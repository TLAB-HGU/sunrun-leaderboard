"""TSAA-style training-distribution augmentation for the CPU direct forecaster.

Adapted from Nochumsohn & Azencot, "Data Augmentation Policy Search for Long-Term
Forecasting", arXiv:2405.00319 (TMLR 2025).  The original searches a distribution
over k sub-policies, each a composition of n time-series transformations, and
selects the policy by validation performance of the trained forecaster.

This module keeps the policy -> transformed training distribution -> same
forecaster mechanism.  The transformations act on the frozen base111 feature
rows instead of raw sequences because training reads the materialized envelope;
this is a labelled CPU adaptation, not a reproduction of the neural pipeline.

A policy is a JSON object (stored as the scalar recipe leaf ``aug_policy``)::

    {"version":1,
     "weights":{"identity":..,"jitter":..,"scale":..,"trend":..,
                "seasonality":..,"mixup":..},
     "magnitude":m, "subpolicy_seed":s, "k":3, "n":2}

Rows are left untouched with probability equal to the normalized identity
weight.  Every other row samples one of k sub-policies; each sub-policy is a
sequence of n (family, magnitude) operations sampled once from the policy
distribution.  All randomness is a deterministic function of
(seed, head band, subpolicy_seed), so checkpoint identities remain reproducible.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

FAMILIES = ("identity", "jitter", "scale", "trend", "seasonality", "mixup")
VERSION = 1


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_policy(text):
    """Return the validated policy dict, or None for the no-augmentation arm."""
    _require(isinstance(text, str) and text, "aug_policy must be a nonempty string")
    if text == "none":
        return None
    body = json.loads(text)
    _require(isinstance(body, dict), "aug_policy must be a JSON object")
    _require(body.get("version") == VERSION, "unsupported aug_policy version")
    weights = body.get("weights")
    _require(isinstance(weights, dict) and set(weights) == set(FAMILIES),
             "aug_policy weights must name exactly " + ",".join(FAMILIES))
    for name, value in weights.items():
        _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                 "invalid augmentation weight: " + name)
    _require(sum(weights.values()) > 0, "augmentation weights sum to zero")
    magnitude = body.get("magnitude")
    _require(type(magnitude) in (int, float) and math.isfinite(magnitude)
             and 0 <= magnitude <= 1, "augmentation magnitude must be in [0,1]")
    seed = body.get("subpolicy_seed")
    _require(type(seed) is int and 0 <= seed < 2**63, "invalid subpolicy_seed")
    for key, upper in (("k", 16), ("n", 8)):
        value = body.get(key)
        _require(type(value) is int and 1 <= value <= upper, "invalid " + key)
    return body


def validate_policy(text):
    parse_policy(text)
    return True


def _columns(features):
    """Column groups used by the transformations (indices into the feature order)."""
    names = list(features)
    index = {name: i for i, name in enumerate(names)}
    speed_lag = [i for i, n in enumerate(names)
                 if n.startswith(("v_lag", "v_m", "v_sd", "rec", "ace_speed", "speed_d"))]
    trend = [i for i in speed_lag if names[i] != "v_lag0"]
    season = [index[n] for n in ("rec26", "rec27", "rec28") if n in index]
    continuous = [i for i, n in enumerate(names) if n != "h"]
    return dict(speed_lag=np.asarray(speed_lag, dtype=int),
                trend=np.asarray(trend, dtype=int),
                season=np.asarray(season, dtype=int),
                continuous=np.asarray(continuous, dtype=int),
                v_lag0=index.get("v_lag0"),
                rec_mean=index.get("rec_mean"))


def _robust_scale(frame, columns, cap=200000):
    """Per-column IQR/1.349 scale, computed on a bounded row sample."""
    values = frame.iloc[:cap, columns].to_numpy(float)
    with np.errstate(all="ignore"):
        q25 = np.nanpercentile(values, 25, axis=0)
        q75 = np.nanpercentile(values, 75, axis=0)
        sd = (q75 - q25) / 1.349
        bad = ~np.isfinite(sd) | (sd <= 0)
        if bad.any():
            sd[bad] = np.nanstd(values[:, bad], axis=0)
        sd[~np.isfinite(sd) | (sd <= 0)] = 0.0
    return sd


def sample_subpolicies(policy):
    """Draw the k x n sub-policy table once from the policy distribution."""
    names = np.asarray(FAMILIES, dtype=object)
    weights = np.asarray([policy["weights"][name] for name in FAMILIES], dtype=float)
    probability = weights / weights.sum()
    rng = np.random.default_rng(policy["subpolicy_seed"])
    table = []
    for _ in range(policy["k"]):
        steps = []
        for _ in range(policy["n"]):
            family = str(rng.choice(names, p=probability))
            magnitude = float(rng.random()) * float(policy["magnitude"])
            steps.append((family, magnitude))
        table.append(steps)
    return table


def _mixup(rng, X, y, rows, groups, columns):
    """Linear interpolation with a same-horizon partner; NaN values fall back."""
    if len(rows) == 0:
        return
    horizon = groups["horizon"]
    by_h = {}
    for row, h in zip(rows, horizon[rows]):
        by_h.setdefault(int(h), []).append(int(row))
    pool = {h: np.asarray(v, dtype=int) for h, v in by_h.items()}
    partner = rows.copy()
    for h, members in pool.items():
        local = rows[horizon[rows] == h]
        if len(members) > 1:
            partner[horizon[rows] == h] = rng.choice(members, size=len(local), replace=True)
    lam = rng.beta(0.4, 0.4, size=len(rows))
    lam = np.maximum(lam, 0.5)  # keep the anchor row dominant
    a = X[rows][:, columns]
    b = X[partner][:, columns]
    mixed = lam[:, None] * a + (1.0 - lam[:, None]) * b
    mixed = np.where(np.isnan(a), b, mixed)
    mixed = np.where(np.isnan(b), a, mixed)
    X[np.ix_(rows, columns)] = mixed
    y[rows] = lam * y[rows] + (1.0 - lam) * y[partner]


def _apply(family, magnitude, rng, X, y, rows, groups, robust_sd):
    if family == "identity" or len(rows) == 0 or magnitude <= 0:
        return
    if family == "jitter":
        columns = groups["continuous"]
        sigma = magnitude * 0.08 * robust_sd
        noise = rng.normal(0.0, 1.0, size=(len(rows), len(columns))) * sigma[None, :]
        block = X[np.ix_(rows, columns)]
        X[np.ix_(rows, columns)] = np.where(np.isfinite(block), block + noise, block)
        return
    if family == "scale":
        columns = groups["speed_lag"]
        factor = 1.0 + magnitude * 0.5 * rng.uniform(-1.0, 1.0, size=len(rows))
        block = X[np.ix_(rows, columns)]
        X[np.ix_(rows, columns)] = np.clip(block * factor[:, None], 0.0, 5000.0)
        y[rows] = np.clip(y[rows] * factor, 0.0, 5000.0)
        return
    if family == "trend":
        anchor = groups["v_lag0"]
        if anchor is None:
            return
        columns = groups["trend"]
        factor = 1.0 + magnitude * 0.8 * rng.uniform(-1.0, 1.0, size=len(rows))
        base = X[np.ix_(rows, [anchor])]
        block = X[np.ix_(rows, columns)]
        warped = base - factor[:, None] * (base - block)
        X[np.ix_(rows, columns)] = np.clip(np.where(np.isfinite(block), warped, block), 0.0, 5000.0)
        return
    if family == "seasonality":
        centre = groups["rec_mean"]
        columns = groups["season"]
        if centre is None or len(columns) == 0:
            return
        factor = 1.0 + magnitude * 0.8 * rng.uniform(-1.0, 1.0, size=len(rows))
        mean = X[np.ix_(rows, [centre])]
        block = X[np.ix_(rows, columns)]
        scaled = mean + factor[:, None] * (block - mean)
        X[np.ix_(rows, columns)] = np.where(np.isfinite(block), scaled, block)
        return
    if family == "mixup":
        _mixup(rng, X, y, rows, groups, groups["continuous"])
        return
    raise ValueError("unknown augmentation family: " + family)


def augment_frame(frame, features, policy_text, seed, band):
    """Return (X, y, current_speed) for one head's training pool.

    ``band`` is the head's (lo, hi) tuple and only seeds the deterministic RNG.
    Rows whose identity draw keeps them clean are bit-identical to the input.
    """
    policy = parse_policy(policy_text)
    columns = list(features)
    y = frame["y"].to_numpy(float)
    current = frame["v_lag0"].to_numpy(float)
    if policy is None:
        return frame[columns].copy(), y, current
    X = frame[columns].to_numpy(float).copy()
    groups = _columns(columns)
    groups["horizon"] = frame["h"].to_numpy(np.int64)
    robust_sd = _robust_scale(frame, groups["continuous"])
    identity_weight = policy["weights"]["identity"] / sum(policy["weights"].values())
    rng = np.random.default_rng(np.random.SeedSequence(
        [int(seed), int(band[0]), int(band[1]), int(policy["subpolicy_seed"])]))
    n = len(frame)
    selected = np.flatnonzero(rng.random(n) >= identity_weight)
    if len(selected):
        table = sample_subpolicies(policy)
        chosen = rng.integers(0, len(table), size=len(selected))
        for number, steps in enumerate(table):
            rows = selected[chosen == number]
            if len(rows) == 0:
                continue
            for family, magnitude in steps:
                _apply(family, magnitude, rng, X, y, rows, groups, robust_sd)
    return pd.DataFrame(X, columns=columns, index=frame.index), y, current


def describe(policy_text):
    policy = parse_policy(policy_text)
    if policy is None:
        return {"mode": "none"}
    return {"mode": "policy", "weights": policy["weights"],
            "magnitude": policy["magnitude"], "k": policy["k"], "n": policy["n"],
            "subpolicy_seed": policy["subpolicy_seed"],
            "subpolicies": [[[family, round(value, 4)] for family, value in steps]
                            for steps in sample_subpolicies(policy)]}
