"""Validate and score a predictions file against truth + regime folds."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HORIZON = 72
MIN_BLOCKS_PER_REGIME = 3  # non-overlapping 72h windows needed before a regime cell is reported
N_BOOT = 1000
REGIMES = [f"{t}_{s}_{f}" for t in ("low", "high") for s in ("low", "high") for f in ("low", "high")]
PERIODS = ("F1", "F2")  # F1 targets 2026-03-01..06-01, F2 2026-06-01..09-30 (UTC, end exclusive)
HSS_KMS = 550.0  # high-speed subset: targets with target_kms >= HSS_KMS
EVENT_GAP_H = 6  # HSS event: hours >= HSS_KMS, runs separated by <= this many quiet hours are one event
EVENT_TOL_H = 12  # forecast and true event peaks match one-to-one when they are at most this far apart
KEY = ["origin_last_input_utc", "horizon_hours"]


def build_truth(series: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Expand folds into (origin, horizon, target_kms) rows from filled series."""
    idx = series.set_index(pd.to_datetime(series["timestamp_utc"], utc=True))
    s = idx["filled_speed_kms"]
    miss = idx["was_missing"].astype(int) if "was_missing" in idx else pd.Series(0, index=idx.index)
    rows = []
    for o in folds["origin_last_input_utc"]:
        ot = pd.Timestamp(o)
        sl = slice(ot + pd.Timedelta(hours=1), ot + pd.Timedelta(hours=HORIZON))
        rows.append(pd.DataFrame({"origin_last_input_utc": o, "horizon_hours": np.arange(1, HORIZON + 1),
                                  "target_kms": s.loc[sl].to_numpy(), "was_missing": miss.loc[sl].to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def n_blocks(origins) -> int:
    """Greedy count of non-overlapping 72h windows among the given fold origins."""
    n, last = 0, None
    for o in sorted(pd.to_datetime(pd.Series(list(origins)), utc=True)):
        if last is None or (o - last) >= pd.Timedelta(hours=HORIZON):
            n, last = n + 1, o
    return n


def _paired_ci(m: pd.DataFrame, ref: pd.DataFrame, seed: int = 0) -> dict:
    """Bootstrap over 72h blocks of mean(se_model - se_ref); negative = model better."""
    d = m.merge(ref[KEY + ["se"]], on=KEY, suffixes=("", "_ref"))
    d["diff"] = d["se"] - d["se_ref"]
    origins = sorted(pd.to_datetime(d["origin_last_input_utc"], utc=True).unique())
    block = {o: i // HORIZON for i, o in enumerate(origins)}  # consecutive hourly origins -> 72h blocks
    d["blk"] = pd.to_datetime(d["origin_last_input_utc"], utc=True).map(block)
    g = d.groupby("blk")["diff"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    boot = g["sum"].to_numpy()[idx].sum(1) / g["count"].to_numpy()[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mse_diff": float(d["diff"].mean()), "ci95": [float(lo), float(hi)], "beats_reference": bool(hi < 0)}


def _events(values: np.ndarray) -> list[int]:
    """Peak hour index of each HSS event in one 72h window; events touching hour 1 or 72 are censored (extent unknown)."""
    active = np.flatnonzero(values >= HSS_KMS)
    if not len(active):
        return []
    peaks = []
    for run in np.split(active, np.flatnonzero(np.diff(active) > EVENT_GAP_H + 1) + 1):
        a, b = int(run[0]), int(run[-1])
        if a > 0 and b < len(values) - 1:
            peaks.append(a + int(np.argmax(values[a:b + 1])))
    return peaks


def _match(true: list[int], pred: list[int]) -> list[tuple[int, int]]:
    """One-to-one peak matches within EVENT_TOL_H: as many matches as possible, then the smallest total timing error."""
    n, k = len(true), len(pred)
    if not n or not k:
        return []
    cost = np.zeros((n + k, n + k))  # dummy rows/columns let any event stay unmatched
    cost[:n, :k] = 1e9
    bonus = (n + k + 1) * (EVENT_TOL_H + 1)
    for i, t in enumerate(true):
        for j, p in enumerate(pred):
            if abs(p - t) <= EVENT_TOL_H:
                cost[i, j] = abs(p - t) - bonus
    rows, cols = linear_sum_assignment(cost)
    return [(true[i], pred[j]) for i, j in zip(rows, cols) if i < n and j < k and cost[i, j] < 0]


def _event_summary(tp: int, fp: int, fn: int, dt: list[int], n_windows: int) -> dict:
    return {"tp": tp, "fp": fp, "fn": fn, "n_windows": n_windows,
            "csi": tp / (tp + fp + fn) if tp + fp + fn else None,
            "pod": tp / (tp + fn) if tp + fn else None,
            "far": fp / (tp + fp) if tp + fp else None,
            "timing_mae_h": float(np.mean(np.abs(dt))) if dt else None,
            "timing_bias_h": float(np.mean(dt)) if dt else None}


def event_scores(m: pd.DataFrame) -> dict:
    """HSS event skill per 72h window (truth as scored, i.e. forward-filled), summed over windows, overall and per period.

    csi = TP / (TP + FP + FN); timing error = forecast peak hour - true peak hour of a matched event (+ = late).
    """
    m = m.sort_values(KEY)
    n = int(m["origin_last_input_utc"].nunique())
    y = m["target_kms"].to_numpy(float).reshape(n, HORIZON)
    p = m["pred_kms"].to_numpy(float).reshape(n, HORIZON)
    period = m["period"].to_numpy()[::HORIZON]
    acc = {}
    for i in range(n):
        true, pred = _events(y[i]), _events(p[i])
        pairs = _match(true, pred)
        acc[i] = (len(pairs), len(pred) - len(pairs), len(true) - len(pairs), [b - a for a, b in pairs])
    out = {}
    for name, sel in [("", np.ones(n, bool))] + [(f"_{q.lower()}", period == q) for q in PERIODS]:
        rows = [acc[i] for i in np.flatnonzero(sel)]
        out[name] = _event_summary(sum(r[0] for r in rows), sum(r[1] for r in rows), sum(r[2] for r in rows),
                                   [d for r in rows for d in r[3]], len(rows))
    return out


def validate(pred: pd.DataFrame, folds: pd.DataFrame) -> list[str]:
    errs = []
    missing_cols = [c for c in KEY + ["pred_kms"] if c not in pred.columns]
    if missing_cols:
        return [f"missing columns: {missing_cols}"]
    n_expected = len(folds) * HORIZON
    if pred.duplicated(KEY).any():
        errs.append(f"{int(pred.duplicated(KEY).sum())} duplicate (origin, horizon) rows")
    if not np.isfinite(pd.to_numeric(pred["pred_kms"], errors="coerce")).all():
        errs.append("pred_kms contains NaN/inf/non-numeric values")
    else:
        if (pred["pred_kms"] <= 0).any() or (pred["pred_kms"] > 3000).any():
            errs.append("pred_kms outside (0, 3000] km/s")
    exp = pd.MultiIndex.from_product([folds["origin_last_input_utc"], range(1, HORIZON + 1)])
    got = pd.MultiIndex.from_frame(pred[KEY].drop_duplicates())
    lacking, extra = exp.difference(got), got.difference(exp)
    if len(lacking):
        errs.append(f"{len(lacking)} of {n_expected} required rows missing, e.g. {lacking[0]}")
    if len(extra):
        errs.append(f"{len(extra)} unexpected rows (origin/horizon not in eval set), e.g. {extra[0]}")
    return errs


def score(pred: pd.DataFrame, truth: pd.DataFrame, folds: pd.DataFrame, naive: pd.DataFrame | None = None) -> dict:
    """Returns overall / per-period / >=550 / per-horizon / per-regime metrics. Assumes validate() passed."""
    if "period" not in folds:  # folds from before the F1/F2 split are all F2
        folds = folds.assign(period="F2")
    m = truth.merge(pred[KEY + ["pred_kms"]], on=KEY, how="inner").merge(
        folds[["origin_last_input_utc", "regime", "period"]], on="origin_last_input_utc")
    if "was_missing" not in m:
        m["was_missing"] = 0
    m["se"] = (m["target_kms"] - m["pred_kms"]) ** 2
    m["ae"] = (m["target_kms"] - m["pred_kms"]).abs()
    subsets = {}  # overall, per period, and each restricted to target_kms >= HSS_KMS
    for name, sel in [("", pd.Series(True, index=m.index))] + [(f"_{p.lower()}", m["period"] == p) for p in PERIODS]:
        subsets[name] = sel
        subsets[f"{name}_ge550"] = sel & (m["target_kms"] >= HSS_KMS)
    by_period = {f"mse{k}": float(m.loc[sel, "se"].mean()) if sel.any() else None for k, sel in subsets.items()}
    by_period.update({f"n_pairs{k}": int(sel.sum()) for k, sel in subsets.items() if k})
    by_regime = {}
    f2 = m[m["period"] == "F2"]  # regime rows stay F2-only: thresholds were frozen on the F2 folds
    for r in REGIMES:
        g = f2[f2["regime"] == r]
        n = int(g["origin_last_input_utc"].nunique())
        nb = n_blocks(g["origin_last_input_utc"].unique()) if n else 0
        by_regime[r] = {"n_folds": n, "n_blocks": nb, "mse": float(g["se"].mean()) if n else None,
                        "reliable": nb >= MIN_BLOCKS_PER_REGIME}
    reliable = [v["mse"] for v in by_regime.values() if v["reliable"]]
    events = event_scores(m)
    out = {
        **by_period,
        **{f"{k}{p}": events[p][k] for p in events for k in ("csi", "timing_mae_h", "timing_bias_h")},
        "events": events,
        "rmse": float(np.sqrt(m["se"].mean())),
        "mae": float(m["ae"].mean()),
        "mse_observed": float(m.loc[m["was_missing"] == 0, "se"].mean()) if (m["was_missing"] == 0).any() else None,
        "regime_balanced_mse": float(np.mean(reliable)) if reliable else None,
        "mse_by_horizon": m.groupby("horizon_hours")["se"].mean().round(4).to_dict(),
        "by_regime": by_regime,
        "n_folds": int(m["origin_last_input_utc"].nunique()),
        "n_pairs": int(len(m)),
    }
    if naive is not None:
        nm = score(naive, truth, folds)["mse"]
        out["skill_vs_naive"] = 1.0 - out["mse"] / nm
        nv = truth.merge(naive[KEY + ["pred_kms"]], on=KEY)
        nv["se"] = (nv["target_kms"] - nv["pred_kms"]) ** 2
        out["vs_naive"] = _paired_ci(m, nv)
    return out
