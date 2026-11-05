"""Bottleneck candidates b01/b02/b05 on the frozen e2000fs_budget_spectral parent.

Single file: data loading, candidate labels/features, spectral XGBoost training
(parent recipe), leaderboard-period evaluation only (F1/F2/F3 with the official
training cutoffs, scored on the leaderboard truth by leaderboard/scorer/competition.py),
paired lead-first S comparison with an origin-block bootstrap, the preregistered gate
and every artifact. No internal validation stage (user rule 2026-11-04).

Experiments (``--experiment``):
  a00_spectral_parent_control      arm A   : unchanged parent, seeds 42/43/44 (shared control)
  b01_spectral_rawlabel_mask       arms B, AB: spectral aux only on raw-complete blocks; AB also
                                              trains point4 on raw (unfilled) labels only
  b02_spectral_input_freshness     arms AB, NC: +63 raw availability/age/never-seen columns;
                                              NC = same 63 dims of training-marginal pseudorandom draws
  b05_spectral_source_travel_lags  arms B, AB: +8 varying CH columns/h; AB = as-of value at
                                              origin+h-tau (tau 96..168h), B = current value repeated

GBM family: CPU only. Leaderboard truth is read only after fitting, for scoring; it never
enters training, feature construction or any fitted coefficient. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
# BLAS stays single-threaded; OMP_NUM_THREADS is left to the launcher (=--threads) because
# OMP_NUM_THREADS=1 caps XGBoost's OpenMP pool regardless of nthread.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path("/home/t-lab01/sunrun")
CAMPAIGN = ROOT / "research/bottleneck_candidates"
DATA = ROOT / "research/rerun_df_ffill/input/df_ffill_trainy.parquet"
RAW = ROOT / "data-v2/input/df.parquet"
PARENT = "e2000fs_budget_spectral"
PARENT_SOURCE = CAMPAIGN / "evidence/e2000fs_budget_spectral/experiment.py"

H, BLOCK, STARTS = 72, 24, (0, 24, 48)
ROUNDS = 900
SEEDS = (42, 43, 44)
CONSTANT = ["v_lag0", "ace_speed_kms_last", "ace_density_cm3_last", "ace_density_cm3_m6",
            "ace_temperature_k_last", "ace_temperature_k_m24", "ace_bt_nt_last", "ace_bt_nt_m6",
            "ace_by_gsm_nt_m24", "ace_epam_p_channel3_last", "ace_sis_gt10_flux_last",
            "ch_ch_t035_top2_area", "ch_ch_t035_meridian15_equatorial_area",
            "ch_ch_t035_meridian30_equatorial_area", "ch_ch_t035_equatorial_area",
            "ch_ch_t055_meridian15_area", "ch_ch_t055_meridian30_equatorial_area",
            "ch_ch_t055_equatorial_area", "img_age_h", "enlil_n_pending", "enlil_kp"]
VARYING = ["rec26", "rec27", "rec28", "rec_mean", "strip_v350_logmed", "strip_v450_logmed",
           "strip_v550_dark", "strip_v650_dark", "strip_v750_logmed"]
OFFICIAL = {"F1": "2025-12-29", "F2": "2026-03-29", "F3": "2026-06-29"}
QUERY_COUNTS = {"F1": 2089, "F2": 2113, "F3": 2113}
PERIODS = tuple(OFFICIAL)          # the only evaluation windows (leaderboard F1/F2/F3)
PRIMARY_SEED = 42                  # seed whose primary-arm forecast is written as F{n}_predictions.npz
LB_DATASET = "tlabtlab/sunrun-lb-store"
SCORER = ROOT / "leaderboard/scorer/competition.py"
MATURITY_H = 144
FRESH_SEED = 20261104
BOOT_SEED = 20261104
BOOT_DRAWS = 2000
BOOT_BLOCK_DAYS = 27
TAUS = (96, 120, 144, 168)
LAG_SOURCES = ("ch_ch_t035_top2_area", "ch_ch_t035_meridian30_equatorial_area")

CONTROL = "a00_spectral_parent_control"
B01, B02, B05 = "b01_spectral_rawlabel_mask", "b02_spectral_input_freshness", "b05_spectral_source_travel_lags"
ARMS = {CONTROL: ("A",), B01: ("B", "AB"), B02: ("AB", "NC"), B05: ("B", "AB")}
ARM_TITLES = {
    (CONTROL, "A"): "unchanged ffilled-y spectral parent",
    (B01, "B"): "ffilled labels; spectral auxiliary only on raw-complete 24h blocks (mask-matched control)",
    (B01, "AB"): "raw-observed point4 labels + same raw-complete spectral eligibility as B",
    (B02, "AB"): "parent 237 + 63 raw availability / log1p(age_h) / never_seen columns",
    (B02, "NC"): "parent 237 + 63 pseudorandom columns drawn from training-only marginals (negative control)",
    (B05, "B"): "parent 237 + 192 columns: current CH snapshot repeated for each (h, tau) (capacity control)",
    (B05, "AB"): "parent 237 + 192 columns: causal as-of CH at origin+h-tau, tau 96/120/144/168h",
}
# Preregistered decision rule (fixed before any fit; see strategy.md).
DECISION = {
    B01: {"primary": ("AB", "A"), "also_required": [("AB", "B")], "descriptive": [("B", "A")]},
    B02: {"primary": ("AB", "A"), "also_required": [("AB", "NC")], "descriptive": [("NC", "A")]},
    B05: {"primary": ("AB", "A"), "also_required": [("AB", "B")], "descriptive": [("B", "A")]},
}
GATE = {"min_improvement_pct": 1.0, "guard_pct": 1.0, "period_guard_pct": 2.0, "worst_seed_pct": 0.5}


def mapped(name):
    return name.replace("ch_ch_", "suvi_", 1) if name.startswith("ch_ch_") else name


def log(**record):
    print(json.dumps(record, default=str), flush=True)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def file_identity(path):
    stat = Path(path).stat()
    return {"path": str(Path(path).resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


# ----------------------------------------------------------------------------- data
def _grid(frame, n):
    if len(frame) != n * H or not np.array_equal(frame.h.to_numpy().reshape(-1, H),
                                                 np.broadcast_to(np.arange(1, H + 1), (n, H))):
        raise ValueError("incomplete origin/horizon grid")


def load_tables(data=DATA, raw=RAW):
    """Train-set grid (ffill features, ffill y, raw y), query grids, raw h==1 timeline."""
    consts = [mapped(c) for c in CONSTANT]
    head_cols = ["set", "origin", "period", "fit_F1", "fit_F2", "fit_F3", *consts]
    head = pd.read_parquet(data, columns=head_cols, filters=[("h", "==", 1)])
    raw_head = pd.read_parquet(raw, columns=["set", "origin", *consts], filters=[("h", "==", 1)])
    for frame in (head, raw_head):
        frame["origin"] = pd.to_datetime(frame.origin, utc=True)
    if not (head.origin.to_numpy() == raw_head.origin.to_numpy()).all() or \
            not (head.set.astype(str).to_numpy() == raw_head.set.astype(str).to_numpy()).all():
        raise ValueError("ffill and raw tables are not row-aligned")
    rows = pd.read_parquet(data, columns=["set", "origin", "h", "y", *VARYING])
    raw_rows = pd.read_parquet(raw, columns=["set", "origin", "h", "y"])
    for frame in (rows, raw_rows):
        frame["origin"] = pd.to_datetime(frame.origin, utc=True)
    out = {"identity": {"data": file_identity(data), "raw": file_identity(raw)}}
    for split in ("train", "query"):
        h1 = head[head.set.astype(str) == split].sort_values("origin", kind="stable")
        r = rows[rows.set.astype(str) == split].sort_values(["origin", "h"], kind="stable")
        rr = raw_rows[raw_rows.set.astype(str) == split].sort_values(["origin", "h"], kind="stable")
        origins = pd.DatetimeIndex(h1.origin)
        if split == "train":
            if origins.has_duplicates:
                raise ValueError("duplicate training origins")
            _grid(r, len(origins))
            if not (pd.DatetimeIndex(r.origin.iloc[::H]) == origins).all():
                raise ValueError("constant and varying grids disagree")
            if not (r.origin.to_numpy() == rr.origin.to_numpy()).all() or not (r.h.to_numpy() == rr.h.to_numpy()).all():
                raise ValueError("raw label rows misaligned")
            out["train"] = {"origins": origins, "C": h1[consts].to_numpy(np.float32),
                            "E": r[VARYING].to_numpy(np.float32).reshape(len(origins), H, len(VARYING)),
                            "y": r.y.to_numpy(np.float64).reshape(len(origins), H),
                            "y_raw": rr.y.to_numpy(np.float64).reshape(len(origins), H),
                            "fit": {p: h1[f"fit_{p}"].to_numpy(bool) for p in OFFICIAL}}
        else:
            out["query"] = {}
            qrows = pd.read_parquet(data, columns=["origin", "h", "period", *VARYING], filters=[("set", "==", "query")])
            qrows["origin"] = pd.to_datetime(qrows.origin, utc=True)
            for period in OFFICIAL:
                ph = h1[h1.period.astype(str) == period].sort_values("origin", kind="stable")
                pr = qrows[qrows.period.astype(str) == period].sort_values(["origin", "h"], kind="stable")
                po = pd.DatetimeIndex(ph.origin)
                _grid(pr, len(po))
                if len(po) != QUERY_COUNTS[period] or not (pd.DatetimeIndex(pr.origin.iloc[::H]) == po).all():
                    raise ValueError(f"unexpected {period} query grid")
                out["query"][period] = {"origins": po, "C": ph[consts].to_numpy(np.float32),
                                        "E": pr[VARYING].to_numpy(np.float32).reshape(len(po), H, len(VARYING))}
    # raw causal timeline of constant snapshots (train/query union; duplicates must agree)
    timeline = raw_head.drop(columns="set")
    dup = timeline[timeline.origin.duplicated(keep=False)]
    if len(dup):
        agree = dup.groupby("origin")[consts].nunique(dropna=False).max(axis=1)
        if (agree > 1).any():
            raise ValueError("raw train/query snapshots disagree for the same origin")
    out["timeline"] = timeline.drop_duplicates("origin").sort_values("origin").set_index("origin")
    return out


# ----------------------------------------------------------------------------- candidate features
def freshness_table(timeline):
    """63 columns: per parent constant raw_available, log1p(age_h), never_seen (origin-causal)."""
    t = (timeline.index.asi8 // 3_600_000_000_000).astype(np.float64)   # whole hours
    columns, names = [], []
    for c in CONSTANT:
        col = mapped(c)
        if col not in timeline:
            raise ValueError(f"no raw provenance for {c}")
        avail = np.isfinite(timeline[col].to_numpy(np.float64))
        last = pd.Series(np.where(avail, t, np.nan)).ffill().to_numpy()
        never = ~np.isfinite(last)
        age = np.where(never, 0.0, t - np.nan_to_num(last))
        if (age < 0).any():
            raise ValueError("negative information age")
        columns += [avail.astype(np.float32), np.log1p(age).astype(np.float32), never.astype(np.float32)]
        names += [f"{c}__raw_available", f"{c}__log1p_age_h", f"{c}__never_seen"]
    return pd.DataFrame(np.column_stack(columns), index=timeline.index, columns=names)


def lookup(table, origins):
    values = table.reindex(origins)
    if values.isna().any().any():
        raise ValueError("origin missing from the causal timeline")
    return values.to_numpy(np.float32)


def pseudorandom_like(pool, origins):
    """Origin-keyed draws from training-only marginals (column-wise)."""
    hours = (origins.asi8 // 3_600_000_000_000).astype(np.int64)
    out = np.empty((len(origins), pool.shape[1]), np.float32)
    cols = np.arange(pool.shape[1])
    for i, hour in enumerate(hours):
        idx = np.random.default_rng([FRESH_SEED, int(hour)]).integers(0, len(pool), pool.shape[1])
        out[i] = pool[idx, cols]
    return out


def asof_grid(timeline):
    """Hourly causal as-of grid (unlimited ffill on raw values, no backward fill)."""
    full = pd.date_range(timeline.index.min(), timeline.index.max(), freq="h")
    cols = [mapped(c) for c in LAG_SOURCES]
    grid = timeline[cols].reindex(full).ffill()
    return full, grid.to_numpy(np.float64)


def lag_block(origins, full, grid, current=False):
    """(n, 72, 8) ordered per h: [q0@tau96..168, q1@tau96..168]."""
    base = full.asi8[0]
    step = 3_600_000_000_000
    o = origins.asi8
    h = np.arange(1, H + 1)
    out = np.full((len(o), H, len(LAG_SOURCES) * len(TAUS)), np.nan, np.float32)
    for j, tau in enumerate(TAUS):
        if current:
            source = np.broadcast_to(o[:, None], (len(o), H))
        else:
            source = o[:, None] + (h[None, :] - tau) * step
        if not current and not (source <= o[:, None] - 24 * step).all():
            raise ValueError("source time later than origin-24h")
        if not (source <= o[:, None]).all():
            raise ValueError("future source")
        index = (source - base) // step
        valid = (index >= 0) & (index < len(full)) & ((source - base) % step == 0)
        for q in range(len(LAG_SOURCES)):
            values = np.full(source.shape, np.nan)
            values[valid] = grid[index[valid], q]
            out[:, :, q * len(TAUS) + j] = values
    return out


def lag_names():
    return [f"{c}@src_tau{tau}" for c in LAG_SOURCES for tau in TAUS]


# ----------------------------------------------------------------------------- design / objective
def feature_names(start, extra_const=(), extra_vary=()):
    hours = range(start + 1, start + BLOCK + 1)
    return (list(CONSTANT) + [f"{v}@h{h}" for h in hours for v in VARYING] + list(extra_const)
            + [f"{v}@h{h}" for h in hours for v in extra_vary])


def design(C, E, start, extra_const=None, extra_vary=None):
    """Parent design (C, E[start:start+24] flattened) with explicit unions appended."""
    parts = [C, E[:, start:start + BLOCK].reshape(len(C), -1)]
    if extra_const is not None:
        parts.append(extra_const)
    if extra_vary is not None:
        parts.append(extra_vary[:, start:start + BLOCK].reshape(len(C), -1))
    return np.hstack(parts).astype(np.float32)


def spectral_objective(target, observed, eligible):
    """Parent point4 + 0.10 spectral objective; spectral term only where ``eligible``."""
    observed = np.asarray(observed, bool)
    raw = np.where(np.nan_to_num(target) >= 550., 4., 1.) * observed
    weights = raw / (raw.sum(0) / observed.sum(0))
    gate = np.asarray(eligible, np.float64)[:, None]
    label = np.nan_to_num(target)

    def objective(prediction, matrix):
        prediction = np.asarray(prediction).reshape(target.shape)
        error = np.where(observed, prediction - label, 0.)
        z = np.fft.fft(error, axis=1, norm="ortho")
        rho = 1 / np.sqrt(1 + abs(z) ** 2 / 50. ** 2)
        gradient = (error * weights + gate * .10 * np.fft.ifft(z * rho, axis=1, norm="ortho").real) * observed
        return gradient.astype(np.float32), ((weights + .10 * gate) * observed).astype(np.float32)
    return objective


def spectral_parameters(threads, seed):
    return dict(max_depth=3, learning_rate=.03, subsample=.8, colsample_bytree=.7,
                min_child_weight=20, reg_lambda=5, objective="reg:squarederror", tree_method="hist",
                device="cpu", eval_metric="rmse", multi_strategy="multi_output_tree", nthread=threads,
                seed=seed, num_target=24)


def label_policy(experiment, arm, y, y_raw, start):
    """(labels, observed, spectral_eligible, keep) for one 24h block."""
    target_ffill = y[:, start:start + BLOCK]
    raw = y_raw[:, start:start + BLOCK]
    keep = np.isfinite(target_ffill).any(axis=1)             # parent row rule (ffill labels)
    raw_complete = np.isfinite(raw).all(axis=1)
    if experiment == B01 and arm == "AB":
        labels, eligible = raw, raw_complete
    elif experiment == B01 and arm == "B":
        labels, eligible = target_ffill, raw_complete
    else:
        labels, eligible = target_ffill, np.ones(len(raw), bool)
    labels, eligible = labels[keep], eligible[keep]
    observed = np.isfinite(labels)
    if not observed.any(axis=0).all():
        raise ValueError("a block output has no observed training labels")
    return labels, observed, eligible, keep


def label_diagnostics(y, y_raw, start, keep, eligible, observed):
    ffill = np.isfinite(y[:, start:start + BLOCK])[keep]
    raw = np.isfinite(y_raw[:, start:start + BLOCK])[keep]
    return {"block": start // BLOCK, "rows": int(keep.sum()), "ffill_labels": int(ffill.sum()),
            "raw_labels": int(raw.sum()), "artificial_labels": int((ffill & ~raw).sum()),
            "artificial_fraction": float((ffill & ~raw).sum() / max(ffill.sum(), 1)),
            "all_raw_missing_origins": int((~raw.any(axis=1)).sum()),
            "spectral_eligible_origins": int(np.asarray(eligible).sum()),
            "point4_cells_used": int(observed.sum())}


def run_lengths(mask):
    """Lengths of consecutive True runs in a 1-D mask."""
    padded = np.concatenate([[False], mask, [False]]).astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return edges[1::2] - edges[::2]


# ----------------------------------------------------------------------------- arm inputs
class Inputs:
    """Builds arm-specific extra columns for any origin set (train/eval/query)."""

    def __init__(self, experiment, tables):
        self.experiment, self.tables = experiment, tables
        self.fresh = freshness_table(tables["timeline"]) if experiment == B02 else None
        self.full, self.grid = asof_grid(tables["timeline"]) if experiment == B05 else (None, None)

    def extras(self, arm, origins, pool_origins=None):
        """(extra_const, extra_vary, const_names, vary_names)."""
        if arm == "A":
            return None, None, [], []
        if self.experiment == B02:
            names = list(self.fresh.columns)
            if arm == "AB":
                return lookup(self.fresh, origins), None, names, []
            pool = lookup(self.fresh, pool_origins)
            return pseudorandom_like(pool, origins), None, [f"nc_{n}" for n in names], []
        if self.experiment == B05:
            names = lag_names() if arm == "AB" else [f"{c}@current_rep{tau}" for c in LAG_SOURCES for tau in TAUS]
            return None, lag_block(origins, self.full, self.grid, current=(arm == "B")), [], names
        return None, None, [], []


# ----------------------------------------------------------------------------- metrics
def score_arrays(pred, truth):
    """Lead-first S (= W) and auxiliary metrics; same algebra as scorer/competition.py."""
    if pred.shape != truth.shape or pred.shape[1] != H or not np.isfinite(pred).all() or (pred <= 0).any():
        raise ValueError("predictions must be positive finite n x 72")
    rows = []
    for h in range(H):
        observed = np.isfinite(truth[:, h])
        y, e = truth[observed, h], pred[observed, h] - truth[observed, h]
        if len(y) < 2 or np.ptp(y) <= 0:
            raise ValueError("insufficient observed variation to score every lead")
        q = np.quantile(y, np.arange(101) / 100, method="linear")
        order = np.argsort(y, kind="stable")
        positions = np.searchsorted(y[order], q[:-1], side="left")
        suffix = np.cumsum((e[order] ** 2)[::-1])[::-1]
        timse = np.dot(np.diff(q), suffix[positions] / (len(y) - positions)) / np.ptp(y)
        rmse = np.sqrt(np.mean(e ** 2))
        rows.append({"lead": h + 1, "S": float(np.sqrt(rmse * np.sqrt(timse))), "rmse": float(rmse),
                     "timse": float(timse), "mae": float(np.mean(np.abs(e))), "bias": float(e.mean()), "n": int(len(y))})
    metrics = {k: float(np.mean([r[k] for r in rows])) for k in ("S", "rmse", "timse", "mae", "bias")}
    observed = np.isfinite(truth)
    errors = pred - truth
    metrics.update(n_pairs=int(observed.sum()), n_origins=int(len(pred)), by_lead=rows,
                   pooled_rmse=float(np.sqrt(np.mean(errors[observed] ** 2))),
                   pooled_mae=float(np.mean(np.abs(errors[observed]))))
    body = observed & (np.nan_to_num(truth) < 550)
    metrics["body_lt550"] = {"n": int(body.sum()), "bias": float(errors[body].mean()) if body.any() else None,
                             "rmse": float(np.sqrt(np.mean(errors[body] ** 2))) if body.any() else None}
    for threshold in (550, 700):
        high = observed & (np.nan_to_num(truth) >= threshold)
        metrics[f"tail_ge{threshold}"] = {"n": int(high.sum()),
                                          "rmse": float(np.sqrt(np.mean(errors[high] ** 2))) if high.any() else None,
                                          "bias": float(errors[high].mean()) if high.any() else None}
    metrics["horizon_bands"] = {f"h{a + 1}-{b}": {k: float(np.mean([r[k] for r in rows[a:b]])) for k in ("S", "rmse", "mae", "bias")}
                                for a, b in ((0, 24), (24, 48), (48, 72))}
    return metrics


def s_only(pred, truth):
    total = 0.0
    for h in range(H):
        observed = np.isfinite(truth[:, h])
        y, e = truth[observed, h], pred[observed, h] - truth[observed, h]
        q = np.quantile(y, np.arange(101) / 100, method="linear")
        order = np.argsort(y, kind="stable")
        positions = np.searchsorted(y[order], q[:-1], side="left")
        suffix = np.cumsum((e[order] ** 2)[::-1])[::-1]
        timse = np.dot(np.diff(q), suffix[positions] / (len(y) - positions)) / np.ptp(y)
        total += np.sqrt(np.sqrt(np.mean(e ** 2)) * np.sqrt(timse))
    return total / H


# ----------------------------------------------------------------------------- leaderboard truth
def load_scorer():
    spec = importlib.util.spec_from_file_location("sunrun_competition", SCORER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_observations(path):
    """Explicit file, else the current revision of the leaderboard store (same file submit.py scores with)."""
    if path is not None:
        return Path(path), None
    from huggingface_hub import HfApi, hf_hub_download
    revision = HfApi().repo_info(LB_DATASET, repo_type="dataset").sha
    return Path(hf_hub_download(LB_DATASET, "competition-observations.parquet", repo_type="dataset",
                                revision=revision)), revision


def lb_context(args):
    competition = load_scorer()
    path, revision = resolve_observations(args.observations)
    context = competition.prepare_context(competition.load_observations(path))
    args.observation_identity = {**file_identity(path), "dataset": LB_DATASET, "revision": revision,
                                 "scorer": file_identity(SCORER), "scorer_version": competition.VERSION}
    return competition, context


def period_grid(context, period):
    """Scored 6 h origins of one leaderboard period and their raw truth (scorer population)."""
    mask = np.asarray(context["period"]) == period
    return pd.DatetimeIndex(context["origins"][mask]), np.asarray(context["truth"][mask], np.float64)


# ----------------------------------------------------------------------------- training
def split_counts(model, n_features):
    """Split usage per feature from the saved JSON trees (get_score is unsupported for multi-output trees)."""
    trees = json.loads(model.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]["trees"]
    counts = np.zeros(n_features, np.int64)
    for tree in trees:
        left, index = np.asarray(tree["left_children"]), np.asarray(tree["split_indices"])
        np.add.at(counts, index[left != -1], 1)
    return counts


def train_predict(experiment, arm, X_by_block, Xq_by_block, y, y_raw, seed, threads, rounds, out, tag):
    """Fit the three 24h blocks; returns predictions (n_query, 72) and per-block reports."""
    out.mkdir(parents=True, exist_ok=True)
    pred = np.empty((len(Xq_by_block[0]), H), np.float64)
    reports = []
    for block, start in enumerate(STARTS):
        labels, observed, eligible, keep = label_policy(experiment, arm, y, y_raw, start)
        matrix = xgb.DMatrix(X_by_block[block][keep], label=np.nan_to_num(labels).astype(np.float32), nthread=threads)
        began = time.perf_counter()
        model = xgb.train(dict(spectral_parameters(threads, seed), base_score=float(labels[observed].mean())), matrix,
                          num_boost_round=rounds, obj=spectral_objective(labels, observed, eligible))
        fit_seconds = time.perf_counter() - began
        model.save_model(out / f"block{block}.ubj")
        pred[:, start:start + BLOCK] = np.clip(model.inplace_predict(Xq_by_block[block]), 1, 3000)
        splits = split_counts(model, X_by_block[block].shape[1])
        n_base = len(CONSTANT) + BLOCK * len(VARYING)
        report = {**label_diagnostics(y, y_raw, start, keep, eligible, observed), "fit_seconds": fit_seconds,
                  "base_score": float(labels[observed].mean()), "features": int(X_by_block[block].shape[1]),
                  "splits_total": int(splits.sum()), "splits_on_added_columns": int(splits[n_base:].sum())}
        reports.append(report)
        log(experiment=experiment, **tag, arm=arm, seed=seed, block=block, fit_seconds=round(fit_seconds, 1))
    if not np.isfinite(pred).all():
        raise ValueError("nonfinite predictions")
    return pred, reports


def build_blocks(C, E, extra_const, extra_vary):
    return [design(C, E, s, extra_const, extra_vary) for s in STARTS]


def lb_run(args, tables, inputs, arm, seed, period, identity):
    """Train with the official cutoff of ``period``, forecast its whole query grid, score the scored grid."""
    out = args.out / "lb" / arm / f"seed{seed}"
    path, meta_path = out / f"{period}.npz", out / f"{period}.json"
    if path.exists() and meta_path.exists():
        if json.loads(meta_path.read_text()).get("identity") != identity:
            raise ValueError(f"checkpoint identity mismatch: {meta_path}")
        log(experiment=args.experiment, stage="lb", period=period, arm=arm, seed=seed, resumed=True)
        return
    tr, query = tables["train"], tables["query"][period]
    fit = np.flatnonzero(tr["fit"][period])
    if not (tr["origins"][fit] + pd.Timedelta(hours=MATURITY_H) <= pd.Timestamp(OFFICIAL[period], tz="UTC")).all():
        raise ValueError("official training cutoff violated")
    grid, truth = period_grid(args.context, period)
    position = query["origins"].get_indexer(grid)
    if (position < 0).any():
        raise ValueError(f"{period}: scored leaderboard origin missing from the query grid")
    rows = np.arange(len(query["origins"]))
    if args.smoke:
        fit = fit[np.linspace(0, len(fit) - 1, min(len(fit), 512)).astype(int)]
        rows = position[:24]
    qo, fo = query["origins"][rows], tr["origins"][fit]
    xc, xv, cn, vn = inputs.extras(arm, fo, pool_origins=fo)
    qc, qv, _, _ = inputs.extras(arm, qo, pool_origins=fo)
    X = build_blocks(tr["C"][fit], tr["E"][fit], xc, xv)
    began = time.perf_counter()
    pred, reports = train_predict(args.experiment, arm, X, build_blocks(query["C"][rows], query["E"][rows], qc, qv),
                                  tr["y"][fit], tr["y_raw"][fit], seed, args.threads, args.rounds,
                                  out / f"{period}_models", {"stage": "lb", "period": period})
    scored = qo.get_indexer(grid)
    keep = scored >= 0
    grid_pred, grid_truth = pred[scored[keep]], truth[keep]
    np.savez_compressed(path, pred=grid_pred, truth=grid_truth, origins=grid[keep].asi8,
                        pred_full=pred, origins_full=qo.asi8)
    names = [feature_names(s, cn, vn) for s in STARTS]
    if any(len(n) != x.shape[1] for n, x in zip(names, X)):
        raise ValueError("feature name/count mismatch")
    added = {}
    if xc is not None or xv is not None:
        extra = xc if xc is not None else xv.reshape(len(xv), -1)
        added = {"nonmissing_fraction": float(np.isfinite(extra).mean()), "columns": len(cn) + BLOCK * len(vn)}
    meta = {"identity": identity, "experiment": args.experiment, "arm": arm,
            "arm_title": ARM_TITLES.get((args.experiment, arm), ARM_TITLES[(CONTROL, "A")]), "seed": seed,
            "period": period, "training_cutoff": OFFICIAL[period], "maturity_hours": MATURITY_H,
            "train_origins": int(len(fit)), "train_start": str(fo.min()), "train_end": str(fo.max()),
            "max_train_target": str(fo.max() + pd.Timedelta(hours=H)), "query_origins": int(len(qo)),
            "query_start": str(qo.min()), "query_end": str(qo.max()), "scored_origins": int(keep.sum()),
            "scored_pairs": int(np.isfinite(grid_truth).sum()), "feature_count_per_block": [x.shape[1] for x in X],
            "features": names, "added_columns": added, "blocks": reports, "seconds": time.perf_counter() - began,
            "evaluation": "leaderboard period truth via the official scorer population; read after fitting only",
            "observations": args.observation_identity, "metrics": score_arrays(grid_pred, grid_truth), "uploaded": False}
    write_json(meta_path, meta)
    log(experiment=args.experiment, stage="lb", period=period, arm=arm, seed=seed, done=True,
        S=round(meta["metrics"]["S"], 4), seconds=round(meta["seconds"], 1))


def write_primary(args, period, arm):
    """Submission-format artifacts (seed 42 of the primary arm); scoring stays in evaluation.json."""
    root = args.out / "lb" / arm / f"seed{PRIMARY_SEED}"
    with np.load(root / f"{period}.npz") as z:
        np.savez_compressed(args.out / f"{period}_predictions.npz", pred=z["pred_full"], origins=z["origins_full"])
    meta = json.loads((root / f"{period}.json").read_text())
    meta.pop("metrics", None)
    write_json(args.out / f"{period}_metadata.json", {**meta, "query_labels_read_for_training": False})


# ----------------------------------------------------------------------------- evaluation
def collect_arm(root, arm, seeds, folds):
    runs = {}
    for seed in seeds:
        parts = []
        for fold in folds:
            with np.load(root / "lb" / arm / f"seed{seed}" / f"{fold}.npz") as z:
                parts.append({"pred": z["pred"], "truth": z["truth"], "origins": z["origins"],
                              "fold": np.full(len(z["pred"]), fold)})
        runs[seed] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return runs


def pct(a, b):
    return 100.0 * (a - b) / b


def compare(cand, ctrl, seeds, folds, draws, label):
    """Paired seed comparison + origin-block bootstrap of the seed-mean delta S."""
    for s in seeds:
        if not (np.array_equal(cand[s]["origins"], ctrl[s]["origins"]) and
                np.array_equal(cand[s]["truth"], ctrl[s]["truth"], equal_nan=True)):
            raise ValueError("candidate/control evaluation grids differ")
    per_seed = {}
    for s in seeds:
        mc, ma = score_arrays(cand[s]["pred"], cand[s]["truth"]), score_arrays(ctrl[s]["pred"], ctrl[s]["truth"])
        periods = {}
        for fold in folds:
            m = cand[s]["fold"] == fold
            periods[fold] = {"candidate": s_only(cand[s]["pred"][m], cand[s]["truth"][m]),
                              "control": s_only(ctrl[s]["pred"][m], ctrl[s]["truth"][m])}
        per_seed[s] = {"candidate": {k: mc[k] for k in ("S", "rmse", "timse", "mae", "bias")},
                       "control": {k: ma[k] for k in ("S", "rmse", "timse", "mae", "bias")},
                       "delta_S": mc["S"] - ma["S"], "delta_S_pct": pct(mc["S"], ma["S"]), "periods": periods,
                       "candidate_detail": {k: mc[k] for k in ("body_lt550", "tail_ge550", "tail_ge700", "horizon_bands")},
                       "control_detail": {k: ma[k] for k in ("body_lt550", "tail_ge550", "tail_ge700", "horizon_bands")}}
    mean = lambda side, key: float(np.mean([per_seed[s][side][key] for s in seeds]))
    summary = {key: {"candidate": mean("candidate", key), "control": mean("control", key),
                     "delta_pct": pct(mean("candidate", key), mean("control", key))} for key in ("S", "rmse", "timse", "mae")}
    period_delta = {f: pct(np.mean([per_seed[s]["periods"][f]["candidate"] for s in seeds]),
                      np.mean([per_seed[s]["periods"][f]["control"] for s in seeds])) for f in folds}
    seed_pct = [per_seed[s]["delta_S_pct"] for s in seeds]
    # bootstrap: resample whole 27-day origin blocks (72h trajectories stay intact)
    origins = pd.DatetimeIndex(cand[seeds[0]]["origins"], tz="UTC")
    block_id = np.asarray((origins - origins.min()) // pd.Timedelta(days=BOOT_BLOCK_DAYS))
    blocks = [np.flatnonzero(block_id == b) for b in np.unique(block_id)]
    rng = np.random.default_rng(BOOT_SEED)
    deltas = np.empty(draws)
    for d in range(draws):
        rows = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        deltas[d] = np.mean([s_only(cand[s]["pred"][rows], cand[s]["truth"][rows]) -
                             s_only(ctrl[s]["pred"][rows], ctrl[s]["truth"][rows]) for s in seeds])
    delta_mean = summary["S"]["candidate"] - summary["S"]["control"]
    boot = {"draws": draws, "block_days": BOOT_BLOCK_DAYS, "effective_blocks": len(blocks), "seed": BOOT_SEED,
            "delta_S_mean": delta_mean, "ci95": [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))],
            "p_delta_ge_0": float(np.mean(deltas >= 0))}
    checks = {"mean_S_improvement_ge_1pct": summary["S"]["delta_pct"] <= -GATE["min_improvement_pct"],
              "bootstrap_upper_lt_0": boot["ci95"][1] < 0,
              "rmse_guard": summary["rmse"]["delta_pct"] <= GATE["guard_pct"],
              "mae_guard": summary["mae"]["delta_pct"] <= GATE["guard_pct"],
              "timse_guard": summary["timse"]["delta_pct"] <= GATE["guard_pct"],
              "period_guard": all(v <= GATE["period_guard_pct"] for v in period_delta.values()),
              "worst_seed_guard": max(seed_pct) <= GATE["worst_seed_pct"]}
    return {"comparison": label, "seeds": list(seeds), "per_seed": {str(k): v for k, v in per_seed.items()},
            "summary": summary, "period_delta_S_pct": period_delta, "seed_delta_S_pct": seed_pct,
            "seed_delta_S_pct_mean": float(np.mean(seed_pct)), "seed_delta_S_pct_std": float(np.std(seed_pct, ddof=1)) if len(seed_pct) > 1 else 0.0,
            "seed_delta_S_pct_worst": float(max(seed_pct)), "bootstrap": boot, "checks": checks,
            "passes_gate": all(checks.values())}


def arm_root(args, arm):
    if args.experiment != CONTROL and arm == "A" and not args.smoke:
        return args.control_out
    return args.out


def wait_for_control(args):
    marker = args.control_out / "lb_complete.json"
    waited = 0
    while not marker.exists():
        if waited == 0:
            log(experiment=args.experiment, stage="wait_control", control=str(args.control_out))
        if waited > args.control_timeout_h * 3600:
            raise TimeoutError("shared control A did not complete")
        time.sleep(120)
        waited += 120
    if (args.control_out / "train.exit").exists() and (args.control_out / "train.exit").read_text().strip() not in ("", "0"):
        raise RuntimeError("shared control failed")


def official_check(args, competition, primary_arm, periods):
    """Score the merged primary forecast with the leaderboard scorer and require agreement with score_arrays."""
    if args.smoke:
        return None
    official = competition.score_competition(pd.read_parquet(args.out / "predictions.parquet"), args.context)
    runs = collect_arm(args.out, primary_arm, (PRIMARY_SEED,), periods)[PRIMARY_SEED]
    local = score_arrays(runs["pred"], runs["truth"])["S"]
    if abs(local - official["total"]["s"]) > 1e-8:
        raise ValueError(f"local S {local} disagrees with the leaderboard scorer {official['total']['s']}")
    return {"arm": primary_arm, "seed": PRIMARY_SEED, "competition": official}


def evaluate(args, seeds, folds, draws, competition):
    if args.experiment == CONTROL:
        runs = collect_arm(args.out, "A", seeds, folds)
        result = {str(s): score_arrays(runs[s]["pred"], runs[s]["truth"]) for s in seeds}
        write_json(args.out / "evaluation.json", {"experiment": CONTROL, "arm": "A", "per_seed": result,
                                                  "evaluation": "leaderboard F1/F2/F3", "observations": args.observation_identity,
                                                  "official_primary": official_check(args, competition, "A", folds),
                                                  "S_mean": float(np.mean([r["S"] for r in result.values()])),
                                                  "S_std": float(np.std([r["S"] for r in result.values()], ddof=1)) if len(seeds) > 1 else 0.0})
        return None
    rule = DECISION[args.experiment]
    arms = {arm: collect_arm(arm_root(args, arm), arm, seeds, folds) for arm in ("A", *ARMS[args.experiment])}
    out = {}
    for cand, ctrl in [rule["primary"], *rule["also_required"], *rule["descriptive"]]:
        label = f"{cand}_vs_{ctrl}"
        out[label] = compare(arms[cand], arms[ctrl], seeds, folds, draws, label)
        log(experiment=args.experiment, stage="evaluate", comparison=label,
            delta_S_pct=round(out[label]["summary"]["S"]["delta_pct"], 4), ci95=out[label]["bootstrap"]["ci95"])
    primary = out["_vs_".join(rule["primary"])]
    required = [out["_vs_".join(pair)]["summary"]["S"]["delta_pct"] < 0 for pair in rule["also_required"]]
    adopt = primary["passes_gate"] and all(required)
    decision = {"rule": {"primary": "_vs_".join(rule["primary"]), "primary_gate": GATE,
                         "also_required_mean_delta_S_lt_0": ["_vs_".join(p) for p in rule["also_required"]]},
                "primary_passes_gate": primary["passes_gate"], "also_required_pass": required, "adopt": adopt,
                "evaluation": "leaderboard F1/F2/F3",
                "submission": "candidate (manual decision, not uploaded)" if adopt else "not a candidate: gate not passed"}
    write_json(args.out / "evaluation.json", {"experiment": args.experiment, "comparisons": out, "decision": decision,
                                              "observations": args.observation_identity,
                                              "official_primary": official_check(args, competition, "AB", folds)})
    seed_summary = {label: {"seed_delta_S_pct": c["seed_delta_S_pct"], "mean": c["seed_delta_S_pct_mean"],
                            "std": c["seed_delta_S_pct_std"], "worst": c["seed_delta_S_pct_worst"]} for label, c in out.items()}
    write_json(args.out / "seed_summary.json", seed_summary)
    log(experiment=args.experiment, stage="decision", adopt=adopt, primary_delta_S_pct=round(primary["summary"]["S"]["delta_pct"], 4))
    return decision


def merge(out, periods):
    frames = []
    for period in periods:
        z = np.load(out / f"{period}_predictions.npz")
        origins = pd.DatetimeIndex(z["origins"], tz="UTC")
        frames.append(pd.DataFrame({"origin_last_input_utc": np.repeat(origins.strftime("%Y-%m-%dT%H:%M:%SZ"), H),
                                    "horizon_hours": np.tile(np.arange(1, H + 1), len(origins)), "pred_kms": z["pred"].ravel()}))
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["origin_last_input_utc", "horizon_hours"]).any():
        raise ValueError("duplicate prediction keys")
    result.to_parquet(out / "predictions.parquet", index=False)
    return len(result)


# ----------------------------------------------------------------------------- provenance
def provenance(args, tables):
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()
    return {"experiment": args.experiment, "arms": {a: ARM_TITLES[(args.experiment, a)] for a in ARMS[args.experiment]},
            "parent": PARENT, "parent_source": str(PARENT_SOURCE), "source": file_identity(__file__),
            "git_commit": git("rev-parse", "HEAD"), "git_dirty_files": git("status", "--porcelain").splitlines(),
            "inputs": tables["identity"], "evaluation_periods": {p: OFFICIAL[p] for p in PERIODS},
            "observations": args.observation_identity, "maturity_hours": MATURITY_H, "seeds": list(args.seeds),
            "rounds": args.rounds, "threads": args.threads, "device": "cpu", "smoke": args.smoke,
            "parameters": {**spectral_parameters(args.threads, "per-seed"), "coefficient": .10, "scale": 50., "point4_threshold": 550., "clip": [1, 3000]},
            "decision_rule": DECISION.get(args.experiment), "gate": GATE,
            "bootstrap": {"draws": args.draws, "block_days": BOOT_BLOCK_DAYS, "seed": BOOT_SEED},
            "parent_features": {"constant": CONSTANT, "varying": VARYING, "block_columns": len(CONSTANT) + BLOCK * len(VARYING)},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "xgboost": xgb.__version__},
            "host": platform.node(), "argv": sys.argv}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", choices=list(ARMS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--control-out", type=Path, default=CAMPAIGN / "runs" / CONTROL)
    parser.add_argument("--control-timeout-h", type=float, default=48)
    parser.add_argument("--observations", type=Path, default=None,
                        help="leaderboard observation parquet (default: current tlabtlab/sunrun-lb-store revision)")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    args.rounds = 2 if args.smoke else ROUNDS
    args.seeds = (42,) if args.smoke else SEEDS
    args.draws = 50 if args.smoke else BOOT_DRAWS
    periods = ("F1",) if args.smoke else PERIODS
    began = time.perf_counter()
    competition, args.context = lb_context(args)
    tables = load_tables(args.data, args.raw)
    log(experiment=args.experiment, stage="loaded", train_origins=len(tables["train"]["origins"]),
        seconds=round(time.perf_counter() - began, 1))
    write_json(args.out / "run_manifest.json", provenance(args, tables))
    identity = {"experiment": args.experiment, "rounds": args.rounds, "smoke": args.smoke, "inputs": tables["identity"],
                "observations": args.observation_identity}
    inputs = Inputs(args.experiment, tables)
    own_arms = ARMS[args.experiment] if args.experiment == CONTROL or not args.smoke else ("A", *ARMS[args.experiment])
    primary_arm = "A" if args.experiment == CONTROL else "AB"
    units = len(args.seeds) * len(own_arms)
    for period in periods:
        unit = 0
        for seed in args.seeds:
            for arm in own_arms:
                lb_run(args, tables, inputs, arm, seed, period, {**identity, "arm": arm, "seed": seed, "period": period})
                unit += 1  # sunrun TUI progress keys (per leaderboard period)
                log(experiment=args.experiment, stage="lb", period=period, arm=arm, seed=seed, unit=unit, units=units)
        write_primary(args, period, primary_arm)
    log(prediction_rows=merge(args.out, periods))
    if args.experiment == CONTROL:
        write_json(args.out / "lb_complete.json", {"arms": ["A"], "seeds": list(args.seeds), "periods": list(periods)})
    elif not args.smoke:
        wait_for_control(args)
    decision = evaluate(args, args.seeds, periods, args.draws, competition)
    complete = {"status": "complete", "experiment": args.experiment, "evaluation": "leaderboard F1/F2/F3 only",
                "decision": decision, "seconds": time.perf_counter() - began}
    if decision is None or not decision["adopt"]:
        complete["publish"] = "not_required"  # read by the sunrun TUI
        complete["publish_reason"] = ("control arm: parent reproduction" if decision is None
                                      else "gate not passed: not a submission candidate")
    write_json(args.out / "complete.json", complete)
    log(experiment=args.experiment, stage="complete", seconds=round(time.perf_counter() - began, 1))


if __name__ == "__main__":
    main()
