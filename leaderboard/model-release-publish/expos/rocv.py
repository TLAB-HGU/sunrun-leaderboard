"""Rolling-origin x multi-seed evaluation (user directive 2026-10-06): the selection standard from now on.

Eight chronological 3-month folds over 2024-06..2026-05 (all preofficial). For fold k the runner trains on rows whose target time
is before fold_start - 72h and predicts every fold origin x 72h, once per seed, writing $EXPOS_OUT/rocv/f{k}_s{seed}.parquet
(columns origin_last_input_utc, horizon_hours, pred_kms). A gain counts only if it is larger than the seed noise:
  - seed-mean MSE improves vs the reference in >= ROCV_MIN_FOLDS of 8 folds,
  - mean relative MSE over folds <= -1%,
  - upper bound mean + 2*SE across seeds of the per-seed relative MSE is < 0 (needs >= 3 seeds),
  - pooled unique-event F1 (seed mean) >= the reference's pooled F1 (seed mean) under the same folds.
"""
import re

import numpy as np
import pandas as pd

from . import core, evaluate

FOLDS = [("2024-06-01", "2024-08-31 23:00"), ("2024-09-01", "2024-11-30 23:00"), ("2024-12-01", "2025-02-28 23:00"),
         ("2025-03-01", "2025-05-31 23:00"), ("2025-06-01", "2025-08-31 23:00"), ("2025-09-01", "2025-11-30 23:00"),
         ("2025-12-01", "2026-02-28 23:00"), ("2026-03-01", "2026-05-28 23:00")]
ROCV_MIN_FOLDS, ROCV_MEAN_MAX, MIN_SEEDS = 6, -0.01, 3
FILE = re.compile(r"f(\d+)_s(\d+)\.parquet$")


def fold_bounds(k):
    a, b = FOLDS[k]
    return pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")


def _files(eid):
    d = core.RUNS / eid / "rocv"
    out = {}
    for p in sorted(d.glob("f*_s*.parquet")) if d.exists() else []:
        m = FILE.search(p.name)
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = p
    return out


def score(eid):
    """Per (fold, seed): MSE on observed targets and event counts."""
    files = _files(eid)
    if not files:
        raise core.ExposError(f"{eid}: no rocv/f{{k}}_s{{seed}}.parquet files")
    folds = sorted({k for k, _ in files})
    if folds != list(range(len(FOLDS))):
        raise core.ExposError(f"{eid}: rocv needs all {len(FOLDS)} folds, got {folds}")
    seeds = sorted({s for _, s in files})
    missing = [(k, s) for k in folds for s in seeds if (k, s) not in files]
    if missing:
        raise core.ExposError(f"{eid}: every fold needs every seed; missing {missing[:5]}")
    filled, observed = evaluate._ace()
    res = {}
    for (k, s), p in files.items():
        df = evaluate._load(p)
        a, b = fold_bounds(k)
        df = df[df["origin"].between(a, b)]
        if df.empty:
            raise core.ExposError(f"{p.name}: no origins inside fold {k}")
        t = df["origin"] + pd.to_timedelta(df["h"], unit="h")
        y = filled.reindex(pd.DatetimeIndex(t)).to_numpy()
        obs = observed.reindex(pd.DatetimeIndex(t)).fillna(False).to_numpy()
        se = (df["y_pred"].to_numpy()[obs] - y[obs]) ** 2
        ev = evaluate._events(df[["origin", "h", "y_pred"]], filled, observed)
        res[(k, s)] = {"mse": float(se.mean()), "rows": int(obs.sum()), "tp": ev["tp"] or 0, "fp": ev["fp"] or 0, "fn": ev["fn"] or 0}
    return {"folds": folds, "seeds": seeds, "cells": res}


def _pooled_f1(cells, seed):
    tp = sum(v["tp"] for (k, s), v in cells.items() if s == seed)
    fp = sum(v["fp"] for (k, s), v in cells.items() if s == seed)
    fn = sum(v["fn"] for (k, s), v in cells.items() if s == seed)
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def summarise(sc):
    seeds, folds, cells = sc["seeds"], sc["folds"], sc["cells"]
    fold_mse = {k: float(np.mean([cells[(k, s)]["mse"] for s in seeds])) for k in folds}
    f1_by_seed = {s: _pooled_f1(cells, s) for s in seeds}
    return {"seeds": seeds, "fold_mse_seedmean": fold_mse,
            "mse_by_seed": {s: {k: cells[(k, s)]["mse"] for k in folds} for s in seeds},
            "f1_by_seed": f1_by_seed, "f1_seedmean": float(np.mean(list(f1_by_seed.values())))}


def evaluate_rocv(eid, reference):
    s = core.state()
    if eid not in s["experiments"] or reference not in s["experiments"]:
        raise core.ExposError("unknown experiment or reference")
    changed = core.check_protected(s)
    if changed:
        raise core.ExposError(f"protected files changed: {changed}")
    cand, ref = summarise(score(eid)), summarise(score(reference))
    folds = sorted(cand["fold_mse_seedmean"])
    rel_fold = {k: cand["fold_mse_seedmean"][k] / ref["fold_mse_seedmean"][k] - 1 for k in folds}
    rel_seed = [float(np.mean([cand["mse_by_seed"][sd][k] / ref["fold_mse_seedmean"][k] - 1 for k in folds]))
                for sd in cand["seeds"]]
    mean_rel = float(np.mean(list(rel_fold.values())))
    se = float(np.std(rel_seed, ddof=1) / np.sqrt(len(rel_seed))) if len(rel_seed) > 1 else float("inf")
    n_imp = sum(v < 0 for v in rel_fold.values())
    reasons = []
    if len(cand["seeds"]) < MIN_SEEDS:
        reasons.append(f"needs >= {MIN_SEEDS} seeds, got {len(cand['seeds'])}")
    if len(set(round(v, 12) for v in rel_seed)) == 1 and len(rel_seed) > 1:
        reasons.append("seeds produced identical results: seed variation was not exercised, seed noise cannot be estimated")
    if n_imp < ROCV_MIN_FOLDS:
        reasons.append(f"improves in {n_imp}/{len(folds)} folds (< {ROCV_MIN_FOLDS})")
    if mean_rel > ROCV_MEAN_MAX:
        reasons.append(f"mean relative {mean_rel:+.4f} > {ROCV_MEAN_MAX}")
    if not mean_rel + 2 * se < 0:
        reasons.append(f"gain not above seed noise: mean {mean_rel:+.4f} + 2*SE {se:.4f} >= 0")
    if cand["f1_seedmean"] < ref["f1_seedmean"]:
        reasons.append(f"pooled F1 seed-mean {cand['f1_seedmean']:.4f} < reference {ref['f1_seedmean']:.4f}")
    gate = {"pass": not reasons, "reasons": reasons, "reference": reference, "rel_by_fold": rel_fold,
            "rel_by_seed": rel_seed, "mean_relative": mean_rel, "seed_se": se, "n_folds_improved": n_imp,
            "f1_seedmean": cand["f1_seedmean"], "reference_f1_seedmean": ref["f1_seedmean"], "seeds": cand["seeds"]}
    return core.append("rocv", id=eid, gate=gate, candidate=cand, reference_summary=ref)
