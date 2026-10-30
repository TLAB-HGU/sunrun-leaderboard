"""Leaderboard-window evaluation (user directive #13, 2026-10-06): every experiment is judged on the same official window
the leaderboard scores (targets 2026-06-01..09-30, 2833 origins x 72h), with the leaderboard scorer and snapshot truth.

Runners never read truth: they write `predictions_official_s{seed}.parquet` (fit on targets < 2026-05-29) and only this
module reads the snapshot. Adoption (user-confirmed bar) needs:
  - >= 3 seeds,
  - seed-mean total MSE below the current best,
  - every seed's total MSE below the current best,
  - seed-mean unique-event F1 >= the champion's official F1,
  - seed-mean MSE of each of the four *_*_high cells not worse than the current best's (user decision 2026-10-06:
    a total-MSE win that comes from flattening and worsens the arrival cells is not adopted).
Current best = lowest seed-mean MSE among earlier lb-eval records that passed or were registered as baselines.
Hindsight-oracle diagnostics (ORACLE_IDS, or "ORACLE UPPER BOUND" in the hypothesis) never count as the current best.
Goal (user directive LB10:20261013, replaces total <= 3289.09 / four cells < 3800 / F1): seed-mean total MSE at least
GOAL_REL_GAIN (10%) below the current best (SOTA) before this evaluation. Cells and F1 are still reported.
"""
import json
import sys

import numpy as np
import pandas as pd

from . import core, evaluate

MIN_SEEDS = 3
NN_MIN_TRAIN_SECONDS = 1800  # directive #14: >= 30 min per seed unless early stopping shows convergence
ORACLE_IDS = {"E0474", "E0481"}  # hindsight IPS shock-arrival diagnostics (2026-10-08)
GOAL_REL_GAIN = 0.10  # LB10:20261013: goal = seed-mean total MSE <= (1 - 0.10) x SOTA


def nn_depth_reasons(eid, seeds):
    """sun-nn depth rule: every seed needs curves_s{seed}.json with per-epoch valid loss and either >= 30 min of training
    or early_stopped=true; otherwise the evaluation is screen-only."""
    out = core.RUNS / eid
    bad = []
    for sd in seeds:
        p = out / f"curves_s{sd}.json"
        try:
            c = json.loads(p.read_text())
            ep = c.get("epochs") or []
            ok_curve = len(ep) >= 2 and all("valid_loss" in e for e in ep)
            ok_budget = (c.get("train_seconds") or 0) >= NN_MIN_TRAIN_SECONDS or c.get("early_stopped") is True
        except (OSError, ValueError, AttributeError):
            ok_curve = ok_budget = False
        if not (ok_curve and ok_budget):
            bad.append(sd)
    return [f"screen-only: seeds {bad} lack curves_s{{seed}}.json with per-epoch valid_loss and "
            f"(train_seconds >= {NN_MIN_TRAIN_SECONDS} or early_stopped=true)"] if bad else []


def _files(eid):
    out = core.RUNS / eid
    seeded = sorted(out.glob("predictions_official_s*.parquet"), key=lambda p: int(p.stem.rsplit("_s", 1)[1]))
    if seeded:
        return {int(p.stem.rsplit("_s", 1)[1]): p for p in seeded}
    single = out / "predictions_official.parquet"
    return {None: single} if single.exists() else {}


def _snapshot():
    snap = core.Path(evaluate.OFFICIAL["snapshot"])
    return (pd.read_parquet(snap / "truth.parquet"), pd.read_parquet(snap / "folds.parquet"),
            pd.read_parquet(snap / "naive.parquet"))


def score_file(path, truth, folds, naive, filled, observed):
    sys.path.insert(0, str(core.LB))
    from scorer import scoring
    pred = pd.read_parquet(path)
    err = scoring.validate(pred, folds)
    if err:
        raise core.ExposError(f"{path.name}: leaderboard validator rejected: {err[:3]}")
    sc = scoring.score(pred, truth, folds, naive)
    return {"mse": float(sc["mse"]), "mse_observed": float(sc["mse_observed"]),
            "cells": {r: float(v["mse"]) for r, v in sc["by_regime"].items()},
            "events": evaluate._events(evaluate._load(path), filled, observed)}


def champion_f1(s):
    off = s.get("official") or {}
    f1 = (((off.get("results") or {}).get("champion") or {}).get("events") or {}).get("f1")
    if f1 is None:
        raise core.ExposError("champion F1 unknown: no official record")
    return float(f1)


def _oracle(x):
    return x.get("id") in ORACLE_IDS or "ORACLE UPPER BOUND" in (x.get("hypothesis") or "")


def best(s, before_seq=None):
    """Lowest seed-mean MSE among passing or baseline lb-eval records (optionally only those before a sequence number)."""
    recs = [r for x in s["experiments"].values() if not _oracle(x) for r in x.get("lbeval", [])
            if (r["gate"]["pass"] or r.get("baseline")) and (before_seq is None or r["seq"] < before_seq)]
    if not recs:
        return None
    r = min(recs, key=lambda r: r["summary"]["mse_seedmean"])
    return {"id": r["id"], "mse_seedmean": r["summary"]["mse_seedmean"], "seq": r["seq"],
            "cells_seedmean": {c: r["summary"]["cells_seedmean"].get(c) for c in evaluate.OFFICIAL["cells"]}}


def lb_eval(eid, baseline=False):
    s = core.state()
    x = s["experiments"].get(eid)
    if not x:
        raise core.ExposError(f"unknown experiment {eid}")
    changed = core.check_protected(s)
    if changed:
        raise core.ExposError(f"protected files changed: {changed}")
    files = _files(eid)
    if not files:
        raise core.ExposError(f"{eid}: no predictions_official_s{{seed}}.parquet (or predictions_official.parquet)")
    truth, folds, naive = _snapshot()
    filled, observed = evaluate._ace()
    per_seed = {("single" if k is None else str(k)): score_file(p, truth, folds, naive, filled, observed)
                for k, p in files.items()}
    mses = [v["mse"] for v in per_seed.values()]
    f1s = [v["events"]["f1"] or 0.0 for v in per_seed.values()]
    cells = {c: float(np.mean([v["cells"].get(c, np.nan) for v in per_seed.values()]))
             for c in next(iter(per_seed.values()))["cells"]}
    summary = {"n_seeds": len(per_seed), "mse_seedmean": float(np.mean(mses)), "mse_by_seed": mses,
               "mse_observed_seedmean": float(np.mean([v["mse_observed"] for v in per_seed.values()])),
               "f1_seedmean": float(np.mean(f1s)), "cells_seedmean": cells}
    ch_f1 = champion_f1(s)
    b = best(s)
    target = (1 - GOAL_REL_GAIN) * b["mse_seedmean"] if b else None
    goal = {"total_mse_10pct_vs_sota": target is not None and summary["mse_seedmean"] <= target}
    reasons = []
    if per_seed.keys() == {"single"} or len(per_seed) < MIN_SEEDS:
        reasons.append(f"needs >= {MIN_SEEDS} seed files predictions_official_s{{seed}}.parquet, got {len(per_seed)}")
    if len(per_seed) > 1 and len(set(round(m, 9) for m in mses)) == 1:
        reasons.append("seeds produced identical results: seed variation was not exercised")
    if b is not None:
        if summary["mse_seedmean"] >= b["mse_seedmean"]:
            reasons.append(f"seed-mean MSE {summary['mse_seedmean']:.2f} >= current best {b['mse_seedmean']:.2f} ({b['id']})")
        worse = [m for m in mses if m >= b["mse_seedmean"]]
        if worse:
            reasons.append(f"{len(worse)}/{len(mses)} seeds not below current best {b['mse_seedmean']:.2f}")
        for c, bc in (b.get("cells_seedmean") or {}).items():
            if bc is not None and not cells.get(c, np.inf) <= bc:
                reasons.append(f"high cell {c} {cells.get(c, np.nan):.1f} worse than current best {bc:.1f}")
    if x["owner"].startswith(core.NN_OWNER):
        reasons += nn_depth_reasons(eid, [k for k in files if k is not None] or ["single"])
    if summary["f1_seedmean"] < ch_f1:
        reasons.append(f"F1 seed-mean {summary['f1_seedmean']:.4f} < champion {ch_f1:.4f}")
    gate = {"pass": (not reasons) and not baseline, "reasons": reasons, "best_before": b, "champion_f1": ch_f1,
            "rel_vs_best": (summary["mse_seedmean"] / b["mse_seedmean"] - 1) if b else None,
            "goal_target_mse": target}
    return core.append("lbeval", id=eid, baseline=bool(baseline), per_seed=per_seed, summary=summary, gate=gate,
                       goal=goal, achieved=all(goal.values()))
