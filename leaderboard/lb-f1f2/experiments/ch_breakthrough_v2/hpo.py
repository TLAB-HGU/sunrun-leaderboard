"""Bounded multi-period HPO for ch_breakthrough_v2 (read-only lanes).

Searches >=16 trials EACH for two promising branches on dev-A/B/C +
Dec2025-Feb2026 selection only:

- ``recurrence``: target-recurrence+CH (horizon-specific trec, pooled fit).
- ``physical``: ACE+CH-quality (physical_ace_ch_quality, pooled fit).

Every trial uses identical candidate params across the four rolling splits,
all 72 horizons, pooled-horizon fits, bounded origin sampling (daily/6h,
<=4000-6000 origins). Selection uses chronological dev + selection ONLY;
holdout is evaluation (never tie-break); official truth is never read or
used. Matched controls are refit at the winning params for ablations.
Event feasibility (600km/s, gap<=6h, 12h tolerance via
``common.score_forecast`` -> ``events.passed``) takes precedence among
passing candidates, else the winner is labelled ``no-event-feasible`` with
best-MSE and NO success claim.

Only after confirming the four refreshed branches exist
(validated/gated, recurrence, propagation, physical_quality), ONE bounded
combined feature model (base ACE+CH + physics + quality + horizon trec) is
evaluated with a single dev/selection-chosen param set (no holdout/official
input, no retune after official).

Artifacts: ONLY ``store/ch-breakthrough-v2/validated/hpo/``.
Lanes (common/gated/recurrence/propagation/physical_quality) are imported
read-only and never modified.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.ch_breakthrough_v2"

from .common import (  # noqa: E402  (frozen foundation, read-only)
    HORIZONS,
    HORIZON_HOURS,
    N_THREADS,
    SEED,
    PooledResidualForecaster,
    as_speed_series,
    build_features,
    file_identity,
    load_ace_hourly,
    load_ch_hourly,
    observed_mask,
    purged_split,
    score_forecast,
    subsample_origins,
    timed_predict,
    utc,
    write_manifest,
)

SPLIT_CFG = {
    "dev-A": {"cutoff": "2024-12-31T23:00:00Z",
              "start": "2025-01-01T00:00:00Z", "end": "2025-03-31T23:00:00Z"},
    "dev-B": {"cutoff": "2025-04-30T23:00:00Z",
              "start": "2025-05-01T00:00:00Z", "end": "2025-07-31T23:00:00Z"},
    "dev-C": {"cutoff": "2025-08-31T23:00:00Z",
              "start": "2025-09-01T00:00:00Z", "end": "2025-11-30T23:00:00Z"},
    "selection": {"cutoff": "2025-11-30T23:00:00Z",
                  "start": "2025-12-01T00:00:00Z", "end": "2026-02-28T23:00:00Z"},
    "holdout": {"cutoff": "2026-02-28T23:00:00Z",
                "start": "2026-03-01T00:00:00Z", "end": "2026-05-31T23:00:00Z"},
}
SEARCH_SPLITS = ("dev-A", "dev-B", "dev-C", "selection")
HOLDOUT_SPLIT = "holdout"
# Official is NEVER touched by HPO (no read, no fit, no selection).
FORBIDDEN_OFFICIAL_CUTOFF = "2026-05-31T23:00:00Z"

BRANCHES = ("recurrence", "physical")
N_TRIALS = 18  # bounded: 16-24 per branch; fixed at 18 (no sweep)
RECURRENCE_ARM = "target-recurrence+CH"
PHYSICAL_ARM = "physical_ace_ch_quality"
RECURRENCE_CONTROLS = ("recent-ACE", "origin-recurrence", "target-recurrence")
PHYSICAL_CONTROLS = ("speed_only", "physical_ace")

DEADLINE_UTC = pd.Timestamp("2026-10-03T07:20:00Z")
DEFAULT_OUTDIR = "store/ch-breakthrough-v2/validated/hpo"


def candidate_grid(n=N_TRIALS) -> list[dict]:
    """Deterministic bounded candidate params (trees vary; not fixed 100-200)."""
    trees = [60, 80, 100, 120, 140, 160, 180, 200, 220, 250,
             70, 90, 110, 130, 170, 190, 210, 240,
             65, 95, 125, 155, 185, 215][:n]
    depths = [3, 4, 5, 6, 4, 5, 3, 6, 4, 5, 3, 4, 5, 6, 4, 3, 5, 6,
              4, 5, 3, 6, 4, 5][:n]
    lrs = [0.03, 0.05, 0.07, 0.10, 0.05, 0.07, 0.03, 0.10, 0.05, 0.07,
           0.04, 0.06, 0.08, 0.12, 0.05, 0.03, 0.07, 0.10,
           0.04, 0.06, 0.08, 0.05, 0.07, 0.10][:n]
    subs = [0.8, 0.9, 0.7, 1.0, 0.8, 0.9, 0.8, 0.7, 0.9, 0.8,
            0.85, 0.75, 0.9, 0.8, 0.85, 0.9, 0.8, 0.75,
            0.8, 0.9, 0.7, 0.85, 0.9, 0.8][:n]
    cols = [0.8, 0.7, 0.9, 0.6, 1.0, 0.8, 0.7, 0.9, 0.8, 0.7,
            0.85, 0.75, 0.8, 0.9, 0.7, 0.8, 0.85, 0.75,
            0.8, 0.7, 0.9, 0.85, 0.8, 0.75][:n]
    mcw = [4, 6, 2, 8, 10, 1, 4, 6, 3, 5,
           4, 7, 2, 9, 5, 6, 3, 8,
           4, 2, 6, 5, 7, 3][:n]
    lam = [1.0, 5.0, 0.5, 2.0, 8.0, 0.0, 3.0, 10.0, 1.5, 4.0,
           2.5, 6.0, 0.2, 7.0, 1.0, 3.5, 5.5, 0.8,
           2.0, 4.5, 1.2, 6.5, 3.0, 9.0][:n]
    out = []
    for i in range(n):
        out.append({
            "n_estimators": int(trees[i]),
            "max_depth": int(depths[i]),
            "learning_rate": float(lrs[i]),
            "subsample": float(subs[i]),
            "colsample_bytree": float(cols[i]),
            "min_child_weight": float(mcw[i]),
            "reg_lambda": float(lam[i]),
        })
    return out


def _past_deadline() -> bool:
    try:
        return pd.Timestamp.now(tz="UTC") >= DEADLINE_UTC
    except Exception:
        return False


def load_inputs(ace_paths, ch_path, raw_pattern):
    """Load + build all HPO feature frames once (causal, read-only lanes)."""
    from .physical_quality import (
        build_ch_quality_frame,
        build_physics_frame,
        load_staged_raw_ace,
        vintage_record,
    )
    from .recurrence import build_recurrence_features

    ace = load_ace_hourly(ace_paths)
    ch = load_ch_hourly(ch_path)
    raw = load_staged_raw_ace(raw_pattern)
    vintage = vintage_record(raw)
    speed = as_speed_series(ace)
    base = build_features(ace, ch, None).replace([np.inf, -np.inf], np.nan)
    rec_full = build_recurrence_features(ace, ch, None, include_ch=True)
    origins_all = base.index.intersection(rec_full.index).sort_values()
    # Physical frames need raw overlap; keep the shared grid causal.
    raw_hours = pd.DatetimeIndex(utc(raw.index)).floor("h")
    origins_all = origins_all[origins_all.isin(raw_hours)].sort_values()
    base = base.reindex(origins_all)
    rec_full = rec_full.reindex(origins_all)
    physics = build_physics_frame(raw, origins_all)
    quality = build_ch_quality_frame(ch, origins_all)
    observed = observed_mask(speed, speed.index)
    raw_files = sorted(glob.glob(raw_pattern))
    raw_parts = [file_identity(p) for p in raw_files]
    raw_digest = hashlib.sha256(
        "".join(p["sha256"] for p in raw_parts).encode()).hexdigest()
    ace_key = ace_paths[0] if isinstance(ace_paths, (list, tuple)) else ace_paths
    identities = {
        "ace": file_identity(ace_key),
        "ch_hourly": file_identity(ch_path),
        "raw_ace_years": {"pattern": raw_pattern, "files": raw_files,
                          "file_identities": raw_parts, "sha256": raw_digest,
                          "bytes": sum(p["bytes"] for p in raw_parts)},
    }
    return {"ace": ace, "ch": ch, "raw": raw, "vintage": vintage,
            "speed": speed, "observed": observed, "base": base,
            "rec_full": rec_full, "physics": physics, "quality": quality,
            "identities": identities}


def eval_origins(feature_index, split, step="daily", max_origins=120,
                 seed=SEED):
    """Purged (>=72h) eval origins for one split + recorded sampling policy."""
    cfg = SPLIT_CFG[split]
    cutoff = utc(cfg["cutoff"])
    grid = pd.date_range(utc(cfg["start"]), utc(cfg["end"]), freq="h")
    grid = pd.DatetimeIndex(sorted(set(grid).intersection(
        set(pd.DatetimeIndex(utc(feature_index))))))
    purge_after = cutoff + pd.Timedelta(hours=HORIZON_HOURS)
    grid = grid[(grid > purge_after)
                & (grid + pd.Timedelta(hours=HORIZON_HOURS) <= utc(cfg["end"]))]
    _, test = purged_split(grid, cutoff, end=utc(cfg["end"]),
                           horizon=HORIZON_HOURS, purge=72)
    if len(test) == 0:
        raise ValueError(f"{split}: no eval origins after purge")
    kept, sampling = subsample_origins(test, step=step,
                                       max_origins=max_origins, seed=seed)
    sampling = {"split_cutoff": cutoff.isoformat(), **sampling}
    return kept, sampling


def _arm_features(branch, arm, frames):
    from .physical_quality import arm_columns as phys_cols
    from .recurrence import arm_columns as rec_cols
    if branch == "recurrence":
        cols = rec_cols(frames["rec_full"], arm)
        return frames["rec_full"][cols], cols, True
    if branch == "physical":
        cols = phys_cols(frames["base"], frames["physics"],
                         frames["quality"], arm)
        parts = []
        for frame in (frames["base"], frames["physics"], frames["quality"]):
            hit = [c for c in cols if c in frame.columns]
            if hit:
                parts.append(frame[hit])
        feats = pd.concat(parts, axis=1)[cols].replace(
            [np.inf, -np.inf], np.nan).astype(np.float32, errors="ignore")
        return feats, cols, False
    if branch == "combined":
        cols = phys_cols(frames["base"], frames["physics"],
                         frames["quality"], PHYSICAL_ARM)
        parts = []
        for frame in (frames["base"], frames["physics"], frames["quality"]):
            hit = [c for c in cols if c in frame.columns]
            if hit:
                parts.append(frame[hit])
        feats = pd.concat(parts, axis=1)[cols].replace(
            [np.inf, -np.inf], np.nan).astype(np.float32, errors="ignore")
        return feats, cols, True  # trec added inside recurrence forecaster
    raise ValueError(f"unknown branch {branch}")


def fit_predict(branch, arm, params, frames, speed, cutoff, eval_origins_idx,
                device="cpu", seed=SEED, max_train_origins=600):
    """Pooled-horizon fit (all 72 horizons) + causal predict for one split."""
    feats, cols, use_target = _arm_features(branch, arm, frames)
    t0 = time.perf_counter()
    if use_target:
        from .recurrence import RecurrenceResidualForecaster
        fore = RecurrenceResidualForecaster(params=dict(params), device=device,
                                            seed=seed, use_target=True)
        fore.fit(feats, speed, cutoff, max_origins=max_train_origins)
    else:
        fore = PooledResidualForecaster(params=dict(params), device=device,
                                        seed=seed)
        fore.fit(feats, speed, cutoff, max_origins=max_train_origins)
    fit_secs = time.perf_counter() - t0
    preds, secs = timed_predict(
        lambda outs, f=fore, af=feats: f.predict(af, speed, outs),
        eval_origins_idx)
    assert set(preds["horizon_hours"]) == set(HORIZONS), "need all 72 horizons"
    keys = pd.MultiIndex.from_arrays(
        [utc(preds["origin_last_input_utc"]), preds["horizon_hours"]])
    assert keys.is_unique and len(keys) == len(eval_origins_idx) * 72
    return preds, fore, fit_secs, secs


def score_split(preds, speed, observed, eval_origins_idx):
    m = score_forecast(preds, speed, observed, origins=eval_origins_idx)
    ev = m.get("events", {}) or {}
    return {
        "mse": float(m["mse"]),
        "mse_observed": float(m.get("mse_observed")),
        "peak_macro_f1": m.get("peak_macro_f1"),
        "peak_bias": m.get("peak_bias"),
        "precision": ev.get("precision"),
        "recall": ev.get("recall"),
        "f1": ev.get("f1"),
        "height_mae_kms": ev.get("height_mae_kms"),
        "time_mae_hours": ev.get("time_mae_hours"),
        "events_passed": bool(ev.get("passed")),
        "imputed_fraction": m.get("imputed_fraction"),
        "unique_observed_peak_timestamps": m.get(
            "unique_observed_peak_timestamps"),
    }


def check_refreshed_branches(validated_root="store/ch-breakthrough-v2/validated"):
    """Confirm the four refreshed lane branches exist before any combined fit."""
    root = Path(validated_root)
    checks = {
        "gated": list((root / "gated").glob("*.summary.json")),
        "recurrence": [root / "recurrence" / "results.json",
                        root / "recurrence" / "summary.json"],
        "propagation": [root / "propagation" / "summary.json"],
        "physical_quality": [root / "physical_quality" / "comparison.json"],
    }
    missing = []
    for branch, paths in checks.items():
        found = [p for p in paths if Path(p).is_file()]
        if branch == "gated" and not found:
            missing.append("gated/*.summary.json")
        elif branch != "gated" and not any(Path(p).is_file() for p in paths):
            missing.append(f"{branch}/{Path(paths[0]).name}")
    return missing


def run_hpo(ace=("store/ch-v1/ace.parquet",), ch_path="store/ch-v1/ch-hourly.parquet",
            raw_pattern="/home/t-lab01/.local/state/sundb/stage/ace_solar/year=*/part.parquet",
            outdir=DEFAULT_OUTDIR, n_trials=N_TRIALS, origin_step="daily",
            max_eval_origins=120, max_train_origins=600, device="cpu",
            seed=SEED):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    started_iso = pd.Timestamp.now(tz="UTC").isoformat()
    t0 = time.perf_counter()
    status = {"status": "incomplete", "started_utc": started_iso}
    frames = load_inputs(list(ace), ch_path, raw_pattern)
    speed, observed = frames["speed"], frames["observed"]
    identities = frames["identities"]
    candidates = candidate_grid(n_trials)
    assert 16 <= len(candidates) <= 24, "protocol bounds 16-24 trials/branch"

    summary = {
        "protocol": {"splits": SPLIT_CFG, "search_splits": list(SEARCH_SPLITS),
                     "holdout": HOLDOUT_SPLIT,
                     "select_basis": ("mean MSE over dev-A/B/C + selection only; "
                                      "holdout is evaluation, never tie-break; "
                                      "official truth never read or used"),
                     "sampling_policy": {"step": origin_step,
                                         "max_eval_origins": max_eval_origins,
                                         "max_train_origins": max_train_origins,
                                         "horizons": "1..72 pooled, identical "
                                         "candidate params across rolling splits"},
                     "event_definition": ("600km/s, quiet gap<=6h, boundary-censored, "
                                          "one-to-one peak matching<=12h"),
                     "deadline_utc": DEADLINE_UTC.isoformat(),
                     "threads_per_fit": N_THREADS,
                     "concurrency": "serial (1 fit at a time, within 2-lease cap)"},
        "branches": {},
    }
    branch_arm = {"recurrence": RECURRENCE_ARM, "physical": PHYSICAL_ARM}
    for branch in BRANCHES:
        if _past_deadline():
            summary["branches"][branch] = {"status": "incomplete",
                                           "reason": "protocol 4h budget reached"}
            continue
        trials = []
        for t_idx, params in enumerate(candidates):
            if _past_deadline():
                trials.append({"trial": t_idx, "status": "incomplete",
                               "reason": "protocol 4h budget reached"})
                continue
            per_split, ok = {}, True
            for split in SEARCH_SPLITS:
                cfg = SPLIT_CFG[split]
                cutoff = utc(cfg["cutoff"])
                # Selection cutoff respected: never fit past the split cutoff
                # (forecaster enforces it); eval starts >=72h after cutoff.
                ev_idx, sampling = eval_origins(
                    frames["base"].index, split, step=origin_step,
                    max_origins=max_eval_origins, seed=seed)
                assert (ev_idx.min() - cutoff) >= pd.Timedelta(hours=72)
                preds, fore, fit_s, inf_s = fit_predict(
                    branch, branch_arm[branch], params, frames, speed,
                    cutoff, ev_idx, device=device, seed=seed,
                    max_train_origins=max_train_origins)
                scored = score_split(preds, speed, observed, ev_idx)
                name = f"{branch}_trial{t_idx:02d}_{split}"
                ppath = outdir / f"{name}.parquet"
                preds.to_parquet(ppath, index=False)
                mpath = outdir / f"{name}.metrics.json"
                mpath.write_text(json.dumps(
                    {"branch": branch, "trial": t_idx, "split": split,
                     "arm": branch_arm[branch], **scored,
                     "sampling": sampling},
                    indent=2, sort_keys=True, default=str) + "\n")
                extra = {"branch": branch, "trial": t_idx, "split": split,
                         "arm": branch_arm[branch], "use_target": branch != "physical",
                         "origins": len(ev_idx), "mse": scored["mse"],
                         "mse_observed": scored["mse_observed"],
                         "peak_macro_f1": scored["peak_macro_f1"],
                         "peak_bias": scored["peak_bias"],
                         "events_passed": scored["events_passed"],
                         "eval_window": [cfg["start"], cfg["end"]],
                         "fit_seconds": fit_s,
                         "max_train_origins": max_train_origins,
                         "vintage_verified": False,
                         "selection_use": "dev/selection only; holdout never "
                         "tie-break; official never used"}
                write_manifest(outdir / f"{name}.manifest.json", mode=name,
                               train_cutoff=cutoff, params=params,
                               tree_counts={"n_estimators": params["n_estimators"]},
                               columns=list(fore.columns),
                               input_identities=identities, seed=seed,
                               inference_seconds_per_origin=inf_s,
                               sampling=sampling, extra=extra,
                               caller_file=__file__,
                               actual_device=fore.actual_device,
                               prediction_path=ppath)
                per_split[split] = {"mse": scored["mse"],
                                    "mse_observed": scored["mse_observed"],
                                    "peak_macro_f1": scored["peak_macro_f1"],
                                    "peak_bias": scored["peak_bias"],
                                    "events_passed": scored["events_passed"],
                                    "n_origins": len(ev_idx),
                                    "fit_seconds": fit_s,
                                    "secs_per_origin": inf_s}
            if not ok:
                continue
            mses = [per_split[s]["mse"] for s in SEARCH_SPLITS]
            select_score = float(np.mean(mses))
            trials.append({"trial": t_idx, "status": "complete",
                           "params": dict(params),
                           "per_split": per_split,
                           "select_score_dev_selection_mean": select_score,
                           "feasible_selection": bool(
                               per_split["selection"]["events_passed"])})
        completed = [t for t in trials if t.get("status") == "complete"]
        if completed:
            feasible = [t for t in completed if t["feasible_selection"]]
            pool = feasible if feasible else completed
            winner = min(pool, key=lambda t: t["select_score_dev_selection_mean"])
            feasibility = ("event-feasible" if feasible
                           else "no-event-feasible")
        else:
            winner, feasibility = None, "no-event-feasible"
        summary["branches"][branch] = {
            "arm": branch_arm[branch],
            "n_trials_requested": len(candidates),
            "n_trials_complete": len(completed),
            "trials": trials,
            "winner": winner,
            "feasibility_label": feasibility,
            # Without an event-feasible candidate there is no success claim.
            "success_claimed": False,
            "selection_note": ("event-feasible candidates take precedence; else "
                               "best-MSE without success claim; holdout/official "
                               "never used for selection"),
        }
        (outdir / "summary.json").write_text(json.dumps(
            summary, indent=2, sort_keys=True, default=str) + "\n")

    # Matched-control ablations at winning params (dev/selection/holdout).
    for branch in BRANCHES:
        info = summary["branches"].get(branch, {})
        winner = (info or {}).get("winner")
        if not winner:
            continue
        controls = (RECURRENCE_CONTROLS if branch == "recurrence"
                    else PHYSICAL_CONTROLS)
        wparams = winner["params"]
        abl = {}
        for arm in controls:
            if _past_deadline():
                abl[arm] = {"status": "incomplete",
                            "reason": "protocol 4h budget reached"}
                continue
            per_split = {}
            for split in (*SEARCH_SPLITS, HOLDOUT_SPLIT):
                if _past_deadline():
                    break
                cfg = SPLIT_CFG[split]
                cutoff = utc(cfg["cutoff"])
                ev_idx, sampling = eval_origins(
                    frames["base"].index, split, step=origin_step,
                    max_origins=max_eval_origins, seed=seed)
                preds, fore, fit_s, inf_s = fit_predict(
                    branch, arm, wparams, frames, speed, cutoff, ev_idx,
                    device=device, seed=seed,
                    max_train_origins=max_train_origins)
                scored = score_split(preds, speed, observed, ev_idx)
                name = f"{branch}_control_{arm.replace('+', 'plus')}_{split}"
                ppath = outdir / f"{name}.parquet"
                preds.to_parquet(ppath, index=False)
                (outdir / f"{name}.metrics.json").write_text(json.dumps(
                    {"branch": branch, "arm": arm, "split": split,
                     "matched_params": True, **scored},
                    indent=2, sort_keys=True, default=str) + "\n")
                write_manifest(outdir / f"{name}.manifest.json", mode=name,
                               train_cutoff=cutoff, params=wparams,
                               tree_counts={"n_estimators": wparams["n_estimators"]},
                               columns=list(fore.columns),
                               input_identities=identities, seed=seed,
                               inference_seconds_per_origin=inf_s,
                               sampling=sampling,
                               extra={"branch": branch, "arm": arm,
                                      "control": True,
                                      "matched_params_of_winner_trial":
                                      winner["trial"],
                                      "split": split, "mse": scored["mse"],
                                      "events_passed": scored["events_passed"],
                                      "holdout_role": ("evaluation only"
                                                       if split == HOLDOUT_SPLIT
                                                       else "dev/selection")},
                               caller_file=__file__,
                               actual_device=fore.actual_device,
                               prediction_path=ppath)
                per_split[split] = {"mse": scored["mse"],
                                    "events_passed": scored["events_passed"]}
            abl[arm] = {"matched_params": True, "per_split": per_split}
        summary["branches"][branch]["ablations_matched_control"] = abl
        (outdir / "summary.json").write_text(json.dumps(
            summary, indent=2, sort_keys=True, default=str) + "\n")

    # Winner holdout evaluation (refit through holdout cutoff; never selection).
    for branch in BRANCHES:
        info = summary["branches"].get(branch, {})
        winner = (info or {}).get("winner")
        if not winner or _past_deadline():
            continue
        cfg = SPLIT_CFG[HOLDOUT_SPLIT]
        cutoff = utc(cfg["cutoff"])
        ev_idx, sampling = eval_origins(
            frames["base"].index, HOLDOUT_SPLIT, step=origin_step,
            max_origins=max_eval_origins, seed=seed)
        preds, fore, fit_s, inf_s = fit_predict(
            branch, branch_arm[branch], winner["params"], frames, speed,
            cutoff, ev_idx, device=device, seed=seed,
            max_train_origins=max_train_origins)
        scored = score_split(preds, speed, observed, ev_idx)
        name = f"{branch}_trial{winner['trial']:02d}_{HOLDOUT_SPLIT}_winner"
        ppath = outdir / f"{name}.parquet"
        preds.to_parquet(ppath, index=False)
        (outdir / f"{name}.metrics.json").write_text(json.dumps(
            {"branch": branch, "trial": winner["trial"],
             "split": HOLDOUT_SPLIT, "role": "evaluation only",
             **scored}, indent=2, sort_keys=True, default=str) + "\n")
        write_manifest(outdir / f"{name}.manifest.json", mode=name,
                       train_cutoff=cutoff, params=winner["params"],
                       tree_counts={"n_estimators":
                                    winner["params"]["n_estimators"]},
                       columns=list(fore.columns),
                       input_identities=identities, seed=seed,
                       inference_seconds_per_origin=inf_s, sampling=sampling,
                       extra={"branch": branch, "trial": winner["trial"],
                              "split": HOLDOUT_SPLIT,
                              "role": "holdout evaluation only, never selection",
                              "mse": scored["mse"],
                              "events_passed": scored["events_passed"]},
                       caller_file=__file__, actual_device=fore.actual_device,
                       prediction_path=ppath)
        summary["branches"][branch]["holdout_winner_eval"] = {
            "mse": scored["mse"], "mse_observed": scored["mse_observed"],
            "peak_macro_f1": scored["peak_macro_f1"],
            "peak_bias": scored["peak_bias"],
            "events_passed": scored["events_passed"]}
        (outdir / "summary.json").write_text(json.dumps(
            summary, indent=2, sort_keys=True, default=str) + "\n")

    # ONE bounded combined model, only after four branches confirmed.
    missing = check_refreshed_branches()
    if missing:
        summary["combined"] = {"status": "skipped",
                               "reason": f"refreshed branches missing: {missing}"}
    elif _past_deadline():
        summary["combined"] = {"status": "incomplete",
                               "reason": "protocol 4h budget reached"}
    else:
        rec_w = (summary["branches"]["recurrence"].get("winner") or {})
        phy_w = (summary["branches"]["physical"].get("winner") or {})
        if rec_w and phy_w:
            pick = ("recurrence" if rec_w["select_score_dev_selection_mean"]
                    <= phy_w["select_score_dev_selection_mean"] else "physical")
            comb_params = dict((rec_w if pick == "recurrence" else phy_w)["params"])
            basis = (f"dev/selection-chosen: {pick} winner trial "
                     f"{(rec_w if pick == 'recurrence' else phy_w)['trial']} "
                     f"(select_score={min(rec_w['select_score_dev_selection_mean'], phy_w['select_score_dev_selection_mean']):.2f}); "
                     f"no holdout/official input; single bounded config, no retune")
        elif rec_w:
            comb_params, basis = dict(rec_w["params"]), "dev/selection-chosen: recurrence winner"
        elif phy_w:
            comb_params, basis = dict(phy_w["params"]), "dev/selection-chosen: physical winner"
        else:
            comb_params, basis = None, "no winners available"
        if comb_params is None:
            summary["combined"] = {"status": "skipped",
                                   "reason": "no branch winners available"}
        else:
            per_split = {}
            for split in (*SEARCH_SPLITS, HOLDOUT_SPLIT):
                if _past_deadline():
                    per_split[split] = {"status": "incomplete",
                                        "reason": "protocol 4h budget reached"}
                    continue
                cfg = SPLIT_CFG[split]
                cutoff = utc(cfg["cutoff"])
                ev_idx, sampling = eval_origins(
                    frames["base"].index, split, step=origin_step,
                    max_origins=max_eval_origins, seed=seed)
                preds, fore, fit_s, inf_s = fit_predict(
                    "combined", PHYSICAL_ARM, comb_params, frames, speed,
                    cutoff, ev_idx, device=device, seed=seed,
                    max_train_origins=max_train_origins)
                scored = score_split(preds, speed, observed, ev_idx)
                name = f"combined_trial00_{split}"
                ppath = outdir / f"{name}.parquet"
                preds.to_parquet(ppath, index=False)
                (outdir / f"{name}.metrics.json").write_text(json.dumps(
                    {"branch": "combined", "split": split,
                     "role": ("evaluation only" if split == HOLDOUT_SPLIT
                              else "dev/selection"), **scored},
                    indent=2, sort_keys=True, default=str) + "\n")
                write_manifest(outdir / f"{name}.manifest.json", mode=name,
                               train_cutoff=cutoff, params=comb_params,
                               tree_counts={"n_estimators":
                                              comb_params["n_estimators"]},
                               columns=list(fore.columns),
                               input_identities=identities, seed=seed,
                               inference_seconds_per_origin=inf_s,
                               sampling=sampling,
                               extra={"branch": "combined",
                                      "features": ("base ACE+CH + physics + "
                                                   "quality + horizon trec"),
                                      "param_basis": basis,
                                      "split": split, "mse": scored["mse"],
                                      "events_passed": scored["events_passed"],
                                      "no_retune_after_official": True},
                               caller_file=__file__,
                               actual_device=fore.actual_device,
                               prediction_path=ppath)
                per_split[split] = {"mse": scored["mse"],
                                    "mse_observed": scored["mse_observed"],
                                    "peak_macro_f1": scored["peak_macro_f1"],
                                    "peak_bias": scored["peak_bias"],
                                    "events_passed": scored["events_passed"]}
            summary["combined"] = {"status": "complete",
                                   "params": comb_params,
                                   "param_basis": basis,
                                   "per_split": per_split,
                                   "feasibility_label": (
                                       "event-feasible"
                                       if per_split.get("selection", {}).get(
                                           "events_passed") else
                                       "no-event-feasible"),
                                   "success_claimed": False}

    done = all((summary["branches"].get(b, {}).get("n_trials_complete", 0) >= 16)
               for b in BRANCHES)
    status = {"status": "complete" if done else "incomplete",
              "started_utc": started_iso,
              "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
              "elapsed_seconds": time.perf_counter() - t0,
              "n_trials_complete": {b: summary["branches"].get(b, {}).get(
                  "n_trials_complete") for b in BRANCHES}}
    if _past_deadline() and not done:
        status["reason"] = "protocol 4h budget reached (measured incomplete)"
    summary["status"] = status["status"]
    (outdir / "summary.json").write_text(json.dumps(
        summary, indent=2, sort_keys=True, default=str) + "\n")
    (outdir / "status.json").write_text(json.dumps(
        status, indent=2, sort_keys=True, default=str) + "\n")
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ace", nargs="+", default=["store/ch-v1/ace.parquet"])
    p.add_argument("--ch-hourly", default="store/ch-v1/ch-hourly.parquet")
    p.add_argument("--raw-pattern",
                   default="/home/t-lab01/.local/state/sundb/stage/ace_solar/year=*/part.parquet")
    p.add_argument("--output-dir", default=DEFAULT_OUTDIR)
    p.add_argument("--n-trials", type=int, default=N_TRIALS)
    p.add_argument("--origin-step", default="daily")
    p.add_argument("--max-eval-origins", type=int, default=120)
    p.add_argument("--max-train-origins", type=int, default=600)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=SEED)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    summary = run_hpo(ace=tuple(args.ace), ch_path=args.ch_hourly,
                      raw_pattern=args.raw_pattern, outdir=args.output_dir,
                      n_trials=args.n_trials, origin_step=args.origin_step,
                      max_eval_origins=args.max_eval_origins,
                      max_train_origins=args.max_train_origins,
                      device=args.device, seed=args.seed)
    print(json.dumps({"status": summary.get("status"),
                      "branches": {b: {"complete": summary["branches"][b].get(
                          "n_trials_complete"),
                          "feasibility": summary["branches"][b].get(
                          "feasibility_label")} for b in BRANCHES},
                      "combined": summary.get("combined", {}).get("status")},
                     indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
