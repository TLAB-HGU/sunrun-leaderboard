"""Frozen GPU HPO confirmation for ch_breakthrough_v2 (Stage1).

Two-branch bounded search on dev-A/B/C + Dec2025-Feb2026 selection ONLY:

- ``recurrence``: target-recurrence+CH (horizon-specific trec, pooled fit).
- ``physical``: ACE+CH-quality (physical_ace_ch_quality, pooled fit).

Reuses frozen ``hpo.candidate_grid(n=18)``, ``hpo.SPLIT_CFG`` cutoffs,
``hpo.eval_origins``/``hpo.fit_predict``/``hpo.score_split`` per-split fitting
interfaces and ``common`` foundation read-only. No new dependency.

Every trial: all 72 horizons, same candidate params across the four rolling
splits, train cap 6000 origins (all if fewer), eval grid 6h (cap 120,
seed 42). Two branch searches run concurrently (at most 2 concurrent GPU
fits) via ``common.acquire_gpu_lease``; each fit resolves an actual
``cuda:N`` string before touching XGBoost (never passes ``device=auto``
unchanged), fail-closed probes via ``common.ensure_device``, and raises
visibly on any GPU bug (never silent CPU fallback). 4 CPU threads per fit
via ``common.N_THREADS``.

Winner per branch: mean dev+selection MSE with event-feasibility precedence.
Peak gate (via ``common.score_forecast`` 600km/s, gap<=6h, +-12h matching):
>=10 UNIQUE genuinely observed peaks, P/R>=0.60, heightMAE<=60. No winner
gets a success claim; ``no-event-feasible`` is labelled explicitly when no
candidate passes. ``frozen-choice.json`` is written BEFORE any ablation or
combined fit, and this module never touches holdout/official.

Matched ablations refit at winner params (recurrence: recent-ACE /
origin-recurrence / target-recurrence; physical: speed_only / physical_ace).
ONE numerical combined model (base ACE+CH + physics + quality + horizon
target-relative recurrence, no imageNN) uses the dev/selection-chosen winner
paramset; ``combined-choice.json`` is written BEFORE the combined fits.

Artifacts: ONLY ``store/ch-breakthrough-v2/validated/hpo-gpu/``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.ch_breakthrough_v2"

from .common import (  # noqa: E402 (frozen foundation, read-only)
    HORIZON_HOURS,
    HORIZONS,
    N_THREADS,
    SEED,
    acquire_gpu_lease,
    code_hash,
    ensure_device,
    runtime_versions,
    utc,
    write_manifest,
)
from .hpo import (  # noqa: E402 (frozen HPO protocol, read-only)
    BRANCHES,
    N_TRIALS,
    PHYSICAL_ARM,
    PHYSICAL_CONTROLS,
    RECURRENCE_ARM,
    RECURRENCE_CONTROLS,
    SEARCH_SPLITS,
    SPLIT_CFG,
    candidate_grid,
    eval_origins,
    fit_predict,
    load_inputs,
    score_split,
)

DEFAULT_OUTDIR = "store/ch-breakthrough-v2/validated/hpo-gpu"
MAX_TRAIN_ORIGINS = 6000
MAX_EVAL_ORIGINS = 120
ORIGIN_STEP = "6h"

BRANCH_ARM = {"recurrence": RECURRENCE_ARM, "physical": PHYSICAL_ARM}

EVENT_GATE_UNIQUE_PEAKS = 10
EVENT_GATE_PR = 0.60
EVENT_GATE_HEIGHT_MAE = 60.0


def ram_snapshot() -> dict:
    """Best-effort RAM snapshot without new dependencies."""
    info: dict = {}
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    info["mem_available_mb"] = int(line.split()[1]) / 1024.0
                    break
    except Exception:
        pass
    try:
        # ru_maxrss is KiB on Linux.
        info["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        pass
    try:
        info["loadavg_1m"] = float(os.getloadavg()[0])
    except Exception:
        pass
    return info


def source_hashes() -> dict:
    """Dependency hashes: common/hpo/recurrence/physical + this runner."""
    here = Path(__file__).resolve()
    root = here.parent
    targets = {
        "common": root / "common.py",
        "hpo": root / "hpo.py",
        "recurrence": root / "recurrence.py",
        "physical_quality": root / "physical_quality.py",
        "hpo_gpu": here,
    }
    out = {}
    for name, path in targets.items():
        try:
            out[name] = {
                "path": str(path),
                "sha256": code_hash(path),
                "bytes": path.stat().st_size,
            }
        except Exception as exc:
            out[name] = {"path": str(path), "error": str(exc)}
    return out


def is_event_feasible(scored: dict) -> bool:
    """Explicit peak gate on genuinely observed peaks via score_forecast.

    Requires >=10 UNIQUE genuinely observed peak timestamps,
    precision/recall >= 0.60, height MAE <= 60 km/s. The +-12h timing
    tolerance is baked into ``common.score_forecast`` one-to-one matching.
    """
    try:
        n_unique = scored.get("unique_observed_peak_timestamps")
        prec = scored.get("precision")
        rec = scored.get("recall")
        hmae = scored.get("height_mae_kms")
        if n_unique is None or int(n_unique) < EVENT_GATE_UNIQUE_PEAKS:
            return False
        if prec is None or rec is None or hmae is None:
            return False
        return (
            float(prec) >= EVENT_GATE_PR
            and float(rec) >= EVENT_GATE_PR
            and float(hmae) <= EVENT_GATE_HEIGHT_MAE
        )
    except Exception:
        return False


def train_pool_stats(feature_index, speed: pd.Series, cutoff, cap: int) -> dict:
    """Mirror forecaster train-pool filtering to record cap/requested/retained."""
    cutoff = utc(cutoff)
    idx = pd.DatetimeIndex(utc(feature_index)).sort_values()
    pool = idx[(idx <= cutoff) & (idx + pd.Timedelta(hours=HORIZON_HOURS) <= cutoff)]
    requested = int(len(pool))
    try:
        depth = speed.sort_index().rolling(648, min_periods=648).mean()
        skipped = int(depth.reindex(pool).isna().sum())
    except Exception:
        skipped = 0
    retained = max(0, requested - skipped)
    if cap is not None:
        retained = min(retained, int(cap))
    return {
        "cap": int(cap) if cap is not None else None,
        "requested": requested,
        "short_history_skipped": skipped,
        "retained": int(retained),
    }


def fit_predict_gpu(branch, arm, params, frames, speed, cutoff, ev_idx,
                    seed=SEED, max_train_origins=MAX_TRAIN_ORIGINS) -> tuple:
    """Pooled fit + predict holding a GPU lease; fail visibly, never CPU."""
    ram_before = ram_snapshot()
    with acquire_gpu_lease() as gpu:
        device = f"cuda:{gpu}"
        if device == "auto" or "auto" in device:
            raise RuntimeError(f"refusing to pass unresolved device {device!r} to XGBoost")
        # Fail-closed CUDA probe: raises visibly when the card is unusable.
        ensure_device(device)
        preds, fore, fit_secs, inf_secs = fit_predict(
            branch, arm, params, frames, speed, cutoff, ev_idx,
            device=device, seed=seed, max_train_origins=max_train_origins)
        actual = str(getattr(fore, "actual_device", device))
        if not actual.startswith("cuda"):
            raise RuntimeError(
                f"GPU fallback detected: requested {device!r} but actual {actual!r}")
        if actual != device:
            raise RuntimeError(
                f"GPU device mismatch: requested {device!r} but actual {actual!r}")
        gpu_info = {"lease_gpu": int(gpu), "device": device,
                    "actual_device": actual}
    ram_after = ram_snapshot()
    return preds, fore, fit_secs, inf_secs, gpu_info, ram_before, ram_after


def _write_trial_artifacts(*, outdir: Path, branch: str, trial: int, split: str,
                           arm: str, params: dict, preds: pd.DataFrame,
                           fore, fit_secs: float, inf_secs: float,
                           scored: dict, ev_idx, sampling: dict, cutoff,
                           identities: dict, seed: int, gpu_info: dict,
                           ram_before: dict, ram_after: dict,
                           train_stats: dict, use_target: bool,
                           max_train_origins: int = MAX_TRAIN_ORIGINS,
                           max_eval_origins: int = MAX_EVAL_ORIGINS,
                           extra_role: str = "dev/selection") -> dict:
    # All-72-horizons evidence: unique (origin, horizon) keys.
    horizons = sorted(set(preds["horizon_hours"].tolist()))
    assert horizons == list(HORIZONS), f"need all 72 horizons, got {horizons[:8]}..."
    keys = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(utc(preds["origin_last_input_utc"])),
         preds["horizon_hours"]])
    assert keys.is_unique, "duplicate forecast keys"
    assert len(keys) == len(ev_idx) * 72, "need len(origins)*72 unique keys"
    name = f"{branch}_trial{trial:02d}_{split}"
    ppath = outdir / f"{name}.parquet"
    preds.to_parquet(ppath, index=False)
    pred_sha = hashlib.sha256(ppath.read_bytes()).hexdigest()
    metrics = {
        "branch": branch, "trial": int(trial), "split": split, "arm": arm,
        **scored,
        "feasible_explicit": bool(is_event_feasible(scored)),
        "n_origins": int(len(ev_idx)),
        "n_keys": int(len(keys)),
        "horizons_complete_1_72": True,
        "sampling": dict(sampling),
        "train": dict(train_stats),
        "lease_gpu": gpu_info["lease_gpu"],
        "actual_device": gpu_info["actual_device"],
    }
    (outdir / f"{name}.metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n")
    sampling_rec = dict(sampling)
    sampling_rec["max_eval_origins"] = int(max_eval_origins)
    extra = {
        "branch": branch, "trial": int(trial), "split": split, "arm": arm,
        "use_target": bool(use_target),
        "origins": int(len(ev_idx)),
        "n_keys": int(len(keys)),
        "horizons_complete_1_72": True,
        "mse": scored["mse"],
        "mse_observed": scored["mse_observed"],
        "events_passed": scored["events_passed"],
        "feasible_explicit": bool(is_event_feasible(scored)),
        "unique_observed_peak_timestamps": scored.get("unique_observed_peak_timestamps"),
        "precision": scored.get("precision"),
        "recall": scored.get("recall"),
        "height_mae_kms": scored.get("height_mae_kms"),
        "time_mae_hours": scored.get("time_mae_hours"),
        "eval_window": [SPLIT_CFG[split]["start"], SPLIT_CFG[split]["end"]],
        "fit_seconds": float(fit_secs),
        "max_train_origins": int(max_train_origins),
        "train_cap": train_stats.get("cap"),
        "train_requested": train_stats.get("requested"),
        "train_retained": train_stats.get("retained"),
        "train_short_history_skipped": train_stats.get("short_history_skipped"),
        "lease_gpu": gpu_info["lease_gpu"],
        "device_requested": gpu_info["device"],
        "ram_before": ram_before,
        "ram_after": ram_after,
        "source_hashes": source_hashes(),
        "threads_per_fit": N_THREADS,
        "role": extra_role,
        "selection_use": ("dev/selection only; holdout never tie-break; "
                          "official never used"),
        "vintage_verified": False,
    }
    manifest = write_manifest(
        outdir / f"{name}.manifest.json", mode=name,
        train_cutoff=cutoff, params=params,
        tree_counts={"n_estimators": params["n_estimators"]},
        columns=list(fore.columns),
        input_identities=identities, seed=seed,
        inference_seconds_per_origin=inf_secs,
        sampling=sampling_rec, extra=extra,
        caller_file=str(Path(__file__).resolve()),
        actual_device=fore.actual_device,
        prediction_path=ppath)
    assert manifest.get("prediction", {}).get("sha256") == pred_sha
    return {"metrics": metrics, "fit_seconds": float(fit_secs),
            "secs_per_origin": float(inf_secs),
            "actual_device": gpu_info["actual_device"],
            "lease_gpu": gpu_info["lease_gpu"]}


def run_branch_worker(branch: str, outdir_str: str, ace, ch_path: str,
                      raw_pattern: str, seed: int = SEED,
                      n_trials: int = N_TRIALS,
                      origin_step: str = ORIGIN_STEP,
                      max_eval_origins: int = MAX_EVAL_ORIGINS,
                      max_train_origins: int = MAX_TRAIN_ORIGINS) -> dict:
    """Run all trials x SEARCH_SPLITS for one branch (one process)."""
    outdir = Path(outdir_str)
    outdir.mkdir(parents=True, exist_ok=True)
    arm = BRANCH_ARM[branch]
    use_target = branch != "physical"
    frames = load_inputs(list(ace), ch_path, raw_pattern)
    speed, observed = frames["speed"], frames["observed"]
    identities = frames["identities"]
    candidates = candidate_grid(n_trials)
    assert 16 <= len(candidates) <= 24, "protocol bounds 16-24 trials/branch"
    # Same eval origins for every trial of this branch (deterministic reuse).
    eval_cache: dict = {}
    for split in SEARCH_SPLITS:
        ev_idx, sampling = eval_origins(
            frames["base"].index, split, step=origin_step,
            max_origins=max_eval_origins, seed=seed)
        cfg = SPLIT_CFG[split]
        cutoff = utc(cfg["cutoff"])
        assert (ev_idx.min() - cutoff) >= pd.Timedelta(hours=72), \
            f"{split}: purge gap violated"
        eval_cache[split] = (ev_idx, sampling, cutoff)
    # Resolve per-split feature frames once (column selection is static).
    trials = []
    for t_idx, params in enumerate(candidates):
        per_split: dict = {}
        ok = True
        error = None
        for split in SEARCH_SPLITS:
            ev_idx, sampling, cutoff = eval_cache[split]
            tstats = train_pool_stats(frames["base"].index, speed, cutoff,
                                      max_train_origins)
            try:
                preds, fore, fit_s, inf_s, gpu_info, ram_b, ram_a = fit_predict_gpu(
                    branch, arm, params, frames, speed, cutoff, ev_idx,
                    seed=seed, max_train_origins=max_train_origins)
            except Exception as exc:  # fail visibly per trial, never CPU retry
                ok = False
                error = f"{type(exc).__name__}: {exc}"
                per_split[split] = {"status": "failed", "error": error}
                break
            scored = score_split(preds, speed, observed, ev_idx)
            rec = _write_trial_artifacts(
                outdir=outdir, branch=branch, trial=t_idx, split=split,
                arm=arm, params=params, preds=preds, fore=fore,
                fit_secs=fit_s, inf_secs=inf_s, scored=scored,
                ev_idx=ev_idx, sampling=sampling, cutoff=cutoff,
                identities=identities, seed=seed, gpu_info=gpu_info,
                ram_before=ram_b, ram_after=ram_a, train_stats=tstats,
                use_target=use_target,
                max_train_origins=max_train_origins,
                max_eval_origins=max_eval_origins)
            per_split[split] = {
                "mse": scored["mse"],
                "mse_observed": scored["mse_observed"],
                "precision": scored["precision"],
                "recall": scored["recall"],
                "height_mae_kms": scored["height_mae_kms"],
                "time_mae_hours": scored["time_mae_hours"],
                "events_passed": scored["events_passed"],
                "feasible_explicit": bool(is_event_feasible(scored)),
                "unique_observed_peak_timestamps": scored.get(
                    "unique_observed_peak_timestamps"),
                "n_origins": len(ev_idx),
                "n_keys": rec["metrics"]["n_keys"],
                "fit_seconds": rec["fit_seconds"],
                "secs_per_origin": rec["secs_per_origin"],
                "actual_device": rec["actual_device"],
                "lease_gpu": rec["lease_gpu"],
            }
        if not ok:
            trials.append({"trial": t_idx, "status": "failed",
                           "params": dict(params), "per_split": per_split,
                           "error": error})
            continue
        mses = [per_split[s]["mse"] for s in SEARCH_SPLITS]
        select_score = float(np.mean(mses))
        trials.append({"trial": t_idx, "status": "complete",
                       "params": dict(params), "per_split": per_split,
                       "select_score_dev_selection_mean": select_score,
                       "feasible_selection": bool(
                           per_split["selection"]["feasible_explicit"])})
    return {"branch": branch, "arm": arm, "trials": trials,
            "eval_policy": {"step": origin_step,
                            "max_eval_origins": max_eval_origins,
                            "max_train_origins": max_train_origins,
                            "seed": seed}}


def pick_winner(trials: list) -> tuple:
    completed = [t for t in trials if t.get("status") == "complete"]
    if not completed:
        return None, "no-event-feasible"
    feasible = [t for t in completed if t.get("feasible_selection")]
    pool = feasible if feasible else completed
    winner = min(pool, key=lambda t: t["select_score_dev_selection_mean"])
    return winner, ("event-feasible" if feasible else "no-event-feasible")


def run_gpu_hpo(ace=("store/ch-v1/ace.parquet",),
                ch_path="store/ch-v1/ch-hourly.parquet",
                raw_pattern="/home/t-lab01/.local/state/sundb/stage/ace_solar/year=*/part.parquet",
                outdir=DEFAULT_OUTDIR, n_trials=N_TRIALS,
                origin_step=ORIGIN_STEP, max_eval_origins=MAX_EVAL_ORIGINS,
                max_train_origins=MAX_TRAIN_ORIGINS, seed=SEED) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    started_iso = pd.Timestamp.now(tz="UTC").isoformat()
    t0 = time.perf_counter()
    candidates = candidate_grid(n_trials)
    assert 16 <= len(candidates) <= 24, "protocol bounds 16-24 trials/branch"

    # Two concurrent branch searches on both GPUs (awaited, no orphans).
    ctx = multiprocessing.get_context("spawn")
    branch_results: dict = {}
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=2, mp_context=ctx) as pool:
        futs = {pool.submit(run_branch_worker, branch, str(outdir),
                            list(ace), ch_path, raw_pattern, seed,
                            n_trials, origin_step, max_eval_origins,
                            max_train_origins): branch for branch in BRANCHES}
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()  # raises visibly on worker crash
            branch_results[res["branch"]] = res

    summary: dict = {
        "protocol": {
            "splits": SPLIT_CFG,
            "search_splits": list(SEARCH_SPLITS),
            "select_basis": ("mean MSE over dev-A/B/C + selection only; "
                             "holdout is evaluation, never tie-break; "
                             "official truth never read or used"),
            "sampling_policy": {"step": origin_step,
                                "max_eval_origins": max_eval_origins,
                                "max_train_origins": max_train_origins,
                                "horizons": ("1..72 pooled, identical candidate "
                                            "params across rolling splits")},
            "event_definition": ("600km/s, quiet gap<=6h, boundary-censored, "
                                "one-to-one peak matching<=12h"),
            "event_gate": {"unique_observed_peak_timestamps>=10": True,
                           "precision>=0.60": True, "recall>=0.60": True,
                           "height_mae_kms<=60": True},
            "threads_per_fit": N_THREADS,
            "concurrency": ("2 branch searches concurrently, at most 2 "
                            "concurrent GPU fits via acquire_gpu_lease"),
            "device_policy": ("actual cuda:N via acquire_gpu_lease; never "
                              "device=auto; fail visibly, never CPU fallback"),
            "versions": runtime_versions(),
            "source_hashes": source_hashes(),
        },
        "branches": {},
    }
    for branch in BRANCHES:
        res = branch_results[branch]
        trials = res["trials"]
        completed = [t for t in trials if t.get("status") == "complete"]
        winner, feasibility = pick_winner(trials)
        summary["branches"][branch] = {
            "arm": BRANCH_ARM[branch],
            "n_trials_requested": len(candidates),
            "n_trials_complete": len(completed),
            "trials": trials,
            "winner": winner,
            "feasibility_label": feasibility,
            "success_claimed": False,
            "selection_note": ("event-feasible candidates take precedence; "
                               "else best-MSE without success claim; "
                               "holdout/official never used for selection"),
        }

    # Freeze winners BEFORE any ablation/combined fit (no holdout/official).
    frozen = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "role": "frozen GPU HPO choice; dev/selection only, pre-replay",
        "sampling_policy": {"step": origin_step,
                            "max_eval_origins": max_eval_origins,
                            "max_train_origins": max_train_origins,
                            "seed": seed},
        "branches": {
            b: {"arm": summary["branches"][b]["arm"],
                "winner_trial": (summary["branches"][b]["winner"] or {}).get("trial"),
                "params": (summary["branches"][b]["winner"] or {}).get("params"),
                "select_score_dev_selection_mean": (
                    summary["branches"][b]["winner"] or {}).get(
                    "select_score_dev_selection_mean"),
                "feasibility_label": summary["branches"][b]["feasibility_label"],
                "n_trials_complete": summary["branches"][b]["n_trials_complete"]}
            for b in BRANCHES},
        "source_hashes": source_hashes(),
        "versions": runtime_versions(),
        "selection_use": "dev/selection only; holdout never tie-break; official never used",
    }
    (outdir / "frozen-choice.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True, default=str) + "\n")
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")

    # Matched-control ablations at winning params (dev/selection only).
    frames = load_inputs(list(ace), ch_path, raw_pattern)
    speed, observed = frames["speed"], frames["observed"]
    identities = frames["identities"]
    for branch in BRANCHES:
        info = summary["branches"].get(branch, {})
        winner = (info or {}).get("winner")
        if not winner:
            continue
        controls = (RECURRENCE_CONTROLS if branch == "recurrence"
                    else PHYSICAL_CONTROLS)
        wparams = winner["params"]
        abl: dict = {}
        for arm in controls:
            per_split: dict = {}
            for split in SEARCH_SPLITS:
                cfg = SPLIT_CFG[split]
                cutoff = utc(cfg["cutoff"])
                ev_idx, sampling = eval_origins(
                    frames["base"].index, split, step=origin_step,
                    max_origins=max_eval_origins, seed=seed)
                tstats = train_pool_stats(frames["base"].index, speed, cutoff,
                                          max_train_origins)
                preds, fore, fit_s, inf_s, gpu_info, ram_b, ram_a = fit_predict_gpu(
                    branch, arm, wparams, frames, speed, cutoff, ev_idx,
                    seed=seed, max_train_origins=max_train_origins)
                scored = score_split(preds, speed, observed, ev_idx)
                # Reuse trial writer with control-specific filename.
                safe_arm = arm.replace("+", "plus").replace("-", "_")
                name = f"{branch}_control_{safe_arm}_{split}"
                ppath = outdir / f"{name}.parquet"
                preds.to_parquet(ppath, index=False)
                (outdir / f"{name}.metrics.json").write_text(json.dumps(
                    {"branch": branch, "arm": arm, "split": split,
                     "matched_params": True, **scored,
                     "feasible_explicit": bool(is_event_feasible(scored))},
                    indent=2, sort_keys=True, default=str) + "\n")
                sampling_rec = dict(sampling)
                sampling_rec["max_eval_origins"] = max_eval_origins
                write_manifest(
                    outdir / f"{name}.manifest.json", mode=name,
                    train_cutoff=cutoff, params=wparams,
                    tree_counts={"n_estimators": wparams["n_estimators"]},
                    columns=list(fore.columns),
                    input_identities=identities, seed=seed,
                    inference_seconds_per_origin=inf_s,
                    sampling=sampling_rec,
                    extra={"branch": branch, "arm": arm, "control": True,
                           "matched_params_of_winner_trial": winner["trial"],
                           "split": split, "mse": scored["mse"],
                           "events_passed": scored["events_passed"],
                           "feasible_explicit": bool(is_event_feasible(scored)),
                           "train_cap": tstats.get("cap"),
                           "train_requested": tstats.get("requested"),
                           "train_retained": tstats.get("retained"),
                           "lease_gpu": gpu_info["lease_gpu"],
                           "device_requested": gpu_info["device"],
                           "ram_before": ram_b, "ram_after": ram_a,
                           "source_hashes": source_hashes(),
                           "threads_per_fit": N_THREADS,
                           "role": "matched ablation, dev/selection only"},
                    caller_file=str(Path(__file__).resolve()),
                    actual_device=fore.actual_device,
                    prediction_path=ppath)
                per_split[split] = {"mse": scored["mse"],
                                    "events_passed": scored["events_passed"],
                                    "feasible_explicit": bool(
                                        is_event_feasible(scored))}
            abl[arm] = {"matched_params": True, "per_split": per_split}
        summary["branches"][branch]["ablations_matched_control"] = abl
        (outdir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")

    # ONE combined model: predeclared dev/selection choice BEFORE the fits.
    rec_w = (summary["branches"]["recurrence"].get("winner") or {})
    phy_w = (summary["branches"]["physical"].get("winner") or {})
    if rec_w and phy_w:
        pick = ("recurrence"
                if rec_w["select_score_dev_selection_mean"]
                <= phy_w["select_score_dev_selection_mean"] else "physical")
        comb_params = dict((rec_w if pick == "recurrence" else phy_w)["params"])
        basis = (f"dev/selection-chosen: {pick} winner trial "
                 f"{(rec_w if pick == 'recurrence' else phy_w)['trial']} "
                 f"(select_score={min(rec_w['select_score_dev_selection_mean'], phy_w['select_score_dev_selection_mean']):.2f}); "
                 f"no holdout/official input; single bounded config, no retune")
    elif rec_w:
        comb_params, basis, pick = dict(rec_w["params"]), \
            "dev/selection-chosen: recurrence winner", "recurrence"
    elif phy_w:
        comb_params, basis, pick = dict(phy_w["params"]), \
            "dev/selection-chosen: physical winner", "physical"
    else:
        comb_params, basis, pick = None, "no winners available", None
    combined_choice = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "role": ("predeclared combined choice; dev/selection only, "
                 "pre-replay"),
        "picked_branch": pick,
        "params": comb_params,
        "param_basis": basis,
        "recurrence_select_score": rec_w.get("select_score_dev_selection_mean"),
        "physical_select_score": phy_w.get("select_score_dev_selection_mean"),
        "features": ("numerical base ACE+CH + physics + quality + horizon "
                     "target-relative recurrence, no imageNN"),
        "source_hashes": source_hashes(),
        "selection_use": ("dev/selection only; no holdout/official input; "
                          "no retune after official"),
    }
    (outdir / "combined-choice.json").write_text(
        json.dumps(combined_choice, indent=2, sort_keys=True, default=str) + "\n")

    if comb_params is not None:
        per_split: dict = {}
        for split in SEARCH_SPLITS:
            cfg = SPLIT_CFG[split]
            cutoff = utc(cfg["cutoff"])
            ev_idx, sampling = eval_origins(
                frames["base"].index, split, step=origin_step,
                max_origins=max_eval_origins, seed=seed)
            tstats = train_pool_stats(frames["base"].index, speed, cutoff,
                                      max_train_origins)
            preds, fore, fit_s, inf_s, gpu_info, ram_b, ram_a = fit_predict_gpu(
                "combined", PHYSICAL_ARM, comb_params, frames, speed,
                cutoff, ev_idx, seed=seed,
                max_train_origins=max_train_origins)
            scored = score_split(preds, speed, observed, ev_idx)
            name = f"combined_trial00_{split}"
            ppath = outdir / f"{name}.parquet"
            preds.to_parquet(ppath, index=False)
            (outdir / f"{name}.metrics.json").write_text(json.dumps(
                {"branch": "combined", "split": split,
                 "role": "dev/selection", **scored,
                 "feasible_explicit": bool(is_event_feasible(scored))},
                indent=2, sort_keys=True, default=str) + "\n")
            sampling_rec = dict(sampling)
            sampling_rec["max_eval_origins"] = max_eval_origins
            write_manifest(
                outdir / f"{name}.manifest.json", mode=name,
                train_cutoff=cutoff, params=comb_params,
                tree_counts={"n_estimators": comb_params["n_estimators"]},
                columns=list(fore.columns),
                input_identities=identities, seed=seed,
                inference_seconds_per_origin=inf_s,
                sampling=sampling_rec,
                extra={"branch": "combined",
                       "features": ("base ACE+CH + physics + quality + "
                                    "horizon trec"),
                       "param_basis": basis,
                       "split": split, "mse": scored["mse"],
                       "events_passed": scored["events_passed"],
                       "feasible_explicit": bool(is_event_feasible(scored)),
                       "train_cap": tstats.get("cap"),
                       "train_requested": tstats.get("requested"),
                       "train_retained": tstats.get("retained"),
                       "lease_gpu": gpu_info["lease_gpu"],
                       "device_requested": gpu_info["device"],
                       "ram_before": ram_b, "ram_after": ram_a,
                       "source_hashes": source_hashes(),
                       "threads_per_fit": N_THREADS,
                       "no_retune_after_official": True},
                caller_file=str(Path(__file__).resolve()),
                actual_device=fore.actual_device,
                prediction_path=ppath)
            per_split[split] = {
                "mse": scored["mse"],
                "mse_observed": scored["mse_observed"],
                "events_passed": scored["events_passed"],
                "feasible_explicit": bool(is_event_feasible(scored))}
        summary["combined"] = {"status": "complete",
                               "params": comb_params,
                               "param_basis": basis,
                               "picked_branch": pick,
                               "per_split": per_split,
                               "feasibility_label": (
                                   "event-feasible"
                                   if per_split.get("selection", {}).get(
                                       "feasible_explicit") else
                                   "no-event-feasible"),
                               "success_claimed": False}
    else:
        summary["combined"] = {"status": "skipped",
                               "reason": "no branch winners available"}

    done = all((summary["branches"].get(b, {}).get("n_trials_complete", 0) >= 16)
               for b in BRANCHES)
    status = {"status": "complete" if done else "incomplete",
              "started_utc": started_iso,
              "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
              "elapsed_seconds": time.perf_counter() - t0,
              "n_trials_complete": {b: summary["branches"].get(b, {}).get(
                  "n_trials_complete") for b in BRANCHES}}
    summary["status"] = status["status"]
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    (outdir / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True, default=str) + "\n")
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ace", nargs="+", default=["store/ch-v1/ace.parquet"])
    p.add_argument("--ch-hourly", default="store/ch-v1/ch-hourly.parquet")
    p.add_argument("--raw-pattern",
                   default="/home/t-lab01/.local/state/sundb/stage/ace_solar/year=*/part.parquet")
    p.add_argument("--output-dir", default=DEFAULT_OUTDIR)
    p.add_argument("--n-trials", type=int, default=N_TRIALS)
    p.add_argument("--origin-step", default=ORIGIN_STEP)
    p.add_argument("--max-eval-origins", type=int, default=MAX_EVAL_ORIGINS)
    p.add_argument("--max-train-origins", type=int, default=MAX_TRAIN_ORIGINS)
    p.add_argument("--seed", type=int, default=SEED)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    summary = run_gpu_hpo(
        ace=tuple(args.ace), ch_path=args.ch_hourly,
        raw_pattern=args.raw_pattern, outdir=args.output_dir,
        n_trials=args.n_trials, origin_step=args.origin_step,
        max_eval_origins=args.max_eval_origins,
        max_train_origins=args.max_train_origins, seed=args.seed)
    print(json.dumps(
        {"status": summary.get("status"),
         "branches": {b: {"complete": summary["branches"][b].get(
             "n_trials_complete"),
             "feasibility": summary["branches"][b].get("feasibility_label")}
             for b in BRANCHES},
         "combined": summary.get("combined", {}).get("status")},
        indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
