"""Two-GPU, single-writer Optuna coordinator for independent CH ablations."""
from __future__ import annotations

import gc
import json
import multiprocessing
import os
import pickle
import resource
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import optuna
import pandas as pd

from .coronal_holes import prefix
from .events import error_diagnostics, event_metrics
from .modeling import (
    DirectForecaster,
    ensure_device,
    passes_gate,
    prediction_metrics,
    split_origins,
)
from .run import save_json
from .tune import suggest_parameters

ARMS = ("ace_only", "ch_only", "ace_ch")
SCREEN_HORIZONS = (6, 12, 24, 36, 48, 72)
TRAIN_END = "2025-07-31T23:00:00Z"
SCREEN_START = "2025-08-01T00:00:00Z"
SCREEN_END = "2025-11-30T23:00:00Z"
SELECT_START = "2025-12-01T00:00:00Z"
SELECT_END = "2026-02-28T23:00:00Z"


def select_columns(columns, arm, threshold=.45):
    if arm not in ARMS:
        raise ValueError(f"unknown feature set {arm}")
    selected = [c for c in columns if (arm != "ch_only" and c.startswith("ace_"))
                or (arm != "ace_only" and c.startswith(prefix(threshold)))]
    if not selected or (arm != "ace_only" and not any(c.startswith(prefix(threshold)) for c in selected)):
        raise ValueError(f"missing feature columns for {arm}/{threshold}")
    return selected


def resources_available(devices, min_ram_gib=8):
    available = next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    if available < min_ram_gib*1024**3:
        return False
    if not any(d.startswith("cuda") for d in devices):
        return True
    command = ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"]
    rows = subprocess.check_output(command, text=True, timeout=10).strip().splitlines()
    usage = {int(parts[0]): float(parts[1])/float(parts[2])
             for parts in (row.split(",") for row in rows)}
    return all(usage[int(d.split(":")[1])] < .85 for d in devices if d.startswith("cuda"))


def _predict_task(task):
    """Isolated candidate. No SQLite or sampler access in this process."""
    from threadpoolctl import threadpool_limits

    started = time.monotonic()
    with threadpool_limits(limits=4):
        ensure_device(task["device"])
        schema = pd.read_parquet(task["features"], columns=[]).index
        if schema.empty:
            raise ValueError("empty feature table")
        import pyarrow.parquet as pq
        columns = select_columns(pq.read_schema(task["features"]).names, task["arm"], task["threshold"])
        frame = pd.read_parquet(task["features"], columns=columns)
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
        ace = pd.read_parquet(task["ace"])
        ace.index = pd.DatetimeIndex(pd.to_datetime(ace.timestamp_utc, utc=True))
        target = ace.filled_speed_kms
        observed = ace.was_missing.eq(0)
        # Physically exclude targets beyond this task's allowed period.
        target, observed = target.loc[:task["end"]], observed.loc[:task["end"]]
        inputs = frame.loc[:task["cutoff"]]
        origins = (pd.DatetimeIndex(pd.to_datetime(task["official_origins"], utc=True))
                   if task.get("official_origins") else split_origins(frame.index, task["start"], task["end"]))
        if task.get("daily_only"):
            origins = origins[origins.hour == 0]
        validation = frame.loc[origins]
        early_origins = split_origins(frame.index, task.get("early_start", task["start"]),
                                     task.get("early_end", task["end"]))
        early_validation = frame.loc[early_origins]
        if inputs.empty or validation.empty:
            raise ValueError("empty training/evaluation split")
        model = DirectForecaster(task["params"], device=task["device"], horizons=task["horizons"],
                                 peak_weight=task["peak_weight"], rise_weight=task["rise_weight"],
                                 weight_cap=5, n_jobs=4)
        # Check deadline between horizons. A running tree fit is bounded by the
        # 2000-tree cap and is allowed to finish cleanly rather than lose artifacts.
        rounds = task.get("rounds")
        for h in task["horizons"]:
            if time.time() >= task["deadline"]:
                raise TimeoutError("candidate exceeded phase deadline")
            part = DirectForecaster(task["params"], device=task["device"], horizons=(h,),
                                    peak_weight=task["peak_weight"], rise_weight=task["rise_weight"],
                                    weight_cap=5, n_jobs=4)
            part.fit(inputs, target, task["cutoff"],
                     validation_features=early_validation if rounds is None else None,
                     validation_end=task.get("early_end", task["end"]) if rounds is None else None,
                     rounds_by_horizon={int(k): v for k, v in rounds.items()} if rounds else None)
            model.models.update(part.models)
            model.columns = part.columns
            del part
        prediction = model.predict(validation)
        result = {"metrics": prediction_metrics(prediction, target, observed),
                  "rounds": model.best_rounds(), "feature_count": len(columns),
                  "runtime": {"seconds": time.monotonic()-started, "device": task["device"],
                              "cpu_threads": 4, "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
        if task.get("prediction_output"):
            prediction.to_parquet(task["prediction_output"], index=False)
        if tuple(task["horizons"]) == tuple(range(1, 73)):
            result["events"] = event_metrics(prediction, target, observed,
                                             origins=origins if task.get("official_origins") else None)
            result["diagnostics"] = error_diagnostics(prediction, target, observed)
        del model, frame
        gc.collect()
        return result


def trial_constraints(trial):
    return tuple(trial.user_attrs.get("constraints", (1., 1., 1.)))


def checkpoint(path, studies):
    payload = {arm: {"sampler": study.sampler, "count": len(study.trials)} for arm, study in studies.items()}
    temp = path.with_suffix(".tmp")
    temp.write_bytes(pickle.dumps(payload))
    temp.replace(path)


def create_studies(output_dir, study_prefix, reference, arms=ARMS):
    # SQLite JSON attributes stringify horizon keys; compare canonical JSON
    # rather than reject a valid resume because Python used integer keys.
    reference = json.loads(json.dumps(reference))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output/"samplers.pkl"
    stored = pickle.loads(path.read_bytes()) if path.exists() else {}
    if stored and set(stored) != set(arms):
        raise ValueError("feature-set changed; use a distinct output directory")
    studies = {}
    for arm in arms:
        sampler = stored[arm]["sampler"] if arm in stored else optuna.samplers.TPESampler(
            seed=42, n_startup_trials=10, constraints_func=trial_constraints)
        study = optuna.create_study(storage=f"sqlite:///{output/'studies.sqlite3'}",
                                    study_name=f"{study_prefix}-{arm}", direction="minimize",
                                    sampler=sampler, load_if_exists=True)
        if len(study.trials) != stored.get(arm, {}).get("count", 0):
            raise RuntimeError("sampler/database checkpoint mismatch; do not silently restart this study")
        saved = study.user_attrs.get("reference")
        if saved is not None and saved != reference:
            raise ValueError("study reference changed")
        study.set_user_attr("reference", reference)
        # Interrupted in-flight work has no valid result and must not be counted
        # as complete. RNG is restored at the persisted ask boundary.
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.RUNNING:
                study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        studies[arm] = study
    checkpoint(path, studies)
    return studies


def completed_candidates(studies):
    result = {}
    for arm, study in studies.items():
        rows = []
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            rows.append({"number": trial.number, "arm": arm, **trial.user_attrs["candidate"],
                         **trial.user_attrs["result"]})
        result[arm] = rows
    equal = min(map(len, result.values()))
    return result, {arm: rows[:equal] for arm, rows in result.items()}


def shortlist(candidates, reference, count=3):
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: (c["metrics"]["mse"], c["number"]))
    by_f1 = sorted(candidates, key=lambda c: (-(c["metrics"]["peak_macro_f1"] or 0), c["number"]))
    feasible = [c for c in ordered if passes_gate(c["metrics"], reference)]
    chosen = []
    for row in ordered[:1]+by_f1[:1]+feasible[:1]+ordered:
        if not any(c["number"] == row["number"] for c in chosen):
            chosen.append(row)
        if len(chosen) == count:
            break
    return chosen


def run_screen(feature_paths, ace_path, output_dir, reference, deadline, devices,
               max_trials=24, study_prefix="ch-v1", worker=_predict_task, prerequisite_check=None, arms=ARMS):
    studies = create_studies(output_dir, study_prefix, reference, arms)
    output = Path(output_dir)
    cursor = 0
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=context) as executor:
        while time.time() < deadline:
            if prerequisite_check is not None:
                prerequisite_check()
            available = [arm for arm in arms if Path(feature_paths[arm]).exists()
                         and sum(t.state == optuna.trial.TrialState.COMPLETE for t in studies[arm].trials) < max_trials]
            if not available:
                if all(Path(feature_paths[a]).exists() for a in arms):
                    break
                time.sleep(10)
                continue
            if not resources_available(devices):
                time.sleep(10)
                continue
            batch = []
            for device in devices:
                arm = next((ARMS[(cursor+i) % 3] for i in range(3)
                            if ARMS[(cursor+i) % 3] in available), None)
                if arm is None:
                    break
                cursor = (ARMS.index(arm)+1) % 3
                # Both devices can serve the same arm while other feature sets
                # are still being extracted; ask/tell order remains fixed.
                pending_for_arm = sum(a == arm for a, _, _ in batch)
                completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in studies[arm].trials)
                if completed+pending_for_arm+1 >= max_trials:
                    available.remove(arm)
                study = studies[arm]
                if not study.trials:
                    study.enqueue_trial({"peak_weight": 1., "rise_weight": 1.})
                trial = study.ask()
                params = suggest_parameters(trial)
                params["n_estimators"] = 2000
                peak = trial.suggest_float("peak_weight", 1, 3)
                rise = trial.suggest_float("rise_weight", 1, 3)
                threshold = trial.suggest_categorical("threshold", [.35, .45, .55]) if arm != "ace_only" else .45
                candidate = {"params": params, "peak_weight": peak, "rise_weight": rise, "threshold": threshold}
                trial.set_user_attr("candidate", candidate)
                task = {**candidate, "arm": arm, "device": device, "features": str(feature_paths[arm]),
                        "ace": str(ace_path), "cutoff": TRAIN_END, "start": SCREEN_START, "end": SCREEN_END,
                        "horizons": SCREEN_HORIZONS, "deadline": deadline}
                batch.append((arm, trial, task))
            checkpoint(output/"samplers.pkl", studies)
            save_json(output/"active-batch.json", {"started_epoch": time.time(),
                      "jobs": [{"arm": a, "trial": t.number, "device": task["device"]} for a, t, task in batch],
                      "cpu_load": list(os.getloadavg()),
                      "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                               "--format=csv,noheader,nounits"], text=True, timeout=10) if any(d.startswith("cuda") for d in devices) else None})
            futures = [executor.submit(worker, task) for _, _, task in batch]
            for (arm, trial, task), future in zip(batch, futures):
                try:
                    result = future.result()
                    metrics = result["metrics"]
                    trial.set_user_attr("result", result)
                    trial.set_user_attr("constraints", [metrics["mse"]-reference["mse"],
                                        reference["peak_macro_f1"]+.02-(metrics["peak_macro_f1"] or 0),
                                        abs(metrics["peak_bias"] if metrics["peak_bias"] is not None else 1e6)-abs(reference["peak_bias"])+10])
                    studies[arm].tell(trial, metrics["mse"])
                    print(f"{arm} trial {trial.number}: MSE={metrics['mse']:.2f} F1={metrics['peak_macro_f1']}", flush=True)
                except Exception as exc:
                    trial.set_user_attr("failure", f"{type(exc).__name__}: {exc}")
                    studies[arm].tell(trial, state=optuna.trial.TrialState.FAIL)
                    print(f"{arm} trial {trial.number} FAILED: {exc}", flush=True)
                    if not isinstance(exc, TimeoutError):
                        checkpoint(output/"samplers.pkl", studies)
                        raise
                checkpoint(output/"samplers.pkl", studies)
            all_rows, balanced = completed_candidates(studies)
            save_json(output/"screen-progress.json", {"all_candidates": all_rows, "balanced_candidates": balanced,
                       "counts": {a: len(r) for a, r in all_rows.items()}, "reference": reference})
    all_rows, balanced = completed_candidates(studies)
    result = {"all_candidates": all_rows, "balanced_candidates": balanced, "reference": reference,
              "counts": {a: len(r) for a, r in all_rows.items()}}
    save_json(output/"screen.json", result)
    return result
