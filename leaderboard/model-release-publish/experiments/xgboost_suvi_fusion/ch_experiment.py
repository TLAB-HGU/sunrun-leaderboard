"""24-hour CH ablation: cache, GPU HPO, frozen selection, holdout and official gates."""
from __future__ import annotations

import argparse
import base64
import fcntl
import importlib.metadata
import json
import multiprocessing
import os
import signal
import struct
import subprocess
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.xgboost_suvi_fusion"

from .ch_hpo import (
    ARMS,
    SCREEN_END,
    SCREEN_HORIZONS,
    SCREEN_START,
    SELECT_END,
    SELECT_START,
    TRAIN_END,
    _predict_task,
    resources_available,
    run_screen,
    shortlist,
)
from .coronal_holes import (
    VERSION,
    build_ch_hourly_cache,
    build_ch_origin_features,
    candidate_mask,
)
from .events import event_metrics
from .features import read_suvi_frame
from .modeling import prediction_metrics, split_origins
from .run import file_identity, load_ace, read_table, save_json

REVISION = "bcf5d5417d99eaa331ddca40581d48953e09a58e"


def write_atomic_parquet(frame, path):
    path = Path(path)
    temp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temp)
    temp.replace(path)


def _png(rgb):
    def chunk(kind, data):
        return struct.pack(">I", len(data))+kind+data+struct.pack(">I", zlib.crc32(kind+data) & 0xffffffff)
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00"+row.tobytes() for row in rgb.astype(np.uint8))
    payload = b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    return payload+chunk(b"IDAT", zlib.compress(raw))+chunk(b"IEND", b"")


def overlays(metadata, output):
    rows = metadata.copy()
    slots = pd.to_datetime(rows.slot, utc=True)
    rows = rows[(slots <= pd.Timestamp(TRAIN_END)) & rows.status.eq("ok") & rows.nc_path.notna()
                & rows.rsun.gt(0)]
    rows = rows[rows.nc_path.map(lambda p: Path(p).is_file())]
    rows = rows.sort_values("slot")
    if rows.empty:
        raise ValueError("no training images for mask quality check")
    output.mkdir(parents=True, exist_ok=True)
    contact = []
    for n, position in enumerate(np.linspace(0, len(rows)-1, min(24, len(rows)), dtype=int)):
        row = rows.iloc[position]
        rad, dqf, header = read_suvi_frame(row.nc_path)
        log = np.log1p(np.maximum(rad, 0))
        finite = log[np.isfinite(log)]
        lo, hi = np.quantile(finite, [.01, .99])
        image = np.nan_to_num(np.clip((log-lo)/max(hi-lo, 1e-9), 0, 1))*255
        gray = np.repeat(image[..., None], 3, axis=2)
        panels = []
        for i, threshold in enumerate((.35, .45, .55)):
            mask, *_ = candidate_mask(rad, dqf, header, threshold)
            rgb = gray.copy()
            rgb[mask] = .45*rgb[mask]+.55*np.array([255, 30, 30])
            encoded = base64.b64encode(_png(rgb[::4, ::4])).decode()
            if threshold == .45:
                contact.append(rgb[::10, ::10].astype(np.uint8))
            panels.append(f'<text x="{i*340+10}" y="40">threshold {threshold}</text>'
                          f'<image x="{i*340}" y="50" width="320" height="320" href="data:image/png;base64,{encoded}"/>')
        svg = '<svg xmlns="http://www.w3.org/2000/svg" width="1020" height="380">'
        svg += '<rect width="100%" height="100%" fill="white"/>'
        svg += f'<text x="10" y="20">{row.slot} — CH candidates (red)</text>'+''.join(panels)+'</svg>'
        (output/f"mask-{n:02d}.svg").write_text(svg)
    tile_h, tile_w = contact[0].shape[:2]
    sheet = np.full((tile_h*((len(contact)+3)//4), tile_w*4, 3), 255, dtype=np.uint8)
    for i, tile in enumerate(contact):
        sheet[(i//4)*tile_h:(i//4+1)*tile_h, (i%4)*tile_w:(i%4+1)*tile_w] = tile
    (output/"mask-contact-sheet.png").write_bytes(_png(sheet))


def extract_features(args):
    from threadpoolctl import threadpool_limits

    output = Path(args.output)
    metadata = read_table(args.suvi_meta)
    hourly = build_ch_hourly_cache(metadata, args.images_root, output/"ch-hourly.parquet", args.image_workers)
    ace = pd.read_parquet(output/"ace-features.parquet")
    with threadpool_limits(limits=1):
        ch = build_ch_origin_features(hourly, ace.index, TRAIN_END)
    write_atomic_parquet(pd.concat([ace, ch], axis=1), output/"all-features.parquet")
    save_json(output/"ch-feature-manifest.json", {"version": VERSION, "rows": len(ch), "features": len(ch.columns),
              "fully_missing_image_rows": int(hourly.isna().all(axis=1).sum()),
              "hourly_identity": file_identity(output/"ch-hourly.parquet"),
              "source": json.loads((output/"ch-hourly.parquet.json").read_text()),
              "origin_identity": file_identity(output/"all-features.parquet")})
    overlays(metadata, output/"masks")
    save_json(output/"feature-ready.json", {"completed_utc": pd.Timestamp.now(tz="UTC").isoformat()})


def prediction_task(candidate, args, stage, deadline, output_path):
    output = Path(args.output)
    times = {
        "selection": (TRAIN_END, SELECT_START, SELECT_END),
        "holdout": (SELECT_END, "2026-03-01T00:00:00Z", "2026-05-31T23:00:00Z"),
        "official": ("2026-05-31T23:00:00Z", "2026-06-01T00:00:00Z", "2026-09-30T23:00:00Z"),
    }
    cutoff, start, end = times[stage]
    task = {k: candidate[k] for k in ("params", "peak_weight", "rise_weight", "threshold", "arm")}
    task.update({"features": str(output/("ace-features.parquet" if candidate["arm"] == "ace_only" else "all-features.parquet")),
                 "ace": str(output/"ace.parquet"), "cutoff": cutoff, "start": start, "end": end,
                 "horizons": tuple(range(1, 73)), "deadline": deadline,
                 "prediction_output": str(output_path)})
    if stage == "selection":
        task.update(early_start=SCREEN_START, early_end=SCREEN_END)
    else:
        task["rounds"] = candidate["rounds"]
    if stage == "official":
        folds = read_table(args.folds)
        task["official_origins"] = pd.to_datetime(folds.origin_last_input_utc, utc=True).astype(str).tolist()
        last_target = pd.to_datetime(folds.origin_last_input_utc, utc=True).max()+pd.Timedelta(hours=72)
        task["end"] = last_target.isoformat()
    return task


def run_tasks(tasks, devices, deadline):
    result = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=context) as executor:
        for start in range(0, len(tasks), len(devices)):
            if time.time() >= deadline:
                break
            while not resources_available(devices):
                if time.time() >= deadline:
                    return result
                time.sleep(10)
            batch = tasks[start:start+len(devices)]
            futures = []
            for task, device in zip(batch, devices):
                cached = Path(task["prediction_output"]).with_suffix(".metrics.json")
                if cached.exists() and Path(task["prediction_output"]).exists():
                    futures.append((task, json.loads(cached.read_text()), None))
                else:
                    futures.append((task, None, executor.submit(_predict_task, {**task, "device": device})))
            for task, cached, future in futures:
                metrics = cached if cached is not None else future.result()
                row = {"arm": task["arm"], "prediction_output": task["prediction_output"], **metrics}
                result.append(row)
                save_json(Path(task["prediction_output"]).with_suffix(".metrics.json"), row)
                print(f"{task['arm']} full72: MSE={metrics['metrics']['mse']:.2f} events={metrics['events']['passed']}", flush=True)
    return result


def freeze_candidates(rows, candidates):
    frozen = {}
    for arm in ARMS:
        available = [(r, c) for r, c in zip(rows, candidates) if c["arm"] == arm]
        if not available:
            continue
        feasible = [(r, c) for r, c in available if r["events"]["passed"]]
        row, candidate = min(feasible or available, key=lambda pair: (pair[0]["metrics"]["mse"], pair[1]["number"]))
        frozen[arm] = {**candidate, **row}
    eligible = [c for a, c in frozen.items() if a != "ace_only" and c["events"]["passed"]]
    selected = min(eligible, key=lambda c: (c["metrics"]["mse"], c["arm"]))["arm"] if eligible else None
    return frozen, selected


def final_gate(candidate, reference_mse):
    return bool(candidate["official"]["mse"] <= .5*reference_mse and candidate["events"]["passed"])


def warm_ace_selection(args):
    """Independent ACE validation while the main coordinator extracts CH."""
    output = Path(args.output)
    lock = (output/"ace-selection.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads((output/"run-state.json").read_text())
    progress = json.loads((output/"hpo/screen-progress.json").read_text())
    if progress["counts"]["ace_only"] < args.max_trials:
        raise ValueError("ACE screening must finish before preselection")
    candidates = shortlist(progress["all_candidates"]["ace_only"], progress["reference"])
    deadline = min(state["deadline_epoch"], state["started_epoch"]+state["budget_hours"]*3600*(20/24))
    tasks = [prediction_task(c, args, "selection", deadline,
                            output/f"selection-{c['arm']}-{c['number']}.parquet") for c in candidates]
    rows = run_tasks(tasks, tuple(args.devices.split(",")), deadline)
    save_json(output/"ace-preselection.json", {"candidates": candidates, "results": rows})


def refresh_events(args):
    """Re-score saved forecasts after the user's causal-fill policy override."""
    output = Path(args.output)
    ace = pd.read_parquet(output/"ace.parquet").set_index("timestamp_utc")
    for path in sorted(output.glob("selection-*.metrics.json")):
        row = json.loads(path.read_text())
        pred = pd.read_parquet(row["prediction_output"])
        row["events"] = event_metrics(pred, ace.filled_speed_kms, ace.was_missing.eq(0))
        save_json(path, row)
        print(path.name, row["events"]["passed"], row["events"]["precision"], row["events"]["recall"], flush=True)
    summary = output/"ace-preselection.json"
    if summary.exists():
        saved = json.loads(summary.read_text())
        saved["results"] = [json.loads(Path(r["prediction_output"]).with_suffix(".metrics.json").read_text())
                            for r in saved["results"]]
        save_json(summary, saved)
    save_json(output/"event-policy.json", {"missing_policy": "forward_fill", "reason": "user: 이전값으로 채워서 진행",
              "imputation": "past observed value only; filled slots disclosed in event reports",
              "source": file_identity(Path(__file__).with_name("events.py"))})


def preview_features(args):
    """Publish the completed pre-December prefix for HPO while later images load.

    A create-only hard link lets the full extractor atomically replace the final
    table later without any race or overwriting a completed final table. HPO
    physically slices to November, so its training/evaluation values are equal.
    """
    from threadpoolctl import threadpool_limits
    output = Path(args.output)
    lock = (output/"preview.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    partial = output/"ch-hourly.parquet.partial"
    while not (output/"all-features.parquet").exists():
        state = json.loads((output/"run-state.json").read_text())
        if state["status"] == "failed" or time.time() >= state["deadline_epoch"]:
            return
        if partial.exists():
            index = pd.read_parquet(partial, columns=[]).index
            if len(index) and index.max() >= pd.Timestamp(SCREEN_END):
                hourly = pd.read_parquet(partial).loc[:SCREEN_END]
                ace = pd.read_parquet(output/"ace-features.parquet").loc[:SCREEN_END]
                with threadpool_limits(limits=1):
                    ch = build_ch_origin_features(hourly, ace.index, TRAIN_END)
                preview = output/"screen-features.parquet"
                write_atomic_parquet(pd.concat([ace, ch], axis=1), preview)
                try:
                    os.link(preview, output/"all-features.parquet")
                except FileExistsError:
                    pass  # final extractor already published the complete table
                save_json(output/"preview-ready.json", {"last_feature_utc": str(ace.index.max()),
                          "rows": len(ace), "identity": file_identity(preview)})
                return
        time.sleep(10)


def official_score(path, ace, folds):
    from scorer.scoring import build_truth, score, validate

    prediction = read_table(path)
    prediction.origin_last_input_utc = pd.to_datetime(prediction.origin_last_input_utc, utc=True)
    folds = folds.copy()
    folds.origin_last_input_utc = pd.to_datetime(folds.origin_last_input_utc, utc=True)
    if "regime" not in folds:
        folds["regime"] = "unclassified"
    errors = validate(prediction, folds)
    if errors:
        raise ValueError("; ".join(errors))
    return score(prediction, build_truth(ace, folds), folds)


def curve_artifact(path, truth, output):
    p = pd.read_parquet(path)
    p.origin_last_input_utc = pd.to_datetime(p.origin_last_input_utc, utc=True)
    origins = pd.DatetimeIndex(p.origin_last_input_utc.unique()).sort_values()
    daily = origins[origins.hour == 0]
    if not len(daily):
        daily = origins
    ranking = []
    for o in daily:
        y = truth.reindex(pd.date_range(o+pd.Timedelta(hours=1), periods=72, freq="h")).to_numpy()
        ranking.append((float(np.max(y)), o))
    panels = []
    for i, (_, o) in enumerate(sorted(ranking, reverse=True)[:8]):
        pred = p[p.origin_last_input_utc.eq(o)].sort_values("horizon_hours").pred_kms.to_numpy()
        y = truth.reindex(pd.date_range(o+pd.Timedelta(hours=1), periods=72, freq="h")).to_numpy()
        x0, y0 = 35+(i % 2)*480, 40+(i//2)*200
        def points(values, x0=x0, y0=y0):
            return ' '.join(f'{x0+j*6:.1f},{y0+150-(v-250)/650*140:.1f}' for j, v in enumerate(values))
        panels.append(f'<text x="{x0}" y="{y0}">{o.isoformat()}</text>'
                      f'<polyline points="{points(y)}" fill="none" stroke="black"/>'
                      f'<polyline points="{points(pred)}" fill="none" stroke="red"/>'
                      f'<text x="{x0}" y="{y0+175}">h1–72; 250–900 km/s; causal-filled truth black, predicted red</text>')
    output.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="850">'
                      '<rect width="100%" height="100%" fill="white"/>'+''.join(panels)+'</svg>')


def execute(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/"run.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = output/"run-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("status") in ("complete", "holdout_gate_failed", "no_event_feasible_candidate", "screen_only_complete"):
        print(f"Experiment already terminal: {state['status']}", flush=True)
        return
    if not state:
        state = {"started_epoch": time.time(), "budget_hours": args.budget_hours,
                 "deadline_epoch": time.time()+args.budget_hours*3600, "revision": REVISION,
                 "version": VERSION, "status": "preparing", "pid": os.getpid()}
    deadline = state["deadline_epoch"]
    # A resume never resets the clock or grants additional compute time.
    if time.time() >= deadline:
        state.update(status="budget_exhausted", pid=os.getpid())
        save_json(state_path, state)
        return
    devices = tuple(args.devices.split(","))
    if len(set(devices)) != len(devices):
        raise ValueError("GPU devices must be distinct")
    state["pid"] = os.getpid()
    state["attempt"] = state.get("attempt", 0)+1
    if "error" in state:
        state["previous_error"] = state.pop("error")
    save_json(state_path, state)
    save_json(output/f"runtime-attempt-{state['attempt']}.json", {
        "python": sys.version, "libraries": {name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scipy", "xgboost", "optuna", "pyarrow")},
        "source": {p.name: file_identity(p) for p in Path(__file__).parent.glob("*.py")},
        "device_info": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total",
                       "--format=csv"], text=True, timeout=10) if "cuda" in args.devices else None})
    identity = {"ace": file_identity(args.ace), "metadata": file_identity(args.suvi_meta),
                "existing_features": file_identity(args.ace_origin_features),
                "folds": file_identity(args.folds), "leader": file_identity(args.leader_predictions),
                "revision": REVISION, "version": VERSION, "devices": devices,
                "max_trials": args.max_trials, "study_prefix": args.study_prefix}
    if args.feature_set != "all":
        identity["feature_set"] = args.feature_set
    identity_path = output/"input-identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != json.loads(json.dumps(identity)):
        raise ValueError("resume input identity changed")
    save_json(identity_path, identity)
    ace = load_ace([args.ace])
    if not (output/"leader-reference.json").exists():
        from scorer.scoring import build_truth
        folds = read_table(args.folds)
        folds.origin_last_input_utc = pd.to_datetime(folds.origin_last_input_utc, utc=True)
        official_truth_path = Path(args.folds).with_name("truth.parquet")
        if not official_truth_path.exists():
            raise ValueError("pinned official truth.parquet is required next to folds")
        expected = read_table(official_truth_path)
        expected.origin_last_input_utc = pd.to_datetime(expected.origin_last_input_utc, utc=True)
        actual = build_truth(ace, folds)
        keys = ["origin_last_input_utc", "horizon_hours"]
        merged = actual.merge(expected, on=keys, suffixes=("_local", "_official"), validate="one_to_one")
        if len(merged) != len(expected) or not np.array_equal(merged.target_kms_local, merged.target_kms_official):
            raise ValueError("local ACE targets differ from pinned official truth")
        if not np.array_equal(merged.was_missing_local, merged.was_missing_official):
            raise ValueError("local ACE missingness differs from official truth")
        leader_score = official_score(args.leader_predictions, ace, folds)
        save_json(output/"leader-reference.json", {"score": leader_score, "target_mse": .5*leader_score["mse"],
                  "verified_truth_pairs": len(merged), "truth_identity": file_identity(official_truth_path)})
    if not (output/"ace.parquet").exists():
        write_atomic_parquet(ace, output/"ace.parquet")
    if not (output/"ace-features.parquet").exists():
        import pyarrow.parquet as pq
        columns = [c for c in pq.read_schema(args.ace_origin_features).names if c.startswith("ace_")]
        write_atomic_parquet(pd.read_parquet(args.ace_origin_features, columns=columns), output/"ace-features.parquet")
    features_process = None
    preview_process = None
    warm_process = None
    features_log = None
    auxiliary_logs = []
    def launch_phase(phase, logfile):
        log = (output/logfile).open("a")
        auxiliary_logs.append(log)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--phase", phase]
        return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        if not (output/"feature-ready.json").exists():
            features_log = (output/"features.log").open("a")
            command = [sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:]]
            command += ["--phase", "features"]
            features_process = subprocess.Popen(command, stdout=features_log, stderr=subprocess.STDOUT, start_new_session=True)
            preview_process = launch_phase("preview-features", "preview.log")
        target = ace.set_index("timestamp_utc").filled_speed_kms
        observed = ace.set_index("timestamp_utc").was_missing.eq(0)
        index = pd.read_parquet(output/"ace-features.parquet", columns=[]).index
        origins = split_origins(index, SCREEN_START, SCREEN_END)
        persistence = pd.DataFrame({"origin_last_input_utc": np.repeat(origins, len(SCREEN_HORIZONS)),
                                    "horizon_hours": np.tile(SCREEN_HORIZONS, len(origins)),
                                    "pred_kms": np.repeat(target.reindex(origins).to_numpy(), len(SCREEN_HORIZONS))})
        reference = prediction_metrics(persistence, target, observed)
        paths = {a: output/("ace-features.parquet" if a == "ace_only" else "all-features.parquet") for a in ARMS}
        state["status"] = "screening"
        save_json(state_path, state)
        screen_path = output/"screen.json"
        if screen_path.exists():
            screen = json.loads(screen_path.read_text())
        else:
            def check_features():
                nonlocal warm_process
                if features_process and features_process.poll() not in (None, 0):
                    raise RuntimeError("CH extraction failed; inspect features.log")
                progress_path = output/"hpo/screen-progress.json"
                if args.feature_set == "all" and warm_process is None and not (output/"ace-preselection.json").exists() and progress_path.exists():
                    progress = json.loads(progress_path.read_text())
                    if progress["counts"]["ace_only"] >= args.max_trials:
                        warm_process = launch_phase("ace-selection", "ace-preselection.log")
            screen = run_screen(paths, output/"ace.parquet", output/"hpo", reference,
                                state["started_epoch"]+args.budget_hours*3600*(16/24), devices,
                                args.max_trials, args.study_prefix, prerequisite_check=check_features,
                                arms=ARMS if args.feature_set == "all" else (args.feature_set,))
            save_json(screen_path, screen)
        if args.feature_set != "all":
            state.update(status="screen_only_complete", eligible_for_submission=False, finished_epoch=time.time())
            save_json(state_path, state)
            save_json(output/"report.json", {"status": state["status"], "screen": screen,
                      "eligible_for_submission": False, "reason": "isolated HPO; full ablation and gates require --feature-set all"})
            return
        if features_process:
            timeout = max(1, deadline-time.time())
            if features_process.wait(timeout=timeout) != 0:
                raise RuntimeError("CH extraction failed; inspect features.log")
        if not (output/"feature-ready.json").exists():
            raise RuntimeError("CH features are not ready")
        if warm_process:
            warm_process.wait(timeout=max(1, deadline-time.time()))
        state["status"] = "selection"
        save_json(state_path, state)
        frozen_path = output/"frozen.json"
        if frozen_path.exists():
            frozen_data = json.loads(frozen_path.read_text())
            frozen, selected = frozen_data["conditions"], frozen_data["selected_arm"]
        else:
            candidates = [c for arm in ARMS for c in shortlist(screen["balanced_candidates"][arm], reference)]
            tasks = [prediction_task(c, args, "selection", deadline,
                                     output/f"selection-{c['arm']}-{c['number']}.parquet") for c in candidates]
            rows = run_tasks(tasks, devices, min(deadline, state["started_epoch"]+args.budget_hours*3600*(20/24)))
            if len(rows) != len(tasks):
                raise TimeoutError("selection incomplete within reserved budget")
            frozen, selected = freeze_candidates(rows, candidates)
            save_json(frozen_path, {"conditions": frozen, "selected_arm": selected,
                      "policy": "event-feasible min-MSE per arm; CH-containing winner selected before holdout",
                      "selected_utc": pd.Timestamp.now(tz="UTC").isoformat()})
        if set(frozen) != set(ARMS):
            raise RuntimeError("not all three ablations have a frozen candidate")
        state["status"] = "holdout"
        save_json(state_path, state)
        holdout_path = output/"holdout.json"
        if holdout_path.exists():
            holdout = json.loads(holdout_path.read_text())
        else:
            tasks = [prediction_task(frozen[a], args, "holdout", deadline, output/f"holdout-{a}.parquet") for a in ARMS]
            rows = run_tasks(tasks, devices, deadline)
            if len(rows) != 3:
                raise TimeoutError("holdout incomplete")
            holdout = {r["arm"]: r for r in rows}
            save_json(holdout_path, holdout)
        for a in ARMS:
            curve_artifact(output/f"holdout-{a}.parquet", target, output/f"holdout-{a}-curves.svg")
        leader_reference = json.loads((output/"leader-reference.json").read_text())
        report = {"selected_arm": selected, "conditions": frozen, "holdout": holdout,
                  "reference": reference, "screen_counts": screen["counts"], "eligible_for_submission": False,
                  "official_evaluated": False, "leader_mse": leader_reference["score"]["mse"],
                  "target_mse": leader_reference["target_mse"], "event_missing_policy": "forward_fill"}
        if selected is None:
            state["status"] = "no_event_feasible_candidate"
        elif not holdout[selected]["events"]["passed"] or holdout[selected]["metrics"]["mse"] >= holdout["ace_only"]["metrics"]["mse"]:
            state["status"] = "holdout_gate_failed"
        else:
            state["status"] = "official"
            save_json(state_path, state)
            tasks = [prediction_task(frozen[a], args, "official", deadline, output/f"official-{a}.parquet") for a in ARMS]
            rows = run_tasks(tasks, devices, deadline)
            if len(rows) != 3:
                raise TimeoutError("official evaluation incomplete")
            folds = read_table(args.folds)
            official = {r["arm"]: {**r, "official": official_score(r["prediction_output"], ace, folds)} for r in rows}
            leader_mse = official_score(args.leader_predictions, ace, folds)["mse"]
            report.update(official=official, official_evaluated=True, leader_mse=leader_mse, target_mse=.5*leader_mse,
                          eligible_for_submission=final_gate(official[selected], leader_mse))
            state["status"] = "complete"
        report["status"] = state["status"]
        save_json(output/"report.json", report)
        lines = ["# Coronal-hole ablation", "", f"Status: {state['status']}; selected: {selected}", "",
                 "| Input | Selection MSE | Holdout MSE | Event precision | Event recall | Peak MAE | Unique events |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for a in ARMS:
            e = holdout[a]["events"]
            lines.append(f"| {a} | {frozen[a]['metrics']['mse']:.2f} | {holdout[a]['metrics']['mse']:.2f} | {e['precision']} | {e['recall']} | {e['height_mae_kms']} | {e['unique_observed_events']} |")
        if "official" in report:
            lines += ["", f"Leader MSE: {report['leader_mse']:.2f}; target: {report['target_mse']:.2f}"]
        coverage = holdout["ace_only"]["events"]
        lines += ["", "Missing truth uses the previous observed value only (no future interpolation).",
                  f"Holdout imputed slots: {coverage['imputed_fraction']:.2%}; distinct observed peak timestamps: {coverage['unique_observed_peak_timestamps']} (gate requires >=10).",
                  f"Official evaluation performed: {report['official_evaluated']}; submission eligible: {report['eligible_for_submission']}.",
                  f"Pinned official leader MSE: {report['leader_mse']:.2f}; 50% target: {report['target_mse']:.2f}. Holdout MSE is from a different period and cannot establish this official target.",
                  "Curve truth lines include the causal-filled slots, not only observed values."]
        (output/"report.md").write_text('\n'.join(lines)+'\n')
        state.update(finished_epoch=time.time(), eligible_for_submission=report["eligible_for_submission"])
        save_json(state_path, state)
    except Exception as exc:
        state.update(status="budget_exhausted" if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)) else "failed",
                     error=f"{type(exc).__name__}: {exc}")
        save_json(state_path, state)
        raise
    finally:
        for process in (features_process, preview_process, warm_process):
            if process and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
        if features_log:
            features_log.close()
        for log in auxiliary_logs:
            log.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("run", "features", "ace-selection", "refresh-events", "preview-features"), default="run")
    p.add_argument("--ace", required=True)
    p.add_argument("--suvi-meta", required=True)
    p.add_argument("--images-root", required=True)
    p.add_argument("--ace-origin-features", required=True)
    p.add_argument("--folds", required=True)
    p.add_argument("--leader-predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--devices", default="cuda:0,cuda:1")
    p.add_argument("--budget-hours", type=float, default=24)
    p.add_argument("--max-trials", type=int, default=24)
    p.add_argument("--study-prefix", default="ch-v1")
    p.add_argument("--feature-set", choices=("all", *ARMS), default="all",
                   help="all: full ablation and gates; a single arm: isolated screening HPO only")
    p.add_argument("--image-workers", type=int, default=12)
    return p


if __name__ == "__main__":
    options = parser().parse_args()
    if options.budget_hours <= 0 or options.max_trials < 1:
        raise ValueError("budget and trial count must be positive")
    if options.phase == "features":
        extract_features(options)
    elif options.phase == "ace-selection":
        warm_ace_selection(options)
    elif options.phase == "refresh-events":
        refresh_events(options)
    elif options.phase == "preview-features":
        preview_features(options)
    else:
        execute(options)
