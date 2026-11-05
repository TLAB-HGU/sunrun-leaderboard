"""Validate a completed retraining run, publish its actual source, and submit it.

Uses an isolated copy of the standard submit client so unrelated local display
policy edits never change the live leaderboard. Existing submissions are kept.
"""
from __future__ import annotations

import argparse
import ast
import base64
import fcntl
import json
import shutil
import sys
import time
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLIENT = HERE / "submit-client"


def client():
    CLIENT.mkdir(exist_ok=True)
    for name in ("submit.py", "board.py", "architecture_code.py", "metric_assets.py", "lbstatus.py"):
        shutil.copy2(ROOT / "leaderboard" / name, CLIENT / name)
    shutil.copytree(ROOT / "leaderboard/scorer", CLIENT / "scorer", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    sys.path.insert(0, str(CLIENT))
    import submit
    return submit


def model_reports(out, family):
    if family == "direct":
        return [json.loads(p.read_text()) for p in sorted(out.glob("F[123]_h[0-9][0-9].json"))
                + sorted(out.glob("F[123]_pooled.json"))]
    if family == "special":
        return [json.loads((out / f"{p}.json").read_text()) for p in ("F1", "F2", "F3")]
    if family in ("strategies", "advanced", "foundation", "structured", "direct", "tcn", "special", "tda",
                  "pooling", "special_hsspeak"):
        return [json.loads((out / f"{p}_metadata.json").read_text()) for p in ("F1", "F2", "F3")]
    paths = sorted(out.glob("F[123]_b[012].json"))
    if len(paths) != 9:
        raise ValueError("Expected nine completed period/block reports")
    return [json.loads(p.read_text()) for p in paths]


def validate_run(experiment):
    out = HERE / "runs" / experiment
    launch = json.loads((out / "launch.json").read_text())
    if (out / "train.exit").read_text().strip() != "0":
        raise ValueError("Training did not complete successfully")
    reports = model_reports(out, launch["family"])
    family = launch["family"]
    expected_parts = (216 if experiment == "e2000_direct_s42" else 3) if family == "direct" else (
        9 if family in ("calib11", "top30", "distributions", "calibration") else 3)
    if len(reports) != expected_parts:
        raise ValueError("Missing full-run fit reports")
    parts_per_period = expected_parts // 3
    if any(sum(r.get("period") == p for r in reports) != parts_per_period for p in ("F1", "F2", "F3")):
        raise ValueError("Fit reports do not cover all three periods")
    data_arg = launch["command"][launch["command"].index("--data") + 1]
    if (not launch.get("training_target_ffill") or Path(data_arg).name != "df_ffill_trainy.parquet"
            or launch["input_identity"]["path"] != data_arg
            or launch["input_identity"]["bytes"] != (HERE / "input/df_ffill_trainy.parquet").stat().st_size):
        raise ValueError("Launch did not use the verified training-target-ffill snapshot")
    if "--smoke" in launch["command"] or not reports:
        raise ValueError("Incomplete or smoke training cannot be submitted")
    if any(r.get("smoke", False) or r.get("experiment") != experiment for r in reports):
        raise ValueError("Smoke or mismatched experiment reports cannot be submitted")
    for report in reports:
        identity = report.get("data", report.get("source"))
        reported_path = identity.get("path") if isinstance(identity, dict) else identity
        if reported_path != data_arg:
            raise ValueError("Fit report input differs from the recorded launch input")
        if isinstance(identity, dict):
            for key in ("bytes", "mtime_ns"):
                if key in identity and identity[key] != launch["input_identity"][key]:
                    raise ValueError("Fit report input identity changed after launch")
        if "rounds" in report:
            rounds = report["rounds"]
            expected = None
            if family in ("direct", "special", "calibration"):
                expected = 900
            elif family == "distributions":
                expected = 300 if experiment.endswith("calib8_qxgb") else 450
            elif family == "calib11":
                expected = 900 if experiment.endswith(("calib11_metricgn", "calib11b_metricwt")) else 300
                if report["period"] == "F3":
                    if experiment.endswith("calib11f_serarounds"):
                        expected = 450
                    elif experiment.endswith(("calib11g_serarounds2", "calib11h_noenlil")):
                        expected = 1200
            if rounds < 300 or (expected is not None and rounds != expected):
                raise ValueError("Training rounds do not match the full original recipe")
        for fit in report.get("fits", []):
            if "rounds" in fit and fit["rounds"] < 300:
                raise ValueError("Smoke fit rounds cannot be submitted")
        if family == "tcn" and report.get("epoch_budget") != 100:
            raise ValueError("TCN must use its full epoch budget")
    models = [p for pattern in ("*.ubj", "*.txt", "*.pkl", "*.pt", "*.pth", "*.joblib", "*_context.npz")
              for p in out.rglob(pattern)]
    if not models or any(p.stat().st_size == 0 for p in models):
        raise ValueError("Fitted model artifacts are missing")
    pred = pd.read_parquet(out / "predictions.parquet")
    keys = ["origin_last_input_utc", "horizon_hours"]
    if len(pred) != 6315 * 72 or pred.duplicated(keys).any():
        raise ValueError("Predictions must contain all 6,315 unique hourly origins x72")
    if not np.isfinite(pred.pred_kms).all() or not pred.pred_kms.gt(0).all():
        raise ValueError("Predictions require positive finite values")
    origin = pd.to_datetime(pred[keys[0]], utc=True)
    rows = pd.read_parquet(HERE / "input/df_ffill_trainy.parquet", columns=["set", "period", "origin", "h"],
                           filters=[("set", "==", "query")])
    rows = rows[rows.period.notna()]
    expected = pd.MultiIndex.from_arrays([pd.to_datetime(rows.origin, utc=True), rows.h])
    actual = pd.MultiIndex.from_arrays([origin, pred[keys[1]]])
    if len(expected.difference(actual)) or len(actual.difference(expected)):
        raise ValueError("Predictions do not match the frozen query grid")
    if (out / "merge.json").exists():
        seconds = json.loads((out / "merge.json").read_text())["inference_seconds_per_fold"]
    else:
        counts = {"F1": 2089, "F2": 2113, "F3": 2113}
        seconds = sum(r["inference_seconds_per_fold"] * r.get("query_origins", counts[r["period"]])
                      for r in reports) / 6315
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Missing measured inference time")
    return out, launch, reports, seconds


def source_for(experiment, family):
    out = HERE / "runs" / experiment
    source = (out / "experiment.py").read_text()
    if source.encode() != (out / "runner_at_launch.py").read_bytes():
        raise ValueError("Executable runner changed after its launch snapshot")
    entry = ast.parse(source).body[-1]
    if not isinstance(entry, ast.If) or "__name__" not in ast.unparse(entry.test):
        raise ValueError("Expected a standalone runner main entry")
    return source.encode()


def publish_sources(files):
    from metric_assets import GitHubAPI
    api = GitHubAPI("TLAB-HGU/sunrun-leaderboard")
    blobs = {path: api.request("POST", "git/blobs", {"content": base64.b64encode(body).decode(),
                                                    "encoding": "base64"})["sha"]
             for path, body in files.items()}
    for attempt in range(4):
        head = api.request("GET", "git/ref/heads/main")["object"]["sha"]
        tree = api.request("GET", f"git/commits/{head}")["tree"]["sha"]
        added = api.request("POST", "git/trees", {"base_tree": tree, "tree": [
            {"path": path, "mode": "100644", "type": "blob", "sha": blob} for path, blob in blobs.items()]})
        commit = api.request("POST", "git/commits", {"message": "Publish reproducible fresh-df ffill retraining source",
                                                      "parents": [head], "tree": added["sha"]})
        try:
            api.request("PATCH", "git/refs/heads/main", {"sha": commit["sha"], "force": False})
            return commit["sha"]
        except urllib.error.HTTPError as exc:
            if exc.code not in (409, 422) or attempt == 3:
                raise
            time.sleep(attempt + 1)


def publish(experiment):
    from huggingface_hub import HfApi, hf_hub_download
    out, launch, reports, seconds = validate_run(experiment)
    if (out / "submission.json").exists():
        raise FileExistsError("Submission receipt already exists; verify it instead of resubmitting")
    manifest = json.loads((HERE / "manifest.json").read_text())
    baseline = next(e for e in manifest["experiments"] if e["experiment"] == experiment)
    prep = json.loads((HERE / "input/ffill_trainy_report.json").read_text())
    effective = {"family": launch["family"], "threads": launch["threads"], "gpu": launch.get("gpu"),
                 "period_recipes": [{k: r[k] for k in ("period", "block", "rounds", "params", "parameters", "features",
                                                       "feature_names", "constant_features", "varying_features", "postprocessing",
                                                       "device", "backend_change", "policy", "epoch_budget", "selected_epoch",
                                                       "optimizer", "history_rule", "target_rule", "actual_component_names",
                                                       "component_settings")
                                     if k in r} for r in reports]}
    for recipe, report in zip(effective["period_recipes"], reports):
        state = report.get("frozen_state", {})
        if "arm_order" in state:
            recipe["arm_order"] = state["arm_order"]
            recipe["actual_arm_count"] = len(state["arm_order"])
            recipe["legacy_inventory_count"] = state.get("legacy_inventory_count")
    config = {"data": {"campaign": "rerun_df_ffill", "source": "data-v2/input/df.parquet",
                       "built_utc": prep["source"]["manifest_built_utc"], "source_bytes": prep["source"]["size_bytes"],
                       "source_mtime_ns": prep["source"]["mtime_ns"], "source_rows": prep["rows"],
                       "prefill": prep["policy"], "target": "training y ffill by target_time; official query truth unchanged",
                       "maturity_hours": 144, "periods": {"F1": 2089, "F2": 2113, "F3": 2113},
                       "original_submission_utc": baseline["original_submission_utc"]},
              "hyperparameters": {"original_recipe": baseline["original_config"]["hyperparameters"],
                                  "effective_recipe": effective}}
    submit = client()
    replay = submit.replay_script_path("seongeun", experiment, config)
    architecture = f"research/rerun_df_ffill/{experiment}/experiment.py"
    source = source_for(experiment, launch["family"])
    compile(source, architecture, "exec")
    local_script = CLIENT / replay
    local_script.parent.mkdir(parents=True, exist_ok=True)
    local_script.write_bytes(source)
    local_architecture = HERE / experiment / "experiment.py"
    local_architecture.parent.mkdir(exist_ok=True)
    local_architecture.write_bytes(source)
    source_receipt = out / "source-publication.json"
    if source_receipt.exists():
        revision = json.loads(source_receipt.read_text())["revision"]
    else:
        revision = publish_sources({architecture: source, replay: source,
                                    "research/rerun_df_ffill/prepare.py": (HERE / "prepare.py").read_bytes(),
                                    "research/rerun_df_ffill/fill_train_y.py": (HERE / "fill_train_y.py").read_bytes(),
                                    "research/rerun_df_ffill/strategy.md": (HERE / "strategy.md").read_bytes()})
        source_receipt.write_text(json.dumps({"revision": revision, "architecture": architecture, "replay": replay}, indent=2) + "\n")
    meta = {"team_member": "seongeun", "experiment": experiment,
            "description": f"Fresh raw-only df.parquet + causal feature ffill rerun of {experiment}. "
                           "Input features and TRAINING y causally forward filled; official evaluation truth unchanged. "
                           "Model architecture and seed preserved; current F1/F2/F3 "
                           "144-hour maturity masks. Full training completed; no evaluation-target calibration or selection. "
                           f"Baseline submission: {baseline['original_submission_utc']}. Existing results retained.",
            "no_future_leakage": True, "config": config, "inference_seconds_per_fold": seconds,
            "architecture_url": f"https://github.com/TLAB-HGU/sunrun-leaderboard/blob/{revision}/{architecture}",
            "f1_training_data_end_utc": "2025-12-29T00:00:00Z",
            "f2_training_data_end_utc": "2026-03-29T00:00:00Z",
            "f3_training_data_end_utc": "2026-06-29T00:00:00Z"}
    (out / "meta.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False))
    dry = submit.submit(str(out / "predictions.parquet"), meta, dry_run=True)
    if dry["competition"]["total"]["replacement_fraction"] != 0:
        raise ValueError("Incomplete official forecast grid; persistence replacements forbidden for rerun completion")
    (out / "dry-run.json").write_text(json.dumps(dry, indent=2) + "\n")
    # Refresh the remote policy inside every render attempt, including CAS retries.
    api = HfApi()
    original_render = submit.render
    import board

    live_policy = {}

    def render_with_live_policy(*args, **kwargs):
        nonlocal live_policy
        head = api.repo_info(submit.SPACE, repo_type="space").sha
        raw = Path(hf_hub_download(submit.SPACE, "visibility.json", repo_type="space", revision=head)).read_text()
        live_policy = json.loads(raw)
        (CLIENT / "visibility.json").write_text(raw)
        return original_render(*args, **kwargs)

    board.visibility_policy = lambda: live_policy
    submit.visibility_policy = lambda: live_policy
    submit.render = render_with_live_policy
    existing = next((r for r in submit.all_results(api, submit.DATASET)
                     if r["team_member"] == "seongeun" and r["experiment"] == experiment
                     and r.get("config") == config), None)
    if existing is None:
        result = submit.submit(str(out / "predictions.parquet"), meta)
    else:
        if existing["competition"] != dry["competition"]:
            raise RuntimeError("Existing rerun identity has different predictions/scores")
        result = existing
        mapping = {"models": [{"member": "seongeun", "experiment": experiment,
                                "url": meta["architecture_url"],
                                **{k: v for k, v in existing.items() if k.startswith("config_")}}]}
        # Reuse the already stored result after a page/verification failure;
        # never submit or retrain the identical model a second time.
        result["_leaderboard"] = submit.publish(api, [], submit.SPACE, architecture=mapping, dataset=submit.DATASET)
    stored = next(r for r in submit.all_results(api, submit.DATASET)
                  if r["team_member"] == "seongeun" and r["experiment"] == experiment
                  and r["submitted_utc"] == result["submitted_utc"])
    if stored["competition"] != result["competition"]:
        raise RuntimeError("Remote result does not match the submitted score")
    head = api.repo_info(submit.SPACE, repo_type="space").sha
    mapping = json.loads(Path(hf_hub_download(submit.SPACE, submit.architecture_code.FILE,
                                            repo_type="space", revision=head)).read_text())
    if not any(e["member"] == "seongeun" and e["experiment"] == experiment
               and e["url"] == meta["architecture_url"] for e in mapping["models"]):
        raise RuntimeError("Remote architecture mapping does not contain the new source")
    import lbstatus

    # The just-verified receipt proves this identity is complete; other active
    # runs are left untouched instead of refetching every historical result.
    lbstatus.leaderboard_submissions = lambda entries, dataset: {("seongeun", experiment): result["submitted_utc"]}
    lbstatus.commit(lambda entries: entries, message=f"Complete fresh-df ffill rerun: {experiment}")
    (out / "submission.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"experiment": experiment, "submitted": True, "metrics": {
        k: result["competition"]["total"][k] for k in ("s", "timse", "rmse", "replacement_fraction")}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    args = parser.parse_args()
    with (HERE / ".publish.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        publish(args.experiment)
