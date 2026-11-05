"""Refit frozen numerical CH submission configurations; never read official truth."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from experiments.ch_breakthrough_v2 import common, hpo

ARMS = {"recurrence": "target-recurrence+CH", "physical": "physical_ace_ch_quality",
        "combined": "physical_ace_ch_quality"}


def validate_config(config):
    branch = config["data"]["feature_branch"]
    if branch not in ARMS or config["data"]["feature_arm"] != ARMS[branch]:
        raise ValueError("unsupported feature branch/arm")
    if common.utc(config["data"]["training_cutoff"]) != common.utc("2026-05-31T23:00:00Z"):
        raise ValueError("training cutoff differs from frozen official replay")
    hyper = config["hyperparameters"]
    if hyper["max_train_origins"] != 6000 or hyper["seed"] != 42:
        raise ValueError("frozen training cap/seed must be 6000/42")
    if not hyper["xgboost"]:
        raise ValueError("missing frozen XGBoost parameters")


def select_origins(folds, feature_index, max_folds=None):
    origins = pd.DatetimeIndex(common.utc(folds["origin_last_input_utc"])).unique().sort_values()
    if not len(origins) or origins.hasnans or not origins.equals(origins.floor("h")):
        raise ValueError("fold origins must be nonempty hourly UTC timestamps")
    if max_folds is not None:
        if max_folds < 1:
            raise ValueError("max-folds must be positive")
        origins = origins[:max_folds]
    missing = origins.difference(feature_index)
    if len(missing):
        raise ValueError(f"fold origin missing from features: {missing[0]}")
    return origins


def validate_predictions(preds, origins):
    keys = pd.MultiIndex.from_arrays([common.utc(preds["origin_last_input_utc"]),
                                      preds["horizon_hours"]])
    expected = pd.MultiIndex.from_product([origins, range(1, 73)])
    if not keys.is_unique or len(keys) != len(expected) or len(expected.difference(keys)):
        raise ValueError("predictions must contain each requested origin and all 72 horizons exactly once")
    if not np.isfinite(preds["pred_kms"]).all():
        raise ValueError("nonfinite predictions")


@contextmanager
def leased_device(requested):
    if requested == "cpu":
        yield "cpu"
    elif requested == "auto":
        with common.acquire_gpu_lease() as gpu:
            device = f"cuda:{gpu}"
            common.ensure_device(device)
            yield device
    else:
        # Same lock files as common.acquire_gpu_lease, but honor an explicit GPU.
        import fcntl
        directory = Path(common.LEASE_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"gpu{requested.split(':')[1]}.lock").open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                common.ensure_device(requested)
                yield requested
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ace", nargs="+", default=["store/ch-v1/ace.parquet"])
    p.add_argument("--ch-hourly", default="store/ch-v1/ch-hourly.parquet")
    p.add_argument("--raw-pattern", default="/home/t-lab01/.local/state/sundb/stage/ace_solar/year=*/part.parquet")
    p.add_argument("--folds", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=("cpu", "cuda:0", "cuda:1", "auto"), default="auto")
    p.add_argument("--max-folds", type=int)
    return p


def run(config, config_sha256=None, argv=None):
    args = parser().parse_args(argv)
    validate_config(config)
    frames = hpo.load_inputs(args.ace, args.ch_hourly, args.raw_pattern)
    branch, arm = config["data"]["feature_branch"], config["data"]["feature_arm"]
    feats, _, _ = hpo._arm_features(branch, arm, frames)
    folds = pd.read_parquet(args.folds)
    origins = select_origins(folds, feats.index, args.max_folds)
    cutoff = common.utc(config["data"]["training_cutoff"])
    if (origins < cutoff).any():
        raise ValueError("official evaluation origins must not precede training cutoff")
    hyper = config["hyperparameters"]
    with leased_device(args.device) as device:
        # fit_predict enforces origin+72h <= cutoff for every training label.
        # Speed keeps observed attrs intact; predict matrices only access <=origin.
        preds, fore, fit_seconds, pipeline_seconds = hpo.fit_predict(
            branch, arm, hyper["xgboost"], frames, frames["speed"], cutoff,
            origins, device=device, seed=hyper["seed"],
            max_train_origins=hyper["max_train_origins"])
        if str(fore.actual_device) != device:
            raise RuntimeError(f"device mismatch/fallback: {fore.actual_device} != {device}")
        # Existing fit_predict timing includes horizon feature construction.
        # Measure a second inference with ALL design preparation outside timer.
        _, matrix, anchor = fore._build_predict_matrix(feats, origins, frames["speed"])
        matrix = matrix[fore.columns]
        started = time.perf_counter()
        values = np.clip(anchor + fore.model.predict(matrix), *common.OUTPUT_RANGE_KMS)
        seconds = (time.perf_counter() - started) / len(origins)
        if not np.allclose(values, preds["pred_kms"], rtol=0, atol=1e-6):
            raise RuntimeError("timed inference differs from original fitted predictions")
    validate_predictions(preds, origins)
    # Official scorer keys are ISO-8601 Z strings, not pandas Timestamps.
    # Formatting is outside inference timing and leaves forecast values intact.
    preds["origin_last_input_utc"] = pd.to_datetime(
        preds["origin_last_input_utc"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(output, index=False)
    manifest = {"config": config, "config_sha256": config_sha256, "versions": common.runtime_versions(),
                "actual_device": device, "fit_seconds": fit_seconds,
                "inference_seconds_per_fold": seconds,
                "inference_timing_scope": "model.predict and residual addition/clipping only; excludes data loading, fit, matrix preparation, output serialization",
                "pipeline_predict_seconds_per_fold": pipeline_seconds,
                "n_origins": len(origins), "n_rows": len(preds),
                "smoke": args.max_folds is not None, "max_folds": args.max_folds,
                "observed_metadata_preserved": "observed" in frames["speed"].attrs,
                "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    output.with_suffix(".manifest.json").write_text(json.dumps(
        manifest, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n")
    print(json.dumps({"output": str(output), "n_rows": len(preds),
                      "inference_seconds_per_fold": seconds, "smoke": manifest["smoke"]}))
    return 0
