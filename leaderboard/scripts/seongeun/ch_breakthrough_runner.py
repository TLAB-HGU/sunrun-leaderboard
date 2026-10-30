"""Refit frozen numerical CH submission configurations; never read official truth."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import glob
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from experiments.ch_breakthrough_v2 import common, hpo

ROOT = Path(__file__).resolve().parents[2]
SOURCE_NAMES = ("common", "hpo", "hpo_gpu", "recurrence", "physical_quality")
ARMS = {"recurrence": "target-recurrence+CH", "physical": "physical_ace_ch_quality",
        "combined": "physical_ace_ch_quality"}


def config_digest(config):
    return hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config, expected_config_sha256=None):
    if expected_config_sha256 is not None and config_digest(config) != expected_config_sha256:
        raise ValueError("CONFIG sha256 mismatch")
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


def validate_hashes(config, ace, ch_hourly, raw_pattern, source_dir=None):
    """Fail closed before loading any potentially altered model/data input."""
    source_dir = Path(source_dir or ROOT / "experiments/ch_breakthrough_v2")
    sources = config["data"]["source_sha256"]
    if set(sources) != set(SOURCE_NAMES):
        raise ValueError("source_sha256 must name every frozen source")
    for name in SOURCE_NAMES:
        if file_digest(source_dir / f"{name}.py") != sources[name]:
            raise ValueError(f"source sha256 mismatch: {name}")
    # The frozen experiments used one complete ACE parquet. Do not silently
    # authenticate only the first path while training on additional files.
    if len(ace) != 1:
        raise ValueError("frozen ACE input requires exactly one parquet")
    raw_files = sorted(glob.glob(raw_pattern))
    if not raw_files:
        raise ValueError("raw ACE pattern matched no files")
    actual = {"ace": file_digest(ace[0]), "ch_hourly": file_digest(ch_hourly),
              "raw_ace_years": hashlib.sha256("".join(
                  file_digest(p) for p in raw_files).encode()).hexdigest()}
    expected = config["data"]["input_sha256"]
    if set(expected) != set(actual):
        raise ValueError("input_sha256 must name all three frozen inputs")
    for name, digest in actual.items():
        if digest != expected[name]:
            raise ValueError(f"input sha256 mismatch: {name}")
    return actual


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


def run(config, expected_config_sha256=None, argv=None):
    args = parser().parse_args(argv)
    validate_config(config, expected_config_sha256)
    identities = validate_hashes(config, args.ace, args.ch_hourly, args.raw_pattern)
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
    manifest = {"config": config, "config_sha256": config_digest(config),
                "input_sha256": identities, "source_sha256": config["data"]["source_sha256"],
                "runner_sha256": file_digest(__file__), "folds_sha256": file_digest(args.folds),
                "prediction_sha256": file_digest(output), "versions": common.runtime_versions(),
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
