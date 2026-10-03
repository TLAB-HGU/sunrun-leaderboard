"""Reproduce a frozen numeric CH-XGBoost leaderboard forecast from cached features.

Candidate entry points keep the immutable configuration and call ``run``. The
feature table is built by ``experiments/xgboost_suvi_fusion`` from the pinned
ACE/SUVI sources; this runner only performs the deterministic frozen fit and
official-fold prediction.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.xgboost_suvi_fusion.ch_hpo import _predict_task  # noqa: E402


def run(config: dict) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ace", default="store/ch-v1/ace.parquet")
    parser.add_argument("--features", default="store/ch-v1/all-features.parquet")
    parser.add_argument("--folds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    folds = pd.read_parquet(args.folds)
    origins = pd.to_datetime(folds.origin_last_input_utc, utc=True)
    task = {
        "arm": config["data"]["feature_set"],
        "threshold": config["data"]["threshold"],
        "params": dict(config["hyperparameters"]["xgboost"]),
        "peak_weight": config["hyperparameters"]["peak_weight"],
        "rise_weight": config["hyperparameters"]["rise_weight"],
        "rounds": {i + 1: n for i, n in enumerate(config["hyperparameters"]["rounds_by_horizon"])},
        "features": args.features,
        "ace": args.ace,
        "cutoff": config["data"]["training_cutoff"],
        "start": str(origins.min()),
        "end": (origins.max() + pd.Timedelta(hours=72)).isoformat(),
        "official_origins": origins.astype(str).tolist(),
        "horizons": tuple(range(1, 73)),
        "deadline": time.time() + 24 * 3600,
        "prediction_output": args.output,
        "device": args.device,
    }
    _predict_task(task)
    # The submission validator compares the origin key to the fold strings.
    # Keep the canonical wire representation while preserving the forecast values.
    prediction = pd.read_parquet(args.output)
    prediction["origin_last_input_utc"] = pd.to_datetime(
        prediction["origin_last_input_utc"], utc=True
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    prediction.to_parquet(args.output, index=False)
