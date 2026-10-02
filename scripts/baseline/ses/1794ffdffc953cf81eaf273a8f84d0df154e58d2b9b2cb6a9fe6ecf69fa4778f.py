"""Refit optimized simple exponential smoothing per fold and forecast 72 hours."""
from __future__ import annotations

import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast.models import SimpleExponentialSmoothingOptimized

HORIZON_HOURS = 72
TRAINING_WINDOW_HOURS = 648
DEFAULT_FOLDS_REPO = "tlabtlab/sunrun-lb-store"
CONFIG = {
    "data": {
        "evaluation_folds": "tlabtlab/sunrun-lb-store/folds.parquet",
        "frequency": "1h",
        "missing_values": "forward_fill",
        "source": "NASA ACE SWEPAM bulk speed",
        "value_column": "filled_speed_kms",
    },
    "hyperparameters": {
        "alpha_bounds": [0.01, 0.99],
        "horizon_hours": HORIZON_HOURS,
        "model": "statsforecast.SimpleExponentialSmoothingOptimized",
        "objective": "training_one_step_squared_error",
        "training_window_hours": TRAINING_WINDOW_HOURS,
    },
}
CONFIG_SHA256 = "1794ffdffc953cf81eaf273a8f84d0df154e58d2b9b2cb6a9fe6ecf69fa4778f"


def verify_config_identity() -> None:
    canonical = json.dumps(CONFIG, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    actual = hashlib.sha256(canonical.encode()).hexdigest()
    if actual != CONFIG_SHA256 or Path(__file__).stem != CONFIG_SHA256:
        raise RuntimeError(f"config identity mismatch: expected {CONFIG_SHA256}, got {actual}")


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def fit_predict(item):
    index, history = item
    fit_started = time.perf_counter()
    model = SimpleExponentialSmoothingOptimized().fit(history)
    fit_seconds = time.perf_counter() - fit_started
    inference_started = time.perf_counter()
    prediction = np.asarray(model.predict(h=HORIZON_HOURS)["mean"])
    inference_seconds = time.perf_counter() - inference_started
    return index, prediction, fit_seconds, inference_seconds


def generate_predictions(series_frame: pd.DataFrame, folds: pd.DataFrame, workers: int) -> tuple[pd.DataFrame, dict]:
    if not {"timestamp_utc", "filled_speed_kms"}.issubset(series_frame.columns):
        raise ValueError("series needs timestamp_utc and filled_speed_kms")
    if "origin_last_input_utc" not in folds:
        raise ValueError("folds is missing origin_last_input_utc")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    timestamps = pd.to_datetime(series_frame["timestamp_utc"], utc=True)
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("series timestamps must be unique and increasing")
    if len(timestamps) > 1 and not (timestamps.diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("series must have exactly one row per hour")
    values = pd.to_numeric(series_frame["filled_speed_kms"], errors="raise").to_numpy(dtype=np.float64)
    origins = pd.DatetimeIndex(pd.to_datetime(folds["origin_last_input_utc"], utc=True))
    if not len(origins):
        raise ValueError("folds must contain at least one origin")
    positions = timestamps.to_numpy().searchsorted(origins.to_numpy())
    if (positions >= len(timestamps)).any() or not (timestamps.iloc[positions].to_numpy() == origins.to_numpy()).all():
        raise ValueError("series does not contain every fold origin")
    if (positions + 1 - TRAINING_WINDOW_HOURS < 0).any():
        raise ValueError(f"at least {TRAINING_WINDOW_HOURS} hours are required before every origin")
    jobs = [(i, values[p - TRAINING_WINDOW_HOURS + 1:p + 1].copy()) for i, p in enumerate(positions)]
    fit_predict(jobs[0])  # warm JIT before worker processes are forked
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(fit_predict, jobs, chunksize=8))
    predictions = np.empty((len(origins), HORIZON_HOURS), dtype=float)
    fit_seconds = inference_seconds = 0.0
    for index, prediction, fit_elapsed, inference_elapsed in results:
        predictions[index] = prediction
        fit_seconds += fit_elapsed
        inference_seconds += inference_elapsed
    if not np.isfinite(predictions).all():
        raise ValueError("SES produced non-finite predictions")
    horizons = np.arange(1, HORIZON_HOURS + 1)
    output = pd.DataFrame({
        "origin_last_input_utc": np.repeat(origins.strftime("%Y-%m-%dT%H:%M:%SZ"), HORIZON_HOURS),
        "horizon_hours": np.tile(horizons, len(origins)),
        "pred_kms": predictions.ravel(),
    })
    return output, {"fit_seconds_per_fold": fit_seconds / len(origins),
                    "inference_seconds_per_fold": inference_seconds / len(origins)}


def main() -> None:
    verify_config_identity()
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", required=True)
    parser.add_argument("--folds")
    parser.add_argument("--output", default="ses.parquet")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.folds:
        folds_path = args.folds
    else:
        from huggingface_hub import hf_hub_download
        folds_path = hf_hub_download(DEFAULT_FOLDS_REPO, "folds.parquet", repo_type="dataset")
    prediction, timing = generate_predictions(read_table(args.series), read_table(folds_path), args.workers)
    prediction.to_parquet(args.output, index=False)
    print(json.dumps({"output": str(Path(args.output).resolve()), "n_folds": int(len(prediction) / HORIZON_HOURS), **timing}))


if __name__ == "__main__":
    main()
