"""Generate the 648-hour seasonal-naive baseline and measure inference time."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON_HOURS = 72
SEASON_LENGTH_HOURS = 648
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
        "horizon_hours": HORIZON_HOURS,
        "season_length_hours": SEASON_LENGTH_HOURS,
        "strategy": "seasonal_naive",
    },
}
CONFIG_SHA256 = "e82281cb844fb18079acceb7579f554d639a65f8e9bba5fd1016b32713f72f77"


def verify_config_identity() -> None:
    canonical = json.dumps(CONFIG, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    actual = hashlib.sha256(canonical.encode()).hexdigest()
    if actual != CONFIG_SHA256 or Path(__file__).stem != CONFIG_SHA256:
        raise RuntimeError(f"config identity mismatch: expected {CONFIG_SHA256}, got {actual}")


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def generate_predictions(series_frame: pd.DataFrame, folds: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    if not {"timestamp_utc", "filled_speed_kms"}.issubset(series_frame.columns):
        raise ValueError("series needs timestamp_utc and filled_speed_kms")
    if "origin_last_input_utc" not in folds:
        raise ValueError("folds is missing origin_last_input_utc")
    timestamps = pd.to_datetime(series_frame["timestamp_utc"], utc=True)
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("series timestamps must be unique and increasing")
    if len(timestamps) > 1 and not (timestamps.diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("series must have exactly one row per hour")
    values = pd.to_numeric(series_frame["filled_speed_kms"], errors="raise").to_numpy()
    origins = pd.DatetimeIndex(pd.to_datetime(folds["origin_last_input_utc"], utc=True))
    if not len(origins):
        raise ValueError("folds must contain at least one origin")
    positions = timestamps.to_numpy().searchsorted(origins.to_numpy())
    if (positions >= len(timestamps)).any() or not (timestamps.iloc[positions].to_numpy() == origins.to_numpy()).all():
        raise ValueError("series does not contain every fold origin")
    if (positions + 1 - SEASON_LENGTH_HOURS < 0).any():
        raise ValueError(f"at least {SEASON_LENGTH_HOURS} hours are required before every origin")
    horizons = np.arange(1, HORIZON_HOURS + 1)
    predictions = np.empty((len(origins), HORIZON_HOURS), dtype=float)
    started = time.perf_counter()
    for i, position in enumerate(positions):
        predictions[i] = values[position + horizons - SEASON_LENGTH_HOURS]
    inference_seconds = time.perf_counter() - started
    output = pd.DataFrame({
        "origin_last_input_utc": np.repeat(origins.strftime("%Y-%m-%dT%H:%M:%SZ"), HORIZON_HOURS),
        "horizon_hours": np.tile(horizons, len(origins)),
        "pred_kms": predictions.ravel(),
    })
    return output, inference_seconds / len(origins)


def main() -> None:
    verify_config_identity()
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", required=True)
    parser.add_argument("--folds")
    parser.add_argument("--output", default="seasonal_naive_648.parquet")
    args = parser.parse_args()
    if args.folds:
        folds_path = args.folds
    else:
        from huggingface_hub import hf_hub_download
        folds_path = hf_hub_download(DEFAULT_FOLDS_REPO, "folds.parquet", repo_type="dataset")
    prediction, seconds = generate_predictions(read_table(args.series), read_table(folds_path))
    prediction.to_parquet(args.output, index=False)
    print(json.dumps({"output": str(Path(args.output).resolve()), "n_folds": int(len(prediction) / HORIZON_HOURS),
                      "inference_seconds_per_fold": seconds}))


if __name__ == "__main__":
    main()
