"""Generate the published mean-reversion baseline and measure inference time.

The input series must be hourly and contain ``timestamp_utc`` and
``filled_speed_kms``. Data loading, rolling-feature construction, and file output
are intentionally excluded from the reported inference time.

Example:
    python scripts/baseline/mean_reversion/095a1fce921eaa40ef3ab29a582af4617eb34dab8327cc7fdd8acdf32560ee66.py \
        --series hourly_series.csv --folds folds.parquet --output mean_reversion.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

HORIZON_HOURS = 72
MEAN_WINDOW_HOURS = 648
TAU_HOURS = 48
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
        "mean_window_hours": MEAN_WINDOW_HOURS,
        "tau_hours": TAU_HOURS,
    },
}
CONFIG_SHA256 = "095a1fce921eaa40ef3ab29a582af4617eb34dab8327cc7fdd8acdf32560ee66"


def verify_config_identity() -> None:
    canonical = json.dumps(CONFIG, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    actual = hashlib.sha256(canonical.encode()).hexdigest()
    if actual != CONFIG_SHA256 or Path(__file__).stem != CONFIG_SHA256:
        raise RuntimeError(f"config identity mismatch: expected {CONFIG_SHA256}, got {actual}")


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def generate_predictions(series_frame: pd.DataFrame, folds: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    required_series = {"timestamp_utc", "filled_speed_kms"}
    missing_series = required_series.difference(series_frame.columns)
    if missing_series:
        raise ValueError(f"series is missing columns: {sorted(missing_series)}")
    if "origin_last_input_utc" not in folds:
        raise ValueError("folds is missing column: origin_last_input_utc")

    timestamps = pd.to_datetime(series_frame["timestamp_utc"], utc=True)
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("series timestamps must be unique and increasing")
    if len(timestamps) > 1 and not (timestamps.diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("series must have exactly one row per hour")

    series = pd.Series(pd.to_numeric(series_frame["filled_speed_kms"], errors="raise").to_numpy(), index=timestamps)
    origins = pd.DatetimeIndex(pd.to_datetime(folds["origin_last_input_utc"], utc=True))
    missing_origins = origins.difference(series.index)
    if len(missing_origins):
        raise ValueError(f"series does not contain fold origin: {missing_origins[0]}")

    rolling_mean = series.rolling(MEAN_WINDOW_HOURS, min_periods=MEAN_WINDOW_HOURS).mean().loc[origins]
    if rolling_mean.isna().any():
        raise ValueError(f"at least {MEAN_WINDOW_HOURS} hourly values are required before every origin")
    last = series.loc[origins].to_numpy()
    mean = rolling_mean.to_numpy()
    horizons = np.arange(1, HORIZON_HOURS + 1)
    decay = np.exp(-horizons / TAU_HOURS)

    predictions = np.empty((len(origins), HORIZON_HOURS), dtype=float)
    started = time.perf_counter()
    for i in range(len(origins)):
        predictions[i] = mean[i] + (last[i] - mean[i]) * decay
    elapsed = time.perf_counter() - started
    seconds_per_fold = elapsed / len(origins) if len(origins) else 0.0

    output = pd.DataFrame({
        "origin_last_input_utc": np.repeat(origins.strftime("%Y-%m-%dT%H:%M:%SZ"), HORIZON_HOURS),
        "horizon_hours": np.tile(horizons, len(origins)),
        "pred_kms": predictions.ravel(),
    })
    return output, seconds_per_fold


def main() -> None:
    verify_config_identity()
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", required=True, help="CSV or parquet hourly ACE series")
    parser.add_argument("--folds", help="folds parquet; downloaded from the private HF store when omitted")
    parser.add_argument("--output", default="mean_reversion.parquet")
    args = parser.parse_args()

    folds_path = args.folds or hf_hub_download(DEFAULT_FOLDS_REPO, "folds.parquet", repo_type="dataset")
    predictions, seconds_per_fold = generate_predictions(read_table(args.series), read_table(folds_path))
    predictions.to_parquet(args.output, index=False)
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "n_folds": int(predictions["origin_last_input_utc"].nunique()),
        "inference_seconds_per_fold": seconds_per_fold,
    }))


if __name__ == "__main__":
    main()
