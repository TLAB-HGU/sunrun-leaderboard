"""jsy301 solar wind 72h leaderboard experiment; the model and all settings are in CONFIG.

The model is trained once on the hourly ACE speed up to ``train_end_utc`` (the first fold origin) and then frozen.
Every fold is forecast on its own from the history ending at its origin, so no row with timestamp > origin is used.
``inference_seconds_per_fold`` is the mean ``time.perf_counter()`` time of that per-fold forecast call; data loading,
training, history slicing and output writing are excluded. The first fold is run once untimed as a warm-up.

Input: hourly ACE files with ``timestamp_utc`` and ``ace_speed_kms`` (sun-db ``timeseries/ace_solar/year=YYYY/part.parquet``
or a CSV export with those columns). Files are applied in order; a later file wins on shared hours. Missing hours are
forward-filled, the same rule as the scoring truth.

Example:
    python scripts/jsy301/<experiment>/<config_sha256>.py \
        --ace solar.csv ace_solar_2022_2026.parquet --folds folds.parquet --output pred.parquet

Environment: Python 3.12 with the library versions in CONFIG["hyperparameters"]["libraries"]; NN models use one GPU.
Generated from neuralforecast/experiments/solar_baselines/lb_experiment_template.py by make_lb_submission.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON_HOURS = 72
DEFAULT_FOLDS_REPO = "tlabtlab/sunrun-lb-store"
CONFIG = {'data': {'evaluation_folds': 'tlabtlab/sunrun-lb-store/folds.parquet',
          'features': ['ace_speed_kms'],
          'frequency': '1h',
          'history_start_utc': '2000-08-01T00:00:00Z',
          'inputs': ['collector spaceweather/solar.csv (hours before 2022-08-22)',
                     'sun-db timeseries/ace_solar year=2022..2026 (identical to solar.csv on the overlap)'],
          'missing_values': 'forward_fill',
          'source': 'NASA ACE SWEPAM bulk speed',
          'train_end_utc': '2026-05-31T23:00:00Z',
          'value_column': 'ace_speed_kms'},
 'hyperparameters': {'estimator': {'colsample_bytree': 0.8,
                                   'learning_rate': 0.03,
                                   'n_estimators': 1000,
                                   'n_jobs': 16,
                                   'num_leaves': 63,
                                   'subsample': 0.8,
                                   'subsample_freq': 1},
                     'horizon_hours': 72,
                     'input_hours': 1000,
                     'lags': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                              23, 24, 48, 72, 96, 168, 336, 648],
                     'libraries': {'lightgbm': '4.7.0',
                                   'mlforecast': '1.1.0',
                                   'numpy': '2.3.2',
                                   'pandas': '2.3.2',
                                   'python': '3.12.14',
                                   'xgboost': '3.4.1'},
                     'model': 'LightGBM',
                     'output_clip_kms': [1.0, 3000.0],
                     'rolling_mean_windows': [6, 24, 168],
                     'rolling_std_windows': [24]}}
CONFIG_SHA256 = "a01d712781728645b017527e66868d23eec8bda42e96137d298fda06fa7a8149"


def verify_config_identity() -> None:
    canonical = json.dumps(CONFIG, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    actual = hashlib.sha256(canonical.encode()).hexdigest()
    if actual != CONFIG_SHA256 or Path(__file__).stem != CONFIG_SHA256:
        raise RuntimeError(f"config identity mismatch: expected {CONFIG_SHA256}, got {actual}")


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, encoding="utf-8-sig")


def build_series(paths: list[str], last_origin: pd.Timestamp) -> pd.Series:
    """Hourly, forward-filled speed from the first valid hour to the last origin (UTC, tz-naive index)."""
    parts = []
    for path in paths:
        table = read_table(path)
        index = pd.to_datetime(table["timestamp_utc"], utc=True).dt.tz_convert(None)
        parts.append(pd.Series(pd.to_numeric(table["ace_speed_kms"], errors="coerce").to_numpy(float), index=index))
    series = pd.concat(parts)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if series.index.max() < last_origin:
        raise ValueError(f"input ends at {series.index.max()}, before the last origin {last_origin}")
    series = series.loc[series.first_valid_index():last_origin]
    return series.reindex(pd.date_range(series.index[0], last_origin, freq="h")).ffill()


# ------------------------------------------------------------------------ models
def fit_model(ds: np.ndarray, y: np.ndarray, train_end: int, accelerator: str):
    """Train on rows ``<= train_end``; return (prepare, predict, info).

    ``prepare(p)`` slices the history ending at row ``p`` (untimed); ``predict(x)`` returns the 72h forecast (timed).
    """
    hp = CONFIG["hyperparameters"]
    kind = hp["model"]
    n_in = hp.get("input_hours")

    def frame(p):
        return pd.DataFrame({"unique_id": "speed", "ds": ds[p - n_in + 1:p + 1], "y": y[p - n_in + 1:p + 1].astype(np.float32)})

    if kind == "HistoricAverage":
        return (lambda p: y[:p + 1]), (lambda x: np.full(HORIZON_HOURS, x.mean())), {}
    if kind == "WindowAverage":
        return (lambda p: y[p - n_in + 1:p + 1]), (lambda x: np.full(HORIZON_HOURS, x.mean())), {}
    if kind == "SeasonalNaive":
        return (lambda p: y[p - n_in + 1:p + 1]), (lambda x: np.resize(x, HORIZON_HOURS)), {}

    if kind in ("AutoARIMA", "AutoETS"):
        from statsforecast import models as sf_models
        model = getattr(sf_models, kind)(season_length=hp["season_length"])
        model.fit(y[train_end - hp["fit_window_hours"] + 1:train_end + 1])
        return (lambda p: y[p - n_in + 1:p + 1]), (lambda x: np.asarray(model.forward(y=x, h=HORIZON_HOURS)["mean"])), {}

    train = pd.DataFrame({"unique_id": "speed", "ds": ds[:train_end + 1], "y": y[:train_end + 1].astype(np.float32)})

    if kind in ("LightGBM", "XGBoost"):
        from mlforecast import MLForecast
        from mlforecast.lag_transforms import RollingMean, RollingStd
        if kind == "LightGBM":
            import lightgbm as lgb
            estimator = lgb.LGBMRegressor(**hp["estimator"], verbosity=-1)
        else:
            import xgboost as xgb
            estimator = xgb.XGBRegressor(**hp["estimator"])
        transforms = [RollingMean(window_size=w) for w in hp["rolling_mean_windows"]]
        transforms += [RollingStd(window_size=w) for w in hp["rolling_std_windows"]]
        mlf = MLForecast(models={kind: estimator}, freq="h", lags=hp["lags"], lag_transforms={1: transforms})
        mlf.fit(train)
        return frame, (lambda x: mlf.predict(h=HORIZON_HOURS, new_df=x)[kind].to_numpy()), {}

    if kind in ("LSTM", "NHITS", "PatchTST"):
        import neuralforecast.models as nf_models
        from neuralforecast import NeuralForecast
        for name in ("pytorch_lightning", "lightning.pytorch", "lightning_fabric"):
            logging.getLogger(name).setLevel(logging.ERROR)
        model = getattr(nf_models, kind)(h=HORIZON_HOURS, alias=kind, **hp["model_kwargs"], accelerator=accelerator, devices=1,
                                         enable_progress_bar=False, logger=False, enable_checkpointing=False,
                                         enable_model_summary=False)
        nf = NeuralForecast(models=[model], freq="h")
        nf.fit(train, val_size=hp["val_size_hours"])
        train_traj, valid_traj = nf.models[0].train_trajectories, nf.models[0].valid_trajectories
        best = min(valid_traj, key=lambda t: t[1]) if valid_traj else None
        info = {"steps_trained": int(train_traj[-1][0]) if train_traj else 0,
                "best_valid_step": int(best[0]) if best else None,
                "best_valid_loss": float(best[1]) if best else None}
        return frame, (lambda x: nf.predict(df=x)[kind].to_numpy()), info

    raise ValueError(f"unknown model {kind}")


def generate_predictions(series: pd.Series, folds: pd.DataFrame, accelerator: str = "gpu") -> tuple[pd.DataFrame, dict]:
    origins = pd.DatetimeIndex(pd.to_datetime(folds["origin_last_input_utc"], utc=True)).tz_convert(None)
    train_end_ts = pd.Timestamp(CONFIG["data"]["train_end_utc"]).tz_convert(None)
    if origins.min() < train_end_ts:
        raise ValueError("train_end_utc is after the first origin: training would see future data")
    positions = series.index.get_indexer(origins)
    if (positions < 0).any():
        raise ValueError("series does not contain every fold origin")
    ds, y = series.index.to_numpy(), series.to_numpy(np.float64)
    n_in = CONFIG["hyperparameters"].get("input_hours") or 0
    if (positions + 1 < n_in).any():
        raise ValueError(f"at least {n_in} hours are required before every origin")

    fit_started = time.perf_counter()
    prepare, predict, info = fit_model(ds, y, series.index.get_loc(train_end_ts), accelerator)
    fit_seconds = time.perf_counter() - fit_started

    predict(prepare(positions[0]))  # warm-up, untimed
    predictions = np.empty((len(origins), HORIZON_HOURS))
    inference_seconds = 0.0
    for i, p in enumerate(positions):
        x = prepare(p)
        started = time.perf_counter()
        predictions[i] = predict(x)
        inference_seconds += time.perf_counter() - started
    if not np.isfinite(predictions).all():
        raise ValueError("non-finite predictions")
    low, high = CONFIG["hyperparameters"]["output_clip_kms"]
    output = pd.DataFrame({
        "origin_last_input_utc": np.repeat(origins.strftime("%Y-%m-%dT%H:%M:%SZ"), HORIZON_HOURS),
        "horizon_hours": np.tile(np.arange(1, HORIZON_HOURS + 1), len(origins)),
        "pred_kms": np.clip(predictions, low, high).ravel(),
    })
    return output, {"fit_seconds": fit_seconds, "inference_seconds_per_fold": inference_seconds / len(origins), **info}


def main() -> None:
    verify_config_identity()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ace", nargs="+", required=True, help="ACE CSV/parquet files, later files win on overlaps")
    parser.add_argument("--folds", help="folds parquet; downloaded from the private HF store when omitted")
    parser.add_argument("--output", default=f"{CONFIG['hyperparameters']['model'].lower()}.parquet")
    parser.add_argument("--accelerator", default="gpu", help="NN models only")
    args = parser.parse_args()
    if args.folds:
        folds_path = args.folds
    else:
        from huggingface_hub import hf_hub_download
        folds_path = hf_hub_download(DEFAULT_FOLDS_REPO, "folds.parquet", repo_type="dataset")
    folds = read_table(folds_path)
    last_origin = pd.to_datetime(folds["origin_last_input_utc"], utc=True).max().tz_convert(None)
    prediction, timing = generate_predictions(build_series(args.ace, last_origin), folds, args.accelerator)
    prediction.to_parquet(args.output, index=False)
    print(json.dumps({"output": str(Path(args.output).resolve()), "n_folds": int(len(prediction) / HORIZON_HOURS), **timing}))


if __name__ == "__main__":
    main()
