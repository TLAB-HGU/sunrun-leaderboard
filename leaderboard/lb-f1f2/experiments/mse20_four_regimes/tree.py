"""Bounded combined-arm XGBoost adapter; orchestration owns split/deadline gates.

Inputs are the causal frames returned by the frozen hpo.load_inputs, already
sliced by the caller to its split end. No labels or official scores are loaded.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from experiments.ch_breakthrough_v2 import hpo
from experiments.ch_breakthrough_v2.common import (
    HORIZON_HOURS, MEAN_WINDOW_HOURS, build_features, utc,
)
from experiments.ch_breakthrough_v2.physical_quality import (
    build_ch_quality_frame, build_physics_frame,
)
from experiments.ch_breakthrough_v2.recurrence import RecurrenceResidualForecaster


def candidate_grid():
    """Fixed preregistration; callers persist this list before running trials."""
    path = Path(__file__).resolve().parents[2] / "store/ch-breakthrough-v2/upload/preparation.json"
    hp = json.loads(path.read_text())[0]["config"]["hyperparameters"]
    variants = [
        ("champion", {}),
        ("depth3", {"max_depth": 3}),
        ("depth4", {"max_depth": 4}),
        ("depth6", {"max_depth": 6, "min_child_weight": 8}),
        ("lambda5", {"reg_lambda": 5.0}),
        ("child10", {"min_child_weight": 10.0, "reg_lambda": 2.0}),
        ("trees200", {"n_estimators": 200, "learning_rate": 0.035}),
        ("trees300_regularized", {"n_estimators": 300, "learning_rate": 0.025,
                                   "max_depth": 4, "reg_lambda": 5.0}),
        ("alpha1", {"reg_alpha": 1.0, "reg_lambda": 2.0}),
    ]
    configs = [{"id": name, "params": {**hp["xgboost"], **changes},
                "max_train_origins": 6000, "seed": 42}
               for name, changes in variants]
    configs.append({"id": "champion_train12000", "params": dict(hp["xgboost"]),
                    "max_train_origins": 12000, "seed": 42})
    return configs


@lru_cache(maxsize=1)
def _canonical_columns():
    """Derive the fixed allowlist from frozen builders, never incoming labels."""
    index = pd.date_range("2000-01-01", periods=1, freq="h", tz="UTC")
    ace = pd.DataFrame({"filled_speed_kms": [400.0], "was_missing": [0]}, index=index)
    ch = pd.DataFrame(index=index)
    raw = pd.DataFrame({name: [1.0] for name in (
        "ace_density_cm3", "ace_temperature_k", "ace_bt_nt", "ace_speed_kms")}, index=index)
    frames = {"base": build_features(ace, ch),
              "physics": build_physics_frame(raw, index),
              "quality": build_ch_quality_frame(ch, index)}
    return tuple(hpo._arm_features("combined", hpo.PHYSICAL_ARM, frames)[1])


def _features(frames):
    features, columns, use_target = hpo._arm_features("combined", hpo.PHYSICAL_ARM, frames)
    if tuple(columns) != _canonical_columns() or not use_target:
        raise ValueError("combined feature schema differs from frozen causal allowlist")
    if not features.index.is_unique or not features.index.is_monotonic_increasing:
        raise ValueError("feature index must be unique and sorted")
    return features


def _origins(origins):
    origins = pd.DatetimeIndex(utc(origins)).sort_values()
    if not len(origins) or not origins.is_unique or not (origins == origins.floor("h")).all():
        raise ValueError("origins must be nonempty, unique and hourly")
    return origins


def _validate_predictions(preds, origins):
    keys = pd.MultiIndex.from_frame(preds[["origin_last_input_utc", "horizon_hours"]])
    expected = pd.MultiIndex.from_product([origins, range(1, 73)])
    if not keys.is_unique or set(keys) != set(expected):
        raise ValueError("prediction keys differ from requested complete 72h grid")
    values = preds["pred_kms"].to_numpy()
    if not (np.isfinite(values) & (values > 0) & (values <= 3000)).all():
        raise ValueError("invalid prediction speeds")


def fit_predict(frames, cutoff, origins, config, device, output_dir):
    """Fit all 72 residual horizons, persist inference artifacts, predict."""
    output_dir = Path(output_dir)
    if (output_dir / "metadata.json").exists() or (output_dir / "model.ubj").exists():
        raise FileExistsError(f"refusing to replace tree artifacts in {output_dir}")
    cutoff, origins = utc(cutoff), _origins(origins)
    features, speed = _features(frames), frames["speed"]
    if not speed.index.is_unique or not speed.index.is_monotonic_increasing:
        raise ValueError("speed index must be unique and sorted")
    cap = config["max_train_origins"]
    if not isinstance(cap, int) or not 0 < cap <= 12000:
        raise ValueError("max_train_origins must be an integer in [1, 12000]")
    # Bound fit inputs themselves, not just labels, before entering frozen code.
    train_features, train_speed = features.loc[:cutoff], speed.loc[:cutoff]
    pool = train_features.index[train_features.index + pd.Timedelta(hours=72) <= cutoff]
    depth = train_speed.rolling(MEAN_WINDOW_HOURS, min_periods=MEAN_WINDOW_HOURS).mean()
    pool = pool[depth.reindex(pool).notna().to_numpy()]
    if len(pool) > cap:
        pool = pool[np.linspace(0, len(pool) - 1, cap, dtype=int)]
    if not len(pool):
        raise ValueError("no training origins with complete history and labels")
    label_max = pool.max() + pd.Timedelta(hours=HORIZON_HOURS)
    if label_max > cutoff:
        raise ValueError("training label exceeds cutoff")
    fore = RecurrenceResidualForecaster(params=dict(config["params"]), device=device,
                                        seed=config["seed"], use_target=True)
    started = time.perf_counter()
    fore.fit(train_features, train_speed, cutoff, max_origins=cap)
    fit_seconds = time.perf_counter() - started
    preds = fore.predict(features, speed, origins)
    _validate_predictions(preds, origins)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.ubj"
    fore.model.get_booster().save_model(model_path)
    metadata = {
        "family": "combined_recurrence_xgboost", "config": config,
        "cutoff": cutoff.isoformat(), "train_label_max": label_max.isoformat(),
        "n_train_origins": len(pool),
        "train_origin_sha256": hashlib.sha256(pool.asi8.tobytes()).hexdigest(),
        "base_columns": fore.base_columns, "columns": fore.columns,
        "use_target": True, "actual_device": fore.actual_device,
        "fit_seconds": fit_seconds, "n_short_history_skipped": fore.n_short_history_skipped,
        "input_identities": frames.get("identities", {}),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "preprocessing": "frozen causal builders; native NaN; no fitted transforms",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return preds, metadata


def predict_saved(frames, origins, output_dir, device):
    """Restore a trusted local UBJ model and reproduce inference without fit."""
    from xgboost import Booster

    output_dir = Path(output_dir)
    meta = json.loads((output_dir / "metadata.json").read_text())
    model_path = output_dir / "model.ubj"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != meta["model_sha256"]:
        raise ValueError("saved tree model hash mismatch")
    features, origins = _features(frames), _origins(origins)
    if list(features.columns) != meta["base_columns"] or meta["use_target"] is not True:
        raise ValueError("saved feature schema mismatch")
    fore = RecurrenceResidualForecaster(params=meta["config"]["params"], device=device,
                                        seed=meta["config"]["seed"], use_target=True)
    booster = Booster(model_file=model_path)
    booster.set_param({"device": device, "nthread": 4})
    fore.base_columns, fore.columns = meta["base_columns"], meta["columns"]
    fore.cutoff = utc(meta["cutoff"])
    if booster.feature_names != fore.columns:
        raise ValueError("saved design schema mismatch")
    origins, design, anchor = fore._build_predict_matrix(features, origins, frames["speed"])
    residual = booster.inplace_predict(design[fore.columns])
    preds = pd.DataFrame({
        "origin_last_input_utc": np.repeat(origins, HORIZON_HOURS),
        "horizon_hours": np.tile(np.arange(1, HORIZON_HOURS + 1), len(origins)),
        "pred_kms": np.clip(anchor + residual, 1.0, 3000.0),
    })
    _validate_predictions(preds, origins)
    return preds
