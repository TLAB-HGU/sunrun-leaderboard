"""Sequential, resumable Optuna search using only the pre-holdout validation set."""
from __future__ import annotations

import gc
import pickle
from pathlib import Path

from .modeling import (
    TRAIN_END,
    VALIDATION_END,
    VALIDATION_START,
    DirectForecaster,
    prediction_metrics,
    split_origins,
    utc,
)


def _sampler_state_path(storage, study_name):
    if not isinstance(storage, str) or not storage.startswith("sqlite:///"):
        return None
    database = Path(storage.removeprefix("sqlite:///"))
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in study_name)
    return database.with_name(f"{database.name}.{safe_name}.sampler.pkl")


def suggest_parameters(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 100, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 100, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [64, 128, 256]),
    }


def run_study(features, target, observed=None, *, storage=None,
              study_name="xgboost-suvi-fusion", n_trials=60, device="cpu",
              n_estimators=2000, train_end=TRAIN_END,
              validation_start=VALIDATION_START, validation_end=VALIDATION_END):
    import optuna

    if utc(validation_end) >= utc("2026-03-01"):
        raise ValueError("HPO must not access the holdout or leaderboard target period")
    if utc(train_end) >= utc(validation_start):
        raise ValueError("Training must end before validation starts")
    # Restrict all inputs passed to objective, including target and missingness.
    target = target.loc[:utc(validation_end)]
    observed = observed.loc[:utc(validation_end)] if observed is not None else None
    train = features.loc[:utc(train_end)]
    validation = features.loc[split_origins(features.index, validation_start, validation_end)]
    if train.empty or validation.empty:
        raise ValueError("Both training and validation periods need feature rows")

    def objective(trial):
        params = suggest_parameters(trial)
        params["n_estimators"] = n_estimators
        peak_weight = trial.suggest_float("peak_weight", 1.0, 5.0)
        rise_weight = trial.suggest_float("rise_weight", 1.0, 5.0)
        model = DirectForecaster(params, device=device, peak_weight=peak_weight,
                                 rise_weight=rise_weight)
        model.fit(train, target, train_end, validation_features=validation,
                  validation_end=validation_end)
        metrics = prediction_metrics(model.predict(validation), target, observed)
        rounds = model.best_rounds()
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("rounds_by_horizon", rounds)
        trial.set_user_attr("devices", device)
        del model
        gc.collect()
        return metrics["mse"]

    state_path = _sampler_state_path(storage, study_name)
    existing_trials = 0
    if storage is not None:
        summaries = optuna.study.get_all_study_summaries(storage)
        summary = next((item for item in summaries if item.study_name == study_name), None)
        existing_trials = summary.n_trials if summary is not None else 0
    if state_path and state_path.exists():
        state = pickle.loads(state_path.read_bytes())
        if state["n_trials"] != existing_trials:
            raise RuntimeError("Optuna sampler checkpoint does not match the study; start a new study")
        sampler = state["sampler"]
    else:
        if existing_trials:
            raise RuntimeError("Existing Optuna study has no sampler checkpoint; start a new study")
        sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10)

    study = optuna.create_study(study_name=study_name, storage=storage,
                               load_if_exists=True, direction="minimize",
                               sampler=sampler,
                               pruner=optuna.pruners.NopPruner())

    def checkpoint(current_study, _trial):
        if state_path is None:
            return
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = state_path.with_suffix(state_path.suffix + ".tmp")
        temporary.write_bytes(pickle.dumps({"n_trials": len(current_study.trials), "sampler": sampler}))
        temporary.replace(state_path)

    remaining = max(0, n_trials - len(study.trials))
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, callbacks=[checkpoint])
    return study
