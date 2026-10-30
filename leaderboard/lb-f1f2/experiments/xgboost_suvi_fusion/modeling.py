"""Direct horizon models and chronological, observation-aware diagnostics."""
from __future__ import annotations

import json
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HORIZONS = tuple(range(1, 73))
TRAIN_END = "2025-07-31T23:00:00Z"
VALIDATION_START = "2025-08-01T00:00:00Z"
VALIDATION_END = "2026-02-28T23:00:00Z"
HOLDOUT_START = "2026-03-01T00:00:00Z"
HOLDOUT_END = "2026-05-31T23:00:00Z"
OUTPUT_RANGE_KMS = (1.0, 3000.0)


def ensure_device(device):
    """Fail closed when XGBoost silently falls back from CUDA to CPU."""
    from xgboost import XGBRegressor

    for requested in str(device).split(","):
        if not requested.startswith("cuda"):
            continue
        probe = XGBRegressor(n_estimators=1, max_depth=1, tree_method="hist", device=requested, n_jobs=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe.fit(np.array([[0.0], [1.0]], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32))
        actual = json.loads(probe.get_booster().save_config())["learner"]["generic_param"]["device"]
        matched = actual.startswith("cuda") if requested == "cuda" else actual == requested
        if not matched:
            raise RuntimeError(f"XGBoost requested {requested!r} but initialized {actual!r}")


def utc(value):
    return pd.to_datetime(value, utc=True)


def check_index(obj):
    if not isinstance(obj.index, pd.DatetimeIndex) or obj.index.tz is None:
        raise ValueError("Inputs require timezone-aware DatetimeIndex")
    if obj.index.has_duplicates or not obj.index.is_monotonic_increasing:
        raise ValueError("Input timestamps must be unique and sorted")


def split_origins(index, start, end, horizon=72):
    """Origins whose complete forecast lies within the specified target period."""
    index = pd.DatetimeIndex(utc(index))
    return index[(index >= utc(start)) & (index + pd.Timedelta(hours=horizon) <= utc(end))]


def horizon_training_data(features, target, horizon, cutoff):
    check_index(features)
    check_index(target)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    times = features.index + pd.Timedelta(hours=horizon)
    values = target.reindex(times).to_numpy(dtype=float)
    mask = (times <= utc(cutoff)) & np.isfinite(values)
    return features.loc[mask], values[mask]


def sample_weights(target, last_value, peak_weight=1.0, rise_weight=1.0, weight_cap=None):
    if peak_weight < 1 or rise_weight < 1:
        raise ValueError("event weights must be at least one")
    weights = np.where(target >= 600, peak_weight, 1.0) * np.where(
        target - last_value >= 100, rise_weight, 1.0
    )
    if weight_cap is not None:
        if weight_cap < 1:
            raise ValueError("weight cap must be at least one")
        weights = np.minimum(weights, weight_cap)
    return weights


class DirectForecaster:
    """One deterministic XGBoost estimator per lead time, sharing parameters."""

    def __init__(self, params=None, *, device="cpu", horizons=HORIZONS,
                 peak_weight=1.0, rise_weight=1.0, early_stopping_rounds=50,
                 weight_cap=None, n_jobs=1):
        self.params = dict(params or {})
        self.devices = tuple(str(device).split(","))
        if not self.devices or any(not value for value in self.devices):
            raise ValueError("at least one nonempty XGBoost device is required")
        self.horizons = tuple(horizons)
        self.peak_weight = peak_weight
        self.rise_weight = rise_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.weight_cap = weight_cap
        self.n_jobs = n_jobs
        self.models = {}

    def fit(self, features, target, cutoff, *, validation_features=None,
            validation_end=None, rounds_by_horizon=None):
        from xgboost import XGBRegressor

        check_index(features)
        check_index(target)
        if validation_features is not None:
            check_index(validation_features)
            if validation_end is None or (validation_features.index <= utc(cutoff)).any():
                raise ValueError("Validation must follow training cutoff and have an end")
        self.columns = list(features.columns)
        def fit_horizon(horizon):
            x, y = horizon_training_data(features, target, horizon, cutoff)
            if x.empty:
                raise ValueError(f"No training labels for horizon {horizon}")
            last = target.reindex(x.index).to_numpy(dtype=float)
            if not np.isfinite(last).all():
                raise ValueError("Training origins need finite last observed/filled speed")
            params = dict(self.params)
            params.update(objective="reg:squarederror", tree_method="hist",
                          device=self.devices[(horizon - 1) % len(self.devices)], random_state=42, n_jobs=self.n_jobs)
            params.setdefault("n_estimators", 2000)
            params.pop("early_stopping_rounds", None)
            if rounds_by_horizon is not None:
                params["n_estimators"] = int(rounds_by_horizon[horizon])
            fit_args = {"sample_weight": sample_weights(y, last, self.peak_weight, self.rise_weight, self.weight_cap),
                        "verbose": False}
            if validation_features is not None:
                vx, vy = horizon_training_data(validation_features, target, horizon, validation_end)
                if vx.empty:
                    raise ValueError(f"No validation labels for horizon {horizon}")
                params["early_stopping_rounds"] = self.early_stopping_rounds
                fit_args["eval_set"] = [(vx[self.columns], vy)]
            model = XGBRegressor(**params)
            model.fit(x, y, **fit_args)
            return horizon, model

        if len(self.devices) == 1:
            self.models = dict(map(fit_horizon, self.horizons))
        else:
            groups = [self.horizons[i::len(self.devices)] for i in range(len(self.devices))]

            def fit_group(group):
                return [fit_horizon(horizon) for horizon in group]

            with ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
                trained_groups = executor.map(fit_group, groups)
                self.models = dict(sorted(item for group in trained_groups for item in group))
        return self

    def best_rounds(self):
        return {h: int(getattr(model, "best_iteration", model.n_estimators - 1)) + 1
                for h, model in self.models.items()}

    def predict(self, features):
        check_index(features)
        if not self.models:
            raise ValueError("Fit the forecaster before predicting")
        rows = [pd.DataFrame({"origin_last_input_utc": features.index,
                              "horizon_hours": h,
                              "pred_kms": np.clip(model.predict(features[self.columns]), *OUTPUT_RANGE_KMS)})
                for h, model in self.models.items()]
        return pd.concat(rows, ignore_index=True).sort_values(
            ["origin_last_input_utc", "horizon_hours"], ignore_index=True)


def prediction_metrics(predictions, target, observed=None):
    """Observed-event diagnostics; missing targets never count as quiet events."""
    p = predictions.copy()
    origins = utc(p["origin_last_input_utc"])
    times = origins + pd.to_timedelta(p["horizon_hours"], unit="h")
    truth = target.reindex(times).to_numpy(dtype=float)
    pred = p["pred_kms"].to_numpy(dtype=float)
    if not len(p) or not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("Metrics require nonempty, finite aligned predictions and targets")
    obs = (np.ones(len(p), dtype=bool) if observed is None
           else observed.reindex(times).fillna(False).to_numpy(dtype=bool))
    last = target.reindex(origins).to_numpy(dtype=float)
    p["truth"], p["pred"], p["observed"], p["last"] = truth, pred, obs, last
    diagnostics = {}
    for h, group in p.groupby("horizon_hours"):
        g = group[group.observed]
        actual, forecast = g.truth >= 600, g.pred >= 600
        tp = int((actual & forecast).sum())
        denominator = int(actual.sum() + forecast.sum())
        rise = (g.truth - g["last"]) >= 100
        diagnostics[int(h)] = {
            "peak_f1": 2 * tp / denominator if denominator else 0.0,
            "peak_bias": float((g.pred[actual] - g.truth[actual]).mean()) if actual.any() else None,
            "peak_count": int(actual.sum()),
            "rise_recall": float(((g.pred - g["last"] >= 100) & rise).sum() / rise.sum()) if rise.any() else None,
        }
    selected = [diagnostics.get(h) for h in (24, 48, 72)]
    complete = all(d is not None and d["peak_count"] > 0 for d in selected)
    return {"mse": float(np.mean((pred - truth) ** 2)),
            "mse_observed": float(np.mean((pred[obs] - truth[obs]) ** 2)) if obs.any() else None,
            "peak_macro_f1": float(np.mean([d["peak_f1"] for d in selected])) if complete else None,
            "peak_bias": float(np.mean([d["peak_bias"] for d in selected])) if complete else None,
            "by_horizon": diagnostics}


def passes_gate(candidate, reference, *, require_observed=False):
    required = ["mse", "peak_macro_f1", "peak_bias"] + (["mse_observed"] if require_observed else [])
    if any(m.get(k) is None or not np.isfinite(m[k]) for m in (candidate, reference) for k in required):
        return False
    return bool(candidate["mse"] < reference["mse"]
                and candidate["peak_macro_f1"] >= reference["peak_macro_f1"] + 0.02
                and abs(candidate["peak_bias"]) <= abs(reference["peak_bias"]) - 10
                and (not require_observed or candidate["mse_observed"] < reference["mse_observed"]))


def select_feasible_trial(trials, reference):
    """Deterministic lowest-MSE feasible candidate, then trial number as tie-break."""
    feasible = [t for t in trials if t.get("metrics") and passes_gate(t["metrics"], reference)]
    return min(feasible, key=lambda t: (t["metrics"]["mse"], t["number"])) if feasible else None
