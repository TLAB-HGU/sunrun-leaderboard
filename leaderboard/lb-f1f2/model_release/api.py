"""Public inference API. Loading pickle is safe only for trusted artifacts."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from .features import FeatureInputs, normalize_origins


class ReleasedModel:
    def __init__(self, bundle):
        if not isinstance(bundle, dict) or bundle.get("format_version") != 1:
            raise ValueError("Unsupported model release format")
        for field in ("experiment", "kind", "models", "state", "columns"):
            if field not in bundle:
                raise ValueError(f"Model bundle missing {field}")
        self.bundle = bundle
        self.experiment = bundle["experiment"]

    def predict(self, origins_utc, inputs):
        origins = normalize_origins(origins_utc)
        config = self.bundle.get("config", {})
        cutoff = (self.bundle["state"].get("cutoff")
                  or config.get("hyperparameters", {}).get("train_target_cutoff_utc")
                  or config.get("data", {}).get("train_target_cutoff_utc"))
        if cutoff is not None and (origins < pd.to_datetime(cutoff, utc=True)).any():
            raise ValueError("Origins must be at or after the fitted training cutoff")
        source = FeatureInputs(inputs, include_enlil=self.bundle["state"].get("include_enlil", False))
        grid = source.build(origins)
        columns = self.bundle["columns"]
        columns = columns["level"] if isinstance(columns, dict) else columns
        missing = set(columns).difference(grid.columns)
        if missing:
            raise ValueError(f"Model feature columns missing: {sorted(missing)}")
        kind = self.bundle["kind"]
        if kind == "residual_analog":
            from .analog_runtime import predict_residual
            predictions = predict_residual(self.bundle, origins, source.build, source.speed)
        elif kind == "event_branch":
            from .analog_runtime import build_d46_state, compute_d46_analog, predict_branch
            state, filled = build_d46_state(source.speed, source.ch)
            settings = self.bundle["state"]
            analog = compute_d46_analog(origins, state, filled, settings["analog_keys"],
                                       settings["metric"], settings["scaler"])
            predictions = predict_branch(self.bundle, grid, analog)
        else:
            from .regular_runtime import predict_bundle
            predictions = predict_bundle(self.bundle, grid)
        predictions = np.asarray(predictions, dtype=float).reshape(-1)
        if len(predictions) != len(grid) or not np.isfinite(predictions).all():
            raise ValueError("Prediction produced nonfinite values or incomplete 72-hour rows")
        return pd.DataFrame({"origin_last_input_utc": grid["origin"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                             "horizon_hours": grid["h"].astype(int), "pred_kms": predictions})


def load_model(path):
    """Load a trusted release PKL and return an object with predict(origins_utc, inputs)."""
    with open(path, "rb") as stream:
        return ReleasedModel(pickle.load(stream))
