import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = next((Path(__file__).parents[1] / "scripts/baseline/mean_reversion").glob("*.py"))
SPEC = importlib.util.spec_from_file_location("mean_reversion_experiment", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_mean_reversion_predictions_and_timing():
    MODULE.verify_config_identity()
    timestamps = pd.date_range("2025-01-01", periods=800, freq="h", tz="UTC")
    values = np.linspace(300.0, 500.0, len(timestamps))
    series = pd.DataFrame({"timestamp_utc": timestamps, "filled_speed_kms": values})
    origins = timestamps[[700, 701]]
    folds = pd.DataFrame({"origin_last_input_utc": origins})

    prediction, seconds = MODULE.generate_predictions(series, folds)

    rolling = pd.Series(values, index=timestamps).rolling(648).mean()
    expected_first = rolling.loc[origins[0]] + (values[700] - rolling.loc[origins[0]]) * np.exp(-1 / 48)
    assert len(prediction) == 2 * 72
    assert prediction.iloc[0]["pred_kms"] == pytest.approx(expected_first)
    assert prediction.groupby("origin_last_input_utc").size().eq(72).all()
    assert seconds > 0


def test_mean_reversion_rejects_non_hourly_series():
    series = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2025-01-01T00:00Z", "2025-01-01T02:00Z"]),
        "filled_speed_kms": [400.0, 410.0],
    })
    folds = pd.DataFrame({"origin_last_input_utc": ["2025-01-01T02:00Z"]})
    with pytest.raises(ValueError, match="one row per hour"):
        MODULE.generate_predictions(series, folds)
