import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).parents[1] / "scripts/baseline"


def load_script(experiment):
    path = next((ROOT / experiment).glob("*.py"))
    spec = importlib.util.spec_from_file_location(f"baseline_{experiment}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample():
    timestamps = pd.date_range("2025-01-01", periods=800, freq="h", tz="UTC")
    series = pd.DataFrame({"timestamp_utc": timestamps, "filled_speed_kms": np.arange(800.0)})
    folds = pd.DataFrame({"origin_last_input_utc": timestamps[[700, 701]]})
    return series, folds


def test_naive_repeats_last_value():
    module = load_script("naive")
    module.verify_config_identity()
    prediction, seconds = module.generate_predictions(*sample())
    assert prediction.groupby("origin_last_input_utc")["pred_kms"].nunique().eq(1).all()
    assert prediction.iloc[0]["pred_kms"] == 700.0
    assert prediction.iloc[72]["pred_kms"] == 701.0
    assert seconds > 0


def test_seasonal_naive_uses_origin_plus_horizon_minus_648():
    module = load_script("seasonal_naive_648")
    module.verify_config_identity()
    prediction, seconds = module.generate_predictions(*sample())
    assert prediction.iloc[0]["pred_kms"] == 53.0  # origin index 700 + horizon 1 - 648
    assert prediction.iloc[71]["pred_kms"] == 124.0
    assert seconds > 0


@pytest.mark.parametrize("experiment", ["naive", "seasonal_naive_648"])
def test_simple_baselines_reject_non_hourly_input(experiment):
    module = load_script(experiment)
    series, folds = sample()
    with pytest.raises(ValueError, match="one row per hour"):
        module.generate_predictions(series.drop(index=10), folds)
