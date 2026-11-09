import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def test_age_advances_across_missing_frames_without_future_fill(tmp_path):
    spec = importlib.util.spec_from_file_location('age_under_test', Path(__file__).with_name('age_repair.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    origins = pd.date_range('2025-01-01', periods=6, freq='h', tz='UTC')
    ages = [np.nan, .25, np.nan, np.nan, .5, np.nan]
    source = pd.DataFrame({'origin': origins, 'h': 1, 'img_age_h': ages})
    # Train/query duplicate origins must not create duplicate repaired features.
    source = pd.concat([source, source.iloc[[2]]], ignore_index=True)
    mod.RAW = tmp_path / 'raw.parquet'
    source.to_parquet(mod.RAW)
    repaired, raw = mod.repaired_age()
    assert repaired.index.is_unique
    assert np.isnan(repaired.iloc[0])
    np.testing.assert_allclose(repaired.iloc[1:], [.25, 1.25, 2.25, .5, 1.5])
    assert raw.isna().sum() == 4


def test_prediction_keys_preserve_origin_lead_alignment():
    spec = importlib.util.spec_from_file_location('age_frame', Path(__file__).with_name('age_repair.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    origins = pd.date_range('2026-01-01', periods=2, freq='6h', tz='UTC')
    predictions = np.arange(144).reshape(2, 72) + 200
    result = mod.frame(origins, predictions)
    assert not result.duplicated(['origin_last_input_utc', 'horizon_hours']).any()
    assert result.iloc[72].origin_last_input_utc == origins[1]
    assert result.iloc[72].horizon_hours == 1
    assert result.iloc[143].pred_kms == predictions[1, 71]
