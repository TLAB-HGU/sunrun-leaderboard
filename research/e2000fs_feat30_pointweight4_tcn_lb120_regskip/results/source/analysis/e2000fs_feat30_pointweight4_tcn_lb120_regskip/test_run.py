"""Regression checks for training boundaries and training-only preprocessing."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def runner():
    spec = importlib.util.spec_from_file_location('feat30_tcn_run', HERE / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_temporal_split_purges_72_hours():
    module = runner()
    origins = pd.date_range('2025-11-25', '2025-12-10', freq='h', tz='UTC')
    fit, val = module.temporal_split(origins)
    assert (origins[fit] + pd.Timedelta(hours=72)).max() < origins[val].min()
    assert not np.intersect1d(fit, val).size
    with pytest.raises(ValueError, match='Empty'):
        module.temporal_split(origins[:10])


def test_normalization_never_fits_validation_or_unused_history():
    module = runner()
    rng = np.random.default_rng(42)
    inputs = dict(history_hourly=rng.normal(size=(360, 30)).astype(np.float32),
                  train=dict(positions=np.array([119, 120, 359]),
                             future=rng.normal(size=(3, 72, 9)).astype(np.float32),
                             y=np.full((3, 72), 450., dtype=np.float32)))
    stats = module.normalization(inputs, np.array([0, 1]))
    inputs['history_hourly'][121:] = 999999.
    inputs['train']['future'][2] = 999999.
    inputs['train']['y'][2] = 999999.
    again = module.normalization(inputs, np.array([0, 1]))
    for name in ('history', 'future'):
        for key in ('median', 'mean', 'std'):
            np.testing.assert_array_equal(stats[name][key], again[name][key])
    assert stats['y_mean'] == again['y_mean'] == 450.
    np.testing.assert_array_equal(stats['weight_mean'], again['weight_mean'])


def test_dataset_weights_use_physical_target_not_standardized_target():
    module = runner()
    inputs = dict(history_hourly=np.ones((121, 30), dtype=np.float32),
                  train=dict(positions=np.array([119, 120]),
                             future=np.ones((2, 72, 9), dtype=np.float32),
                             y=np.full((2, 72), 450., dtype=np.float32)))
    inputs['train']['y'][0, 0] = 600.
    stats = module.normalization(inputs, np.array([0, 1]))
    dataset = module.Windows(inputs['history_hourly'], inputs['train'], [0, 1], stats)
    hist, future, target, weight = dataset[0]
    assert hist.shape == (120, 30) and future.shape == (72, 9)
    assert weight[0] == pytest.approx(1.6)
    assert weight[1] == 1.
    assert dataset[1][3][0] == pytest.approx(.4)
    assert target[0] > target[1]


def test_complete_prediction_grid_preserves_all_origins_and_horizons():
    module = runner()
    origins = pd.date_range('2026-02-28T23:00Z', periods=3, freq='h')
    pred = np.arange(216, dtype=float).reshape(3, 72)
    frame = module.frame(origins, pred)
    assert len(frame) == 216
    assert not frame.duplicated(['origin_last_input_utc', 'horizon_hours']).any()
    np.testing.assert_array_equal(frame.pred_kms.to_numpy().reshape(3, 72), pred)
    assert set(frame.horizon_hours) == set(range(1, 73))
    with pytest.raises(ValueError, match='Invalid'):
        module.frame(origins, pred[:, :71])


def test_internal_gate_requires_both_total_and_short_horizon_improvement():
    module = runner()
    previous = {'mse': 8960., 'mse_h1': 6189.}
    assert module.internal_decision({'mse': 8000., 'mse_h1': 3000.}, previous)['accepted']
    assert not module.internal_decision({'mse': 8000., 'mse_h1': 7000.}, previous)['accepted']
    assert not module.internal_decision({'mse': 9000., 'mse_h1': 3000.}, previous)['accepted']
    assert not module.internal_decision(previous, previous)['accepted']
    with pytest.raises(ValueError, match='finite'):
        module.internal_decision({'mse': float('nan'), 'mse_h1': 0.}, previous)
