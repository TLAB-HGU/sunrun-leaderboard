"""Protocol tests without model training or official evaluation labels."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as e


def test_baseline_objective_is_exact():
    rng = np.random.default_rng(11)
    target = rng.uniform(250, 800, (40, 24))
    prediction = rng.uniform(250, 800, target.shape).astype(np.float32)
    expected = e.baseline.spectral_objective(target)(prediction, None)
    actual = e.make_objective(target, e.normalize_config({}))(prediction, None)
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)
        assert left.dtype == np.float32
    # The installed 2.1.4 multi-target objective contract returns two matching
    # (rows, num_target) arrays. No training is needed to check this contract.
    assert all(value.shape == target.shape for value in actual)


def test_spectral_removal_changes_gradient_and_hessian():
    target = np.array([[400., 600.], [600., 400.]])
    prediction = target + np.array([[2., -3.], [-4., 5.]])
    config = e.normalize_config({'spectral': False})
    gradient, hessian = e.make_objective(target, config)(prediction, None)
    weights = np.where(target >= 550., 4., 1.) / 2.5
    np.testing.assert_array_equal(gradient, ((prediction - target) * weights).astype(np.float32))
    np.testing.assert_array_equal(hessian, weights.astype(np.float32))
    plain = e.normalize_config({'spectral': False, 'tail_weight': 1})
    gradient, hessian = e.make_objective(target, plain)(prediction, None)
    np.testing.assert_array_equal(gradient, (prediction - target).astype(np.float32))
    np.testing.assert_array_equal(hessian, np.ones_like(hessian))


def test_feature_removal_removes_all_expanded_columns():
    config = e.normalize_config({})
    names = e.feature_names(config)
    assert len(e.expanded_names(names, 0, 24)) == 237
    reduced = e.feature_names(e.normalize_config({'exclude_features': ['rec26', 'v_lag0']}))
    columns = e.expanded_names(reduced, 24, 24)
    assert len(columns) == 212
    assert 'v_lag0' not in columns
    assert not any(column.startswith('rec26@') for column in columns)
    grid = {'C': np.ones((3, len(reduced['constant']))),
            'E': np.ones((3, 72, len(reduced['varying'])))}
    assert e.design(grid, 24, 24).shape == (3, len(columns))


def test_baseline_offset_and_scope_ablations():
    rng = np.random.default_rng(12)
    truth = rng.uniform(250, 800, (400, 72))
    prior = rng.uniform(250, 800, (400, 72))
    config = e.normalize_config({})
    offset, rows = e.estimate_offset(prior, truth, config)
    expected = .25 * (truth[-336:] - prior[-336:]).mean(0)
    np.testing.assert_array_equal(offset, expected)
    assert rows == 336
    global_offset, _ = e.estimate_offset(prior, truth, dict(config, offset_scope='global'))
    np.testing.assert_array_equal(global_offset, np.full(72, expected.mean()))
    block_offset, _ = e.estimate_offset(prior, truth, dict(config, offset_scope='block'))
    for start in (0, 24, 48):
        np.testing.assert_array_equal(block_offset[start:start + 24],
                                      np.full(24, expected[start:start + 24].mean()))
    all_offset, rows = e.estimate_offset(prior, truth, dict(config, offset_window='all', alpha=1))
    np.testing.assert_array_equal(all_offset, (truth - prior).mean(0))
    assert rows == 400


def test_clipping_and_no_offset_are_separate_operations():
    raw = np.full((2, 72), 400.)
    raw[:, 0], raw[:, 1] = -10., 3010.
    offset = np.zeros(72)
    offset[0], offset[1] = -5., 5.
    config = e.normalize_config({})
    corrected, parent = e.apply_offset(raw, offset, config)
    np.testing.assert_array_equal(parent, np.clip(raw, 1, 3000))
    np.testing.assert_array_equal(corrected, np.clip(np.clip(raw, 1, 3000) + offset, 1, 3000))
    corrected, parent = e.apply_offset(raw, offset, dict(config, clip_component=False, clip_final=False))
    np.testing.assert_array_equal(parent, raw)
    np.testing.assert_array_equal(corrected, raw + offset)
    corrected, parent = e.apply_offset(raw, offset, dict(config, offset=False))
    np.testing.assert_array_equal(corrected, parent)


def test_prior_clipping_changes_calibration_residuals():
    prior = np.full((336, 72), -5.)
    truth = np.full((336, 72), 400.)
    config = e.normalize_config({})
    clipped, _ = e.estimate_offset(prior, truth, config)
    unclipped, _ = e.estimate_offset(prior, truth, dict(config, clip_component=False))
    np.testing.assert_array_equal(clipped, np.full(72, .25 * 399))
    np.testing.assert_array_equal(unclipped, np.full(72, .25 * 405))
    with pytest.raises(ValueError, match='Insufficient'):
        e.estimate_offset(prior[:10], truth[:10], config)


def test_query_loader_never_requests_target(monkeypatch):
    requested = []
    origins = pd.date_range('2026-01-01', periods=2, freq='h', tz='UTC')

    def read_parquet(path, columns, filters):
        requested.extend(columns)
        if columns == ['origin', 'v_lag0']:
            return pd.DataFrame({'origin': origins, 'v_lag0': [400., 410.]})
        return pd.DataFrame({'origin': np.repeat(origins, 72),
                             'h': np.tile(np.arange(1, 73), 2), 'rec26': np.ones(144)})

    monkeypatch.setattr(e.baseline.pd, 'read_parquet', read_parquet)
    grid, truth = e.baseline.load_grid('unused', 'F1',
                                      {'constant': ['v_lag0'], 'varying': ['rec26']}, False)
    assert truth is None and 'y' not in requested
    assert grid['E'].shape == (2, 72, 1)


def test_runner_rejects_query_truth_before_fitting(monkeypatch, tmp_path):
    grid = {'origins': pd.date_range('2025-01-01', periods=2, freq='h', tz='UTC')}
    monkeypatch.setattr(e.baseline, 'load_grid', lambda *args: (grid, np.ones((2, 72))))
    monkeypatch.setattr(e, 'fit_predict', lambda *args: pytest.fail('fit must not be reached'))
    with pytest.raises(ValueError, match='evaluation targets'):
        e.run_period(e.normalize_config({}), 'unused', tmp_path, 'F1')


def test_reuse_preserves_predictions_and_rejects_wrong_seed(tmp_path):
    source = tmp_path / 'baseline_s42'
    model = source / 'F1_models'
    model.mkdir(parents=True)
    config = e.normalize_config({})
    identity = {'data': {'path': 'fixed', 'bytes': 123, 'mtime_ns': 1}}
    truth = np.tile(np.arange(72.), (400, 1)) + 500
    prior = np.full((400, 72), 450.)
    raw = np.full((8, 72), 400.)
    origins = pd.date_range('2026-01-01', periods=8, freq='h', tz='UTC').asi8
    cal_origins = pd.date_range('2025-11-01', periods=400, freq='h', tz='UTC').asi8
    np.savez(model / 'prior_predictions.npz', raw=prior, training_truth=truth, origins=cal_origins)
    np.savez(source / 'F1_raw_predictions.npz', raw=raw, origins=origins)
    report = dict(smoke=False, config=config, identity=identity, frozen_state={}, train_origins=500,
                  train_start='2022-01-01', train_end='2025-12-01', inference_seconds_per_fold=.01)
    (source / 'F1_metadata.json').write_text(json.dumps(report))
    reused = e.normalize_config({'reuse_run': 'baseline_s42', 'offset_scope': 'global'})
    dest = tmp_path / 'offset_global_s42'
    result = e.run_period(reused, 'unused', dest, 'F1', identity=identity)
    assert result['identity']['reused_from'] == 'baseline_s42'
    with np.load(dest / 'F1_raw_predictions.npz') as z:
        np.testing.assert_array_equal(z['raw'], raw)
        assert np.ptp(z['offset']) == 0
    with pytest.raises(ValueError, match='training configuration'):
        e.reuse_arrays(dict(reused, seed=43), dest, 'F1', identity)
    with pytest.raises(ValueError, match='data identity'):
        e.reuse_arrays(reused, dest, 'F1', {'data': {}})


def test_full_historical_matrix_remains_supported():
    source = Path(__file__).resolve().parents[1] / 'calib_offset_roll_ablation_20261104'
    assignment = json.loads((source / 'assignment.full-superseded.json').read_text())
    assert len(assignment['experiments']) == 162
    for recipe in assignment['experiments']:
        assert not e.normalize_config(recipe['config'])['spectral_hessian_only']


def test_hessian_only_preserves_hessian_and_removes_spectral_gradient():
    target = np.array([[400., 600.], [600., 400.]])
    prediction = target + np.array([[2., -3.], [-4., 5.]])
    control = e.normalize_config({'spectral_hessian_only': True})
    grad, hess = e.make_objective(target, control)(prediction, None)
    _, baseline_hess = e.make_objective(target, e.normalize_config({}))(prediction, None)
    off_grad, _ = e.make_objective(target, e.normalize_config({'spectral': False}))(prediction, None)
    np.testing.assert_array_equal(grad, off_grad)
    np.testing.assert_array_equal(hess, baseline_hess)
    assert [k for k in e.DEFAULTS if control[k] != e.DEFAULTS[k]] == ['spectral_hessian_only']
    with pytest.raises(ValueError, match='Hessian-only'):
        e.normalize_config({'spectral': False, 'spectral_hessian_only': True})
    with pytest.raises(ValueError, match='boolean'):
        e.normalize_config({'spectral_hessian_only': 1})


def test_legacy_replay_defaults_missing_hessian_flag(tmp_path):
    source = tmp_path / 'baseline_s42'
    models = source / 'F1_models'
    models.mkdir(parents=True)
    old_config = dict(e.DEFAULTS)
    old_config.pop('spectral_hessian_only')
    identity = {'data': {'path': 'fixed', 'bytes': 123, 'mtime_ns': 1}}
    (source / 'F1_metadata.json').write_text(json.dumps(dict(smoke=False, config=old_config, identity=identity)))
    np.savez(models / 'prior_predictions.npz', raw=np.ones((336, 72)), training_truth=np.ones((336, 72)), origins=np.arange(336))
    np.savez(source / 'F1_raw_predictions.npz', raw=np.ones((8, 72)), origins=np.arange(8))
    config = e.normalize_config({'reuse_run': 'baseline_s42', 'offset': False})
    assert e.reuse_arrays(config, tmp_path / 'replay', 'F1', identity)[0].shape == (336, 72)
    with pytest.raises(ValueError, match='training configuration'):
        e.reuse_arrays(dict(config, spectral_hessian_only=True), tmp_path / 'replay', 'F1', identity)


def test_locked_cli_rejects_name_and_config_changes(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(e, 'score_outputs', lambda out, obs: None)
    args = ['--data', 'unused', '--out', str(tmp_path), '--observations', 'unused', '--score-only']
    e.main(args, locked_experiment='baseline_s42', locked_config={'seed': 42})
    with pytest.raises(SystemExit):
        e.main(args + ['--experiment', 'other'], locked_experiment='baseline_s42', locked_config={'seed': 42})
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'seed': 43}))
    with pytest.raises(ValueError, match='locked experiment configuration'):
        e.main(args + ['--config', str(config)], locked_experiment='baseline_s42', locked_config={'seed': 42})
