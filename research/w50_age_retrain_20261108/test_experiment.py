import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def experiment():
    spec = importlib.util.spec_from_file_location('age_training_test', Path(__file__).with_name('experiment.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_causal_age_keeps_leading_missing_and_advances_stale_frames(experiment, tmp_path):
    origins = pd.date_range('2024-01-01', periods=6, freq='h', tz='UTC')
    data = pd.DataFrame({'origin': origins, 'h': 1, 'img_age_h': [np.nan, .25, np.nan, np.nan, .5, np.nan]})
    data = pd.concat([data, data.iloc[[2]]], ignore_index=True)
    path = tmp_path / 'raw.parquet'
    data.to_parquet(path)
    actual = experiment.repaired_age(path)
    assert np.isnan(actual.iloc[0])
    np.testing.assert_array_equal(actual.iloc[1:], [.25, 1.25, 2.25, .5, 1.5])
    # Appending a later frame cannot retroactively change earlier repaired ages.
    later = pd.DataFrame({'origin': [origins[-1] + pd.Timedelta(hours=1)], 'h': 1, 'img_age_h': .1})
    pd.concat([data, later]).to_parquet(path)
    pd.testing.assert_series_equal(experiment.repaired_age(path).iloc[:6], actual)


def test_conflicting_duplicate_raw_age_fails(experiment, tmp_path):
    path = tmp_path / 'raw.parquet'
    pd.DataFrame({'origin': ['2024-01-01'] * 2, 'h': 1, 'img_age_h': [.1, .2]}).to_parquet(path)
    with pytest.raises(ValueError, match='Conflicting'):
        experiment.repaired_age(path)


def test_repair_changes_only_age_and_does_not_mutate_parent(experiment):
    names = {'constant': ['other', 'img_age_h'], 'varying': ['rec27']}
    origins = pd.date_range('2024-01-01', periods=3, freq='h', tz='UTC')
    grid = {'origins': origins, 'C': np.array([[7, .1], [8, .1], [9, .1]], np.float32),
                'E': np.arange(216, dtype=np.float32).reshape(3, 72, 1)}
    before = grid['C'].copy()
    changed, receipt = experiment.repair_grid(grid, names, pd.Series([.1, 1.1, 2.1], index=origins), training=False)
    np.testing.assert_array_equal(grid['C'], before)
    np.testing.assert_array_equal(changed['C'][:, 0], before[:, 0])
    np.testing.assert_array_equal(changed['E'], grid['E'])
    np.testing.assert_allclose(changed['C'][:, 1], [.1, 1.1, 2.1])
    assert receipt['changed_origins'] == 2
    assert receipt['unchanged_other_inputs']


def test_query_missing_source_and_evaluation_labels_rejected(experiment, monkeypatch):
    baseline, _ = experiment.parent_modules()
    origins = pd.date_range('2024-01-01', periods=1, tz='UTC')
    grid = {'origins': origins, 'C': np.zeros((1, 1)), 'E': np.zeros((1, 72, 1))}
    names = {'constant': ['img_age_h'], 'varying': ['rec27']}
    with pytest.raises(ValueError, match='Invalid causal'):
        experiment.repair_grid(grid, names, pd.Series([np.nan], index=origins), training=False)
    monkeypatch.setattr(baseline, 'load_grid', lambda *args: (grid, np.ones((1, 72))))
    experiment.bind_repaired_loader(baseline, pd.Series([1.], index=origins), [])
    with pytest.raises(ValueError, match='evaluation labels'):
        baseline.load_grid(None, 'F1', names, False)


@pytest.mark.parametrize('seed', [42, 43, 44])
def test_matched_parent_features_and_hyperparameters(experiment, seed):
    baseline, core = experiment.parent_modules()
    config = experiment.fixed_config(core, seed, f'age_retrain_s{seed}')
    names = experiment.assert_parent_contract(core, config, seed)
    assert len(names['constant']) == 20
    assert len(names['varying']) == 8
    assert len(core.expanded_names(names, 0, 24)) == 212
    assert 'reuse_run' not in config
    assert core.baseline is baseline


def test_fresh_paths_receive_repaired_train_calibration_and_query(experiment, monkeypatch, tmp_path):
    baseline, core = experiment.parent_modules()
    config = experiment.fixed_config(core, 42, 'age_retrain_s42')
    names = core.feature_names(config)
    col = names['constant'].index('img_age_h')
    train_origins = pd.date_range('2024-01-01', periods=1000, freq='6h', tz='UTC')
    query_origins = pd.date_range('2026-01-01', periods=3, freq='6h', tz='UTC')

    def grid(origins):
        return {'origins': origins, 'C': np.zeros((len(origins), 20), np.float32),
                    'E': np.zeros((len(origins), 72, 8), np.float32)}

    train, query = grid(train_origins), grid(query_origins)
    y = np.full((1000, 72), 400., np.float64)
    def loader(data, period, received_names, training):
        assert received_names == names
        return (train, y) if training else (query, None)
    monkeypatch.setattr(baseline, 'load_grid', loader)
    monkeypatch.setitem(baseline.QUERY_COUNTS, 'F1', 3)
    ages = pd.Series(np.arange(1003, dtype=np.float32), index=train_origins.append(query_origins))
    receipts, calls = [], []
    experiment.bind_repaired_loader(baseline, ages, receipts)

    def fake_fit(train_grid, query_grid, target, received_config, threads, out, period, stage, progress, smoke):
        assert not smoke
        assert received_config == config
        for item in (train_grid, query_grid):
            np.testing.assert_array_equal(item['C'][:, col], ages.reindex(item['origins']).to_numpy())
        np.testing.assert_array_equal(target, np.full(target.shape, 400.))
        calls.append((stage, train_grid['origins'], query_grid['origins']))
        return np.full((len(query_grid['origins']), 72), 390.), 0.

    monkeypatch.setattr(core, 'fit_predict', fake_fit)
    report = core.run_period(config, 'unused.parquet', tmp_path, 'F1', 1, False, {})
    assert [call[0] for call in calls] == ['prior', 'full']
    assert calls[0][1].max() + pd.Timedelta(hours=144) <= calls[0][2].min()
    assert calls[1][1].equals(train_origins)
    assert calls[1][2].equals(query_origins)
    assert [r['training'] for r in receipts] == [True, False]
    assert report['rounds'] == 900
    assert report['frozen_state']['offset_rows'] == 336
    np.testing.assert_allclose(report['frozen_state']['offset'], 2.5)
    # Loader repair never mutates the original source matrix.
    assert (train['C'] == 0).all()
