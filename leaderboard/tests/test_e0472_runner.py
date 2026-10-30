"""Regression checks for temporal boundaries and the E0472 runner contract."""
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from experiments.e0472_ablation_hpo.configs import DEFAULT_CONFIG, ENLIL_COLUMNS, resolve_window, select_columns, validate_config
from experiments.e0472_ablation_hpo.metrics import prediction_frame
from experiments.e0472_ablation_hpo.recipe import HOUR, fit_alpha, residual, source_guard, trailing, utc


def test_defaults_preserve_recipe_and_validate():
    cfg = validate_config({})
    assert cfg['xgb']['n_estimators'] == 900
    assert cfg['xgb']['learning_rate'] == .03
    assert cfg['xgb']['max_depth'] == 3
    assert cfg['k'] == 15 and cfg['standardize']
    cfg['xgb']['max_depth'] = 6
    assert DEFAULT_CONFIG['xgb']['max_depth'] == 3
    with pytest.raises(ValueError):
        validate_config({'xgb': {'device': 'cuda'}})
    with pytest.raises(ValueError):
        validate_config({'xgb': {'n_estimators': 10}})


def test_drop_prevents_enlil_derivative_reconstruction_and_speed_aliases():
    cols = ['h', 'v_lag0', 'v_m24', 'ace_speed_kms_last', 'ace_speed_kms_m6', 'speed_d24'] + list(ENLIL_COLUMNS)
    kept, removed = select_columns(cols, validate_config({'drop_columns': ['enlil_next_arr_h']}))
    assert set(removed) == {'enlil_next_arr_h', 'enlil_dt_target'}
    assert 'enlil_kp' in kept
    kept, _ = select_columns(cols, validate_config({'drop_groups': ['v_lag', 'v_roll']}))
    assert kept == ['h'] + list(ENLIL_COLUMNS)
    with pytest.raises(ValueError):
        select_columns(cols, validate_config({'drop_groups': ['plasma']}))


def test_smoke_and_search_cannot_open_confirm_or_official():
    for stage in ('smoke', 'search'):
        for window in ('W0', 'W1', 'W2', 'official'):
            with pytest.raises(ValueError):
                resolve_window(stage, window)
    _, (_, start, end) = resolve_window('smoke')
    origins = pd.date_range(utc(start), utc(end), freq='3h')
    assert len(origins) == 56
    assert origins.max() + 72*HOUR < utc('2026-01-01')


def test_availability_clock_does_not_shift_labels():
    times = pd.date_range('2025-01-01', periods=30, freq='h', tz='UTC')
    labels = pd.Series(np.arange(30.), index=times)
    available = labels.copy()
    available.index += 2*HOUR
    assert trailing(available, times[15:16])[0, -1] == 13
    assert labels.reindex(times[15:16]).iloc[0] == 15


def test_residual_future_pool_poison_and_positive_control():
    times = pd.date_range('2024-01-01', periods=3500, freq='h', tz='UTC')
    labels = pd.Series(400 + np.sin(np.arange(len(times))/80)*80, index=times)
    ace = labels.copy()
    ace.index += 2*HOUR
    ct = times[50:1900]
    cp = np.repeat(labels.reindex(ct).to_numpy()[:, None], 72, axis=1)
    origins = times[2700:2701]
    query = np.full((1, 72), 450.)
    hind = np.full((1, 72), 440.)
    cutoff = times[1800]
    cfg = validate_config({})
    original, scales = residual(origins, ace, labels, query, hind, ct, cp, cfg, cutoff)
    poisoned = labels.copy()
    poisoned.loc[poisoned.index + 72*HOUR > cutoff] += 1e6
    future, _ = residual(origins, ace, poisoned, query, hind, ct, cp, cfg, cutoff, scales)
    np.testing.assert_allclose(original, future)
    admitted = labels.copy()
    admitted.loc[admitted.index + 72*HOUR <= cutoff] += 100
    positive, _ = residual(origins, ace, admitted, query, hind, ct, cp, cfg, cutoff, scales)
    np.testing.assert_allclose(positive-original, 100)
    cfg['analog_weighting'] = 'distance'
    distance, _ = residual(origins, ace, labels, query, hind, ct, cp, cfg, cutoff, scales)
    assert np.isfinite(distance).all()


def test_alpha_is_fresh_grid_fit_and_missing_residual_falls_back():
    base = np.full((2, 72), 400.)
    dv = np.full((2, 72), 100.)
    labels = np.full((2, 72), 420.)
    alpha, _ = fit_alpha(base, dv, labels)
    assert alpha == pytest.approx(.2)
    assert fit_alpha(base, dv, base)[0] == 0
    assert fit_alpha(base, np.full_like(dv, np.nan), labels)[0] == 0


def test_source_guard_pushes_cap_and_restores_reader(tmp_path):
    path = tmp_path / 'data.parquet'
    times = pd.date_range('2025-01-01', periods=4, freq='h', tz='UTC')
    pd.DataFrame({'timestamp_utc': times, 'value': range(4)}).to_parquet(path)
    reader = pd.read_parquet
    access = {}
    with source_guard(times[1], access, allowed_paths=[path]):
        assert len(pd.read_parquet(path)) == 2
        with pytest.raises(ValueError):
            pd.read_parquet(tmp_path/'predictions_official.parquet')
    assert pd.read_parquet is reader
    assert len(access[str(path)]['sha256']) == 64


def test_prediction_grid_is_complete():
    times = pd.date_range('2025-01-01', periods=2, freq='3h', tz='UTC')
    pred = prediction_frame(times, np.full((2, 72), 400))
    assert len(pred) == 144 and not pred.duplicated(['origin_last_input_utc', 'horizon_hours']).any()
    with pytest.raises(ValueError):
        prediction_frame(times, np.full((2, 71), 400))


def test_runner_synthetic_end_to_end(tmp_path, monkeypatch):
    from experiments.e0472_ablation_hpo import runner, recipe, metrics
    times = pd.date_range('2022-01-01', '2025-12-31', freq='h', tz='UTC')
    class FakeFeatures:
        def __init__(self, cap):
            self.labels = self.observed = pd.Series(400., index=times)
            self.access = self.code_hashes = {}
        def poison(self, origin):
            return {'passed': True, 'positive_control': True}
    class FakeFit:
        def __init__(self, features, cfg, cutoff, seed, jobs):
            self.cols, self.dropped = ['h'], []
            self.train_receipt = {'rows': 42, 'latest_target': str(cutoff-72*HOUR)}
            self.seed = seed
        def paths(self, origins):
            return np.full((len(origins), 72), 400.+self.seed)
    monkeypatch.setattr(recipe, 'Features', FakeFeatures)
    monkeypatch.setattr(recipe, 'Fit', FakeFit)
    monkeypatch.setattr(metrics, 'preofficial_metrics', lambda pred, filled, labels: {'mse': float(((pred.pred_kms-400)**2).mean())})
    cfgpath = tmp_path/'config.json'
    cfgpath.write_text(json.dumps({'residual': False}))
    monkeypatch.setenv('EXPOS_OUT', str(tmp_path))
    args = SimpleNamespace(config=cfgpath, stage='smoke', seed=1, n_jobs=1, window=None)
    runner.run(args)
    result = json.loads((tmp_path/'metrics_CAL_s1.json').read_text())
    assert result['mse'] == 1
    assert json.loads((tmp_path/'poison.json').read_text())['passed']
    pred = pd.read_parquet(tmp_path/'predictions_CAL_s1.parquet')
    assert len(pred) == 56*72
