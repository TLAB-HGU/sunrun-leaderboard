"""Synthetic-only exact solver, artifact, and causal feature checks."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from test_mse20_adaptive_features import frames
from experiments.mse20_exact_ridge import ridge


def config(target='mr', days=None):
    return dict(next(c for c in ridge.candidate_grid() if c['target'] == target and c['history_days'] == days), max_train_origins=12)


def test_grid_and_solver_reference():
    assert len(ridge.candidate_grid()) == 8
    for c in ridge.candidate_grid():
        ridge.validate_config(c)
    rng = np.random.default_rng(42)
    x = rng.normal(size=(18, 1016)); y = rng.normal(size=(18, 72))
    w, b, diagnostics = ridge.solve_ridge(x, y, .01)
    xc, yc = x - x.mean(0), y - y.mean(0)
    # Independent dual-form reference instead of repeating the primal solve.
    expected = xc.T @ np.linalg.solve(xc @ xc.T + len(x) * .01 * np.eye(len(x)), yc)
    np.testing.assert_allclose(w, expected, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(b, y.mean(0) - x.mean(0) @ expected, atol=1e-10)
    assert max(diagnostics.values()) <= 1e-8
    for badx, bady in ((x[:, :-1], y), (x, y[:, :-1]), (x * np.nan, y), (x, y * np.nan)):
        with pytest.raises(ValueError):
            ridge.solve_ridge(badx, bady, .01)
    for changes in ({'alpha': 0}, {'max_train_origins': 6001}, {'max_train_origins': True}, {'seed': 1}):
        with pytest.raises(ValueError):
            ridge.validate_config(dict(config(), **changes))


@pytest.mark.parametrize('target', ['mr', 'persistence'])
def test_restore_subset_and_postcutoff_invariance(tmp_path, target):
    f = frames(); cutoff = f['speed'].index[850]; origins = f['speed'].index[[1000, 1001, 1002]]
    c = config(target)
    pred, meta = ridge.fit_predict(f, cutoff, origins, c, 'cpu', tmp_path / 'fit')
    restored = ridge.predict_saved(f, origins, tmp_path / 'fit', 'cpu')
    np.testing.assert_allclose(pred.pred_kms, restored.pred_kms, atol=1e-10, rtol=0)
    for i in range(3):
        subset = ridge.predict_saved(f, origins[i:i+1], tmp_path / 'fit', 'cpu')
        np.testing.assert_allclose(pred.pred_kms[i*72:(i+1)*72], subset.pred_kms, atol=1e-10, rtol=0)
    altered = copy.deepcopy(f)
    for key, value in altered.items():
        value.loc[value.index > cutoff] = False if key == 'observed' else 1800.
    _, other = ridge.fit_predict(altered, cutoff, origins, c, 'cpu', tmp_path / 'poison')
    assert meta['scaler'] == other['scaler']
    assert meta['train_origin_sha256'] == other['train_origin_sha256']
    for key in ('coef', 'intercept'):
        np.testing.assert_array_equal(np.load(tmp_path / 'fit/model.npz')[key], np.load(tmp_path / 'poison/model.npz')[key])
    assert pd.Timestamp(meta['train_label_max']) <= cutoff


def test_all_72_future_poison_and_drop(tmp_path):
    f = frames(); origin = f['speed'].index[[1000]]
    ridge.fit_predict(f, f['speed'].index[850], origin, config(), 'cpu', tmp_path)
    baseline = ridge.predict_saved(f, origin, tmp_path, 'cpu')
    for horizon in range(1, 73):
        timestamp = origin[0] + pd.Timedelta(hours=horizon)
        for drop in (False, True):
            changed = copy.deepcopy(f)
            for key, value in changed.items():
                assert timestamp in value.index
                if drop:
                    changed[key] = value.drop(timestamp)
                else:
                    value.loc[timestamp] = False if key == 'observed' else 1800.
            actual = ridge.predict_saved(changed, origin, tmp_path, 'cpu')
            np.testing.assert_allclose(actual.pred_kms, baseline.pred_kms, atol=1e-10, rtol=0)
    f['base']['future_regime'] = 1
    with pytest.raises(ValueError, match='schema'):
        ridge.predict_saved(f, origin, tmp_path, 'cpu')


def test_recent_origins_keep_full_lookback(tmp_path, monkeypatch):
    f = frames(19000); cutoff = f['speed'].index[-100]
    captured = []
    real = ridge.adaptive.inputs
    def checked(bounded, origins, target):
        if origins.max() < cutoff:
            assert bounded['speed'].index.min() == f['speed'].index.min()
            assert origins.min() >= cutoff - pd.Timedelta(days=730)
            raw = real(bounded, origins, target)
            assert np.isfinite(raw[2][:, :216]).all()
            captured.append(origins)
            return raw
        return real(bounded, origins, target)
    monkeypatch.setattr(ridge.adaptive, 'inputs', checked)
    _, meta = ridge.fit_predict(f, cutoff, f['speed'].index[[-90]], config(days=730), 'cpu', tmp_path)
    assert captured and len(captured[0]) == 12
    assert pd.Timestamp(meta['training_origins'][0]) < cutoff - pd.Timedelta(days=729)


@pytest.mark.parametrize('tamper', ['metadata', 'model', 'schema', 'cutoff', 'scaler', 'source', 'shape'])
def test_tamper_and_nooverwrite(tmp_path, tamper):
    f = frames(); origin = f['speed'].index[[1000]]; cutoff = f['speed'].index[850]
    ridge.fit_predict(f, cutoff, origin, config(), 'cpu', tmp_path)
    with pytest.raises(FileExistsError):
        ridge.fit_predict(f, cutoff, origin, config(), 'cpu', tmp_path)
    mpath = tmp_path / 'metadata.json'; meta = json.loads(mpath.read_text())
    if tamper == 'metadata':
        mpath.write_text(mpath.read_text() + ' ')
    elif tamper == 'model':
        with (tmp_path / 'model.npz').open('ab') as out:
            out.write(b'tamper')
    else:
        if tamper == 'schema': meta['schema']['history_hours'] = 999
        elif tamper == 'cutoff': meta['cutoff'] = f['speed'].index[700].isoformat()
        elif tamper == 'scaler': meta['scaler']['aux_scale'][0] = 0
        elif tamper == 'source': meta['source_sha256'] = '0' * 64
        elif tamper == 'shape':
            np.savez(tmp_path / 'model.npz', coef=np.zeros((1, 72)), intercept=np.zeros(72))
            meta['model_sha256'] = ridge.digest(tmp_path / 'model.npz')
        mpath.write_text(json.dumps(meta))
        (tmp_path / 'metadata.sha256').write_text(ridge.digest(mpath))
    with pytest.raises(ValueError):
        ridge.predict_saved(f, origin, tmp_path, 'cpu')


def test_matrix_and_failed_solver_rejected(monkeypatch):
    with pytest.raises(ValueError, match='dimensions'):
        ridge._matrix((np.zeros((2, 2, 168)), np.zeros((2, 679)), np.zeros((2, 72))))
    with pytest.raises(ValueError, match='nonfinite'):
        ridge._matrix((np.full((2, 2, 168), np.nan), np.zeros((2, 680)), np.zeros((2, 72))))
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(6, 1016)), rng.normal(size=(6, 72))
    monkeypatch.setattr(np.linalg, 'solve', lambda a, b: np.zeros_like(b))
    with pytest.raises(ValueError, match='diagnostics'):
        ridge.solve_ridge(x, y, .01)
