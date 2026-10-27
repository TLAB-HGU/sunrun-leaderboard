import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from xgboost import XGBRegressor

from experiments.e2000_njobs_all.model import (
    LoggedMultiOutputRegressor, ModelConfig, PooledRegressor,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(90, 4)).astype(np.float32)
    Y = np.column_stack((400 + 40 * X[:, 0], 600 - 50 * X[:, 1], 800 + 30 * X[:, 2]))
    mask = np.ones(Y.shape, dtype=bool)
    mask[::4, 0] = False
    mask[1::4, 1] = False
    Y[~mask] = np.nan
    context = dict(code_hashes={'model': 'synthetic'}, data_hashes={'source': 'synthetic'},
                   feature_names=['a', 'b', 'c', 'd'], cutoff='2026-03-01T00:00:00Z',
                   grid={'train': 'synthetic', 'validation': 'synthetic'},
                   source_availability={'synthetic': True}, stage='internal')
    config = ModelConfig(horizons=3, n_estimators=12, learning_rate=.2, min_child_weight=1, n_jobs=1)
    return X[:60], Y[:60], mask[:60], (X[60:], Y[60:], mask[60:]), context, config


def _fit(model_class, data, path, **kwargs):
    X, Y, mask, val, context, config = data
    return model_class(config).fit(X, Y, observed_mask=mask, validation=val,
                                  output_dir=path, context=context, **kwargs)


def test_all_cpu_request_is_passed_to_each_xgboost(data, tmp_path):
    X, Y, mask, val, context, _ = data
    config = ModelConfig(horizons=3, n_estimators=2, min_child_weight=1)
    assert config.n_jobs == config.params()['n_jobs'] == -1
    model = LoggedMultiOutputRegressor(config).fit(
        X, Y, observed_mask=mask, validation=val, output_dir=tmp_path,
        context=context)
    assert model.n_jobs == 1  # independent horizon fits avoid nested full-CPU jobs
    assert all(e.get_params()['n_jobs'] == -1 and e.random_state == 42 for e in model.estimators_)
    assert model.predict(val[0]).shape == val[1].shape
    for invalid in (0, -2, True):
        with pytest.raises(ValueError):
            ModelConfig(n_jobs=invalid)


def test_independent_horizon_semantics_and_native_losses(data, tmp_path):
    X, Y, mask, val, context, config = data
    model = _fit(LoggedMultiOutputRegressor, data, tmp_path)
    pred = model.predict(val[0])
    assert pred.shape == val[1].shape
    assert len(model.estimators_) == 3
    assert all(e.random_state == 42 for e in model.estimators_)
    assert isinstance(clone(model), LoggedMultiOutputRegressor)
    for h, estimator in enumerate(model.estimators_):
        expected = XGBRegressor(**config.params()).fit(X[mask[:, h]], Y[mask[:, h], h])
        np.testing.assert_array_equal(pred[:, h], expected.predict(val[0]))
        log = pd.read_csv(tmp_path / f'loss_history_s42_h{h+1:02d}.csv')
        assert list(log) == ['step', 'train_mse', 'validation_mse']
        assert list(log.step) == list(range(1, config.n_estimators + 1))
        observed = val[2][:, h]
        mse = np.mean((pred[observed, h] - val[1][observed, h]) ** 2)
        assert log.validation_mse.iloc[-1] == pytest.approx(mse, rel=2e-6)
        assert np.isfinite(log.to_numpy()).all()
    # Absolute output must not receive an extra observed-speed offset.
    assert 300 < pred[:, 0].mean() < 500
    assert 700 < pred[:, 2].mean() < 900


def test_completed_resume_does_not_refit(data, tmp_path, monkeypatch):
    first = _fit(LoggedMultiOutputRegressor, data, tmp_path).predict(data[3][0])
    def forbidden(*args, **kwargs):
        raise AssertionError('completed checkpoint unexpectedly refit')
    monkeypatch.setattr(XGBRegressor, 'fit', forbidden)
    resumed = _fit(LoggedMultiOutputRegressor, data, tmp_path)
    np.testing.assert_array_equal(first, resumed.predict(data[3][0]))


@pytest.mark.parametrize('change', ['features', 'data', 'mask', 'stage', 'config', 'code'])
def test_identity_drift_rejected(data, tmp_path, change):
    _fit(LoggedMultiOutputRegressor, data, tmp_path)
    X, Y, mask, val, context, config = data
    X, mask, context = X.copy(), mask.copy(), dict(context)
    if change == 'features':
        context['feature_names'] = ['b', 'a', 'c', 'd']
    elif change == 'data':
        X[0, 0] += 1
    elif change == 'mask':
        mask[2, 2] = False
    elif change == 'stage':
        context['stage'] = 'official'
    elif change == 'code':
        context['code_hashes'] = {'model': 'changed'}
    else:
        config = replace(config, n_estimators=13)
    with pytest.raises(ValueError, match='identity'):
        _fit(LoggedMultiOutputRegressor, (X, Y, mask, val, context, config), tmp_path)


@pytest.mark.parametrize('artifact', ['model_s42_h01.checkpoint', 'loss_history_s42_h01.csv'])
def test_artifact_corruption_rejected(data, tmp_path, artifact):
    _fit(LoggedMultiOutputRegressor, data, tmp_path)
    with (tmp_path / artifact).open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        _fit(LoggedMultiOutputRegressor, data, tmp_path)


def test_partial_checkpoint_is_not_reused(data, tmp_path):
    (tmp_path / 'model_s42_h01.checkpoint').write_bytes(b'partial')
    model = _fit(LoggedMultiOutputRegressor, data, tmp_path)
    quarantined = list((tmp_path / 'quarantine').glob('s42_h01_*/model_s42_h01.checkpoint'))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b'partial'
    assert model.predict(data[3][0]).shape == data[3][1].shape


def test_interrupted_head_resumes_without_refitting_completed_heads(data, tmp_path, monkeypatch):
    from experiments.e2000_njobs_all import model as module
    original_atomic = module._atomic_text

    def interrupt_receipt(path, text):
        if path.name == 'receipt_s42_h02.json':
            raise InterruptedError('simulated interruption before receipt commit')
        original_atomic(path, text)

    monkeypatch.setattr(module, '_atomic_text', interrupt_receipt)
    with pytest.raises(InterruptedError):
        _fit(LoggedMultiOutputRegressor, data, tmp_path)
    first_receipt = (tmp_path / 'receipt_s42_h01.json').read_bytes()
    orphan_paths = [tmp_path / 'model_s42_h02.checkpoint', tmp_path / 'loss_history_s42_h02.csv']
    orphan_bytes = {path.name: path.read_bytes() for path in orphan_paths}
    monkeypatch.setattr(module, '_atomic_text', original_atomic)
    original_fit = XGBRegressor.fit
    fit_calls = []

    def counted_fit(self, *args, **kwargs):
        fit_calls.append(1)
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(XGBRegressor, 'fit', counted_fit)
    resumed = _fit(LoggedMultiOutputRegressor, data, tmp_path)
    assert len(fit_calls) == 2  # Interrupted h02 and never-started h03 only.
    assert (tmp_path / 'receipt_s42_h01.json').read_bytes() == first_receipt
    quarantine = list((tmp_path / 'quarantine').glob('s42_h02_*'))
    assert len(quarantine) == 1
    for name, payload in orphan_bytes.items():
        assert (quarantine[0] / name).read_bytes() == payload
    assert resumed.predict(data[3][0]).shape == data[3][1].shape


def test_pooled_exact_rows_and_prediction_order(data, tmp_path):
    X, Y, mask, val, context, config = data
    model = _fit(PooledRegressor, data, tmp_path)
    rows, horizons = np.nonzero(mask)
    flat_X = np.column_stack((X[rows], horizons + 1))
    expected = XGBRegressor(**config.params()).fit(flat_X, Y[rows, horizons])
    pred = model.predict(val[0])
    for h in range(config.horizons):
        direct = expected.predict(np.column_stack((val[0], np.full(len(val[0]), h+1))))
        np.testing.assert_array_equal(pred[:, h], direct)
    assert model.receipts_[0]['train_rows'] == mask.sum()
    np.testing.assert_array_equal(pred, _fit(PooledRegressor, data, tmp_path).predict(val[0]))
    loss = pd.read_csv(tmp_path / 'loss_history_s42_pooled.csv')
    assert loss.validation_mse.iloc[-1] == pytest.approx(
        np.mean((pred[val[2]] - val[1][val[2]])**2), rel=2e-6)


def test_final_validation_roles(data, tmp_path):
    X, Y, mask, val, context, config = data
    with pytest.raises(ValueError, match='only for official'):
        LoggedMultiOutputRegressor(config).fit(
            X, Y, observed_mask=mask, output_dir=tmp_path, context=context)
    context = {**context, 'stage': 'official'}
    model = LoggedMultiOutputRegressor(config).fit(
        X, Y, observed_mask=mask, validation=(X, Y, mask), output_dir=tmp_path / 'overlap',
        context=context, validation_role='overlapping_diagnostic')
    assert model.receipts_[0]['validation_role'] == 'overlapping_diagnostic'
    LoggedMultiOutputRegressor(config).fit(
        X, Y, observed_mask=mask, output_dir=tmp_path / 'none', context=context, validation_role='none')
    log = pd.read_csv(tmp_path / 'none/loss_history_s42_h01.csv')
    assert log.validation_mse.isna().all()
    receipt = json.loads((tmp_path / 'none/receipt_s42_h01.json').read_text())
    assert receipt['validation_rows'] == 0


def test_bad_masks_and_missing_horizon_fail(data, tmp_path):
    X, Y, mask, val, context, config = data
    with pytest.raises(ValueError, match='boolean'):
        LoggedMultiOutputRegressor(config).fit(X, Y, observed_mask=mask.astype(int),
            validation=val, output_dir=tmp_path, context=context)
    mask[:, 2] = False
    with pytest.raises(ValueError, match='every horizon'):
        LoggedMultiOutputRegressor(config).fit(X, Y, observed_mask=mask,
            validation=val, output_dir=tmp_path, context=context)


def test_full_72_output_contract(data, tmp_path):
    X, _, _, _, context, _ = data
    Y = 400 + X[:, :1] + np.arange(72)[None, :]
    config = ModelConfig(n_estimators=2, min_child_weight=1, n_jobs=1)
    model = LoggedMultiOutputRegressor(config).fit(
        X[:40], Y[:40], validation=(X[40:], Y[40:], None),
        output_dir=tmp_path, context=context)
    assert len(model.estimators_) == 72
    assert model.predict(X[:2]).shape == (2, 72)
    assert len(list(tmp_path.glob('loss_history_s42_h*.csv'))) == 72
    assert all(e.random_state == 42 for e in model.estimators_)
