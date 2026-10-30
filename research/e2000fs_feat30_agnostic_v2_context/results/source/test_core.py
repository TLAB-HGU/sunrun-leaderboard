"""Integration contracts for frozen data and native XGBoost fitting."""

import json

import numpy as np
import pandas as pd
import pytest

import core as C
from context import context_objective


@pytest.fixture
def chronological_data():
    origins = pd.date_range('2025-01-01', '2026-05-30', freq='h', tz='UTC')
    return {'train': {'origins': origins}}


@pytest.mark.parametrize('stage,cutoff,lo,hi', C.FOLDS)
@pytest.mark.parametrize('smoke', [False, True])
def test_three_chronological_splits_respect_target_maturity(chronological_data, stage, cutoff, lo, hi, smoke):
    fit, valid = C.split(chronological_data, stage, smoke=smoke)
    origins = chronological_data['train']['origins']
    cutoff = pd.Timestamp(cutoff, tz='UTC')
    assert len(fit) and len(valid)
    assert not np.intersect1d(fit, valid).size
    assert (origins[fit] + pd.Timedelta(hours=72) < cutoff).all()
    assert origins[fit].max() + pd.Timedelta(hours=72) < origins[valid].min()
    assert origins[valid].min() >= pd.Timestamp(lo, tz='UTC')
    assert origins[valid].max() < pd.Timestamp(hi, tz='UTC') + pd.Timedelta(days=1)
    assert np.all(np.diff(fit) > 0) and np.all(np.diff(valid) > 0)
    if smoke:
        assert len(fit) <= 64 and len(valid) <= 16
    else:
        # A target ending exactly at the exclusive cutoff is not mature.
        boundary = origins.get_loc(cutoff - pd.Timedelta(hours=72))
        assert boundary not in fit and boundary - 1 in fit


@pytest.mark.parametrize('stage', ['f1', 'final'])
def test_final_training_splits_keep_exclusive_72h_cutoff(chronological_data, stage):
    fit, valid = C.split(chronological_data, stage)
    origins = chronological_data['train']['origins']
    cutoff = pd.Timestamp(C.CUTOFFS[stage], tz='UTC')
    assert (origins[fit] + pd.Timedelta(hours=72) < cutoff).all()
    np.testing.assert_array_equal(valid, C.split(chronological_data, 'fold2')[1])


def make_bundle(n=4):
    return {'C': np.arange(n * 21, dtype=np.float32).reshape(n, 21),
            'E': np.arange(n * 72 * 9, dtype=np.float32).reshape(n, 72, 9),
            'Y': np.arange(n * 72, dtype=np.float32).reshape(n, 72),
            'origins': pd.date_range('2025-01-01', periods=n, freq='h', tz='UTC')}


def test_design_669_and_237_mapping_and_take_order():
    bundle = make_bundle()
    full = C.design(bundle)
    assert full.shape == (4, 669) and full.dtype == np.float32
    np.testing.assert_array_equal(full[:, :21], bundle['C'])
    np.testing.assert_array_equal(full[:, 21:], bundle['E'].reshape(4, 648))
    for block in range(3):
        design = C.design(bundle, block)
        assert design.shape == (4, 237)
        np.testing.assert_array_equal(design[:, :21], bundle['C'])
        np.testing.assert_array_equal(design[:, 21:], bundle['E'][:, block * 24:(block + 1) * 24].reshape(4, 216))
    chosen = C.take(bundle, np.array([3, 1]))
    np.testing.assert_array_equal(C.design(chosen), full[[3, 1]])
    np.testing.assert_array_equal(chosen['Y'], bundle['Y'][[3, 1]])


def test_design_target_and_later_origin_poison_with_positive_control():
    bundle = make_bundle()
    baseline = C.design(bundle)
    poisoned = {k: v.copy() for k, v in bundle.items()}
    poisoned['Y'][:] = np.nan
    np.testing.assert_array_equal(C.design(poisoned), baseline)
    poisoned['C'][1:] = -1e6
    poisoned['E'][1:] = 1e6
    np.testing.assert_array_equal(C.design(poisoned)[0], baseline[0])
    # Positive control changes an available same-origin feature and must act.
    poisoned['E'][0, 30, 4] += 300.
    assert not np.array_equal(C.design(poisoned)[0], baseline[0])
    np.testing.assert_array_equal(C.design(poisoned, 0)[0], C.design(bundle, 0)[0])
    assert not np.array_equal(C.design(poisoned, 1)[0], C.design(bundle, 1)[0])
    np.testing.assert_array_equal(C.design(bundle), baseline)


@pytest.mark.parametrize('change', [{'smoke': True}, {'identity': 'different'}, {'hashes': {'source': 'different'}}])
def test_freeze_rejects_changed_mode_identity_or_hash(tmp_path, change):
    directory = tmp_path / 'frozen'
    original = {'identity': 'fixed', 'smoke': False, 'hashes': {'source': 'abc'}, 'method': 'event'}
    C.freeze(directory, original)
    C.freeze(directory, original)
    with pytest.raises(RuntimeError, match='changed code/data/config/mode'):
        C.freeze(directory, original | change)
    assert json.loads((directory / 'manifest.json').read_text()) == original


def tiny_training():
    rng = np.random.default_rng(47)
    x = rng.normal(size=(12, 237)).astype(np.float32)
    y = (400. + x[:, :1] * np.linspace(5., 35., 24)[None, :]).astype(np.float32)
    return x, y


def test_native_scalar_quantiles_five_outputs_and_checkpoint_reload(tmp_path):
    x, y = tiny_training()
    folder = tmp_path / 'quantile'
    overrides = dict(n_estimators=2, min_child_weight=1.,
                     multi_strategy='one_output_per_tree', objective='reg:quantileerror',
                     quantile_alpha=np.array([.1, .3, .5, .7, .9]), eval_metric='quantile')
    model = C.fit_model(folder, x, y[:, 0], x[:4], y[:4, 0], smoke=True, overrides=overrides)
    pred = C.prediction(model, x[:4])
    assert pred.shape == (4, 5) and np.isfinite(pred).all()
    loaded = C.fit_model(folder, x, y[:, 0], x[:4], y[:4, 0], smoke=True, overrides=overrides)
    np.testing.assert_array_equal(C.prediction(loaded, x[:4]), pred)
    history = pd.read_csv(folder / 'loss.csv')
    assert len(history) == 2
    assert {'train_quantile', 'validation_quantile'} <= set(history)
    receipt = json.loads((folder / 'receipt.json').read_text())
    assert receipt['reload_parity'] and receipt['native_loss_history']
    assert receipt['rows'] == 12 and receipt['targets'] == 1
    assert receipt['sha256'] == C.digest(folder / 'model.ubj')


def test_context_zero_strength_matches_native_squared_error_fit(tmp_path):
    x, y = tiny_training()
    common = {'n_estimators': 2, 'min_child_weight': 1.}
    control = C.fit_model(tmp_path / 'control', x, y, x[:4], y[:4], smoke=True, overrides=common)
    contextual = C.fit_model(tmp_path / 'context0', x, y, x[:4], y[:4], smoke=True,
                             overrides=common | {'objective': context_objective(np.linspace(0., 1., len(y)), 0.)})
    np.testing.assert_array_equal(C.prediction(control, x), C.prediction(contextual, x))
    control_config = json.loads(control.save_config())['learner']['learner_model_param']
    contextual_config = json.loads(contextual.save_config())['learner']['learner_model_param']
    assert control_config['base_score'] == contextual_config['base_score']
    assert C.prediction(contextual, x).shape == (12, 24)


def test_fit_model_rejects_corrupt_checkpoint(tmp_path):
    x, y = tiny_training()
    folder = tmp_path / 'corrupt'
    C.fit_model(folder, x, y, x[:4], y[:4], smoke=True, overrides={'n_estimators': 2})
    with (folder / 'model.ubj').open('ab') as f:
        f.write(b'corrupted')
    with pytest.raises(RuntimeError, match='Corrupt checkpoint'):
        C.fit_model(folder, x, y, x[:4], y[:4], smoke=True)
