"""Regression tests for the transfer-gated window protocol and the read-outs."""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

import hist
import run


@pytest.fixture(scope='module')
def data():
    _, bundle = run.load_data()
    return bundle


def test_windows_are_the_protocol_windows():
    assert list(run.WINDOWS) == ['D1', 'D2', 'D3', 'C']
    assert run.WINDOWS['D1'] == ('2024-12-04', '2025-02-22')
    assert run.WINDOWS['C'] == ('2025-12-04', '2026-02-22')
    assert run.DEV == ('D1', 'D2', 'D3')


def test_maturity_boundary_is_strict(data):
    origins = data['train']['origins']
    for name in run.WINDOWS:
        fit, evaluate = run.window_split(origins, name)
        start = pd.Timestamp(run.WINDOWS[name][0], tz='UTC')
        mature = origins[fit] + pd.Timedelta(hours=run.HORIZON)
        assert (mature < start).all()
        # one grid step later would reach into the window, so the boundary is tight
        later = origins[fit].max() + pd.Timedelta(hours=3)
        assert later + pd.Timedelta(hours=run.HORIZON) >= start
        assert origins[evaluate].min() >= start
        assert not np.intersect1d(fit, evaluate).size
        assert origins[fit].max() < origins[evaluate].min()


def test_cutoff_splits_are_mature(data):
    origins = data['train']['origins']
    for stage, cutoff in run.CUTOFF.items():
        fit = run.cutoff_split(origins, stage)
        assert (origins[fit] + pd.Timedelta(hours=run.HORIZON) < pd.Timestamp(cutoff, tz='UTC')).all()
        assert len(fit)


def test_future_poison_does_not_reach_the_fit_rows(data):
    """Poisoning every target that overlaps a window must not change any fit label."""
    origins = data['train']['origins']
    clean = data['train']['Y']
    for name in run.WINDOWS:
        fit, _ = run.window_split(origins, name)
        start = pd.Timestamp(run.WINDOWS[name][0], tz='UTC')
        stamps = origins.values[:, None] + np.arange(1, run.HORIZON + 1).astype('timedelta64[h]')
        exposed = pd.DatetimeIndex(stamps.ravel(), tz='UTC') >= start
        poisoned = clean.copy()
        poisoned.reshape(-1)[exposed] = 1e6
        np.testing.assert_array_equal(poisoned[fit], clean[fit])
        assert (poisoned[fit] != 1e6).all()


def test_design_is_the_parent_237_column_block(data):
    bundle = data['train']
    for block in range(3):
        X = run.design(bundle, block)
        assert X.shape == (len(bundle['origins']), 237)
        np.testing.assert_array_equal(X[:, :21], bundle['C'])
        np.testing.assert_array_equal(X[:, 21:237], bundle['E'][:, block * 24:(block + 1) * 24].reshape(len(X), -1))


def test_band_metrics_match_a_hand_computed_example():
    truth = np.full((2, run.HORIZON), 500.)
    truth[1, :] = 600.
    pred = truth + 10.
    metrics = run.band_metrics(truth, pred)
    assert metrics['mse_lt550'] == pytest.approx(100.)
    assert metrics['mse_ge550'] == pytest.approx(100.)
    assert metrics['n_pairs_lt550'] == run.HORIZON
    assert metrics['n_pairs_ge550'] == run.HORIZON
    assert metrics['mse'] == pytest.approx(100.)
    assert run.reweighted(metrics, .25) == pytest.approx(100.)


def test_band_metrics_separate_the_bands():
    truth = np.full((1, run.HORIZON), 300.)
    truth[0, -1] = 900.
    pred = truth.copy()
    pred[0, 0] = 310.
    pred[0, -1] = 800.
    metrics = run.band_metrics(truth, pred)
    assert metrics['mse_ge550'] == pytest.approx(10000.)
    assert metrics['mse_lt550'] == pytest.approx(100. / 71.)
    assert metrics['bias_ge550'] == pytest.approx(-100.)


def test_band_metrics_reject_degenerate_input():
    with pytest.raises(ValueError):
        run.band_metrics(np.full((1, 72), 300.), np.full((1, 72), 300.))
    with pytest.raises(ValueError):
        run.band_metrics(np.full((1, 72), 300.), np.full((2, 72), 300.))


def test_read_out_variants_share_one_training_run():
    assert run.ARMS['hist_weighted_raw'][0] == run.ARMS['hist_weighted_corrected'][0] == 'hist_weighted'
    assert run.ARMS['hist_weighted_corrected'][1] == 'corrected'
    assert run.ARMS['hist_weighted_raw'][1] is None
    assert set(run.ARMS) == {'parent', 'reference', 'hist_unweighted',
                             'hist_weighted_raw', 'hist_weighted_corrected'}


def test_hessian_conversion_keeps_regularisation_scale():
    params, h_ref = run.hist_params()
    k = len(hist.EDGES) - 1
    assert params['num_class'] == k
    assert h_ref == pytest.approx((1. / k) * (1. - 1. / k))
    assert params['reg_lambda'] == pytest.approx(5. * h_ref)
    assert params['min_child_weight'] == pytest.approx(20. * h_ref)
    assert params['max_depth'] == 3 and params['learning_rate'] == 0.03


def test_histogram_head_learns_and_reloads(tmp_path):
    """Smoke: a tiny head fits, predicts absolute speeds and reloads identically."""
    rng = np.random.default_rng(0)
    n = 480
    X = rng.normal(size=(n, 12)).astype(np.float32)
    center = 450. + 60. * X[:, 0]
    Y = center[:, None] + 25. * rng.normal(size=(n, run.BLOCK))
    fit = np.arange(n // 2)
    evaluate = np.arange(n // 2, n)
    result = run.fit_histogram(tmp_path, X, Y, fit, evaluate, factor=1.0, rounds=40)
    assert result['raw'].shape == (len(evaluate), run.BLOCK)
    assert np.isfinite(result['raw']).all()
    # absolute target round-trip: a direct speed read-out, never a delta
    assert np.abs(result['raw'].mean() - Y[evaluate].mean()) < 100.
    assert (result['raw'] > 240.).all() and (result['raw'] < 1150.).all()
    assert result['reload_max_abs_delta'] < 1e-6
    assert list(result['curves']) == ['train_ce', 'valid_ce', 'train_rmse', 'valid_rmse']
    assert result['curves']['train_ce'].shape == (run.BLOCK, 40)
    assert np.isfinite(result['curves']['train_ce'][:, -1]).all()
    reloaded = xgb.Booster(model_file=str(tmp_path / 'h00.ubj'))
    margins = reloaded.predict(xgb.DMatrix(X[evaluate], nthread=1), output_margin=True)
    assert margins.shape == (len(evaluate), len(hist.EDGES) - 1)


def test_corrected_read_out_differs_from_raw_only_under_weighting(tmp_path):
    rng = np.random.default_rng(1)
    n = 480
    X = rng.normal(size=(n, 10)).astype(np.float32)
    Y = (450. + 60. * X[:, 0])[:, None] + 30. * rng.normal(size=(n, run.BLOCK))
    fit, evaluate = np.arange(n // 2), np.arange(n // 2, n)
    plain = run.fit_histogram(tmp_path / 'a', X, Y, fit, evaluate, factor=1.0, rounds=30)
    np.testing.assert_allclose(plain['raw'], plain['corrected'])
    weighted = run.fit_histogram(tmp_path / 'b', X, Y, fit, evaluate, factor=4.0, rounds=30)
    assert not np.allclose(weighted['raw'], weighted['corrected'])
    assert weighted['corrected'].mean() < weighted['raw'].mean()


def test_window_p_train_uses_training_targets_only(data):
    origins = data['train']['origins']
    for name in run.WINDOWS:
        fit, evaluate = run.window_split(origins, name)
        p = run.window_p_train(data['train']['Y'], fit)
        assert 0. < p < 0.5
        assert p == pytest.approx(float((data['train']['Y'][fit] >= 550.).mean()))


def test_manifest_round_trip_allows_resume(tmp_path):
    """The resume guard must accept the JSON round-trip of its own manifest."""
    manifest = dict(identity='abc', windows=dict(D1=('2024-12-04', '2025-02-22')),
                    bins=[240., 290.], rounds=300, smoke=False)
    run.ensure_identity(tmp_path, manifest)
    run.ensure_identity(tmp_path, manifest)  # would raise before the 2026-10-28 fix
    changed = dict(manifest, rounds=299)
    with pytest.raises(RuntimeError):
        run.ensure_identity(tmp_path, changed)
