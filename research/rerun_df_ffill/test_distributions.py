"""Parity with pinned distribution sources and missing-target exclusion."""
import importlib.util
from pathlib import Path

import distributions as d
import numpy as np
import pytest
import xgboost as xgb


def original(experiment):
    path = Path(__file__).parent / "source_archive" / f"{experiment}.py"
    spec = importlib.util.spec_from_file_location(experiment, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("experiment", d.EXPERIMENTS[1:])
def test_feature_and_booster_parity(experiment):
    old = original(experiment)
    assert d.CONSTANT == [x.replace("ch_ch_", "suvi_") for x in old.CONSTANT]
    assert d.VARYING == old.VARYING
    assert d.CYCLES == old.CYCLES == 2
    assert d.GROUP_ROUNDS == old.GROUP_ROUNDS == 450
    assert d.K == old.K == 2
    for key, value in d.PARAMS.items():
        assert old.PARAMS[key] == value


@pytest.mark.parametrize("legacy", [False, True])
def test_finite_group_scales_objectives_and_posterior_match_original(legacy):
    old = original("e2000fs_calib9_lnmix" if legacy else "e2000fs_calib10_xgboostlss")
    rng = np.random.default_rng(17)
    y = np.exp(rng.normal(6., .2, (30, 3)))
    layout, old_layout = d.Layout(3, 2), old.Layout(3, 2)
    init, old_init = d.marginal_init(y), old.marginal_init(y, 2)
    for value, expected in zip(init, old_init):
        np.testing.assert_array_equal(value, expected)
    offsets = layout.to_offsets(*init)
    scales = d.group_scale(offsets, np.log(y), layout, legacy)
    expected_scales = old.group_scale(offsets, np.log(y), old_layout)
    np.testing.assert_allclose(list(scales.values()), list(expected_scales.values()), rtol=1e-14)
    theta = np.tile(offsets, (len(y), 1))
    matrix = xgb.DMatrix(rng.normal(size=(len(y), 2)), label=np.log(y).astype(np.float32), nthread=1)
    for group in layout.groups:
        base = theta[:, layout.index(*group)].copy()
        predicted = rng.normal(0, .02, y.shape)
        expected = old.group_block_objective(old_layout, group, base, theta.copy(), scales[group])(predicted, matrix)
        actual = d.group_objective(layout, group, base, theta.copy(), scales[group], np.isfinite(y))(predicted, matrix)
        for value, reference in zip(actual, expected):
            np.testing.assert_allclose(value, reference, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(d.mixture_mean(theta, layout)[0], old.mixture_mean(theta, old_layout)[0])


def test_missing_marginal_initialization_matches_each_observed_subset():
    rng = np.random.default_rng(7)
    y = np.exp(rng.normal(6., .2, (80, 3)))
    y[::3, 0] = np.nan
    y[1::4, 1] = np.nan
    p, m, s = d.marginal_init(y)
    for lead in range(3):
        expected = d.gaussian_mixture_em(np.log(y[np.isfinite(y[:, lead]), lead]), 2)
        for value, reference in zip((p[:, lead], m[:, lead], s[:, lead]), expected[:3]):
            np.testing.assert_array_equal(value, reference)


def test_missing_group_gradients_and_hessians_zero_and_finite_difference():
    rng = np.random.default_rng(71)
    y = np.exp(rng.normal(6., .2, (12, 3)))
    observed = np.ones(y.shape, bool)
    observed[0, 1] = False
    layout = d.Layout(3, 2)
    theta = np.tile(layout.to_offsets(*d.marginal_init(y)), (len(y), 1))
    z = np.log(y).astype(np.float32)
    matrix = xgb.DMatrix(rng.normal(size=(len(y), 2)), label=z, nthread=1)
    for group in layout.groups:
        cols = layout.index(*group)
        obj = d.group_objective(layout, group, theta[:, cols].copy(), theta.copy(), 1., observed)
        grad, hess = obj(np.zeros_like(y), matrix)
        assert grad[0, 1] == hess[0, 1] == 0
        eps = 1e-5
        upper, lower = theta.copy(), theta.copy()
        upper[:, cols] += eps
        lower[:, cols] -= eps
        numeric = (d.nll_and_responsibilities(upper, z, layout)[0] - d.nll_and_responsibilities(lower, z, layout)[0]) / (2 * eps)
        np.testing.assert_allclose(grad[observed], numeric[observed], rtol=2e-5, atol=2e-5)


def test_quantile_gradient_readout_matches_submitted_original_config():
    old = original("e2000fs_calib8_qxgb")
    rng = np.random.default_rng(22)
    horizons, n = 4, 11
    labels = rng.normal(size=(n, horizons))
    pred = rng.normal(size=(n, horizons * len(d.LEVELS)))
    matrix = xgb.DMatrix(rng.normal(size=(n, 3)), label=np.tile(labels, (1, len(d.LEVELS))), nthread=1)
    expected = old.objective(d.LEVELS, .5, "gaussian", 0, 1, horizons)(pred, matrix)
    actual = d.quantile_objective(np.ones(labels.shape, bool), horizons)(pred, matrix)
    for value, reference in zip(actual, expected):
        np.testing.assert_array_equal(value, reference)
    q = np.sort(pred.reshape(n, len(d.LEVELS), horizons).transpose(0, 2, 1), axis=2)
    np.testing.assert_array_equal(d.cdf_mean(q), old.cdf_mean(q, d.LEVELS))
    mask = np.ones(labels.shape, bool)
    mask[0, 2] = False
    grad, hess = d.quantile_objective(mask, horizons)(pred, matrix)
    assert (grad[0, 2::horizons] == 0).all() and (hess[0, 2::horizons] == 0).all()


@pytest.mark.parametrize("legacy", [True, False])
def test_complete_finite_training_matches_original(tmp_path, legacy):
    old = original("e2000fs_calib9_lnmix" if legacy else "e2000fs_calib10_xgboostlss")
    rng = np.random.default_rng(88)
    x, xq = rng.normal(size=(50, 4)).astype(np.float32), rng.normal(size=(3, 4)).astype(np.float32)
    y = np.exp(rng.normal(6, .15, (50, d.BLOCK)))
    expected = old.fit_cyclic(x, xq, y, old.Layout(d.BLOCK, 2), 2, 1, cycles=1)[0]
    pred, extras, info = d.fit_cyclic(x, xq, y, 2, 1, tmp_path, "test", legacy=legacy, cycles=1)
    np.testing.assert_allclose(extras["theta"], expected, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(pred, old.mixture_mean(expected, old.Layout(d.BLOCK, 2))[0], rtol=1e-7)
    assert len(list(tmp_path.glob("*.ubj"))) == 5
    assert len(info["history"]) == 5


@pytest.mark.parametrize("family", ["quantile", "mixture"])
def test_missing_target_training_persists_finite_models(tmp_path, family):
    rng = np.random.default_rng(81)
    x, xq = rng.normal(size=(60, 4)).astype(np.float32), rng.normal(size=(3, 4)).astype(np.float32)
    y = np.exp(rng.normal(6, .15, (60, d.BLOCK)))
    y[0] = np.nan
    y[1::5, ::2] = np.nan
    if family == "mixture":
        pred, _, info = d.fit_cyclic(x, xq, y, 2, 1, tmp_path, "missing", cycles=1)
    else:
        pred, _, info = d.fit_quantiles(x, xq, y, 2, 1, tmp_path, "missing")
    assert pred.shape == (3, d.BLOCK)
    assert np.isfinite(pred).all() and (pred > 0).all()
    assert info["dropped_no_target"] == 1
