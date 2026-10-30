import numpy as np
import pytest

from methods import (augment_design, extreme_weights, haar_transform24,
                     inverse_haar_transform24, shape_objective)


@pytest.mark.parametrize('shape,axis', [((24,), -1), ((4, 24, 9), 1), ((2, 3, 24), -1)])
def test_haar_roundtrip_and_energy(shape, axis):
    x = np.random.default_rng(3).normal(size=shape)
    encoded = haar_transform24(x, axis=axis)
    np.testing.assert_allclose(inverse_haar_transform24(encoded, axis=axis), x, atol=2e-15)
    np.testing.assert_allclose(np.sum(encoded**2), np.sum(x**2), rtol=1e-14)


def test_haar_order_and_constant_curve():
    x = np.arange(24.)
    encoded = haar_transform24(x)
    np.testing.assert_allclose(encoded[12:], (x[::2] - x[1::2]) / np.sqrt(2.))
    constant = haar_transform24(np.ones(24))
    np.testing.assert_allclose(constant[:3], np.sqrt(8.))
    np.testing.assert_array_equal(constant[3:], np.zeros(21))


@pytest.mark.parametrize('function', [haar_transform24, inverse_haar_transform24])
def test_haar_rejects_wrong_width(function):
    with pytest.raises(ValueError, match='24'):
        function(np.zeros((2, 23)))


def test_design_mapping_and_difference_only_control():
    c = np.arange(42.).reshape(2, 21)
    e = np.arange(2 * 24 * 9.).reshape(2, 24, 9)
    out = augment_design(c, e)
    assert out.shape == (2, 660)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[:, :21], c)
    np.testing.assert_array_equal(out[:, 21:237], e.reshape(2, 216))
    np.testing.assert_array_equal(out[:, 237:444], np.full((2, 207), 9.))
    np.testing.assert_allclose(out[:, 444:], haar_transform24(e, axis=1).reshape(2, 216), rtol=1e-6)
    np.testing.assert_array_equal(augment_design(c, e, include_wavelet=False), out[:, :444])


def test_design_future_poison_and_positive_control():
    # An already-available horizon-feature curve is not future observed truth.
    # The transform takes one origin and block only, never later origin rows.
    rng = np.random.default_rng(4)
    c, e = rng.normal(size=(5, 21)), rng.normal(size=(5, 72, 9))
    base = augment_design(c[:1], e[:1, :24])
    cp, ep = c.copy(), e.copy()
    cp[1:] = 1e8
    ep[1:] = -1e8
    ep[0, 24:] = 1e8
    np.testing.assert_array_equal(base, augment_design(cp[:1], ep[:1, :24]))
    ep[0, 12, 2] += 10.
    changed = augment_design(cp[:1], ep[:1, :24])
    assert not np.array_equal(base[:, 237:444], changed[:, 237:444])
    assert not np.array_equal(base[:, 444:], changed[:, 444:])


@pytest.mark.parametrize('factor', [2., 4.])
def test_extreme_weights_threshold_ramp_and_normalization(factor):
    y = np.full((4, 24), 400.)
    y[1] = 550.
    y[2, 6:] = 500.
    y[3, 6:] = 499.9
    weights = extreme_weights(y, factor)
    np.testing.assert_allclose(weights / weights[0], [1, factor, factor, 1])
    np.testing.assert_allclose(weights.mean(), 1.)
    np.testing.assert_array_equal(extreme_weights(y[1:3], factor), np.ones(2))


def test_extreme_weights_do_not_count_falls_or_cross_block_rises():
    y = np.full((2, 24), 400.)
    y[0, :6] = 540.
    y[1] = 520.
    np.testing.assert_array_equal(extreme_weights(y, 4), np.ones(2))


@pytest.mark.parametrize('lam', [0., .25, 1.])
def test_shape_gradient_and_hessian_majorizer(lam):
    rng = np.random.default_rng(5)
    truth, pred = rng.normal(size=(2, 24)), rng.normal(size=(2, 24))
    gradient, diagonal = shape_objective(lam)(truth.ravel(), pred)

    def loss(value):
        err = value - truth
        return .5 * np.sum(err**2) + .5 * lam * np.sum(np.diff(err, axis=1)**2)

    numeric = np.empty_like(pred)
    for index in np.ndindex(pred.shape):
        plus, minus = pred.copy(), pred.copy()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        numeric[index] = (loss(plus) - loss(minus)) / 2e-5
    np.testing.assert_allclose(gradient, numeric, atol=2e-9)
    difference = np.diff(np.eye(24), axis=0)
    exact_hessian = np.eye(24) + lam * difference.T @ difference
    assert np.linalg.eigvalsh(np.diag(diagonal[0]) - exact_hessian).min() > -1e-12
    assert np.all(diagonal > 0)
    if lam == 0:
        np.testing.assert_array_equal(gradient, pred - truth)
        np.testing.assert_array_equal(diagonal, np.ones_like(pred))


def test_shape_xgboost_zero_lambda_matches_builtin():
    import xgboost as xgb

    rng = np.random.default_rng(6)
    x = rng.normal(size=(40, 4)).astype(np.float32)
    y = (400 + x[:, :1] * np.arange(1, 25)[None, :]).astype(np.float32)
    kwargs = dict(n_estimators=3, max_depth=2, n_jobs=1, tree_method='hist',
                  multi_strategy='multi_output_tree', base_score=400., seed=42)
    standard = xgb.XGBRegressor(objective='reg:squarederror', **kwargs).fit(x, y)
    custom = xgb.XGBRegressor(objective=shape_objective(0.), **kwargs).fit(x, y)
    np.testing.assert_array_equal(standard.predict(x), custom.predict(x))


def test_reject_invalid_weights_and_lambda():
    with pytest.raises(ValueError, match='finite'):
        extreme_weights(np.full((1, 24), np.nan), 2)
    with pytest.raises(ValueError, match='factor'):
        extreme_weights(np.ones((1, 24)), 0)
    with pytest.raises(ValueError, match='lam'):
        shape_objective(-1)
