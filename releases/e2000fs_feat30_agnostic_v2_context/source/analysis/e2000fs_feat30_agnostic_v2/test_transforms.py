import numpy as np
import pytest

from transforms import (LOWER, UPPER, _curve_jacobian, event_curve,
                        fit_event_parameters, haar72, inverse_haar72, parameters)


def test_haar_roundtrip_energy_and_order():
    rng = np.random.default_rng(42)
    y = rng.normal(size=(5, 72))
    original = y.copy()
    encoded = haar72(y)
    np.testing.assert_allclose(inverse_haar72(encoded), y, atol=2e-15)
    np.testing.assert_allclose(np.sum(encoded**2, axis=1), np.sum(y**2, axis=1))
    np.testing.assert_allclose(encoded[:, 36:], (y[:, ::2] - y[:, 1::2]) / np.sqrt(2.))
    np.testing.assert_array_equal(y, original)


def test_haar_constant():
    encoded = haar72(np.full((2, 72), 400.))
    np.testing.assert_allclose(encoded[:, :9], 400. * np.sqrt(8.))
    np.testing.assert_array_equal(encoded[:, 9:], np.zeros((2, 63)))


def test_event_curve_formula_and_clipped_bounds():
    p = np.array([[400., 200., 12., 6., 100., 12., 6.]])
    curve = event_curve(p)[0]
    np.testing.assert_allclose(curve[[11, 14, 17, 29, 32, 35]], [400, 500, 600, 600, 550, 500])
    bad = np.array([[-100., 2000., -10., -3., 2000., -5., 0.],
                    [4000., -5., 90., 90., -2., 100., 90.]])
    before = bad.copy()
    out = event_curve(bad)
    np.testing.assert_array_equal(out, event_curve(np.clip(bad, LOWER, UPPER)))
    assert np.isfinite(out).all() and out.min() >= 1 and out.max() <= 3000
    np.testing.assert_array_equal(bad, before)


def test_event_jacobian_away_from_knots():
    p = np.array([400., 180., 12.3, 8.2, 90., 11.8, 9.4])
    curve, jac = _curve_jacobian(p)
    np.testing.assert_allclose(curve, event_curve(p[None])[0])
    numeric = np.empty((72, 7))
    for j in range(7):
        plus, minus = p.copy(), p.copy()
        plus[j] += 1e-5
        minus[j] -= 1e-5
        numeric[:, j] = (event_curve(plus[None])[0] - event_curve(minus[None])[0]) / 2e-5
    np.testing.assert_allclose(jac, numeric, atol=1e-8)


def test_fit_rise_and_decay_reconstructs_fixture_and_preserves_labels():
    expected = np.array([[350., 250., 18., 12., 180., 12., 18.]])
    y = event_curve(expected)
    original = y.copy()
    fit = fit_event_parameters(y)
    assert fit['params'].shape == (1, 7)
    assert fit['reconstruction_rmse'][0] < .1
    assert fit['success'][0]
    assert not fit['no_rise'][0]
    np.testing.assert_allclose(event_curve(fit['params']), y, atol=.3)
    np.testing.assert_array_equal(y, original)
    np.testing.assert_array_equal(parameters(y), fit['params'])


def test_no_rise_constant_and_decay():
    p = np.array([[520., 0., 0., 1., 180., 0., 36.]])
    y = np.vstack([np.full(72, 450.), event_curve(p)[0]])
    fit = fit_event_parameters(y)
    assert fit['no_rise'].all()
    np.testing.assert_array_equal(fit['params'][:, 1], np.zeros(2))
    assert np.max(fit['reconstruction_rmse']) < .1
    assert (fit['params'] >= LOWER).all() and (fit['params'] <= UPPER).all()
    assert (np.diff(event_curve(fit['params']), axis=1) <= 1e-10).all()


def test_multiple_rises_diagnostic_and_reproducibility():
    y = np.full((1, 72), 350.)
    y[:, 12:24] = 550.
    y[:, 36:48] = 600.
    a, b = fit_event_parameters(y, max_nfev=12), fit_event_parameters(y, max_nfev=12)
    assert a['multiple_rises'][0] == 2
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_empty_batches():
    empty = np.empty((0, 72))
    assert haar72(empty).shape == inverse_haar72(empty).shape == (0, 72)
    assert event_curve(np.empty((0, 7))).shape == (0, 72)
    result = fit_event_parameters(empty)
    assert result['params'].shape == (0, 7)
    for key in ['reconstruction_rmse', 'success', 'no_rise', 'multiple_rises']:
        assert result[key].shape == (0,)


@pytest.mark.parametrize('function,width', [(haar72, 72), (inverse_haar72, 72),
                                            (fit_event_parameters, 72), (event_curve, 7)])
def test_invalid_shape_or_nonfinite_rejected(function, width):
    with pytest.raises(ValueError, match='shape'):
        function(np.zeros(width))
    with pytest.raises(ValueError, match='finite'):
        function(np.full((1, width), np.nan))


def test_invalid_optimizer_budget():
    with pytest.raises(ValueError, match='max_nfev'):
        fit_event_parameters(np.zeros((1, 72)), max_nfev=0)
