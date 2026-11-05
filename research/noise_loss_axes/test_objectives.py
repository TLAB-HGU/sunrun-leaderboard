"""Independent density, finite-difference, and stability checks for NLL heads."""
import numpy as np
import pytest
from objective_reference import nll_and_grad, predict_mean
from scipy.special import expit
from scipy.stats import lognorm, norm


def example(family):
    if family == "gaussian":
        return np.array([[390., np.log(40.)], [520., np.log(90.)]]), np.array([420., 610.])
    if family == "lognormal":
        return np.array([[5.96, -2.], [6.25, -1.5]]), np.array([420., 610.])
    return np.array([[.5, 5.96, 6.25, -2., -1.5],
                     [-.7, 6.05, 6.4, -1.7, -1.9]]), np.array([420., 610.])


@pytest.mark.parametrize("family", ["gaussian", "lognormal", "mixture"])
def test_density_and_mean_against_scipy(family):
    theta, y = example(family)
    loss, _, hessian = nll_and_grad(theta, y, family)
    if family == "gaussian":
        expected = -norm.logpdf(y, loc=theta[:, 0], scale=np.exp(theta[:, 1]))
        mean = norm.mean(loc=theta[:, 0], scale=np.exp(theta[:, 1]))
    elif family == "lognormal":
        expected = -lognorm.logpdf(y, s=np.exp(theta[:, 1]), scale=np.exp(theta[:, 0]))
        mean = lognorm.mean(s=np.exp(theta[:, 1]), scale=np.exp(theta[:, 0]))
    else:
        pi = expit(theta[:, 0])
        distributions = [lognorm(s=np.exp(theta[:, 3+j]), scale=np.exp(theta[:, 1+j])) for j in range(2)]
        expected = -np.log(pi * distributions[0].pdf(y) + (1-pi) * distributions[1].pdf(y))
        mean = pi * distributions[0].mean() + (1-pi) * distributions[1].mean()
    np.testing.assert_allclose(loss, expected, rtol=1e-12)
    np.testing.assert_allclose(predict_mean(theta, family), mean, rtol=1e-12)
    assert (hessian > 0).all()


@pytest.mark.parametrize("family", ["gaussian", "lognormal", "mixture"])
def test_analytic_gradients_match_weighted_finite_difference(family):
    theta, y = example(family)
    weights = np.where(y >= 550., 4., 1.)
    _, analytic, _ = nll_and_grad(theta, y, family, weights=weights)
    eps = 1e-5
    for column in range(theta.shape[1]):
        upper, lower = theta.copy(), theta.copy()
        upper[:, column] += eps
        lower[:, column] -= eps
        numeric = (nll_and_grad(upper, y, family, weights=weights)[0]
                   - nll_and_grad(lower, y, family, weights=weights)[0]) / (2*eps)
        np.testing.assert_allclose(analytic[:, column], numeric, rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize("family", ["gaussian", "lognormal", "mixture"])
def test_missing_mask_and_zero_weight_are_exactly_excluded(family):
    theta, y = example(family)
    theta = np.tile(theta[0], (4, 1))
    y = np.array([y[0], np.nan, -10., np.inf])
    loss, grad, hessian = nll_and_grad(theta, y, family,
                                     weights=[4., 1., 0., 1.], mask=[True, True, True, False])
    reference = nll_and_grad(theta[0], y[0], family)
    for actual, expected in zip((loss, grad, hessian), reference):
        np.testing.assert_allclose(actual[0], expected * 4.)
        assert (actual[1:] == 0).all()


@pytest.mark.parametrize("family", ["gaussian", "lognormal", "mixture"])
def test_scale_floor_has_consistent_gradient(family):
    theta, y = example(family)
    columns = [1] if family != "mixture" else [3, 4]
    theta[:, columns] = -20.
    _, grad, hessian = nll_and_grad(theta, y, family, sigma_floor=.1)
    assert (grad[:, columns] == 0).all()
    assert (hessian > 0).all()
    upper = theta.copy()
    upper[:, columns] += .01
    np.testing.assert_array_equal(nll_and_grad(theta, y, family, sigma_floor=.1)[0],
                                  nll_and_grad(upper, y, family, sigma_floor=.1)[0])


def test_extreme_mixture_logits_and_label_switching():
    theta, y = example("mixture")
    switched = theta[:, [0, 2, 1, 4, 3]].copy()
    switched[:, 0] *= -1
    np.testing.assert_allclose(nll_and_grad(theta, y, "mixture")[0],
                               nll_and_grad(switched, y, "mixture")[0])
    np.testing.assert_allclose(predict_mean(theta, "mixture"), predict_mean(switched, "mixture"))
    theta[:, 0] = [-1000., 1000.]
    assert all(np.isfinite(a).all() for a in nll_and_grad(theta, y, "mixture"))
    assert np.isfinite(predict_mean(theta, "mixture")).all()


def test_readout_is_arithmetic_mean_and_gaussian_not_clipped():
    assert predict_mean(np.array([-2., 0.]), "gaussian") == -2.
    mu, sigma = 6., .3
    assert predict_mean(np.array([mu, np.log(sigma)]), "lognormal") > np.exp(mu)


def test_invalid_active_targets_and_weights_fail():
    theta, _ = example("lognormal")
    with pytest.raises(ValueError, match="positive"):
        nll_and_grad(theta, [0., 400.], "lognormal")
    with pytest.raises(ValueError, match="nonnegative"):
        nll_and_grad(theta, [400., 500.], "lognormal", weights=[1., -1.])
