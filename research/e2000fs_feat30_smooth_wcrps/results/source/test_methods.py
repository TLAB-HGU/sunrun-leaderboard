"""Numerical and actual LightGBM checkpoint regression checks."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('distribution_methods', Path(__file__).with_name('methods.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.mark.parametrize('method', m.METHODS)
def test_loss_gradients_match_finite_difference(method):
    y = np.array([285., 410., 610., 900.])
    mu = np.log([320., 440., 580., 760.])
    rho = np.log([.12, .18, .25, .3])
    objective = m.DistributionObjective(y, method)
    loss, grad, hess = objective.evaluate(mu, rho)
    assert np.isfinite(loss) and np.isfinite(grad).all() and (hess > 0).all()
    eps = 1e-6
    for coordinate in range(2):
        for i in range(len(y)):
            plus, minus = [mu.copy(), rho.copy()], [mu.copy(), rho.copy()]
            plus[coordinate][i] += eps
            minus[coordinate][i] -= eps
            finite = len(y) * (objective.evaluate(*plus, False) - objective.evaluate(*minus, False)) / (2 * eps)
            np.testing.assert_allclose(grad[i, coordinate], finite, rtol=2e-5, atol=2e-5)


def test_crps_quadrature_matches_analytic_and_refinement():
    y = np.array([250., 410., 550., 900., 1200.])
    mu = np.log([280., 430., 520., 850., 1000.])
    sigma = np.array([.05, .15, .3, .2, .4])
    exact = m.lognormal_crps(y, mu, sigma).mean()
    q64 = m.DistributionObjective(y, 'lognormal_crps').evaluate(mu, np.log(sigma), False)
    q128 = m.DistributionObjective(y, 'lognormal_crps', nodes=128).evaluate(mu, np.log(sigma), False)
    np.testing.assert_allclose(q64, exact, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(q64, q128, rtol=1e-5, atol=1e-4)


def test_weights_are_coordinate_based_and_scores_valid():
    y = np.array([300., 700.])
    objective = m.DistributionObjective(y, 'smooth_wcrps')
    plain = m.DistributionObjective(y, 'lognormal_crps')
    ratio = np.divide(objective.weights, plain.weights, out=np.ones_like(plain.weights), where=plain.weights != 0)
    np.testing.assert_allclose(ratio, 1. + m.ndtr((objective.z - 550) / 100))
    assert np.ptp(ratio[0]) > .9
    scores = m.distribution_scores(y, np.log(y), np.array([.1, .2]))
    assert all(np.isfinite(value) for value in scores.values())
    assert 0 <= scores['coverage90'] <= 1
    assert 0 <= scores['p_ge550_brier'] <= 1


@pytest.mark.parametrize('method', m.METHODS)
def test_actual_training_reload_offsets_and_tamper(tmp_path, method):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(120, 5))
    y = np.exp(6. + .15 * x[:, 0] + rng.normal(0, .08, size=len(x)))
    params = dict(n_estimators=4, learning_rate=.03, num_leaves=7, max_depth=3,
                  min_child_samples=5, min_child_weight=.01, n_jobs=1,
                  subsample=.8, subsample_freq=1, colsample_bytree=.9,
                  reg_alpha=.1, reg_lambda=.1, random_state=42)
    directory = tmp_path / method
    model = m.fit_distribution(x[:90], y[:90], x[90:], y[90:], params, method, directory)
    pred = m.predict_distribution(model, x[90:])
    assert (pred['sigma'] > 0).all() and np.isfinite(pred['mean']).all()
    restored = m.load_distribution(directory)
    for key, value in m.predict_distribution(restored, x[90:]).items():
        np.testing.assert_array_equal(pred[key], value)
    assert model['mu_offset'] > 5
    assert np.std(pred['mean']) > .01
    history = m.pd.read_csv(directory / 'loss.csv')
    assert len(history) == 4 and np.isfinite(history.to_numpy()).all()
    assert history.train_loss.iloc[-1] < history.train_loss.iloc[0]
    metadata = json.loads((directory / 'model.json').read_text())
    assert metadata['parameters']['num_leaves'] == params['num_leaves']
    assert metadata['parameters']['learning_rate'] == params['learning_rate']
    with (directory / 'mu.txt').open('a') as out:
        out.write('\nchanged\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        m.load_distribution(directory)


def test_invalid_parameters_fail_instead_of_clipping():
    with pytest.raises(FloatingPointError):
        m.DistributionObjective([400.], 'lognormal_nll').evaluate(np.array([6.]), np.array([3.]))
    with pytest.raises(ValueError):
        m.DistributionObjective([0.], 'smooth_wcrps')
