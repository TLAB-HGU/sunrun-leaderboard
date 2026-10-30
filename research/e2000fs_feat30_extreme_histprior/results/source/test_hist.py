"""Unit tests for the HL-Gauss histogram head and its target-shift correction."""

import numpy as np
import pytest

import hist


def test_threshold_is_an_exact_bin_edge():
    assert 550. in hist.EDGES
    fast = hist.EDGES[hist.FAST_BIN]
    assert fast == 550.
    assert (hist.EDGES[hist.FAST_BIN:-1] >= 550.).all()
    assert (hist.EDGES[1:hist.FAST_BIN + 1] <= 550.).all()
    assert np.all(np.diff(hist.EDGES) > 0)


def test_bin_index_clips_outside_support():
    assert hist.bin_index([100.])[0] == 0
    assert hist.bin_index([5000.])[0] == len(hist.EDGES) - 2
    assert hist.bin_index([550.])[0] == hist.FAST_BIN
    assert hist.bin_index([549.999])[0] == hist.FAST_BIN - 1
    with pytest.raises(ValueError):
        hist.bin_index([np.nan])


@pytest.mark.parametrize('value', [249.4, 300., 424.6, 549.9, 550.1, 700., 1087.2])
def test_soft_labels_form_a_distribution(value):
    q = hist.soft_labels(np.array([value, value + 5.]))
    assert q.shape == (2, len(hist.EDGES) - 1)
    assert (q >= 0).all()
    np.testing.assert_allclose(q.sum(axis=1), 1., atol=1e-6)
    assert q.dtype == np.float32


def test_soft_labels_are_local():
    value = 430.
    q = hist.soft_labels(np.array([value]))[0]
    index = hist.bin_index(np.array([value]))[0]
    assert q[index] > 0.5
    near = q[max(index - 2, 0):index + 3].sum()
    assert near > 0.99
    assert np.flatnonzero(q > 1e-3).max() - np.flatnonzero(q > 1e-3).min() <= 5


def test_soft_labels_handle_out_of_support_values():
    low = hist.soft_labels(np.array([0.]))[0]
    high = hist.soft_labels(np.array([5000.]))[0]
    assert low[0] > 0.95 and low[1:].sum() < 0.05
    assert high[-1] > 0.95 and high[:-1].sum() < 0.05
    np.testing.assert_allclose([low.sum(), high.sum()], [1., 1.], atol=1e-9)


def test_pointwise_weight_is_a_label_function_only():
    values = np.array([300., 520., 549.9, 550., 900.])
    weight = hist.point_weights(values, 4.)
    np.testing.assert_allclose(weight, np.array([1., 1., 1., 4., 4.]) / 2.2)
    assert hist.point_weights(values, 1.) is None
    # every value inside one bin gets the same weight, so the weight is constant
    # on the support of every class and the posterior correction is exact
    grid = np.linspace(240.1, 1149.9, 4000)
    weight = hist.point_weights(grid, 4.)
    per_bin = [np.unique(weight[hist.bin_index(grid) == k]) for k in range(len(hist.EDGES) - 1)]
    assert all(len(values) == 1 for values in per_bin)
    np.testing.assert_allclose(weight.mean(), 1., rtol=1e-12)


def test_correction_inverts_a_synthetic_target_shift():
    """Exact recovery of p(y|x) from a weight-corrupted class posterior."""
    rng = np.random.default_rng(0)
    k = len(hist.EDGES) - 1
    truth = rng.dirichlet(np.full(k, 2.))
    binw = hist.bin_weight(4.)
    shifted = truth * binw
    shifted = shifted / shifted.sum()
    margins = np.log(shifted)[None, :]
    reps = hist.representatives(np.array([420., 560.]))
    raw, _ = hist.histogram_mean(margins, reps)
    corrected, posterior = hist.histogram_mean(margins, reps, binw)
    np.testing.assert_allclose(posterior[0], truth, atol=1e-12)
    assert abs(corrected[0] - truth @ reps) < 1e-9
    assert corrected[0] < raw[0]


def test_histogram_mean_is_the_posterior_mean():
    margins = np.zeros((3, len(hist.EDGES) - 1))
    reps = (hist.EDGES[:-1] + hist.EDGES[1:]) / 2.
    mean, posterior = hist.histogram_mean(margins, reps)
    np.testing.assert_allclose(posterior, 1. / len(reps))
    np.testing.assert_allclose(mean, reps.mean())
    assert np.all((mean > 240.) & (mean < 1150.))


def test_representatives_use_conditional_means_and_are_increasing():
    values = np.array([300., 320., 500., 560., 900.])
    reps = hist.representatives(values)
    assert np.all(np.diff(reps) > 0)
    assert reps[hist.bin_index(np.array([300.]))[0]] == pytest.approx(310.)
    assert reps[hist.bin_index(np.array([560.]))[0]] == pytest.approx(560.)


def test_softmax_is_stable_for_large_margins():
    margins = np.array([[1000., -1000., 0.] + [0.] * 13])
    posterior = hist.softmax(margins)
    assert np.isfinite(posterior).all()
    np.testing.assert_allclose(posterior.sum(axis=1), 1., atol=1e-12)
    with pytest.raises(ValueError):
        hist.softmax(np.zeros((2, 3)))


def test_class_objective_matches_cross_entropy_gradient():
    rng = np.random.default_rng(1)
    k = len(hist.EDGES) - 1
    q = rng.dirichlet(np.full(k, 3.), size=4)
    margins = rng.normal(scale=.2, size=(4, k))

    class Matrix:
        def __init__(self, label):
            self.label = label

        def get_label(self):
            return self.label

    lookup = lambda dmat: (q, None)  # noqa: E731
    objective = hist.class_objective(lookup)
    gradient, hessian = objective(margins, Matrix(np.arange(4.)))
    posterior = hist.softmax(margins)
    np.testing.assert_allclose(gradient, posterior - q, atol=1e-12)
    np.testing.assert_allclose(hessian, np.maximum(posterior * (1. - posterior), 1e-6))
    np.testing.assert_allclose(gradient.sum(axis=1), 0., atol=1e-12)
    assert (hessian > 0).all()


def test_pointwise_weights_reach_a_custom_objective():
    """Guards the xgboost 2.1.4 trap: DMatrix.weight is ignored under a custom obj."""
    import xgboost as xgb
    rng = np.random.default_rng(3)
    n, p = 300, 6
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = 450. + 60. * X[:, 0] + 20. * rng.normal(size=n)
    q = hist.soft_labels(y)
    params = dict(objective='multi:softprob', num_class=len(hist.EDGES) - 1, max_depth=3,
                  learning_rate=0.3, nthread=1, seed=0, base_score=0.0,
                  disable_default_eval_metric=1, reg_lambda=0.3)
    margins = {}
    for tag, factor in (('unit', 1.0), ('fast', 4.0)):
        registry = hist.LabelRegistry()
        dm = registry.add(xgb.DMatrix(X, label=np.arange(n, dtype=np.float32)), q,
                          hist.point_weights(y, factor))
        booster = xgb.train(params, dm, num_boost_round=20,
                            obj=hist.class_objective(registry), verbose_eval=False)
        margins[tag] = booster.predict(xgb.DMatrix(X), output_margin=True)
    assert not np.array_equal(margins['unit'], margins['fast'])
    assert np.abs(margins['unit'] - margins['fast']).max() > 1e-3
