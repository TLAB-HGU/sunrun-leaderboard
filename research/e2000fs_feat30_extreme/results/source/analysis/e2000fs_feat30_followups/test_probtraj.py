import numpy as np
import pytest

from probtraj import LeafTrajectoryDistribution, distribution_summary, weighted_crps


def trajectories(values):
    return np.repeat(np.asarray(values, dtype=float)[:, None], 72, axis=1)


def test_inverse_occupancy_and_whole_trajectory_weights():
    leaves = np.array([[1, 1], [1, 2], [2, 2]])
    targets = trajectories([100, 200, 300]) + np.arange(72)
    model = LeafTrajectoryDistribution().fit(leaves, targets)
    pred = next(model.iter_predictions([[1, 2]]))
    # Shared leaf score: [1/2, 1/2+1/2, 1/2].
    np.testing.assert_array_equal(pred['indices'], [1, 0, 2])
    np.testing.assert_allclose(pred['weights'], [.5, .25, .25])
    np.testing.assert_allclose(pred['mean'], 200 + np.arange(72))
    assert not pred['fallback']
    assert pred['weights'].sum() == 1


def test_tree_identity_does_not_cross_match_equal_leaf_ids():
    model = LeafTrajectoryDistribution().fit([[1, 2], [2, 1]], trajectories([10, 20]))
    pred = next(model.iter_predictions([[1, 2]]))
    np.testing.assert_array_equal(pred['indices'], [0])
    np.testing.assert_array_equal(pred['mean'], trajectories([10])[0])


def test_fallback_uses_all_training_rows_even_when_k_is_one():
    model = LeafTrajectoryDistribution().fit([[1], [1], [2]], trajectories([10, 20, 90]))
    pred = next(model.iter_predictions([[99]], k=1))
    assert pred['fallback']
    np.testing.assert_array_equal(pred['indices'], [0, 1, 2])
    np.testing.assert_allclose(pred['weights'], [1/3] * 3)
    np.testing.assert_allclose(pred['mean'], 40)


def test_ties_are_resolved_by_training_index_and_batches_agree():
    model = LeafTrajectoryDistribution().fit(np.ones((5, 3)), trajectories(range(5)))
    queries = np.ones((3, 3))
    a = list(model.iter_predictions(queries, k=2, batch_size=1))
    b = list(model.iter_predictions(queries, k=2, batch_size=3))
    for first, second in zip(a, b):
        np.testing.assert_array_equal(first['indices'], [0, 1])
        np.testing.assert_array_equal(first['indices'], second['indices'])
        np.testing.assert_allclose(first['weights'], [.5, .5])


def test_crps_matches_pairwise_definition_and_point_mass():
    rng = np.random.default_rng(42)
    samples = rng.normal(size=(7, 4))
    weights = np.arange(1, 8, dtype=float)
    weights /= weights.sum()
    truth = rng.normal(size=4)
    expected = weights @ np.abs(samples-truth) - .5 * np.sum(
        weights[:, None, None] * weights[None, :, None] *
        np.abs(samples[:, None, :] - samples[None, :, :]), axis=(0, 1))
    np.testing.assert_allclose(weighted_crps(samples, weights, truth), expected, atol=1e-14)
    np.testing.assert_allclose(weighted_crps([[2, 4]], [1], [1, 6]), [1, 2])
    np.testing.assert_allclose(weighted_crps([[0], [2]], [.5, .5], [1]), [.5])


def test_summary_preserves_never_crossing_probability_and_ramp():
    samples = trajectories([400, 400, 400])
    samples[1, 5:] = 600  # first crossing at h6
    samples[2, 71] = 700  # first crossing at h72
    summary = distribution_summary(samples, [.5, .3, .2], samples[0])
    assert summary['never550_probability'] == .5
    assert summary['any550_probability'] == .5
    assert summary['first550_probability'][5] == .3
    assert summary['first550_probability'][71] == .2
    assert summary['first550_never_sentinel'] == 73
    np.testing.assert_array_equal(summary['first550_quantiles'], [6, 72, 73])
    np.testing.assert_array_equal(summary['peak_quantiles'], [400, 400, 700])
    np.testing.assert_array_equal(summary['max_6h_rise_quantiles'], [0, 0, 300])
    assert summary['any550_brier'] == .25
    assert np.all(summary['crps'] >= 0)
    assert np.isclose(summary['first550_probability'].sum() + summary['never550_probability'], 1)


def test_no_positive_rise_for_falling_trajectories():
    samples = np.tile(np.arange(72, 0, -1), (2, 1))
    summary = distribution_summary(samples, [1, 1])
    np.testing.assert_array_equal(summary['max_6h_rise_quantiles'], [0, 0, 0])


@pytest.mark.parametrize('leaves', [[[np.nan]], [[1.5]], [[-1]]])
def test_invalid_leaf_identifiers_rejected(leaves):
    with pytest.raises(ValueError):
        LeafTrajectoryDistribution().fit(leaves, trajectories([1]))


def test_invalid_shapes_weights_and_unfitted_model_rejected():
    with pytest.raises(ValueError):
        LeafTrajectoryDistribution().fit([[1]], np.zeros((1, 71)))
    with pytest.raises(ValueError):
        next(LeafTrajectoryDistribution().iter_predictions([[1]]))
    with pytest.raises(ValueError):
        weighted_crps([[1], [2]], [1, -1], [0])
    model = LeafTrajectoryDistribution().fit([[1]], trajectories([1]))
    with pytest.raises(ValueError):
        next(model.iter_predictions([[1, 2]]))
    with pytest.raises(ValueError):
        next(model.iter_predictions([[1]], k=0))
