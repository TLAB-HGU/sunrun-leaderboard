import inspect

import numpy as np
import pytest

from quantiles import (
    couple_samples, independent_samples, mean_of_samples, pit_ranks,
    probability_metrics, quantile_samples,
)


def forecast(values, n=1):
    return np.tile(np.asarray(values, dtype=float), (n, 72, 1))


def test_quantile_crossing_repair_interpolation_and_clamped_tails():
    q = forecast([9, 1, 7, 3, 5])
    original = q.copy()
    samples = quantile_samples(q, n_samples=10)
    np.testing.assert_allclose(samples[0, :, 0], [1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9])
    np.testing.assert_array_equal(q, original)
    assert samples.shape == (1, 10, 72)


def test_pit_interpolation_ties_and_outside_range():
    q = forecast([10, 20, 20, 20, 40], n=5)
    y = np.repeat(np.array([5, 15, 20, 30, 45])[:, None], 72, axis=1)
    ranks = pit_ranks(y, q)
    np.testing.assert_allclose(ranks[:, 0], [0, .3, .5, .7, 1])
    np.testing.assert_allclose(pit_ranks(np.full((1, 72), 12), forecast([12] * 5)), .5)


def test_coupling_preserves_every_hourly_marginal_and_mean():
    rng = np.random.default_rng(8)
    samples = rng.normal(size=(2, 128, 72))
    library = rng.uniform(size=(31, 72))
    coupled = couple_samples(samples, library)
    independent = independent_samples(samples)
    np.testing.assert_array_equal(np.sort(coupled, axis=1), np.sort(samples, axis=1))
    np.testing.assert_array_equal(np.sort(independent, axis=1), np.sort(samples, axis=1))
    np.testing.assert_allclose(mean_of_samples(coupled), mean_of_samples(samples), atol=1e-15)


def test_coherent_library_ranks_produce_coherent_trajectories():
    samples = np.tile(np.arange(128, dtype=float)[None, :, None], (1, 1, 72))
    library = np.tile(np.linspace(0, 1, 128)[:, None], (1, 72))
    coupled = couple_samples(samples, library)
    np.testing.assert_array_equal(coupled, samples)
    independent = independent_samples(samples)
    assert not np.array_equal(independent[:, :, 0], independent[:, :, 1])
    np.testing.assert_array_equal(independent, independent_samples(samples))


def test_stable_ties_and_short_library_are_deterministic():
    samples = np.tile(np.arange(4, dtype=float)[None, :, None], (1, 1, 72))
    library = np.full((1, 72), .5)
    np.testing.assert_array_equal(couple_samples(samples, library, count=4), samples)
    # Inference APIs accept only predictions, historical ranks, and fixed controls.
    assert 'y' not in inspect.signature(couple_samples).parameters
    assert 'y' not in inspect.signature(quantile_samples).parameters
    assert 'y' not in inspect.signature(independent_samples).parameters


def test_crps_matches_pairwise_definition_and_handcalculation():
    rng = np.random.default_rng(9)
    samples = rng.normal(size=(2, 7, 72))
    truth = rng.normal(size=(2, 72))
    expected = np.abs(samples-truth[:, None]).mean(axis=1) - .5 * np.abs(
        samples[:, :, None] - samples[:, None, :]).mean(axis=(1, 2))
    np.testing.assert_allclose(probability_metrics(truth, samples)['crps'], expected, atol=1e-14)
    pair = np.tile(np.array([0., 2.])[None, :, None], (1, 1, 72))
    np.testing.assert_allclose(probability_metrics(np.ones((1, 72)), pair)['crps'], .5)


def test_first_crossing_probabilities_and_no_event_censoring():
    samples = np.full((1, 4, 72), 400.)
    samples[0, 1, 5:] = 600
    samples[0, 2, 71] = 700
    result = probability_metrics(np.full((1, 72), 400.), samples)
    assert result['never550_probability'][0] == .5
    assert result['first550_probability'][0, 5] == .25
    assert result['first550_probability'][0, 71] == .25
    np.testing.assert_array_equal(result['first550_quantiles'], [[6, 72, 73]])
    np.testing.assert_array_equal(result['peak_quantiles'], [[400, 400, 700]])
    np.testing.assert_array_equal(result['max_6h_rise_quantiles'], [[0, 0, 300]])
    assert result['first550_never_sentinel'] == 73
    assert result['any550_brier_mean'] == .25
    assert result['brier'][0, 71] == .25
    np.testing.assert_allclose(result['first550_probability'].sum(axis=1) + result['never550_probability'], 1)


def test_no_event_and_falling_trajectory_summary():
    samples = np.tile(np.arange(72, 0, -1)[None, None, :], (2, 3, 1))
    result = probability_metrics(np.zeros((2, 72)), samples)
    np.testing.assert_array_equal(result['first550_quantiles'], np.full((2, 3), 73))
    np.testing.assert_array_equal(result['max_6h_rise_quantiles'], np.zeros((2, 3)))
    np.testing.assert_array_equal(result['never550_probability'], [1, 1])


def test_marginal_scores_invariant_to_coupling_but_event_scores_can_change():
    samples = np.tile(np.array([400., 400., 600., 600.])[None, :, None], (1, 1, 72))
    truth = np.full((1, 72), 400.)
    independent = independent_samples(samples)
    original = probability_metrics(truth, samples)
    shuffled = probability_metrics(truth, independent)
    np.testing.assert_array_equal(original['crps'], shuffled['crps'])
    np.testing.assert_array_equal(original['brier'], shuffled['brier'])
    assert original['any550_probability'][0] == .5
    assert shuffled['any550_probability'][0] == 1


@pytest.mark.parametrize('bad', [np.zeros((72, 5)), np.zeros((1, 71, 5)), np.full((1, 72, 5), np.nan)])
def test_invalid_forecasts_rejected(bad):
    with pytest.raises(ValueError):
        quantile_samples(bad)


def test_invalid_parameters_and_library_rejected():
    q = forecast([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        quantile_samples(q, n_samples=0)
    with pytest.raises(ValueError):
        quantile_samples(q, levels=[.1, .1, .5, .7, .9])
    samples = quantile_samples(q)
    with pytest.raises(ValueError):
        couple_samples(samples, np.zeros((1, 72)), count=64)
    with pytest.raises(ValueError):
        couple_samples(samples, np.full((1, 72), 2))
    with pytest.raises(ValueError):
        probability_metrics(np.zeros((1, 71)), samples)
    with pytest.raises(ValueError):
        mean_of_samples(np.empty((1, 0, 72)))
