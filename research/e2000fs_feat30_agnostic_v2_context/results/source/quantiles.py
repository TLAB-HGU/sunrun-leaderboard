"""Model-independent quantile interpolation and empirical trajectory coupling.

Shapes are batch-first: forecasts (N, 72, 5), observations (N, 72), and
trajectory samples (N, K, 72). Only ``pit_ranks`` and ``probability_metrics``
accept observations; inference adapters never consult verifying outcomes.
"""
from __future__ import annotations

import numpy as np


LEVELS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
SUMMARY_LEVELS = np.array([0.1, 0.5, 0.9])


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _forecasts(qpred, levels):
    qpred = np.asarray(qpred, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    if levels.shape != (5,) or not np.isfinite(levels).all() or not np.all(np.diff(levels) > 0):
        raise ValueError('five finite strictly increasing quantile levels are required')
    if np.any(levels <= 0) or np.any(levels >= 1):
        raise ValueError('quantile levels must lie strictly between zero and one')
    if qpred.ndim != 3 or qpred.shape[0] == 0 or qpred.shape[1:] != (72, 5):
        raise ValueError('quantile forecasts must have shape (N, 72, 5), N > 0')
    if not np.isfinite(qpred).all():
        raise ValueError('quantile forecasts must be finite')
    return np.sort(qpred, axis=-1), levels


def _samples(samples):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 3 or min(samples.shape[:2]) == 0 or samples.shape[2] != 72:
        raise ValueError('samples must have shape (N, K, 72), N and K > 0')
    if not np.isfinite(samples).all():
        raise ValueError('samples must be finite')
    return samples


def _observations(y, n):
    y = np.asarray(y, dtype=np.float64)
    if y.shape != (n, 72) or not np.isfinite(y).all():
        raise ValueError('observations must be finite with shape (N, 72)')
    return y


def quantile_samples(qpred, levels=LEVELS, n_samples=128):
    """Interpolate sorted marginal quantiles on midpoint probability levels.

    Outside the supplied levels, the inverse CDF is clamped to the smallest
    and largest predicted quantiles. Crossing repair sorts only forecast values.
    """
    qpred, levels = _forecasts(qpred, levels)
    count = _positive_integer(n_samples, 'n_samples')
    u = (np.arange(count) + 0.5) / count
    result = np.empty((len(qpred), count, 72))
    for row in range(len(qpred)):
        for horizon in range(72):
            result[row, :, horizon] = np.interp(u, levels, qpred[row, horizon])
    return result


def pit_ranks(y, qpred, levels=LEVELS):
    """Approximate PIT values for a historical calibration/rank library.

    Duplicate forecast values use the midpoint of their quantile levels.
    Observations strictly below/above the forecast range receive 0/1; exact
    endpoint ties retain their midpoint level. This deterministic interpolation
    is a rank estimate, not randomized PIT for the clamped distribution's atoms.
    """
    qpred, levels = _forecasts(qpred, levels)
    y = _observations(y, len(qpred))
    result = np.empty_like(y)
    for row in range(len(qpred)):
        for horizon in range(72):
            values, first, counts = np.unique(qpred[row, horizon], return_index=True, return_counts=True)
            midpoint = (levels[first] + levels[first + counts - 1]) / 2
            result[row, horizon] = np.interp(y[row, horizon], values, midpoint, left=0, right=1)
    return result


def couple_samples(marginals, rank_library, count=128):
    """Reorder marginal samples using deterministic historical PIT ranks.

    Select ``count`` equally spaced library rows, including both endpoints.
    A short library repeats rows. Stable ordering resolves ties by selected-row
    position. The same selected rows couple all horizons and forecast origins.
    """
    samples = _samples(marginals)
    count = _positive_integer(count, 'count')
    if count != samples.shape[1]:
        raise ValueError('count must equal marginal sample count to preserve marginals')
    library = np.asarray(rank_library, dtype=np.float64)
    if library.ndim != 2 or library.shape[0] == 0 or library.shape[1] != 72:
        raise ValueError('rank_library must have shape (M, 72), M > 0')
    if not np.isfinite(library).all() or np.any(library < 0) or np.any(library > 1):
        raise ValueError('rank library values must be finite PIT ranks in [0, 1]')
    indices = np.linspace(0, len(library) - 1, count, dtype=np.int64)
    order = np.argsort(library[indices], axis=0, kind='stable')
    ranks = np.argsort(order, axis=0, kind='stable')
    ordered = np.sort(samples, axis=1)
    return np.take_along_axis(ordered, np.broadcast_to(ranks, ordered.shape), axis=1)


def independent_samples(marginals, seed=42):
    """Apply independent reproducible sample permutations at each horizon."""
    samples = np.sort(_samples(marginals), axis=1)
    rng = np.random.default_rng(seed)
    result = np.empty_like(samples)
    for row in range(len(samples)):
        for horizon in range(72):
            result[row, :, horizon] = samples[row, rng.permutation(samples.shape[1]), horizon]
    return result


def mean_of_samples(samples):
    return _samples(samples).mean(axis=1)


def probability_metrics(y, sample):
    """Scores and trajectory summaries; first crossing 73 denotes no event.

    CRPS is exact for the equal-weight empirical distribution, not the fair
    finite-ensemble estimator. Quantiles use the empirical inverse CDF to keep
    first-crossing hours integral. Scores include all observations/false alarms.
    """
    sample = _samples(sample)
    y = _observations(y, len(sample))
    count = sample.shape[1]
    ordered = np.sort(sample, axis=1)
    coefficients = (2 * np.arange(1, count + 1) - count - 1) / count ** 2
    half_pair_distance = np.sum(ordered * coefficients[None, :, None], axis=1)
    crps = np.maximum(np.abs(sample - y[:, None]).mean(axis=1) - half_pair_distance, 0)
    high = sample >= 550
    hourly_probability = high.mean(axis=1)
    any_high = high.any(axis=2)
    any_probability = any_high.mean(axis=1)
    first = np.where(any_high, high.argmax(axis=2) + 1, 73)
    first_mass = np.stack([np.bincount(row, minlength=74)[1:] / count for row in first])
    peak = sample.max(axis=2)
    ramp = np.maximum(0, (sample[:, :, 6:] - sample[:, :, :-6]).max(axis=2))
    def quantiles(values):
        return np.quantile(values, SUMMARY_LEVELS, axis=1, method='inverted_cdf').T

    brier = (hourly_probability - (y >= 550)) ** 2
    any_brier = (any_probability - (y >= 550).any(axis=1)) ** 2
    return {
        'mean': sample.mean(axis=1),
        'crps': crps,
        'crps_mean': float(crps.mean()),
        'brier': brier,
        'brier_mean': float(brier.mean()),
        'any550_brier': any_brier,
        'any550_brier_mean': float(any_brier.mean()),
        'hourly_exceedance': hourly_probability,
        'any550_probability': any_probability,
        'quantile_levels': SUMMARY_LEVELS.copy(),
        'peak_quantiles': quantiles(peak),
        'max_6h_rise_quantiles': quantiles(ramp),
        'first550_quantiles': quantiles(first),
        'first550_probability': first_mass[:, :72],
        'never550_probability': first_mass[:, 72],
        'first550_never_sentinel': 73,
    }
