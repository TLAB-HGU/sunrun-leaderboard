"""Conditional distributions over whole observed trajectories using tree leaves.

The caller supplies leaves from models trained on mature targets only. This
module never fits a forecasting model or adds independently sampled hourly noise.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def _leaves(values, columns=None):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError('leaves must be a two-dimensional array with trees')
    if columns is not None and values.shape[1] != columns:
        raise ValueError('query and training tree counts differ')
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.floor(values)):
        raise ValueError('leaf identifiers must be finite nonnegative integers')
    return values


def _distribution(samples, weights):
    samples = np.asarray(samples, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError('samples must have nonempty sample and horizon axes')
    if weights.shape != (samples.shape[0],):
        raise ValueError('one weight is required per complete trajectory')
    if not np.isfinite(samples).all() or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('samples and nonnegative weights must be finite')
    total = weights.sum()
    if total <= 0:
        raise ValueError('distribution must have positive total weight')
    return samples, weights / total


class LeafTrajectoryDistribution:
    """Inverse-occupancy leaf proximity with deterministic donor selection.

    Donor i receives sum_t I[leaf_i,t == leaf_query,t] / occupancy_t.
    The sparse membership matrix has one entry per training origin and tree.
    Query batches materialize only batch_size x n_train proximity scores.
    """

    def fit(self, train_leaves, targets):
        leaves = _leaves(train_leaves)
        targets = np.asarray(targets, dtype=np.float64)
        if not len(leaves) or targets.shape != (len(leaves), 72) or not np.isfinite(targets).all():
            raise ValueError('training requires finite targets with shape (n_origins, 72)')
        n, self.n_trees = leaves.shape
        codes = np.empty((n, self.n_trees), dtype=np.int32)
        self.leaf_values, self.offsets, inverse_counts = [], [], []
        offset = 0
        for tree in range(self.n_trees):
            values, inverse, counts = np.unique(leaves[:, tree], return_inverse=True, return_counts=True)
            if offset + len(values) > np.iinfo(np.int32).max:
                raise ValueError('too many distinct tree leaves')
            self.leaf_values.append(values)
            self.offsets.append(offset)
            codes[:, tree] = inverse + offset
            inverse_counts.append(1.0 / counts)
            offset += len(values)
        inverse_counts = np.concatenate(inverse_counts)
        membership = sparse.csr_matrix(
            (inverse_counts[codes.ravel()], codes.ravel(),
             np.arange(0, (n + 1) * self.n_trees, self.n_trees, dtype=np.int64)),
            shape=(n, offset),
        )
        self.membership_transpose = membership.T
        self.n_leaf_codes = offset
        self.targets = targets.copy()
        return self

    def _query_membership(self, leaves):
        rows, cols = [], []
        for tree, (values, offset) in enumerate(zip(self.leaf_values, self.offsets)):
            at = np.searchsorted(values, leaves[:, tree])
            valid = at < len(values)
            valid[valid] &= values[at[valid]] == leaves[valid, tree]
            rows.append(np.flatnonzero(valid))
            cols.append(at[valid] + offset)
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                                 shape=(len(leaves), self.n_leaf_codes))

    def iter_predictions(self, query_leaves, k=128, batch_size=8):
        """Yield dictionaries containing indices, weights, mean and fallback.

        Normal queries return at most k donors with positive proximity, ordered
        by descending proximity then training index. Unmatched queries return
        *all* training donors uniformly, preserving the empirical distribution.
        """
        if not hasattr(self, 'targets'):
            raise ValueError('fit must be called before prediction')
        leaves = _leaves(query_leaves, self.n_trees)
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError('k must be a positive integer')
        if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        for start in range(0, len(leaves), batch_size):
            scores = (self._query_membership(leaves[start:start + batch_size])
                      @ self.membership_transpose).toarray()
            for score in scores:
                indices = np.flatnonzero(score > 0)
                fallback = not len(indices)
                if fallback:
                    indices = np.arange(len(self.targets))
                    weights = np.full(len(indices), 1.0 / len(indices))
                else:
                    order = np.lexsort((indices, -score[indices]))[:k]
                    indices = indices[order]
                    weights = score[indices] / score[indices].sum()
                yield {'indices': indices, 'weights': weights,
                       'mean': weights @ self.targets[indices], 'fallback': fallback}


def weighted_crps(samples, weights, truth):
    """Exact empirical CRPS per horizon, in O(h * n log n) time."""
    samples, weights = _distribution(samples, weights)
    truth = np.asarray(truth, dtype=np.float64)
    if truth.shape != (samples.shape[1],) or not np.isfinite(truth).all():
        raise ValueError('truth must be finite and match the horizon axis')
    order = np.argsort(samples, axis=0, kind='stable')
    ordered = np.take_along_axis(samples, order, axis=0)
    w = weights[order]
    cumulative = np.cumsum(w, axis=0)
    half_pair_distance = np.sum(w * ordered * (2 * cumulative - w - 1), axis=0)
    crps = weights @ np.abs(samples - truth) - half_pair_distance
    # Floating-point cancellation can produce a tiny negative score at a point mass.
    return np.maximum(crps, 0.0)


def _quantiles(values, weights, probabilities):
    order = np.argsort(values, kind='stable')
    values, weights = values[order], weights[order]
    positive = weights > 0
    values, weights = values[positive], weights[positive]
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return values[np.searchsorted(cumulative, probabilities, side='left')]


def distribution_summary(samples, weights, truth=None):
    """Summarize complete trajectories; hour 73 means no crossing within 72h.

    First-crossing outputs include the probability of never reaching 550 km/s;
    their quantiles are unconditional and may therefore contain sentinel 73.
    """
    samples, weights = _distribution(samples, weights)
    if samples.shape[1] != 72:
        raise ValueError('trajectory summaries require 72 horizons')
    probabilities = np.array([0.1, 0.5, 0.9])
    high = samples >= 550.0
    any_high = high.any(axis=1)
    first = np.where(any_high, high.argmax(axis=1) + 1, 73)
    first_mass = np.bincount(first, weights=weights, minlength=74)[1:]
    peaks = samples.max(axis=1)
    ramp = np.maximum(0, (samples[:, 6:] - samples[:, :-6]).max(axis=1))
    result = {
        'mean': weights @ samples,
        'hourly_exceedance': weights @ high,
        'quantile_levels': probabilities,
        'peak_quantiles': _quantiles(peaks, weights, probabilities),
        'max_6h_rise_quantiles': _quantiles(ramp, weights, probabilities),
        'first550_quantiles': _quantiles(first, weights, probabilities),
        'first550_probability': first_mass[:72],
        'never550_probability': float(first_mass[72]),
        'any550_probability': float(weights @ any_high),
        'first550_never_sentinel': 73,
    }
    if truth is not None:
        truth = np.asarray(truth, dtype=np.float64)
        result['crps'] = weighted_crps(samples, weights, truth)
        result['brier'] = (result['hourly_exceedance'] - (truth >= 550)) ** 2
        result['any550_brier'] = float((result['any550_probability'] - bool(np.any(truth >= 550))) ** 2)
    return result
