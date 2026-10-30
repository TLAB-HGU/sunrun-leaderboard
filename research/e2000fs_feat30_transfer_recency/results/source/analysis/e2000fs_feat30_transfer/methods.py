"""Fixed causal tree adaptations and fail-closed transfer gates."""
import numpy as np
import pandas as pd

KEYS = ('mse', 'mse_lt550', 'mse_ge550')


def mature_indices(origins, cutoff):
    return np.flatnonzero(origins + pd.Timedelta(hours=72) < cutoff)


def training_policy(origins, indices, lane):
    times = origins[indices]
    age = np.asarray((times.max() - times) / pd.Timedelta(days=1))
    if lane == 'recent_window':
        return indices[age <= 365], None
    if lane == 'recency':
        weights = .1 + .9 * np.exp(-np.log(2) * age / 180)
        return indices, weights / weights.mean()
    return indices, None


def speed_mask(names, selected):
    constant = [n for n in names['constant'] if n in selected]
    varying = [n for n in names['varying'] if n in selected]
    return np.array([n == 'v_lag0' or n.startswith('ace_speed') or
                     n in ('rec26', 'rec27', 'rec28', 'rec_mean')
                     for n in constant + varying * 24])


def center(blocks, past, mask):
    anchor = np.asarray(past, dtype=float).mean(axis=1)
    if not np.isfinite(anchor).all():
        raise ValueError('Nonfinite causal anchor')
    transformed = []
    for block in blocks:
        value = block.copy()
        value[:, mask] -= anchor[:, None]
        transformed.append(value)
    return transformed, anchor


def gate(candidate, baseline, improvements=False):
    ratios = (.99, 1., .95) if improvements else (1., 1., 1.)
    return all(candidate.get(k) is not None and baseline.get(k) is not None
               and np.isfinite(candidate[k]) and np.isfinite(baseline[k])
               and candidate[k] <= baseline[k] * ratio
               for k, ratio in zip(KEYS, ratios))


def development_gate(candidate, baseline):
    return (all(gate(candidate[f'dev{i}'], baseline[f'dev{i}']) for i in range(3))
            and gate(candidate['pooled'], baseline['pooled'], True))
