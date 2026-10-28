"""Model-agnostic context-asymmetric loss; no prediction correction."""
import numpy as np
from scipy.special import expit


def event_labels(y):
    y = np.asarray(y)
    if y.ndim != 2 or y.shape[1] != 24 or not np.isfinite(y).all():
        raise ValueError('Expected finite (n,24) labels')
    return ((y.max(1) >= 550) | ((y[:, 6:] - y[:, :-6]).max(1) >= 100)).astype(np.int32)


def purged_blocks(origins, folds=5):
    origins = np.asarray(origins).astype('datetime64[ns]')
    gap = np.timedelta64(72, 'h')
    for held in np.array_split(np.arange(len(origins)), folds):
        if not len(held):
            continue
        keep = np.flatnonzero((origins < origins[held[0]] - gap) |
                              (origins > origins[held[-1]] + gap))
        if not len(keep):
            raise ValueError('No training data remain after 72h purge')
        yield keep, held


def asymmetric_loss(y, p, q, strength, margin=20., scale=10.):
    e = np.asarray(y) - np.asarray(p)
    q = np.asarray(q).reshape(-1, 1)
    return .5 * e**2 + strength * q * scale**2 * np.logaddexp(0, (e - margin) / scale)


def context_objective(q, strength, margin=20., scale=10.):
    q = np.asarray(q, dtype=float).reshape(-1, 1)
    if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)) or strength < 0 or scale <= 0:
        raise ValueError('Invalid context-loss configuration')

    def objective(y_true, y_pred):
        p = np.asarray(y_pred)
        y = np.asarray(y_true).reshape(p.shape)
        if p.ndim != 2 or len(p) != len(q):
            raise ValueError('Context probability must match training rows')
        e = y - p
        probability = expit((e - margin) / scale)
        return (-e - strength * q * scale * probability,
                np.ones_like(p) + strength * q * probability * (1 - probability))

    return objective
