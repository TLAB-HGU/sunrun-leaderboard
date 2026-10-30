"""Symmetric smooth error-tail loss; a shortfall-risk-inspired surrogate."""
import numpy as np
from scipy.special import expit


def loss(y, prediction, alpha=3., cutoff=100., scale=50.):
    error = np.asarray(prediction, dtype=float) - np.asarray(y, dtype=float)
    z = (error**2 - cutoff**2) / scale**2
    return .5 * error**2 + .5 * alpha * scale**2 * np.logaddexp(0., z)


def objective(alpha=3., cutoff=100., scale=50.):
    if not np.isfinite([alpha, cutoff, scale]).all() or alpha < 0 or cutoff < 0 or scale <= 0:
        raise ValueError('Invalid smooth-tail loss configuration')

    def derivatives(y, prediction):
        error = np.asarray(prediction, dtype=float) - np.asarray(y, dtype=float)
        probability = expit((error**2 - cutoff**2) / scale**2)
        grad = error * (1. + alpha * probability)
        hess = 1. + alpha * probability + 2. * alpha * error**2 / scale**2 * probability * (1. - probability)
        return grad, hess

    return derivatives
