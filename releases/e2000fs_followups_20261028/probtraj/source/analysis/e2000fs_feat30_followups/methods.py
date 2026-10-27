"""Non-neural, block-local adaptations for the frozen feat30 study."""

import numpy as np


def haar_transform24(x, axis=-1):
    """Orthonormal three-level Haar coefficients: coarse3, detail3/6/12."""
    values = np.moveaxis(np.asarray(x, dtype=np.float64), axis, -1)
    if values.shape[-1] != 24:
        raise ValueError('Haar input axis must contain exactly 24 values')
    details = []
    for _ in range(3):
        even, odd = values[..., ::2], values[..., 1::2]
        details.append((even - odd) / np.sqrt(2.))
        values = (even + odd) / np.sqrt(2.)
    coefficients = np.concatenate([values, *reversed(details)], axis=-1)
    return np.moveaxis(coefficients, -1, axis)


def inverse_haar_transform24(x, axis=-1):
    """Invert the coefficient ordering used by :func:`haar_transform24`."""
    coefficients = np.moveaxis(np.asarray(x, dtype=np.float64), axis, -1)
    if coefficients.shape[-1] != 24:
        raise ValueError('Haar coefficient axis must contain exactly 24 values')
    values = coefficients[..., :3]
    offset = 3
    for width in (3, 6, 12):
        detail = coefficients[..., offset:offset + width]
        restored = np.empty((*values.shape[:-1], width * 2), dtype=np.float64)
        restored[..., ::2] = (values + detail) / np.sqrt(2.)
        restored[..., 1::2] = (values - detail) / np.sqrt(2.)
        values = restored
        offset += width
    return np.moveaxis(values, -1, axis)


def augment_design(constants, varying, include_wavelet=True):
    """Append block-local differences and optionally Haar coefficients.

    Inputs must already follow the parent's original constant/varying order.
    The first 237 columns are unchanged. Difference and coefficient arrays are
    flattened time/coefficient-major, with nine features per position. No other
    origin, observed future, or neighboring block is read.
    """
    constants = np.asarray(constants)
    varying = np.asarray(varying)
    if constants.ndim != 2 or constants.shape[1] != 21:
        raise ValueError('constants must have shape (n, 21)')
    if varying.shape != (len(constants), 24, 9):
        raise ValueError('varying must have shape (n, 24, 9)')
    parts = [constants, varying.reshape(len(constants), 216),
             np.diff(varying, axis=1).reshape(len(constants), 207)]
    if include_wavelet:
        parts.append(haar_transform24(varying, axis=1).reshape(len(constants), 216))
    return np.concatenate(parts, axis=1).astype(np.float32)


def extreme_weights(y, factor):
    """Mean-one origin weights from mature 24-hour training targets only."""
    y = np.asarray(y)
    if y.ndim != 2 or y.shape[1] != 24 or not len(y):
        raise ValueError('targets must have nonempty shape (n, 24)')
    if not np.isfinite(y).all():
        raise ValueError('training targets must be finite')
    if not np.isfinite(factor) or factor < 1:
        raise ValueError('factor must be finite and at least one')
    active = (y.max(axis=1) >= 550.) | ((y[:, 6:] - y[:, :-6]).max(axis=1) >= 100.)
    weights = np.where(active, float(factor), 1.)
    return weights / weights.mean()


def shape_objective(lam):
    """XGBoost objective: .5*sum(e²) + .5*lam*sum(diff(e)²).

    The factor .5 matches XGBoost's squared-error gradient/Hessian at lam=0,
    preserving the meaning of regularization. Off-diagonal Hessian entries are
    replaced by a positive diagonal majorizer, I + 2*lam*diag(degree).
    """
    if not np.isfinite(lam) or lam < 0:
        raise ValueError('lam must be finite and nonnegative')
    degree = np.full(24, 2., dtype=np.float64)
    degree[[0, -1]] = 1.

    def objective(y_true, y_pred):
        prediction = np.asarray(y_pred)
        if prediction.ndim != 2 or prediction.shape[1] != 24:
            raise ValueError('predictions must have shape (n, 24)')
        truth = np.asarray(y_true).reshape(prediction.shape)
        error = prediction - truth
        gradient = error.copy()
        differences = np.diff(error, axis=1)
        gradient[:, :-1] -= lam * differences
        gradient[:, 1:] += lam * differences
        hessian = np.broadcast_to(1. + 2. * lam * degree, error.shape).copy()
        return gradient, hessian

    return objective
