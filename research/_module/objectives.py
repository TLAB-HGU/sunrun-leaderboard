"""XGBoost custom objectives shared by more than one board row (gradient, hessian per cell)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .common import CUTOFFS

TAIL_THRESHOLD, TAIL_WEIGHT = 550., 4.
SPECTRAL_SCALE = 50.


def tail_weights(target):
    """Targets >= 550 km/s weigh 4, then normalised to mean 1 per lead (over observed cells)."""
    if not np.isfinite(target).all():
        raise ValueError('Training targets must be finite')
    observed = np.isfinite(target)
    raw = np.where(target >= TAIL_THRESHOLD, TAIL_WEIGHT, 1.) * observed
    return observed, raw / (raw.sum(0) / observed.sum(0))


def point4(target):
    """Tail-weighted squared error: gradient = w * error, hessian = w (no spectral term)."""
    observed, weights = tail_weights(target)

    def objective(prediction, matrix):
        prediction = np.asarray(prediction).reshape(target.shape)
        error = np.where(observed, prediction - np.nan_to_num(target), 0.)
        return (error * weights * observed).astype(np.float32), (weights * observed).astype(np.float32)
    return objective


def point4_spectral(target, coefficient):
    """Tail-weighted squared error plus a Huber-like FFT term on the block's lead error
    (Hessian = weight + coefficient)."""
    observed, weights = tail_weights(target)

    def objective(prediction, matrix):
        prediction = np.asarray(prediction).reshape(target.shape)
        error = np.where(observed, prediction - np.nan_to_num(target), 0.)
        z = np.fft.fft(error, axis=1, norm='ortho')
        rho = 1 / np.sqrt(1 + abs(z) ** 2 / SPECTRAL_SCALE ** 2)
        gradient = (error * weights + coefficient * np.fft.ifft(
            z * rho, axis=1, norm='ortho').real) * observed
        hessian = (weights + coefficient) * observed
        return gradient.astype(np.float32), hessian.astype(np.float32)
    return objective


def point4_spectral_hessian(target, coefficient):
    """Tail-weighted squared error; the spectral coefficient is kept only in the Hessian."""
    observed, weights = tail_weights(target)

    def objective(prediction, matrix):
        prediction = np.asarray(prediction).reshape(target.shape)
        error = np.where(observed, prediction - np.nan_to_num(target), 0.)
        gradient, hessian = error * weights, (weights + coefficient) * observed
        return gradient.astype(np.float32), hessian.astype(np.float32)
    return objective


def fit_speed_cdf(table, period):
    """loc = 0 lognormal maximum-likelihood fit (closed form) to the observed, unfilled targets of this
    period's training rows, one value per target timestamp."""
    rows = table.loc[table[f'fit_{period}'].to_numpy(), ['target_time', 'y_observed']].drop_duplicates('target_time')
    times = pd.to_datetime(rows.target_time, utc=True)
    speed = rows.y_observed.to_numpy(np.float64)
    keep = np.isfinite(speed) & (speed > 0)
    if keep.sum() < 1000 or not (times.max() < pd.Timestamp(CUTOFFS[period], tz='UTC')):
        raise ValueError('CDF fit needs pre-cutoff observed training targets')
    log_speed = np.log(speed[keep])
    return dict(family='lognormal', loc=0., sigma=float(log_speed.std()), scale=float(np.exp(log_speed.mean())),
                n=int(keep.sum()), first_target=str(times[keep].min()), last_target=str(times[keep].max()))


def cdf_weighted(target, cdf, floor=None, span=None):
    """Squared error with weight F(target) (or floor + span * F(target)), F the fitted lognormal CDF,
    normalised per lead; gradient = w * error, hessian = w."""
    if not np.isfinite(target).all() or not (target > 0).all():
        raise ValueError('Training targets must be finite and positive')
    observed = np.isfinite(target)
    if floor is None:
        raw = ndtr(np.log(target / cdf['scale']) / cdf['sigma']) * observed
    else:
        raw = (floor + span * ndtr(np.log(target / cdf['scale']) / cdf['sigma'])) * observed
    weights = raw / (raw.sum(0) / observed.sum(0))

    def objective(prediction, matrix):
        prediction = np.asarray(prediction).reshape(target.shape)
        error = np.where(observed, prediction - np.nan_to_num(target), 0.)
        return (error * weights * observed).astype(np.float32), (weights * observed).astype(np.float32)
    return objective


def timse_coefficients(target):
    """Official 100-quantile TIMSE coefficients of one training lead; they sum to one."""
    target = np.asarray(target, dtype=np.float64)
    if target.ndim != 1 or len(target) < 2 or not np.isfinite(target).all():
        raise ValueError('TIMSE requires at least two finite training targets')
    spread = np.ptp(target)
    if spread == 0:
        raise ValueError('Constant training targets make TIMSE undefined')
    quantiles = np.quantile(target, np.arange(101) / 100, method='linear')
    order = np.argsort(target, kind='stable')
    positions = np.searchsorted(target[order], quantiles[:-1], side='left')
    increments = np.zeros(len(target), dtype=np.float64)
    np.add.at(increments, positions,
              np.diff(quantiles) / spread / (len(target) - positions))
    coefficients = np.empty(len(target), dtype=np.float64)
    coefficients[order] = np.cumsum(increments)
    return coefficients
