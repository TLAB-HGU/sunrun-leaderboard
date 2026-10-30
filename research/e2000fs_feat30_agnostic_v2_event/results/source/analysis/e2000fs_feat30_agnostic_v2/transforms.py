"""Direct output representations, fitted only from mature training labels.

The event representation has one linear rise, a plateau, and one linear decay.
Its reconstruction diagnostics describe representation error, not forecast skill.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks


LOWER = np.array([1., 0., 0., 1., 0., 0., 1.])
UPPER = np.array([3000., 1500., 72., 48., 1500., 72., 48.])
HORIZONS = np.arange(1., 73.)


def _matrix(values, width, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f'{name} must have shape (n, {width})')
    if not np.isfinite(values).all():
        raise ValueError(f'{name} must be finite')
    return values


def haar72(y):
    """Orthonormal coefficients ordered coarse9, detail9, detail18, detail36."""
    values = _matrix(y, 72, 'y')
    details = []
    for _ in range(3):
        even, odd = values[:, ::2], values[:, 1::2]
        details.append((even - odd) / np.sqrt(2.))
        values = (even + odd) / np.sqrt(2.)
    return np.concatenate([values, *reversed(details)], axis=1)


def inverse_haar72(coefficients):
    """Invert :func:`haar72` without changing or clipping the input."""
    coefficients = _matrix(coefficients, 72, 'coefficients')
    values = coefficients[:, :9]
    offset = 9
    for width in (9, 18, 36):
        details = coefficients[:, offset:offset + width]
        restored = np.empty((len(values), 2 * width), dtype=np.float64)
        restored[:, ::2] = (values + details) / np.sqrt(2.)
        restored[:, 1::2] = (values - details) / np.sqrt(2.)
        values = restored
        offset += width
    return values


def event_curve(params):
    """Decode bounded [baseline, amplitude, onset, rise_duration, decay,
    plateau_gap, decay_duration] into speeds at horizons 1..72.

    Predictions are clipped to parameter bounds first and [1, 3000] km/s last.
    This is the same decoding operation used during parameter fitting.
    """
    params = np.clip(_matrix(params, 7, 'params'), LOWER, UPPER)
    b, a, tau, duration, c, gap, decay_duration = params.T[:, :, None]
    rise = np.clip((HORIZONS - tau) / duration, 0., 1.)
    decay = np.clip((HORIZONS - tau - duration - gap) / decay_duration, 0., 1.)
    return np.clip(b + a * rise - c * decay, 1., 3000.)


def _curve_jacobian(params):
    """Analytic local Jacobian; ramp knots use the inactive-side derivative."""
    b, a, tau, duration, c, gap, decay_duration = params
    u = (HORIZONS - tau) / duration
    v = (HORIZONS - tau - duration - gap) / decay_duration
    rise, decay = np.clip(u, 0., 1.), np.clip(v, 0., 1.)
    active_u = (u > 0.) & (u < 1.)
    active_v = (v > 0.) & (v < 1.)
    raw = b + a * rise - c * decay
    jac = np.column_stack([
        np.ones(72), rise,
        -a / duration * active_u + c / decay_duration * active_v,
        -a * u / duration * active_u + c / decay_duration * active_v,
        -decay,
        c / decay_duration * active_v,
        c * v / decay_duration * active_v,
    ])
    jac[(raw < 1.) | (raw > 3000.)] = 0.
    return np.clip(raw, 1., 3000.), jac


def _rise_count(y):
    """Count local six-hour rises >=100 km/s, separated by >=6 hours.

    Padding includes endpoint maxima. A flat maximum is counted once by
    scipy.signal.find_peaks. This is a target-shape diagnostic, not a physical
    event identifier; several shoulders of one event can count separately.
    """
    changes = y[6:] - y[:-6]
    peaks, _ = find_peaks(np.r_[-np.inf, changes, -np.inf], height=100., distance=6)
    return len(peaks)


def fit_event_parameters(y, max_nfev=80):
    """Deterministically fit mature training-label curves with bounded LS.

    Starts use the largest six-hour-rise location and onsets 0, 24, 48, with
    rise duration 6 and plateau gap 12. A non-rising target (maximum six-hour
    change <=0) fixes amplitude exactly at zero and fits decay with the other
    six parameters; an additional immediate-decay start handles such curves.
    The smallest SSE wins, then the norm of bound-normalized parameters, then
    parameter tuple. ``success`` is the winning optimizer's convergence flag;
    a finite best fit can be returned with False when the evaluation cap is hit.
    ``multiple_rises`` stores a count, so values >1 flag multiple-rise targets.
    """
    y = _matrix(y, 72, 'y')
    if not isinstance(max_nfev, (int, np.integer)) or max_nfev < 1:
        raise ValueError('max_nfev must be a positive integer')
    params = np.empty((len(y), 7), dtype=np.float64)
    reconstruction_rmse = np.empty(len(y), dtype=np.float64)
    success = np.zeros(len(y), dtype=bool)
    no_rise = np.zeros(len(y), dtype=bool)
    multiple_rises = np.zeros(len(y), dtype=np.int64)
    for i, target in enumerate(y):
        changes = target[6:] - target[:-6]
        no_rise[i] = changes.max() <= 0.
        multiple_rises[i] = _rise_count(target)
        baseline = np.clip(target[:3].mean(), LOWER[0], UPPER[0])
        amplitude = 0. if no_rise[i] else np.clip(target.max() - baseline, 1., 1500.)
        decay = np.clip(target.max() - target[-3:].mean(), 0., 1500.)
        starts = [np.array([baseline, amplitude, tau, 6., decay, 12., 12.])
                  for tau in sorted({0., 24., 48., float(np.argmax(changes) + 1)})]
        if no_rise[i]:
            starts.append(np.array([np.clip(target[0], 1., 3000.), 0., 0., 1.,
                                    np.clip(target[0] - target[-1], 0., 1500.), 0., 48.]))
        free = np.array([0, 2, 3, 4, 5, 6] if no_rise[i] else range(7))

        def full_parameters(free_values):
            result = np.zeros(7)
            result[free] = free_values
            return result

        def residual(free_values):
            return _curve_jacobian(full_parameters(free_values))[0] - target

        def jacobian(free_values):
            return _curve_jacobian(full_parameters(free_values))[1][:, free]

        candidates = []
        for start in starts:
            start = np.clip(start, LOWER, UPPER)
            result = least_squares(residual, start[free], jac=jacobian,
                                   bounds=(LOWER[free], UPPER[free]),
                                   x_scale='jac', max_nfev=max_nfev)
            fitted = full_parameters(result.x)
            error = event_curve(fitted[None, :])[0] - target
            sse = float(error @ error)
            norm = float(np.linalg.norm((fitted - LOWER) / (UPPER - LOWER)))
            candidates.append((sse, norm, tuple(fitted), bool(result.success)))
        best = min(candidates, key=lambda item: item[:3])
        params[i] = best[2]
        reconstruction_rmse[i] = np.sqrt(best[0] / 72.)
        success[i] = best[3]
    return {'params': params, 'reconstruction_rmse': reconstruction_rmse,
            'success': success, 'no_rise': no_rise, 'multiple_rises': multiple_rises}


def parameters(y, max_nfev=80):
    """Convenience target encoder returning only the fitted (n, 7) matrix."""
    return fit_event_parameters(y, max_nfev=max_nfev)['params']
