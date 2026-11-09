"""Existing official TIMSE/block-bootstrap kernel; no model selection logic."""
import json

import numpy as np

METRICS = ("s", "timse", "rmse", "trmse")


def metric_weights(truth, indices):
    """Linear MSE/TIMSE coefficients, recomputing empirical quantiles per draw.

    Duplicate sampled origins retain multiplicity. With identity indices these
    coefficients reproduce competition.score_competition exactly.
    """
    n, leads = truth.shape
    mse = np.zeros((n, leads))
    timse = np.zeros_like(mse)
    for h in range(leads):
        take = indices[np.isfinite(truth[indices, h])]
        y = truth[take, h]
        if len(y) < 2 or np.ptp(y) == 0:
            raise ValueError('Bootstrap draw has undefined TIMSE population')
        order = np.argsort(y, kind='stable')
        sorted_y = y[order]
        q = np.quantile(y, np.arange(101) / 100, method='linear')
        pos = np.searchsorted(sorted_y, q[:-1], side='left')
        increments = np.zeros(len(y))
        np.add.at(increments, pos, np.diff(q) / (len(y) - pos) / (q[-1] - q[0]))
        tail_weights = np.cumsum(increments)
        mse[:, h] = np.bincount(take, minlength=n) / len(y)
        timse[:, h] = np.bincount(take[order], weights=tail_weights, minlength=n)
    return mse, timse


def scores_from_weights(squared_errors, mse_weights, timse_weights):
    rmse = np.sqrt(np.einsum('anh,nh->ah', squared_errors, mse_weights, optimize=True))
    trmse = np.sqrt(np.einsum('anh,nh->ah', squared_errors, timse_weights, optimize=True))
    return {"s": np.sqrt(rmse * trmse).mean(1), "rmse": rmse.mean(1),
                "trmse": trmse.mean(1), "timse": (trmse ** 2).mean(1)}


def circular_draw(context, rng, block_hours):
    pieces = []
    for period in ('F1', 'F2', 'F3'):
        rows = np.flatnonzero(context['period'] == period)
        # Official origin grid is six-hourly. Keep period populations fixed.
        if len(rows) > 1 and not np.all(np.diff(context['origins'][rows].asi8) == 6 * 3600 * 10**9):
            raise ValueError('Expected the official six-hour origin grid')
        length = max(1, block_hours // 6)
        starts = rng.integers(0, len(rows), size=int(np.ceil(len(rows) / length)))
        positions = (starts[:, None] + np.arange(length)) % len(rows)
        pieces.append(rows[positions.ravel()[:len(rows)]])
    return np.concatenate(pieces)


def bootstrap_scores(context, arrays, count, block_hours, seed=20261104):
    errors = np.where(np.isfinite(context['truth'])[None],
                      (arrays - context['truth'][None]) ** 2, 0.)
    exact = scores_from_weights(errors, *metric_weights(context['truth'], np.arange(len(context['origins']))))
    result = {key: np.empty((count, len(arrays))) for key in METRICS}
    rng = np.random.default_rng(seed)
    accepted = attempts = 0
    while accepted < count:
        attempts += 1
        if attempts > max(100, count * 10):
            raise RuntimeError('Too many undefined bootstrap populations')
        try:
            weights = metric_weights(context['truth'], circular_draw(context, rng, block_hours))
        except ValueError:
            continue
        values = scores_from_weights(errors, *weights)
        for key in METRICS:
            result[key][accepted] = values[key]
        accepted += 1
        if accepted % 100 == 0:
            print(json.dumps({"stage": 'analysis_bootstrap', "unit": accepted, "units": count}), flush=True)
    return exact, result, attempts - accepted


def bh_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.minimum.accumulate((values[order] * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty(len(values))
    result[order] = np.minimum(adjusted, 1.)
    return result.tolist()


def distribution(draws, observed):
    if not len(draws):
        return {"ci95": None, "p_centered_bootstrap": None}
    lo, hi = np.quantile(draws, [.025, .975])
    p = (1 + np.count_nonzero(np.abs(draws - observed) >= abs(observed))) / (len(draws) + 1)
    return {"ci95": [float(lo), float(hi)], "p_centered_bootstrap": float(p)}
