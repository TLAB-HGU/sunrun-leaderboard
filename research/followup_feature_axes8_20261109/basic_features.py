"""Past-only derived feature additions for preregistered axes 1–4."""
from __future__ import annotations

import numpy as np
import pandas as pd

SHAPE_FEATURES = (
    'slope_norm', 'delta_norm', 'curv', 'peak_pos', 'trough_pos',
    'frac_inc', 'ac1', 'r2_lin',
)
_SPECS = {
    1: (('v_diff24_48', 'v_diff48_72', 'v_curv0_24_48'), (),
        ('v_lag0', 'v_lag24', 'v_lag48', 'v_lag72')),
    2: ((), ('rec_spread',), ('rec26', 'rec27', 'rec28')),
    3: ((), ('rec_mean_minus_v0', 'rec_mean_minus_v24'),
        ('rec_mean', 'v_lag0', 'v_lag24')),
    4: (SHAPE_FEATURES, (), ('origin',)),
}


def describe(axis):
    """Names by model channel and required frame/history input columns."""
    if str(axis) not in ('1', '2', '3', '4'):
        raise ValueError(f'Unsupported basic feature axis: {axis}')
    constant, varying, sources = _SPECS[int(axis)]
    return {'constant': constant, 'varying': varying, 'sources': sources,
            'requires_history': int(axis) == 4,
            'history_sources': ('origin', 'v_lag0') if int(axis) == 4 else ()}


def _require(frame, names):
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing causal input columns: {sorted(missing)}')


def _origins(values):
    origins = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    if origins.isna().any() or (origins.floor('h') != origins).any():
        raise ValueError('Origins must be finite exact UTC hours')
    return origins


def _shape_features(bins):
    # Exact morphology math from post_ablation_five_axes_20261108/feature_models.py.
    if np.isfinite(bins).sum() < 9:
        return None
    values = pd.Series(bins).interpolate(limit=2, limit_direction='both').to_numpy()
    if not np.isfinite(values).all() or values.std() < 10.:
        return None
    times = np.arange(len(values), dtype=float)
    centered = times - times.mean()
    mean, std = values.mean(), values.std()
    slope = np.sum(centered * (values - mean)) / np.sum(centered ** 2)
    residual = values - (mean + slope * centered)
    adjacent_std = values[:-1].std() * values[1:].std()
    if adjacent_std == 0:
        return None
    result = np.array([
        slope * 12. / std, (values[-1] - values[0]) / std,
        np.polyfit(times, values, 2)[0] * 1e3,
        np.argmax(values) / 11., np.argmin(values) / 11.,
        (np.diff(values) > 0).mean(), np.corrcoef(values[:-1], values[1:])[0, 1],
        1. - np.sum(residual ** 2) / np.sum((values - mean) ** 2),
    ])
    return result if np.isfinite(result).all() else None


def _morphology(frame, history):
    if history is None:
        raise ValueError('Axis 4 requires separate hourly origin/v_lag0 history')
    _require(history, ('origin', 'v_lag0'))
    if 'h' in history:
        history = history.loc[history['h'] == 1]
    hourly = history.loc[:, ['origin', 'v_lag0']].copy()
    hourly['origin'] = _origins(hourly['origin'])
    hourly['v_lag0'] = hourly['v_lag0'].astype(float)
    if np.isinf(hourly['v_lag0']).any():
        raise ValueError('Infinite history values are invalid')
    duplicate = hourly.origin.duplicated(keep=False)
    if (hourly.loc[duplicate].groupby('origin').v_lag0.nunique(dropna=False) > 1).any():
        raise ValueError('Duplicate origins disagree on causal input values')
    origins = _origins(frame['origin'])
    result = pd.DataFrame(np.nan, index=frame.index, columns=SHAPE_FEATURES)
    if hourly.empty or frame.empty:
        return result
    hourly = hourly.drop_duplicates('origin').set_index('origin').sort_index()
    # No later origin enters any window; v_lag0 at O is physical O-1h.
    hourly = hourly.loc[hourly.index <= origins.max()]
    if hourly.empty:
        return result
    hourly = hourly.reindex(pd.date_range(hourly.index.min(), origins.max(), freq='h'))
    values = np.pad(hourly.v_lag0.to_numpy(), (71, 0), constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(values, 72).reshape(-1, 12, 6)
    finite = np.isfinite(windows)
    counts = finite.sum(2)
    bins = np.where(finite, windows, 0.).sum(2) / np.maximum(counts, 1)
    bins[counts < 2] = np.nan
    shapes = pd.DataFrame(np.nan, index=hourly.index, columns=SHAPE_FEATURES)
    for origin in origins.unique():
        pos = hourly.index.get_indexer([origin])[0]
        if pos >= 0:
            computed = _shape_features(bins[pos])
            if computed is not None:
                shapes.loc[origin] = computed
    result.iloc[:, :] = shapes.reindex(origins).to_numpy()
    return result


def build(axis, frame, history=None):
    """Return additions only, without mutating inputs or consulting targets.

    Arithmetic uses strict missing propagation per output, including population
    recurrence SD. Unusable morphology windows retain eight missing values.
    """
    spec = describe(axis)
    _require(frame, spec['sources'])
    axis = int(axis)
    if axis == 4:
        return _morphology(frame, history)
    values = frame.loc[:, list(spec['sources'])].astype(float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError('Infinite source values are invalid')
    if axis == 1:
        outputs = (values.v_lag24 - values.v_lag48,
                   values.v_lag48 - values.v_lag72,
                   values.v_lag0 - 2 * values.v_lag24 + values.v_lag48)
    elif axis == 2:
        outputs = (np.std(values.to_numpy(), axis=1, ddof=0),)
    else:
        outputs = (values.rec_mean - values.v_lag0, values.rec_mean - values.v_lag24)
    return pd.DataFrame(dict(zip(spec['constant'] + spec['varying'], outputs)),
                        index=frame.index)
