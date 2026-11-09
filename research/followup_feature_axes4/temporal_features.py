"""Frozen, target-free temporal additions for follow-up axes 3 and 4.

Axis 3 compares the 72 hourly v_lag0 samples ending at origin O with the
corresponding window ending at O - (24*d + offset) hours, d=26/27/28.
Offsets are every integer in [-6, 6]. Each Pearson correlation requires at
least 54 finite pairs and positive variance in both windows. Select the
largest correlation; differences <=1e-12 are ties, resolved by nominal lag,
then smallest absolute offset, then negative offset. Positive offsets mean
an older historical window. Only correlation and offset are added: recurrence
forecast trajectories are never shifted. v_lag0 at O is already available
at O under the immutable prepared-input convention.

Axis 4 adds A(O)-A(O-24h) and A(O)-A(O-48h), where A is the prepared t045
meridian30 equatorial area. This source exists in prepared inputs but is not
a retained D feature. Exact UTC-hour lookups preserve missingness; there is
no new filling, interpolation, rate duplication or target access. These are
fixed visible-region changes, not tracked coronal-hole intrinsic growth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CH_SOURCE = 'suvi_t045_meridian30_equatorial_area'
WINDOW_HOURS = 72
MIN_PAIRS = 54
CORRELATION_TIE_ATOL = 1e-12
OFFSETS = (0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6)
RECURRENCE_COLUMNS = tuple(
    name for day in (26, 27, 28)
    for name in (f'rec{day}_past72_corr', f'rec{day}_past72_offset_h')
)
CH_COLUMNS = ('suvi_t045_meridian30_equatorial_d24',
              'suvi_t045_meridian30_equatorial_d48')


def describe(axis):
    """Return appended model columns and exact required source columns."""
    if axis == 3:
        columns, source = RECURRENCE_COLUMNS, 'v_lag0'
    elif axis == 4:
        columns, source = CH_COLUMNS, CH_SOURCE
    else:
        raise ValueError(f'Unsupported temporal feature axis: {axis}')
    return {'constant': columns, 'varying': (), 'sources': ('origin', source)}


def _hourly(frame, source):
    missing = {'origin', source} - set(frame.columns)
    if missing:
        raise ValueError(f'Missing temporal input columns: {sorted(missing)}')
    origins = pd.DatetimeIndex(pd.to_datetime(frame['origin'], utc=True),
                               name='origin_utc')
    if origins.isna().any() or not origins.equals(origins.floor('h')):
        raise ValueError('Origins must be finite exact UTC hours')
    values = frame[source].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError('Infinite temporal input values are invalid')
    series = pd.Series(values, index=origins, name=source)
    if series.groupby(level=0).nunique(dropna=False).gt(1).any():
        raise ValueError('Duplicate origins disagree on causal input values')
    series = series[~series.index.duplicated()].sort_index()
    if not series.empty:
        series = series.reindex(pd.date_range(series.index.min(),
                                             series.index.max(), freq='h',
                                             name='origin_utc'))
    return origins, series


def _recurrence(series):
    result = pd.DataFrame(np.nan, index=series.index,
                          columns=RECURRENCE_COLUMNS)
    for day in (26, 27, 28):
        best = np.full(len(series), np.nan)
        offsets = np.full(len(series), np.nan)
        for offset in OFFSETS:
            past = series.shift(24 * day + offset)
            corr = series.rolling(WINDOW_HOURS, min_periods=MIN_PAIRS).corr(past)
            # corr requires paired support and positive variance. Nonfinite
            # results (including constant windows) never become features.
            corr = corr.to_numpy()
            eligible = np.isfinite(corr)
            corr = np.clip(corr, -1., 1.)
            improve = eligible & (~np.isfinite(best) |
                                  (corr > best + CORRELATION_TIE_ATOL))
            best[improve] = corr[improve]
            offsets[improve] = offset
        result[f'rec{day}_past72_corr'] = best
        result[f'rec{day}_past72_offset_h'] = offsets
    return result


def build(frame, axis):
    """Return additions indexed by origin_utc, preserving input origin order.

    Pass the full origin history, not a parquet row-group or query-only slice.
    Only describe(axis)['sources'] are read; extra columns such as y are ignored.
    Duplicate matching origins are supported; conflicting values fail closed.
    """
    spec = describe(axis)
    origins, series = _hourly(frame, spec['sources'][1])
    if series.empty:
        return pd.DataFrame(index=origins, columns=spec['constant'], dtype=float)
    if axis == 3:
        result = _recurrence(series)
    else:
        result = pd.DataFrame({name: series - series.shift(hours)
                               for name, hours in zip(CH_COLUMNS, (24, 48))})
    return result.reindex(origins)
