"""df.parquet -> causally filled model table -> (origin x 72-lead) training and query grids."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .common import CUTOFFS, H, file_identity

METADATA = ('set', 'origin', 'h', 'target_time', 'period', 'fit_F1', 'fit_F2', 'fit_F3', 'y')
PURGE_HOURS = 144


def timeline(origin, horizon):
    """Representative rows, original-row -> unique (origin, h) indices, and h boundaries."""
    if len(origin) == 0:
        raise ValueError('The source table is empty')
    if np.isnat(origin).any() or pd.isna(horizon).any():
        raise ValueError('origin and h must not contain missing values')
    order = np.lexsort((origin, horizon))
    sorted_origin, sorted_h = origin[order], horizon[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = (sorted_origin[1:] != sorted_origin[:-1]) | (sorted_h[1:] != sorted_h[:-1])
    representatives = order[first]
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.cumsum(first, dtype=np.int64) - 1
    unique_h = horizon[representatives]
    boundaries = np.r_[0, np.flatnonzero(unique_h[1:] != unique_h[:-1]) + 1, len(unique_h)]
    return representatives, inverse, boundaries


def fill_feature(values, representatives, inverse, boundaries, name):
    """Validate repeated inputs, then forward-fill independently along each h timeline."""
    unique = values[representatives]
    repeated = unique[inverse]
    equal = (values == repeated) | (np.isnan(values) & np.isnan(repeated))
    if not equal.all():
        bad = int(np.flatnonzero(~equal)[0])
        raise ValueError(f'Conflicting duplicate (origin, h) values in {name!r}, source row {bad}')
    del repeated, equal
    before = int(np.isnan(values).sum())
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        segment = unique[start:stop]
        valid = ~np.isnan(segment)
        previous = np.maximum.accumulate(np.where(valid, np.arange(len(segment)), -1))
        take = previous >= 0
        segment[take] = segment[previous[take]]
    filled = unique[inverse]
    after = int(np.isnan(filled).sum())
    return filled, dict(missing_before=before, missing_after=after, filled=before - after)


def fill_training_labels(metadata):
    """y with training rows filled from the last observed target timestamp; query y unchanged."""
    time_axis = pd.to_datetime(metadata.target_time, utc=True)
    if time_axis.isna().any():
        raise ValueError('target_time must be present')
    timeline_frame = pd.DataFrame({'time': time_axis, 'y': metadata.y.to_numpy()})
    if timeline_frame.groupby('time').y.nunique(dropna=False).gt(1).any():
        raise ValueError('The same target timestamp has inconsistent source labels')
    timeline_frame = timeline_frame.drop_duplicates('time').sort_values('time').set_index('time')
    filled = timeline_frame.y.ffill()
    labels = metadata.y.to_numpy(copy=True)
    training = metadata['set'].eq('train').to_numpy()
    labels[training] = filled.reindex(pd.DatetimeIndex(time_axis[training])).to_numpy()
    original = metadata.y.to_numpy()
    if not np.array_equal(labels[~training], original[~training], equal_nan=True):
        raise RuntimeError('Query truth was modified')
    if not np.array_equal(labels[np.isfinite(original)], original[np.isfinite(original)], equal_nan=True):
        raise RuntimeError('Observed labels were modified')
    return labels


def load_table(path, features, logs, keep_observed=False):
    """The model table: metadata plus the experiment's features, causally filled. keep_observed adds the
    unfilled target as y_observed (for fits that must see only observed labels)."""
    began = time.perf_counter()
    features = list(features)
    table = pd.read_parquet(path, columns=[*METADATA, *features])
    origin = pd.DatetimeIndex(table['origin']).asi8.view('datetime64[ns]')
    representatives, inverse, boundaries = timeline(origin, table['h'].to_numpy())
    stats = {}
    for name in features:
        table[name], stats[name] = fill_feature(table[name].to_numpy(), representatives, inverse,
                                                boundaries, name)
    training = table['set'].eq('train').to_numpy()
    missing = int(table['y'][training].isna().sum())
    if keep_observed:
        table['y_observed'] = table['y'].to_numpy(copy=True)
    table['y'] = fill_training_labels(table)
    logs('data', 'table_ready', source=file_identity(path), rows=len(table),
         unique_origin_h_keys=len(representatives), features=features,
         feature_ffill=stats, training_y_missing_before=missing,
         training_y_missing_after=int(np.isnan(table['y'].to_numpy()[training]).sum()),
         policy='feature ffill by h over sorted origin (train+query union, leading NaNs kept); '
                'training y ffill on sorted target_time; query y unchanged',
         seconds=round(time.perf_counter() - began, 3))
    return table


def load_grid(table, period, training, constant, varying):
    """Constant (origin) and varying (origin x 72 leads) arrays for training or query origins."""
    if training:
        mask = table[f'fit_{period}'].to_numpy()
    else:
        mask = (table['set'].eq('query') & table['period'].eq(period)).to_numpy()
    selected = table[mask]
    constant_rows = selected.loc[selected['h'].eq(1).to_numpy(), ['origin', *constant]].sort_values(
        'origin', kind='stable')
    origins = pd.DatetimeIndex(pd.to_datetime(constant_rows.origin, utc=True))
    if origins.has_duplicates:
        raise ValueError('duplicate input origins')
    columns = ['origin', 'h', *varying] + (['y'] if training else [])
    rows = selected[columns].sort_values(['origin', 'h'], kind='stable')
    if len(rows) != len(origins) * H or not np.array_equal(
            rows.h.to_numpy().reshape(-1, H), np.broadcast_to(np.arange(1, H + 1), (len(origins), H))):
        raise ValueError('incomplete origin/horizon grid')
    if not origins.equals(pd.DatetimeIndex(pd.to_datetime(rows.origin.iloc[::H], utc=True))):
        raise ValueError('constant and varying grids disagree')
    if training and not (origins + pd.Timedelta(hours=PURGE_HOURS)
                         <= pd.Timestamp(CUTOFFS[period], tz='UTC')).all():
        raise ValueError('training cutoff violated')
    grid = {'origins': origins, 'C': constant_rows[list(constant)].to_numpy(np.float32),
            'E': rows[list(varying)].to_numpy(np.float32).reshape(len(origins), H, len(varying))}
    if np.isinf(grid['C']).any() or np.isinf(grid['E']).any():
        raise ValueError('Infinite feature values are not supported')
    return grid, (rows.y.to_numpy(np.float64).reshape(len(origins), H) if training else None)


def subset(grid, indices):
    return {key: value[indices] for key, value in grid.items()}
