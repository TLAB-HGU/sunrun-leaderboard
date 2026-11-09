"""Additive, causal SUVI strip reprojection from coherent raw frames."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / 'data-v2/input/df.parquet'
SPECS = ((350, 'logmed'), (450, 'logmed'), (550, 'dark'), (750, 'logmed'))
NAMES = tuple(f'strip_v{v}_{s}_reprojected' for v, s in SPECS)
GRID = tuple(f'suvi_lon{i:02d}_lat1_{s}' for s in ('log_median', 'dark_fraction') for i in range(12))
EDGES = -90.0 + 15.0 * np.arange(13)


def describe():
    return {'constant': [], 'varying': list(NAMES),
            'raw_prerequisites': ['origin', 'img_age_h', *GRID],
            'query_prerequisites': ['origin', 'h'],
            'publication_time_verified': False}


def prepare_history(raw):
    """Keep whole actual frames; reject conflicting duplicates and invalid ages."""
    data = raw.loc[:, ['origin', 'img_age_h', *GRID]].copy()
    data['origin'] = pd.to_datetime(data['origin'], utc=True)
    if data['origin'].isna().any():
        raise ValueError('Missing frame origin')
    if data.groupby('origin').nunique(dropna=False).gt(1).any().any():
        raise ValueError('Conflicting raw frame duplicates')
    data = data.drop_duplicates('origin').sort_values('origin')
    ages = data['img_age_h'].to_numpy(float)
    if np.isinf(ages).any() or (ages[np.isfinite(ages)] < 0).any():
        raise ValueError('Frame age must be nonnegative or missing')
    data = data.loc[np.isfinite(ages)].copy()
    data['observed_at'] = data['origin'] - pd.to_timedelta(data['img_age_h'], unit='h')
    if not data['observed_at'].is_monotonic_increasing:
        raise ValueError('Frame observation times must be monotonic')
    return data.reset_index(drop=True)


def load_history(path=RAW_PATH):
    return prepare_history(pd.read_parquet(path, columns=['origin', 'img_age_h', *GRID],
                                          filters=[('h', '==', 1)]))


def build(frame, history):
    """Project last available whole frame at each origin for its forecast lead.

    History must be the output of prepare_history/load_history. Availability is
    bounded by raw origin, not merely observation time; no future row is used.
    Ordinary sums intentionally reproduce the original NaN propagation.
    """
    origins = pd.DatetimeIndex(pd.to_datetime(frame['origin'], utc=True))
    horizons = frame['h'].to_numpy(float)
    if origins.isna().any() or not np.isfinite(horizons).all() or (horizons < 1).any():
        raise ValueError('Queries require finite positive horizons and valid origins')
    output = np.full((len(frame), len(NAMES)), np.nan)
    if history.empty or frame.empty:
        return pd.DataFrame(output, index=frame.index, columns=NAMES)
    available = pd.DatetimeIndex(history['origin']).asi8
    if not pd.Index(available).is_monotonic_increasing or pd.Index(available).has_duplicates:
        raise ValueError('History must have unique sorted origins')
    source = np.searchsorted(available, origins.asi8, side='right') - 1
    observed = pd.DatetimeIndex(history['observed_at']).asi8
    grids = {s: history[[f'suvi_lon{i:02d}_lat1_{suffix}' for i in range(12)]].to_numpy(float)
             for s, suffix in [('logmed', 'log_median'), ('dark', 'dark_fraction')]}
    # Bound temporary weights for full 72-horizon input tables.
    for start in range(0, len(frame), 65536):
        rows = np.arange(start, min(start + 65536, len(frame)))
        rows = rows[source[rows] >= 0]
        idx = source[rows]
        age = (origins.asi8[rows] - observed[idx]) / 3.6e12
        if (age < 0).any():
            raise ValueError('Future observation in frame history')
        for k, (speed, stat) in enumerate(SPECS):
            lam = 0.55 * (147858620.7 / speed / 3600.0 - (horizons[rows] + age))
            weights = np.clip(np.minimum(lam[:, None] + 5.0, EDGES[1:])
                              - np.maximum(lam[:, None] - 5.0, EDGES[:-1]), 0, None) / 10.0
            total = weights.sum(axis=1)
            numerator = (weights * grids[stat][idx]).sum(axis=1)
            output[rows, k] = np.where(total > 0, numerator / np.where(total > 0, total, 1), np.nan)
    return pd.DataFrame(output, index=frame.index, columns=NAMES)
