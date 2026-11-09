"""Target-free ACE minute support and speed–density coupling features.

Both axes use the 56 exact minute slots in (origin-60min, origin-4min].
The four-minute release delay is a frozen research assumption, not proof of
historical receipt. Missing/invalid slots remain missing; no filling occurs.
Axis 1 uses status-zero, finite positive speed. Its last-valid age is confined
to this window. Axis 2 additionally requires finite positive density at the
same minute: correlation needs 30 paired samples, and population covariance
of adjacent-minute changes needs 20 valid consecutive pairs. These thresholds
and windows are fixed before scores. Neither builder reads targets.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MINUTE_NS = 60_000_000_000
SLOTS = 56
FEATURES = {
    1: ['ace_valid_fraction60', 'ace_max_gap60_min', 'ace_last_valid_age60_min'],
    2: ['ace_speed_density_corr60', 'ace_dspeed_ddensity_cov60'],
}


def describe(axis):
    return {'constant': FEATURES[int(axis)].copy(), 'varying': []}


def _index(values):
    index = pd.DatetimeIndex(pd.to_datetime(values, utc=True)).as_unit('ns')
    if index.hasnans or np.any(index.asi8 % MINUTE_NS):
        raise ValueError('Origins and ACE samples must be valid minute-aligned timestamps')
    return index


def _samples(minutes, axis):
    columns = ['S', 'spd'] + (['dens'] if axis == 2 else [])
    frame = minutes[columns].copy()
    frame.index = _index(frame.index)
    # A retrospective conflicting copy cannot be resolved by choosing latest.
    for _, group in frame[frame.index.duplicated(keep=False)].groupby(level=0):
        if len(group.drop_duplicates()) > 1:
            raise ValueError('Conflicting ACE sample versions')
    return frame[~frame.index.duplicated()].sort_index()


def load_minutes(source_root, start, end):
    """Read cached days intersecting [start, end]; missing days stay missing.

    Callers include their one-hour lookback in start. The existing NOAA parser
    retains status and missing markers. No download or source mutation occurs.
    """
    path = ROOT / 'research/ace_1m_to_hourly_20261108/experiment.py'
    spec = spec_from_file_location('_axes4_ace_text_parser', path)
    parser = module_from_spec(spec)
    spec.loader.exec_module(parser)
    start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
    if pd.isna(start) or pd.isna(end) or end < start:
        raise ValueError('Invalid ACE cache interval')
    frames = []
    for day in pd.date_range(start.floor('D'), end.floor('D')):
        source = Path(source_root) / f'{day:%Y%m%d}.txt'
        if source.exists():
            frame, diagnostics = parser.parse_text('swepam', source.read_text())
            if diagnostics['bad_lines']:
                raise ValueError(f'Malformed ACE source: {source}')
            if len(frame):
                frames.append(frame[['S', 'spd', 'dens']])
    if not frames:
        return pd.DataFrame(columns=['S', 'spd', 'dens'],
                            index=pd.DatetimeIndex([], tz='UTC', name='timestamp_utc'))
    frame = _samples(pd.concat(frames), 2)
    return frame.loc[(frame.index >= start) & (frame.index <= end)]


def build(origins, minutes, axis):
    """Return fixed-axis float features in original origin order (duplicates kept)."""
    axis = int(axis)
    names = describe(axis)['constant']
    index = _index(origins).rename('origin_utc')
    frame = _samples(minutes, axis)
    times = frame.index.asi8
    values = frame.to_numpy(dtype=float)
    result = np.full((len(index), len(names)), np.nan)
    for row, origin in enumerate(index.asi8):
        first = origin - 59 * MINUTE_NS
        left = np.searchsorted(times, first, side='left')
        right = np.searchsorted(times, origin - 4 * MINUTE_NS, side='right')
        window = np.full((SLOTS, values.shape[1]), np.nan)
        positions = (times[left:right] - first) // MINUTE_NS
        window[positions] = values[left:right]
        speed = window[:, 1]
        valid = (window[:, 0] == 0) & np.isfinite(speed) & (speed > 0)
        if axis == 1:
            valid_slots = np.flatnonzero(valid)
            # Sentinel boundaries include missing leading and trailing runs.
            longest_gap = np.diff(np.r_[-1, valid_slots, SLOTS]).max() - 1
            age = 59 - valid_slots[-1] if len(valid_slots) else np.nan
            result[row] = [valid.sum() / SLOTS, longest_gap, age]
            continue
        density = window[:, 2]
        valid &= np.isfinite(density) & (density > 0)
        if valid.sum() >= 30:
            x, y = speed[valid], density[valid]
            x, y = x - x.mean(), y - y.mean()
            denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
            if denominator > 0:
                result[row, 0] = np.clip(np.dot(x, y) / denominator, -1., 1.)
        adjacent = valid[:-1] & valid[1:]
        if adjacent.sum() >= 20:
            # Difference the fixed grid first; never bridge an absent minute.
            dx, dy = np.diff(speed)[adjacent], np.diff(density)[adjacent]
            result[row, 1] = np.mean((dx - dx.mean()) * (dy - dy.mean()))
    return pd.DataFrame(result, index=index, columns=names)
