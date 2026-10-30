"""Causal Fe195 clips with physical radiance and fixed solar coordinates.

Historical metadata describe observations, not guaranteed historical arrival.
Only rows with cutoff_applied certify the collector's live arrival cutoff.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from experiments.xgboost_suvi_fusion.features import read_suvi_frame

DEFAULT_METADATA = Path('/home/t-lab01/.local/state/sundb/stage/suvi_hourly')
DEFAULT_IMAGES = Path('/home/t-lab01/sunrun/sun-img/data')


def utc(value):
    return pd.to_datetime(value, utc=True)


def inventory_sources(metadata_root=DEFAULT_METADATA, images_root=DEFAULT_IMAGES,
                      start='2022-01-01', end='2026-05-31T23:59:59Z'):
    """Inventory actual local selected NetCDF twins, retaining explicit gaps.

    Reads only metadata partitions through May 2026. No official scoring truth
    is read. Missing raw files are recorded, never inferred from derived CH data.
    """
    start, end = utc(start), utc(end)
    if end >= utc('2026-06-01'):
        raise ValueError('inventory must end before official June 2026 period')
    parts = sorted(p for p in Path(metadata_root).glob('year=*/*.parquet')
                   if start.year <= int(p.parent.name.split('=')[1]) <= end.year)
    if not parts:
        raise FileNotFoundError(f'no SUVI metadata partitions: {metadata_root}')
    frame = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    frame['slot'], frame['obs_end'] = utc(frame.slot), utc(frame.obs_end)
    frame = frame.loc[frame.slot.between(start, end) & frame.sat.str.upper().eq('G18')
                      & frame.channel.eq('Fe195')].sort_values('slot').copy()
    if frame.slot.duplicated().any():
        raise ValueError('duplicate metadata slots')
    paths, sizes, mtimes = [], [], []
    root = Path(images_root)
    for row in frame.itertuples():
        path = None
        if row.status == 'ok' and pd.notna(row.obs_end):
            if row.obs_end > row.slot:
                raise ValueError('observation end exceeds source slot')
            filename = str(row.filename).replace('.fits', '.nc')
            candidate = root / 'g18' / row.obs_end.strftime('%Y/%j') / filename
            if candidate.is_file():
                path = candidate.resolve()
        stat = path.stat() if path else None
        paths.append(str(path) if path else None)
        sizes.append(stat.st_size if stat else 0)
        mtimes.append(stat.st_mtime_ns if stat else 0)
    frame['raw_path'], frame['size_bytes'], frame['mtime_ns'] = paths, sizes, mtimes
    frame['raw_available'] = frame.raw_path.notna()
    invalid_geometry = np.zeros(len(frame), dtype=bool)
    for column in ('rsun', 'crpix1', 'crpix2', 'crota'):
        if column in frame:
            values = pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype=float)
            invalid_geometry |= ~np.isfinite(values)
            if column == 'rsun':
                invalid_geometry |= values <= 0
    frame['invalid_geometry'] = invalid_geometry
    frame['path'] = frame.raw_path.where(~frame.invalid_geometry, None)
    frame['available'] = frame.path.notna()
    frame.attrs['arrival_limitation'] = ('Historical obs_end <= slot does not prove live '
        'arrival; cutoff_applied indicates collector live LastModified enforcement.')
    return frame.reset_index(drop=True)


@dataclass(frozen=True)
class SourceIndex:
    slots: pd.DatetimeIndex
    ends: pd.DatetimeIndex
    paths: np.ndarray


def prepare_inventory(inventory):
    """Prepare once for O(log N) causal lookup per requested frame."""
    rows = inventory.loc[inventory.path.notna()].copy()
    rows['slot'], rows['obs_end'] = utc(rows.slot), utc(rows.obs_end)
    rows = rows.sort_values(['slot', 'obs_end'])
    if rows.obs_end.isna().any() or (rows.obs_end > rows.slot).any():
        raise ValueError('invalid observation end in source inventory')
    return SourceIndex(pd.DatetimeIndex(rows.slot), pd.DatetimeIndex(rows.obs_end),
                       rows.path.to_numpy())


def select_clip(inventory, origin, mode='video', max_age_hours=6):
    """Select trailing 16 x 3h slots, or repeat one last frame 16 times.

    Every fallback is bounded by actual observation age, not merely slot age.
    Availability follows the selected hourly slot as well as observation end.
    Pass prepare_inventory's result to reuse its sorted index across origins.
    """
    origin = utc(origin)
    if mode not in ('video', 'single') or max_age_hours < 0:
        raise ValueError('invalid clip mode or maximum frame age')
    index = inventory if isinstance(inventory, SourceIndex) else prepare_inventory(inventory)
    requested = ([origin] * 16 if mode == 'single' else
                 pd.date_range(origin-pd.Timedelta(hours=45), origin, periods=16))
    records = []
    for at in requested:
        pos = int(index.slots.searchsorted(at, side='right'))-1
        earliest = at-pd.Timedelta(hours=max_age_hours)
        while pos >= 0 and index.slots[pos] >= earliest:
            if earliest <= index.ends[pos] <= at:
                break
            pos -= 1
        found = pos >= 0 and index.slots[pos] >= earliest and earliest <= index.ends[pos] <= at
        records.append({'requested_time': at, 'path': index.paths[pos] if found else None,
                        'obs_end': index.ends[pos] if found else pd.NaT,
                        'source_slot': index.slots[pos] if found else pd.NaT,
                        'missing': not found,
                        'filled': found and index.slots[pos] < at,
                        'age_hours': (at-index.ends[pos]).total_seconds()/3600 if found
                                     else float(max_age_hours+1)})
    return pd.DataFrame(records)


def canonical_radiance(rad, dqf, header, size=384):
    """Map solar coordinates to a north-up square; radius is 40% of width.

    Uses the existing reader's FITS 1-based CRPIX, pixel RSUN and PC/CROTA
    convention. The 2.5-R-wide canvas retains the full disk without cropping.
    SOLAR_B0 is recorded; no latitude deprojection distorts the visible disk.
    """
    rad, dqf = np.asarray(rad), np.asarray(dqf)
    if rad.ndim != 2 or rad.shape != dqf.shape:
        raise ValueError('RAD/DQF must have equal 2-D shape')
    h = {str(k).upper(): v for k, v in header.items()}
    def num(key, default=None):
        value = h.get(key, default)
        if value is None:
            raise ValueError(f'missing geometry {key}')
        return float(np.asarray(value).item())
    radius = num('RSUN')
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('RSUN must be finite and positive')
    angle = np.deg2rad(num('CROTA', 0))
    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    if all(k in h for k in ('PC1_1', 'PC1_2', 'PC2_1', 'PC2_2')):
        matrix = np.array([[num('PC1_1'), num('PC1_2')], [num('PC2_1'), num('PC2_2')]])
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-8:
        raise ValueError('invalid orientation matrix')
    yy, xx = np.indices((size, size), dtype=np.float64)
    x, y = (xx-(size-1)/2)/(.4*size), ((size-1)/2-yy)/(.4*size)
    src = np.linalg.solve(matrix, np.stack((x.ravel(), y.ravel()))*radius)
    coords = [src[1].reshape(size, size)+num('CRPIX2')-1,
              src[0].reshape(size, size)+num('CRPIX1')-1]
    valid = (dqf == 0) & np.isfinite(rad) & (rad >= 0)
    weight = map_coordinates(valid.astype(float), coords, order=1, mode='constant', cval=0.)
    values = map_coordinates(np.where(valid, rad, 0.).astype(float), coords,
                             order=1, mode='constant', cval=0.)
    good = (weight >= .999) & (x*x+y*y <= 1.)
    values = np.where(good, values/np.maximum(weight, 1e-12), 0.)
    return values.astype(np.float32), good, {'center': [(size-1)/2]*2,
        'radius': .4*size, 'north_up': True, 'solar_b0': num('SOLAR_B0', 0),
        'valid_disk_fraction': float(good.sum()/max(1, (x*x+y*y <= 1.).sum()))}


@dataclass(frozen=True)
class IntensityTransform:
    log_low: float
    log_high: float
    train_cutoff: str
    fit_paths: tuple[str, ...] = ()

    def to_dict(self):
        return asdict(self)

    def apply(self, radiance, valid):
        z = (np.log1p(np.maximum(radiance, 0))-self.log_low)/max(self.log_high-self.log_low, 1e-6)
        return np.where(valid, 2*np.clip(z, 0, 1)-1, 0).astype(np.float32)


def fit_intensity(inventory, train_cutoff, max_frames=64):
    """Fit one fixed log-radiance q1/q99 transform on training images only."""
    cutoff = utc(train_cutoff)
    eligible = inventory.loc[(utc(inventory.slot) <= cutoff) & (utc(inventory.obs_end) <= cutoff)
                             & inventory.path.notna()].sort_values('slot').drop_duplicates('path')
    if eligible.empty or max_frames < 1:
        raise ValueError('no training images for intensity transform')
    positions = np.unique(np.linspace(0, len(eligible)-1, min(max_frames, len(eligible))).astype(int))
    paths = tuple(eligible.iloc[positions].path)
    samples = []
    for path in paths:
        rad, valid, _ = canonical_radiance(*read_suvi_frame(path), size=96)
        samples.append(np.log1p(rad[valid]))
    values = np.concatenate(samples)
    if not len(values):
        raise ValueError('training images have no valid radiance pixels')
    low, high = np.quantile(values, [.01, .99])
    return IntensityTransform(float(low), float(high), cutoff.isoformat(), paths)


def preprocess_frame(path, transform, size=384):
    rad, valid, geometry = canonical_radiance(*read_suvi_frame(path), size=size)
    image = np.repeat(transform.apply(rad, valid)[None], 3, axis=0)
    return image, valid, geometry


def build_clip(selection, transform, size=384):
    if len(selection) != 16:
        raise ValueError('V-JEPA clip requires 16 frames')
    images, valid_fractions, cache = [], [], {}
    for row in selection.itertuples():
        if row.path is None or pd.isna(row.path):
            image, fraction = np.zeros((3, size, size), dtype=np.float32), 0.
        else:
            if utc(row.obs_end) > utc(row.requested_time):
                raise ValueError('future frame in clip')
            if row.path not in cache:
                cache[row.path] = preprocess_frame(row.path, transform, size)
            image, _, geometry = cache[row.path]
            fraction = geometry['valid_disk_fraction']
        images.append(image)
        valid_fractions.append(fraction)
    stats = {'missing_fraction': float(selection.missing.mean()),
             'filled_fraction': float(selection.filled.mean()),
             'mean_age_hours': float(selection.age_hours.mean()),
             'last_age_hours': float(selection.age_hours.iloc[-1]),
             'valid_disk_fraction': float(np.mean(valid_fractions))}
    return np.stack(images, axis=1), stats
