"""Exact causal 120-hour source plan and chronological 16-hour clip coverage."""
from __future__ import annotations

import numpy as np
import pandas as pd

HOUR_NS = 3_600_000_000_000
NAT = np.iinfo(np.int64).min
STARTS = np.array([0, 15, 30, 45, 60, 75, 90, 104], dtype=np.int64)
CLIP_OFFSETS = STARTS[:, None] + np.arange(16)
QUALITY = ['missing_fraction', 'filled_fraction', 'mean_age_hours',
           'last_age_hours', 'valid_disk_fraction']
QUALITY_COLUMNS = ['window_' + q for q in QUALITY] + [
    f'clip{i}_' + q for i in range(8) for q in QUALITY]


def origins_ns(origins):
    arr = np.asarray(origins)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int64)
    else:
        arr = pd.DatetimeIndex(pd.to_datetime(origins, utc=True)).as_unit('ns').asi8
    if arr.ndim != 1 or not len(arr) or np.any(arr % HOUR_NS):
        raise ValueError('origins must be nonempty hourly timestamps in nanoseconds')
    if len(np.unique(arr)) != len(arr):
        raise ValueError('duplicate origins')
    return arr


def coverage_manifest():
    counts = np.bincount(CLIP_OFFSETS.ravel(), minlength=120)
    if not np.all(counts >= 1) or counts.size != 120:
        raise ValueError('incomplete hourly coverage')
    return {'requested_hourly_slots': 120, 'clip_starts': STARTS.tolist(),
            'clip_offsets': CLIP_OFFSETS.tolist(), 'slot_multiplicities': counts.tolist(),
            'unique_slots': 120, 'encoded_slots_per_origin': 128,
            'overlap_slots': np.flatnonzero(counts > 1).tolist(),
            'aggregation': 'chronological concatenation of eight 3840D clips'}


def plan_sources(inventory, origins):
    origins = origins_ns(origins)
    rows = inventory.loc[inventory.available & inventory.path.notna()].copy()
    rows['slot'] = pd.to_datetime(rows.slot, utc=True)
    rows['obs_end'] = pd.to_datetime(rows.obs_end, utc=True)
    rows = rows.sort_values('slot').reset_index(drop=True)
    slots = pd.DatetimeIndex(rows.slot).as_unit('ns').asi8
    ends = pd.DatetimeIndex(rows.obs_end).as_unit('ns').asi8
    if np.any(ends == NAT) or np.any(ends > slots) or rows.slot.duplicated().any():
        raise ValueError('invalid source observation time or duplicate slot')
    requested = origins[:, None] - np.arange(119, -1, -1)[None, :] * HOUR_NS
    ids = np.full(requested.shape, -1, np.int64)
    # Search older sources if a newer source has stale observation metadata.
    positions = np.searchsorted(slots, requested, side='right') - 1
    pending = positions >= 0
    while np.any(pending):
        rr, cc = np.where(pending)
        pos = positions[rr, cc]
        at = requested[rr, cc]
        viable = (slots[pos] >= at-6*HOUR_NS)
        good = viable & (ends[pos] <= at) & (ends[pos] >= at-6*HOUR_NS)
        ids[rr[good], cc[good]] = pos[good]
        positions[rr, cc] -= 1
        pending[rr, cc] = viable & ~good & (positions[rr, cc] >= 0)
    source_ends = np.full(ids.shape, NAT, np.int64)
    age = np.full(ids.shape, 7., np.float64)
    filled = np.zeros(ids.shape, bool)
    valid = ids >= 0
    source_ends[valid] = ends[ids[valid]]
    age[valid] = (requested[valid] - source_ends[valid]) / HOUR_NS
    filled[valid] = slots[ids[valid]] < requested[valid]
    if np.any(source_ends > requested) or np.any(requested > origins[:, None]):
        raise ValueError('future source or requested slot')
    return rows, ids, source_ends, age, filled


def deduplicate_clips(origins, ids, ends, age, filled):
    origins = origins_ns(origins)
    clip_ends = origins[:, None] - (119 - CLIP_OFFSETS[:, -1])[None, :] * HOUR_NS
    unique, first, inverse = np.unique(clip_ends.ravel(), return_index=True, return_inverse=True)
    origin_rows, clip_rows = np.unravel_index(first, clip_ends.shape)
    offsets = CLIP_OFFSETS[clip_rows]
    result = {'clip_ends': unique, 'origin_to_clip': inverse.reshape(-1, 8)}
    for name, values in [('source_ids', ids), ('source_end_ns', ends),
                         ('age_hours', age), ('filled', filled)]:
        result[name] = values[origin_rows[:, None], offsets]
        # A deduplicated end must mean exactly the same entire causal window.
        for clip in range(8):
            if not np.array_equal(result[name][result['origin_to_clip'][:, clip]], values[:, CLIP_OFFSETS[clip]]):
                raise ValueError('conflicting source plan for identical clip end')
    return result


def quality_summary(ids, age, filled, validity):
    return np.stack([(ids < 0).mean(-1), filled.mean(-1), age.mean(-1),
                     age[..., -1], validity.mean(-1)], axis=-1).astype(np.float32)


def all_quality(ids, age, filled, validity):
    return np.concatenate([quality_summary(ids, age, filled, validity)] + [
        quality_summary(ids[:, c], age[:, c], filled[:, c], validity[:, c])
        for c in CLIP_OFFSETS], axis=1)
