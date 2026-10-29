"""Hourly, availability-aware feat30 history with frozen forecast grids.

The historical channel for a horizon-varying feature is its h=1 value at
that historical origin. No future observation is used as a decoder input.
"""
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / 'analysis/forecast_strategies_e2000/outputs/cache'
PARENT = ROOT / 'analysis/e2000fs_dirmo_24_feat30_lgbm/logs/full_01/manifest.json'
ARCHIVE = ROOT / 'analysis/forecast_strategies_e2000/publication/worktree'
METADATA = ROOT / 'research/e0472-multioutput-s42-20261026/relocated-v1/corrected_metadata.parquet'
HOUR_NS = 3_600_000_000_000
LOOKBACK = 120


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def ns(origins):
    return pd.DatetimeIndex(origins).as_unit('ns').asi8


def causal_prefill(values):
    """Missing leading values remain missing for training-only imputation."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or np.isinf(values).any():
        raise ValueError('Expected a finite-or-NaN two dimensional feature table')
    return pd.DataFrame(values).ffill().to_numpy(np.float32)


def window_positions(hourly_origins, origins):
    hourly_origins = np.asarray(hourly_origins, dtype=np.int64)
    if len(hourly_origins) < LOOKBACK or not np.all(np.diff(hourly_origins) == HOUR_NS):
        raise ValueError('History index must be a complete hourly timeline')
    oo = ns(origins)
    positions = np.searchsorted(hourly_origins, oo)
    if (positions < LOOKBACK - 1).any() or (positions >= len(hourly_origins)).any():
        raise ValueError('Missing complete 120-hour lookback')
    if not np.array_equal(hourly_origins[positions], oo):
        raise ValueError('Origin is not an exact hour in the history timeline')
    return positions


def future_features(bundle, names, selected):
    varying = [name for name in selected if name in names['varying']]
    indices = [names['varying'].index(name) for name in varying]
    return np.asarray(bundle['E'][:, :, indices], dtype=np.float32), varying


@contextmanager
def production_helpers():
    """Load preserved source helpers; only process-local source paths change."""
    lb = ROOT / 'leaderboard'
    # Several workspace source paths are broken archival symlinks. The release
    # retains readable source, so add those concrete directories explicitly.
    for rel in ('d147_e0421_lean', 'd148_enlil_live', 'd143_axisA_analog/a2b',
                'd1_longitude_strips', 'headroom_audit'):
        path = str(ARCHIVE / 'experiments' / rel)
        if path not in sys.path:
            sys.path.append(path)
    for rel in ('cumulative_lane12_20261024', 'cumulative_20261024_lane9'):
        path = str(lb / 'experiments' / rel)
        if path not in sys.path:
            sys.path.insert(0, path)
    if str(lb) not in sys.path:
        sys.path.insert(0, str(lb))
    prepare = importlib.import_module('experiments.e2000.prepare')
    hr = importlib.import_module('headroom')
    ia = importlib.import_module('info_ablation')
    st = importlib.import_module('strips')
    le = importlib.import_module('lean_enlil')
    ch = importlib.import_module('prepare_base111_v2')
    root = prepare.source_root()
    updates = [(hr, 'LB', root.parent), (hr, 'OUT', root / 'headroom-audit'),
               (ia, 'LB', root.parent), (ia, 'OUT', root / 'headroom-audit'),
               (st, 'SUVI_CACHE', root / 'suvi-fusion/suvi-hourly-v2.parquet'),
               (st, 'SUNIMG', ROOT / 'data-v2/suvi'), (st, 'resolve_nc', prepare.relocated_nc),
               (le, 'LIVE', root / 'donki-live'), (ch, 'CH', root / 'ch-v1/ch-hourly.parquet')]
    saved = [(module, name, getattr(module, name)) for module, name, _ in updates]
    try:
        for module, name, value in updates:
            setattr(module, name, value)
        yield root, prepare, hr, ia, st, le, ch
    finally:
        for module, name, value in reversed(saved):
            setattr(module, name, value)


def select_strip_slots(origins, meta, block, resolve_nc):
    """Vectorized equivalent of the production newest-eligible-slot rule."""
    slots = ns(meta['slot'])
    obs = ns(meta['obs_end'])
    s3 = ns(meta['s3_modified'])
    finite = block.reindex(pd.DatetimeIndex(meta['slot'])).notna().all(axis=1).to_numpy()
    paths = [Path(resolve_nc(str(p))) if pd.notna(p) and str(p) else None for p in meta.nc_path]
    present = np.array([p is not None and p.is_file() and p.stat().st_size > 0 for p in paths])
    good = (meta.status.to_numpy() == 'ok') & finite & present & meta.obs_end.notna().to_numpy() & meta.s3_modified.notna().to_numpy()
    ids = np.flatnonzero(good)
    out = np.full(len(origins), -1, dtype=np.int64)
    if len(ids):
        available = np.maximum(slots[ids] + HOUR_NS, s3[ids])
        order = np.argsort(available, kind='stable')
        best = np.maximum.accumulate(ids[order])
        j = np.searchsorted(available[order], ns(origins), side='right') - 1
        ok = j >= 0
        out[ok] = best[j[ok]]
    age = np.full(len(origins), np.nan)
    ok = out >= 0
    age[ok] = (ns(origins)[ok] - obs[out[ok]]) / HOUR_NS
    return out, age


def strip_h1(origins, meta, block, selected, st):
    """Production strip geometry at h=1, avoiding 71 discarded horizons."""
    ids, age = select_strip_slots(origins, meta, block, st.resolve_nc)
    result = pd.DataFrame(index=origins)
    result['img_age_h'] = age.astype(np.float32)
    good = (ids >= 0) & (age <= st.STALE_H)
    # Production indexes its cache rows using selected metadata indices. Match
    # by slot explicitly so missing metadata/cache slots cannot shift values.
    chosen = pd.DatetimeIndex(meta.slot.iloc[np.maximum(ids, 0)])
    rows = block.reindex(chosen)
    edges = -90. + st.CELL_W * np.arange(13)
    for name in selected:
        if not name.startswith('strip_'):
            continue
        _, speed, stat = name.split('_')
        lam = st.lam(1, int(speed[1:]), age)
        weights = np.clip(np.minimum(lam[:, None] + st.STRIP_HALF, edges[1:])
                          - np.maximum(lam[:, None] - st.STRIP_HALF, edges[:-1]), 0, None) / (2 * st.STRIP_HALF)
        denom = weights.sum(axis=1)
        cover_cols = sorted([c for c in rows if c.endswith(st.STAT_SUFFIX['cover'])], key=lambda c: int(c[3:5]))
        value_cols = sorted([c for c in rows if c.endswith(st.STAT_SUFFIX[stat])], key=lambda c: int(c[3:5]))
        cover = (weights * rows[cover_cols].to_numpy()).sum(axis=1) / np.maximum(denom, 1e-12)
        value = (weights * rows[value_cols].to_numpy()).sum(axis=1) / np.maximum(denom, 1e-12)
        value[~(good & (denom > 0) & (cover >= st.COVER_MIN))] = np.nan
        result[name] = value.astype(np.float32)
    return result


def hourly_features(origins, selected, helpers):
    """Rebuild selected feature families at each historical origin, lag=0."""
    root, prepare, hr, ia, st, le, ch = helpers
    raw_speed = pd.read_parquet(root / 'ch-v1/ace.parquet')
    raw_speed['timestamp_utc'] = pd.to_datetime(raw_speed.timestamp_utc, utc=True)
    speed = raw_speed.drop_duplicates('timestamp_utc', keep='last').set_index('timestamp_utc').filled_speed_kms.sort_index()
    speed = speed.reindex(pd.date_range(speed.index.min(), speed.index.max(), freq='h', tz='UTC'))
    print(f'Hourly features: ACE and recurrence, {len(origins)} hours', flush=True)
    result = hr.ace_features(origins).join(ia.speed_features(origins, speed))
    for days in (26, 27, 28):
        result[f'rec{days}'] = speed.reindex(origins + pd.Timedelta(hours=1) - pd.Timedelta(days=days)).to_numpy()
    result['rec_mean'] = result[['rec26', 'rec27', 'rec28']].mean(axis=1)
    print('Hourly features: origin-available ENLIL forecasts', flush=True)
    result = result.join(ia.enlil_features(origins, le.enlil_runs()))
    meta = pd.read_parquet(METADATA).sort_values('slot').reset_index(drop=True)
    for name in ('slot', 'obs_end', 's3_modified'):
        meta[name] = pd.to_datetime(meta[name], utc=True)
    ch_meta = meta.loc[meta.status.eq('ok') & meta.obs_end.notna() & meta.s3_modified.notna()].drop_duplicates('slot').copy()
    ch_meta['available_at'] = pd.concat([ch_meta.slot, ch_meta.s3_modified, ch_meta.obs_end + pd.Timedelta(hours=1)], axis=1).max(axis=1)
    print('Hourly features: availability-filtered CH and h1 strips', flush=True)
    ch_cols = [name for name in selected if name.startswith('ch_')]
    values, _ = ch.select_filtered(ch.load_ch30(ch_cols), ch_meta, origins, tolerance_hours=6.)
    result = result.join(values)
    cache = pd.read_parquet(st.SUVI_CACHE)
    cache.index = pd.to_datetime(cache.index, utc=True)
    block = cache[sorted(c for c in cache if '_lat1_' in c)]
    result = result.join(strip_h1(origins, meta, block, selected, st))
    return result[selected].to_numpy(np.float32)


def _smoke_indices(origins):
    fit = np.flatnonzero(origins + pd.Timedelta(hours=72) < pd.Timestamp('2025-12-04', tz='UTC'))[-64:]
    val = np.flatnonzero((origins >= pd.Timestamp('2025-12-04', tz='UTC')) & (origins <= pd.Timestamp('2026-02-22T23:00:00Z')))[:32]
    return np.concatenate([fit, val])


def _frozen_inputs(smoke):
    parent = json.loads(PARENT.read_text())
    names = json.loads((CACHE / 'names.json').read_text())
    selected = parent['selected']
    if len(selected) != 30 or len(set(selected)) != 30:
        raise ValueError('Frozen feat30 identity invalid')
    for path in CACHE.iterdir():
        if path.is_file() and digest(path) != parent['hashes'].get(str(path)):
            raise ValueError(f'Frozen parent input changed: {path}')
    speed = pd.read_parquet(CACHE / 'speed.parquet')['speed']
    stages = {}
    for stage, filename in (('train', 'train'), ('f1', 'f1'), ('final', 'official')):
        with np.load(CACHE / f'{filename}.npz') as bundle:
            origins = pd.DatetimeIndex(bundle['origins'], tz='UTC')
            future, varying = future_features(bundle, names, selected)
            baseline = np.empty((len(origins), 30), dtype=np.float32)
            for j, name in enumerate(selected):
                baseline[:, j] = bundle['C'][:, names['constant'].index(name)] if name in names['constant'] else bundle['E'][:, 0, names['varying'].index(name)]
        ix = _smoke_indices(origins) if smoke and stage == 'train' else np.arange(min(16, len(origins))) if smoke else np.arange(len(origins))
        origins, future, baseline = origins[ix], future[ix], baseline[ix]
        times = origins.values[:, None] + np.arange(1, 73).astype('timedelta64[h]')[None, :]
        y = speed.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy(np.float64).reshape(-1, 72)
        if not np.isfinite(y).all() or len(varying) != 9:
            raise ValueError('Invalid frozen targets or varying channel count')
        stages[stage] = dict(origins=origins, future=future, y=y, baseline=baseline)
    return parent, selected, varying, stages


def load_inputs(out, smoke=False):
    out = Path(out)
    cache_dir = out / 'inputs'
    cache_dir.mkdir(parents=True, exist_ok=True)
    parent, selected, varying, stages = _frozen_inputs(smoke)
    all_origins = pd.DatetimeIndex(np.concatenate([ns(s['origins']) for s in stages.values()]), tz='UTC')
    hourly = pd.date_range(all_origins.min() - pd.Timedelta(hours=119), all_origins.max(), freq='h', tz='UTC')
    for stage in stages.values():
        stage['positions'] = window_positions(ns(hourly), stage['origins'])
    needed = np.ones(len(hourly), dtype=bool)
    if smoke:
        needed[:] = False
        for stage in stages.values():
            for pos in stage['positions']:
                needed[pos - 119:pos + 1] = True
    with production_helpers() as helpers:
        root, prepare, hr, ia, st, le, ch = helpers
        sources = [p for p in prepare.source_files(root) if p.is_file()]
        sources += [METADATA, PARENT, Path(__file__), *sorted(CACHE.iterdir())]
        modules = [prepare, hr, ia, st, le, ch, ch.L9]
        sources += [Path(m.__file__) for m in modules]
        identity = dict(version=1, smoke=bool(smoke), lookback=120, lag_hours=0,
                        selected=selected, varying=varying, historical_varying_horizon=1,
                        first_hour=str(hourly.min()), last_hour=str(hourly.max()),
                        generated_hours=int(needed.sum()), prefill='causal ffill; leading NaN preserved',
                        source_sha256={str(p): digest(p) for p in sorted(set(sources))})
        identity_path = cache_dir / 'identity.json'
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError('Input cache identity changed; use a new run output')
        else:
            write_json(identity_path, identity)
        array_path, receipt_path = cache_dir / 'hourly.npy', cache_dir / 'receipt.json'
        if array_path.exists() and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if digest(array_path) != receipt['hourly_sha256']:
                raise ValueError('Hourly cache hash changed')
            history = np.load(array_path, mmap_mode='r')
        else:
            print(f'Building {int(needed.sum())} actual hourly feat30 rows (lag0)', flush=True)
            raw = np.full((len(hourly), 30), np.nan, dtype=np.float32)
            raw[needed] = hourly_features(hourly[needed], selected, helpers)
            # The frozen table is already forward-filled on its origin grid;
            # compare only mutually finite raw rebuild cells, never fill a
            # missing candidate using parent values to manufacture parity.
            parity = {}
            for name, stage in stages.items():
                ix = np.linspace(0, len(stage['origins']) - 1, min(16, len(stage['origins'])), dtype=int)
                fresh, old = raw[stage['positions'][ix]], stage['baseline'][ix]
                finite = np.isfinite(fresh) & np.isfinite(old)
                close = np.isclose(fresh, old, rtol=1e-5, atol=1e-4)
                mismatch = finite & ~close
                parity[name] = dict(rows=len(ix), finite_pairs=int(finite.sum()),
                                    mismatches={col: int(mismatch[:, j].sum()) for j, col in enumerate(selected) if mismatch[:, j].any()},
                                    compared_by_feature={col: int(finite[:, j].sum()) for j, col in enumerate(selected)})
            if any(row['mismatches'] for row in parity.values()):
                write_json(cache_dir / 'parity_failure.json', parity)
                raise ValueError('Hourly builder differs from frozen parent on finite cells; see parity_failure.json')
            history = causal_prefill(raw)
            temp = array_path.with_suffix('.npy.tmp')
            with temp.open('wb') as target:
                np.save(target, history)
            temp.replace(array_path)
            receipt = dict(hourly_sha256=digest(array_path), shape=list(history.shape), parity=parity,
                           generated_hours=int(needed.sum()), nan_after_prefill={col: int(np.isnan(history[:, j]).sum()) for j, col in enumerate(selected)})
            write_json(receipt_path, receipt)
    for stage in stages.values():
        del stage['baseline']
    return dict(history_hourly=history, hourly_origins=ns(hourly), selected=selected,
                varying=varying, identity=identity, receipt=receipt, **stages)
