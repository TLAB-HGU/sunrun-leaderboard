"""Origin-causal ACE Alfvénicity features from the existing hourly MAG/SWEPAM store.

ACE velocity is archived here only as a scalar speed.  We therefore use its
anti-sunward radial projection on GSM x, and record that approximation in every
feature-cache manifest.  No source data is downloaded or read beyond the run cap.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[2]
STORE_ROOT = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store')

SOURCE_DIR = Path('/home/t-lab01/.local/state/sundb/stage/ace_solar')
CACHE_DIR = STORE_ROOT / 'solar-inc-alfven72-20261014-cache'
SOURCE_MANIFEST = Path(__file__).with_name('ace_source_manifest.json')
ALFVEN_COLUMNS = (
    'sigma_c_1h', 'sigma_c_3h', 'sigma_c_6h',
    'sigma_r_1h', 'sigma_r_3h', 'sigma_r_6h',
    'alfven_ratio_1h', 'alfven_ratio_3h', 'alfven_ratio_6h',
    'vb_corr_3h', 'vb_corr_6h', 'alfven_mach_median_6h',
    'valid_fraction', 'alfven_missing',
)
WINDOWS = (1, 3, 6)
MU0 = 4.0 * math.pi * 1e-7
PROTON_MASS_KG = 1.67262192369e-27
INPUT_COLUMNS = (
    'ace_bx_gsm_nt', 'ace_by_gsm_nt', 'ace_bz_gsm_nt',
    'ace_density_cm3', 'ace_speed_kms', 'ace_mag_status', 'ace_swepam_status',
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_index(values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True))


def _valid(frame: pd.DataFrame) -> np.ndarray:
    numeric = frame[list(INPUT_COLUMNS[:5])].to_numpy(dtype=float)
    bmag = np.linalg.norm(numeric[:, :3], axis=1)
    return (np.isfinite(numeric).all(axis=1)
            & (numeric[:, 3] > 0)
            & (numeric[:, 4] > 0)
            & (bmag > 0)
            & (frame.ace_mag_status.to_numpy() == 0)
            & (frame.ace_swepam_status.to_numpy() == 0))


def _series(frame: pd.DataFrame):
    """Return projected velocity and Alfvén velocity vectors in km/s."""
    speed = frame.ace_speed_kms.to_numpy(dtype=float)
    density = frame.ace_density_cm3.to_numpy(dtype=float) * 1e6 * PROTON_MASS_KG
    b_tesla = frame[['ace_bx_gsm_nt', 'ace_by_gsm_nt', 'ace_bz_gsm_nt']].to_numpy(dtype=float) * 1e-9
    # GSM x is sunward; the measured solar-wind speed is projected anti-sunward.
    velocity = np.column_stack((-speed, np.zeros(len(speed)), np.zeros(len(speed))))
    alfven_velocity = b_tesla / np.sqrt(MU0 * density[:, None]) / 1000.0
    return velocity, alfven_velocity, speed / np.linalg.norm(alfven_velocity, axis=1)


def _window_stats(frame: pd.DataFrame, min_samples: int):
    names = ('sigma_c', 'sigma_r', 'alfven_ratio', 'vb_corr', 'mach')
    empty = dict.fromkeys(names, np.nan)
    good = frame.loc[_valid(frame)]
    if len(good) < min_samples:
        return empty, len(good)
    velocity, alfven, mach = _series(good)
    dv = velocity - velocity.mean(axis=0, keepdims=True)
    db = alfven - alfven.mean(axis=0, keepdims=True)
    ev = float(np.mean(np.sum(dv * dv, axis=1)))
    eb = float(np.mean(np.sum(db * db, axis=1)))
    denom = ev + eb
    corr_denom = float(np.std(dv[:, 0]) * np.std(db[:, 0]))
    corr = (float(np.clip(np.mean((dv[:, 0] - dv[:, 0].mean()) * (db[:, 0] - db[:, 0].mean())) / corr_denom,
                          -1.0, 1.0))
            if corr_denom > 0 else np.nan)
    return {
        'sigma_c': float(2.0 * np.mean(np.sum(dv * db, axis=1)) / denom) if denom > 0 else np.nan,
        'sigma_r': float((ev - eb) / denom) if denom > 0 else np.nan,
        'alfven_ratio': float(eb / ev) if ev > 0 else np.nan,
        'vb_corr': corr,
        'mach': float(np.median(mach)),
    }, len(good)


class AlfvenicityFeatureStore:
    def __init__(self, cap):
        self.cap = pd.Timestamp(cap)
        self.cap = self.cap.tz_localize('UTC') if self.cap.tz is None else self.cap.tz_convert('UTC')
        self.source_paths = sorted(SOURCE_DIR.glob('year=*/part.parquet'))
        if not self.source_paths:
            raise FileNotFoundError(f'no existing ACE MAG/SWEPAM parquet shards under {SOURCE_DIR}')
        self.source_hashes = {str(p): file_sha256(p) for p in self.source_paths}
        self.source_manifest = json.loads(SOURCE_MANIFEST.read_text())
        if self.source_hashes != self.source_manifest.get('source_hashes'):
            raise RuntimeError('pinned ACE MAG/SWEPAM source hashes changed; register a new experiment')
        cap_value = self.cap.to_pydatetime()
        self.source_raw = pd.concat([pd.read_parquet(
            p, columns=['timestamp_utc', *INPUT_COLUMNS],
            filters=[('timestamp_utc', '<=', cap_value)])
                                     for p in self.source_paths], ignore_index=True)
        self.source_raw['timestamp_utc'] = pd.to_datetime(self.source_raw.timestamp_utc, utc=True)
        self.source_raw = (self.source_raw.loc[self.source_raw.timestamp_utc <= self.cap]
                           .drop_duplicates('timestamp_utc', keep='last')
                           .sort_values('timestamp_utc').set_index('timestamp_utc'))
        if self.source_raw.empty:
            raise ValueError('no ACE MAG/SWEPAM rows at or before the run cap')
        self.input_first = str(self.source_raw.index.min())
        self.input_last = str(self.source_raw.index.max())
        code_hash = file_sha256(Path(__file__))
        key_payload = {
            'code_sha256': code_hash,
            'source_sha256': self.source_hashes,
            'source_cap': str(self.cap),
            'input_columns': INPUT_COLUMNS,
            'windows_hours': WINDOWS,
            'minimum_samples': {'1h': 2, '3h': 2, '6h': 3},
            'velocity_projection': 'scalar ace_speed_kms projected anti-sunward along GSM x',
            'latency_hours': 2,
        }
        self.cache_key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_path = CACHE_DIR / f'{self.cache_key}.parquet'
        self.manifest_path = CACHE_DIR / f'{self.cache_key}.json'
        self.cache_sha256 = None
        if self.cache_path.exists() and self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
            actual = file_sha256(self.cache_path)
            if manifest.get('cache_key') == self.cache_key and manifest.get('cache_sha256') == actual:
                self.table = pd.read_parquet(self.cache_path).set_index('origin_last_input_utc')
                self.table.index = _utc_index(self.table.index)
                self.cache_sha256 = actual
            else:
                self._build_cache(key_payload)
        else:
            self._build_cache(key_payload)

    def _one(self, origin, frame=None):
        origin = pd.Timestamp(origin)
        origin = origin.tz_localize('UTC') if origin.tz is None else origin.tz_convert('UTC')
        cutoff = origin - pd.Timedelta(hours=2)
        frame = self.source_raw if frame is None else frame
        # Use closed windows ending at the latest ACE sample whose conservative
        # two-hour availability time is no later than this origin.
        values = {}
        for width in WINDOWS:
            begin = cutoff - pd.Timedelta(hours=width)
            left = frame.index.searchsorted(begin, side='left')
            right = frame.index.searchsorted(cutoff, side='right')
            window = frame.iloc[left:right]
            minimum = 3 if width == 6 else 2
            stats, _ = _window_stats(window, minimum)
            for name in ('sigma_c', 'sigma_r', 'alfven_ratio'):
                values[f'{name}_{width}h'] = stats[name]
            if width in (3, 6):
                values[f'vb_corr_{width}h'] = stats['vb_corr']
            if width == 6:
                values['alfven_mach_median_6h'] = stats['mach']
                values['valid_fraction'] = float(_valid(window).sum() / 7.0)
                values['alfven_missing'] = float(not np.isfinite(stats['sigma_c']))
        return {name: values.get(name, np.nan) for name in ALFVEN_COLUMNS}

    def _build_cache(self, key_payload):
        observation_times = self.source_raw.index
        origins = observation_times + pd.Timedelta(hours=2)
        origins = origins[origins <= self.cap]
        records = [self._one(origin) for origin in origins]
        table = pd.DataFrame(records, index=origins)
        table.index.name = 'origin_last_input_utc'
        self.table = table
        temporary = self.cache_path.with_suffix('.parquet.tmp')
        table.reset_index().to_parquet(temporary, index=False)
        temporary.replace(self.cache_path)
        self.cache_sha256 = file_sha256(self.cache_path)
        manifest = {
            **key_payload, 'cache_key': self.cache_key, 'cache_sha256': self.cache_sha256,
            'cache_rows': len(table), 'source_observation_first': self.input_first,
            'source_observation_last': self.input_last, 'local_store_range': self.source_manifest['local_observation_range_utc'],
            'cache_path': str(self.cache_path),
        }
        tmp_manifest = self.manifest_path.with_suffix('.json.tmp')
        tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        tmp_manifest.replace(self.manifest_path)

    def features(self, origins):
        idx = _utc_index(origins)
        result = self.table.reindex(idx)
        absent = result.isna().all(axis=1)
        for pos in np.flatnonzero(absent.to_numpy()):
            result.iloc[pos] = self._one(idx[pos])
        result.index = idx
        return result.loc[:, ALFVEN_COLUMNS]

    def poison(self, origin):
        origin = pd.Timestamp(origin)
        origin = origin.tz_localize('UTC') if origin.tz is None else origin.tz_convert('UTC')
        cutoff = origin - pd.Timedelta(hours=2)
        base = pd.DataFrame([self._one(origin)], columns=ALFVEN_COLUMNS)
        after_future = self.source_raw.copy()
        future = after_future.index > cutoff
        mutate = [c for c in INPUT_COLUMNS if c not in ('ace_mag_status', 'ace_swepam_status')]
        after_future.loc[future, mutate] = after_future.loc[future, mutate] + 1e5
        future_value = pd.DataFrame([self._one(origin, after_future)], columns=ALFVEN_COLUMNS)
        future_invariant = bool(np.allclose(base.to_numpy(), future_value.to_numpy(), equal_nan=True))

        valid = _valid(self.source_raw)
        six_hour_start = cutoff - pd.Timedelta(hours=6)
        past_positions = np.flatnonzero(valid & (self.source_raw.index >= six_hour_start)
                                        & (self.source_raw.index <= cutoff))
        if not len(past_positions):
            return {'alfven_future_invariant': future_invariant, 'alfven_positive_control': False,
                    'alfven_poison_passed': False, 'alfven_poison_reason': 'no admissible joint MAG/SWEPAM point'}
        past_time = self.source_raw.index[past_positions[-1]]
        after_past = self.source_raw.copy()
        after_past.loc[past_time, 'ace_speed_kms'] += 137.0
        past_value = pd.DataFrame([self._one(origin, after_past)], columns=ALFVEN_COLUMNS)
        positive = bool(not np.allclose(base.to_numpy(), past_value.to_numpy(), equal_nan=True))
        return {'alfven_future_invariant': future_invariant,
                'alfven_positive_control': positive,
                'alfven_poison_passed': bool(future_invariant and positive),
                'alfven_poison_origin': str(origin),
                'alfven_positive_control_observation': str(past_time),
                'alfven_source_range': [self.input_first, self.input_last],
                'alfven_cache_key': self.cache_key,
                'alfven_cache_sha256': self.cache_sha256}

    def report(self, origins):
        values = self.features(origins)
        report = {'n_origins': len(values), 'cache_key': self.cache_key,
                  'cache_sha256': self.cache_sha256, 'source_range': [self.input_first, self.input_last],
                  'local_store_range': self.source_manifest['local_observation_range_utc'],
                  'source_hashes': self.source_hashes, 'features': {}}
        for name in ALFVEN_COLUMNS:
            x = pd.to_numeric(values[name], errors='coerce').to_numpy(dtype=float)
            finite = x[np.isfinite(x)]
            summary = {'coverage': float(np.isfinite(x).mean()), 'n_finite': int(len(finite))}
            if len(finite):
                summary.update(median=float(np.median(finite)), p05=float(np.quantile(finite, .05)),
                               p95=float(np.quantile(finite, .95)))
            report['features'][name] = summary
        return report
