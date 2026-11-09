"""Causal, target-free external feature builders for follow-up axes 6–8.

Builders are pure; cache loaders never download. Archive existence is not proof
of historical exposure. Consult external-readiness.json before scheduling.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEATURES = {
    6: ['ace_speed_sd60', 'ace_speed_iqr60', 'ace_speed_range60'],
    7: ['ace_speed_slope60', 'ace_speed_mean15_minus_prev45'],
    8: ['cme_issued_speed', 'cme_issued_half_angle', 'cme_issued_lon',
        'cme_issued_lat', 'cme_issue_age_hours'],
}


def describe(axis):
    axis = int(axis)
    return {'axis': axis, 'features': FEATURES[axis].copy(), 'kind': 'constant',
            'window': '(origin-60min, origin-4min]' if axis != 8 else '[origin-72h, origin]',
            'missing': 'NaN; no future filling',
            'source': 'ACE SWEPAM status=0, finite positive speed' if axis != 8 else 'issued CME summary only'}


def _module(relative):
    spec = spec_from_file_location('_external_' + Path(relative).stem, ROOT / relative)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ace_minutes(cache_root, start, end):
    """Read available daily text files; absent days remain absent, never filled."""
    parser = _module('research/ace_1m_to_hourly_20261108/experiment.py')
    frames = []
    for day in pd.date_range(pd.Timestamp(start).floor('D'), pd.Timestamp(end).floor('D')):
        path = Path(cache_root) / f'{day:%Y%m%d}.txt'
        if path.exists():
            frame, _ = parser.parse_text('swepam', path.read_text())
            if len(frame):
                frames.append(frame[['S', 'spd']])
    return pd.concat(frames) if frames else pd.DataFrame(columns=['S', 'spd'], index=pd.DatetimeIndex([], tz='UTC'))


def load_cme_archive(directory):
    import json
    parser = _module('analysis/w50_audit_20261108/notification_archive_probe.py')
    records = []
    for path in sorted(Path(directory).glob('????-??.json')):
        for row in json.loads(path.read_text()):
            records.append(parser.parse_message(row))
    return records


def _ace(frame):
    values = frame[['S', 'spd']].copy()
    values.index = pd.to_datetime(values.index, utc=True)
    # Never resolve a conflicting historical duplicate by choosing its latest copy.
    for _, group in values[values.index.duplicated(keep=False)].groupby(level=0):
        if len(group.drop_duplicates()) > 1:
            raise ValueError('Conflicting ACE sample versions')
    values = values[~values.index.duplicated()].sort_index()
    return values.loc[(values.S == 0) & np.isfinite(values.spd) & (values.spd > 0), 'spd']


def _cme(records):
    seen = {}
    for record in records:
        key = record.get('message_id')
        if not key:
            raise ValueError('CME message requires stable message_id')
        if key in seen and seen[key] != record:
            raise ValueError('Conflicting CME message versions')
        seen[key] = record
    valid = []
    for row in seen.values():
        if not row.get('single_cme_forecast_summary'):
            continue
        fields = ['speed_kms', 'half_angle_deg', 'longitude_deg', 'latitude_deg']
        if len(row.get('event_ids', [])) != 1 or any(row.get(k) is None for k in fields):
            continue
        vals = np.asarray([row[k] for k in fields], dtype=float)
        if not np.isfinite(vals).all() or not (vals[0] > 0 and 0 < vals[1] <= 180 and abs(vals[2]) <= 180 and abs(vals[3]) <= 90):
            continue
        stamp = pd.Timestamp(row['conservative_issue_time'])
        if stamp.tzinfo is None or pd.isna(stamp):
            raise ValueError('CME issue timestamp must be timezone-aware')
        valid.append((stamp.value, vals))
    valid.sort(key=lambda pair: pair[0])
    for a, b in zip(valid, valid[1:]):
        if a[0] == b[0] and not np.array_equal(a[1], b[1]):
            raise ValueError('Ambiguous CME messages sharing issue time')
    return valid


def build(axis, origins, *, ace_minutes=None, cme_records=None):
    """Return float columns aligned to input origin order, including duplicates.

    ACE needs a UTC-indexed frame with S and spd. CME needs records returned by
    load_cme_archive. Callers must check source coverage/provenance separately.
    """
    axis = int(axis)
    index = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
    if index.hasnans:
        raise ValueError('Origins must be valid timestamps')
    result = np.full((len(index), len(FEATURES[axis])), np.nan)
    minute = pd.Timedelta(minutes=1).value
    if axis in (6, 7):
        if ace_minutes is None:
            raise ValueError('ace_minutes is required')
        series = _ace(ace_minutes)
        times, speeds = series.index.asi8, series.to_numpy(float)
        for i, origin in enumerate(index.asi8):
            left = np.searchsorted(times, origin - 60 * minute, side='right')
            right = np.searchsorted(times, origin - 4 * minute, side='right')
            v, t = speeds[left:right], times[left:right]
            if not len(v):
                continue
            if axis == 6:
                q = np.quantile(v, [.25, .75], method='linear')
                result[i] = [v.std(ddof=0), q[1] - q[0], v.max() - v.min()]
            else:
                x = (t - origin) / minute
                if len(v) >= 2:
                    centered = x - x.mean()
                    result[i, 0] = np.dot(centered, v - v.mean()) / np.dot(centered, centered)
                recent = t > origin - 15 * minute
                if recent.any() and (~recent).any():
                    result[i, 1] = v[recent].mean() - v[~recent].mean()
    else:
        if cme_records is None:
            raise ValueError('cme_records is required')
        records = _cme(cme_records)
        stamps = np.asarray([row[0] for row in records], dtype=np.int64)
        for i, origin in enumerate(index.asi8):
            pos = np.searchsorted(stamps, origin, side='right') - 1
            if pos >= 0 and origin - stamps[pos] <= 72 * 60 * minute:
                result[i] = [*records[pos][1], (origin - stamps[pos]) / (60 * minute)]
    return pd.DataFrame(result, index=index, columns=FEATURES[axis])


# Controller-facing name; archive records retain issue/version metadata.
load_cme_records = load_cme_archive


def probe_readiness(axis, *, start='2022-10-01', end='2026-09-26T23:00:00Z',
                    cache_root=None, archive_root=None):
    """Target-free filesystem readiness. Recheck on every controller iteration.

    ACE archival 4-minute publication is an explicit assumed availability rule,
    not receipt proof. Existing daily files are checked for nonzero byte size;
    row-level missingness is retained by build, not imputed.
    """
    import subprocess
    axis = int(axis)
    describe(axis)
    start = pd.to_datetime(start, utc=True)
    end = pd.to_datetime(end, utc=True)
    processes = subprocess.run(['ps', '-eo', 'pid,args'], check=True, capture_output=True,
                               text=True).stdout.splitlines()
    collectors = [line.strip() for line in processes
                  if 'experiment.py fetch' in line and 'ace_1m_to_hourly' in line]
    if axis in (6, 7):
        root = Path(cache_root or ROOT / 'data-v2/raw/ACE/daily-txt/swepam')
        wanted = [root / f'{day:%Y%m%d}.txt'
                  for day in pd.date_range((start - pd.Timedelta(hours=1)).floor('D'), end.floor('D'))]
        missing = [str(path) for path in wanted if not path.exists() or path.stat().st_size == 0]
        available = [path for path in wanted if str(path) not in missing]
        return {'axis': axis, 'ready': not missing, 'state': 'blocked' if missing else 'ready',
                'blocked_reason': 'missing_daily_cache' if missing else None,
                'providers': ['NOAA SWPC ACE SWEPAM daily text'], 'cache_root': str(root),
                'wanted_count': len(wanted), 'available_count': len(available),
                'missing_count': len(missing), 'missing_paths': missing,
                'bytes': sum(path.stat().st_size for path in available),
                'coverage_check': 'nonempty daily files; not a guarantee of valid minute rows',
                'required_start': str(start - pd.Timedelta(hours=1)), 'required_end': str(end),
                'collector_processes': collectors,
                'collector_status': 'running' if collectors else 'not_running',
                'collector_command': '.venv/bin/python research/ace_1m_to_hourly_20261108/experiment.py fetch --start '
                                     + str((start - pd.Timedelta(hours=1)).date()) + ' --end ' + str(end.date())
                                     + ' --device cpu --threads 1',
                'availability_assumption': 'sample timestamp +4min; historical daily publication is not a receipt log'}
    root = Path(archive_root or ROOT / 'analysis/w50_audit_20261108/notification_archive')
    months = pd.period_range((start - pd.Timedelta(hours=72)).date(), end.date(), freq='M')
    missing = [str(root / f'{month}.json') for month in months if not (root / f'{month}.json').exists()]
    return {'axis': 8, 'ready': False, 'state': 'blocked',
            'blocked_reason': 'immutable_as_issued_body_provenance_unproven',
            'providers': ['NASA DONKI official issued notification archive'],
            'archive_root': str(root), 'wanted_count': len(months),
            'missing_count': len(missing), 'missing_paths': missing,
            'required_start': str(start - pd.Timedelta(hours=72)), 'required_end': str(end),
            'collector_status': 'not_running',
            'readiness_requirements': ['All required archive months and receipt metadata',
                                       'Evidence that historical issued body versions were not retrospectively overwritten'],
            'notes': 'Issue time and freshly downloaded archive receipt alone do not establish immutable historical body.'}
