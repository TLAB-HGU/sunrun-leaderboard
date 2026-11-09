"""ace_1m_to_hourly: how well can ACE 1-minute (SWEPAM, MAG) and 5-minute (EPAM, SIS) lists approximate the NASA monthly
hourly files, and what does that buy the 72 h forecast?

One file, subcommands in the order of research/ace_1m_to_hourly_20261108/strategy.md (thresholds fixed there, before running):

    fetch       download the daily 1m/5m lists (resumable, polite)            -> data-v2/raw/ACE/daily-txt/<product>/YYYYMMDD.txt
    check       Stage 0 gates (completeness, not-revised, snapshot vs final)   -> <out>/integrity.json
    identify    Stage 1 monthly-rule identification and validity analysis      -> <out>/identification.json, validity.json
    estimate    Stage 2 causal estimators for the just-closed hour             -> <out>/estimators_{F1,F2,F3}.json
    downstream  Stage 3 effect on the 72 h forecast with the F1-F3 boosters    -> <out>/downstream.json

Everything is fitted on data up to FIT_END (before the F1 cutoff); 2026 hours are only scored. CPU only. No file
fingerprints are computed anywhere: integrity is byte length, row counts, time order and direct comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import numpy as np
import pandas as pd

EXPERIMENT = 'ace_1m_to_hourly'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DAILY_DIR = ROOT / 'data-v2/raw/ACE/daily-txt'
MONTHLY_DIR = ROOT / 'data-v2/raw/ACE/monthly-txt'
SNAPSHOT_DIR = HERE / 'data/snapshots'
URL = 'https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/'
USER_AGENT = 'sunrun-ace-1m-study/1.0 (research; polite; contact via repository owner)'

START_DAY = '2022-08-01'
SEED = 20261108
# split dates, fixed in strategy.md before running
SEL_END = pd.Timestamp('2024-12-31 23:00', tz='UTC')
CONF_END = pd.Timestamp('2025-12-28 23:00', tz='UTC')
FIT_END = CONF_END
PERIODS = {'F1': ('2026-01-01', '2026-04-01'), 'F2': ('2026-04-01', '2026-07-01'), 'F3': ('2026-07-01', '2026-09-30')}

# product -> (daily file suffix, monthly product name, cadence minutes, expected rows/day)
PRODUCTS = {'swepam': ('swepam_1m', 'swepam', 1, 1440), 'mag': ('mag_1m', 'mag', 1, 1440),
            'epam': ('epam_5m', 'epam', 5, 288), 'sis': ('sis_5m', 'sis', 5, 288)}
# field layout after the six time fields; (column, status column) and the missing marker (same as the monthly files)
SPEC = {
    'swepam': ([('S', None), ('dens', 'S'), ('spd', 'S'), ('temp', 'S')],
               {'dens': -9999.9, 'spd': -9999.9, 'temp': -1.0e5}),
    'mag': ([('S', None), ('bx', 'S'), ('by', 'S'), ('bz', 'S'), ('bt', 'S'), ('lat', 'S'), ('lon', 'S')], -999.9),
    'epam': ([('Se', None), ('e1', 'Se'), ('e2', 'Se'), ('Sp', None), ('p1', 'Sp'), ('p2', 'Sp'), ('p3', 'Sp'),
              ('p4', 'Sp'), ('p5', 'Sp'), ('anis', 'Sp')], {'anis': -1.0, '*': -1.0e5}),
    'sis': ([('S10', None), ('f10', 'S10'), ('S30', None), ('f30', 'S30')], -1.0e5),
}
# model variables -> (product, column in the daily list, column in the monthly file)
VARIABLES = {'speed': ('swepam', 'spd'), 'density': ('swepam', 'dens'), 'temperature': ('swepam', 'temp'),
             'bt': ('mag', 'bt'), 'by': ('mag', 'by'), 'epam_p3': ('epam', 'p3'), 'sis_gt10': ('sis', 'f10')}
PUBLICATION_LAG_MIN = {'swepam': 4.0, 'mag': 4.0, 'epam': 10.0, 'sis': 10.0}


def log(record: dict, path: Path | None = None) -> None:
    line = json.dumps(record, default=str)
    print(line, flush=True)


def revision() -> str:
    out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, capture_output=True, check=False)
    return out.stdout.strip()


def versions() -> dict:
    import scipy
    info = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__)
    for name in ('sklearn', 'xgboost'):
        try:
            info[name] = __import__(name).__version__
        except ImportError:
            pass
    return info


# ------------------------------------------------------------------------------------------------- parsing
def parse_text(product: str, text: str) -> tuple[pd.DataFrame, dict]:
    """Raw list text -> frame indexed by row time (UTC) with status and value columns as float, markers kept.

    Returns (frame, problems) where problems counts malformed lines. Row time is the time field of the list.
    """
    fields, _ = SPEC[product]
    names = [n for n, _ in fields]
    rows, bad, header = [], 0, {}
    for line in text.splitlines():
        if line.startswith(':Created:'):
            header['created'] = line[len(':Created:'):].strip()
        if not line or line[0] in ':#':
            continue
        parts = line.split()
        if len(parts) != 6 + len(fields) or not parts[0].isdigit():
            bad += 1
            continue
        rows.append(parts[:4] + parts[6:])
    if not rows:
        return pd.DataFrame(columns=names), dict(bad_lines=bad, **header)
    raw = pd.DataFrame(rows, columns=['yr', 'mo', 'da', 'hhmm'] + names)
    hhmm = raw['hhmm'].astype(int)
    t = pd.to_datetime(dict(year=raw.yr.astype(int), month=raw.mo.astype(int), day=raw.da.astype(int)), utc=True)
    t = t + pd.to_timedelta(hhmm // 100, unit='h') + pd.to_timedelta(hhmm % 100, unit='m')
    frame = raw[names].astype(float)
    frame.index = pd.DatetimeIndex(t)
    return frame, dict(bad_lines=bad, **header)


def missing_marker(product: str, column: str) -> float:
    _, missing = SPEC[product]
    return missing.get(column, missing.get('*')) if isinstance(missing, dict) else missing


def clean_column(product: str, frame: pd.DataFrame, column: str) -> pd.Series:
    """Values with NaN where the status digit is not 0 or the missing marker is present (the monthly-file rule)."""
    fields, _ = SPEC[product]
    status = dict(fields)[column]
    v = frame[column].copy()
    bad = np.isclose(v, missing_marker(product, column), rtol=0, atol=1e-6) | ~np.isfinite(v)
    if status is not None:
        bad |= frame[status].ne(0)
    return v.where(~bad)


def daily_path(product: str, day: datetime | pd.Timestamp) -> Path:
    return DAILY_DIR / product / f'{day:%Y%m%d}.txt'


def read_daily(product: str, day) -> pd.DataFrame | None:
    path = daily_path(product, day)
    if not path.exists():
        return None
    frame, _ = parse_text(product, path.read_text())
    return frame


def read_range(product: str, start, end) -> pd.DataFrame:
    """Concatenated daily lists for days start..end inclusive (missing days are skipped)."""
    parts = [f for f in (read_daily(product, d) for d in pd.date_range(start, end, freq='D', tz='UTC')) if f is not None and len(f)]
    out = pd.concat(parts).sort_index()
    return out[~out.index.duplicated(keep='last')]


def read_monthly(product: str) -> pd.DataFrame:
    """All local monthly hourly files of a product, indexed by label time (start of the averaged hour)."""
    monthly_name = PRODUCTS[product][1]
    parts = []
    for path in sorted(MONTHLY_DIR.glob(f'*_ace_{monthly_name}_1h.txt')):
        frame, _ = parse_text(product, path.read_text())
        if len(frame):
            parts.append(frame)
    out = pd.concat(parts).sort_index()
    return out[~out.index.duplicated(keep='last')]


# ------------------------------------------------------------------------------------------------- fetch
_lock = threading.Lock()


def get(url: str, tries: int = 5) -> bytes | None:
    """Body of url, None on 404. Other failures are retried with exponential back-off, then raised."""
    failure: Exception | None = None
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT, 'Accept-Encoding': 'identity'})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            failure = error
        except OSError as error:
            failure = error
        time.sleep(min(60.0, 2.0 ** attempt))
    raise RuntimeError(f'GET failed {url}: {failure}')


def fetch_one(job: tuple[str, pd.Timestamp], delay: float) -> tuple[str, str, str]:
    product, day = job
    path = daily_path(product, day)
    if path.exists() and path.stat().st_size > 0:
        return product, f'{day:%Y%m%d}', 'have'
    time.sleep(delay)
    body = get(f'{URL}{day:%Y%m%d}_ace_{PRODUCTS[product][0]}.txt')
    if body is None:
        return product, f'{day:%Y%m%d}', 'missing'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_bytes(body)
    temporary.replace(path)
    return product, f'{day:%Y%m%d}', 'added'


def cmd_fetch(args) -> int:
    last = pd.Timestamp(args.end, tz='UTC') if args.end else pd.Timestamp.now(tz='UTC').normalize() - pd.Timedelta(days=1)
    days = list(pd.date_range(pd.Timestamp(args.start, tz='UTC'), last, freq='D'))
    jobs = [(p, d) for p in PRODUCTS for d in days]
    log(dict(event='fetch_start', jobs=len(jobs), first=str(days[0].date()), last=str(days[-1].date()), workers=args.workers))
    counts: dict[str, int] = {}
    began = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        for i, (product, day, status) in enumerate(pool.map(lambda j: fetch_one(j, args.delay), jobs), 1):
            counts[status] = counts.get(status, 0) + 1
            if status == 'missing':
                log(dict(event='missing', product=product, day=day))
            if i % 200 == 0 or i == len(jobs):
                log(dict(event='fetch_progress', done=i, total=len(jobs), seconds=round(time.time() - began), **counts))
    log(dict(event='fetch_done', **counts))
    return 0 if counts.get('missing', 0) == 0 else 1


# ------------------------------------------------------------------------------------------------- check (Stage 0)
def expected_created(day: pd.Timestamp) -> str:
    nxt = day + pd.Timedelta(days=1)
    return f'{nxt:%Y %b %d} 0011 UT'


def check_file(product: str, day: pd.Timestamp) -> dict:
    path = daily_path(product, day)
    if not path.exists():
        return dict(ok=False, problem='absent')
    frame, info = parse_text(product, path.read_text())
    cadence, expected = PRODUCTS[product][2], PRODUCTS[product][3]
    problems = []
    if info['bad_lines']:
        problems.append(f"bad_lines={info['bad_lines']}")
    if len(frame) != expected:
        problems.append(f'rows={len(frame)}')
    if len(frame):
        if not frame.index.is_monotonic_increasing:
            problems.append('not_monotonic')
        if frame.index.has_duplicates:
            problems.append('duplicates')
        steps = frame.index.to_series().diff().dropna().dt.total_seconds().div(60)
        if len(steps) and (steps != cadence).any():
            problems.append(f'cadence_breaks={int((steps != cadence).sum())}')
        if frame.index.min() < day or frame.index.max() >= day + pd.Timedelta(days=1):
            problems.append('outside_day')
    if info.get('created') != expected_created(day):
        problems.append(f"created={info.get('created')}")
    return dict(ok=not problems, problem=';'.join(problems), rows=len(frame))


def compare_snapshot(product: str, snapshot_text: str, final_text: str) -> dict:
    """Rows of a live 2-hour list against the same rows of the final daily list."""
    snap, _ = parse_text(product, snapshot_text)
    final, _ = parse_text(product, final_text)
    common = snap.index.intersection(final.index)
    differing = [(str(t), c) for t in common for c in snap.columns
                 if not (snap.at[t, c] == final.at[t, c] or (np.isnan(snap.at[t, c]) and np.isnan(final.at[t, c])))]
    return dict(snapshot_rows=len(snap), compared=len(common), missing_in_final=len(snap.index.difference(final.index)),
                differing_cells=len(differing), first_differences=differing[:5])


def cmd_check(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    last = pd.Timestamp(args.end, tz='UTC') if args.end else pd.Timestamp.now(tz='UTC').normalize() - pd.Timedelta(days=1)
    days = list(pd.date_range(pd.Timestamp(args.start, tz='UTC'), last, freq='D'))
    report: dict = dict(experiment=EXPERIMENT, stage='check', git_commit=revision(), versions=versions(),
                        data_range=[str(days[0].date()), str(days[-1].date())])
    # gate 1: completeness
    gate1, bad_days = {}, {}
    for product in PRODUCTS:
        results = [check_file(product, d) for d in days]
        complete = sum(r['ok'] for r in results)
        gate1[product] = dict(days=len(days), complete=complete, share=round(complete / len(days), 5))
        bad_days[product] = [(f'{d:%Y%m%d}', r['problem']) for d, r in zip(days, results) if not r['ok']][:25]
    report['gate1_completeness'] = dict(per_product=gate1, first_deviations=bad_days,
                                        passed=all(v['share'] >= 0.99 for v in gate1.values()))
    # monthly coverage
    monthly = {p: read_monthly(p) for p in ('swepam', 'mag', 'epam', 'sis')}
    report['monthly_coverage'] = {p: dict(hours=len(f), first=str(f.index.min()), last=str(f.index.max())) for p, f in monthly.items()}
    # gate 2: not revised (direct byte comparison of a seeded random sample, re-downloaded now)
    rng = random.Random(SEED)
    sample = sorted(rng.sample(days[:-3], min(args.sample, len(days) - 3)))
    mismatches, compared = [], 0
    for d in sample:
        for product in PRODUCTS:
            path = daily_path(product, d)
            if not path.exists():
                continue
            body = get(f'{URL}{d:%Y%m%d}_ace_{PRODUCTS[product][0]}.txt')
            time.sleep(args.delay)
            compared += 1
            if body != path.read_bytes():
                mismatches.append(f'{product} {d:%Y%m%d}')
    report['gate2_not_revised'] = dict(sample_days=len(sample), files_compared=compared, mismatching_files=mismatches,
                                       seed=SEED, passed=(not mismatches and compared > 0))
    # gate 3: live snapshot vs final daily list
    gate3 = []
    for path in sorted(SNAPSHOT_DIR.glob('*.txt')):
        stem, _, stamp = path.stem.partition('@')                      # e.g. ace-swepam@20261108T0421Z
        product = {'ace-swepam': 'swepam', 'ace-magnetometer': 'mag', 'ace-epam': 'epam', 'ace-sis': 'sis'}.get(stem)
        if product is None:
            continue
        snap, _ = parse_text(product, path.read_text())
        if not len(snap):
            continue
        day = snap.index.min().normalize()
        final = daily_path(product, day)
        complete = (day + pd.Timedelta(days=1, minutes=11)) <= pd.Timestamp.now(tz='UTC')
        if not final.exists() or not complete:
            gate3.append(dict(file=path.name, product=product, status='final list not complete yet'))
            continue
        gate3.append(dict(file=path.name, product=product, **compare_snapshot(product, path.read_text(), final.read_text())))
    report['gate3_snapshot_vs_final'] = gate3
    (out / 'integrity.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    log(dict(event='check_done', gate1=report['gate1_completeness']['passed'], gate2=report['gate2_not_revised']['passed'],
             output=str(out / 'integrity.json')))
    return 0 if report['gate1_completeness']['passed'] and report['gate2_not_revised']['passed'] else 1


# ------------------------------------------------------------------------------------------------- shared stage tools
def hours_in(start, end) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start, tz='UTC') if not isinstance(start, pd.Timestamp) else start,
                         pd.Timestamp(end, tz='UTC') if not isinstance(end, pd.Timestamp) else end, freq='h')


def split_of(labels: pd.DatetimeIndex) -> pd.Series:
    """sel / conf / F1 / F2 / F3 for each label hour (anything later than F3's end is 'after')."""
    out = pd.Series('sel', index=labels)
    out[labels > SEL_END] = 'conf'
    out[labels > CONF_END] = 'gap'          # 2025-12-29 .. 2025-12-31: between the fit limit and F1
    for name, (a, b) in PERIODS.items():
        out[(labels >= pd.Timestamp(a, tz='UTC')) & (labels < pd.Timestamp(b, tz='UTC'))] = name
    out[labels >= pd.Timestamp(PERIODS['F3'][1], tz='UTC')] = 'after'
    return out


def tolerance(variable: str, value: np.ndarray) -> np.ndarray:
    """File rounding: one decimal for speed, density, Bt, By (0.05 plus float noise); three significant digits otherwise."""
    if variable in ('speed', 'density', 'bt', 'by'):
        return np.full(np.shape(value), 0.0501)
    return np.abs(value) * 0.0051 + 1e-12


_FRAMES: dict = {}


def product_frame(product: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    key = (product, start.normalize(), end.normalize())
    if key not in _FRAMES:
        _FRAMES[key] = read_range(product, start.normalize(), end.normalize())
    return _FRAMES[key]


class Series:
    """One cleaned variable of a list on a regular grid, with cumulative sums for O(1) window sums and counts."""

    def __init__(self, product: str, column: str, start, end, policy: str = 'S0'):
        """policy 'S0': samples with status 0 only; 'any': every sample with a valid value (status ignored)."""
        self.product, self.column, self.cadence, self.policy = product, column, PRODUCTS[product][2], policy
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        self.grid = pd.date_range(start, end, freq=f'{self.cadence}min')
        frame = product_frame(product, start, end).reindex(self.grid)
        self.frame = frame
        if policy == 'S0':
            self.v = clean_column(product, frame, column)
        else:
            raw = frame[column]
            self.v = raw.where(~(np.isclose(raw, missing_marker(product, column), rtol=0, atol=1e-6) | ~np.isfinite(raw)))
        values = self.v.to_numpy(float)
        self.good = np.isfinite(values)
        self.x = np.where(self.good, values, 0.0)
        self.cs = np.r_[0.0, np.cumsum(self.x)]
        self.cg = np.r_[0, np.cumsum(self.good.astype(np.int64))]
        self.size = len(self.grid)

    def pos(self, labels: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray(np.round((labels - self.grid[0]) / pd.Timedelta(minutes=self.cadence))).astype(np.int64)

    def window(self, p: np.ndarray, o: int, n: int):
        """Sum, count of nominal samples, and in-range flag for windows [p+o, p+o+n) (steps of the cadence)."""
        a, b = p + o, p + o + n
        ok = (a >= 0) & (b <= self.size)
        a2, b2 = np.clip(a, 0, self.size), np.clip(b, 0, self.size)
        return self.cs[b2] - self.cs[a2], self.cg[b2] - self.cg[a2], ok

    def matrix(self, p: np.ndarray, o: int, n: int) -> np.ndarray:
        """Samples of the windows as a (hours, n) array, NaN where missing or out of range."""
        idx = p[:, None] + o + np.arange(n)[None, :]
        inside = (idx >= 0) & (idx < self.size)
        vals = self.v.to_numpy(float)[np.clip(idx, 0, self.size - 1)]
        return np.where(inside, vals, np.nan)


def monthly_target(product: str, column: str, labels: pd.DatetimeIndex) -> np.ndarray:
    monthly = read_monthly(product)
    return clean_column(product, monthly, column).reindex(labels).to_numpy(float)


def monthly_status(product: str, column: str, labels: pd.DatetimeIndex) -> np.ndarray:
    monthly = read_monthly(product)
    status = dict(SPEC[product][0])[column]
    return monthly[status].reindex(labels).to_numpy(float)


# ------------------------------------------------------------------------------------------------- stage 1: identify
CLEAN_FRAC = 0.98          # a window is "clean" when at least this share of its samples is nominal
MIN_CLEAN_HOURS = 300


def score_cell(series: Series, p: np.ndarray, target: np.ndarray, variable: str, o: int, n: int) -> dict | None:
    total, count, ok = series.window(p, o, n)
    clean = ok & (count >= CLEAN_FRAC * n) & np.isfinite(target) & (count > 0)
    if clean.sum() < MIN_CLEAN_HOURS:
        return None
    est = total[clean] / count[clean]
    diff = np.abs(target[clean] - est)
    tol = tolerance(variable, target[clean])
    return dict(policy=series.policy, offset=o * series.cadence, length=n * series.cadence, clean_hours=int(clean.sum()),
                match=float((diff <= tol).mean()), median=float(np.median(diff)), p95=float(np.quantile(diff, 0.95)))


def coarse_search(series: Series, labels: pd.DatetimeIndex, variable: str, offsets, lengths, top: int = 20) -> list[dict]:
    p = series.pos(labels)
    target = monthly_target(series.product, series.column, labels)
    cells = []
    for o in offsets:
        for n in lengths:
            r = score_cell(series, p, target, variable, o, n)
            if r:
                cells.append(r)
    cells.sort(key=lambda c: (-c['match'], c['median']))
    return cells[:top]


AGGREGATORS = ('mean', 'median', 'trim10', 'trim20', 'last', 'block5')


def aggregate(mat: np.ndarray, how: str, cadence: int) -> np.ndarray:
    from scipy import stats
    if how == 'mean':
        return np.nanmean(mat, axis=1)
    if how == 'median':
        return np.nanmedian(mat, axis=1)
    if how in ('trim10', 'trim20'):
        cut = 0.10 if how == 'trim10' else 0.20
        out = np.full(len(mat), np.nan)
        for i, row in enumerate(mat):
            r = row[np.isfinite(row)]
            if len(r):
                out[i] = stats.trim_mean(r, cut)
        return out
    if how == 'last':
        idx = np.where(np.isfinite(mat), np.arange(mat.shape[1])[None, :], -1).max(1)
        return np.where(idx >= 0, mat[np.arange(len(mat)), np.clip(idx, 0, None)], np.nan)
    if how == 'block5':                                  # mean of 5-sample block means (nominal samples only)
        n = mat.shape[1] // 5 * 5
        if n == 0:
            return np.nanmean(mat, axis=1)
        blocks = np.nanmean(mat[:, :n].reshape(len(mat), -1, 5), axis=2)
        return np.nanmean(blocks, axis=1)
    raise ValueError(how)


def refine(series: Series, labels, variable: str, cells: list[dict], min_counts=(1, 10, 20, 30, 40, 50, 55)) -> list[dict]:
    p = series.pos(labels)
    target = monthly_target(series.product, series.column, labels)
    results = []
    for cell in cells:
        o, n = cell['offset'] // series.cadence, cell['length'] // series.cadence
        mat = series.matrix(p, o, n)
        have = np.isfinite(mat).sum(1)
        usable = np.isfinite(target) & (have >= CLEAN_FRAC * n)
        for how in AGGREGATORS:
            with np.errstate(all='ignore'):
                est = aggregate(mat[usable], how, series.cadence)
            diff = np.abs(target[usable] - est)
            fine = np.isfinite(diff)
            tol = tolerance(variable, target[usable][fine])
            results.append(dict(policy=series.policy, offset=cell['offset'], length=cell['length'], aggregator=how, hours=int(fine.sum()),
                                match=float((diff[fine] <= tol).mean()), median=float(np.median(diff[fine])),
                                p95=float(np.quantile(diff[fine], 0.95))))
    results.sort(key=lambda r: (-r['match'], r['median']))
    return results


def evaluate_rule(series: Series, labels, variable: str, rule: dict) -> dict:
    """Score one fixed rule on the clean hours of the given labels."""
    p = series.pos(labels)
    target = monthly_target(series.product, series.column, labels)
    o, n = rule['offset'] // series.cadence, rule['length'] // series.cadence
    mat = series.matrix(p, o, n)
    have = np.isfinite(mat).sum(1)
    usable = np.isfinite(target) & (have >= CLEAN_FRAC * n)
    with np.errstate(all='ignore'):
        est = aggregate(mat[usable], rule['aggregator'], series.cadence)
    diff = np.abs(target[usable] - est)
    fine = np.isfinite(diff)
    if not fine.any():
        return dict(hours=0)
    tol = tolerance(variable, target[usable][fine])
    return dict(hours=int(fine.sum()), match=float((diff[fine] <= tol).mean()), median=float(np.median(diff[fine])),
                p95=float(np.quantile(diff[fine], 0.95)))


def kernel_profile(series: Series, labels, lags_back: int = 90, lags_fwd: int = 150, min_good: float = 0.93) -> dict:
    """Non-negative least squares of the monthly value on the samples around the label (sum of weights = 1)."""
    from scipy.optimize import nnls
    p = series.pos(labels)
    target = monthly_target(series.product, series.column, labels)
    span = (lags_back + lags_fwd) // series.cadence
    mat = series.matrix(p, -lags_back // series.cadence, span)
    good = np.isfinite(mat).mean(1) >= min_good
    keep = good & np.isfinite(target)
    if keep.sum() < 200:
        return dict(hours=int(keep.sum()), weights=None)
    X = pd.DataFrame(mat[keep]).T.interpolate(limit_direction='both').T.to_numpy()
    y = target[keep]
    scale = np.nanstd(y)
    A = np.vstack([X / scale, 1e3 * np.ones((1, X.shape[1]))])
    b = np.r_[y / scale, 1e3]
    w, _ = nnls(A, b)
    centre = float((w * (np.arange(span) * series.cadence - lags_back)).sum() / max(w.sum(), 1e-12))
    return dict(hours=int(keep.sum()), weights=[round(float(x), 5) for x in w], lags_start=-lags_back, cadence=series.cadence,
                weight_centre_min=centre, weight_sum=float(w.sum()),
                support=[int(i * series.cadence - lags_back) for i in np.flatnonzero(w > 0.005)][:1] +
                        [int(i * series.cadence - lags_back) for i in np.flatnonzero(w > 0.005)][-1:])


def cmd_identify(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp('2022-08-01', tz='UTC')
    end = pd.Timestamp('2026-10-01', tz='UTC') - pd.Timedelta(minutes=1)
    labels_all = hours_in(start + pd.Timedelta(hours=6), pd.Timestamp('2026-09-30 23:00', tz='UTC'))
    split = split_of(labels_all)
    report: dict = dict(experiment=EXPERIMENT, stage='identify', git_commit=revision(), versions=versions(),
                        fit_range=[str(labels_all[0]), str(FIT_END)], clean_fraction=CLEAN_FRAC, variables={})
    variables = [v for v in VARIABLES if (not args.variables or v in args.variables)]
    for variable in variables:
        product, column = VARIABLES[variable]
        began = time.time()
        series = Series(product, column, start, end)
        cad = series.cadence
        offsets = range(args.offset_min // cad, args.offset_max // cad + 1)
        lengths = range(max(1, args.length_min // cad), args.length_max // cad + 1)
        sel_labels = labels_all[(split == 'sel').to_numpy()]
        assert sel_labels.max() <= FIT_END
        coarse = coarse_search(series, sel_labels, variable, offsets, lengths)
        rules = refine(series, sel_labels, variable, coarse)
        best = rules[0]
        conf = evaluate_rule(series, labels_all[(split == 'conf').to_numpy()], variable, best)
        by_year = {}
        for year in (2022, 2023, 2024):
            idx = (split == 'sel').to_numpy() & (labels_all.year == year)
            if idx.any():
                by_year[str(year)] = evaluate_rule(series, labels_all[idx], variable, best)
        kernel = kernel_profile(series, sel_labels)
        scored = {name: evaluate_rule(series, labels_all[(split == name).to_numpy()], variable, best) for name in PERIODS}
        sd_train = float(np.nanstd(monthly_target(product, column, sel_labels)))
        exact = (conf.get('match', 0) >= 0.99 and all(y.get('match', 0) >= 0.99 for y in by_year.values()))
        approx = (conf.get('median', 1e9) <= 0.005 * sd_train and conf.get('p95', 1e9) <= 0.03 * sd_train)
        report['variables'][variable] = dict(
            product=product, column=column, sd_train=sd_train, best_rule=best, runner_up=rules[1:4],
            coarse_top=coarse[:5], sel_clean_hours=best['hours'], conf=conf, by_year=by_year, kernel=kernel,
            scored_after_fit=scored, verdict='exact' if exact else ('approximate' if approx else 'noise floor'),
            seconds=round(time.time() - began, 1))
        log(dict(event='identified', variable=variable, rule=best, conf=conf, verdict=report['variables'][variable]['verdict']))
    (out / 'identification.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    return 0


# ------------------------------------------------------------------------------------------------- stage 2: estimators
LOG_VARIABLES = ('density', 'temperature', 'epam_p3', 'sis_gt10')      # positive, heavy-tailed: learned models work in logs
K_GRID = (5, 10, 15, 30, 45)                                           # "last k minutes" estimators (samples for 5-min lists)
GBM_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_RULE = dict(offset=0, length=60, aggregator='mean')


def visible_end(series: Series, t0: pd.DatetimeIndex, lag_min: float) -> np.ndarray:
    """Exclusive end index of the samples published by t0: sample time + lag <= t0."""
    steps = (t0 - pd.Timedelta(minutes=lag_min) - series.grid[0]) / pd.Timedelta(minutes=series.cadence)
    return np.floor(np.asarray(steps, dtype=float)).astype(np.int64) + 1     # numpy, not a pandas Index


def span_mean(series: Series, start: np.ndarray, end: np.ndarray):
    """Mean and nominal count of samples [start, end) per hour (end < start gives NaN)."""
    a, b = np.clip(start, 0, series.size), np.clip(end, 0, series.size)
    b = np.maximum(a, b)
    count = series.cg[b] - series.cg[a]
    total = series.cs[b] - series.cs[a]
    with np.errstate(all='ignore'):
        return np.where(count > 0, total / count, np.nan), count


def design_for(series: Series, labels: pd.DatetimeIndex, rule: dict, lag_min: float) -> dict:
    """Everything an estimator may use for the hour labelled L, as published at t0 = L + 1 h."""
    cad = series.cadence
    t0 = labels + pd.Timedelta(hours=1)
    end = visible_end(series, t0, lag_min)
    pL = series.pos(labels)
    o, n = rule['offset'] // cad, rule['length'] // cad
    a, b_full = pL + o, pL + o + n
    d: dict = {'end': end}
    d['E1'], d['E1_count'] = span_mean(series, a, np.minimum(b_full, end))
    d['O1'], d['O1_count'] = span_mean(series, a, b_full)                 # oracle: whole rule window, after t0 included
    for k in K_GRID:
        d[f'E2_{k}'], _ = span_mean(series, end - max(1, k // cad), end)
    tail = d['E2_10']
    unseen = np.maximum(0, b_full - np.maximum(a, end))
    seen_total = d['E1'] * d['E1_count']
    with np.errstate(all='ignore'):
        e3 = np.where(d['E1_count'] + unseen > 0,
                      (np.nan_to_num(seen_total) + unseen * np.where(np.isfinite(tail), tail, d['E1'])) / np.maximum(d['E1_count'] + unseen, 1), np.nan)
    d['E3'] = np.where(np.isfinite(e3), e3, tail)
    # learned-model features from the visible tail only
    win = 60 // cad
    mat = series.matrix(end, -win, win)                                    # last `win` samples before the visible end
    with np.errstate(all='ignore'):
        d['f_mean60'] = np.nanmean(mat, 1)
        d['f_std30'] = np.nanstd(mat[:, -(30 // cad):], 1)
        half = win // 2
        d['f_slope'] = np.nanmean(mat[:, half:], 1) - np.nanmean(mat[:, :half], 1)
        d['f_median60'] = np.nanmedian(mat, 1)
    d['f_nominal60'] = np.isfinite(mat).mean(1)
    last_good = np.where(np.isfinite(mat), np.arange(win)[None, :], -1).max(1)
    d['f_since_good'] = np.where(last_good >= 0, (win - 1 - last_good) * cad, win * cad)
    d['f_last'] = np.where(last_good >= 0, mat[np.arange(len(mat)), np.clip(last_good, 0, None)], np.nan)
    d['f_hour'] = labels.hour.to_numpy()
    return d


def previous_monthly(product: str, column: str, labels: pd.DatetimeIndex) -> dict:
    """What the monthly files hold at t0: labels <= L - 1 h (the file refreshes at :10). E0 = forward fill of those."""
    full = pd.date_range(labels.min() - pd.Timedelta(days=3), labels.max(), freq='h')
    m = pd.Series(monthly_target(product, column, full), index=full)
    prev1 = m.shift(1).reindex(labels).to_numpy()                      # M(L-1h), NaN if that hour is invalid
    ffilled = m.shift(1).ffill().reindex(labels).to_numpy()            # E0
    last3 = m.shift(1).rolling(3, min_periods=1).mean().reindex(labels).to_numpy()
    return dict(E0=ffilled, prev1=prev1, prev3=last3)


def stats_block(est: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict:
    ok = mask & np.isfinite(est) & np.isfinite(truth)
    if ok.sum() < 20:
        return dict(hours=int(ok.sum()))
    e = est[ok] - truth[ok]
    return dict(hours=int(ok.sum()), rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))), bias=float(np.mean(e)))


def learned_features(d: dict, prev: dict, base: np.ndarray, log: bool) -> np.ndarray:
    names = ['f_mean60', 'f_std30', 'f_slope', 'f_median60', 'f_nominal60', 'f_since_good', 'f_last', 'E1', 'E2_15', 'E3']
    cols = [d[n] for n in names] + [prev['prev1'], prev['prev3'], prev['E0'], d['E1_count'].astype(float)]
    X = np.column_stack(cols).astype(float)
    if log:
        with np.errstate(all='ignore'):
            X[:, [0, 3, 6, 7, 8, 9, 10, 11, 12]] = np.log(np.where(X[:, [0, 3, 6, 7, 8, 9, 10, 11, 12]] > 0, X[:, [0, 3, 6, 7, 8, 9, 10, 11, 12]], np.nan))
    X = np.column_stack([X, np.isfinite(base).astype(float)])
    return X


def fit_ridge_gbm(X: dict, y: dict, seeds=GBM_SEEDS) -> dict:
    """Ridge (alpha picked on conf) and small GBM (rounds picked on conf, 5 seeds). Fit data: sel, then sel+conf refit."""
    from sklearn.linear_model import Ridge
    import xgboost as xgb
    fit, val = X['sel'], X['conf']
    med = np.nanmedian(fit, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)

    def fill(a):
        out = np.where(np.isfinite(a), a, med[None, :])
        return out
    mu, sd = fill(fit).mean(0), fill(fit).std(0) + 1e-9
    Z = lambda a: (fill(a) - mu) / sd                                  # noqa: E731
    best_alpha, best = None, 1e18
    for alpha in (0.1, 1, 10, 100, 1000):
        model = Ridge(alpha=alpha).fit(Z(fit), y['sel'])
        err = float(np.mean((model.predict(Z(val)) - y['conf']) ** 2))
        if err < best:
            best_alpha, best = alpha, err
    both = np.vstack([fit, val]), np.r_[y['sel'], y['conf']]
    ridge = Ridge(alpha=best_alpha).fit(Z(both[0]), both[1])
    models = dict(ridge=ridge, alpha=best_alpha, Z=Z, gbm=[], rounds=[])
    for seed in seeds:
        params = dict(max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, reg_lambda=5,
                      objective='reg:squarederror', tree_method='hist', device='cpu', nthread=1, seed=seed, verbosity=0)
        dtr, dva = xgb.DMatrix(fit, label=y['sel']), xgb.DMatrix(val, label=y['conf'])
        booster = xgb.train(params, dtr, num_boost_round=400, evals=[(dva, 'conf')], early_stopping_rounds=30, verbose_eval=False)
        rounds = booster.best_iteration + 1
        dall = xgb.DMatrix(both[0], label=both[1])
        models['gbm'].append(xgb.train(params, dall, num_boost_round=rounds))
        models['rounds'].append(int(rounds))
    return models


def cmd_estimate(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ident_path = Path(args.identification)
    ident = json.loads(ident_path.read_text())['variables'] if ident_path.exists() else {}
    start = pd.Timestamp('2022-08-01', tz='UTC')
    end = pd.Timestamp('2026-10-01', tz='UTC') - pd.Timedelta(minutes=1)
    labels_all = hours_in(start + pd.Timedelta(hours=30), pd.Timestamp('2026-09-30 22:00', tz='UTC'))
    split = split_of(labels_all).to_numpy()
    report: dict = dict(experiment=EXPERIMENT, stage='estimate', git_commit=revision(), versions=versions(),
                        fit_range=[str(labels_all[0]), str(FIT_END)], lags_min=PUBLICATION_LAG_MIN, variables={})
    variables = [v for v in VARIABLES if (not args.variables or v in args.variables)]
    for variable in variables:
        product, column = VARIABLES[variable]
        rule = ident.get(variable, {}).get('best_rule') or DEFAULT_RULE
        rule = {k: rule[k] for k in ('offset', 'length', 'aggregator')}
        series = Series(product, column, start, end)
        lag = args.lag if args.lag is not None else PUBLICATION_LAG_MIN[product]
        d = design_for(series, labels_all, rule, lag)
        prev = previous_monthly(product, column, labels_all)
        truth = monthly_target(product, column, labels_all)
        log_target = variable in LOG_VARIABLES
        # simple estimators: pick the best on sel+conf (never on the scored periods)
        fit_mask = np.isin(split, ('sel', 'conf'))
        assert labels_all[fit_mask].max() <= FIT_END
        simple = {'E0': prev['E0'], 'E1': d['E1'], 'E3': d['E3'], **{f'E2_{k}': d[f'E2_{k}'] for k in K_GRID}}
        filled = {name: np.where(np.isfinite(v), v, prev['E0']) for name, v in simple.items()}
        fit_rmse = {name: stats_block(v, truth, fit_mask).get('rmse', 1e18) for name, v in filled.items() if name != 'E0'}
        best_simple = min(fit_rmse, key=fit_rmse.get)
        base = filled[best_simple]
        # learned models on the residual (log ratio for heavy-tailed variables)
        X = learned_features(d, prev, base, log_target)
        with np.errstate(all='ignore'):
            resid = (np.log(truth) - np.log(base)) if log_target else (truth - base)
        usable = np.isfinite(resid) & np.isfinite(base) & (np.abs(resid) < 1e6)
        Xs = {k: X[(split == k) & usable] for k in ('sel', 'conf')}
        ys = {k: resid[(split == k) & usable] for k in ('sel', 'conf')}
        models = fit_ridge_gbm(Xs, ys)
        pred = {}
        Zall = models['Z'](X)
        r_ridge = models['ridge'].predict(Zall)
        import xgboost as xgb
        r_gbm = np.array([m.predict(xgb.DMatrix(X)) for m in models['gbm']])
        back = (lambda r: base * np.exp(r)) if log_target else (lambda r: base + r)    # noqa: E731
        pred['E4_ridge'] = back(r_ridge)
        gbm_est = np.array([back(r) for r in r_gbm])
        pred['E5_gbm_mean'] = gbm_est.mean(0)
        results: dict = {'rule': rule, 'best_simple': best_simple, 'ridge_alpha': models['alpha'], 'gbm_rounds': models['rounds']}
        names = {'E0': prev['E0'], **{k: v for k, v in filled.items() if k != 'E0'}, **pred, 'O1_oracle': d['O1']}
        periods = {k: split == k for k in ('sel', 'conf', 'F1', 'F2', 'F3')}
        for pname, mask in periods.items():
            block = {}
            for name, est in names.items():
                block[name] = stats_block(est, truth, mask)
            # worst seed for the GBM
            block['E5_gbm_worst_seed'] = max((stats_block(g, truth, mask) for g in gbm_est), key=lambda b: b.get('rmse', 0))
            block['E5_gbm_best_seed'] = min((stats_block(g, truth, mask) for g in gbm_est), key=lambda b: b.get('rmse', 1e18))
            e0 = block['E0'].get('rmse')
            for name, b in block.items():
                if e0 and b.get('rmse') is not None:
                    b['rmse_vs_E0'] = round(b['rmse'] / e0 - 1, 4)
            results[pname] = block
        report['variables'][variable] = results
        log(dict(event='estimated', variable=variable, best_simple=best_simple, F3_E0=results['F3']['E0'].get('rmse'),
                 F3_best=results['F3'][best_simple].get('rmse'), F3_gbm=results['F3']['E5_gbm_mean'].get('rmse')))
        np.savez_compressed(out / f'hourly_estimates_{variable}.npz', labels=labels_all.asi8, truth=truth,
                            **{k: np.asarray(v, dtype=np.float32) for k, v in names.items()}, gbm=gbm_est.astype(np.float32))
    (out / 'estimators.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    return 0


# ------------------------------------------------------------------------------------------------- stage 1c: validity (H2)
VALID_OFFSET_TRAIN = ('sel',)


def status_counts(series: Series, p: np.ndarray, o: int, n: int, end: np.ndarray | None = None) -> np.ndarray:
    """Counts of status digit groups in windows [p+o, min(p+o+n, end)): 0, 1, 2-8, 9, and rows outside the list."""
    fields, _ = SPEC[series.product]
    status_col = dict(fields)[series.column]
    st = series.frame[status_col].to_numpy(float)
    groups = [(st == 0), (st == 1), ((st >= 2) & (st <= 8)), (st == 9), ~np.isfinite(st)]
    cums = [np.r_[0, np.cumsum(g.astype(np.int64))] for g in groups]
    a = p + o
    b = p + o + n if end is None else np.minimum(p + o + n, end)
    a2, b2 = np.clip(a, 0, series.size), np.clip(np.maximum(a, b), 0, series.size)
    return np.column_stack([c[b2] - c[a2] for c in cums]).astype(float)


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float('nan')


def cmd_validity(args) -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, export_text
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ident_path = Path(args.identification)
    ident = json.loads(ident_path.read_text())['variables'] if ident_path.exists() else {}
    start = pd.Timestamp('2022-08-01', tz='UTC')
    end = pd.Timestamp('2026-10-01', tz='UTC') - pd.Timedelta(minutes=1)
    labels = hours_in(start + pd.Timedelta(hours=30), pd.Timestamp('2026-09-30 22:00', tz='UTC'))
    split = split_of(labels).to_numpy()
    report: dict = dict(experiment=EXPERIMENT, stage='validity', git_commit=revision(), versions=versions(),
                        fit_range=[str(labels[0]), str(FIT_END)], products={})
    for variable in [v for v in ('speed', 'bt', 'epam_p3', 'sis_gt10') if (not args.variables or v in args.variables)]:
        product, column = VARIABLES[variable]
        series = Series(product, column, start, end)
        cad = series.cadence
        rule = ident.get(variable, {}).get('best_rule') or DEFAULT_RULE
        o, n = rule['offset'] // cad, rule['length'] // cad
        p = series.pos(labels)
        t0 = labels + pd.Timedelta(hours=1)
        vend = visible_end(series, t0, PUBLICATION_LAG_MIN[product])
        target = np.isfinite(monthly_target(product, column, labels)).astype(int)
        full = status_counts(series, p, o, n)
        wide = status_counts(series, p, o - 30 // cad, n + 60 // cad)
        trunc = status_counts(series, p, o, n, vend)
        m = pd.Series(target, index=labels)
        prev = np.column_stack([m.shift(1).fillna(1).to_numpy(), m.shift(2).fillna(1).to_numpy(), m.shift(6).fillna(1).to_numpy()])
        hod = np.isin(labels.hour, (3, 9, 15, 21)).astype(float)[:, None]
        sets = {'full': np.column_stack([full, wide, hod]), 'full_no_hour': np.column_stack([full, wide]),
                'truncated': np.column_stack([trunc, prev, hod]), 'truncated_no_hour': np.column_stack([trunc, prev])}
        fit_mask = split == 'sel'
        assert labels[fit_mask].max() <= FIT_END
        result: dict = {'valid_share': {k: float(target[split == k].mean()) for k in ('sel', 'conf', 'F1', 'F2', 'F3')}}
        for name, X in sets.items():
            tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=0).fit(X[fit_mask], target[fit_mask])
            logit = LogisticRegression(max_iter=2000, C=1.0).fit((X[fit_mask] - X[fit_mask].mean(0)) / (X[fit_mask].std(0) + 1e-9), target[fit_mask])
            block = {}
            for k in ('conf', 'F1', 'F2', 'F3'):
                mk = split == k
                pt = tree.predict_proba(X[mk])[:, 1]
                pl = logit.predict_proba((X[mk] - X[fit_mask].mean(0)) / (X[fit_mask].std(0) + 1e-9))[:, 1]
                inv = target[mk] == 0
                block[k] = dict(
                    tree=dict(auc=auc_score(target[mk], pt), error=float(((pt >= 0.5) != (target[mk] == 1)).mean()),
                              nan_recall=float((pt[inv] < 0.5).mean()) if inv.any() else float('nan')),
                    logit=dict(auc=auc_score(target[mk], pl), error=float(((pl >= 0.5) != (target[mk] == 1)).mean()),
                               nan_recall=float((pl[inv] < 0.5).mean()) if inv.any() else float('nan')))
            result[name] = block
            if name == 'truncated':
                result['tree_rules_truncated'] = export_text(tree, feature_names=['n0', 'n1', 'n2_8', 'n9', 'nout', 'prev1', 'prev2', 'prev6', 'hod'])
                np.savez_compressed(out / f'p_valid_{variable}.npz', labels=labels.asi8,
                                    tree=tree.predict_proba(X)[:, 1].astype(np.float32),
                                    logit=logit.predict_proba((X - X[fit_mask].mean(0)) / (X[fit_mask].std(0) + 1e-9))[:, 1].astype(np.float32))
        # simple baselines on the scored periods
        base = {}
        for k in ('conf', 'F1', 'F2', 'F3'):
            mk = split == k
            by_hour = np.where(np.isin(labels[mk].hour, (3, 9, 15, 21)), 0, 1)
            base[k] = dict(always_valid_error=float((target[mk] == 0).mean()), hour_rule_error=float((by_hour != target[mk]).mean()),
                           window_all_nominal_error=float(((full[mk][:, 0] >= 0.98 * n).astype(int) != target[mk]).mean()))
        result['baselines'] = base
        report['products'][variable] = result
        log(dict(event='validity', variable=variable, truncated_conf_tree_auc=result['truncated']['conf']['tree']['auc'],
                 full_conf_tree_auc=result['full']['conf']['tree']['auc']))
    (out / 'validity.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    return 0


# ------------------------------------------------------------------------------------------------- stage 3: downstream
BOOSTER_ROOT = ROOT / 'research/rerun_df_ffill_20261103/runs/e2000fs_calib_offset_roll'
TABLE = ROOT / 'research/rerun_df_ffill_20261103/input/df_ffill_trainy.parquet'
# model feature -> (variable, statistic over hourly bins ending at the origin bin)
ACE_FEATURES = {'v_lag0': ('speed', 1), 'ace_speed_kms_last': ('speed', 1), 'ace_density_cm3_last': ('density', 1),
                'ace_density_cm3_m6': ('density', 6), 'ace_temperature_k_last': ('temperature', 1),
                'ace_temperature_k_m24': ('temperature', 24), 'ace_bt_nt_last': ('bt', 1), 'ace_bt_nt_m6': ('bt', 6),
                'ace_by_gsm_nt_m24': ('by', 24), 'ace_epam_p_channel3_last': ('epam_p3', 1), 'ace_sis_gt10_flux_last': ('sis_gt10', 1)}
H3_THRESHOLD = {'speed': 0.40, 'density': 0.30, 'temperature': 0.30, 'bt': 0.30, 'by': 0.30, 'epam_p3': 0.20, 'sis_gt10': 0.20}


def bin_series(variable: str, hours: pd.DatetimeIndex) -> np.ndarray:
    """Cleaned monthly value of the hour labelled L, stored at bin L + 1 h (what the training table calls the bin)."""
    product, column = VARIABLES[variable]
    return monthly_target(product, column, hours - pd.Timedelta(hours=1))


def window_features(base: np.ndarray, newest: np.ndarray, width: int) -> np.ndarray:
    """Mean of the `width` hourly bins ending at each origin, with the newest bin replaced by `newest` (NaN-aware)."""
    x = np.where(np.isfinite(base), base, 0.0)
    c = np.isfinite(base).astype(float)
    cs, cc = np.r_[0.0, np.cumsum(x)], np.r_[0.0, np.cumsum(c)]
    idx = np.arange(len(base))
    lo = np.clip(idx - (width - 1), 0, len(base))
    prev_sum, prev_cnt = cs[idx] - cs[lo], cc[idx] - cc[lo]               # bins lo .. idx-1
    new = np.where(np.isfinite(newest), newest, 0.0)
    cnt = prev_cnt + np.isfinite(newest)
    with np.errstate(all='ignore'):
        return np.where(cnt > 0, (prev_sum + new) / cnt, np.nan)


def paired_bootstrap(se: dict, n: np.ndarray, days: np.ndarray, reference: str, oracle: str, draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    uniq = np.unique(days)
    index = {d: np.flatnonzero(days == d) for d in uniq}
    out = {name: [] for name in se}
    gap = {name: [] for name in se}
    diff = {name: [] for name in se}
    for _ in range(draws):
        pick = np.concatenate([index[d] for d in rng.choice(uniq, len(uniq))])
        total = n[pick].sum()
        rm = {name: np.sqrt(se[name][pick].sum() / total) for name in se}
        for name in se:
            out[name].append(rm[name])
            diff[name].append(rm[name] - rm[reference])
            gap[name].append((rm[reference] - rm[name]) / max(rm[reference] - rm[oracle], 1e-9))
    q = lambda a: [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]      # noqa: E731
    return {name: dict(rmse_ci=q(out[name]), diff_vs_reference_ci=q(diff[name]), gap_closed_ci=q(gap[name])) for name in se}


def cmd_downstream(args) -> int:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import xgboost as xgb
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    est_report = json.loads((Path(args.estimators) / 'estimators.json').read_text())['variables']
    meta = json.loads((BOOSTER_ROOT / 'F3_metadata.json').read_text())['features']
    cnames, vnames = meta['constant'], meta['varying']
    mapped_name = lambda c: c.replace('ch_ch_', 'suvi_', 1) if c.startswith('ch_ch_') else c      # noqa: E731
    # which estimator each variable would use, decided on conf only, and whether it passes H3 on F1/F2/F3
    chosen, passes = {}, {}
    for variable, r in est_report.items():
        cands = {k: r['conf'][k].get('rmse', 1e18) for k in ('E1', 'E3', 'E4_ridge', 'E5_gbm_mean') if k in r['conf']}
        cands.update({r['best_simple']: r['conf'][r['best_simple']].get('rmse', 1e18)})
        chosen[variable] = min(cands, key=cands.get)
        passes[variable] = all(r[k][chosen[variable]].get('rmse_vs_E0', 0) <= -H3_THRESHOLD[variable] for k in PERIODS)
    hours = pd.date_range('2025-11-01', '2026-10-01', freq='h', tz='UTC')            # origins (bin times), with warm-up history
    arrays = {v: np.load(Path(args.estimators) / f'hourly_estimates_{v}.npz') for v in est_report}
    base = {v: bin_series(v, hours) for v in sorted({x[0] for x in ACE_FEATURES.values()})}

    def newest_for(variant: str, variable: str) -> np.ndarray:
        """Value of the newest bin (label O - 1 h) for each origin hour under a variant."""
        arr = arrays.get(variable)
        labels = hours - pd.Timedelta(hours=1)
        pos = pd.Series(np.arange(len(arr['labels'])), index=pd.DatetimeIndex(arr['labels'], tz='UTC')).reindex(labels).to_numpy()
        take = lambda key: np.where(np.isfinite(pos), arr[key][np.nan_to_num(pos).astype(int)], np.nan)      # noqa: E731
        copy = np.r_[np.nan, base[variable][:-1]]                                       # container E0: copy the previous bin, NaN kept
        if variant == 'standard':
            return base[variable]
        if variant == 'E0':
            return copy
        if variant.startswith('est:'):
            est = take(variant[4:])
            return np.where(np.isfinite(est), est, copy)
        raise ValueError(variant)

    def build(variant_by_variable: dict) -> dict:
        feats = {}
        for name, (variable, width) in ACE_FEATURES.items():
            newest = newest_for(variant_by_variable.get(variable, 'E0'), variable)
            feats[name] = pd.Series(window_features(base[variable], newest, width), index=hours).ffill()
        return feats

    # origins and table rows
    table = ds.dataset(TABLE).to_table(columns=['origin', 'h', 'y', 'period'] + [mapped_name(c) for c in cnames] + vnames,
                                        filter=(ds.field('set') == 'query')).to_pandas()
    table = table[table.origin.dt.hour.isin([0, 6, 12, 18]) & table.period.isin(list(PERIODS))].sort_values(['origin', 'h'])
    origins = pd.DatetimeIndex(table.origin.iloc[::72].to_numpy())
    period_of = table.period.iloc[::72].to_numpy()
    n = len(origins)
    y = table.y.to_numpy(float).reshape(n, 72)
    C = table[[mapped_name(c) for c in cnames]].iloc[::72].to_numpy(np.float32)
    E = table[vnames].to_numpy(np.float32).reshape(n, 72, len(vnames))
    ace_cols = [cnames.index(k) for k in ACE_FEATURES]
    boosters, offsets = {}, {}
    for period in PERIODS:
        boosters[period] = []
        for b in range(3):
            booster = xgb.Booster()
            booster.load_model(str(BOOSTER_ROOT / f'{period}_models/full/spectral/b{b}.ubj'))
            boosters[period].append(booster)
        offsets[period] = np.array(json.loads((BOOSTER_ROOT / f'{period}_models/frozen_calibration.json').read_text())['offset'])

    def predict(constants: np.ndarray) -> np.ndarray:
        pred = np.empty((n, 72))
        for period in PERIODS:
            rows = np.flatnonzero(period_of == period)
            parts = []
            for booster, start in zip(boosters[period], (0, 24, 48)):
                x = np.hstack([constants[rows], E[rows, start:start + 24].reshape(len(rows), -1)]).astype(np.float32)
                parts.append(np.asarray(booster.inplace_predict(x), dtype=float))
            pred[rows] = np.clip(np.clip(np.hstack(parts), 1, 3000) + offsets[period], 1, 3000)
        return pred

    def constants_for(feats: dict) -> np.ndarray:
        out = C.copy()
        for name, series in feats.items():
            out[:, cnames.index(name)] = series.reindex(origins).to_numpy(np.float32)
        return out

    # gate: the recomputed 'standard' features must reproduce the table
    std_feats = build({v: 'standard' for v in est_report})
    recomputed = constants_for(std_feats)
    parity = {name: float(np.nanmax(np.abs(recomputed[:, cnames.index(name)] - C[:, cnames.index(name)]) /
                                    (np.abs(C[:, cnames.index(name)]) + 1e-6))) for name in ACE_FEATURES}
    gate_ok = max(parity.values()) < 1e-3
    report: dict = dict(experiment=EXPERIMENT, stage='downstream', git_commit=revision(), versions=versions(),
                        origins=int(n), periods={k: int((period_of == k).sum()) for k in PERIODS},
                        chosen_estimator=chosen, passes_h3=passes, table_parity_relative_max=parity, parity_gate=gate_ok)
    log(dict(event='parity', gate=gate_ok, worst=max(parity.values())))
    if not gate_ok and not args.force:
        (out / 'downstream.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
        return 1
    ok = np.isfinite(y)
    variants = {'V1_standard': {v: 'standard' for v in est_report}, 'V0_E0': {v: 'E0' for v in est_report}}
    for label, picker in (('V2_E1', lambda v: 'est:E1'), ('V3_best_simple', lambda v: 'est:' + est_report[v]['best_simple']),
                          ('V4_learned', lambda v: 'est:' + ('E5_gbm_mean' if est_report[v]['conf']['E5_gbm_mean'].get('rmse', 1e18) <= est_report[v]['conf']['E4_ridge'].get('rmse', 1e18) else 'E4_ridge')),
                          ('V4b_chosen', lambda v: 'est:' + chosen[v])):
        variants[label] = {v: picker(v) for v in est_report}
    variants['V4c_chosen_passing_only'] = {v: ('est:' + chosen[v] if passes[v] else 'E0') for v in est_report}
    variants['V4d_speed_only'] = {v: ('est:' + chosen[v] if v == 'speed' else 'E0') for v in est_report}
    # H5 ablation: replace only the `_last` features of the chosen estimator, keep rolling features on E0
    results, se, counts = {}, {}, {}
    day = np.array([f'{o:%Y-%m-%d}' for o in origins])
    preds = {}
    for name, mapping in variants.items():
        preds[name] = predict(constants_for(build(mapping)))
    last_only = build({v: 'E0' for v in est_report})
    chosen_feats = build({v: 'est:' + chosen[v] for v in est_report})
    for name, (variable, width) in ACE_FEATURES.items():
        if width == 1:
            last_only[name] = chosen_feats[name]
    preds['V4e_last_only'] = predict(constants_for(last_only))
    for name, pred in preds.items():
        err = np.where(ok, pred - y, 0.0)
        se[name], counts[name] = (err ** 2).sum(1), ok.sum(1)
    n_pairs = counts['V0_E0']
    for name in preds:
        per = {}
        for scope, mask in [('all', np.ones(n, bool))] + [(k, period_of == k) for k in PERIODS]:
            per[scope] = dict(rmse=float(np.sqrt(se[name][mask].sum() / n_pairs[mask].sum())))
            horizon = {}
            for lo, hi in ((1, 6), (7, 24), (25, 48), (49, 72)):
                e = np.where(ok[mask][:, lo - 1:hi], preds[name][mask][:, lo - 1:hi] - y[mask][:, lo - 1:hi], 0.0)
                horizon[f'h{lo}-{hi}'] = float(np.sqrt((e ** 2).sum() / max(ok[mask][:, lo - 1:hi].sum(), 1)))
            per[scope]['by_horizon'] = horizon
        results[name] = per
    base_rmse = {k: results['V0_E0'][k]['rmse'] for k in results['V0_E0']}
    orac_rmse = {k: results['V1_standard'][k]['rmse'] for k in results['V1_standard']}
    for name in results:
        for scope in results[name]:
            gap = base_rmse[scope] - orac_rmse[scope]
            results[name][scope]['gap_closed'] = float((base_rmse[scope] - results[name][scope]['rmse']) / gap) if gap > 1e-9 else None
            results[name][scope]['vs_E0_pct'] = float(results[name][scope]['rmse'] / base_rmse[scope] - 1)
    boot = {}
    for scope, mask in [('all', np.ones(n, bool))] + [(k, period_of == k) for k in PERIODS]:
        boot[scope] = paired_bootstrap({k: v[mask] for k, v in se.items()}, n_pairs[mask], day[mask], 'V0_E0', 'V1_standard', args.bootstrap, SEED)
    report.update(results=results, bootstrap=boot, bootstrap_draws=args.bootstrap, seed=SEED)
    (out / 'downstream.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    log(dict(event='downstream', V0=results['V0_E0']['F3']['rmse'], V1=results['V1_standard']['F3']['rmse'],
             V4=results['V4_learned']['F3']['rmse']))
    return 0


# ------------------------------------------------------------------------------------------------- entry point
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', choices=(EXPERIMENT,), default=EXPERIMENT)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('fetch', 'check'):
        p = sub.add_parser(name)
        p.add_argument('--start', default=START_DAY)
        p.add_argument('--end', help='last UTC day (default: yesterday)')
        p.add_argument('--delay', type=float, default=0.25, help='seconds slept before each request')
        p.add_argument('--device', choices=('cpu',), default='cpu')
        p.add_argument('--threads', type=int, default=1)
        p.add_argument('--out', default=str(HERE / 'runs/stage0'))
        if name == 'fetch':
            p.add_argument('--workers', type=int, default=2)
        else:
            p.add_argument('--sample', type=int, default=60)
    p = sub.add_parser('identify')
    p.add_argument('--variables', nargs='*', choices=list(VARIABLES))
    p.add_argument('--offset-min', type=int, default=-180)
    p.add_argument('--offset-max', type=int, default=180)
    p.add_argument('--length-min', type=int, default=1)
    p.add_argument('--length-max', type=int, default=180)
    p.add_argument('--device', choices=('cpu',), default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--out', default=str(HERE / 'runs/stage1'))
    p = sub.add_parser('estimate')
    p.add_argument('--variables', nargs='*', choices=list(VARIABLES))
    p.add_argument('--identification', default=str(HERE / 'runs/stage1/identification.json'))
    p.add_argument('--lag', type=float, help='publication lag in minutes (default per product)')
    p.add_argument('--device', choices=('cpu',), default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--out', default=str(HERE / 'runs/stage2'))
    p = sub.add_parser('validity')
    p.add_argument('--variables', nargs='*', choices=list(VARIABLES))
    p.add_argument('--identification', default=str(HERE / 'runs/stage1/identification.json'))
    p.add_argument('--device', choices=('cpu',), default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--out', default=str(HERE / 'runs/stage1'))
    p = sub.add_parser('downstream')
    p.add_argument('--estimators', default=str(HERE / 'runs/stage2'))
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--force', action='store_true', help='continue even if the table parity gate fails')
    p.add_argument('--device', choices=('cpu',), default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--out', default=str(HERE / 'runs/stage3'))
    args = parser.parse_args()
    return {'fetch': cmd_fetch, 'check': cmd_check, 'identify': cmd_identify, 'estimate': cmd_estimate,
            'validity': cmd_validity, 'downstream': cmd_downstream}[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
