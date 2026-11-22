"""Union of the nine overnight arms that improved on the leaderboard in at least two seeds, on the seam row.

Published on explicit user request (2026-11-22) as an exception: every component was selected on F1/F2/F3 leaderboard
scores, most are below the 1 % gate, so the score is leaderboard-selected, not a clean generalisation estimate, and the
row is not a cumulative baseline. All fitting uses pre-cutoff training data only; the composition, order and constants
were fixed before the union was scored. Because selection used the evaluation periods, the row does NOT meet the
leaderboard rule "trained (and tuned) only on data before each cutoff"; no exception approval is recorded.

  GBM      smooth_tail_nospec's 223 inputs + 5 origin constants (proton entropy and N/B co-compression in every block;
           SWPC HSS coverage, proton ram pressure and SRS eruptive sunspot area in h49-72), A900 at 600 rounds
           (widths 225 / 225 / 228); S-cap 6 x 30 and the 336 x 0.25 offset are refitted
  post-hoc seam xf8 -> 5 % weighted ridge on the 223 inputs -> 3 h same-issue phase advance ->
           seed-agreement uplift (<= 40 km/s) -> fixed equal mean of seeds 42/43/44

Usage (CPU only):
  CUDA_VISIBLE_DEVICES= python experiment.py --threads 6 [--donki JSON --srs DIR --discussion DIR]
  CUDA_VISIBLE_DEVICES= python experiment.py --predict-only [...]        # GBM and ridge from weights/
  CUDA_VISIBLE_DEVICES= python experiment.py --smoke --out DIR [...]     # 2-round wiring check, no post-hoc, no score
Without the three corpus arguments the public DONKI, NCEI SRS and SWPC discussion texts are downloaded. --discussion
needs txt/ and hdr/ (HTTP response headers: Last-Modified is the availability time).
Outputs: weights/F{1,2,3}/{s{seed},ridge}.npz; results/{s42,s43,s44,mean3}_{predictions.parquet,score.json} and logs/.
"""
from __future__ import annotations

import argparse
import dataclasses
import email.utils
import importlib.util
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'  # ridge: deterministic single-thread BLAS
os.environ['OMP_NUM_THREADS'] = sys.argv[sys.argv.index('--threads') + 1] if '--threads' in sys.argv[:-1] else '6'

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from _module import calibration as cal, common, gbm, public_events  # noqa: E402
from _module.table import load_grid, load_table, subset  # noqa: E402

EXPERIMENT = 'e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam_union9_lbsel'
PARENT = 'e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec'
SEAM_ROW = 'e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam'


def _load(row):
    loader = importlib.util.spec_from_file_location(f'{row}_experiment', HERE.parent / row / 'experiment.py')
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


ANCHOR, SEAM = _load(PARENT), _load(SEAM_ROW)
GLOBAL = ['ace_log_proton_entropy', 'ace_nb_cocompression_6h']
LATE = ['swpc_hss_coverage49_72', 'ace_proton_ram_pressure_npa', 'srs_earthward_eruptive_area']
ADDED, LATE_START = GLOBAL + LATE, 48
RIDGE_WEIGHT, RIDGE_PENALTY = .05, .01
H, MISSING = common.H, -999.
HOUR = pd.Timedelta(hours=1)
REPORTS = 'https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports'


@dataclasses.dataclass
class UnionSpec(gbm.GBMSpec):
    def design(self, grid, start, size):
        x = super().design(grid, start, size)
        tail = x[:, -len(ADDED):]
        tail[tail == MISSING] = np.nan
        return x if start == LATE_START else np.ascontiguousarray(x[:, :-len(LATE)])

    def feature_names(self, start, size):
        names = super().feature_names(start, size)
        return names if start == LATE_START else names[:-len(LATE)]


_launch = {k: v for k, v in ANCHOR.SPEC.launch.items() if k != 'source_ablation'}
_launch.update(parent_experiment=PARENT, seam_row=SEAM_ROW, added_constant=ADDED, added_varying=[], removed_features=[],
               classification='user-requested exception: union of leaderboard-selected arms', block_widths=[225, 225, 228],
               selection='arms improved in >= 2 seeds on the F1/F2/F3 leaderboard (LB-selected)',
               a_stage={**ANCHOR.SPEC.launch['a_stage'], 'rounds': 600}, tree_budget=600 + 180,
               postprocess=['seam xf8', 'ridge 0.05', 'phase advance 3 h', 'seed-agreement uplift 40'],
               source_ablation='~/sunrun-overnight-loop union9_lbsel')
_fields = {f.name: getattr(ANCHOR.SPEC, f.name) for f in dataclasses.fields(ANCHOR.SPEC)}
_fields.update(experiment=EXPERIMENT, constant=[*ANCHOR.SPEC.constant, *ADDED], threads=6, rounds=600, launch=_launch)
SPEC = UnionSpec(**_fields)


# ---- public corpora --------------------------------------------------------------------------------------------------
def get(url):
    """(body, headers), or None for a file the archive does not have."""
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read(), response.headers
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if attempt == 3:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
        time.sleep(5 * (attempt + 1))


def fetch_srs(folder):
    folder.mkdir(parents=True)

    def one(day):
        name = f'{day:%Y%m%d}SRS.txt'
        found = get(f'{REPORTS}/solar_region_summaries/{day:%Y/%m}/{name}')
        if found:
            (folder / name).write_bytes(found[0])

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(one, pd.date_range('2022-09-25', '2026-09-26', freq='D')))
    return folder


def fetch_discussion(folder):
    (folder / 'txt').mkdir(parents=True)
    (folder / 'hdr').mkdir()
    jobs = []
    for month in pd.date_range('2024-03-01', '2026-09-01', freq='MS'):
        base = f'{REPORTS}/discussion/{month:%Y/%m}'
        index = get(base + '/')
        if index:
            names = set(re.findall(r'\d{12}forecast_discussion\.txt', index[0].decode(errors='replace')))
            jobs += [(base, name) for name in sorted(names)]

    def one(job):
        found = get('/'.join(job))
        if found:
            (folder / 'txt' / job[1]).write_bytes(found[0])
            (folder / 'hdr' / job[1]).write_text(''.join(f'{k}: {v}\n' for k, v in found[1].items()))

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(one, jobs))
    return folder


def issued_at(text):
    match = re.search(r'^:Issued:\s*(\d{4} \w{3} \d{1,2} \d{4}) UTC', text, re.M)
    return pd.Timestamp(pd.to_datetime(match.group(1), format='%Y %b %d %H%M'), tz='UTC') if match else None


# ---- NOAA Solar Region Summary: Earth-facing eruptive sunspot area -----------------------------------------------------
SRS_CLASSES = {'FKC', 'EKC', 'DKC', 'FKI', 'EKI', 'DKI'}
SRS_LOCATION = re.compile(r'^([NS])(\d{2})([EW])(\d{2})$')


def srs_report(text, day):
    issued = issued_at(text)
    if issued is None:
        return None
    available = max(issued, day + pd.Timedelta(minutes=40)) + HOUR
    lines = text.splitlines()
    try:
        first = next(i for i, l in enumerate(lines) if l.strip().startswith('I.') and 'Regions with Sunspots' in l)
        last = next(i for i, l in enumerate(lines) if i > first and l.strip().startswith('IA.'))
    except StopIteration:
        return available, np.nan
    body = [l for l in lines[first + 1:last] if l.strip()]
    if not body or body[0].split()[:4] != ['Nmbr', 'Location', 'Lo', 'Area']:
        return available, np.nan
    total = 0.
    for line in body[1:]:
        token = line.split()
        if token == ['None']:
            continue
        if len(token) < 5:
            return available, np.nan
        location = SRS_LOCATION.match(token[1])
        try:
            area = float(token[3])
        except ValueError:
            return available, np.nan
        if location is None or not 0 <= area <= 10000:
            return available, np.nan
        if token[4].upper() in SRS_CLASSES and int(location.group(2)) <= 40 and int(location.group(4)) <= 45:
            total += area
    return available, float(np.log1p(total / 100.))


def srs_area(folder, origins):
    rows = []
    for day in pd.date_range('2022-09-25', '2026-09-26', freq='D', tz='UTC'):
        path = folder / f'{day:%Y%m%d}SRS.txt'
        if path.exists():
            report = srs_report(path.read_text(errors='replace'), day)
            if report is not None:
                rows.append(report)
    reports = pd.DataFrame(rows, columns=['a', 'b']).sort_values('a', kind='stable')
    a, b = pd.DatetimeIndex(reports.a).as_unit('ns').asi8, reports.b.to_numpy(float)
    o = origins.as_unit('ns').asi8
    index = np.searchsorted(a, o, side='right') - 1
    at = np.maximum(index, 0)
    use = (index >= 0) & ((o - a[at]) / 3.6e12 <= 36) & np.isfinite(b[at])
    return np.where(use, b[at], MISSING).astype(np.float32)


# ---- SWPC Forecast Discussion: share of h49-72 on dates named for coronal-hole high-speed streams ----------------------
MONTHS = {m: i for i, m in enumerate('jan feb mar apr may jun jul aug sep oct nov dec'.split(), 1)}
_M = '(' + '|'.join(MONTHS) + r')\w*'
HSS_DATE = re.compile(rf'\b(\d{{1,2}})\s+{_M}\s*-\s*(\d{{1,2}})\s+{_M}\b'
                      rf'|\b(\d{{1,2}})\s*-\s*(\d{{1,2}})\s+{_M}\b'
                      rf'|\bthrough\s+(\d{{1,2}})\s+{_M}\b'
                      rf'|\b(\d{{1,2}})\s+{_M}\b')
HSS = re.compile(r'\bhss\b|high[- ]speed streams?')
HSS_POSITIVE = re.compile(r'\b(increas|enhanc|arriv|onset\b|persist|continu|influenc|geoeffective\b)')
HSS_NEGATIVE = re.compile(r'\b(cmes?\b|transient|no\b|not\b|neither\b|unlikely\b|decreas|declin|diminish|wane|weakening\b)')


def hss_dates(text, issue):
    """Set of UTC dates; None = ambiguous."""
    def resolve(day, month):
        days = (issue.floor('D') + pd.Timedelta(days=k) for k in range(-1, 4))
        hit = [d for d in days if d.day == day and d.month == month]
        if len(hit) != 1:
            raise ValueError('date')
        return hit[0]

    section = re.search(r'^Solar Wind\s*$(.*?)^Geospace\s*$', text, re.M | re.S)
    if not section or '.Forecast...' not in section.group(1):
        return None
    forecast = re.sub(r'[‐-―−]', '-', section.group(1).split('.Forecast...', 1)[1])
    forecast = re.sub(r'\s+', ' ', forecast).strip()
    dates = set()
    for sentence in re.split(r'(?<=\.)\s+(?=[A-Z0-9])', forecast):
        s = sentence.lower()
        if not HSS.search(s):
            continue
        if HSS_NEGATIVE.search(s):
            return None
        if not HSS_POSITIVE.search(s):
            continue
        found = list(HSS_DATE.finditer(s))
        if len(found) != 1:
            return None
        g = found[0].groups()
        try:
            if g[0]:
                low, high = resolve(int(g[0]), MONTHS[g[1]]), resolve(int(g[2]), MONTHS[g[3]])
            elif g[4]:
                low, high = resolve(int(g[4]), MONTHS[g[6]]), resolve(int(g[5]), MONTHS[g[6]])
            elif g[7]:
                low, high = issue.floor('D'), resolve(int(g[7]), MONTHS[g[8]])
            else:
                low = high = resolve(int(g[9]), MONTHS[g[10]])
        except ValueError:
            return None
        if high < low:
            return None
        dates.update(pd.date_range(low, high, freq='D'))
    return dates


def hss_coverage(folder, origins):
    rows, seen = [], {}
    for path in sorted((folder / 'txt').glob('*forecast_discussion.txt')):
        text, modified = path.read_text(errors='replace'), None
        for line in (folder / 'hdr' / path.name).read_text(errors='replace').splitlines():
            if line.lower().startswith('last-modified:'):
                modified = pd.Timestamp(email.utils.parsedate_to_datetime(line.split(':', 1)[1].strip())).tz_convert('UTC')
        issue = issued_at(text)
        if issue is None or modified is None:
            continue
        body = re.sub(r'^:Product:.*$', '', text, flags=re.M).strip()
        if issue in seen:
            if seen[issue] != body:
                raise ValueError(f'different bodies for one issue time {issue}')
            continue
        seen[issue] = body
        rows.append((issue, max(issue, modified) + HOUR, hss_dates(text, issue)))
    reports = pd.DataFrame(rows, columns=['issue', 'a', 'dates']).sort_values('a', kind='stable').reset_index(drop=True)
    index = np.searchsorted(pd.DatetimeIndex(reports.a).as_unit('ns').asi8, origins.as_unit('ns').asi8, side='right') - 1
    out = np.full(len(origins), MISSING, np.float32)
    leads = pd.to_timedelta(np.arange(49, 73), unit='h')
    for j, (origin, i) in enumerate(zip(origins, index)):
        if i < 0:
            continue
        report = reports.iloc[i]
        if origin - report.issue > 18 * HOUR or report.dates is None:
            continue
        out[j] = np.float32(np.mean([day in report.dates for day in (origin + leads).floor('D')]))
    return out


# ---- ACE features ----------------------------------------------------------------------------------------------------
def entropy(t, n):
    t, n = np.asarray(t, np.float64), np.asarray(n, np.float64)
    ok = np.isfinite(t) & np.isfinite(n) & (t > 0) & (n > 0)
    out = np.full(len(t), MISSING, np.float64)
    out[ok] = np.log10(t[ok]) - 5. - (2. / 3.) * np.log10(n[ok])
    return out.astype(np.float32)


def cocompression(n, b):
    """Consecutive hourly series -> median over the last 6 h of min(log2 n, log2 Bt rise vs the preceding 24 h)."""
    n, b = np.asarray(n, np.float64), np.asarray(b, np.float64)
    valid = np.isfinite(n) & np.isfinite(b) & (n > 0) & (b > 0)
    ln = np.where(valid, np.log2(np.where(valid, n, 1)), np.nan)
    lb = np.where(valid, np.log2(np.where(valid, b, 1)), np.nan)
    out = np.full(len(n), MISSING, np.float64)
    for i in range(29, len(n)):
        before, recent = slice(i - 29, i - 5), slice(i - 5, i + 1)
        rv, wv = valid[before], valid[recent]
        if rv.sum() < 12 or wv.sum() < 3:
            continue
        nr, br = np.median(ln[before][rv]), np.median(lb[before][rv])
        out[i] = np.median(np.minimum(ln[recent][wv] - nr, lb[recent][wv] - br))
    return out.astype(np.float32)


def ram_pressure(n, v):
    n, v = (np.asarray(x, np.float32).astype(np.float64) for x in (n, v))
    ok = np.isfinite(n) & np.isfinite(v) & (n > 0) & (v > 0)
    return np.where(ok, 1.6726e-6 * n * v ** 2, MISSING).astype(np.float32)


def add_features(source, destination, srs, discussion):
    """Copy the table row group by row group and append the five columns."""
    raw = pd.read_parquet(source, columns=['origin', 'h', 'ace_density_cm3_last', 'ace_bt_nt_last'])
    per = raw[raw.h.eq(1)].groupby('origin')[['ace_density_cm3_last', 'ace_bt_nt_last']]
    if (per.nunique(dropna=False) > 1).any().any():
        raise ValueError('conflicting duplicate origin values')
    series = per.first().sort_index()
    origins = pd.DatetimeIndex(series.index)
    if not (np.diff(origins.as_unit('ns').asi8) == 3600 * 10 ** 9).all():
        raise ValueError('origins are not consecutive hours')
    values = {'ace_nb_cocompression_6h': cocompression(series.ace_density_cm3_last, series.ace_bt_nt_last),
              'swpc_hss_coverage49_72': hss_coverage(discussion, origins),
              'srs_earthward_eruptive_area': srs_area(srs, origins)}
    del raw, per, series
    filled = load_table(source, ['ace_density_cm3_last', 'ace_speed_kms_last'], lambda *a, **k: None)
    first = filled[filled.h.eq(1)].drop_duplicates('origin').set_index('origin').reindex(origins)
    values['ace_proton_ram_pressure_npa'] = ram_pressure(first.ace_density_cm3_last, first.ace_speed_kms_last)
    del filled, first
    parquet, writer = pq.ParquetFile(source), None
    for i in range(parquet.num_row_groups):
        table = parquet.read_row_group(i)
        position = origins.get_indexer(pd.DatetimeIndex(table['origin'].to_pandas()))
        if (position < 0).any():
            raise ValueError('origin without features')
        columns = {name: value[position] for name, value in values.items()}
        columns['ace_log_proton_entropy'] = entropy(table['ace_temperature_k_last'].to_numpy(zero_copy_only=False),
                                                    table['ace_density_cm3_last'].to_numpy(zero_copy_only=False))
        for name in ADDED:
            table = table.append_column(name, pa.array(columns[name], type=pa.float32()))
        writer = writer or pq.ParquetWriter(destination, table.schema, compression='zstd')
        writer.write_table(table, row_group_size=table.num_rows)
    writer.close()


# ---- fixed post-hoc part ---------------------------------------------------------------------------------------------
def scale(x, stats=None):
    x = np.array(x, np.float64)
    if stats is None:
        median = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in x.T])
    else:
        median, mean, std = stats
    i, j = np.nonzero(~np.isfinite(x))
    x[i, j] = median[j]
    if stats is None:
        mean, std = x.mean(0), x.std(0)
        std[std < 1e-12] = 1.
    return np.column_stack([np.ones(len(x)), (x - mean) / std]), (median, mean, std)


def ridge_fit(x, y, weights):
    z, stats = scale(x)
    diagonal = np.diag([0.] + [1.] * (z.shape[1] - 1))
    theta = np.empty((z.shape[1], y.shape[1]))
    for lead in range(y.shape[1]):
        zw = z.T * weights[:, lead]
        left, right = zw @ z + RIDGE_PENALTY * weights[:, lead].sum() * diagonal, zw @ y[:, lead]
        theta[:, lead] = np.linalg.solve(left, right)
        if np.abs(left @ theta[:, lead] - right).max() / max(1., np.abs(right).max()) > 1e-9:
            raise ValueError('ridge linear system residual')
    return (*stats, theta)


def ridge_forecast(data, weights, predict_only, logs):
    """Deterministic weighted ridge on the parent's 223 inputs with its own training-only offset."""
    parent, names = ANCHOR.SPEC, ('median', 'mean', 'std', 'theta')
    table = load_table(data, parent.constant + parent.varying, lambda *a, **k: None, keep_observed=parent.keep_observed)
    starts, parts = range(0, H, parent.block), []
    for period in common.CUTOFFS:
        query, _ = load_grid(table, period, False, parent.constant, parent.varying)
        path = weights / period / 'ridge.npz'
        if not predict_only:
            train, y = load_grid(table, period, True, parent.constant, parent.varying)
            reference, calibration, _ = cal.prior_split(train['origins'])
            held, saved = subset(train, calibration), {}
            prior = np.empty((len(calibration), H))
            for stage, rows in (('prior', reference), ('full', np.arange(len(y)))):
                grid = subset(train, rows)
                for b, start in enumerate(starts):
                    target = y[rows][:, start:start + parent.block]
                    model = ridge_fit(parent.design(grid, start, parent.block), target,
                                      parent.make_objective(target, None).weights)
                    if stage == 'prior':
                        prior[:, start:start + parent.block] = scale(parent.design(held, start, parent.block),
                                                                     model[:3])[0] @ model[3]
                    for name, value in zip(names, model):
                        saved[f'{stage}_b{b}_{name}'] = value
                    logs('run', 'ridge_fit', period=period, stage=stage, block=b, train_rows=len(rows))
            saved['offset'] = cal.estimate_offset(prior, y[calibration])[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **saved)
        with np.load(path) as arrays:
            blocks = []
            for b, start in enumerate(starts):
                *stats, theta = (arrays[f'full_b{b}_{name}'] for name in names)
                blocks.append(scale(parent.design(query, start, parent.block), stats)[0] @ theta)
            parts.append(cal.apply_offset(np.hstack(blocks), arrays['offset']))
    return np.vstack(parts)


def blend(gbm_forecast, ridge):
    return gbm_forecast + RIDGE_WEIGHT * (ridge - gbm_forecast)


def phase_advance(forecast):
    h = np.arange(1, H + 1)
    ahead = np.minimum(h + 6, H)
    gate = np.clip(np.minimum((h - 24) / 6, (72 - h) / 6), 0, 1)
    weight = np.where((h >= 25) & (h <= 71), 3 * gate / np.maximum(ahead - h, 1), 0.)
    return np.clip(forecast + weight[None, :] * (forecast[:, ahead - 1] - forecast), 1., 3000.)


def uplift(forecasts):
    stack = np.stack(forecasts)
    mean, spread = stack.mean(0), stack.std(0, ddof=1)
    lead = np.clip((np.arange(1, H + 1) - 24) / 24, 0, 1)
    agree = np.clip(1 - spread / 5, 0, 1)
    band = np.clip(np.minimum((mean - 500) / 50, (650 - mean) / 50), 0, 1)
    delta = 40 * lead[None, :] * agree * band
    return [np.where(delta > 0, np.clip(f + delta, 1., 3000.), f) for f in stack]


def postprocess(forecasts, ridge):
    """seam xf8 -> ridge blend -> phase advance -> uplift, per seed."""
    smooth_ridge = SEAM.smooth(ridge)
    return uplift([phase_advance(blend(SEAM.smooth(f), smooth_ridge)) for f in forecasts])


# ---- entry point -----------------------------------------------------------------------------------------------------
def main():
    if {'-h', '--help'}.intersection(sys.argv[1:]):
        gbm.main(SPEC, __doc__, HERE)
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--donki', type=Path, help='saved DONKI CME notifications JSON')
    parser.add_argument('--srs', type=Path, help='saved NOAA SRS text folder')
    parser.add_argument('--discussion', type=Path, help='saved SWPC discussion folder (txt/, hdr/)')
    parser.add_argument('--data', type=Path, default=common.DATA)
    known, rest = parser.parse_known_args()
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument('--out', type=Path)
    shared.add_argument('--weights', type=Path)
    shared.add_argument('--observations', type=Path)
    shared.add_argument('--seeds', type=int, nargs='+', default=list(common.SEEDS))
    shared.add_argument('--periods', nargs='+', default=list(common.CUTOFFS))
    shared.add_argument('--predict-only', action='store_true')
    shared.add_argument('--smoke', action='store_true')
    args = shared.parse_known_args(rest)[0]
    common.output_paths(args, HERE, EXPERIMENT)
    with tempfile.TemporaryDirectory(prefix=f'{EXPERIMENT}-') as folder:
        work = Path(folder)
        corpus = known.donki or public_events.fetch_notifications(work / 'donki_notifications.json')
        srs = known.srs or fetch_srs(work / 'srs')
        discussion = known.discussion or fetch_discussion(work / 'discussion')
        base, table = work / 'base.parquet', work / 'table.parquet'
        SPEC.launch['donki'] = {**public_events.augment(known.data, corpus, base),
                                'df_source': common.file_identity(known.data)}
        add_features(base, table, srs, discussion)
        gbm.score = lambda *a, **k: None  # the pre-transform forecasts are never scored
        sys.argv = [sys.argv[0], *rest, '--data', str(table)]
        gbm.main(SPEC, __doc__, HERE)
        if args.smoke or sorted(args.seeds) != list(common.SEEDS) or args.periods != list(common.CUTOFFS):
            return
        logs = common.run_logs(args, EXPERIMENT)
        results = args.out / 'results'
        raw = [pd.read_parquet(results / f's{seed}_predictions.parquet') for seed in common.SEEDS]
        ridge = ridge_forecast(base, args.weights, args.predict_only, logs)
        final = postprocess([f.pred_kms.to_numpy(np.float64).reshape(-1, H) for f in raw], ridge)
        observations = common.observations_path(args.observations)
        frames = []
        for seed, frame, matrix in zip(common.SEEDS, raw, final):
            out = frame[common.KEYS].copy()
            out['pred_kms'] = matrix.ravel()
            scores = common.score(out, observations)
            common.write_results(results, f's{seed}', out, scores)
            logs(f's{seed}', 'scored', total_s=scores['competition']['total']['s'])
            frames.append(out)
        ensemble = common.mean3(frames)
        scores = common.score(ensemble, observations)
        common.write_results(results, 'mean3', ensemble, scores)
        logs('run', 'complete', postprocess=_launch['postprocess'], total_s=scores['competition']['total']['s'])


if __name__ == '__main__':
    main()
