"""ace_delay_decision: what the ACE publication delay costs e2000fs_calib_offset_roll, and how to handle it.

At an on-the-hour t0 the NASA monthly ACE files hold rows up to label t0-2h (refresh ~:10), while the training
table uses label t0-1h. This file measures, on the official scorer S over F1/F2/F3 and seeds 42/43/44 (thresholds in
strategy.md, fixed before running):

    views   ACE inputs under each visibility: raw monthly files -> the container's own feature code.
            gates G1 (raw rebuild == df), G2 (container k=0 == df), G6 (shift parquet); writes input/df_shift{k}.parquet
    score   stored seed-42/43/44 boosters (+ trained shift models) on every view -> S, RMSE, MAE, lead buckets,
            paired day-block bootstrap; gates G3 (stored predictions / recorded S) and G4 (ffill_cost RMSE)
    train   the e2000fs_calib_offset_roll training path (default configuration only) on a shift-k input

The leaderboard truth is used for scoring only. CPU only. No file fingerprints anywhere.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import numpy as np
import pandas as pd

EXPERIMENT = 'ace_delay_decision'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'research/rerun_df_ffill_20261103/input/df_ffill_trainy.parquet'
OBSERVATIONS = ROOT / 'research/offset_roll_complete_ablation_20261106/competition-observations.parquet'
BASE_RUNS = ROOT / 'research/offset_roll_complete_ablation_20261106/runs'
RAW = ROOT / 'data-v2/raw/ACE/monthly-txt'
BENCHLAB = ROOT / 'benchlab-submission'
SCORER = ROOT / 'leaderboard/scorer/competition.py'
ACE_1M_ESTIMATES = ROOT / 'research/ace_1m_to_hourly_20261108/runs/stage2'

SEEDS = (42, 43, 44)
PERIODS = ('F1', 'F2', 'F3')
RECORDED_S = {42: 98.7626868541542, 43: 99.93218855943535, 44: 100.21277355950158}
FFILL_COST_F3_S42 = dict(standard=55.67, container=55.90)          # benchlab-submission/tests/ffill_cost.py, 2026-11-07
# the 11 ACE-derived constant inputs of the model; everything else stays byte-identical
ACE11 = ['v_lag0', 'ace_speed_kms_last', 'ace_density_cm3_last', 'ace_density_cm3_m6', 'ace_temperature_k_last',
         'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6', 'ace_by_gsm_nt_m24', 'ace_epam_p_channel3_last',
         'ace_sis_gt10_flux_last']
# k bins late: a row labelled L is visible at t0 when L + (10 + 60 k) min <= t0  (k=1 is the live situation, :10 refresh)
DELAY_MIN = {0: 10, 1: 70, 2: 130}
SHIFTS = (1, 2)
LATE_SHARE = 2 / 42                       # hours whose refresh did not arrive within 55 min (hourly/ace_watch.jsonl)
BOOT_DRAWS, BOOT_SEED, MIX_DRAWS = 2000, 20261109, 20
BUCKETS = ((1, 6), (7, 24), (25, 48), (49, 72))
RAW_END = pd.Timestamp('2026-10-01', tz='UTC')
# 1-minute variables of research/ace_1m_to_hourly_20261108 -> hourly table column of the container
ONE_MINUTE = {'speed': 'ace_speed_kms', 'density': 'ace_density_cm3', 'temperature': 'ace_temperature_k',
              'bt': 'ace_bt_nt', 'by': 'ace_by_gsm_nt', 'epam_p3': 'ace_epam_p_channel3', 'sis_gt10': 'ace_sis_gt10_flux'}
# research/ace_1m_to_hourly_20261108 H3 (RMSE reduction vs forward fill needed in each of F1, F2, F3)
H3_THRESHOLD = {'speed': 0.40, 'density': 0.30, 'temperature': 0.30, 'bt': 0.30, 'by': 0.30, 'epam_p3': 0.20, 'sis_gt10': 0.20}

# training configuration of the baseline default path (runs/baseline_s42/experiment.py DEFAULTS)
TAIL_WEIGHT, SPECTRAL_COEFFICIENT, SPECTRAL_SCALE, ROUNDS, BLOCK = 4., .1, 50., 900, 24
OFFSET_WINDOW, ALPHA = 336, .25
POLICY = 'training-only frozen calibration; official labels used only for scoring'


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


baseline = load_module('baseline', BASE_RUNS / 'baseline_s42/baseline.py')     # shared runner: loader, split, params


def names() -> dict:
    return baseline.selected_names(baseline.FEATURES)


def log(record: dict) -> None:
    print(json.dumps(record, default=str), flush=True)


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, default=str, allow_nan=False) + '\n')
    tmp.replace(path)


def file_identity(path: Path) -> dict:
    st = Path(path).stat()
    return dict(path=str(Path(path).resolve()), bytes=st.st_size, mtime_ns=st.st_mtime_ns)


def provenance() -> dict:
    import xgboost as xgb
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=False)
    dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, capture_output=True,
                           text=True, check=False)
    return dict(git_commit=git.stdout.strip(), tracked_changes=bool(dirty.stdout.strip()),
                versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                              xgboost=xgb.__version__))


# ------------------------------------------------------------------------------------------------ views
def benchlab():
    sys.path.insert(0, str(BENCHLAB))
    from pipeline.collect import ace as collect                              # noqa: E402
    from pipeline.contract import HOUR, LOOKBACK_ORIGINS                     # noqa: E402
    from pipeline.preprocess import ace_features                            # noqa: E402
    return collect, ace_features, HOUR, LOOKBACK_ORIGINS


def scored_origins() -> tuple[pd.DatetimeIndex, np.ndarray, dict]:
    competition = load_module('competition', SCORER)
    context = competition.prepare_context(competition.load_observations(OBSERVATIONS))
    return context['origins'], context['period'], context


def visible(collect, text: str, t0: pd.Timestamp, delay: pd.Timedelta) -> str:
    """collect.as_of with an explicit delay: header lines plus rows with label + delay <= t0."""
    kept = [line for line in text.splitlines()
            if (label := collect.row_label(line)) is None or label + delay <= t0]
    return '\n'.join(kept) + '\n'


def container_features(collect, af, hour, lookback, texts: dict, t0: pd.Timestamp, k: int, replace=None) -> pd.Series:
    """ACE features exactly as benchlab-submission computes them at t0 when rows are k bins late."""
    delay = pd.Timedelta(minutes=DELAY_MIN[k])
    bundle = {p: [visible(collect, texts[(p, m)], t0, delay) for m in collect.months_for(t0) if (p, m) in texts]
              for p in af.SPEC}
    table, _ = af.hourly_bins(bundle, t0)
    if replace is not None:
        table = replace(table, t0)
    window = pd.date_range(t0 - (lookback - 1) * hour, t0, freq='h')
    return af.features_at(table, window).ffill().iloc[-1][ACE11]


def oracle_series(af) -> pd.DataFrame:
    """k=0 features on the full hourly grid from the final monthly files, causally forward filled (the df rule)."""
    bundle = {p: [f.read_text() for f in sorted(RAW.glob(f'*_ace_{p}_1h.txt'))] for p in af.SPEC}
    table, _ = af.hourly_bins(bundle, RAW_END)
    return af.features_at(table, table.index)[ACE11].ffill()


def same(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """float32 rounding tolerance; NaN equals NaN."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    close = np.abs(a - b) <= 1e-6 * np.maximum(1.0, np.abs(a))
    return close | (np.isnan(a) & np.isnan(b))


def write_shift(oracle: pd.DataFrame, k: int, out: Path) -> dict:
    """Copy of the training table with the 11 ACE columns replaced by the k=0 feature of origin - k h."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    src = pq.ParquetFile(DATA)
    shifted = oracle.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=k)                  # value of origin O - k stored at O
    tmp = out.with_suffix('.tmp.parquet')
    writer = pq.ParquetWriter(tmp, src.schema_arrow)
    missing = 0
    for g in range(src.num_row_groups):
        t = src.read_row_group(g)
        origin = pd.DatetimeIndex(t['origin'].to_pandas())
        for name in ACE11:
            i = t.schema.get_field_index(name)
            values = shifted[name].reindex(origin).to_numpy(np.float32)
            missing += int(np.isnan(values).sum())
            t = t.set_column(i, t.schema.field(i), pa.array(values, type=t.schema.field(i).type, mask=np.isnan(values)))
        writer.write_table(t, row_group_size=t.num_rows)
    writer.close()
    tmp.replace(out)
    # G6: every other column identical, the 11 columns = oracle(O - k)
    a, b = pq.ParquetFile(DATA), pq.ParquetFile(out)
    other = [n for n in a.schema_arrow.names if n not in ACE11]
    unequal, ace_bad = [], 0
    for g in range(a.num_row_groups):
        ta, tb = a.read_row_group(g), b.read_row_group(g)
        unequal += [n for n in other if not ta.column(n).equals(tb.column(n))]
        origin = pd.DatetimeIndex(tb['origin'].to_pandas())
        for name in ACE11:
            want = oracle[name].reindex(origin - pd.Timedelta(hours=k)).to_numpy(np.float32)
            ace_bad += int((~same(tb[name].to_numpy(zero_copy_only=False), want)).sum())
    return dict(path=str(out), bytes=out.stat().st_size, rows=b.metadata.num_rows, row_groups=b.num_row_groups,
                schema_equal=a.schema_arrow.equals(b.schema_arrow), other_columns_unequal=sorted(set(unequal)),
                ace_cells_wrong=ace_bad, ace_cells_nan=missing,
                passed=a.schema_arrow.equals(b.schema_arrow) and not unequal and ace_bad == 0)


def one_minute_replacer(estimates: dict, mode: str, variables=None):
    """Hourly table rewrite for C_1m_last (newest bin from the 1m estimator) or C_1m_full (all bins from 1m).

    `variables` limits the rewrite (C_1m_last_h3: only the variables that pass ace_1m H3); the rest keep the
    container's forward-filled copy."""
    def replace(table: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
        table = table.copy()
        newest = t0                                                         # bin t0 = label t0-1h, unpublished
        for variable, column in ONE_MINUTE.items():
            if variables is not None and variable not in variables:
                continue
            arr = estimates[variable]
            if mode == 'full':
                bins = table.index[table.index < newest]
                o1 = arr['O1'].reindex(bins - pd.Timedelta(hours=1)).to_numpy()
                table.loc[bins, column] = o1
            est = arr['estimate'].get(newest - pd.Timedelta(hours=1), np.nan)
            if np.isfinite(est):
                table.loc[newest, column] = est
            # else: keep the container's forward-filled copy (what serving would do)
        return table
    return replace


def load_estimates(root: Path) -> tuple[dict, dict, dict]:
    """research/ace_1m_to_hourly_20261108 stage2 output: the estimator chosen on conf (as its downstream does),
    plus whether it passes that study's H3 on F1, F2 and F3 (preregistered there; uses the scored periods' inputs)."""
    report = json.loads((root / 'estimators.json').read_text())['variables']
    out, chosen, passes = {}, {}, {}
    for variable in ONE_MINUTE:
        r = report[variable]
        cands = {k: r['conf'][k].get('rmse', 1e18) for k in ('E1', 'E3', 'E4_ridge', 'E5_gbm_mean') if k in r['conf']}
        cands[r['best_simple']] = r['conf'][r['best_simple']].get('rmse', 1e18)
        chosen[variable] = min(cands, key=cands.get)
        passes[variable] = all(r[p][chosen[variable]].get('rmse_vs_E0', 0) <= -H3_THRESHOLD[variable] for p in PERIODS)
        with np.load(root / f'hourly_estimates_{variable}.npz') as z:
            labels = pd.DatetimeIndex(z['labels'], tz='UTC')
            out[variable] = dict(estimate=pd.Series(z[chosen[variable]].astype(float), index=labels),
                                 O1=pd.Series(z['O1_oracle'].astype(float), index=labels))
    return out, chosen, passes


def cmd_views(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    collect, af, hour, lookback = benchlab()
    began = time.time()
    report = dict(experiment=EXPERIMENT, stage='views', **provenance(), data=file_identity(DATA),
                  raw_dir=str(RAW), ace_features=ACE11, delay_min=DELAY_MIN)
    oracle = oracle_series(af)
    log(dict(stage='views', unit=1, units=5, step='oracle', rows=len(oracle)))
    # G1: rebuilt k=0 features == df for every train + query origin
    df = pd.read_parquet(DATA, columns=['set', 'origin'] + ACE11, filters=[('h', '==', 1)])
    df['origin'] = pd.to_datetime(df.origin, utc=True)
    first_train = df.loc[df.set == 'train', 'origin'].min()
    g1 = {}
    for name in ACE11:
        a = df[name].to_numpy(np.float64)
        b = oracle[name].reindex(df.origin).to_numpy(np.float64)
        bad = ~same(a, b)
        leading = bad & np.isnan(a) & (df.origin == first_train).to_numpy()
        g1[name] = dict(rows=len(a), mismatch=int(bad.sum()), leading_nan_exempt=int(leading.sum()))
    g1_pass = all(v['mismatch'] == v['leading_nan_exempt'] for v in g1.values())
    report['G1_raw_rebuild_equals_df'] = dict(per_feature=g1, passed=g1_pass)
    log(dict(stage='views', unit=2, units=5, step='G1', passed=g1_pass))
    if not g1_pass:
        write_json(out / 'views.json', report)
        return 1
    # container views on the scored origins
    origins, period, _ = scored_origins()
    texts = {}
    for product in af.SPEC:
        for path in RAW.glob(f'*_ace_{product}_1h.txt'):
            texts[(product, path.name[:6])] = path.read_text()
    views = {f'ffill{k}': np.empty((len(origins), len(ACE11)), np.float32) for k in DELAY_MIN}
    for i, t0 in enumerate(origins):
        for k in DELAY_MIN:
            views[f'ffill{k}'][i] = container_features(collect, af, hour, lookback, texts, t0, k).to_numpy(np.float32)
    views['oracle'] = oracle.reindex(origins).to_numpy(np.float32)
    for k in SHIFTS:
        views[f'shift{k}'] = oracle.reindex(origins - pd.Timedelta(hours=k)).to_numpy(np.float32)
    dfq = df[df.set == 'query'].drop_duplicates('origin').set_index('origin')
    g2 = {name: int((~same(views['ffill0'][:, j], dfq[name].reindex(origins).to_numpy())).sum()) for j, name in enumerate(ACE11)}
    g2_pass = not any(g2.values())
    report['G2_container_k0_equals_df'] = dict(mismatch_per_feature=g2, origins=len(origins), passed=g2_pass)
    report['view_change_vs_oracle'] = {
        v: {name: dict(changed=int((~same(views[v][:, j], views['oracle'][:, j])).sum()),
                       mean_abs=float(np.nanmean(np.abs(views[v][:, j] - views['oracle'][:, j]))))
            for j, name in enumerate(ACE11)} for v in views if v != 'oracle'}
    log(dict(stage='views', unit=3, units=5, step='container views + G2', passed=g2_pass,
             seconds=round(time.time() - began)))
    # 1-minute views when the ace_1m estimates exist
    est_root = Path(args.estimates)
    if (est_root / 'estimators.json').exists():
        estimates, chosen, passes = load_estimates(est_root)
        passing = [v for v in ONE_MINUTE if passes[v]]
        for label, mode, variables in (('1m_last', 'last', None), ('1m_last_h3', 'last', passing), ('1m_full', 'full', None)):
            replace = one_minute_replacer(estimates, mode, variables)
            arr = np.empty((len(origins), len(ACE11)), np.float32)
            for i, t0 in enumerate(origins):
                arr[i] = container_features(collect, af, hour, lookback, texts, t0, 1, replace).to_numpy(np.float32)
            views[label] = arr
            log(dict(stage='views', step=label, seconds=round(time.time() - began)))
        report['one_minute'] = dict(estimates=str(est_root), chosen_estimator=chosen, passes_h3=passes,
                                    h3_variables=passing)
    else:
        report['one_minute'] = dict(estimates=str(est_root), status='not available yet')
    np.savez_compressed(out / 'views.npz', origins=origins.asi8, period=period, names=np.array(ACE11), **views)
    log(dict(stage='views', unit=4, units=5, step='1m views', available='1m_last' in views))
    # shift inputs for training (G6)
    if not args.skip_shift_inputs:
        (HERE / 'input').mkdir(exist_ok=True)
        report['G6_shift_inputs'] = {k: write_shift(oracle, k, HERE / f'input/df_shift{k}.parquet') for k in SHIFTS}
    passed = g1_pass and g2_pass and all(v['passed'] for v in report.get('G6_shift_inputs', {}).values())
    report.update(passed=passed, seconds=round(time.time() - began))
    write_json(out / 'views.json', report)
    log(dict(stage='views', unit=5, units=5, step='G6', passed=passed))
    return 0 if passed else 1


# ------------------------------------------------------------------------------------------------ scoring
class Population:
    """Per-lead truth statistics of the official scorer for one set of origin rows (lead-timse-6h-v1)."""

    def __init__(self, truth: np.ndarray):
        self.leads = []
        for col in range(truth.shape[1]):
            valid = np.isfinite(truth[:, col])
            observed = truth[valid, col]
            q = np.quantile(observed, np.arange(101) / 100, method='linear')
            order = np.argsort(observed, kind='stable')
            positions = np.searchsorted(observed[order], q[:-1], side='left')
            self.leads.append((valid, observed, order, positions, np.diff(q), float(q[-1] - q[0])))

    def score(self, pred: np.ndarray) -> dict:
        s, rmse, sq_sum, abs_sum, n = np.empty(72), np.empty(72), 0., 0., 0
        for col, (valid, observed, order, positions, widths, span) in enumerate(self.leads):
            error = pred[valid, col] - observed
            squared = error ** 2
            suffix = np.cumsum(squared[order][::-1])[::-1]
            timse = float(np.dot(widths, suffix[positions] / (len(observed) - positions)) / span)
            rmse[col] = np.sqrt(squared.mean())
            s[col] = np.sqrt(rmse[col] * np.sqrt(timse))
            sq_sum, abs_sum, n = sq_sum + squared.sum(), abs_sum + np.abs(error).sum(), n + len(error)
        return dict(s=float(s.mean()), lead_s=s, lead_rmse=rmse, rmse=float(rmse.mean()),
                    pooled_rmse=float(np.sqrt(sq_sum / n)), mae=float(abs_sum / n))


def summary(result: dict) -> dict:
    out = {k: result[k] for k in ('s', 'rmse', 'pooled_rmse', 'mae')}
    for lo, hi in BUCKETS:
        out[f's_h{lo}-{hi}'] = float(result['lead_s'][lo - 1:hi].mean())
        out[f'rmse_h{lo}-{hi}'] = float(result['lead_rmse'][lo - 1:hi].mean())
    return out


def stored_model(seed: int, period: str):
    import xgboost as xgb
    run = BASE_RUNS / f'baseline_s{seed}' / f'{period}_models'
    boosters = []
    for b in range(3):
        booster = xgb.Booster()
        booster.load_model(str(run / f'full/b{b}.ubj'))
        boosters.append(booster)
    offset = np.asarray(json.loads((run / 'frozen_calibration.json').read_text())['offset'], float)
    return boosters, offset


def forecast(boosters, offset, C: np.ndarray, E: np.ndarray) -> np.ndarray:
    """runs/baseline_s*/experiment.py inference: 3 x 24 h blocks, clip, + horizon offset, clip."""
    parts = [np.asarray(b.inplace_predict(np.hstack([C, E[:, s:s + BLOCK].reshape(len(C), -1)]).astype(np.float32)), float)
             for b, s in zip(boosters, (0, 24, 48))]
    return np.clip(np.clip(np.hstack(parts), 1., 3000.) + offset[None, :], 1., 3000.)


def frame_of(origins: pd.DatetimeIndex, pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(dict(origin_last_input_utc=np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), 72),
                             horizon_hours=np.tile(np.arange(1, 73), len(origins)), pred_kms=pred.ravel()))


def cmd_score(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    competition = load_module('competition', SCORER)
    began = time.time()
    z = np.load(Path(args.views) / 'views.npz')
    origins, period = pd.DatetimeIndex(z['origins'], tz='UTC'), z['period']
    assert list(z['names']) == ACE11
    views = {k: z[k] for k in z.files if k not in ('origins', 'period', 'names')}
    context = competition.prepare_context(competition.load_observations(OBSERVATIONS))
    assert context['origins'].equals(origins)
    feat = names()
    cols = [feat['constant'].index(n) for n in ACE11]
    query = {}
    for p in PERIODS:
        grid, _ = baseline.load_grid(DATA, p, feat, False)
        query[p] = grid
    report = dict(experiment=EXPERIMENT, stage='score', **provenance(), data=file_identity(DATA),
                  observations=file_identity(OBSERVATIONS), scorer=file_identity(SCORER), scorer_version=competition.VERSION,
                  features=feat, replaced_features=ACE11, kept_features=[n for n in feat['constant'] if n not in ACE11] + feat['varying'],
                  views=sorted(views), gates={})
    view_of = {'A_oracle': 'oracle', 'A_ffill': 'ffill1', 'A_ffill2': 'ffill2', 'C_1m_last': '1m_last',
               'C_1m_last_h3': '1m_last_h3', 'C_1m_full': '1m_full'}
    preds: dict[tuple[str, int], np.ndarray] = {}
    g3 = {}
    units = len(SEEDS) * len(PERIODS)
    unit = 0
    for seed in SEEDS:
        for p in PERIODS:
            boosters, offset = stored_model(seed, p)
            grid = query[p]
            pos = pd.Series(np.arange(len(grid['origins'])), index=grid['origins'])
            rows = np.flatnonzero(period == p)
            take = pos.reindex(origins[rows]).to_numpy()
            assert np.isfinite(take).all()
            take = take.astype(int)
            # G3a: stored hourly predictions are reproduced from the stored boosters on the df inputs
            with np.load(BASE_RUNS / f'baseline_s{seed}' / f'{p}_predictions.npz') as stored:
                assert np.array_equal(stored['origins'], grid['origins'].asi8)
                full = forecast(boosters, offset, grid['C'], grid['E'])
                g3[f's{seed}_{p}'] = float(np.abs(full - stored['pred']).max())
            for condition, view in view_of.items():
                if view not in views:
                    continue
                C = grid['C'][take].copy()
                C[:, cols] = views[view][rows]
                preds.setdefault((condition, seed), np.full((len(origins), 72), np.nan))[rows] = forecast(
                    boosters, offset, C, grid['E'][take])
            # A_mix: k=1 or (late refresh) k=2 per origin, fixed seeds
            rng = np.random.default_rng(BOOT_SEED + seed)
            for d in range(MIX_DRAWS):
                late = rng.random(len(rows)) < LATE_SHARE
                C = grid['C'][take].copy()
                C[:, cols] = np.where(late[:, None], views['ffill2'][rows], views['ffill1'][rows])
                preds.setdefault((f'A_mix#{d}', seed), np.full((len(origins), 72), np.nan))[rows] = forecast(
                    boosters, offset, C, grid['E'][take])
            unit += 1
            log(dict(stage='score', period=p, seed=seed, unit=unit, units=units))
    report['gates']['G3_stored_prediction_max_abs'] = g3
    # trained shift models (B): their own query predictions
    for k in SHIFTS:
        for seed in SEEDS:
            run = HERE / 'runs' / f'B_shift{k}_s{seed}'
            files = [run / f'{p}_predictions.npz' for p in PERIODS]
            if not all(f.exists() for f in files) or not (run / 'train.exit').exists() \
                    or (run / 'train.exit').read_text().strip() != '0':
                continue
            pred = np.full((len(origins), 72), np.nan)
            for p, f in zip(PERIODS, files):
                with np.load(f) as zz:
                    pos = pd.Series(np.arange(len(zz['origins'])), index=pd.DatetimeIndex(zz['origins'], tz='UTC'))
                    rows = np.flatnonzero(period == p)
                    pred[rows] = zz['pred'][pos.reindex(origins[rows]).to_numpy().astype(int)]
            preds[(f'B_shift{k}', seed)] = pred
    # official scorer on every prediction; G3b recorded S
    official = {}
    for (condition, seed), pred in preds.items():
        assert np.isfinite(pred).all()
        result = competition.score_competition(frame_of(origins, pred), context)
        assert result['total']['n_replaced'] == 0
        official[(condition, seed)] = result
    g3b = {seed: abs(official[('A_oracle', seed)]['total']['s'] - RECORDED_S[seed]) for seed in SEEDS}
    report['gates']['G3_recorded_S_abs_diff'] = g3b
    g3_pass = max(g3.values()) <= 1e-6 and max(g3b.values()) <= 1e-6
    # scopes for my Population (checked against the official numbers)
    truth = context['truth']
    scopes = {'total': np.ones(len(origins), bool), **{p.lower(): period == p for p in PERIODS}}
    pops = {name: Population(truth[mask]) for name, mask in scopes.items()}
    results, parity = {}, 0.
    for (condition, seed), pred in preds.items():
        entry = {}
        for name, mask in scopes.items():
            mine = pops[name].score(pred[mask])
            parity = max(parity, abs(mine['s'] - official[(condition, seed)][name]['s']))
            entry[name] = summary(mine)
        results.setdefault(condition, {})[seed] = entry
    report['gates']['scorer_parity_max_abs_S'] = parity
    # G4: ffill_cost reproduction (F3, seed 42, pooled RMSE over verified pairs, as that script prints)
    f3 = period == 'F3'
    ok = np.isfinite(truth[f3])

    def lead_mean_rmse(pred):
        err = np.where(ok, pred[f3] - truth[f3], np.nan)
        return float(np.sqrt(np.nanmean(err ** 2, 0)).mean())
    g4 = dict(standard=lead_mean_rmse(preds[('A_oracle', 42)]), container=lead_mean_rmse(preds[('A_ffill', 42)]),
              recorded=FFILL_COST_F3_S42)
    g4['passed'] = abs(g4['standard'] - 55.67) < 0.006 and abs(g4['container'] - 55.90) < 0.006
    report['gates']['G4_ffill_cost'] = g4
    report['gates']['passed'] = g3_pass and parity <= 1e-9 and g4['passed']
    log(dict(stage='score', step='gates', **{k: v for k, v in report['gates'].items() if k != 'G3_stored_prediction_max_abs'}))
    # A_mix: mean over draws
    mix = [c for c in results if c.startswith('A_mix#')]
    if mix:
        results['A_mix'] = {}
        for seed in SEEDS:
            draws = [results[c][seed] for c in mix]
            results['A_mix'][seed] = {scope: {m: float(np.mean([d[scope][m] for d in draws])) for m in draws[0][scope]}
                                      for scope in scopes}
            results['A_mix'][seed]['total']['s_draw_range'] = [float(min(d['total']['s'] for d in draws)),
                                                               float(max(d['total']['s'] for d in draws))]
        for c in mix:
            del results[c]
            for seed in SEEDS:
                preds.pop((c, seed), None)
    for condition, by_seed in results.items():
        seeds = sorted(by_seed)
        by_seed_out = {str(s): by_seed[s] for s in seeds}
        mean = {scope: {m: float(np.mean([by_seed[s][scope][m] for s in seeds])) for m in by_seed[seeds[0]][scope]
                        if not isinstance(by_seed[seeds[0]][scope][m], list)} for scope in scopes}
        results[condition] = dict(seeds=seeds, by_seed=by_seed_out, mean=mean)
    report['results'] = results
    # paired comparisons with a stratified UTC-day block bootstrap of the seed-mean delta S
    comparisons = [('A_ffill', 'A_oracle'), ('A_ffill2', 'A_oracle'), ('A_ffill2', 'A_ffill'),
                   ('B_shift1', 'A_ffill'), ('B_shift1', 'A_oracle'), ('B_shift2', 'A_ffill2'), ('B_shift2', 'A_oracle'),
                   ('C_1m_last', 'A_ffill'), ('C_1m_last', 'A_oracle'), ('C_1m_last_h3', 'A_ffill'), ('C_1m_last_h3', 'A_oracle'),
                   ('C_1m_full', 'A_oracle'), ('C_1m_full', 'A_ffill'), ('B_shift1', 'C_1m_last')]
    days = np.array([f'{o:%Y-%m-%d}' for o in origins])
    blocks = {p: {d: np.flatnonzero((period == p) & (days == d)) for d in np.unique(days[period == p])} for p in PERIODS}
    rng = np.random.default_rng(BOOT_SEED)
    pairs = []
    for a, b in comparisons:
        seeds = sorted(set(s for c, s in preds if c == a) & set(s for c, s in preds if c == b))
        if seeds:
            pairs.append((a, b, seeds))
    draws = {(a, b): {scope: [] for scope in scopes} for a, b, _ in pairs}
    needed = sorted({(c, s) for a, b, seeds in pairs for c in (a, b) for s in seeds})
    for d in range(args.bootstrap):
        pick = {p: np.concatenate([blocks[p][x] for x in rng.choice(list(blocks[p]), len(blocks[p]))]) for p in PERIODS}
        pick['total'] = np.concatenate([pick[p] for p in PERIODS])
        for scope in scopes:
            rows = pick['total'] if scope == 'total' else pick[scope.upper()]
            pop = Population(truth[rows])
            s = {key: pop.score(preds[key][rows])['s'] for key in needed}
            for a, b, seeds in pairs:
                draws[(a, b)][scope].append(float(np.mean([s[(a, x)] - s[(b, x)] for x in seeds])))
        if (d + 1) % 200 == 0:
            log(dict(stage='bootstrap', unit=d + 1, units=args.bootstrap, seconds=round(time.time() - began)))
    paired = {}
    for a, b, seeds in pairs:
        entry = dict(seeds=seeds)
        for scope in scopes:
            per_seed = {str(x): results[a]['by_seed'][str(x)][scope]['s'] - results[b]['by_seed'][str(x)][scope]['s']
                        for x in seeds}
            mean_delta = float(np.mean(list(per_seed.values())))
            ref = float(np.mean([results[b]['by_seed'][str(x)][scope]['s'] for x in seeds]))
            buckets = {f'h{lo}-{hi}': float(np.mean([results[a]['by_seed'][str(x)][scope][f's_h{lo}-{hi}'] -
                                                     results[b]['by_seed'][str(x)][scope][f's_h{lo}-{hi}'] for x in seeds]))
                       for lo, hi in BUCKETS}
            entry[scope] = dict(delta_s_by_seed=per_seed, delta_s_mean=mean_delta, relative=mean_delta / ref,
                                ci95=[float(np.quantile(draws[(a, b)][scope], .025)),
                                      float(np.quantile(draws[(a, b)][scope], .975))],
                                delta_s_by_bucket=buckets)
        paired[f'{a} - {b}'] = entry
    report['paired'] = paired
    report['bootstrap'] = dict(draws=args.bootstrap, seed=BOOT_SEED, block='UTC day, stratified by period')
    report['decisions'] = decide(paired, results)
    report['seconds'] = round(time.time() - began)
    write_json(out / 'results.json', report)
    np.savez_compressed(out / 'predictions.npz', origins=origins.asi8,
                        **{f'{c}__s{s}': p.astype(np.float32) for (c, s), p in preds.items()})
    log(dict(stage='score', step='done', gates=report['gates']['passed'], seconds=report['seconds']))
    return 0 if report['gates']['passed'] else 1


def decide(paired: dict, results: dict) -> dict:
    """The preregistered rules of strategy.md."""
    out = {}
    q2 = paired.get('A_ffill - A_oracle')
    if q2:
        rel = q2['total']['relative']
        out['Q2'] = dict(relative=rel, delta_s=q2['total']['delta_s_mean'], ci95=q2['total']['ci95'],
                         verdict='negligible' if rel < .005 else ('moderate' if rel < .01 else 'large'))
    q1 = paired.get('B_shift1 - A_ffill')
    if q1:
        t = q1['total']
        complete = len(q1['seeds']) == 3
        adopt = complete and all(v < 0 for v in t['delta_s_by_seed'].values()) and t['delta_s_mean'] <= -.10 and t['ci95'][1] < 0
        out['Q1'] = dict(seeds=q1['seeds'], delta_s=t['delta_s_mean'], by_seed=t['delta_s_by_seed'], ci95=t['ci95'],
                         complete=complete, adopt_delay_aware_training=adopt)
    q3 = paired.get('C_1m_last - A_ffill')
    if q3:
        t = q3['total']
        out['Q3_last'] = dict(delta_s=t['delta_s_mean'], by_seed=t['delta_s_by_seed'], ci95=t['ci95'],
                              s_condition=all(v <= 0 for v in t['delta_s_by_seed'].values()) and len(q3['seeds']) == 3,
                              note='also needs ace_1m H3/H4 and the H0 gate (see strategy.md)')
    q3h = paired.get('C_1m_last_h3 - A_ffill')
    if q3h:
        t = q3h['total']
        out['Q3_last_h3_exploratory'] = dict(delta_s=t['delta_s_mean'], by_seed=t['delta_s_by_seed'], ci95=t['ci95'],
                                             note='variables kept by ace_1m H3, which looks at F1-F3 inputs')
    q3f = paired.get('C_1m_full - A_oracle')
    if q3f:
        out['Q3_full'] = dict(relative=q3f['total']['relative'], delta_s=q3f['total']['delta_s_mean'],
                              ci95=q3f['total']['ci95'], small=abs(q3f['total']['relative']) < .005)
    return out


# ------------------------------------------------------------------------------------------------ training (B)
def sample_weights(target: np.ndarray, observed: np.ndarray) -> np.ndarray:
    raw = np.where(target >= 550., TAIL_WEIGHT, 1.) * observed
    return raw / (raw.sum(0) / observed.sum(0))


def make_objective(target: np.ndarray):
    if not np.isfinite(target).all():
        raise ValueError('Training targets must be finite')
    observed = np.isfinite(target)
    weights = sample_weights(target, observed)

    def objective(prediction, matrix):
        prediction = np.asarray(prediction).reshape(target.shape)
        error = np.where(observed, prediction - np.nan_to_num(target), 0.)
        z = np.fft.fft(error, axis=1, norm='ortho')
        rho = 1 / np.sqrt(1 + abs(z) ** 2 / SPECTRAL_SCALE ** 2)
        gradient = (error * weights + SPECTRAL_COEFFICIENT * np.fft.ifft(z * rho, axis=1, norm='ortho').real) * observed
        hessian = (weights + SPECTRAL_COEFFICIENT) * observed
        return gradient.astype(np.float32), hessian.astype(np.float32)
    return objective


def design(grid: dict, start: int) -> np.ndarray:
    return np.hstack([grid['C'], grid['E'][:, start:start + BLOCK].reshape(len(grid['C']), BLOCK * grid['E'].shape[2])]
                     ).astype(np.float32)


def fit_predict(train, query, target, seed, threads, out: Path, period, stage, progress, smoke=False):
    import xgboost as xgb
    out.mkdir(parents=True, exist_ok=True)
    predictions = np.empty((len(query['origins']), baseline.H), np.float64)
    for block, start in enumerate(range(0, baseline.H, BLOCK)):
        began = time.perf_counter()
        y = target[:, start:start + BLOCK]
        matrix = xgb.DMatrix(design(train, start), label=y.astype(np.float32), nthread=threads)
        params = dict(baseline.PARAMS, seed=seed, device='cpu', nthread=threads, num_target=BLOCK, base_score=float(y.mean()))
        model = xgb.train(params, matrix, num_boost_round=2 if smoke else ROUNDS, obj=make_objective(y))
        model.save_model(out / f'b{block}.ubj')
        predictions[:, start:start + BLOCK] = model.inplace_predict(design(query, start))
        progress['unit'] += 1
        log(dict(period=period, stage=stage, component='spectral', block=block, horizon=start + BLOCK,
                 fit_seconds=round(time.perf_counter() - began, 3), **progress))
        del model, matrix
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite component forecasts')
    return predictions


def run_period(data: Path, out: Path, period: str, seed: int, shift: int, threads: int, smoke: bool, identity: dict) -> dict:
    began = time.perf_counter()
    model_dir = out / f'{period}_models'
    model_dir.mkdir(parents=True, exist_ok=True)
    feat = names()
    train, y = baseline.load_grid(data, period, feat, True)
    query, _ = baseline.load_grid(data, period, feat, False)
    if len(query['origins']) != baseline.QUERY_COUNTS[period]:
        raise ValueError('Unexpected official query grid')
    if not np.isfinite(y).all():
        raise ValueError('Prepared training targets must be finite')
    reference, calibration, boundary = baseline.prior_split(train['origins'])
    full = np.arange(len(y))
    if smoke:
        reference = reference[np.linspace(0, len(reference) - 1, min(len(reference), 128)).astype(int)]
        calibration = calibration[np.linspace(0, len(calibration) - 1, min(len(calibration), 64)).astype(int)]
        full = np.linspace(0, len(y) - 1, min(len(y), 192)).astype(int)
        query = baseline.subset(query, np.arange(min(len(query['origins']), 8)))
    progress = dict(unit=0, units=2 * (baseline.H // BLOCK))
    prior = fit_predict(baseline.subset(train, reference), baseline.subset(train, calibration), y[reference],
                        seed, threads, model_dir / 'prior', period, 'prior', progress, smoke)
    truth, cal_origins = y[calibration], train['origins'][calibration].asi8
    raw = fit_predict(baseline.subset(train, full), query, y[full], seed, threads, model_dir / 'full', period, 'full',
                      progress, smoke)
    rows = min(OFFSET_WINDOW, len(truth))
    if rows < OFFSET_WINDOW and not smoke:
        raise ValueError('Insufficient training calibration rows for frozen window')
    offset = ALPHA * (truth[-rows:] - np.clip(prior, 1., 3000.)[-rows:]).mean(0)
    state = dict(calibration_rows=len(calibration), reference_rows=len(reference),
                 reference_start=str(train['origins'][reference].min()), reference_end=str(train['origins'][reference].max()),
                 calibration_start=str(train['origins'][calibration].min()),
                 calibration_end=str(train['origins'][calibration].max()), split_boundary=str(boundary), purge_hours=144,
                 policy=POLICY, window=OFFSET_WINDOW, alpha=ALPHA, scope='horizon', applied=True, offset=offset.tolist(),
                 offset_estimator='mean', offset_condition='none', offset_rows=rows,
                 offset_start=str(pd.Timestamp(cal_origins[-rows], tz='UTC')), offset_end=str(pd.Timestamp(cal_origins[-1], tz='UTC')))
    write_json(model_dir / 'frozen_calibration.json', state)
    np.savez_compressed(model_dir / 'prior_predictions.npz', origins=cal_origins, raw=prior,
                        spectral=np.clip(prior, 1., 3000.), training_truth=truth)
    component = np.clip(raw, 1., 3000.)
    prediction = np.clip(component + offset[None, :], 1., 3000.)
    origins = query['origins'].asi8
    np.savez_compressed(out / f'{period}_predictions.npz', pred=prediction, origins=origins)
    np.savez_compressed(out / f'{period}_no_offset_predictions.npz', pred=component, origins=origins)
    np.savez_compressed(out / f'{period}_raw_predictions.npz', raw=raw, component=component,
                        preclip=component + offset[None, :], offset=np.broadcast_to(offset[None, :], component.shape).copy(),
                        origins=origins)
    report = dict(experiment=out.name, period=period, smoke=smoke, shift_bins=shift, seed=seed, policy=POLICY,
                  features=feat, feature_count=sum(map(len, feat.values())), shifted_features=ACE11,
                  kept_features=[n for n in feat['constant'] if n not in ACE11] + feat['varying'],
                  parameters=dict(baseline.PARAMS, seed=seed, nthread=threads), rounds=2 if smoke else ROUNDS,
                  tail_weight=TAIL_WEIGHT, spectral=dict(coefficient=SPECTRAL_COEFFICIENT, scale=SPECTRAL_SCALE),
                  block_size=BLOCK, training_cutoff=baseline.CUTOFFS[period], maturity_hours=144,
                  train_origins=len(full), query_origins=len(prediction),
                  train_start=str(train['origins'][full].min()), train_end=str(train['origins'][full].max()),
                  query_start=str(pd.Timestamp(origins.min(), tz='UTC')), query_end=str(pd.Timestamp(origins.max(), tz='UTC')),
                  frozen_state=state, end_to_end_seconds=time.perf_counter() - began, identity=identity)
    write_json(out / f'{period}_metadata.json', report)
    return report


def cmd_train(args) -> int:
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU GBM runs require CUDA_VISIBLE_DEVICES="" before startup')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = DATA if args.shift == 0 else HERE / f'input/df_shift{args.shift}.parquet'
    if args.shift and not args.smoke:
        views = json.loads((HERE / 'runs/views/views.json').read_text())
        if not views.get('passed') or not views['G6_shift_inputs'][str(args.shift)]['passed']:
            raise ValueError('shift input did not pass G6')
    identity = dict(data=file_identity(data), experiment=file_identity(Path(__file__)),
                    baseline=file_identity(Path(baseline.__file__)), **provenance())
    for period in ([args.period] if args.period else PERIODS):
        run_period(data, out, period, args.seed, args.shift, args.threads, args.smoke, identity)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', default=EXPERIMENT)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', choices=('cpu',), default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('views')
    p.add_argument('--estimates', default=str(ACE_1M_ESTIMATES))
    p.add_argument('--skip-shift-inputs', action='store_true')
    p = sub.add_parser('score')
    p.add_argument('--views', default=str(HERE / 'runs/views'))
    p.add_argument('--bootstrap', type=int, default=BOOT_DRAWS)
    p = sub.add_parser('train')
    p.add_argument('--shift', type=int, choices=(0, *SHIFTS), required=True, help='0 only for the G5 smoke check')
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--period', choices=PERIODS)
    p.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    return {'views': cmd_views, 'score': cmd_score, 'train': cmd_train}[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
