"""fianl_v1 HPO on two internal 2025 folds (training rows only, no 2026 data, no board truth).

Pre-registered (2026-11-26, before any trial ran):
  folds    V1 = targets 2025-07-01..2025-10-01, V2 = targets 2025-10-01..2025-12-26 (UTC, end exclusive).
           Training rows of a fold: fit_F1 rows with origin + 144 h <= fold start - 72 h (the board's rule:
           cutoff = period start - 72 h, 144 h purge). Scored origins: 6-hourly, whole 72 h window inside the fold.
  score    official lead-first S (scorer lead-timse-6h-v1 formula) on unfilled observed speeds, per fold.
  model    fianl_v1 unchanged except PARAMS / ROUNDS of the trial; seed 42 only; seam xf8 -> 3 h phase advance;
           the prior-model offset is left out of the search (it needs a second fit per fold).
  trials   20 = trial 0 (fianl_v1 defaults) + 19 random draws, rng seed 20261126, drawn before any fit.
  select   lowest mean of V1 S and V2 S. Reported with the per-fold change against trial 0.

Amendment 2026-11-26 (fianl_v2 shallow search, user request "depth 1-2", before any v2 trial ran):
  model    fianl_v2 (team research/fianl_v2, local patches) on the 2026-11-26 post-fetch df (live rows dropped,
           query y nulled). Folds, rows, score, seed and selection rule as above.
  trials   20 = trial 0 (fianl_v2 defaults, depth 3) + trial 1 (fianl_v1 HPO winner t16, depth 2) + 18 random draws,
           rng seed 20261127: depth {1, 2}; rounds {300, 450, 600, 900, 1200} (1200 added because depth-1 trees
           need more of them); other ranges unchanged. Mode make-trials-shallow.

usage: python hpo.py make-trials OUT
       CUDA_VISIBLE_DEVICES='' python hpo.py work OUT --data df.parquet --experiment experiment.py --threads 8
       python hpo.py report OUT
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

FOLDS = {'V1': ('2025-07-01', '2025-10-01'), 'V2': ('2025-10-01', '2025-12-26')}
H = 72
BASE = dict(max_depth=3, learning_rate=0.03, rounds=600, min_child_weight=20, reg_lambda=5,
            subsample=0.8, colsample_bytree=0.7)


def make_trials(out):
    rng = np.random.default_rng(20261126)
    trials = [dict(BASE)]
    for _ in range(19):
        trials.append(dict(
            max_depth=int(rng.choice([2, 3, 4, 5])),
            learning_rate=float(np.exp(rng.uniform(np.log(0.015), np.log(0.06)))),
            rounds=int(rng.choice([300, 450, 600, 900])),
            min_child_weight=float(np.exp(rng.uniform(np.log(5), np.log(80)))),
            reg_lambda=float(np.exp(rng.uniform(np.log(1), np.log(20)))),
            subsample=float(rng.uniform(0.6, 1.0)),
            colsample_bytree=float(rng.uniform(0.5, 0.9))))
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'trials.json'
    if path.exists():
        raise SystemExit('trials.json exists; trials are drawn once')
    path.write_text(json.dumps(trials, indent=1))
    print(json.dumps(trials, indent=1))


V1_T16 = dict(max_depth=2, learning_rate=0.038375442815769975, rounds=450, min_child_weight=9.248543255203462,
              reg_lambda=5.231301586893702, subsample=0.7110334592419305, colsample_bytree=0.6862170728197281)


def make_trials_shallow(out):
    rng = np.random.default_rng(20261127)
    trials = [dict(BASE), dict(V1_T16)]
    for _ in range(18):
        trials.append(dict(
            max_depth=int(rng.choice([1, 2])),
            learning_rate=float(np.exp(rng.uniform(np.log(0.015), np.log(0.06)))),
            rounds=int(rng.choice([300, 450, 600, 900, 1200])),
            min_child_weight=float(np.exp(rng.uniform(np.log(5), np.log(80)))),
            reg_lambda=float(np.exp(rng.uniform(np.log(1), np.log(20)))),
            subsample=float(rng.uniform(0.6, 1.0)),
            colsample_bytree=float(rng.uniform(0.5, 0.9))))
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'trials.json'
    if path.exists():
        raise SystemExit('trials.json exists; trials are drawn once')
    path.write_text(json.dumps(trials, indent=1))
    for i, t in enumerate(trials):
        print(i, {k: round(v, 4) if isinstance(v, float) else v for k, v in t.items()})


def lead_scores(pred, truth):
    """Official per-lead S on one population: pred, truth (n, 72); NaN truth excluded per lead."""
    rows = []
    for c in range(H):
        valid = np.isfinite(truth[:, c])
        observed = truth[valid, c]
        error = pred[valid, c] - observed
        squared = error ** 2
        quantiles = np.quantile(observed, np.arange(101) / 100, method='linear')
        order = np.argsort(observed, kind='stable')
        positions = np.searchsorted(observed[order], quantiles[:-1], side='left')
        suffix = np.cumsum(squared[order][::-1])[::-1]
        tail = suffix[positions] / (len(observed) - positions)
        timse = float(np.dot(np.diff(quantiles), tail) / (quantiles[-1] - quantiles[0]))
        rmse = float(np.sqrt(squared.mean()))
        rows.append(dict(s=float(np.sqrt(rmse * np.sqrt(timse))), timse=timse, rmse=rmse,
                         bias=float(error.mean()), n=int(valid.sum())))
    counts = [r['n'] for r in rows]
    return dict(s=float(np.mean([r['s'] for r in rows])), timse=float(np.mean([r['timse'] for r in rows])),
                rmse=float(np.mean([r['rmse'] for r in rows])),
                bias=float(np.average([r['bias'] for r in rows], weights=counts)),
                s_blocks=[float(np.mean([r['s'] for r in rows[a:a + 24]])) for a in (0, 24, 48)])


def fold_origins(start, end):
    first = pd.Timestamp(start, tz='UTC') - pd.Timedelta(hours=1)
    last = pd.Timestamp(end, tz='UTC') - pd.Timedelta(hours=73)
    return pd.date_range(first.ceil('6h'), last.floor('6h'), freq='6h')


def load_experiment(path):
    spec = importlib.util.spec_from_file_location('fianl_v1_experiment', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_truth(data):
    frame = pd.read_parquet(data, columns=['set', 'target_time', 'y'], filters=[('set', '==', 'train')])
    frame = frame.dropna(subset=['y']).drop_duplicates('target_time')
    series = pd.Series(frame.y.to_numpy(np.float64), index=pd.DatetimeIndex(pd.to_datetime(frame.target_time, utc=True)))
    return series.sort_index()


def work(args):
    out = args.out
    trials = json.loads((out / 'trials.json').read_text())
    E = load_experiment(args.experiment)
    logs = E.Logs(out / 'logs', f'w{args.worker}')
    table = E.load_table(args.data, logs)
    truth_series = raw_truth(args.data)
    origin = pd.DatetimeIndex(pd.to_datetime(table['origin'], utc=True))
    folds = {}
    for name, (start, end) in FOLDS.items():
        cutoff = pd.Timestamp(start, tz='UTC') - pd.Timedelta(hours=72)
        table[f'fit_{name}'] = (table['fit_F1'].to_numpy()
                                & np.asarray(origin + pd.Timedelta(hours=E.PURGE_HOURS) <= cutoff))
        query = fold_origins(start, end)
        table[f'fit_{name}q'] = (table['set'].eq('train').to_numpy() & np.asarray(origin.isin(query)))
        E.CUTOFFS[name] = cutoff.strftime('%Y-%m-%d %H:%M:%S')
        E.CUTOFFS[f'{name}q'] = '2100-01-01'
        train, y = E.load_grid(table, name, True)
        grid, _ = E.load_grid(table, f'{name}q', True)
        if not grid['origins'].equals(query):
            raise ValueError(f'{name}: query grid incomplete')
        if train['origins'].max() + pd.Timedelta(hours=E.PURGE_HOURS + 72) > pd.Timestamp(start, tz='UTC'):
            raise ValueError(f'{name}: training rows reach into the fold')
        truth = np.column_stack([truth_series.reindex(query + pd.Timedelta(hours=h)).to_numpy() for h in range(1, H + 1)])
        folds[name] = (train, y, grid, truth)
        logs('hpo', 'fold_ready', worker=args.worker, fold=name, train_origins=len(train['origins']),
             train_end=str(train['origins'].max()), query_origins=len(query),
             truth_coverage=float(np.isfinite(truth).mean()))
    run_args = SimpleNamespace(threads=args.threads, smoke=args.smoke)
    for index, trial in enumerate(trials):
        claim = out / 'claims' / f'{index:02d}'
        claim.parent.mkdir(exist_ok=True)
        try:
            os.close(os.open(claim, os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        E.PARAMS.update(max_depth=trial['max_depth'], learning_rate=trial['learning_rate'],
                        min_child_weight=trial['min_child_weight'], reg_lambda=trial['reg_lambda'],
                        subsample=trial['subsample'], colsample_bytree=trial['colsample_bytree'])
        rounds = 2 if args.smoke else trial['rounds']
        result = dict(trial=index, params=trial, worker=args.worker, threads=args.threads, folds={})
        began = time.perf_counter()
        for name, (train, y, grid, truth) in folds.items():
            tick = time.perf_counter()
            raw, _, _ = E.fit_predict(train, grid, y, 42, run_args, rounds, logs, 'hpo', name, f'trial{index}',
                                      dict(unit=0, units=3))
            final = E.phase_advance(E.seam_smooth(E.component_values(raw)))
            result['folds'][name] = dict(**lead_scores(final, truth), seconds=round(time.perf_counter() - tick, 1))
            np.save(out / 'preds' / f't{index:02d}_{name}.npy', final) if (out / 'preds').is_dir() else None
        result['mean_s'] = float(np.mean([f['s'] for f in result['folds'].values()]))
        result['seconds'] = round(time.perf_counter() - began, 1)
        (out / 'results').mkdir(exist_ok=True)
        (out / 'results' / f't{index:02d}.json').write_text(json.dumps(result, indent=1))
        logs('hpo', 'trial_done', **{k: v for k, v in result.items() if k != 'params'})


def report(out):
    rows = []
    for path in sorted((out / 'results').glob('t*.json')):
        r = json.loads(path.read_text())
        rows.append(dict(trial=r['trial'], mean_s=r['mean_s'], V1=r['folds']['V1']['s'], V2=r['folds']['V2']['s'],
                         V1_bias=r['folds']['V1']['bias'], V2_bias=r['folds']['V2']['bias'],
                         minutes=r['seconds'] / 60, **r['params']))
    frame = pd.DataFrame(rows).sort_values('mean_s')
    if (frame.trial == 0).any():
        base = frame[frame.trial == 0].iloc[0]
        for c in ('mean_s', 'V1', 'V2'):
            frame[f'd{c}_%'] = 100 * (frame[c] / base[c] - 1)
    pd.set_option('display.width', 250)
    print(frame.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['make-trials', 'make-trials-shallow', 'work', 'report'])
    ap.add_argument('out', type=Path)
    ap.add_argument('--data', type=Path)
    ap.add_argument('--experiment', type=Path)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.mode == 'make-trials':
        make_trials(args.out)
    elif args.mode == 'make-trials-shallow':
        make_trials_shallow(args.out)
    elif args.mode == 'work':
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise SystemExit('CPU only: CUDA_VISIBLE_DEVICES=""')
        work(args)
    else:
        report(args.out)


if __name__ == '__main__':
    main()
