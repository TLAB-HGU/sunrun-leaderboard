"""auto_arima: non-seasonal AutoARIMA refitted on the last 648 h of every origin.

usage: experiment.py --out DIR [--smoke] [--workers N]
Exact with statsforecast==2.0.3 and numba==0.67.0 (leaderboard/requirements-baselines.txt); the shared environment's 2.1.1 gives 9045.53.
Data is only read from data-v2. Scores mse 9046.96 on the leaderboard.
"""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import argparse
import concurrent.futures
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast.models import AutoARIMA

DATA = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store/e2000fs-lag0')
WINDOW = 648  # training hours ending at the origin
ARIMA = dict(seasonal=False, season_length=1, max_p=5, max_q=5, max_d=2, ic='aicc', stepwise=True, nmodels=94,
             approximation=False, test='kpss', allowmean=True, allowdrift=True)


def targets(speed, origins):
    times = origins.values[:, None] + np.arange(1, 73).astype('timedelta64[h]')[None, :]
    return speed.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, 72)


def fit_predict(history):
    return np.asarray(AutoARIMA(**ARIMA).fit(history).predict(h=72)['mean'])


def main(out, smoke, workers):
    # Load
    speed = pd.read_parquet(DATA / 'speed.parquet')['speed']
    origins = pd.DatetimeIndex(np.concatenate([np.load(DATA / f'{grid}.npz')['origins'] for grid in ('f1', 'official')]), tz='UTC')
    f1 = len(np.load(DATA / 'f1.npz')['origins'])
    positions = speed.index.searchsorted(origins)
    if smoke:
        positions, origins, f1 = positions[::500], origins[::500], len(origins[:f1:500])

    # Refit and forecast once per origin
    values = speed.to_numpy(np.float64)
    histories = [values[p - WINDOW + 1:p + 1].copy() for p in positions]
    fit_predict(histories[0])  # warm JIT before worker processes are forked
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        pred = np.stack(list(pool.map(fit_predict, histories, chunksize=8)))

    # Score: the hourly speed series is the official truth series
    assert (smoke or pred.shape == (4970, 72)) and np.isfinite(pred).all()
    pd.DataFrame({'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), 72),
                  'horizon_hours': np.tile(np.arange(1, 73), len(origins)),
                  'pred_kms': pred.ravel()}).to_parquet(out / 'predictions.parquet', index=False)
    errors = (pred - targets(speed, origins)) ** 2
    scores = dict(mse=float(errors.mean()), mse_f1=float(errors[:f1].mean()), mse_f2=float(errors[f1:].mean()))
    (out / 'scores.json').write_text(json.dumps(scores, indent=2) + '\n')
    print(json.dumps(scores), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    main(args.out, args.smoke, args.workers)
