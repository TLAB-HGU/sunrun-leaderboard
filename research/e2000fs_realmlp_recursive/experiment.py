"""e2000fs_realmlp_recursive: RealMLP with the Recursive strategy, end to end.

RealMLP-TD on the newest 5000 cells; each model trains up to 256 epochs and the selection window fixes its stop epoch.
One scalar model applied 72 times on its own forecasts.
All 111 features of the E2000 recipe. Query trajectories start empty and consume the model's own forecasts only.
F1 forecasts use targets before 2026-02-26, F2 forecasts targets before 2026-05-29.

usage: experiment.py --out DIR [--smoke]
Data is only read from data-v2. Results depend on the GPU stack; the archive is results/forecast.npz (mse 19308.84).
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')  # GPU0 is reserved
os.environ.setdefault('CC', '/usr/bin/gcc-12')  # this host ships no unversioned cc

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pytabkit import RealMLP_TD_Regressor
import torch

DATA = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store/e2000fs-lag0')
LAG = 24  # recent speed hours handed to every model
CONTEXT_ROWS = 5000
BATCH = 256
SEED = 42
SHARED = True  # one model for all horizons
EPOCHS = 256
STAGES = (('f1', '2026-02-26'), ('official', '2026-05-29'))  # forecast grid, training target cutoff


def targets(speed, origins):
    times = origins.values[:, None] + np.arange(1, 73).astype('timedelta64[h]')[None, :]
    return speed.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, 72)


def take(grid, index):
    return {k: grid[k][index] for k in ('C', 'E', 'past')}


def features(grid, previous, rows, horizons):
    """Origin constants, the 24 h ending at the previous horizon, this horizon's features and the lead."""
    trajectory = np.hstack([grid['past'], previous])
    lag = trajectory[rows[:, None], horizons[:, None] + np.arange(LAG)]
    return np.hstack((grid['C'][rows], lag, grid['E'][rows, horizons], (horizons + 1)[:, None])).astype(np.float32)


def context_cells(origins, head, limit):
    """The newest origin-horizon cells, oldest first."""
    order = np.argsort(pd.DatetimeIndex(origins).asi8, kind='stable')
    order = order[-((limit + 72 - 1) // 72):]
    return np.repeat(order, 72)[-limit:], np.tile(np.arange(72), len(order))[-limit:]


def new_model(smoke, stop_epoch=None):
    """Without ``stop_epoch`` the model trains the full schedule; the refit stops at the selected epoch."""
    params = dict(random_state=SEED, device='cuda', n_threads=4, n_cv=1, n_refit=0, n_repeats=1, n_ens=1, val_fraction=0.0,
                  n_epochs=2 if smoke else EPOCHS, verbosity=0, train_metric_name='mse', val_metric_name='rmse')
    if stop_epoch is not None:
        params['stop_epoch'] = stop_epoch
    return RealMLP_TD_Regressor(**params)


def select_epoch(fitting, Y, origins, valid, Yv, head, smoke):
    """Fit on the selection window and read the epoch with the best validation RMSE."""
    rows, horizons = context_cells(origins, head, 32 if smoke else CONTEXT_ROWS)
    vr = np.arange(len(Yv))
    if SHARED:
        vr, vh = np.repeat(vr, 72), np.tile(np.arange(72), len(vr))
    else:
        vh = np.full(len(vr), head, dtype=int)
    model = new_model(smoke)
    model.fit(features(fitting, Y, rows, horizons), Y[rows, horizons], X_val=features(valid, Yv, vr, vh), y_val=Yv[vr, vh])
    stop = model.fit_params_['stop_epoch']
    return int(stop['rmse'] if isinstance(stop, dict) else stop)


def predict_batch(model, X):
    return np.asarray(model.predict(X), dtype=np.float64).reshape(-1)


def main(out, smoke):
    # Load
    train = dict(np.load(DATA / 'train.npz'))
    grids = {stage: dict(np.load(DATA / f'{stage}.npz')) for stage, _ in STAGES}
    speed = pd.read_parquet(DATA / 'speed.parquet')['speed']
    origins = pd.DatetimeIndex(train['origins'], tz='UTC')
    y = targets(speed, origins)
    torch.set_num_threads(4)
    context_rows, batch, heads = (32, 8, 3) if smoke else (CONTEXT_ROWS, BATCH, 72)
    start = pd.Timestamp('2025-12-01', tz='UTC')
    select_fit = np.flatnonzero(origins + pd.Timedelta(hours=72) < start)
    select_val = np.flatnonzero((origins >= pd.Timestamp('2025-12-04', tz='UTC')) & (origins <= pd.Timestamp('2026-02-22T23:00:00Z')))
    if smoke:
        select_fit, select_val = select_fit[-40:], select_val[:2]
    epochs = {}  # one epoch per trained model, chosen on the selection window only

    # Fit in context and forecast
    frames, errors = [], {}
    for stage, cutoff in STAGES:
        fit = np.flatnonzero(origins + pd.Timedelta(hours=72) < pd.Timestamp(cutoff, tz='UTC'))
        fitting, Y, dates = take(train, fit), y[fit], origins[fit]
        grid = grids[stage]
        query = pd.DatetimeIndex(grid['origins'], tz='UTC')
        n = len(query)
        pred = np.full((n, 72), np.nan)
        model = None
        for head in range(72):
            if head < heads:
                if model is None:
                    rows, horizons = context_cells(dates, head, context_rows)
                    key = 0 if SHARED else head
                    if key not in epochs:
                        epochs[key] = select_epoch(take(train, select_fit), y[select_fit], origins[select_fit], take(train, select_val), y[select_val], head, smoke)
                    model = new_model(smoke, epochs[key])
                    model.fit(features(fitting, Y, rows, horizons), Y[rows, horizons])
                Xq = features(grid, pred, np.arange(n), np.full(n, head, dtype=int))
                pred[:, head] = np.clip(np.concatenate([predict_batch(model, Xq[i:i + batch]) for i in range(0, n, batch)]), 1., 3000.)
                if not SHARED:
                    model = None
                    torch.cuda.empty_cache()
            else:  # smoke: keep the row count, skip the remaining horizons
                pred[:, head] = pred[:, head - 1]
            print(f'{stage} head {head + 1}', flush=True)
        errors[stage] = (pred - targets(speed, query)) ** 2
        frames.append(pd.DataFrame({'origin_last_input_utc': np.repeat(query.strftime('%Y-%m-%dT%H:%M:%SZ'), 72),
                                    'horizon_hours': np.tile(np.arange(1, 73), len(query)),
                                    'pred_kms': pred.ravel()}))

    # Score: the hourly speed series is the official truth series
    frame = pd.concat(frames, ignore_index=True)
    assert len(frame) == 357840 and np.isfinite(frame.pred_kms).all()
    frame.to_parquet(out / 'predictions.parquet', index=False)
    scores = dict(mse=float(np.concatenate([e.ravel() for e in errors.values()]).mean()),
                  mse_f1=float(errors['f1'].mean()), mse_f2=float(errors['official'].mean()))
    (out / 'scores.json').write_text(json.dumps(scores, indent=2) + '\n')
    print(json.dumps(scores), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    main(args.out, args.smoke)
