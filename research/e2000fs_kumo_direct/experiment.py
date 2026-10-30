"""e2000fs_kumo_direct: Kumo with the Direct strategy, end to end.

NVIDIA Kumo-Tabular v1.0.1 (large, 8 estimators), pretrained: in-context inference on the newest 5000 cells, no gradient training.
One scalar model per horizon.
All 111 features of the E2000 recipe. Query trajectories start empty and consume the model's own forecasts only.
F1 forecasts use targets before 2026-02-26, F2 forecasts targets before 2026-05-29.

usage: experiment.py --out DIR [--smoke]
Data is only read from data-v2. Results depend on the GPU stack; the archive is results/forecast.npz (mse 4391.09).
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')  # GPU0 is reserved
os.environ.setdefault('CC', '/usr/bin/gcc-12')  # this host ships no unversioned cc

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sdm
from sdm.models import KumoTabular
import torch

DATA = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store/e2000fs-lag0')
LAG = 24  # recent speed hours handed to every model
CONTEXT_ROWS = 5000
BATCH = 256
SEED = 42
SHARED = False  # one model for all horizons
ESTIMATORS = 8
STAGES = (('f1', '2026-02-26'), ('official', '2026-05-29'))  # forecast grid, training target cutoff


def targets(speed, origins):
    times = origins.values[:, None] + np.arange(1, 73).astype('timedelta64[h]')[None, :]
    return speed.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, 72)


def take(grid, index):
    return {k: grid[k][index] for k in ('C', 'E', 'past')}


def features(grid, previous, rows, horizons):
    """Origin constants, the last 24 h of speed and this horizon's features."""
    return np.hstack((grid['C'][rows], grid['past'][rows], grid['E'][rows, horizons])).astype(np.float32)


def context_cells(origins, head, limit):
    """The newest origins for this horizon, oldest first."""
    rows = np.argsort(pd.DatetimeIndex(origins).asi8, kind='stable')[-limit:]
    return rows, np.full(len(rows), head, dtype=int)


def table(X, target=False):
    X = np.asarray(X, dtype=np.float32)
    columns = ['target'] if target else [f'f{i:03}' for i in range(X.shape[1])]
    frame = pd.DataFrame(X, columns=columns)
    return sdm.TableTensor.from_pandas(frame, stypes={c: sdm.Stype.numerical for c in columns}, device='cuda')


def new_model(smoke):
    return KumoTabular(task='regression', size='small' if smoke else 'large', device='cuda')


def fit_model(model, X, y, smoke):
    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
        model.fit(x=table(X), y=table(y[:, None], True), num_estimators=1 if smoke else ESTIMATORS, estimator_batch_size=1,
                  generator=torch.Generator(device='cuda').manual_seed(SEED))
    torch.cuda.synchronize()


def predict_batch(model, X):
    """Point forecast: the mean of the 999 predicted quantiles."""
    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
        return model.predict(table(X)).numerical.detach().float().cpu().numpy().astype(np.float64).mean(axis=1)


def main(out, smoke):
    # Load
    train = dict(np.load(DATA / 'train.npz'))
    grids = {stage: dict(np.load(DATA / f'{stage}.npz')) for stage, _ in STAGES}
    speed = pd.read_parquet(DATA / 'speed.parquet')['speed']
    origins = pd.DatetimeIndex(train['origins'], tz='UTC')
    y = targets(speed, origins)
    torch.set_num_threads(4)
    context_rows, batch, heads = (32, 8, 3) if smoke else (CONTEXT_ROWS, BATCH, 72)

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
                    model = new_model(smoke)
                    fit_model(model, features(fitting, Y, rows, horizons), Y[rows, horizons], smoke)
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
