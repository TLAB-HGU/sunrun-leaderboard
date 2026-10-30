"""e2000fs_recmo_24: RecMO-24: one 24-output model applied 3 times on its own forecasts, end to end.

All 111 features of the E2000 recipe. Training uses observed (filled) speeds as lags and trajectories;
inference uses the model's own forecasts only.
F1 forecasts use models fitted on targets before 2026-02-26, F2 forecasts on targets before 2026-05-29.

usage: experiment.py --out DIR [--smoke]
Data is only read from data-v2. Reproduces results/forecast.npz (mse 3608.39).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

DATA = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store/e2000fs-lag0')
PARAMS = dict(n_estimators=900, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
              min_child_weight=20, reg_lambda=5, random_state=42, objective='reg:squarederror',
              tree_method='hist', device='cpu', eval_metric='rmse')
LAG = 24  # recent speed hours handed to every model
SIGMA = 24  # block length in hours
SPANS = [(s, s + SIGMA - 1) for s in range(1, 73, SIGMA)]
SHARED_THREADS = 12
STAGES = (('f1', '2026-02-26'), ('official', '2026-05-29'))  # forecast grid, training target cutoff
VALID = ('2025-12-04', '2026-02-22T23:00:00Z')  # loss curves only


def targets(speed, origins):
    times = origins.values[:, None] + np.arange(1, 73).astype('timedelta64[h]')[None, :]
    return speed.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, 72)


def take(grid, index):
    return {k: grid[k][index] for k in ('C', 'E', 'past')}


def fit_xgb(X, Y, Xv, Yv, rounds, threads, curve):
    """Multi-output trees for a block longer than one hour; train and validation MSE per round in ``curve``."""
    params = dict(PARAMS, n_estimators=rounds, n_jobs=threads)
    if Y.shape[1] > 1:
        params['multi_strategy'] = 'multi_output_tree'
    model = XGBRegressor(**params)
    model.fit(X, Y if Y.shape[1] > 1 else Y[:, 0], eval_set=[(X, Y), (Xv, Yv)], verbose=False)
    history = model.evals_result()
    pd.DataFrame({'round': np.arange(1, rounds + 1), 'train_mse': np.square(history['validation_0']['rmse']),
                  'validation_mse': np.square(history['validation_1']['rmse'])}).to_csv(curve, index=False)
    return model


def features_recmo(grid, trajectory, s, e):
    """Origin constants, the 24 h ending at horizon s-1, the block's horizon features, and the lead."""
    n = len(grid['C'])
    lag = trajectory[:, s - 1:s - 1 + LAG]
    return np.hstack([grid['C'], lag, grid['E'][:, s - 1:e].reshape(n, -1), np.full((n, 1), s, np.float32)]).astype(np.float32)


def fit_recmo(grid, Y, valid, Yv, rounds, curve):
    T, Tv = np.hstack([grid['past'].astype(np.float64), Y]), np.hstack([valid['past'].astype(np.float64), Yv])
    X = np.vstack([features_recmo(grid, T, s, e) for s, e in SPANS])
    y = np.vstack([Y[:, s - 1:e] for s, e in SPANS])
    Xv = np.vstack([features_recmo(valid, Tv, s, e) for s, e in SPANS])
    yv = np.vstack([Yv[:, s - 1:e] for s, e in SPANS])
    return fit_xgb(X, y, Xv, yv, rounds, SHARED_THREADS, curve)


def predict_recmo(model, grid):
    n = len(grid['C'])
    T = np.hstack([grid['past'].astype(np.float64), np.full((n, 72), np.nan)])
    for s, e in SPANS:
        raw = np.asarray(model.predict(features_recmo(grid, T, s, e)), dtype=np.float64)
        T[:, LAG + s - 1:LAG + e] = raw.reshape(n, e - s + 1)
    return T[:, LAG:]


def main(out, smoke):
    # Load
    train = dict(np.load(DATA / 'train.npz'))
    grids = {stage: dict(np.load(DATA / f'{stage}.npz')) for stage, _ in STAGES}
    speed = pd.read_parquet(DATA / 'speed.parquet')['speed']
    origins = pd.DatetimeIndex(train['origins'], tz='UTC')
    y = targets(speed, origins)
    val = np.flatnonzero((origins >= pd.Timestamp(VALID[0], tz='UTC')) & (origins <= pd.Timestamp(VALID[1])))
    rounds = 2 if smoke else PARAMS['n_estimators']
    step = 64 if smoke else 1
    val = val[::step]
    valid, y_val = take(train, val), y[val]

    # Train and forecast
    frames, errors = [], {}
    for stage, cutoff in STAGES:
        fit = np.flatnonzero(origins + pd.Timedelta(hours=72) < pd.Timestamp(cutoff, tz='UTC'))[::step]
        fitting, y_fit, fit_origins = take(train, fit), y[fit], origins[fit]
        grid = grids[stage]
        query = pd.DatetimeIndex(grid['origins'], tz='UTC')
        model = fit_recmo(fitting, y_fit, valid, y_val, rounds, out / f'{stage}_loss.csv')
        pred = np.clip(predict_recmo(model, grid), 1., 3000.)
        print(f'{stage}: fit {len(fit)} origins', flush=True)
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
