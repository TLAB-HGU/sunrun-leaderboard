"""e1604_hsspeak25: peak-and-timing correction of the E0472 base forecast, end to end.

Two HistGradientBoosting heads predict the peak speed and its hour for each base forecast of three seeds. A Gaussian bump around that
hour lifts horizons 25-72 by 0.15 of the gap between the predicted peak and the base minimum; the three seeds are averaged. F2 fits use
every prepared training row (the original contract); F1 fits use rows whose 72 h label window ends before 2026-02-26.
Inputs are the prepared causal features and frozen E0472 base forecasts from the ACE+2h era.

usage: experiment.py --out DIR
Data is only read from data-v2. Reproduces results/e1604_hsspeak25.parquet (mse 3674.00).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

STORE = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store')
PARAMS = dict(max_iter=500, learning_rate=0.04, max_depth=4, min_samples_leaf=50, random_state=0)
WIDTH, SHRINK = 12., 0.15
F1_CUTOFF = pd.Timestamp('2026-02-26T00:00:00Z')


def targets(speed, origins):
    times = origins.values[:, None] + np.arange(1, 73).astype('timedelta64[h]')[None, :]
    return speed.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, 72)


def predict(models, x, base):
    """Seed-averaged correction of the (seed, origin, horizon) base forecasts."""
    peak = models[0].predict(x.reshape(-1, x.shape[-1])).reshape(x.shape[:2])
    hour = models[1].predict(x.reshape(-1, x.shape[-1])).reshape(x.shape[:2])
    hh = np.arange(1, 73, dtype=float)
    bump = np.exp(-((hh[None, None, :] - np.clip(hour, 1, 72)[:, :, None]) ** 2) / (2 * WIDTH ** 2))
    bump /= np.nanmax(bump, axis=2, keepdims=True)
    lift = SHRINK * (peak[:, :, None] - np.nanmin(base, axis=2, keepdims=True)) * bump
    lift[:, :, :24] = 0.0
    return (base + np.nan_to_num(lift)).mean(axis=0)


def main(out):
    # Load
    with np.load(STORE / 'e1604/prepared_inputs.npz') as z:
        train_x, peak, hour = z['train_x'], z['train_peak'], z['train_hour']
        x2, base2, origins2 = z['official_x'], z['base'], pd.to_datetime(z['origins_ns'], utc=True)
    with np.load(STORE / 'e1604/hsspeak_f1_inputs.npz') as z:
        train_origins = pd.to_datetime(z['train_origins_ns'], utc=True)
        x1, base1, origins1 = z['f1_x'], z['f1_base'], pd.to_datetime(z['f1_origins_ns'], utc=True)
    speed = pd.read_parquet(STORE / 'e2000fs-lag0/speed.parquet')['speed']
    f1_rows = np.asarray(train_origins + pd.Timedelta(hours=72) < F1_CUTOFF)

    # Train and forecast
    with threadpool_limits(limits=1):
        models2 = [HistGradientBoostingRegressor(**PARAMS).fit(train_x, y) for y in (peak, hour)]
        models1 = [HistGradientBoostingRegressor(**PARAMS).fit(train_x[f1_rows], y[f1_rows]) for y in (peak, hour)]
        pred1, pred2 = predict(models1, x1, base1), predict(models2, x2, base2)

    # Score: the hourly speed series is the official truth series
    origins = origins1.append(origins2)
    pred = np.vstack([pred1, pred2])
    assert pred.shape == (4970, 72) and np.isfinite(pred).all()
    pd.DataFrame({'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), 72),
                  'horizon_hours': np.tile(np.arange(1, 73), len(origins)),
                  'pred_kms': pred.ravel()}).to_parquet(out / 'predictions.parquet', index=False)
    errors = (pred - targets(speed, origins)) ** 2
    scores = dict(mse=float(errors.mean()), mse_f1=float(errors[:len(origins1)].mean()), mse_f2=float(errors[len(origins1):].mean()))
    (out / 'scores.json').write_text(json.dumps(scores, indent=2) + '\n')
    print(json.dumps(scores), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    main(args.out)
