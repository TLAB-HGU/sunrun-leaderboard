"""naive: the last observed speed held for 72 hours.

usage: experiment.py --out DIR
Data is only read from data-v2. Scores mse 9029.70 on the leaderboard.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store/e2000fs-lag0')


def targets(speed, origins):
    times = origins.values[:, None] + np.arange(1, 73).astype('timedelta64[h]')[None, :]
    return speed.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, 72)


def main(out):
    # Load
    speed = pd.read_parquet(DATA / 'speed.parquet')['speed']
    origins = pd.DatetimeIndex(np.concatenate([np.load(DATA / f'{grid}.npz')['origins'] for grid in ('f1', 'official')]), tz='UTC')
    f1 = len(np.load(DATA / 'f1.npz')['origins'])

    # Forecast
    pred = np.repeat(speed.reindex(origins).to_numpy()[:, None], 72, axis=1)

    # Score: the hourly speed series is the official truth series
    assert pred.shape == (4970, 72) and np.isfinite(pred).all()
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
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    main(args.out)
