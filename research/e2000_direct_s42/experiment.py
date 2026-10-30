"""e2000_direct_s42: 72 direct XGBoost models, end to end.

One XGBoost model per horizon, seed 42, frozen E2000 recipe.
The 111 E0472 features keep their historical ACE+2h timing: 90 origin-invariant features once, 20 horizon-dependent features
expanded across all 72 leads, 1,530 inputs. Training uses observed labels only, with targets before 2026-02-26 for F1 and
before 2026-05-29 for F2. The original training module survives only as bytecode; its fitted models are under results/.

usage: experiment.py --out DIR [--smoke]
Data is only read from data-v2. Reproduces results/E1987/combined (mse 3818.32).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from xgboost import XGBRegressor

STORE = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store')
DATA = STORE / 'e2000-ace2h'
VARYING = ('rec26', 'rec27', 'rec28', 'rec_mean') + tuple(f'strip_v{v}_{s}' for v in (350, 450, 550, 650, 750)
                                                          for s in ('logmed', 'dark', 'cover')) + ('enlil_dt_target',)
PARAMS = dict(n_estimators=900, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, min_child_weight=20, reg_lambda=5,
              random_state=42, objective='reg:squarederror', tree_method='hist', device='cpu', eval_metric='rmse')
WORKERS = 6  # independent fits, one thread each
STAGES = (('features_f1.parquet', '2026-02-26'), ('official_causal.parquet', '2026-05-29'))  # forecast grid, target cutoff


def targets(speed, origins):
    times = origins.values[:, None] + np.arange(1, 73).astype('timedelta64[h]')[None, :]
    return speed.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, 72)


def wide(path, features):
    """One row per origin: invariant features once, varying features once per lead."""
    frame = pd.read_parquet(DATA / path)
    frame['origin'] = pd.to_datetime(frame.origin, utc=True)
    frame = frame.sort_values(['origin', 'h']).reset_index(drop=True)
    origins = pd.DatetimeIndex(frame.origin.unique())
    columns = []
    for name in features:
        a = frame[name].to_numpy(dtype=np.float32).reshape(len(origins), 72)
        columns.append(a if name in VARYING else a[:, :1])
    return origins, np.hstack(columns)


def fit_horizon(X, y, rounds):
    model = XGBRegressor(**dict(PARAMS, n_estimators=rounds, n_jobs=1))
    model.fit(X, y, verbose=False)
    return model


def main(out, smoke):
    # Load
    features = [n for n in json.loads((DATA / 'features_v3.json').read_text())['features'] if n != 'h']
    speed = pd.read_parquet(STORE / 'e2000fs-lag0/speed.parquet')['speed']
    origins, X = wide('train_complete_causal.parquet', features)
    labels = pd.read_parquet(DATA / 'train_v3.parquet', columns=['origin', 'h', 'y']).set_index(['origin', 'h']).y
    index = pd.MultiIndex.from_product([origins, range(1, 73)], names=['origin', 'h'])
    y = labels.reindex(index).to_numpy(float).reshape(-1, 72)
    times = origins.tz_localize(None).to_numpy(dtype='datetime64[ns]')[:, None] + np.arange(1, 73).astype('timedelta64[h]')[None, :]
    rounds, step = (2, 100) if smoke else (PARAMS['n_estimators'], 1)

    # Train and forecast: one model per horizon on its observed targets
    frames, errors = [], []
    for path, cutoff in STAGES:
        query, Xq = wide(path, features)
        mask = np.isfinite(y) & (times < np.datetime64(cutoff))
        rows = [np.flatnonzero(mask[:, h])[::step] for h in range(72)]
        models = Parallel(n_jobs=WORKERS)(delayed(fit_horizon)(X[rows[h]], y[rows[h], h], rounds) for h in range(72))
        pred = np.column_stack([m.predict(Xq) for m in models]).astype(np.float64)
        print(f'{cutoff}: {len(rows[0])} origins at horizon 1', flush=True)
        assert pred.shape == (len(query), 72) and np.isfinite(pred).all()
        errors.append((pred - targets(speed, query)) ** 2)
        frames.append(pd.DataFrame({'origin_last_input_utc': np.repeat(query.strftime('%Y-%m-%dT%H:%M:%SZ'), 72),
                                    'horizon_hours': np.tile(np.arange(1, 73), len(query)),
                                    'pred_kms': pred.ravel()}))

    # Score: the hourly speed series is the official truth series
    frame = pd.concat(frames, ignore_index=True)
    assert len(frame) == 357840
    frame.to_parquet(out / 'predictions.parquet', index=False)
    scores = dict(mse=float(np.concatenate([e.ravel() for e in errors]).mean()), mse_f1=float(errors[0].mean()), mse_f2=float(errors[1].mean()))
    (out / 'scores.json').write_text(json.dumps(scores, indent=2) + '\n')
    print(json.dumps(scores), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    main(args.out, args.smoke)
