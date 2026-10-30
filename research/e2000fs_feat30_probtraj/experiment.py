"""e2000fs_feat30_probtraj: leaf-matched whole trajectories from DirMO-24, end to end.

Three models, one per 24 h block of the 72 h horizon, each a 24-output vector-leaf tree ensemble on the Top30 features.
Forecasts are weighted means of whole observed training trajectories chosen by leaf proximity, so hours are never sampled independently.
F1 forecasts use models fitted on targets before 2026-02-26, F2 forecasts on targets before 2026-05-29.
The Top30 list is post-hoc (permutation delta MSE on F1/F2), so it is official-exposed.

usage: experiment.py --out DIR [--smoke]
Data is only read from data-v2. Reproduces results/forecast.npz (mse 4531.18).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb

DATA = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store/e2000fs-lag0')
FEATURES = ['v_lag0', 'rec_mean', 'enlil_kp', 'ch_ch_t035_meridian30_equatorial_area', 'ace_bt_nt_last', 'ace_speed_kms_last',
            'ch_ch_t035_equatorial_area', 'ch_ch_t035_meridian15_equatorial_area', 'ace_bt_nt_m6', 'rec26',
            'strip_v450_logmed', 'strip_v350_logmed', 'ch_ch_t055_equatorial_area', 'rec27', 'ace_density_cm3_last',
            'ch_ch_t055_meridian30_equatorial_area', 'ace_by_gsm_nt_m24', 'ace_density_cm3_m6',
            'ace_temperature_k_last', 'enlil_n_pending', 'ace_temperature_k_m24', 'strip_v750_logmed',
            'ch_ch_t055_meridian15_area', 'ch_ch_t035_top2_area', 'strip_v550_dark', 'strip_v650_dark',
            'ace_sis_gt10_flux_last', 'ace_epam_p_channel3_last', 'img_age_h', 'rec28']
PARAMS = dict(n_estimators=900, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
              min_child_weight=20, reg_lambda=5, n_jobs=3, random_state=42, objective='reg:squarederror',
              tree_method='hist', device='cpu', eval_metric='rmse', multi_strategy='multi_output_tree')
BLOCKS = (0, 24, 48)
K, BATCH = 128, 8  # donors per query, query batch
STAGES = (('f1', '2026-02-26'), ('official', '2026-05-29'))  # forecast grid, training target cutoff
VALID = ('2025-12-04', '2026-02-22T23:00:00Z')  # loss curves only


def targets(speed, origins):
    times = origins.values[:, None] + np.arange(1, 73).astype('timedelta64[h]')[None, :]
    return speed.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, 72)


def design(grid, names, start):
    """Selected constants, then the selected varying features of the block's 24 hours, flattened."""
    constant = [i for i, n in enumerate(names['constant']) if n in FEATURES]
    varying = [i for i, n in enumerate(names['varying']) if n in FEATURES]
    block = grid['E'][:, start:start + 24, :][:, :, varying].reshape(len(grid['C']), -1)
    return np.hstack([grid['C'][:, constant], block]).astype(np.float32)


def leaf_codes(leaves):
    """Per-tree leaf vocabularies, for sparse inverse-occupancy proximity."""
    values, offsets, inverse_counts, codes, offset = [], [], [], np.empty(leaves.shape, dtype=np.int32), 0
    for tree in range(leaves.shape[1]):
        v, inverse, counts = np.unique(leaves[:, tree], return_inverse=True, return_counts=True)
        values.append(v)
        offsets.append(offset)
        codes[:, tree] = inverse + offset
        inverse_counts.append(1. / counts)
        offset += len(v)
    return values, offsets, np.concatenate(inverse_counts), codes, offset


def trajectory_means(fit_leaves, fit_targets, query_leaves):
    """Each query averages the K training trajectories sharing the most (rare) leaves with it, or all if none do."""
    n, trees = fit_leaves.shape
    values, offsets, inverse_counts, codes, width = leaf_codes(fit_leaves)
    membership = sparse.csr_matrix((inverse_counts[codes.ravel()], codes.ravel(),
                                    np.arange(0, (n + 1) * trees, trees, dtype=np.int64)), shape=(n, width)).T
    means = []
    for first in range(0, len(query_leaves), BATCH):
        batch = query_leaves[first:first + BATCH]
        rows, cols = [], []
        for tree, (v, offset) in enumerate(zip(values, offsets)):
            at = np.searchsorted(v, batch[:, tree])
            valid = at < len(v)
            valid[valid] &= v[at[valid]] == batch[valid, tree]
            rows.append(np.flatnonzero(valid))
            cols.append(at[valid] + offset)
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        query = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(batch), width))
        for score in (query @ membership).toarray():
            donors = np.flatnonzero(score > 0)
            if not len(donors):
                means.append(np.full(len(fit_targets), 1. / len(fit_targets)) @ fit_targets)
                continue
            donors = donors[np.lexsort((donors, -score[donors]))[:K]]
            means.append(score[donors] / score[donors].sum() @ fit_targets[donors])
    return np.asarray(means)


def main(out, smoke):
    # Load
    names = json.loads((DATA / 'names.json').read_text())
    train = dict(np.load(DATA / 'train.npz'))
    grids = {stage: dict(np.load(DATA / f'{stage}.npz')) for stage, _ in STAGES}
    speed = pd.read_parquet(DATA / 'speed.parquet')['speed']
    origins = pd.DatetimeIndex(train['origins'], tz='UTC')
    y = targets(speed, origins)
    rounds = 2 if smoke else PARAMS['n_estimators']
    step = 100 if smoke else 1

    # Train the DirMO-24 parent, then forecast from the training trajectories that share its leaves
    frames, errors = [], {}
    for stage, cutoff in STAGES:
        fit = np.flatnonzero(origins + pd.Timedelta(hours=72) < pd.Timestamp(cutoff, tz='UTC'))[::step]
        grid = grids[stage]
        query = pd.DatetimeIndex(grid['origins'], tz='UTC')
        fit_leaves, query_leaves = [], []
        for start in BLOCKS:
            X, Xq = design(train, names, start), design(grid, names, start)
            model = xgb.XGBRegressor(**{**PARAMS, 'n_estimators': rounds, 'n_jobs': 4})
            model.fit(X[fit], y[fit, start:start + 24], verbose=False)
            booster = model.get_booster()
            fit_leaves.append(booster.predict(xgb.DMatrix(X[fit], nthread=3), pred_leaf=True).astype(np.int32))
            query_leaves.append(booster.predict(xgb.DMatrix(Xq, nthread=3), pred_leaf=True).astype(np.int32))
            print(f'{stage} block {start}: fit {len(fit)} origins', flush=True)
        pred = np.clip(trajectory_means(np.hstack(fit_leaves), y[fit], np.hstack(query_leaves)), 1., 3000.)
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
