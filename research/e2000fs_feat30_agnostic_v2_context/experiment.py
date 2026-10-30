"""e2000fs_feat30_agnostic_v2_context: context-aware asymmetric loss, end to end.

Three models, one per 24 h block of the 72 h horizon, each a 24-output vector-leaf tree ensemble on the Top30 features. Their loss
charges under-forecasts more when an out-of-fold classifier expects a fast-wind event in that block (strength 0.25).
F1 forecasts use models fitted on targets before 2026-02-26, F2 forecasts on targets before 2026-05-29.
The Top30 list is post-hoc (permutation delta MSE on F1/F2), so it is official-exposed.

usage: experiment.py --out DIR [--smoke]
Data is only read from data-v2. Reproduces results/forecast.npz (mse 3260.72).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
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
STRENGTH, MARGIN, SCALE = .25, 20., 10.
STAGES = (('f1', '2026-02-26'), ('official', '2026-05-29'))  # forecast grid, training target cutoff


def targets(speed, origins):
    times = origins.values[:, None] + np.arange(1, 73).astype('timedelta64[h]')[None, :]
    return speed.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, 72)


def design(grid, names, start):
    """Selected constants, then the selected varying features of the block's 24 hours, flattened."""
    constant = [i for i, n in enumerate(names['constant']) if n in FEATURES]
    varying = [i for i, n in enumerate(names['varying']) if n in FEATURES]
    block = grid['E'][:, start:start + 24, :][:, :, varying].reshape(len(grid['C']), -1)
    return np.hstack([grid['C'][:, constant], block]).astype(np.float32)


def purged_folds(origins):
    """Five contiguous folds; origins within 72 h of the held fold leave its training set."""
    origins = np.asarray(origins).astype('datetime64[ns]')
    gap = np.timedelta64(72, 'h')
    for held in np.array_split(np.arange(len(origins)), 5):
        yield np.flatnonzero((origins < origins[held[0]] - gap) | (origins > origins[held[-1]] + gap)), held


def event_probabilities(X, block_y, origins, rounds):
    """Out-of-fold probability that this 24 h block reaches 550 km/s or rises 100 km/s in 6 h."""
    event = ((block_y.max(1) >= 550) | ((block_y[:, 6:] - block_y[:, :-6]).max(1) >= 100)).astype(np.int32)
    params = {k: v for k, v in PARAMS.items() if k != 'multi_strategy'}
    params.update(n_estimators=rounds, objective='binary:logistic', eval_metric='logloss')
    probabilities = np.full(len(event), np.nan)
    for fit, held in purged_folds(origins):
        if len(np.unique(event[fit])) < 2:
            probabilities[held] = event[fit].mean()
            continue
        model = xgb.XGBClassifier(**params, base_score=float(np.clip(np.mean(event[fit]), 1e-4, 1 - 1e-4)))
        model.fit(X[fit], np.asarray(event[fit], dtype=np.float32), verbose=False)
        probabilities[held] = model.get_booster().predict(xgb.DMatrix(X[held], nthread=3))
    return probabilities


def context_loss(q):
    """Squared error plus a softplus penalty on under-forecasts beyond MARGIN, scaled by the event probability."""
    q = np.asarray(q, dtype=float).reshape(-1, 1)

    def objective(y_true, y_pred):
        p = np.asarray(y_pred)
        e = np.asarray(y_true).reshape(p.shape) - p
        probability = expit((e - MARGIN) / SCALE)
        return -e - STRENGTH * q * SCALE * probability, np.ones_like(p) + STRENGTH * q * probability * (1 - probability)

    return objective


def main(out, smoke):
    # Load
    names = json.loads((DATA / 'names.json').read_text())
    train = dict(np.load(DATA / 'train.npz'))
    grids = {stage: dict(np.load(DATA / f'{stage}.npz')) for stage, _ in STAGES}
    speed = pd.read_parquet(DATA / 'speed.parquet')['speed']
    origins = pd.DatetimeIndex(train['origins'], tz='UTC')
    y = targets(speed, origins)
    rounds = 3 if smoke else PARAMS['n_estimators']
    step = 50 if smoke else 1

    # Train and forecast
    frames, errors = [], {}
    for stage, cutoff in STAGES:
        fit = np.flatnonzero(origins + pd.Timedelta(hours=72) < pd.Timestamp(cutoff, tz='UTC'))[::step]
        grid = grids[stage]
        query = pd.DatetimeIndex(grid['origins'], tz='UTC')
        pred = np.empty((len(query), 72))
        for start in BLOCKS:
            X, Xq = design(train, names, start)[fit], design(grid, names, start)
            block_y = y[fit, start:start + 24]
            q = event_probabilities(X, block_y, origins[fit].values, rounds)
            model = xgb.XGBRegressor(**{**PARAMS, 'n_estimators': rounds, 'objective': context_loss(q),
                                        'base_score': float(np.asarray(block_y, dtype=np.float32).mean())})
            model.fit(X, np.asarray(block_y, dtype=np.float32), verbose=False)
            pred[:, start:start + 24] = model.get_booster().predict(xgb.DMatrix(Xq, nthread=3))
            print(f'{stage} block {start}: fit {len(fit)} origins', flush=True)
        pred = np.clip(pred, 1, 3000)
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
