"""e2000fs_feat30_wavelet: DirMO-24 with difference and Haar inputs, end to end.

Three models, one per 24 h block of the 72 h horizon, each a 24-output vector-leaf tree ensemble on the Top30 features.
Each block also sees hourly differences and Haar coefficients of its varying features.
F1 forecasts use models fitted on targets before 2026-02-26, F2 forecasts on targets before 2026-05-29.
The Top30 list is post-hoc (permutation delta MSE on F1/F2), so it is official-exposed.

usage: experiment.py --out DIR [--smoke]
Data is only read from data-v2. Reproduces results/forecast.npz (mse 3288.02).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
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
STAGES = (('f1', '2026-02-26'), ('official', '2026-05-29'))  # forecast grid, training target cutoff
VALID = ('2025-12-04', '2026-02-22T23:00:00Z')  # loss curves only


def targets(speed, origins):
    times = origins.values[:, None] + np.arange(1, 73).astype('timedelta64[h]')[None, :]
    return speed.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, 72)


def haar24(x):
    """Orthonormal three-level Haar coefficients along the hour axis: coarse3, detail3/6/12."""
    values = np.moveaxis(np.asarray(x, dtype=np.float64), 1, -1)
    details = []
    for _ in range(3):
        even, odd = values[..., ::2], values[..., 1::2]
        details.append((even - odd) / np.sqrt(2.))
        values = (even + odd) / np.sqrt(2.)
    return np.moveaxis(np.concatenate([values, *reversed(details)], axis=-1), -1, 1)


def design(grid, names, start):
    """Selected constants, the selected varying features of the block's 24 hours, their hourly differences and Haar coefficients."""
    constant = [i for i, n in enumerate(names['constant']) if n in FEATURES]
    varying = [i for i, n in enumerate(names['varying']) if n in FEATURES]
    C, E = grid['C'][:, constant], grid['E'][:, start:start + 24, :][:, :, varying]
    return np.concatenate([C, E.reshape(len(C), 216), np.diff(E, axis=1).reshape(len(C), 207),
                           haar24(E).reshape(len(C), 216)], axis=1).astype(np.float32)


def main(out, smoke):
    # Load
    names = json.loads((DATA / 'names.json').read_text())
    train = dict(np.load(DATA / 'train.npz'))
    grids = {stage: dict(np.load(DATA / f'{stage}.npz')) for stage, _ in STAGES}
    speed = pd.read_parquet(DATA / 'speed.parquet')['speed']
    origins = pd.DatetimeIndex(train['origins'], tz='UTC')
    y = targets(speed, origins)
    val = np.flatnonzero((origins >= pd.Timestamp(VALID[0], tz='UTC')) & (origins <= pd.Timestamp(VALID[1])))
    rounds = 3 if smoke else PARAMS['n_estimators']
    step = 100 if smoke else 1
    val = val[::step]

    # Train and forecast
    frames, errors = [], {}
    for stage, cutoff in STAGES:
        fit = np.flatnonzero(origins + pd.Timedelta(hours=72) < pd.Timestamp(cutoff, tz='UTC'))[::step]
        grid = grids[stage]
        query = pd.DatetimeIndex(grid['origins'], tz='UTC')
        pred = np.empty((len(query), 72))
        for b, start in enumerate(BLOCKS):
            X, Xq = design(train, names, start), design(grid, names, start)
            block_y = y[:, start:start + 24].astype(np.float32)
            model = xgb.XGBRegressor(**{**PARAMS, 'n_estimators': rounds, 'base_score': float(block_y[fit].mean())})
            model.fit(X[fit], block_y[fit], eval_set=[(X[fit], block_y[fit]), (X[val], block_y[val])], verbose=False)
            curve = model.evals_result()
            pd.DataFrame({'round': np.arange(1, rounds + 1), 'train_mse': np.square(curve['validation_0']['rmse']),
                          'validation_mse': np.square(curve['validation_1']['rmse'])}).to_csv(out / f'{stage}_b{b}_loss.csv', index=False)
            pred[:, start:start + 24] = np.clip(model.predict(Xq), 1., 3000.)
            print(f'{stage} b{b}: fit {len(fit)} origins', flush=True)
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
