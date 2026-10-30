"""e2000fs_budget_spectral: point4 loss plus a spectral penalty, end to end.

Three models, one per 24 h block of the 72 h horizon, each a 24-output vector-leaf tree ensemble on the Top30 features.
point4 weighted L2 plus a smooth penalty on the frequency content of the 24 h error (coefficient .10, scale 50).
Training keeps a 72 h purge after the 72 h label window (origin + 144 h <= cutoff); cutoffs are 2026-02-26 for F1 and 2026-05-29 for F2.
The Top30 list is post-hoc (permutation delta MSE on F1/F2), so it is official-exposed.

usage: experiment.py --out DIR [--smoke]
Data is only read from data-v2. Reproduces results/forecast.npz (mse 3495.59).
"""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ['OMP_NUM_THREADS'] = '3'

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
PARAMS = dict(max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, min_child_weight=20, reg_lambda=5,
              objective='reg:squarederror', tree_method='hist', device='cpu', eval_metric='rmse',
              multi_strategy='multi_output_tree', nthread=3, seed=42, num_target=24)
ROUNDS = 900
BLOCKS = (0, 24, 48)
COEFFICIENT, SCALE = .10, 50.
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


def point4_weights(y):
    """Each target at or above 550 km/s counts four times; every horizon's weights have mean 1."""
    w = np.where(y >= 550., 4., 1.)
    return (w / w.mean(axis=0)).astype(np.float32)


def point4_spectral(pred, matrix):
    """point4 weighted L2 plus a smooth L1 penalty on the orthonormal FFT of the 24 h error."""
    target = matrix.get_label().reshape(pred.shape)
    w = point4_weights(target)
    error = pred - target
    z = np.fft.fft(error, axis=1, norm='ortho')
    rho = 1 / np.sqrt(1 + abs(z) ** 2 / SCALE ** 2)
    grad = error * w + COEFFICIENT * np.fft.ifft(z * rho, axis=1, norm='ortho').real
    return grad.astype(np.float32), (w + COEFFICIENT).astype(np.float32)


def main(out, smoke):
    # Load
    names = json.loads((DATA / 'names.json').read_text())
    train = dict(np.load(DATA / 'train.npz'))
    grids = {stage: dict(np.load(DATA / f'{stage}.npz')) for stage, _ in STAGES}
    speed = pd.read_parquet(DATA / 'speed.parquet')['speed']
    origins = pd.DatetimeIndex(train['origins'], tz='UTC')
    y = targets(speed, origins)
    rounds = 2 if smoke else ROUNDS
    step = 100 if smoke else 1
    X = [design(train, names, start) for start in BLOCKS]

    # Train and forecast
    frames, errors = [], {}
    for stage, cutoff in STAGES:
        cutoff = pd.Timestamp(cutoff, tz='UTC')
        fit = np.flatnonzero(origins + pd.Timedelta(hours=144) <= cutoff)[::step]
        grid = grids[stage]
        query = pd.DatetimeIndex(grid['origins'], tz='UTC')
        pred = np.empty((len(query), 72))
        for b, start in enumerate(BLOCKS):
            target = y[fit, start:start + 24].astype(np.float32)
            matrix = xgb.DMatrix(X[b][fit], label=target, nthread=3)
            model = xgb.train(dict(PARAMS, base_score=float(target.mean())), matrix, num_boost_round=rounds, obj=point4_spectral)
            pred[:, start:start + 24] = np.clip(model.inplace_predict(design(grid, names, start)), 1., 3000.)
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
