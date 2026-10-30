"""e2000fs_feat30_pointweight4: tuned LightGBM with point-weighted L2, end to end.

72 independent LightGBM models, one per horizon, sharing their 24 h block's inputs. Each model weights its own targets at or above 550 km/s four times.
F1 forecasts use models fitted on targets before 2026-02-26, F2 forecasts on targets before 2026-05-29.
The Top30 list is post-hoc (permutation delta MSE on F1/F2), so it is official-exposed.

usage: experiment.py --out DIR [--smoke]
Data is only read from data-v2. Reproduces results/forecast.npz (mse 3732.27).
"""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '2'

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

DATA = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store/e2000fs-lag0')
FEATURES = ['v_lag0', 'rec_mean', 'enlil_kp', 'ch_ch_t035_meridian30_equatorial_area', 'ace_bt_nt_last', 'ace_speed_kms_last',
            'ch_ch_t035_equatorial_area', 'ch_ch_t035_meridian15_equatorial_area', 'ace_bt_nt_m6', 'rec26',
            'strip_v450_logmed', 'strip_v350_logmed', 'ch_ch_t055_equatorial_area', 'rec27', 'ace_density_cm3_last',
            'ch_ch_t055_meridian30_equatorial_area', 'ace_by_gsm_nt_m24', 'ace_density_cm3_m6',
            'ace_temperature_k_last', 'enlil_n_pending', 'ace_temperature_k_m24', 'strip_v750_logmed',
            'ch_ch_t055_meridian15_area', 'ch_ch_t035_top2_area', 'strip_v550_dark', 'strip_v650_dark',
            'ace_sis_gt10_flux_last', 'ace_epam_p_channel3_last', 'img_age_h', 'rec28']
# Winner of 150 TPE trials on the internal window (study 1, trial 32).
PARAMS = dict(n_estimators=400, objective='regression', learning_rate=0.011665375590900416, max_depth=6, num_leaves=39,
              subsample=0.8158360436957313, subsample_freq=1, colsample_bytree=0.796156879318864, min_child_samples=76,
              min_child_weight=1.7495117760617724, reg_alpha=12.889193670860928, reg_lambda=9.785792859825623,
              random_state=42, device_type='cpu', verbosity=-1, n_jobs=2)
BLOCKS = (0, 24, 48)
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


def sample_weights(y):
    """Only this horizon's own target at or above 550 km/s counts four times; mean 1."""
    w = np.where(y >= 550., 4., 1.)
    return w / w.mean()


def main(out, smoke):
    # Load
    names = json.loads((DATA / 'names.json').read_text())
    train = dict(np.load(DATA / 'train.npz'))
    grids = {stage: dict(np.load(DATA / f'{stage}.npz')) for stage, _ in STAGES}
    speed = pd.read_parquet(DATA / 'speed.parquet')['speed']
    origins = pd.DatetimeIndex(train['origins'], tz='UTC')
    y = targets(speed, origins)
    val = np.flatnonzero((origins >= pd.Timestamp(VALID[0], tz='UTC')) & (origins <= pd.Timestamp(VALID[1])))
    params = dict(PARAMS, n_estimators=2) if smoke else PARAMS
    step = 100 if smoke else 1
    val = val[::step]
    X = {start: design(train, names, start) for start in BLOCKS}

    # Train and forecast: one scalar model per horizon, sharing its block's inputs
    frames, errors = [], {}
    for stage, cutoff in STAGES:
        fit = np.flatnonzero(origins + pd.Timedelta(hours=72) < pd.Timestamp(cutoff, tz='UTC'))[::step]
        grid = grids[stage]
        query = pd.DatetimeIndex(grid['origins'], tz='UTC')
        Xq = {start: design(grid, names, start) for start in BLOCKS}
        pred = np.empty((len(query), 72))
        for h in range(72):
            start = h // 24 * 24
            history = {}
            model = lgb.LGBMRegressor(**params)
            model.fit(X[start][fit], y[fit, h], sample_weight=sample_weights(y[fit, h]), eval_set=[(X[start][fit], y[fit, h]), (X[start][val], y[val, h])],
                      eval_names=['train', 'validation'], eval_metric='l2', callbacks=[lgb.record_evaluation(history)])
            pd.DataFrame({'round': np.arange(1, len(history['train']['l2']) + 1), 'train_mse': history['train']['l2'],
                          'validation_mse': history['validation']['l2']}).to_csv(out / f'{stage}_h{h + 1:02}_loss.csv', index=False)
            pred[:, h] = np.clip(model.predict(Xq[start]), 1., 3000.)
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
