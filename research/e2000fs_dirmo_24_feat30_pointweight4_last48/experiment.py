"""e2000fs_dirmo_24_feat30_pointweight4_last48: DirMO-24 XGBoost on the Top30 features with point-weighted L2, end to end.

Three models, one per 24 h block of the 72 h horizon, each a 24-output vector-leaf tree ensemble. Each target at or
above 550 km/s counts four times in the squared error (horizons 25-72 only; horizons 1-24 keep ordinary L2), and every horizon's weights
are normalized to mean 1. F1 forecasts use models fitted on targets before 2026-02-26, F2 forecasts on targets before
2026-05-29. The Top30 list is post-hoc (permutation delta MSE on F1/F2), so it is official-exposed.

usage: experiment.py --out DIR [--smoke]
Data is only read from data-v2. Reproduces results/forecast.npz (mse 3510.23).
"""
import argparse
import json
from functools import partial
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
              multi_strategy='multi_output_tree', nthread=4, seed=42, num_target=24)
ROUNDS = 900
BLOCKS = (0, 24, 48)
FIRST_WEIGHTED_BLOCK = 1
HIGH_SPEED, HIGH_WEIGHT = 550., 4.
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


def weighted_l2(pred, matrix, block):
    """Gradient and hessian of the point-weighted squared error."""
    target = matrix.get_label().reshape(pred.shape)
    if block < FIRST_WEIGHTED_BLOCK:
        w = np.ones_like(target, dtype=np.float32)
    else:
        w = np.where(target >= HIGH_SPEED, HIGH_WEIGHT, 1.)
        w = (w / w.mean(axis=0)).astype(np.float32)
    return (pred - target) * w, w


def main(out, smoke):
    # Load
    names = json.loads((DATA / 'names.json').read_text())
    train = dict(np.load(DATA / 'train.npz'))
    grids = {stage: dict(np.load(DATA / f'{stage}.npz')) for stage, _ in STAGES}
    speed = pd.read_parquet(DATA / 'speed.parquet')['speed']
    origins = pd.DatetimeIndex(train['origins'], tz='UTC')
    y = targets(speed, origins)
    val = np.flatnonzero((origins >= pd.Timestamp(VALID[0], tz='UTC')) & (origins <= pd.Timestamp(VALID[1])))
    rounds = 2 if smoke else ROUNDS
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
            assert X.shape[1] == Xq.shape[1] == 237
            block_y = y[:, start:start + 24].astype(np.float32)
            fitting = xgb.DMatrix(X[fit], label=block_y[fit], nthread=4)
            valid = xgb.DMatrix(X[val], label=block_y[val], nthread=4)
            history = {}
            settings = dict(PARAMS, base_score=float(block_y[fit].mean()))
            model = xgb.train(settings, fitting, num_boost_round=rounds, obj=partial(weighted_l2, block=b),
                              evals=[(fitting, 'train'), (valid, 'validation')], evals_result=history, verbose_eval=False)
            pd.DataFrame({'round': np.arange(1, rounds + 1), 'train_mse': np.square(history['train']['rmse']),
                          'validation_mse': np.square(history['validation']['rmse'])}).to_csv(out / f'{stage}_b{b}_loss.csv', index=False)
            model.save_model(out / f'{stage}_b{b}.ubj')
            pred[:, start:start + 24] = np.clip(model.inplace_predict(Xq), 1., 3000.)
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
