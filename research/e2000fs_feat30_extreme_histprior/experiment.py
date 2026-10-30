"""e2000fs_feat30_extreme_histprior: HL-Gauss histogram heads with fast-wind weights, end to end.

72 XGBoost multi:softprob models, one per horizon, sharing their 24 h block's Top30 inputs. Targets are Gaussian-smoothed
histograms over 16 bins with 550 km/s on an edge, targets at or above 550 km/s count four times, and the point forecast is the
mean of the predicted histogram with no prior correction (the hist_weighted_raw read-out).
F1 forecasts use models fitted on targets before 2026-02-26, F2 forecasts on targets before 2026-05-29.
The Top30 list is post-hoc (permutation delta MSE on F1/F2), so it is official-exposed.

usage: experiment.py --out DIR [--smoke]
Data is only read from data-v2. Reproduces results/forecast.npz (mse 3538.42).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
import xgboost as xgb

DATA = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store/e2000fs-lag0')
FEATURES = ['v_lag0', 'rec_mean', 'enlil_kp', 'ch_ch_t035_meridian30_equatorial_area', 'ace_bt_nt_last', 'ace_speed_kms_last',
            'ch_ch_t035_equatorial_area', 'ch_ch_t035_meridian15_equatorial_area', 'ace_bt_nt_m6', 'rec26',
            'strip_v450_logmed', 'strip_v350_logmed', 'ch_ch_t055_equatorial_area', 'rec27', 'ace_density_cm3_last',
            'ch_ch_t055_meridian30_equatorial_area', 'ace_by_gsm_nt_m24', 'ace_density_cm3_m6',
            'ace_temperature_k_last', 'enlil_n_pending', 'ace_temperature_k_m24', 'strip_v750_logmed',
            'ch_ch_t055_meridian15_area', 'ch_ch_t035_top2_area', 'strip_v550_dark', 'strip_v650_dark',
            'ace_sis_gt10_flux_last', 'ace_epam_p_channel3_last', 'img_age_h', 'rec28']
EDGES = np.array([240., 290., 340., 390., 440., 490., 520., 550., 580., 610., 640., 680., 730., 800., 900., 1000., 1150.])
FAST_BIN = 7  # first bin at or above 550 km/s
K = len(EDGES) - 1
H_REF = (1. / K) * (1. - 1. / K)  # softmax Hessian scale; keeps the regression regularisation comparable
PARAMS = dict(objective='multi:softprob', num_class=K, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
              min_child_weight=20. * H_REF, reg_lambda=5. * H_REF, nthread=3, seed=42, tree_method='hist', device='cpu',
              base_score=0.0, disable_default_eval_metric=1)
ROUNDS = 300
HIGH_WEIGHT = 4.
BLOCKS = (0, 24, 48)
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


def bin_index(y):
    return np.clip(np.searchsorted(EDGES, y, side='right') - 1, 0, K - 1)


def soft_labels(y):
    """Gaussian probability mass of each bin around y; bandwidth half the bin width, within 10-50 km/s."""
    pinned = np.clip(y, EDGES[0], EDGES[-1])
    index = bin_index(pinned)
    band = np.minimum(np.maximum(0.5 * (EDGES[index + 1] - EDGES[index]), 10.), 50.)[:, None]
    q = np.clip(ndtr((EDGES[None, 1:] - pinned[:, None]) / band) - ndtr((EDGES[None, :-1] - pinned[:, None]) / band), 0., None)
    return (q / q.sum(axis=1, keepdims=True)).astype(np.float32)


def softmax(margins):
    z = np.asarray(margins, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def representatives(values):
    """Training mean speed of each bin, or its midpoint when empty; shared by all horizons of a block."""
    index = bin_index(values)
    out = (EDGES[:-1] + EDGES[1:]) / 2.
    for k in range(K):
        if (index == k).any():
            out[k] = values[index == k].mean()
    return np.maximum(out, 1.)


def fit_head(X, y, rounds):
    """Soft-label cross-entropy; both gradient and Hessian carry the sample weight, as DMatrix weights are ignored here."""
    labels = np.asarray(soft_labels(y), dtype=np.float64)
    weights = np.ones(K)
    weights[FAST_BIN:] = HIGH_WEIGHT
    w = weights[bin_index(y)]
    w = (w / w.mean())[:, None]

    def objective(margins, matrix):
        p = softmax(margins)
        return (p - labels) * w, np.maximum(p * (1. - p), 1e-6) * w

    matrix = xgb.DMatrix(X, label=np.arange(len(X), dtype=np.float32), nthread=3)
    return xgb.train(PARAMS, matrix, num_boost_round=rounds, obj=objective, verbose_eval=False)


def main(out, smoke):
    # Load
    names = json.loads((DATA / 'names.json').read_text())
    train = dict(np.load(DATA / 'train.npz'))
    grids = {stage: dict(np.load(DATA / f'{stage}.npz')) for stage, _ in STAGES}
    speed = pd.read_parquet(DATA / 'speed.parquet')['speed']
    origins = pd.DatetimeIndex(train['origins'], tz='UTC')
    y = targets(speed, origins)
    rounds = 3 if smoke else ROUNDS
    step = 50 if smoke else 1

    # Train and forecast
    frames, errors = [], {}
    for stage, cutoff in STAGES:
        fit = np.flatnonzero(origins + pd.Timedelta(hours=72) < pd.Timestamp(cutoff, tz='UTC'))[::step]
        grid = grids[stage]
        query = pd.DatetimeIndex(grid['origins'], tz='UTC')
        pred = np.empty((len(query), 72))
        for start in BLOCKS:
            X, Xq = design(train, names, start), xgb.DMatrix(design(grid, names, start), nthread=3)
            reps = representatives(y[fit, start:start + 24].ravel())
            for h in range(start, start + 24):
                model = fit_head(X[fit], y[fit, h], rounds)
                pred[:, h] = softmax(model.predict(Xq, output_margin=True)) @ reps
            print(f'{stage} block {start}: fit {len(fit)} origins', flush=True)
        pred = np.clip(pred, 1., 3000.)
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
