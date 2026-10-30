"""e2000fs_feat30_smooth_wcrps: lognormal LightGBM trained on smooth-weighted CRPS, end to end.

72 lognormal LightGBM models, one per horizon, sharing their 24 h block's inputs. Point forecast: the predictive mean.
CRPS weight rises smoothly around 550 km/s (width 100) on integration coordinates, never on whole observations.
F1 forecasts use models fitted on targets before 2026-02-26, F2 forecasts on targets before 2026-05-29.
The Top30 list is post-hoc (permutation delta MSE on F1/F2), so it is official-exposed.

usage: experiment.py --out DIR [--smoke]
Data is only read from data-v2. Reproduces results/forecast.npz (mse 3289.44).
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
from scipy.special import ndtr

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
NODES, LOWER, UPPER = 64, 1., 6000.  # CRPS integration
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


def quadrature(y):
    """Gauss-Legendre nodes in log speed, split at each observation, over [LOWER, UPPER] km/s."""
    q, w = np.polynomial.legendre.leggauss(NODES)
    edges = [np.full_like(y, np.log(LOWER)), np.log(y), np.full_like(y, np.log(UPPER))]
    zz, ww, ii = [], [], []
    for k in range(2):
        half = (edges[k + 1] - edges[k])[:, None] / 2
        z = np.exp((edges[k + 1] + edges[k])[:, None] / 2 + half * q)
        zz.append(z)
        ww.append(half * w * z)
        ii.append(np.full_like(z, float(k)))
    return np.concatenate(zz, axis=1), np.concatenate(ww, axis=1), np.concatenate(ii, axis=1)


def cdf_derivatives(z, mu, sigma):
    a = (np.log(z) - mu[:, None]) / sigma[:, None]
    density = np.exp(-.5 * a * a) / np.sqrt(2. * np.pi)
    return ndtr(a), -density / sigma[:, None], -a * density


def make_gradients(y):
    """Smooth tail-weighted CRPS; Gauss-Newton diagonal as Hessian."""
    z, weights, observed = quadrature(y)
    weights = weights * (1. + ndtr((z - 550.) / 100.))

    def gradients(mu, rho):
        sigma = np.exp(rho)
        cdf, dmu, drho = cdf_derivatives(z, mu, sigma)
        error = cdf - observed
        grad = np.column_stack([2. * np.sum(weights * error * d, axis=1) for d in (dmu, drho)])
        hess = np.column_stack([2. * np.sum(weights * d ** 2, axis=1) for d in (dmu, drho)])
        return grad, np.maximum(hess, 1e-8)
    return gradients


def fit_horizon(X, y, rounds):
    """Boost mu and rho = log sigma in turn; each round both boosters use the same gradients."""
    gradients = make_gradients(y)
    config = {k: v for k, v in PARAMS.items() if k not in ('n_estimators', 'n_jobs')}
    config.update(objective='none', metric='None', num_threads=PARAMS['n_jobs'], boost_from_average=False, feature_pre_filter=False)
    offsets = float(np.log(y).mean()), float(np.log(max(float(np.std(np.log(y))), .05)))
    boosters = [lgb.Booster(config, train_set=lgb.Dataset(X, label=y, init_score=np.full(len(y), offset), free_raw_data=False))
                for offset in offsets]

    def score(booster):
        captured = []
        booster.eval_train(feval=lambda pred, data: (captured.append(np.asarray(pred).copy()), ('raw', 0., False))[1])
        return captured[0]

    grad, hess = gradients(np.full(len(y), offsets[0]), np.full(len(y), offsets[1]))
    for _ in range(rounds):
        for c, booster in enumerate(boosters):
            g, h = grad[:, c].copy(), hess[:, c].copy()
            booster.update(fobj=lambda pred, data, g=g, h=h: (g, h))
        grad, hess = gradients(score(boosters[0]), score(boosters[1]))
    return boosters, offsets


def predict_mean(boosters, offsets, X):
    """Lognormal predictive mean exp(mu + sigma^2 / 2)."""
    mu = boosters[0].predict(X, num_threads=PARAMS['n_jobs']) + offsets[0]
    sigma = np.exp(boosters[1].predict(X, num_threads=PARAMS['n_jobs']) + offsets[1])
    return np.exp(mu + .5 * sigma ** 2)


def main(out, smoke):
    # Load
    names = json.loads((DATA / 'names.json').read_text())
    train = dict(np.load(DATA / 'train.npz'))
    grids = {stage: dict(np.load(DATA / f'{stage}.npz')) for stage, _ in STAGES}
    speed = pd.read_parquet(DATA / 'speed.parquet')['speed']
    origins = pd.DatetimeIndex(train['origins'], tz='UTC')
    y = targets(speed, origins)
    rounds = 3 if smoke else PARAMS['n_estimators']
    step = 100 if smoke else 1
    X = {start: design(train, names, start) for start in BLOCKS}

    # Train and forecast: one mu/rho pair per horizon, sharing its block's inputs
    frames, errors = [], {}
    for stage, cutoff in STAGES:
        fit = np.flatnonzero(origins + pd.Timedelta(hours=72) < pd.Timestamp(cutoff, tz='UTC'))[::step]
        grid = grids[stage]
        query = pd.DatetimeIndex(grid['origins'], tz='UTC')
        Xq = {start: design(grid, names, start) for start in BLOCKS}
        pred = np.empty((len(query), 72))
        for h in range(72):
            start = h // 24 * 24
            boosters, offsets = fit_horizon(X[start][fit], y[fit, h], rounds)
            pred[:, h] = np.clip(predict_mean(boosters, offsets, Xq[start]), 1., 3000.)
            print(f'{stage} h{h + 1}: fit {len(fit)} origins', flush=True)
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
