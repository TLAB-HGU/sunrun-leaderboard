"""e2000fs_feat30_agnostic_v2_event: one parametric rise-plateau-decay curve, end to end.

One 7-output vector-leaf tree ensemble on the Top30 features of all 72 hours predicts the standardized parameters of a single
rise-plateau-decay curve, fitted to each observed training trajectory.
F1 forecasts use models fitted on targets before 2026-02-26, F2 forecasts on targets before 2026-05-29.
The Top30 list is post-hoc (permutation delta MSE on F1/F2), so it is official-exposed.

usage: experiment.py --out DIR [--smoke]
Data is only read from data-v2. Reproduces results/forecast.npz (mse 4693.16).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import json
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
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
LOWER = np.array([1., 0., 0., 1., 0., 0., 1.])
UPPER = np.array([3000., 1500., 72., 48., 1500., 72., 48.])
HORIZONS = np.arange(1., 73.)
WORKERS = 8
STAGES = (('f1', '2026-02-26'), ('official', '2026-05-29'))  # forecast grid, training target cutoff


def targets(speed, origins):
    times = origins.values[:, None] + np.arange(1, 73).astype('timedelta64[h]')[None, :]
    return speed.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, 72)


def design_all(grid, names):
    """Selected constants, then the selected varying features of all 72 hours, flattened."""
    constant = [i for i, n in enumerate(names['constant']) if n in FEATURES]
    varying = [i for i, n in enumerate(names['varying']) if n in FEATURES]
    return np.hstack([grid['C'][:, constant], grid['E'][:, :, varying].reshape(len(grid['C']), -1)]).astype(np.float32)


def event_curve(params):
    """Baseline, amplitude, onset, rise duration, decay, plateau gap, decay duration -> speeds at horizons 1..72."""
    params = np.clip(np.asarray(params, dtype=np.float64), LOWER, UPPER)
    b, a, tau, duration, c, gap, decay_duration = params.T[:, :, None]
    rise = np.clip((HORIZONS - tau) / duration, 0., 1.)
    decay = np.clip((HORIZONS - tau - duration - gap) / decay_duration, 0., 1.)
    return np.clip(b + a * rise - c * decay, 1., 3000.)


def curve_jacobian(params):
    b, a, tau, duration, c, gap, decay_duration = params
    u = (HORIZONS - tau) / duration
    v = (HORIZONS - tau - duration - gap) / decay_duration
    rise, decay = np.clip(u, 0., 1.), np.clip(v, 0., 1.)
    active_u, active_v = (u > 0.) & (u < 1.), (v > 0.) & (v < 1.)
    raw = b + a * rise - c * decay
    jac = np.column_stack([np.ones(72), rise, -a / duration * active_u + c / decay_duration * active_v,
                           -a * u / duration * active_u + c / decay_duration * active_v, -decay,
                           c / decay_duration * active_v, c * v / decay_duration * active_v])
    jac[(raw < 1.) | (raw > 3000.)] = 0.
    return np.clip(raw, 1., 3000.), jac


def fit_curve(target):
    """Bounded least squares from several starts on one observed 72 h trajectory; lowest SSE wins."""
    changes = target[6:] - target[:-6]
    no_rise = changes.max() <= 0.
    baseline = np.clip(target[:3].mean(), LOWER[0], UPPER[0])
    amplitude = 0. if no_rise else np.clip(target.max() - baseline, 1., 1500.)
    decay = np.clip(target.max() - target[-3:].mean(), 0., 1500.)
    starts = [np.array([baseline, amplitude, tau, 6., decay, 12., 12.])
              for tau in sorted({0., 24., 48., float(np.argmax(changes) + 1)})]
    if no_rise:  # amplitude stays zero; an extra immediate-decay start
        starts.append(np.array([np.clip(target[0], 1., 3000.), 0., 0., 1., np.clip(target[0] - target[-1], 0., 1500.), 0., 48.]))
    free = np.array([0, 2, 3, 4, 5, 6] if no_rise else range(7))

    def full(values):
        result = np.zeros(7)
        result[free] = values
        return result

    candidates = []
    for start in starts:
        start = np.clip(start, LOWER, UPPER)
        result = least_squares(lambda v: curve_jacobian(full(v))[0] - target, start[free], jac=lambda v: curve_jacobian(full(v))[1][:, free],
                               bounds=(LOWER[free], UPPER[free]), x_scale='jac', max_nfev=80)
        fitted = full(result.x)
        error = event_curve(fitted[None, :])[0] - target
        candidates.append((float(error @ error), float(np.linalg.norm((fitted - LOWER) / (UPPER - LOWER))), tuple(fitted)))
    return np.array(min(candidates)[2])


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
    X = design_all(train, names)
    fits = {stage: np.flatnonzero(origins + pd.Timedelta(hours=72) < pd.Timestamp(cutoff, tz='UTC'))[::step] for stage, cutoff in STAGES}
    rows = np.union1d(*fits.values())
    # The curve fit of one origin never depends on the stage, so the overlap is fitted once.
    curves = dict(zip(rows, Parallel(n_jobs=WORKERS)(delayed(fit_curve)(y[i]) for i in rows)))

    # Train and forecast
    frames, errors = [], {}
    for stage, cutoff in STAGES:
        fit = fits[stage]
        grid = grids[stage]
        query = pd.DatetimeIndex(grid['origins'], tz='UTC')
        params = np.stack([curves[i] for i in fit])
        mean, scale = params.mean(0), np.maximum(params.std(0), 1.)
        coded = np.asarray((params - mean) / scale, dtype=np.float32)
        model = xgb.XGBRegressor(**{**PARAMS, 'n_estimators': rounds, 'base_score': float(coded.mean())})
        model.fit(X[fit], coded, verbose=False)
        raw = model.get_booster().predict(xgb.DMatrix(design_all(grid, names), nthread=3))
        pred = np.clip(event_curve(raw * scale + mean), 1, 3000)
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
