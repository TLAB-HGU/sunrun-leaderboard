"""Five controlled CPU loss arms on a pre-F1 temporal holdout.

The 237-column design, filled input snapshot, tree settings and fixed 300 rounds
are shared. SERA/plain L2 use km/s. Distribution arms use y/500 and initial Fisher
coordinate scaling, unweighted NLL and max_delta_step=.5 for solver stability.
No validation target enters training, initialization, stopping or model selection.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / 'research/rerun_df_ffill_20261103/input/df_ffill_trainy.parquet'
TRUTH = ROOT / 'data-v2/input/df.parquet'
START = pd.Timestamp('2025-07-01', tz='UTC')
END = pd.Timestamp('2025-10-01', tz='UTC')
CUTOFF = pd.Timestamp('2025-06-29', tz='UTC')
H, BLOCK, TARGET_SCALE = 72, 24, 500.0
SIGMA_FLOOR = 0.02
ARMS = ('sera_l2', 'plain_l2', 'gaussian', 'lognormal', 'mixture')
PARAMS = {'max_depth': 3, 'learning_rate': .03, 'subsample': .8, 'colsample_bytree': .7,
              'min_child_weight': 20, 'reg_lambda': 5, 'objective': 'reg:squarederror',
              'tree_method': 'hist', 'device': 'cpu', 'multi_strategy': 'multi_output_tree',
              'verbosity': 0, 'disable_default_eval_metric': True}
CONSTANT = ['v_lag0', 'ace_speed_kms_last', 'ace_density_cm3_last', 'ace_density_cm3_m6',
            'ace_temperature_k_last', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6',
            'ace_by_gsm_nt_m24', 'ace_epam_p_channel3_last', 'ace_sis_gt10_flux_last',
            'suvi_t035_top2_area', 'suvi_t035_meridian15_equatorial_area',
            'suvi_t035_meridian30_equatorial_area', 'suvi_t035_equatorial_area',
            'suvi_t055_meridian15_area', 'suvi_t055_meridian30_equatorial_area',
            'suvi_t055_equatorial_area', 'img_age_h', 'enlil_n_pending', 'enlil_kp']
VARYING = ['rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed',
           'strip_v450_logmed', 'strip_v550_dark', 'strip_v650_dark', 'strip_v750_logmed']

# Embedded reference loss implementation follows. This file is standalone.
PARAMETER_COUNTS = {"gaussian": 2, "lognormal": 2, "mixture": 5}
LOG_2PI = np.log(2.0 * np.pi)


def _parameters(theta, family):
    if family not in PARAMETER_COUNTS:
        raise ValueError(f"Unknown family: {family}")
    values = np.asarray(theta, dtype=np.float64)
    scalar = values.ndim == 1
    if scalar:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != PARAMETER_COUNTS[family]:
        raise ValueError(f"theta must have shape (n, {PARAMETER_COUNTS[family]})")
    if not np.isfinite(values).all():
        raise ValueError("Distribution parameters must be finite")
    return values, scalar


def _scales(rho, sigma_floor):
    if not np.isfinite(sigma_floor) or sigma_floor <= 0:
        raise ValueError("sigma_floor must be finite and positive")
    lower = np.log(sigma_floor)
    effective = np.maximum(rho, lower)
    # The floor is part of the loss, so its derivative is zero below the floor.
    # At the kink we choose the zero subgradient. A curvature floor keeps the
    # learner's numerical denominator positive, but does not change the gradient.
    active = rho > lower
    return effective, np.exp(-2.0 * effective), active


def nll_and_grad(theta, y, family, *, weights=None, mask=None,
                 sigma_floor=1e-4, hessian_floor=1e-8):
    """Return (per-cell NLL, gradient, positive approximate diagonal Hessian).

    gaussian/lognormal: theta=(mu, log_sigma).
    mixture: theta=(logit_pi1, mu1, mu2, log_sigma1, log_sigma2), where
    pi1=sigmoid(logit_pi1). Component means need not be ordered.

    y, weights, mask accept scalars or vectors of length n. Nonfinite labels and
    explicitly masked cells contribute exactly zero. Nonpositive *active*
    lognormal labels are errors. Zero-weight rows are excluded. Inactive rows may
    contain invalid labels, but all parameters must remain finite. A 1-D theta
    returns a scalar loss and 1-D gradient/Hessian.
    """
    theta, scalar = _parameters(theta, family)
    if not np.isfinite(hessian_floor) or hessian_floor <= 0:
        raise ValueError("hessian_floor must be finite and positive")
    n = len(theta)
    labels = np.broadcast_to(np.asarray(y, dtype=np.float64), (n,))
    weight = np.ones(n) if weights is None else np.broadcast_to(
        np.asarray(weights, dtype=np.float64), (n,))
    if not np.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("weights must be finite and nonnegative")
    valid = np.isfinite(labels) & (weight > 0)
    if mask is not None:
        valid &= np.broadcast_to(np.asarray(mask, dtype=bool), (n,))
    if family != "gaussian" and (labels[valid] <= 0).any():
        raise ValueError("Lognormal targets must be positive")
    safe = np.where(valid, labels, 1.0)
    z = safe if family == "gaussian" else np.log(safe)
    grad, curvature = np.zeros_like(theta), np.zeros_like(theta)
    if family in ("gaussian", "lognormal"):
        rho, inv_var, active = _scales(theta[:, 1], sigma_floor)
        residual = z - theta[:, 0]
        squared = residual**2 * inv_var
        loss = 0.5 * (LOG_2PI + squared) + rho
        grad[:, 0] = -residual * inv_var
        grad[:, 1] = (1.0 - squared) * active
        curvature[:, 0] = inv_var
        curvature[:, 1] = 2.0 * active
    else:
        eta = theta[:, 0]
        log_pi = np.column_stack((-np.logaddexp(0., -eta),
                                  -np.logaddexp(0., eta)))
        rho, inv_var, active = _scales(theta[:, 3:5], sigma_floor)
        residual = z[:, None] - theta[:, 1:3]
        squared = residual**2 * inv_var
        log_joint = log_pi - 0.5 * (LOG_2PI + squared) - rho
        log_density = np.logaddexp(log_joint[:, 0], log_joint[:, 1])
        responsibility = np.exp(log_joint - log_density[:, None])
        pi = np.exp(log_pi)
        loss = -log_density
        grad[:, 0] = pi[:, 0] - responsibility[:, 0]
        grad[:, 1:3] = -responsibility * residual * inv_var
        grad[:, 3:5] = responsibility * (1.0 - squared) * active
        curvature[:, 0] = pi[:, 0] * pi[:, 1]
        curvature[:, 1:3] = responsibility * inv_var
        curvature[:, 3:5] = 2.0 * responsibility * active
    if family != "gaussian":
        loss += z
    loss = np.where(valid, loss * weight, 0.)
    grad = np.where(valid[:, None], grad * weight[:, None], 0.)
    curvature = np.where(valid[:, None],
                         np.maximum(curvature, hessian_floor) * weight[:, None], 0.)
    if not all(np.isfinite(value).all() for value in (loss, grad, curvature)):
        raise FloatingPointError("Nonfinite distribution loss or derivative")
    if scalar:
        return loss[0], grad[0], curvature[0]
    return loss, grad, curvature


def predict_mean(theta, family, *, sigma_floor=1e-4):
    """Return the arithmetic conditional mean, without positivity clipping.

    Gaussian means may be negative. Any forecast-domain clipping is a separate
    readout policy and must be reported by the experiment runner.
    """
    theta, scalar = _parameters(theta, family)
    if family == "gaussian":
        mean = theta[:, 0].copy()
    elif family == "lognormal":
        rho, _, _ = _scales(theta[:, 1], sigma_floor)
        mean = np.exp(theta[:, 0] + 0.5 * np.exp(2.0 * rho))
    else:
        rho, _, _ = _scales(theta[:, 3:5], sigma_floor)
        log_pi = np.column_stack((-np.logaddexp(0., -theta[:, 0]),
                                  -np.logaddexp(0., theta[:, 0])))
        component_log_mean = theta[:, 1:3] + 0.5 * np.exp(2.0 * rho)
        mean = np.exp(np.logaddexp(log_pi[:, 0] + component_log_mean[:, 0],
                                  log_pi[:, 1] + component_log_mean[:, 1]))
    if not np.isfinite(mean).all():
        raise FloatingPointError("Nonfinite distribution mean")
    return mean[0] if scalar else mean

def feature_names(block):
    return CONSTANT + [f'{f}@h{h}' for h in range(block*BLOCK+1, (block+1)*BLOCK+1) for f in VARYING]


def validation_origins():
    return pd.date_range((START-pd.Timedelta(hours=1)).ceil('6h'),
                         (END-pd.Timedelta(hours=73)).floor('6h'), freq='6h')


def check_split(train_origins, valid_origins):
    if not len(train_origins) or not len(valid_origins):
        raise ValueError('Empty training/validation origins')
    if not (train_origins + pd.Timedelta(hours=144) <= CUTOFF).all():
        raise ValueError('Training maturity boundary violated')
    if not ((valid_origins + pd.Timedelta(hours=1) >= START) &
            (valid_origins + pd.Timedelta(hours=72) < END)).all():
        raise ValueError('Validation targets leave the registered window')
    if not (valid_origins.hour % 6 == 0).all() or train_origins.intersection(valid_origins).size:
        raise ValueError('Invalid or overlapping origin grids')


def load_block(data, truth_path, block, smoke=False):
    """Project one block; holdout truth always comes from unfilled observations."""
    train_end = CUTOFF-pd.Timedelta(hours=144)
    filters = [[('set', '==', 'train'), ('origin', '<=', train_end)],
               [('set', '==', 'train'), ('origin', '>=', START-pd.Timedelta(hours=1)),
                ('origin', '<=', END-pd.Timedelta(hours=73))]]
    constants = pd.read_parquet(data, columns=['origin', *CONSTANT],
                                filters=[f+[('h', '==', 1)] for f in filters], use_threads=False)
    constants = constants.sort_values('origin').set_index('origin')
    if not constants.index.is_unique:
        raise ValueError('Duplicate input origin')
    hours = np.arange(block*BLOCK+1, (block+1)*BLOCK+1)
    rows = pd.read_parquet(data, columns=['origin', 'h', 'y', *VARYING],
                           filters=[f+[('h', '>=', int(hours[0])), ('h', '<=', int(hours[-1]))] for f in filters],
                           use_threads=False).sort_values(['origin', 'h'])
    origins = pd.DatetimeIndex(rows.origin.iloc[::BLOCK])
    if len(rows) != len(origins)*BLOCK or not np.array_equal(rows.h.to_numpy().reshape(-1, BLOCK),
                                                          np.tile(hours, (len(origins), 1))):
        raise ValueError('Incomplete input block')
    origin_grid = pd.DatetimeIndex(rows.origin).asi8.reshape(-1, BLOCK)
    if not origins.is_unique or not (origin_grid == origin_grid[:, :1]).all():
        raise ValueError('Inconsistent per-row block origins')
    c = constants.reindex(origins).to_numpy(np.float32)
    varying = rows[VARYING].to_numpy(np.float32).reshape(len(origins), -1)
    x = np.column_stack((c, varying)).astype(np.float32)
    y = rows.y.to_numpy(np.float64).reshape(len(origins), BLOCK)
    train = np.where(origins <= train_end)[0]
    valid = np.where(origins.isin(validation_origins()))[0]
    if smoke:
        train = train[np.linspace(0, len(train)-1, min(512, len(train)), dtype=int)]
        valid = valid[:24]
    ot, ov = origins[train], origins[valid]
    check_split(ot, ov)
    if not smoke and not ov.equals(validation_origins()):
        raise ValueError('Input lacks the complete registered validation grid')
    observations = pd.read_parquet(truth_path, columns=['target_time', 'y'],
                                    filters=[('set', '==', 'train'), ('h', '==', 1),
                                             ('target_time', '>=', START), ('target_time', '<', END)],
                                    use_threads=False).set_index('target_time').y
    if not observations.index.is_unique:
        raise ValueError('Duplicate raw observation timestamp')
    targets = ov.asi8[:, None] + pd.to_timedelta(hours, unit='h').asi8[None, :]
    truth = observations.reindex(pd.to_datetime(targets.ravel(), utc=True)).to_numpy().reshape(len(ov), BLOCK)
    return x[train], y[train], x[valid], truth, ot, ov


def sera_weights(y):
    observed = np.isfinite(y)
    low, high = np.nanmin(y, axis=0), np.nanmax(y, axis=0)
    if (high <= low).any():
        raise ValueError('SERA needs varying training labels')
    raw = np.where(observed, np.maximum((y-low)/(high-low), .001), np.nan)
    return np.where(observed, raw/np.nanmedian(raw, axis=0), 0.)


def initialise(y, family):
    """Training-only initial distribution and immutable Fisher coordinate scales."""
    from sklearn.mixture import GaussianMixture
    p = PARAMETER_COUNTS[family]
    offset = np.zeros((y.shape[1], p))
    coefficient = np.ones_like(offset)
    for h in range(y.shape[1]):
        v = y[np.isfinite(y[:, h]), h]/TARGET_SCALE
        z = v if family == 'gaussian' else np.log(v)
        if family == 'mixture':
            model = GaussianMixture(2, n_init=3, max_iter=300, random_state=42,
                                    reg_covar=SIGMA_FLOOR**2).fit(z[:, None])
            if not model.converged_:
                raise RuntimeError('Training-only mixture EM did not converge')
            order = np.argsort(model.means_[:, 0])
            pi = model.weights_[order]
            mu = model.means_[order, 0]
            sigma = np.sqrt(model.covariances_[order, 0, 0])
            offset[h] = [np.log(pi[0]/pi[1]), *mu, *np.log(sigma)]
            coefficient[h] = [1/np.sqrt(pi.prod()), sigma[0]/np.sqrt(pi[0]),
                               sigma[1]/np.sqrt(pi[1]), 1/np.sqrt(2*pi[0]), 1/np.sqrt(2*pi[1])]
        else:
            sigma = max(float(z.std()), SIGMA_FLOOR*1.01)
            offset[h] = [z.mean(), np.log(sigma)]
            coefficient[h] = [sigma, 1/np.sqrt(2)]
    return offset, coefficient


def theta_from_margin(margin, offset, coefficient):
    theta = np.asarray(margin, dtype=float).reshape(-1, *offset.shape)*coefficient+offset
    scale_columns = [1] if offset.shape[1] == 2 else [3, 4]
    if not np.isfinite(theta).all() or np.any(theta[:, :, scale_columns] > np.log(2.0)):
        raise FloatingPointError('Nonfinite parameters or sigma ceiling exceeded; optimisation failure')
    return theta


def distribution_cdf(theta, values, family):
    values = np.asarray(values, dtype=float)
    z = values if family == 'gaussian' else np.log(np.maximum(values, np.finfo(float).tiny))
    if family == 'mixture':
        pi = np.exp(-np.logaddexp(0., -theta[..., 0]))
        sig = np.maximum(np.exp(theta[..., 3:5]), SIGMA_FLOOR)
        return pi*ndtr((z-theta[..., 1])/sig[..., 0])+(1-pi)*ndtr((z-theta[..., 2])/sig[..., 1])
    sigma = np.maximum(np.exp(theta[..., 1]), SIGMA_FLOOR)
    return ndtr((z-theta[..., 0])/sigma)


def distribution_metrics(theta, y, family):
    loss = nll_and_grad(theta, y/TARGET_SCALE, family, sigma_floor=SIGMA_FLOOR)[0]+np.log(TARGET_SCALE)
    pit = distribution_cdf(theta, y/TARGET_SCALE, family)
    return {'nll': float(loss.mean()), 'coverage50': float(np.mean((pit >= .25) & (pit <= .75))),
            'coverage90': float(np.mean((pit >= .05) & (pit <= .95))),
            'coverage95': float(np.mean((pit >= .025) & (pit <= .975))),
            'pit_mean': float(pit.mean()), 'pit_hist': np.histogram(pit, bins=np.linspace(0, 1, 11))[0].tolist()}


def score_arrays(pred, truth, theta=None, family=None):
    """Same lead-first TIMSE and W algebra as scorer/competition.py."""
    if pred.shape != truth.shape or pred.shape[1] != H or not np.isfinite(pred).all() or (pred <= 0).any():
        raise ValueError('Predictions must be positive finite n x 72')
    rows = []
    for h in range(H):
        observed = np.isfinite(truth[:, h])
        y, e = truth[observed, h], pred[observed, h]-truth[observed, h]
        if len(y) < 2 or np.ptp(y) <= 0:
            raise ValueError('Insufficient observed variation to score every lead')
        q = np.quantile(y, np.arange(101)/100, method='linear')
        order = np.argsort(y, kind='stable')
        positions = np.searchsorted(y[order], q[:-1], side='left')
        suffix = np.cumsum((e[order]**2)[::-1])[::-1]
        timse = np.dot(np.diff(q), suffix[positions]/(len(y)-positions))/np.ptp(y)
        rmse = np.sqrt(np.mean(e**2))
        row = {'lead': h+1, 'W': float(np.sqrt(rmse*np.sqrt(timse))), 'rmse': float(rmse),
                   'timse': float(timse), 'mae': float(np.mean(np.abs(e))), 'bias': float(e.mean()), 'n': len(y)}
        if theta is not None:
            row['distribution'] = distribution_metrics(theta[observed, h], y, family)
        rows.append(row)
    metrics = {key: float(np.mean([r[key] for r in rows])) for key in ('W', 'rmse', 'timse', 'mae', 'bias')}
    observed = np.isfinite(truth)
    errors = pred-truth
    metrics.update(n_pairs=int(observed.sum()), n_origins=len(pred), by_lead=rows,
                   pooled_rmse=float(np.sqrt(np.mean(errors[observed]**2))),
                   pooled_mae=float(np.mean(np.abs(errors[observed]))), pooled_bias=float(errors[observed].mean()))
    for threshold in (550, 700):
        high = observed & (truth >= threshold)
        metrics[f'tail_{threshold}'] = {'n': int(high.sum()),
            'rmse': float(np.sqrt(np.mean(errors[high]**2))) if high.any() else None,
            'bias': float(errors[high].mean()) if high.any() else None}
    if theta is not None:
        metrics['distribution'] = distribution_metrics(theta[observed], truth[observed], family)
        scale_columns = [1] if theta.shape[-1] == 2 else [3, 4]
        sigma = np.maximum(np.exp(theta[:, :, scale_columns]), SIGMA_FLOOR)
        metrics['distribution'].update(sigma_min=float(sigma.min()), sigma_max=float(sigma.max()),
                                        sigma_floor_fraction=float(np.mean(sigma <= SIGMA_FLOOR)))
        if family == 'mixture':
            pi = np.exp(-np.logaddexp(0., -theta[..., 0]))
            metrics['distribution'].update(pi1_min=float(pi.min()), pi1_max=float(pi.max()),
                                            pi1_mean=float(pi.mean()))
    metrics['horizon_bands'] = {f'{a+1}-{b}': {key: float(np.mean([r[key] for r in rows[a:b]]))
                                              for key in ('W', 'rmse', 'mae', 'bias')}
                                for a, b in ((0, 24), (24, 48), (48, 72))}
    return metrics


def file_identity(path):
    path = Path(path)
    stat = path.stat()
    return {'path': str(path.resolve()), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def fit_block(args, block, identity):
    prefix = args.out/f'b{block}'
    meta_path = prefix.with_suffix('.json')
    if meta_path.exists() and prefix.with_suffix('.npz').exists():
        old = json.loads(meta_path.read_text())
        if old['identity'] != identity or old['features'] != feature_names(block):
            raise ValueError('Checkpoint identity mismatch')
        print(json.dumps({'resumed': block}), flush=True)
        return
    started = time.perf_counter()
    x, y, xq, truth, ot, ov = load_block(args.data, args.truth, block, args.smoke)
    observed = np.isfinite(y)
    keep = observed.any(axis=1)
    x, y, observed, ot = x[keep], y[keep], observed[keep], ot[keep]
    if not observed.any(axis=0).all():
        raise ValueError('Missing complete training output')
    params = dict(PARAMS, nthread=args.threads, seed=args.seed)
    extras = {}
    if args.arm in ('sera_l2', 'plain_l2'):
        weights = sera_weights(y) if args.arm == 'sera_l2' else observed.astype(float)
        labels = np.where(observed, y, 0.)
        matrix = xgb.DMatrix(x, label=labels.astype(np.float32), nthread=args.threads)
        params['base_score'] = float(np.nanmean(y))
        def objective(pred, _):
            return ((pred-labels)*weights).astype(np.float32), weights.astype(np.float32)
        def metric(pred, _):
            return 'train_rmse', float(np.sqrt(np.mean((pred[observed]-y[observed])**2)))
    else:
        offset, coefficient = initialise(y, args.arm)
        p = PARAMETER_COUNTS[args.arm]
        labels = np.zeros((len(x), BLOCK*p), dtype=np.float32)
        matrix = xgb.DMatrix(x, label=labels, nthread=args.threads)
        params.update(base_score=0., max_delta_step=.5)
        def objective(pred, _):
            theta = theta_from_margin(pred, offset, coefficient)
            _, grad, hess = nll_and_grad(theta.reshape(-1, p), (y/TARGET_SCALE).ravel(), args.arm,
                                        sigma_floor=SIGMA_FLOOR)
            grad = grad.reshape(len(x), BLOCK, p)*coefficient
            hess = hess.reshape(len(x), BLOCK, p)*coefficient**2
            return grad.reshape(labels.shape).astype(np.float32), hess.reshape(labels.shape).astype(np.float32)
        def metric(pred, _):
            theta = theta_from_margin(pred, offset, coefficient)
            loss = nll_and_grad(theta.reshape(-1, p), (y/TARGET_SCALE).ravel(), args.arm,
                                sigma_floor=SIGMA_FLOOR)[0].reshape(y.shape)
            return 'train_nll_kms', float(loss[observed].mean()+np.log(TARGET_SCALE))
        extras.update(offset=offset, coefficient=coefficient)
    history = {}
    model = xgb.train(params, matrix, num_boost_round=args.rounds, obj=objective,
                      evals=[(matrix, 'train')], custom_metric=metric,
                      evals_result=history, verbose_eval=50)
    model.save_model(prefix.with_suffix('.ubj'))
    raw = model.inplace_predict(xq)
    if args.arm in ('sera_l2', 'plain_l2'):
        pred = raw
    else:
        theta = theta_from_margin(raw, offset, coefficient)
        pred = predict_mean(theta.reshape(-1, p), args.arm, sigma_floor=SIGMA_FLOOR).reshape(len(xq), BLOCK)*TARGET_SCALE
        extras.update(theta=theta, cdf=distribution_cdf(theta, truth/TARGET_SCALE, args.arm))
    if not np.isfinite(pred).all() or (pred <= 0).any():
        raise FloatingPointError('Invalid point prediction; no replacement/clipping allowed')
    np.savez_compressed(prefix.with_suffix('.npz'), pred=pred, truth=truth, origins=ov.asi8, **extras)
    write_json(prefix.with_suffix('.loss.json'), history)
    report = {'identity': identity, 'block': block, 'features': feature_names(block), 'feature_count': x.shape[1],
              'train_origins': len(x), 'valid_origins': len(xq), 'observed_training_labels': int(observed.sum()),
              'train_origin_start': str(ot.min()), 'train_origin_end': str(ot.max()),
              'max_train_target_time': str(ot.max()+pd.Timedelta(hours=(block+1)*BLOCK)),
              'validation_origin_start': str(ov.min()), 'validation_origin_end': str(ov.max()),
              'params': params, 'seconds': time.perf_counter()-started}
    write_json(meta_path, report)
    print(json.dumps({'completed_block': block, 'seconds': report['seconds']}), flush=True)


def assemble(args, identity):
    parts = []
    for block in range(3):
        report = json.loads((args.out/f'b{block}.json').read_text())
        if report['identity'] != identity:
            raise ValueError('Block run identities disagree')
        with np.load(args.out/f'b{block}.npz') as z:
            parts.append({k: z[k].copy() for k in z.files})
    if not all(np.array_equal(p['origins'], parts[0]['origins']) for p in parts):
        raise ValueError('Block origin grids disagree')
    pred, truth = (np.concatenate([p[key] for p in parts], axis=1) for key in ('pred', 'truth'))
    theta = np.concatenate([p['theta'] for p in parts], axis=1) if args.arm in PARAMETER_COUNTS else None
    metrics = score_arrays(pred, truth, theta, args.arm if theta is not None else None)
    origins = pd.to_datetime(parts[0]['origins'], utc=True)
    frame = pd.DataFrame({'origin': np.repeat(origins, H), 'h': np.tile(np.arange(1, H+1), len(origins)),
                          'pred_kms': pred.ravel(), 'y': truth.ravel()})
    frame.to_parquet(args.out/'predictions.parquet', index=False)
    write_json(args.out/'metrics.json', {'arm': args.arm, 'seed': args.seed, 'identity': identity, **metrics})
    write_json(args.out/'complete.json', {'status': 'complete', 'exit_code': 0, 'rows': len(frame),
                                         'scored_pairs': metrics['n_pairs'], 'identity': identity})
    print(json.dumps({k: metrics[k] for k in ('W', 'rmse', 'mae', 'bias', 'n_pairs')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rounds', type=int, default=300)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--truth', type=Path, default=TRUTH)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--score', action='store_true')
    parser.add_argument('--source-commit', default=os.environ.get('SUNRUN_SOURCE_COMMIT'))
    args = parser.parse_args()
    if args.smoke:
        args.rounds = 2
    args.out = args.out or HERE/'results'/args.arm/f'seed{args.seed}'
    args.out.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=False)
    commit = revision.stdout.strip() if revision.returncode == 0 else args.source_commit
    if not commit:
        raise ValueError('A git checkout or explicit --source-commit is required for provenance')
    commit_origin = 'git_checkout' if revision.returncode == 0 else 'explicit_source_workspace_commit'
    identity = {'arm': args.arm, 'seed': args.seed, 'rounds': args.rounds, 'smoke': args.smoke,
                'data': file_identity(args.data), 'truth': file_identity(args.truth),
                'train_cutoff': str(CUTOFF), 'maturity_hours': 144, 'validation_start': str(START),
                'validation_end': str(END), 'source': file_identity(__file__),
                'target_scale': TARGET_SCALE, 'sigma_floor': SIGMA_FLOOR, 'sigma_ceiling': 2.0,
                'distribution_max_delta_step': .5,
                'mixture_init': {'n_init': 3, 'max_iter': 300, 'seed': 42, 'convergence_required': True},
                'git_commit': commit, 'source_commit_origin': commit_origin,
                'libraries': {'numpy': np.__version__, 'pandas': pd.__version__, 'xgboost': xgb.__version__, 'sklearn': sklearn.__version__}}
    path = args.out/'run.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError('Output directory already belongs to another configuration')
    source_copy = args.out/'source.py'
    source_text = Path(__file__).read_text()
    if source_copy.exists() and source_copy.read_text() != source_text:
        raise ValueError('Saved source differs from current runner')
    source_copy.write_text(source_text)
    write_json(path, identity)
    if not args.score:
        for block in range(3):
            fit_block(args, block, identity)
    assemble(args, identity)


if __name__ == '__main__':
    main()
