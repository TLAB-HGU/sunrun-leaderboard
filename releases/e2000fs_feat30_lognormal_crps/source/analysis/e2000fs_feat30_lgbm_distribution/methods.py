"""Continuous lognormal LightGBM adaptations; no forecast routing or mixtures.

CRPS training uses deterministic log-space Gauss--Legendre integration, split
at each observation, over [1, 6000] km/s. Smooth weights belong to integration
coordinates, never whole observations. The tail-calibration arm adds a batch
marginal CDF calibration penalty; this is not the published TMCB algorithm.
"""
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

METHODS = ('lognormal_nll', 'lognormal_crps', 'smooth_wcrps', 'tailcal_crps')
QUADRATURE_NODES = 64
LOWER, UPPER = 1., 6000.
THRESHOLD, TRANSITION = 550., 100.
SQRT2PI = np.sqrt(2. * np.pi)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def _targets(y):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if not len(y) or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError('Targets must be finite positive speeds')
    return y


def _parameters(mu, rho):
    mu, rho = np.asarray(mu, dtype=np.float64), np.asarray(rho, dtype=np.float64)
    if mu.shape != rho.shape or not np.isfinite(mu).all() or not np.isfinite(rho).all():
        raise ValueError('Invalid lognormal parameter arrays')
    # Fail closed rather than quietly clipping a distribution or its derivative.
    if (np.abs(mu) > 20).any() or (rho < -8).any() or (rho > 1.5).any():
        raise FloatingPointError('Distribution parameters left numerical safety bounds')
    return mu, np.exp(rho)


def _quadrature(y, nodes=QUADRATURE_NODES):
    """Integrate separately below/above y; quadrature is fixed during fitting."""
    y = _targets(y)
    if (y < LOWER).any() or (y > UPPER).any():
        raise ValueError('Training targets must be inside documented quadrature bounds')
    q, w = np.polynomial.legendre.leggauss(nodes)
    edges = [np.full_like(y, np.log(LOWER)), np.log(y), np.full_like(y, np.log(UPPER))]
    zz, ww, ii = [], [], []
    for k in range(2):
        half = (edges[k + 1] - edges[k])[:, None] / 2
        logz = (edges[k + 1] + edges[k])[:, None] / 2 + half * q
        z = np.exp(logz)
        zz.append(z)
        ww.append(half * w * z)
        ii.append(np.full_like(z, float(k)))
    return np.concatenate(zz, axis=1), np.concatenate(ww, axis=1), np.concatenate(ii, axis=1)


def _common_quadrature(nodes=QUADRATURE_NODES):
    q, w = np.polynomial.legendre.leggauss(nodes)
    lo, hi = np.log(LOWER), np.log(UPPER)
    z = np.exp((lo + hi) / 2 + (hi - lo) / 2 * q)
    return z, w * (hi - lo) / 2 * z


def _cdf_derivatives(z, mu, sigma):
    a = (np.log(z) - mu[:, None]) / sigma[:, None]
    density = np.exp(-.5 * a * a) / SQRT2PI
    return ndtr(a), -density / sigma[:, None], -a * density


class DistributionObjective:
    """Loss and SUM-loss gradients; Hessians are positive diagonal surrogates."""

    def __init__(self, y, method, nodes=QUADRATURE_NODES):
        self.y = _targets(y)
        if method not in METHODS:
            raise ValueError(f'Unknown method: {method}')
        self.method = method
        if method != 'lognormal_nll':
            self.z, self.weights, self.observed = _quadrature(self.y, nodes)
            if method == 'smooth_wcrps':
                self.weights *= 1. + ndtr((self.z - THRESHOLD) / TRANSITION)
            if method == 'tailcal_crps':
                self.cal_z, self.cal_weights = _common_quadrature(nodes)
                self.cal_weights *= ndtr((self.cal_z - THRESHOLD) / TRANSITION)
                self.cal_observed = (self.y[:, None] <= self.cal_z).astype(float)

    def evaluate(self, mu, rho, derivatives=True):
        mu, sigma = _parameters(mu, rho)
        if mu.shape != self.y.shape:
            raise ValueError('Parameters and targets must align')
        if self.method == 'lognormal_nll':
            r = mu - np.log(self.y)
            loss = np.log(self.y) + np.log(sigma) + .5 * (r / sigma)**2 + np.log(SQRT2PI)
            if not derivatives:
                return float(loss.mean())
            grad = np.column_stack((r / sigma**2, 1. - (r / sigma)**2))
            hess = np.column_stack((1. / sigma**2, np.full_like(mu, 2.)))
            return float(loss.mean()), grad, hess
        cdf, dmu, drho = _cdf_derivatives(self.z, mu, sigma)
        error = cdf - self.observed
        loss = float(np.mean(np.sum(self.weights * error**2, axis=1)))
        if derivatives:
            grad = np.column_stack([2. * np.sum(self.weights * error * d, axis=1) for d in (dmu, drho)])
            hess = np.column_stack([2. * np.sum(self.weights * d**2, axis=1) for d in (dmu, drho)])
        if self.method == 'tailcal_crps':
            cdf, dmu, drho = _cdf_derivatives(self.cal_z[None, :], mu, sigma)
            residual = np.mean(cdf - self.cal_observed, axis=0)
            loss += float(np.sum(self.cal_weights * residual**2))
            if derivatives:
                # n * mean-CDF penalty: its SUM-loss gradient has no extra 1/n.
                grad += np.column_stack([2. * np.sum(self.cal_weights * residual * d, axis=1) for d in (dmu, drho)])
                hess += np.column_stack([2. / len(mu) * np.sum(self.cal_weights * d**2, axis=1) for d in (dmu, drho)])
        if not np.isfinite(loss):
            raise FloatingPointError('Nonfinite distribution loss')
        if not derivatives:
            return loss
        return loss, grad, np.maximum(hess, 1e-8)


def lognormal_crps(y, mu, sigma):
    y = _targets(y)
    mu, sigma = np.asarray(mu), np.asarray(sigma)
    a = (np.log(y) - mu) / sigma
    mean = np.exp(mu + .5 * sigma**2)
    return y * (2. * ndtr(a) - 1.) - 2. * mean * (ndtr(a - sigma) + ndtr(sigma / np.sqrt(2.)) - 1.)


def distribution_scores(y, mu, sigma):
    y, mu, sigma = _targets(y), np.asarray(mu).reshape(-1), np.asarray(sigma).reshape(-1)
    if y.shape != mu.shape or mu.shape != sigma.shape or (sigma <= 0).any():
        raise ValueError('Distribution score arrays must match')
    _parameters(mu, np.log(sigma))
    pit = ndtr((np.log(y) - mu) / sigma)
    q = ndtri(.95)
    low, high = np.exp(mu - q * sigma), np.exp(mu + q * sigma)
    exceedance = ndtr((mu - np.log(THRESHOLD)) / sigma)
    # Bound intermediate quadrature memory for full 357840-point evaluation.
    weighted = 0.
    for start in range(0, len(y), 2048):
        sl = slice(start, start + 2048)
        weighted += len(y[sl]) * DistributionObjective(y[sl], 'smooth_wcrps').evaluate(mu[sl], np.log(sigma[sl]), False)
    return dict(crps=float(lognormal_crps(y, mu, sigma).mean()), smooth_wcrps=float(weighted / len(y)),
                nll=float(DistributionObjective(y, 'lognormal_nll').evaluate(mu, np.log(sigma), False)),
                coverage90=float(np.mean((y >= low) & (y <= high))), interval_width90=float(np.mean(high - low)),
                pit_mean=float(pit.mean()), pit_variance=float(pit.var()),
                p_ge550_brier=float(np.mean((exceedance - (y >= THRESHOLD))**2)))


def _native_predictions(booster, valid):
    captured = []

    def capture(pred, data):
        captured.append(np.asarray(pred).copy())
        return 'raw_mean', float(np.mean(pred)), False

    if valid:
        booster.eval_valid(feval=capture)
    else:
        booster.eval_train(feval=capture)
    return captured[0]


def predict_distribution(model, X):
    mu = model['mu_booster'].predict(X, num_threads=model['threads']) + model['mu_offset']
    rho = model['rho_booster'].predict(X, num_threads=model['threads']) + model['rho_offset']
    mu, sigma = _parameters(mu, rho)
    mean = np.exp(mu + .5 * sigma**2)
    if not np.isfinite(mean).all():
        raise FloatingPointError('Nonfinite lognormal mean')
    return dict(mu=mu, sigma=sigma, mean=mean)


def load_distribution(output_dir):
    directory = Path(output_dir)
    receipt = json.loads((directory / 'receipt.json').read_text())
    if receipt.get('complete') is not True or set(receipt.get('sha256', {})) != {'mu.txt', 'rho.txt', 'model.json', 'loss.csv'}:
        raise ValueError('Incomplete distribution checkpoint receipt')
    for name, expected in receipt['sha256'].items():
        if _digest(directory / name) != expected:
            raise ValueError(f'Distribution checkpoint hash mismatch: {name}')
    metadata = json.loads((directory / 'model.json').read_text())
    return dict(mu_booster=lgb.Booster(model_file=str(directory / 'mu.txt')),
                rho_booster=lgb.Booster(model_file=str(directory / 'rho.txt')),
                **{key: metadata[key] for key in ('mu_offset', 'rho_offset', 'method', 'threads')})


def fit_distribution(X_train, y_train, X_valid, y_valid, params, method, output_dir, *, smoke=False):
    """Train one horizon. Caller owns run identity and stage/cutoff validation."""
    y_train, y_valid = _targets(y_train), _targets(y_valid)
    X_train, X_valid = np.asarray(X_train), np.asarray(X_valid)
    if X_train.ndim != 2 or X_valid.ndim != 2 or X_train.shape[1] != X_valid.shape[1]:
        raise ValueError('Feature matrices must have the same columns')
    if len(X_train) != len(y_train) or len(X_valid) != len(y_valid) or np.isinf(X_train).any() or np.isinf(X_valid).any():
        raise ValueError('Invalid feature alignment or infinities')
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'receipt.json').exists():
        raise FileExistsError('Caller must verify run identity before loading completed horizon')
    config = dict(params)
    rounds = int(config.pop('n_estimators', 400))
    if smoke:
        rounds = min(rounds, 3)
    threads = int(config.pop('n_jobs', config.pop('num_threads', 2)))
    config.update(objective='none', metric='None', num_threads=threads, verbosity=-1,
                  boost_from_average=False, feature_pre_filter=False)
    init_mu = float(np.log(y_train).mean())
    init_rho = float(np.log(max(float(np.std(np.log(y_train))), .05)))
    train_objective = DistributionObjective(y_train, method)
    valid_objective = DistributionObjective(y_valid, method)
    boosters = []
    for offset in (init_mu, init_rho):
        train = lgb.Dataset(X_train, label=y_train, init_score=np.full(len(y_train), offset), free_raw_data=False)
        valid = lgb.Dataset(X_valid, label=y_valid, init_score=np.full(len(y_valid), offset), reference=train, free_raw_data=False)
        booster = lgb.Booster(config, train_set=train)
        booster.add_valid(valid, 'validation')
        boosters.append(booster)
    mu, rho = np.full(len(y_train), init_mu), np.full(len(y_train), init_rho)
    _, grad, hess = train_objective.evaluate(mu, rho)
    history = []
    for iteration in range(rounds):
        for coordinate, booster in enumerate(boosters):
            g, h = grad[:, coordinate].copy(), hess[:, coordinate].copy()
            booster.update(fobj=lambda pred, data, g=g, h=h: (g, h))
        mu, rho = [_native_predictions(booster, False) for booster in boosters]
        vmu, vrho = [_native_predictions(booster, True) for booster in boosters]
        train_loss, grad, hess = train_objective.evaluate(mu, rho)
        valid_loss = valid_objective.evaluate(vmu, vrho, False)
        train_mean = np.exp(mu + .5 * np.exp(2. * rho))
        valid_mean = np.exp(vmu + .5 * np.exp(2. * vrho))
        history.append(dict(round=iteration + 1, train_loss=train_loss, validation_loss=valid_loss,
                            train_mse=float(np.mean((train_mean - y_train)**2)),
                            validation_mse=float(np.mean((valid_mean - y_valid)**2))))
        if (iteration + 1) % 25 == 0:
            progress_path = directory / 'loss.csv.tmp'
            pd.DataFrame(history).to_csv(progress_path, index=False)
            progress_path.replace(directory / 'loss.csv')
    model = dict(mu_booster=boosters[0], rho_booster=boosters[1], mu_offset=init_mu,
                 rho_offset=init_rho, method=method, threads=threads)
    predicted = predict_distribution(model, X_valid)
    np.testing.assert_allclose(predicted['mu'], vmu, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.log(predicted['sigma']), vrho, rtol=1e-12, atol=1e-12)
    for name, booster in zip(('mu', 'rho'), boosters):
        temp = directory / f'{name}.txt.tmp'
        booster.save_model(str(temp))
        temp.replace(directory / f'{name}.txt')
    metadata = dict(method=method, mu_offset=init_mu, rho_offset=init_rho, threads=threads,
                    parameters=config, requested_rounds=rounds, actual_rounds=[b.current_iteration() for b in boosters],
                    target='positive absolute speed km/s; lognormal mean exp(mu+sigma^2/2)',
                    quadrature=dict(nodes_per_side=QUADRATURE_NODES, lower=LOWER, upper=UPPER, split_at_observation=True),
                    weight=dict(floor=1., amplitude=1., center=THRESHOLD, width=TRANSITION),
                    curvature='Fisher diagonal for NLL; positive Gauss-Newton diagonal for integrated scores',
                    tailcal='CRPS plus unit-weight squared batch marginal CDF residual integrated with smooth tail weight; not TMCB',
                    lightgbm_version=lgb.__version__)
    _write(directory / 'model.json', metadata)
    history_path = directory / 'loss.csv.tmp'
    pd.DataFrame(history).to_csv(history_path, index=False)
    history_path.replace(directory / 'loss.csv')
    restored = dict(model, mu_booster=lgb.Booster(model_file=str(directory / 'mu.txt')),
                    rho_booster=lgb.Booster(model_file=str(directory / 'rho.txt')))
    for key, value in predict_distribution(restored, X_valid).items():
        np.testing.assert_array_equal(value, predicted[key])
    _write(directory / 'receipt.json', dict(sha256={name: _digest(directory / name) for name in ('mu.txt', 'rho.txt', 'model.json', 'loss.csv')},
                                          complete=True, fit_rows=len(y_train), validation_rows=len(y_valid), reload_parity=True))
    return restored
