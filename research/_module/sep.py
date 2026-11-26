"""Centered Gaussian-smoothed SEP adaptation; train-only histogram parameters."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr

from . import calibration
from .objectives import SPECTRAL_SCALE
from .table import load_grid


def fit_histogram(observed, minimum=1000):
    """Fit the first sustained below-uniform crossing after the rightmost mode."""
    values = np.asarray(observed, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < minimum:
        raise ValueError('SEP needs sufficient positive original h1 observations')
    low, high = float(values.min()), float(values.max())
    spread = high - low
    if spread <= 0:
        raise ValueError('SEP needs positive target range')
    bin_size = 10. ** (np.floor(np.log10(high)) - 1)
    bins = int(spread / bin_size) + 1
    counts, edges = np.histogram(values, bins=bins)
    p = counts / counts.sum()
    mode = int(np.flatnonzero(counts == counts.max())[-1])
    crossings = [i for i in range(1, bins - 1)
                 if p[i] < 1 / bins and p[i-1] >= 1 / bins and p[i+1] < 1 / bins]
    eligible = [i for i in crossings if i > mode]
    if not eligible:
        raise ValueError('SEP histogram has no sustained internal crossing after rightmost mode')
    return {'gamma': float(edges[eligible[0]]), 'range': spread, 'minimum': low,
                'observations': int(values.size), 'bin_size': float(bin_size), 'bins': bins,
                'counts': counts.tolist(), 'edges': edges.tolist(), 'modal_bin': mode,
                'crossings': crossings, 'eligible_crossings': eligible}


def smooth_terms(z, lambda_, delta):
    """Uncentered dimensionless loss and its first two exact derivatives."""
    if not np.isfinite([lambda_, delta]).all() or lambda_ <= 0 or delta <= 0:
        raise ValueError('SEP lambda and delta must be finite and positive')
    z = np.asarray(z, dtype=np.float64)
    if not np.isfinite(z).all():
        raise ValueError('SEP residuals must be finite')
    b = 1. / lambda_
    with np.errstate(over='ignore', invalid='ignore'):
        u = np.exp(-z + delta**2 / 2 + log_ndtr(delta - z / delta))
        v = np.exp(b*z + b*b*delta**2 / 2 + log_ndtr(z / delta + b*delta))
        density = np.exp(-.5*(z/delta)**2) / np.sqrt(2*np.pi)
        result = (u+v-1, -u+b*v, u+b*b*v+(1+b)*density/delta)
    if not all(np.isfinite(value).all() for value in result):
        raise ValueError('Nonfinite SEP loss or derivatives')
    return result


def center_root(lambda_, delta):
    return float(brentq(lambda z: float(smooth_terms(z, lambda_, delta)[1]),
                        -1., 1., xtol=1e-15))


def point_terms(error, target, parameters, lambda_, delta, q):
    """Physical unscaled point loss, gradient and diagonal Hessian."""
    error = np.asarray(error, dtype=np.float64)
    if not np.isfinite(error).all() or not np.isfinite(target).all():
        raise ValueError('SEP targets and predictions must be finite')
    spread = parameters['range']
    if not np.isfinite(spread) or spread <= 0:
        raise ValueError('SEP range must be finite and positive')
    loss, gradient, hessian = .5*error**2, error.copy(), np.ones_like(error)
    extreme = np.asarray(target) >= parameters['gamma']
    f, g, h = smooth_terms(error[extreme]/spread + q, lambda_, delta)
    loss[extreme] = .5*spread**2*(f-smooth_terms(q, lambda_, delta)[0])
    gradient[extreme], hessian[extreme] = .5*spread*g, .5*h
    return loss, gradient, hessian


@dataclass(frozen=True)
class SmoothSEP:
    constant: tuple
    varying: tuple
    lambda_: float = 1.
    delta: float = .01
    block: int = 24
    spectral_coefficient: float = .1

    def prepare_period(self, table, period):
        grid, filled = load_grid(table, period, True, self.constant, self.varying)
        selected = table.loc[table[f'fit_{period}'].to_numpy()].sort_values(['origin', 'h'], kind='stable')
        raw = selected.y_observed.to_numpy(np.float64).reshape(filled.shape)
        observed = np.isfinite(raw)
        if not np.array_equal(filled[observed], raw[observed]):
            raise ValueError('Original and filled SEP labels disagree')
        reference, _, _ = calibration.prior_split(grid['origins'])
        subsets = [('prior', reference), ('full', np.arange(len(filled))),
                   ('smoke_prior', reference[np.linspace(0, len(reference)-1, min(len(reference), 128)).astype(int)]),
                   ('smoke_full', np.linspace(0, len(filled)-1, min(len(filled), 192)).astype(int))]
        lookup = []
        for name, indices in subsets:
            labels = filled[indices]
            original_h1 = raw[indices, 0]
            for start in range(0, labels.shape[1], self.block):
                lookup.append({'period': period, 'subset': name, 'start': start,
                                   'target': labels[:, start:start+self.block], 'original_h1': original_h1})
        return lookup

    def make_objective(self, target, context):
        target = np.asarray(target, dtype=np.float64)
        matches = [entry for entry in context if entry['target'].shape == target.shape
                   and np.array_equal(entry['target'], target)]
        if len(matches) != 1:
            raise ValueError('Fitted SEP block must have one exact original-label context')
        entry = matches[0]
        smoke = entry['subset'].startswith('smoke_')
        parameters = fit_histogram(entry['original_h1'], minimum=2 if smoke else 1000)
        q = center_root(self.lambda_, self.delta)
        # XGBoost stores this scalar base prediction as float32.
        initial = float(np.float32(target.mean()))
        _, _, initial_h = point_terms(initial-target, target, parameters, self.lambda_, self.delta, q)
        scale = 1. / float(initial_h.mean())
        coefficient = self.spectral_coefficient
        if not np.isfinite([scale, coefficient]).all() or scale <= 0 or coefficient < 0:
            raise ValueError('Invalid SEP normalization or spectral coefficient')
        record = {'event': 'fit_sep', 'period': entry['period'], 'subset': entry['subset'],
                      'block': entry['start']//self.block, 'smoke': smoke, 'minimum_observations': 2 if smoke else 1000,
                      'parameters': parameters, 'lambda_': self.lambda_, 'delta': self.delta, 'q': q, 'scale': scale,
                      'initial_prediction': initial, 'spectral_coefficient': coefficient}
        print(json.dumps(record, allow_nan=False), flush=True)

        def objective(prediction, matrix):
            error = np.asarray(prediction, dtype=np.float64).reshape(target.shape)-target
            _, gradient, hessian = point_terms(error, target, parameters, self.lambda_, self.delta, q)
            z = np.fft.fft(error, axis=1, norm='ortho')
            rho = 1 / np.sqrt(1 + abs(z)**2 / SPECTRAL_SCALE**2)
            gradient = scale*gradient + coefficient*np.fft.ifft(z*rho, axis=1, norm='ortho').real
            hessian = scale*hessian + coefficient
            with np.errstate(over='ignore'):
                gradient, hessian = gradient.astype(np.float32), hessian.astype(np.float32)
            if not np.isfinite(gradient).all() or not np.isfinite(hessian).all() or not (hessian > 0).all():
                raise ValueError('SEP requires finite gradients and positive finite Hessians')
            return gradient, hessian
        return objective
