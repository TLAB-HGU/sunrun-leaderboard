"""CDF weighted squared error, fitted only to original labels of the actual fit.

Exact target-array matching resolves the existing GBM runner's period context to
its prior/full block. A calibration population cannot be silently substituted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtr

from . import calibration
from .table import load_grid


def fit_cdf(observed):
    values = np.asarray(observed, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 2:
        raise ValueError('Insufficient original observed targets for CDF')
    logs = np.log(values)
    sigma = float(logs.std())
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('CDF needs nonconstant positive observations')
    return dict(log_scale=float(logs.mean()), sigma=sigma, observations=len(values))


@dataclass(frozen=True)
class RelevanceCDF:
    """Pooling scope is the sole distinction between the two CDF-loss arms."""
    constant: tuple
    varying: tuple
    pooled: bool = False
    block: int = 24

    def prepare_period(self, table, period):
        grid, filled = load_grid(table, period, True, self.constant, self.varying)
        selected = table.loc[table[f'fit_{period}'].to_numpy()].sort_values(['origin', 'h'], kind='stable')
        raw = selected.y_observed.to_numpy(np.float64).reshape(filled.shape)
        times = pd.to_datetime(selected.target_time, utc=True).to_numpy().reshape(filled.shape)
        if not np.array_equal(filled[np.isfinite(raw)], raw[np.isfinite(raw)]):
            raise ValueError('Original and filled labels disagree')
        reference, _, _ = calibration.prior_split(grid['origins'])
        subsets = [('prior', reference), ('full', np.arange(len(filled))),
                   ('smoke_prior', reference[np.linspace(0, len(reference)-1, min(len(reference),128)).astype(int)]),
                   ('smoke_full', np.linspace(0, len(filled)-1, min(len(filled),192)).astype(int))]
        lookup = []
        for name, indices in subsets:
            labels, original, stamps = filled[indices], raw[indices], times[indices]
            pooled_original = None
            if self.pooled:
                _, unique = np.unique(stamps.reshape(-1), return_index=True)
                pooled_original = original.reshape(-1)[unique]
            for start in range(0, labels.shape[1], self.block):
                lookup.append(dict(subset=name, start=start, target=labels[:,start:start+self.block],
                                   original=original[:,start:start+self.block],
                                   pooled_original=pooled_original))
        return lookup

    def make_objective(self, target, context):
        matches = [entry for entry in context
                   if entry['target'].shape == target.shape and np.array_equal(entry['target'], target)]
        if len(matches) != 1:
            raise ValueError('Fitted block must have one exact original-label context')
        entry = matches[0]
        raw = entry['original']
        cdfs = ([fit_cdf(entry['pooled_original'])] * target.shape[1] if self.pooled
                else [fit_cdf(raw[:,h]) for h in range(target.shape[1])])
        raw_weights = np.column_stack([
            1. + 3. * ndtr((np.log(target[:,h])-cdf['log_scale'])/cdf['sigma'])
            for h, cdf in enumerate(cdfs)])
        weights = raw_weights / raw_weights.mean(axis=0)
        if not np.isfinite(weights).all() or not np.allclose(weights.mean(axis=0), 1.):
            raise ValueError('Invalid normalized relevance weights')
        print(json.dumps(dict(event='fit_cdf', subset=entry['subset'], block=entry['start']//self.block,
                              scope='pooled' if self.pooled else 'lead', lambda_=3, gamma=1,
                              cdfs=cdfs, ess=((weights.sum(0)**2)/(weights**2).sum(0)).tolist()),
                         allow_nan=False), flush=True)
        def objective(prediction, matrix):
            error = np.asarray(prediction).reshape(target.shape)-target
            return (weights*error).astype(np.float32), weights.astype(np.float32)
        return objective
