"""InHaR-inspired temporal hardness for A900; teachers never see future origins.

This is a rank-weighting adaptation, not a reproduction of InHaR RO. Cache
identity is checked against direct input arrays, without hashes or checksums.
"""
from __future__ import annotations

import fcntl
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import calibration, objectives
from .table import load_grid, subset


def temporal_splits(origins):
    origins = pd.DatetimeIndex(origins)
    if len(origins) < 2 or not origins.is_monotonic_increasing or origins.has_duplicates:
        raise ValueError('Teacher origins must be strictly chronological')
    eligible = np.flatnonzero(origins >= origins[0] + pd.Timedelta(days=365, hours=144))
    if len(eligible) < 3:
        raise ValueError('Insufficient post-warmup teacher origins')
    splits = []
    for query in np.array_split(eligible, 3):
        train = np.flatnonzero(origins + pd.Timedelta(hours=144) <= origins[query[0]])
        if len(train) < 2:
            raise ValueError('Insufficient purged teacher prefix')
        splits.append((train, query))
    return splits


def rank_hardness(original, prediction, *, min_eligible=1000):
    observed = np.isfinite(original)
    counts = observed.sum(0)
    if np.any(counts == 0):
        raise ValueError('Every lead needs original observations')
    gamma = np.where(observed, original ** 2, 0.).sum(0) / counts
    if not np.isfinite(gamma).all() or np.any(gamma <= 0):
        raise ValueError('Invalid observed target normalization')
    eligible = np.isfinite(prediction).all(1) & (observed.sum(1) >= 12)
    if eligible.sum() < min_eligible:
        raise ValueError(f'Hardness mechanism gate: {eligible.sum()} < {min_eligible} eligible origins')
    ih = -np.expm1(-np.abs(prediction - original) / gamma)
    score = np.where(observed[eligible], ih[eligible], 0.).sum(1) / observed[eligible].sum(1)
    q = np.zeros(len(original), dtype=np.float64)
    q[eligible] = rankdata(score, method='average') / (eligible.sum() + 1.)
    return q, eligible, gamma


def permute_blocks(q, eligible, origins, seed=20261113):
    """Permute chronological seven-day chunks including the final partial chunk."""
    indices = np.flatnonzero(eligible)
    if not len(indices):
        raise ValueError('Cannot permute an empty eligible population')
    stamps = pd.DatetimeIndex(origins)[indices]
    groups = np.asarray((stamps - stamps[0]) // pd.Timedelta(days=7))
    boundaries = np.r_[0, np.flatnonzero(np.diff(groups)) + 1, len(indices)]
    chunks = [q[indices[a:b]] for a, b in itertools.pairwise(boundaries)]
    permutation = np.random.default_rng(seed).permutation(len(chunks))
    shuffled = q.copy()
    shuffled[indices] = np.concatenate([chunks[i] for i in permutation])
    if not np.array_equal(np.sort(q[indices]), np.sort(shuffled[indices])):
        raise RuntimeError('Null control changed the rank multiset')
    return shuffled, permutation


def weighted_objective(target, modifier):
    modifier = np.asarray(modifier, dtype=np.float64)
    if modifier.shape != (len(target),) or not np.isfinite(modifier).all() or np.any(modifier <= 0):
        raise ValueError('Invalid origin modifier')
    _, weights = objectives.tail_weights(target)
    if not np.all(modifier == modifier[0]):
        weights = weights * modifier[:, None]
        weights /= weights.mean(0)

    def objective(prediction, matrix):
        error = np.asarray(prediction).reshape(target.shape) - target
        z = np.fft.fft(error, axis=1, norm='ortho')
        rho = 1 / np.sqrt(1 + abs(z) ** 2 / objectives.SPECTRAL_SCALE ** 2)
        gradient = error * weights + .1 * np.fft.ifft(z * rho, axis=1, norm='ortho').real
        return gradient.astype(np.float32), (weights + .1).astype(np.float32)

    return objective, weights


@dataclass(frozen=True)
class TemporalHardness:
    constant: tuple
    varying: tuple
    cache_root: str | Path
    mode: str = 'hard'
    parent_constant: tuple | None = None
    threads: int = 2

    def __post_init__(self):
        if self.mode not in ('hard', 'null'):
            raise ValueError('Hardness mode must be hard or null')
        if self.parent_constant is not None and tuple(self.constant[:len(self.parent_constant)]) != tuple(self.parent_constant):
            raise ValueError('Parent constants must be preserved as an ordered prefix')
        if self.threads < 1:
            raise ValueError('Teacher threads must be positive')

    def prepare_period(self, table, period):
        grid, filled = load_grid(table, period, True, self.constant, self.varying)
        selected = table.loc[table[f'fit_{period}'].to_numpy(), ['origin', 'h', 'y_observed']]
        raw = selected.sort_values(['origin', 'h'], kind='stable').y_observed.to_numpy(np.float64).reshape(filled.shape)
        if not np.array_equal(filled[np.isfinite(raw)], raw[np.isfinite(raw)]):
            raise ValueError('Original and filled labels disagree')
        reference, _, _ = calibration.prior_split(grid['origins'])
        populations = [('prior', reference), ('full', np.arange(len(filled))),
                       ('smoke_prior', reference[np.linspace(0, len(reference)-1, min(len(reference), 128)).astype(int)]),
                       ('smoke_full', np.linspace(0, len(filled)-1, min(len(filled), 192)).astype(int))]
        context = []
        for name, indices in populations:
            fit_grid = subset(grid, indices)
            labels, observed = filled[indices], raw[indices]
            for start in range(0, labels.shape[1], 24):
                context.append({'period': period, 'subset': name, 'start': start, 'grid': fit_grid,
                                    'target': labels[:, start:start+24], 'original': observed[:, start:start+24]})
        return context

    def _design(self, grid, start):
        varying = grid['E'][:, start:start+24].reshape(len(grid['C']), -1)
        if self.parent_constant is None:
            return np.column_stack([grid['C'], varying]).astype(np.float32)
        base = len(self.parent_constant)
        return np.column_stack([grid['C'][:, :base], varying, grid['C'][:, base:]]).astype(np.float32)

    def _teacher(self, entry, x, rounds):
        import xgboost as xgb

        from .gbm import PARAMS

        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise RuntimeError('CPU teacher requires CUDA_VISIBLE_DEVICES empty')
        origins, y = entry['grid']['origins'], entry['target']
        prediction = np.full(y.shape, np.nan)
        records = []
        for unit, (train, query) in enumerate(temporal_splits(origins), 1):
            params = dict(PARAMS, seed=42, nthread=self.threads, num_target=y.shape[1],
                          base_score=float(y[train].mean()), device='cpu')
            matrix = xgb.DMatrix(x[train], label=y[train].astype(np.float32), nthread=self.threads)
            model = xgb.train(params, matrix, num_boost_round=rounds,
                              obj=objectives.point4_spectral(y[train], .1), verbose_eval=False)
            prediction[query] = model.predict(xgb.DMatrix(x[query], nthread=self.threads))
            record = {'period': entry['period'], 'subset': entry['subset'], 'block': entry['start']//24,
                          'stage': 'teacher', 'unit': unit, 'units': 3, 'train_rows': len(train), 'query_rows': len(query),
                          'teacher_last_origin': str(origins[train[-1]]),
                          'teacher_last_target_bound': str(origins[train[-1]] + pd.Timedelta(hours=144)),
                          'first_prediction_origin': str(origins[query[0]]),
                          'last_prediction_origin': str(origins[query[-1]]),
                          'last_prediction_target_bound': str(origins[query[-1]] + pd.Timedelta(hours=144)),
                          'actual_fit_target_bound': str(origins[-1] + pd.Timedelta(hours=144))}
            print(json.dumps(record), flush=True)
            records.append(record)
        return prediction, records

    def _cached(self, entry):
        import xgboost as xgb

        from .gbm import PARAMS

        smoke = entry['subset'].startswith('smoke_')
        rounds, minimum = (2, 1) if smoke else (900, 1000)
        x = self._design(entry['grid'], entry['start'])
        metadata = {'version': 1, 'period': entry['period'], 'subset': entry['subset'], 'start': entry['start'],
                        'smoke': smoke, 'teacher_rounds': rounds, 'min_eligible': minimum, 'teacher_seed': 42,
                        'constant': list(self.constant), 'varying': list(self.varying),
                        'parent_constant': None if self.parent_constant is None else list(self.parent_constant),
                        'params': PARAMS, 'xgboost': xgb.__version__, 'threads': self.threads,
                        'warmup_days': 365, 'purge_hours': 144, 'segments': 3, 'spectral': .1}
        identity = {'x': x, 'target': entry['target'], 'original': entry['original'],
                        'origins': entry['grid']['origins'].asi8}
        directory = Path(self.cache_root) / ('smoke' if smoke else 'full') / entry['period'] / entry['subset']
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"block{entry['start']//24}.npz"
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    if json.loads(str(saved['metadata'])) != metadata or any(
                        not np.array_equal(saved[key], value, equal_nan=True) for key, value in identity.items()
                    ):
                        raise ValueError(f'Stale hardness cache: {path}')
                    prediction = saved['prediction']
                    q, eligible, gamma = rank_hardness(entry['original'], prediction, min_eligible=minimum)
                    if not np.array_equal(q, saved['q']) or not np.array_equal(eligible, saved['eligible']):
                        raise ValueError('Cached hardness ranks fail recomputation')
                    return q, eligible, gamma, True
            prediction, provenance = self._teacher(entry, x, rounds)
            q, eligible, gamma = rank_hardness(entry['original'], prediction, min_eligible=minimum)
            # Teacher residuals remain training-only diagnostic evidence.
            observed = np.isfinite(entry['original'])
            error = prediction - entry['original']
            counts = observed[eligible].sum(1)
            summaries = {
                'speed': np.where(observed[eligible], entry['original'][eligible], 0.).sum(1) / counts,
                'absolute_error': np.where(observed[eligible], abs(error[eligible]), 0.).sum(1) / counts,
                'bias': np.where(observed[eligible], error[eligible], 0.).sum(1) / counts,
            }
            correlations = {name: (float(np.corrcoef(q[eligible], values)[0, 1])
                                   if np.std(values) > 0 and np.std(q[eligible]) > 0 else None)
                            for name, values in summaries.items()}
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream, **identity, metadata=json.dumps(metadata, sort_keys=True),
                         prediction=prediction, q=q, eligible=eligible, gamma=gamma,
                         provenance=json.dumps(provenance), correlations=json.dumps(correlations, allow_nan=False))
            temporary.replace(path)
            return q, eligible, gamma, False

    def make_objective(self, target, context):
        matches = [entry for entry in context if entry['target'].shape == target.shape
                   and np.array_equal(entry['target'], target)]
        if len(matches) != 1:
            raise ValueError('Fitted block must have one exact original-label context')
        entry = matches[0]
        q, eligible, gamma, hit = self._cached(entry)
        permutation = []
        if self.mode == 'null':
            q, permutation = permute_blocks(q, eligible, entry['grid']['origins'])
            permutation = permutation.tolist()
        callback, weights = weighted_objective(target, 1. + q)
        observed = np.isfinite(entry['original'])
        origins = entry['grid']['origins']
        time_groups = np.asarray((origins - origins[0]) // pd.Timedelta(days=7))
        groups, inverse = np.unique(time_groups, return_inverse=True)
        time_mass = np.bincount(inverse, weights=weights.sum(1)).tolist()
        print(json.dumps({'event': 'hardness', 'period': entry['period'], 'subset': entry['subset'],
                              'block': entry['start']//24, 'mode': self.mode, 'smoke': entry['subset'].startswith('smoke_'),
                              'cache_hit': hit, 'eligible': int(eligible.sum()), 'rows': len(target), 'gamma': gamma.tolist(),
                              'modifier_min': float((1+q).min()), 'modifier_max': float((1+q).max()),
                              'ess': ((weights.sum(0)**2)/(weights**2).sum(0)).tolist(),
                              'observed_weight_mass': float(weights[observed].sum()),
                              'filled_weight_mass': float(weights[~observed].sum()),
                              'permutation': permutation, 'seven_day_groups': groups.tolist(),
                              'seven_day_weight_mass': time_mass}, allow_nan=False), flush=True)
        return callback
