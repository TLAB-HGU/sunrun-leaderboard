"""Multi-step forecasting strategies on the E2000 recipe (Direct itself is out of scope).

Every strategy is built from three block learners over horizons 1..72 split into blocks of size sigma:
  RecMO(sigma)    one shared sigma-output model, applied 72/sigma times on its own trajectory
  DirMO(sigma)    one model per block, origin information only
  DirRecMO(sigma) one model per block, plus the trajectory of earlier blocks
Recursive = RecMO(1), MIMO = DirMO(72), DirRec = DirRecMO(1). Rectify wraps a base and a residual learner
(residual targets use purged out-of-fold base forecasts). Multi-output blocks use XGBoost joint trees.
Teacher forcing: training lags/trajectories are observed (filled) speeds; inference uses own predictions only.
"""
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from xgboost import XGBRegressor

from data import H, LAG, LB

if str(LB) not in sys.path:
    sys.path.insert(0, str(LB))
from experiments.e2000.model import ModelConfig  # noqa: E402

PURGE_H = 72
FOLDS = 4
PRED_MIN, PRED_MAX = 1., 3000.
# Measured on this 24-thread host: ~1e4-row fits are fastest at 1-4 threads and 10-50x slower at 24;
# the stacked shared design (~7e5 rows) peaks near 12. Per-block models therefore run as parallel processes.
SHARED_THREADS = int(os.environ.get('FS_SHARED_THREADS', '12'))
BLOCK_THREADS = int(os.environ.get('FS_BLOCK_THREADS', '4'))
WORKERS = int(os.environ.get('FS_WORKERS', '6'))


def recipe(n_estimators=None, n_jobs=None):
    """The frozen E2000 recipe (seed 42); only rounds (smoke runs) and threads may change."""
    cfg = ModelConfig(n_jobs=n_jobs or SHARED_THREADS)
    return replace(cfg, n_estimators=n_estimators) if n_estimators else cfg


class Inputs:
    """ctx (n, k) origin-level features, E (n, 72, m) horizon features, past (n, LAG) speeds ending at origin."""

    def __init__(self, ctx, E, past):
        self.ctx, self.E, self.past = (np.asarray(ctx, np.float32), np.asarray(E, np.float32),
                                       np.asarray(past, np.float32))
        if not (len(self.ctx) == len(self.E) == len(self.past)) or self.E.shape[1] != H:
            raise ValueError('inconsistent input shapes')

    @classmethod
    def of(cls, bundle, extra=None, past=None):
        ctx = bundle.C if extra is None else np.hstack([bundle.C, extra])
        return cls(ctx, bundle.E, bundle.past if past is None else past)

    def __len__(self):
        return len(self.ctx)

    def take(self, index):
        return Inputs(self.ctx[index], self.E[index], self.past[index])


def blocks(sigma):
    if H % sigma:
        raise ValueError('sigma must divide 72')
    return [(s, s + sigma - 1) for s in range(1, H + 1, sigma)]


def _fit_xgb(tag, X, Y, Xv, Yv, cfg, root):
    Y = Y.reshape(len(Y), -1)
    Yv = Yv.reshape(len(Yv), -1)
    params = cfg.params()
    if Y.shape[1] > 1:
        params['multi_strategy'] = 'multi_output_tree'
    model = XGBRegressor(**params)
    model.fit(X, Y if Y.shape[1] > 1 else Y[:, 0], eval_set=[(X, Y), (Xv, Yv)], verbose=False)
    if root is not None:
        ev = model.evals_result()
        frame = pd.DataFrame({'step': np.arange(1, cfg.n_estimators + 1),
                              'train_mse': np.square(ev['validation_0']['rmse']),
                              'validation_mse': np.square(ev['validation_1']['rmse'])})
        path = Path(root) / f'{tag}_s42.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return model


class Logger:
    """Native per-round loss of every fit: step,train_mse,validation_mse in (km/s)^2. ``seconds`` is wall time."""

    def __init__(self, root=None):
        self.root = None if root is None else Path(root)
        self.seconds = 0.

    def fit(self, tag, X, Y, Xv, Yv, cfg):
        start = time.perf_counter()
        model = _fit_xgb(tag, X, Y, Xv, Yv, cfg, self.root)
        self.seconds += time.perf_counter() - start
        return model

    def fit_many(self, jobs, cfg):
        """jobs: [(tag, X, Y, Xv, Yv)]; independent fits run as WORKERS processes x BLOCK_THREADS threads."""
        cfg = replace(cfg, n_jobs=min(cfg.n_jobs, BLOCK_THREADS))
        start = time.perf_counter()
        if len(jobs) == 1 or WORKERS == 1:
            models = [_fit_xgb(*job, cfg, self.root) for job in jobs]
        else:
            models = Parallel(n_jobs=min(WORKERS, len(jobs)), backend='loky', max_nbytes='16M')(
                delayed(_fit_xgb)(*job, cfg, self.root) for job in jobs)
        self.seconds += time.perf_counter() - start
        return models


def _predict(model, X, width):
    return np.asarray(model.predict(X), dtype=np.float64).reshape(len(X), width)


def _require_finite_targets(Y):
    if not np.isfinite(Y).all():
        raise ValueError('training targets must be finite (filled speed series)')


class Block:
    """Shared machinery: trajectory = [past LAG speeds | 72 future values]."""
    shared = False

    def __init__(self, sigma, cfg, logger, tag):
        self.sigma, self.cfg, self.logger, self.tag = sigma, cfg, logger, tag
        self.models = {}

    @staticmethod
    def trajectory(inp, Y):
        return np.hstack([inp.past.astype(np.float64), Y])

    def features(self, inp, traj, s, e):
        raise NotImplementedError

    def fit(self, inp, Y, inp_v, Y_v):
        _require_finite_targets(Y)
        T, Tv = self.trajectory(inp, Y), self.trajectory(inp_v, Y_v)
        spans = blocks(self.sigma)
        if self.shared:
            X = np.vstack([self.features(inp, T, s, e) for s, e in spans])
            y = np.vstack([Y[:, s - 1:e] for s, e in spans])
            Xv = np.vstack([self.features(inp_v, Tv, s, e) for s, e in spans])
            yv = np.vstack([Y_v[:, s - 1:e] for s, e in spans])
            self.models[0] = self.logger.fit(self.tag, X, y, Xv, yv, self.cfg)
        else:
            jobs = [(f'{self.tag}_b{b:02}', self.features(inp, T, s, e), Y[:, s - 1:e],
                     self.features(inp_v, Tv, s, e), Y_v[:, s - 1:e]) for b, (s, e) in enumerate(spans)]
            self.models = dict(enumerate(self.logger.fit_many(jobs, self.cfg)))
        return self

    def predict(self, inp):
        T = self.trajectory(inp, np.full((len(inp), H), np.nan))
        for b, (s, e) in enumerate(blocks(self.sigma)):
            model = self.models[0 if self.shared else b]
            T[:, LAG + s - 1:LAG + e] = _predict(model, self.features(inp, T, s, e), e - s + 1)
        return T[:, LAG:]


class RecMO(Block):
    shared = True

    def features(self, inp, traj, s, e):
        lag = traj[:, s - 1:s - 1 + LAG]  # the LAG values ending at horizon s-1
        lead = np.full((len(inp), 1), s, np.float32)
        return np.hstack([inp.ctx, lag, inp.E[:, s - 1:e].reshape(len(inp), -1), lead]).astype(np.float32)


class DirMO(Block):
    def features(self, inp, traj, s, e):
        return np.hstack([inp.ctx, inp.past, inp.E[:, s - 1:e].reshape(len(inp), -1)]).astype(np.float32)


class DirRecMO(Block):
    def features(self, inp, traj, s, e):
        earlier = traj[:, LAG:LAG + s - 1]  # horizons 1..s-1: observed in training, predicted at inference
        return np.hstack([inp.ctx, inp.past, earlier,
                          inp.E[:, s - 1:e].reshape(len(inp), -1)]).astype(np.float32)


KINDS = {'RecMO': RecMO, 'DirMO': DirMO, 'DirRecMO': DirRecMO}


def purged_folds(origins, k=FOLDS, purge_h=PURGE_H):
    """Contiguous time folds; training origins within purge_h of a held-out fold are dropped on both sides."""
    origins = pd.DatetimeIndex(origins)
    parts = np.array_split(np.arange(len(origins)), k)
    gap = pd.Timedelta(hours=purge_h)
    out = []
    for held in parts:
        lo, hi = origins[held[0]], origins[held[-1]]
        keep = (origins < lo - gap) | (origins > hi + gap)
        out.append((np.flatnonzero(keep), held))
    return out


class Rectify:
    """Base forecast + residual learner. Residual inputs add the past speeds and the full base trajectory."""

    def __init__(self, base, residual, cfg, logger, tag, base_cache=None):
        (self.bk, self.bs), (self.rk, self.rs) = base, residual
        self.cfg, self.logger, self.tag = cfg, logger, tag
        self.cache = {} if base_cache is None else base_cache

    def _base(self, tag):
        return KINDS[self.bk](self.bs, self.cfg, self.logger, tag)

    @staticmethod
    def residual_inputs(inp, B):
        return Inputs(np.hstack([inp.ctx, inp.past, B]), inp.E, np.zeros_like(inp.past))

    def fit(self, inp, Y, inp_v, Y_v, origins):
        # Base fits are shared across Rectify/Stratify variants and with the standalone block strategy.
        oof_key, full_key = ('oof', self.bk, self.bs), ('full', self.bk, self.bs)
        if oof_key not in self.cache:
            oof = np.full_like(Y, np.nan)
            for f, (tr, held) in enumerate(purged_folds(origins)):
                m = self._base(f'base_{self.bk}{self.bs}_oof{f}').fit(inp.take(tr), Y[tr], inp_v, Y_v)
                oof[held] = m.predict(inp.take(held))
            self.cache[oof_key] = oof
        if full_key not in self.cache:
            self.cache[full_key] = self._base(f'base_{self.bk}{self.bs}_full').fit(inp, Y, inp_v, Y_v)
        oof, self.base = self.cache[oof_key], self.cache[full_key]
        Bv = self.base.predict(inp_v)
        self.res = KINDS[self.rk](self.rs, self.cfg, self.logger, f'res_{self.rk}{self.rs}').fit(
            self.residual_inputs(inp, oof), Y - oof, self.residual_inputs(inp_v, Bv), Y_v - Bv)
        return self

    def predict(self, inp):
        B = self.base.predict(inp)
        return B + self.res.predict(self.residual_inputs(inp, B))


def spec(name):
    """Strategy name -> constructor arguments."""
    fixed = {'Recursive': ('RecMO', 1), 'DirRec': ('DirRecMO', 1), 'MIMO': ('DirMO', 72)}
    if name in fixed:
        return {'kind': 'block', 'block': fixed[name]}
    if name == 'Rectify':
        return {'kind': 'rectify', 'base': ('RecMO', 1), 'residual': ('DirMO', 1)}
    head, _, rest = name.partition('-')
    if head in KINDS:
        return {'kind': 'block', 'block': (head, int(rest))}
    if head == 'RectifyMO':
        return {'kind': 'rectify', 'base': ('RecMO', int(rest)), 'residual': ('DirMO', int(rest))}
    if head == 'Stratify':  # Stratify-<base>-<residual>-<sigma>
        b, r, s = rest.split('-')
        return {'kind': 'rectify', 'base': (b, int(s)), 'residual': (r, int(s))}
    raise ValueError('unknown strategy ' + name)


def build(name, cfg, logger, base_cache=None):
    sp = spec(name)
    if sp['kind'] == 'block':
        kind, sigma = sp['block']
        return KINDS[kind](sigma, cfg, logger, name)
    return Rectify(sp['base'], sp['residual'], cfg, logger, name, base_cache)


def fit_predict(name, cfg, fit, Y, valid, Y_v, targets, *, origins, log_root=None, base_cache=None):
    """Fit one strategy and predict each Inputs in ``targets``.

    Returns (predictions, fit seconds, inference seconds per origin). Inference covers only the
    predict call on prepared inputs (leaderboard rule: no load/train/feature preparation/output).
    """
    logger = Logger(None if log_root is None else Path(log_root) / name)
    model = build(name, cfg, logger, base_cache)
    if isinstance(model, Rectify):
        model.fit(fit, Y, valid, Y_v, origins)
    else:
        model.fit(fit, Y, valid, Y_v)
        if base_cache is not None:
            base_cache[('full',) + spec(name)['block']] = model
    preds, per_origin = {}, {}
    for k, v in targets.items():
        start = time.perf_counter()
        raw = model.predict(v)
        per_origin[k] = (time.perf_counter() - start) / len(v)
        preds[k] = np.clip(raw, PRED_MIN, PRED_MAX)
    return preds, logger.seconds, per_origin


STANDALONE = (['Recursive', 'DirRec', 'MIMO']
              + [f'{k}-{s}' for k in ('RecMO', 'DirMO', 'DirRecMO') for s in (6, 12, 24)])
RECTIFY = ['Rectify'] + [f'RectifyMO-{s}' for s in (6, 12, 24)]
STRATIFY = [f'Stratify-{b}-{r}-{s}' for s in (6, 24) for b in KINDS for r in KINDS]
