"""Training-only relative RMSE/TIMSE risks with a positive curvature proxy.

The proxy is the residual coefficient, not the exact Hessian or a bound.
No risk floors, dynamic rescaling, acceptance gates, or evaluation labels.
"""
from __future__ import annotations

import time

import numpy as np
import xgboost as xgb

from .gbm import load_booster
from .objectives import timse_coefficients


def _positive(value, name):
    if not np.isfinite(value).all() or not (np.asarray(value) > 0).all():
        raise ValueError(f'{name} must be finite and strictly positive')


def _callback_arrays(gradient, curvature):
    with np.errstate(over='ignore', under='ignore'):
        g, h = gradient.astype(np.float32), curvature.astype(np.float32)
    if not np.isfinite(g).all() or np.any((gradient != 0) & (g == 0)):
        raise ValueError('Float32 gradient overflow or underflow')
    _positive(h, 'float32 curvature')
    return g, h


def _mixture(r, u, tau):
    _positive(np.array([r, u]), 'relative risks')
    if tau is None:
        value, weights = .5 * r + .5 * u, np.array([.5, .5])
    else:
        # Shift before division so large, close finite risks remain representable.
        largest = max(r, u)
        with np.errstate(over='ignore', under='ignore'):
            exponentials = np.exp((np.array([r, u]) - largest) / tau)
        weights = exponentials / exponentials.sum()
        value = largest + tau * (np.log(exponentials.sum()) - np.log(2.))
    _positive(weights, 'mixture weights (underflow is not floored)')
    _positive(value, 'objective value')
    return float(value), weights


class BalancedAverageObjective:
    """Exact gradient of fixed_scale*(R/R0 + sqrt(T/T0))/2."""
    arm = 'B-AVG'
    tau = None

    def __init__(self, target):
        self.target = np.array(target, dtype=np.float64, copy=True)
        if (self.target.ndim != 2 or self.target.shape[0] < 2 or
                self.target.shape[1] < 1 or not np.isfinite(self.target).all()):
            raise ValueError('Target must be a finite nonempty two-dimensional matrix')
        self.c = np.column_stack([timse_coefficients(y) for y in self.target.T])
        if not np.isfinite(self.c).all() or not np.allclose(self.c.sum(0), 1., rtol=1e-12, atol=1e-12):
            raise ValueError('Invalid TIMSE coefficients')
        with np.errstate(over='ignore', invalid='ignore'):
            self.base_score = float(np.float32(self.target.mean()))
        if not np.isfinite(self.base_score):
            raise ValueError('Nonfinite float32 base score')
        self.initial_prediction = np.full(self.target.shape, self.base_score)
        initial = self._risks(self.initial_prediction)
        self.R0, self.U0 = initial['R'], initial['U']
        self.fixed_scale = float(1 / self.coefficient(self.initial_prediction).mean())
        _positive(self.fixed_scale, 'fixed scale')
        self.initial = self.risks(self.initial_prediction)
        self.calls = 0
        _callback_arrays(*self.derivatives(self.initial_prediction))

    def errors(self, prediction):
        prediction = np.asarray(prediction, dtype=np.float64)
        if prediction.shape != self.target.shape or not np.isfinite(prediction).all():
            raise ValueError('Prediction must have the finite target shape')
        with np.errstate(over='ignore', invalid='ignore'):
            errors = prediction - self.target
        if not np.isfinite(errors).all():
            raise ValueError('Nonfinite residual')
        return errors

    def _risks(self, prediction):
        e = self.errors(prediction)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            squared = e * e
            m = squared.mean(0)
            t = (self.c * squared).sum(0)
            s = np.sqrt(np.sqrt(m) * np.sqrt(t))
        _positive(m, 'MSE'); _positive(t, 'TIMSE'); _positive(s, 'S')
        R, T = float(np.sqrt(m).mean()), float(t.mean())
        U = float(np.sqrt(T))
        _positive(np.array([R, T, U]), 'aggregate risks')
        return dict(R=R, T=T, U=U, rmse=R, timse=T, s=float(s.mean()),
                    mse_by_lead=m.tolist(), rmse_by_lead=np.sqrt(m).tolist(),
                    timse_by_lead=t.tolist(), s_by_lead=s.tolist())

    def risks(self, prediction):
        result = self._risks(prediction)
        r, u = result['R'] / self.R0, result['U'] / self.U0
        value, pi = _mixture(r, u, self.tau)
        return dict(result, r=r, u=u, pi_R=float(pi[0]), pi_U=float(pi[1]),
                    objective_unscaled=value, objective_value=value * self.fixed_scale,
                    fixed_scale=self.fixed_scale)

    def coefficient(self, prediction):
        risk = self._risks(prediction)
        _, pi = _mixture(risk['R'] / self.R0, risk['U'] / self.U0, self.tau)
        n, h = self.target.shape
        with np.errstate(over='ignore', under='ignore', divide='ignore'):
            a = (pi[0] / (h * self.R0 * n * np.sqrt(risk['mse_by_lead']))[None, :] + \
                pi[1] * self.c / (h * self.U0 * risk['U']))
        _positive(a, 'curvature proxy')
        return a

    def value(self, prediction):
        return self.risks(prediction)['objective_value']

    def derivatives(self, prediction):
        curvature = self.fixed_scale * self.coefficient(prediction)
        gradient = curvature * self.errors(prediction)
        if not np.isfinite(gradient).all():
            raise ValueError('Nonfinite gradient')
        _positive(curvature, 'scaled curvature proxy')
        return gradient, curvature

    def __call__(self, prediction, matrix):
        self.calls += 1
        return _callback_arrays(*self.derivatives(prediction))


class BalancedMaxObjective(BalancedAverageObjective):
    """Smooth maximum of the relative risks; tau=0.1 is fixed before training."""
    arm = 'B-MAX'
    tau = .1


class _Trace(xgb.callback.TrainingCallback):
    def __init__(self, objective, matrix):
        self.objective, self.matrix = objective, matrix
        self.records = [dict(round=0, **objective.initial)]

    def after_iteration(self, model, epoch, evals_log):
        if (epoch + 1) % 50 == 0:
            self.records.append(dict(round=epoch + 1, **self.objective.risks(model.predict(self.matrix))))
        return False


def _fit(objective_type, spec, x, xq, y, seed, threads, rounds, context, smoke):
    if not isinstance(rounds, int) or rounds < 1:
        raise ValueError('rounds must be a positive integer')
    objective = objective_type(y)
    matrix = xgb.DMatrix(x, label=np.asarray(y, np.float32), nthread=threads)
    params = dict(spec.params, seed=seed, device='cpu', nthread=threads,
                  num_target=objective.target.shape[1], base_score=objective.base_score)
    trace = _Trace(objective, matrix)
    model = xgb.train(params, matrix, num_boost_round=rounds, obj=objective, callbacks=[trace])
    final = objective.risks(model.predict(matrix))
    if trace.records[-1]['round'] != rounds:
        trace.records.append(dict(round=rounds, **final))
    artifact = np.frombuffer(bytes(model.save_raw('ubj')), np.uint8)
    began = time.perf_counter()
    forecast = model.inplace_predict(xq)
    seconds = time.perf_counter() - began
    replay = load_booster(artifact, threads)
    if (not np.array_equal(replay.inplace_predict(xq), forecast) or
            replay.num_boosted_rounds() != rounds or objective.calls != rounds):
        raise ValueError('Checkpoint replay or round/callback count mismatch')
    return forecast, artifact, seconds, dict(replay_max_abs_error=0., balanced_loss=dict(
        arm=objective.arm, tau=objective.tau, initial=objective.initial, final=final,
        fixed_scale=objective.fixed_scale, R0=objective.R0, U0=objective.U0,
        rounds=rounds, trees=rounds, objective_calls=objective.calls, records=trace.records,
        curvature='positive residual-coefficient proxy; not exact Hessian or bound'))


def fit_avg_block(spec, x, xq, y, seed, threads, rounds, context, smoke):
    return _fit(BalancedAverageObjective, spec, x, xq, y, seed, threads, rounds, context, smoke)


def fit_max_block(spec, x, xq, y, seed, threads, rounds, context, smoke):
    return _fit(BalancedMaxObjective, spec, x, xq, y, seed, threads, rounds, context, smoke)
