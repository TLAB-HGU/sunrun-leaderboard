"""Training-only normalized joint loss and actual-tree S/R/T acceptance.

All scaling is a single frozen block scalar. S curvature is a positive
Newton proxy, not the exact Hessian of this row-coupled objective.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import xgboost as xgb

from .gbm import load_booster, predict_block
from .objectives import timse_coefficients

ALPHAS = (1., .5, .25, .125, .0625, .03125, .015625)


def _positive(value, name):
    if not np.isfinite(value).all() or not (np.asarray(value) > 0).all():
        raise ValueError(f'{name} must be finite and strictly positive')


def _callback_arrays(gradient, curvature):
    with np.errstate(over='ignore', under='ignore'):
        g, h = gradient.astype(np.float32), curvature.astype(np.float32)
    if not np.isfinite(g).all():
        raise ValueError('Nonfinite float32 gradient')
    if np.any((gradient != 0) & (g == 0)):
        raise ValueError('Float32 gradient underflow')
    _positive(h, 'float32 curvature')
    return g, h


class TrainingRisk:
    def __init__(self, target):
        self.target = np.asarray(target, np.float64)
        if self.target.ndim != 2 or not np.isfinite(self.target).all():
            raise ValueError('Target must be a finite two-dimensional matrix')
        self.c = np.column_stack([timse_coefficients(column) for column in self.target.T])
        if not np.isfinite(self.c).all() or not np.allclose(self.c.sum(0), 1., rtol=1e-12, atol=1e-12):
            raise ValueError('Invalid TIMSE coefficients')
        self.base_score = float(np.float32(self.target.mean()))
        if not np.isfinite(self.base_score):
            raise ValueError('Nonfinite float32 base score')
        self.initial_prediction = np.full(self.target.shape, self.base_score)
        self.initial = self.risks(self.initial_prediction)

    def errors(self, prediction):
        prediction = np.asarray(prediction, np.float64).reshape(self.target.shape)
        if not np.isfinite(prediction).all():
            raise ValueError('Prediction must be finite')
        return prediction - self.target

    def risks(self, prediction):
        e = self.errors(prediction)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            squared = e * e
            m = squared.mean(0)
            t = (self.c * squared).sum(0)
            # Split powers avoid overflow in M*T while preserving its definition.
            s = np.sqrt(np.sqrt(m) * np.sqrt(t))
        _positive(m, 'MSE'); _positive(t, 'TIMSE'); _positive(s, 'S')
        _positive(np.array([s.mean(), np.sqrt(m).mean(), t.mean()]), 'Aggregate risk')
        return dict(s=float(s.mean()), rmse=float(np.sqrt(m).mean()), timse=float(t.mean()),
                    mse_by_lead=m.tolist(), rmse_by_lead=np.sqrt(m).tolist(),
                    timse_by_lead=t.tolist(), s_by_lead=s.tolist())

    def metric_gradients(self, prediction):
        e = self.errors(prediction)
        r = self.risks(prediction)
        m, t, s = (np.asarray(r[key]) for key in ('mse_by_lead', 'timse_by_lead', 's_by_lead'))
        n, h = e.shape
        return dict(s=.5 * s * (1 / (n * m) + self.c / t) * e / h,
                    rmse=e / (n * np.sqrt(m) * h), timse=2 * self.c * e / h)


class NormalizedJointObjective(TrainingRisk):
    """L=mean_h(.5*M/M0+.5*T/T0), multiplied by H*fixed_scale."""
    def __init__(self, target):
        super().__init__(target)
        m = np.asarray(self.initial['mse_by_lead'])
        t = np.asarray(self.initial['timse_by_lead'])
        self.weights = 1 / (len(self.target) * m) + self.c / t
        _positive(self.weights, 'Joint weights')
        self.fixed_scale = float(1 / self.weights.mean())
        _positive(self.fixed_scale, 'Joint scale')
        self.curvature = self.weights * self.fixed_scale
        _callback_arrays(np.zeros_like(self.curvature), self.curvature)

    def value(self, prediction):
        return float(.5 * np.sum(self.curvature * self.errors(prediction) ** 2))

    def derivatives(self, prediction):
        return self.curvature * self.errors(prediction), self.curvature

    def __call__(self, prediction, matrix):
        return _callback_arrays(*self.derivatives(prediction))


class DirectScoreObjective(TrainingRisk):
    """Exact derivative of fixed_scale*sum(S_h), with positive diagonal proxy."""
    def __init__(self, target):
        super().__init__(target)
        self.fixed_scale = float(1 / self.coefficient(self.initial_prediction).mean())
        _positive(self.fixed_scale, 'Direct S scale')
        _callback_arrays(*self.derivatives(self.initial_prediction))

    def coefficient(self, prediction):
        r = self.risks(prediction)
        m, t, s = (np.asarray(r[key]) for key in ('mse_by_lead', 'timse_by_lead', 's_by_lead'))
        coefficient = .5 * s * (1 / (len(self.target) * m) + self.c / t)
        _positive(coefficient, 'Direct S curvature proxy')
        return coefficient

    def value(self, prediction):
        return float(self.fixed_scale * sum(self.risks(prediction)['s_by_lead']))

    def derivatives(self, prediction):
        curvature = self.fixed_scale * self.coefficient(prediction)
        return curvature * self.errors(prediction), curvature

    def __call__(self, prediction, matrix):
        return _callback_arrays(*self.derivatives(prediction))


def gate_reasons(candidate, current, initial, tolerances, require_joint):
    reasons = []
    if not all(np.isfinite(candidate[k]) for k in ('s', 'rmse', 'timse')):
        return ['nonfinite_candidate']
    if candidate['s'] >= current['s'] or current['s'] - candidate['s'] < tolerances['s']:
        reasons.append('s_not_decreased')
    if require_joint:
        for key in ('rmse', 'timse'):
            if candidate[key] > current[key] + tolerances[key]:
                reasons.append(f'{key}_step_increase')
            if candidate[key] > initial[key] + tolerances[key]:
                reasons.append(f'{key}_initial_bound')
    return reasons


def _buffer(model):
    return np.frombuffer(bytes(model.save_raw('ubj')), np.uint8)


def _finish(model, xq, threads, count, metadata):
    artifact = _buffer(model)
    began = time.perf_counter()
    forecast = model.inplace_predict(xq)
    seconds = time.perf_counter() - began
    replay = load_booster(artifact, threads)
    if not np.array_equal(replay.inplace_predict(xq), forecast) or replay.num_boosted_rounds() != count:
        raise ValueError('Checkpoint replay or tree count mismatch')
    return forecast, artifact, seconds, dict(replay_max_abs_error=0., joint_loss=metadata)


def fit_joint_block(spec, x, xq, y, seed, threads, rounds, context, smoke):
    objective = NormalizedJointObjective(y)
    matrix = xgb.DMatrix(x, label=np.asarray(y, np.float32), nthread=threads)
    params = dict(spec.params, seed=seed, device='cpu', nthread=threads,
                  num_target=y.shape[1], base_score=objective.base_score)
    model = xgb.train(params, matrix, num_boost_round=rounds, obj=objective)
    final = objective.risks(model.inplace_predict(x))
    return _finish(model, xq, threads, rounds,
                   dict(arm='J', fixed_scale=objective.fixed_scale, initial=objective.initial,
                        final=final, accepted_trees=rounds, tree_fit_attempts=rounds))


@dataclass(frozen=True)
class JointGateFitter:
    require_joint: bool = False

    def launch_record(self):
        return dict(arm='D-RT' if self.require_joint else 'D-S', alphas=list(ALPHAS),
                    seed_schedule='seed+iteration', tolerance_relative=1e-10,
                    curvature='positive diagonal proxy', initial='float32 scalar training mean')

    predict_block = staticmethod(predict_block)

    def fit_block(self, spec, x, xq, y, seed, threads, rounds, context, smoke):
        risk = DirectScoreObjective(y)
        matrix = xgb.DMatrix(x, label=np.asarray(y, np.float32), nthread=threads)
        params = dict(spec.params, seed=seed, device='cpu', nthread=threads,
                      num_target=y.shape[1], base_score=risk.base_score)
        model = xgb.train(params, matrix, num_boost_round=0)
        current_prediction = model.inplace_predict(x).astype(np.float64).reshape(y.shape)
        if not np.array_equal(current_prediction, risk.initial_prediction):
            raise ValueError('Initial booster prediction differs from frozen base score')
        current = risk.initial
        tolerances = {k: 1e-10 * max(1., abs(current[k])) for k in ('s', 'rmse', 'timse')}
        records, attempts, accepted = [], 0, 0
        stop_reason = 'proposal_budget_exhausted'
        for iteration in range(rounds):
            before = _buffer(model)
            g, h = risk(current_prediction, matrix)
            metric_gradients = risk.metric_gradients(current_prediction)
            def objective(prediction, dmatrix):
                return g, h
            taken = False
            for alpha in ALPHAS:
                attempts += 1
                began = time.perf_counter()
                candidate = xgb.train(dict(params, seed=seed + iteration,
                                          learning_rate=params['learning_rate'] * alpha),
                                      matrix, num_boost_round=1, obj=objective,
                                      xgb_model=load_booster(before, threads))
                prediction = candidate.inplace_predict(x).astype(np.float64).reshape(y.shape)
                try:
                    proposed = risk.risks(prediction)
                except ValueError as error:
                    records.append(dict(iteration=iteration, seed=seed + iteration, alpha=alpha,
                                        accepted=False, reasons=['undefined_candidate_risk'], error=str(error)))
                    continue
                reasons = gate_reasons(proposed, current, risk.initial, tolerances, self.require_joint)
                direction = prediction - current_prediction
                record = dict(iteration=iteration, seed=seed + iteration, alpha=alpha,
                              accepted=not reasons, reasons=reasons, current=current, candidate=proposed,
                              seconds=time.perf_counter() - began,
                              gradient_dot_direction={k: float(np.sum(v * direction))
                                                      for k, v in metric_gradients.items()},
                              lead_changes={k: (np.asarray(proposed[k]) - np.asarray(current[k])).tolist()
                                            for k in ('rmse_by_lead', 'timse_by_lead', 's_by_lead')})
                records.append(record)
                if not reasons:
                    model, current_prediction, current = candidate, prediction, proposed
                    accepted += 1
                    taken = True
                    break
            if not taken:
                if not np.array_equal(_buffer(model), before):
                    raise ValueError('Rejected candidate mutated accepted booster')
                stop_reason = 'all_alphas_rejected'
                break
        metadata = dict(self.launch_record(), initial=risk.initial, final=current,
                        fixed_scale=risk.fixed_scale, tolerances=tolerances,
                        proposed_iterations=iteration + 1 if rounds else 0,
                        accepted_trees=accepted, tree_fit_attempts=attempts,
                        stop_reason=stop_reason, records=records,
                        strict_final_joint_decrease=all(current[k] < risk.initial[k] - tolerances[k]
                                                        for k in ('rmse', 'timse')))
        return _finish(model, xq, threads, accepted, metadata)


# Public entry points used by the single-arm experiment specifications.
DirectSObjective = DirectScoreObjective
fit_s_gate = JointGateFitter(False).fit_block
fit_rt_gate = JointGateFitter(True).fit_block
