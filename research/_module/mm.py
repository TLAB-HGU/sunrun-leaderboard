"""Matched dynamic and stage-fixed majorization of training-sample Direct-S.

Both fitters continue one vector-leaf booster from a scalar training mean and
accept a stage only when its stage-start quadratic majorizer and exact S do not
increase. Only the coefficient refresh frequency differs between the modes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import xgboost as xgb

from .gbm import load_booster, predict_block
from .objectives import timse_coefficients


class ZeroTrainingRisk(ValueError):
    """An exact zero MSE or TIMSE makes the unsmoothed S derivative undefined."""


class TrainingScore:
    """Exact unscaled sum of lead S values and its quadratic majorizer."""

    def __init__(self, target):
        self.target = np.array(target, dtype=np.float64, copy=True)
        if (self.target.ndim != 2 or self.target.shape[0] < 2 or self.target.shape[1] < 1
                or not np.isfinite(self.target).all()):
            raise ValueError('Training targets require a finite matrix with at least two rows')
        self.tail = np.column_stack([timse_coefficients(column) for column in self.target.T])

    def error(self, prediction):
        prediction = np.asarray(prediction, dtype=np.float64)
        if prediction.shape != self.target.shape or not np.isfinite(prediction).all():
            raise ValueError('Prediction must be finite and match training targets')
        return prediction - self.target

    def terms(self, prediction):
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            squared = self.error(prediction) ** 2
            mse = squared.mean(axis=0)
            timse = (self.tail * squared).sum(axis=0)
            score = (mse * timse) ** .25
        if not all(np.isfinite(value).all() for value in (mse, timse, score)):
            raise ValueError('Nonfinite training risk')
        return mse, timse, score

    def curvature(self, prediction):
        """Exact diagonal Hessian 2c of the tangent quadratic (not Hessian of S)."""
        mse, timse, score = self.terms(prediction)
        if np.any(mse == 0) or np.any(timse == 0):
            raise ZeroTrainingRisk('Zero training risk at one or more leads')
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            weight = .5 * score * (1. / (len(self.target) * mse) + self.tail / timse)
        if not np.isfinite(weight).all() or not (weight > 0).all():
            raise ValueError('Majorizer curvature must be positive and finite')
        return weight

    def value(self, prediction):
        return float(self.terms(prediction)[2].sum())

    def surrogate(self, prediction, anchor, curvature=None):
        """Tangent majorizer including its constant, summed over the block's leads."""
        weight = self.curvature(anchor) if curvature is None else curvature
        with np.errstate(over='raise', invalid='raise'):
            value = .5 * self.value(anchor) + .5 * np.sum(weight * self.error(prediction) ** 2)
        if not np.isfinite(value):
            raise ValueError('Nonfinite majorizer')
        return float(value)

    def summary(self, prediction):
        mse, timse, score = self.terms(prediction)
        return dict(mean_s=float(score.mean()), mse_by_lead=mse.tolist(),
                    timse_by_lead=timse.tolist(), s_by_lead=score.tolist())


@dataclass(frozen=True)
class StagewiseS:
    """GBMSpec block fitter: mode='frozen' for MM, 'dynamic' for its control.

    One scalar normalizes initial mean curvature to one in both modes. It stays
    fixed for the whole block, preserving lead ratios and MM descent. Reported
    S and Q remain unscaled. No per-lead or per-stage renormalization is applied.
    """

    mode: str = 'frozen'
    stage_rounds: int = 30

    def __post_init__(self):
        if self.mode not in ('frozen', 'dynamic'):
            raise ValueError('mode must be frozen or dynamic')
        if type(self.stage_rounds) is not int or self.stage_rounds < 1:
            raise ValueError('stage_rounds must be a positive integer')

    def launch_record(self):
        return dict(coefficient_refresh=self.mode, stage_rounds=self.stage_rounds,
                    initial_scale='one fixed block scalar: 1 / mean(initial 2c)',
                    acceptance='stage-start Q and exact training S nonincrease; first rejection stops',
                    initialization='float32 scalar mean of this fitted target matrix')

    def fit_block(self, spec, x, xq, y, seed, threads, rounds, context, smoke):
        """Return query predictions, ordinary UBJSON array, replay timing and fit metadata."""
        if type(rounds) is not int or rounds < 1:
            raise ValueError('rounds must be a positive integer')
        target = np.asarray(y, dtype=np.float64)
        loss = TrainingScore(target)
        if target.shape[1] != spec.block:
            raise ValueError('Target output width differs from GBMSpec block')
        matrix = xgb.DMatrix(x, label=target.astype(np.float32), nthread=threads)
        params = dict(spec.params, seed=seed, device='cpu', nthread=threads,
                      num_target=spec.block, base_score=float(np.float32(target.mean())))
        if params.get('tree_method') != 'hist' or params.get('multi_strategy') != 'multi_output_tree':
            raise ValueError('Stagewise S requires hist multi_output_tree')
        model = xgb.train(params, matrix, num_boost_round=0)
        current = model.inplace_predict(x).astype(np.float64).reshape(target.shape)
        initial = loss.summary(current)
        fixed_scale = float(1. / loss.curvature(current).mean())
        if not np.isfinite(fixed_scale) or fixed_scale <= 0:
            raise ValueError('Invalid fixed objective scale')
        accepted_rounds, proposed_rounds, records = 0, 0, []
        stop_reason = 'proposal_budget_exhausted'
        while proposed_rounds < rounds:
            stage = len(records)
            anchor = current.copy()
            try:
                fixed_weight = loss.curvature(anchor)
            except ZeroTrainingRisk:
                stop_reason = 'zero_risk_lead'
                break
            q_before = loss.surrogate(anchor, anchor, fixed_weight)
            s_before = loss.value(anchor)
            count = min(self.stage_rounds, rounds - proposed_rounds)
            previous = np.frombuffer(bytes(model.save_raw('ubj')), np.uint8)
            stage_params = dict(params, seed=seed + stage)
            calls = 0

            def objective(prediction, _matrix):
                nonlocal calls
                prediction = np.asarray(prediction, dtype=np.float64).reshape(target.shape)
                weight = fixed_weight if self.mode == 'frozen' else loss.curvature(prediction)
                hessian = fixed_scale * weight
                gradient = hessian * loss.error(prediction)
                gradient, hessian = gradient.astype(np.float32), hessian.astype(np.float32)
                if not np.isfinite(gradient).all() or not np.isfinite(hessian).all() or not (hessian > 0).all():
                    raise ValueError('Invalid float32 gradient or Hessian')
                calls += 1
                return gradient, hessian

            try:
                proposal = xgb.train(stage_params, matrix, num_boost_round=count,
                                     obj=objective, xgb_model=model)
            except ZeroTrainingRisk:
                model = load_booster(previous, threads)
                stop_reason = 'zero_risk_lead_during_proposal'
                records.append(dict(stage=stage, seed=seed + stage, accepted=False,
                                    reason=stop_reason, objective_calls=calls,
                                    requested_rounds=count, q_before=q_before, s_before=s_before))
                break
            proposed_rounds += count
            candidate = proposal.inplace_predict(x).astype(np.float64).reshape(target.shape)
            q_after = loss.surrogate(candidate, anchor, fixed_weight)
            s_after = loss.value(candidate)
            accepted = q_after <= q_before and s_after <= s_before
            reason = 'accepted' if accepted else 'surrogate_or_s_increased'
            records.append(dict(stage=stage, seed=seed + stage, proposed_rounds=count,
                                objective_calls=calls, accepted=accepted, reason=reason,
                                q_before=q_before, q_after=q_after, s_before=s_before, s_after=s_after,
                                curvature_mean=float(fixed_weight.mean()),
                                curvature_min=float(fixed_weight.min()), curvature_max=float(fixed_weight.max())))
            if not accepted:
                model = load_booster(previous, threads)
                stop_reason = reason
                break
            model, current = proposal, candidate
            accepted_rounds += count
        buffer = np.frombuffer(bytes(model.save_raw('ubj')), np.uint8)
        forecast = model.inplace_predict(xq)
        began = time.perf_counter()
        replay = load_booster(buffer, threads)
        replayed = replay.inplace_predict(xq)
        seconds = time.perf_counter() - began
        if not np.array_equal(replayed, forecast) or replay.num_boosted_rounds() != accepted_rounds:
            raise ValueError('Stagewise S checkpoint failed exact prediction/round replay')
        return forecast, buffer, seconds, dict(
            replay_max_abs_error=0., mm=dict(
                **self.launch_record(), fixed_block_scale=fixed_scale,
                initial_prediction=params['base_score'], proposal_budget=rounds,
                proposed_rounds=proposed_rounds, accepted_rounds=accepted_rounds,
                accepted_stages=sum(record['accepted'] for record in records),
                stop_reason=stop_reason, initial=initial, final=loss.summary(current), stages=records))

    predict_block = staticmethod(predict_block)
