"""A900 + direct-S continuation boosting (e2000fs_calib_offset_roll_scap / _splus).

Each A900 block booster is continued with the training-sample competition score
S = mean_h (MSE_h * TIMSE_h)^(1/4) plus a robust Fourier term, in stages of proposal trees, each stage
accepted with an exact Armijo line search; scap additionally caps every lead's training MSE
(mse_cap x the A900 booster's training MSE).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import xgboost as xgb

from .gbm import PARAMS, load_booster
from .objectives import timse_coefficients


@dataclass(frozen=True)
class Continuation:
    objective: str
    spectral_coefficient: float = .1
    spectral_scale: float = 50.
    proposal_rounds: int = 180
    stage_rounds: int = 30
    step_sizes: tuple = (1., .5, .25, .125, .0625, .03125, .015625)
    armijo: float = 1e-4
    mse_cap: float | None = None

    def launch_record(self):
        record = dict(objective=self.objective, proposal_rounds=self.proposal_rounds,
                      stage_rounds=self.stage_rounds, step_sizes=self.step_sizes, armijo=self.armijo)
        if self.mse_cap is not None:
            record['mse_cap'] = self.mse_cap
        return record

    def fit_block(self, spec, x, xq, y, seed, threads, rounds, context, smoke):
        """A900 booster, then the continuation; returns the query forecast, the block artifact
        (A900 bytes, stage bytes, accepted alphas), inference seconds and log fields."""
        proposal_rounds = 2 if smoke else self.proposal_rounds
        matrix = xgb.DMatrix(x, label=y.astype(np.float32), nthread=threads)
        params = dict(PARAMS, seed=seed, device='cpu', nthread=threads, num_target=spec.block,
                      base_score=float(y.mean()))
        model = xgb.train(params, matrix, num_boost_round=rounds, obj=spec.make_objective(y, context))
        reference = np.frombuffer(bytes(model.save_raw('ubj')), np.uint8)
        del model, matrix
        forecast, stages, record = self.continue_block(load_booster(reference, threads), x, xq,
                                                       y.astype(np.float64), seed, threads, proposal_rounds)
        alphas = [s['alpha'] for s in record['stages']]
        inference = time.perf_counter()
        replay = block_forecast(reference, stages, alphas, xq, threads)
        seconds = time.perf_counter() - inference
        replay_error = float(np.max(np.abs(replay - forecast)))
        if replay_error != 0. or load_booster(reference, threads).num_boosted_rounds() != rounds:
            raise ValueError('Saved boosters did not exactly reproduce their forecasts/rounds')
        return forecast, dict(reference=reference, stages=stages, alphas=alphas), seconds, dict(
            proposal_rounds=proposal_rounds, continuation=record, replay_max_abs_error=replay_error)

    @staticmethod
    def predict_block(artifact, xq, threads):
        return block_forecast(artifact['reference'], artifact['stages'], artifact['alphas'], xq, threads)

    def continue_block(self, reference, x, xq, y, seed, threads, proposal_rounds):
        """Continue one A900 booster on its training rows. Returns the query forecast (A900 + sum of
        accepted alpha x stage forecast), the stage boosters' UBJSON bytes and the stage records."""
        loss = TrainingLoss(y, self.spectral_coefficient, self.spectral_scale)
        current = reference.inplace_predict(x).astype(np.float64).reshape(y.shape)
        query = reference.inplace_predict(xq).astype(np.float64).reshape(len(xq), y.shape[1])
        budget = None if self.mse_cap is None else self.mse_cap * np.mean((current - y) ** 2, axis=0)
        settings = dict(PARAMS, device='cpu', nthread=threads, num_target=y.shape[1], base_score=0.)
        matrix = xgb.DMatrix(x, label=y.astype(np.float32), nthread=threads)
        initial = loss.summary(current)
        buffers, stages, used = [], [], 0
        while used < proposal_rounds:
            stage, rounds = len(stages), min(self.stage_rounds, proposal_rounds - used)
            base = current.copy()

            def objective(delta, _matrix, base=base):
                actual = base + np.asarray(delta).reshape(y.shape)
                return loss.gradient(actual).astype(np.float32), loss.curvature(actual).astype(np.float32)

            settings['seed'] = seed + stage
            model = xgb.train(settings, matrix, num_boost_round=rounds, obj=objective)
            direction = model.inplace_predict(x).astype(np.float64).reshape(y.shape)
            alpha, decision = self.choose_step(loss, current, direction, budget)
            if alpha:
                current += alpha * direction
                query += alpha * model.inplace_predict(xq).astype(np.float64).reshape(len(xq), y.shape[1])
            used += rounds
            buffers.append(np.frombuffer(bytes(model.save_raw('ubj')), np.uint8))
            record = dict(stage=stage, rounds=rounds, seed=settings['seed'], alpha=alpha, **decision,
                          training_mean_s=loss.summary(current)['mean_s'])
            if budget is not None:
                record['max_relative_budget_violation'] = float(np.maximum(
                    (np.mean((current - y) ** 2, axis=0) - budget) / budget, 0.).max())
            stages.append(record)
        final = loss.summary(current)
        record = dict(fixed_block_scale=loss.scale, initial_prediction=loss.initial_prediction,
                      stages=stages, accepted_stages=sum(s['alpha'] > 0 for s in stages),
                      accepted_proposal_rounds=sum(s['rounds'] for s in stages if s['alpha'] > 0),
                      training_mean_s_initial=initial['mean_s'], training_mean_s_final=final['mean_s'],
                      training_mse_initial=initial['mse_by_lead'], training_mse_final=final['mse_by_lead'])
        return query, buffers, record

    def choose_step(self, loss, current, direction, budget):
        """Exact Armijo line search on the training objective; with a budget the grid is restricted to
        min(step, feasible upper) and every accepted step keeps each lead's raw training MSE within it."""
        current, direction = np.asarray(current), np.asarray(direction)
        before = loss.value(current)
        if direction.shape != current.shape or not np.isfinite(direction).all():
            raise ValueError('Invalid or nonfinite proposal direction')
        if budget is None:
            derivative = float(np.sum(loss.gradient(current) * direction))
            if not np.isfinite(derivative):
                raise ValueError('Nonfinite directional derivative')
            if derivative < 0:
                for alpha in self.step_sizes:
                    after = loss.value(current + alpha * direction)
                    if np.isfinite(after) and after < before and after <= before + self.armijo * alpha * derivative:
                        return alpha, dict(before=before, after=after, reason='accepted')
            reason = 'no_descent_direction' if derivative >= 0 else 'line_search_rejected'
            return 0., dict(before=before, after=before, reason=reason)
        upper = feasible_upper(current - loss.target, direction, budget)
        derivative = float(np.sum(loss.gradient(current) * direction))
        if not np.isfinite(derivative):
            raise ValueError('Nonfinite directional derivative')
        if derivative < 0:
            for alpha in dict.fromkeys(min(step, upper) for step in self.step_sizes):
                if alpha <= 0:
                    continue
                candidate = current + alpha * direction
                after = loss.value(candidate)
                feasible = bool((np.mean((candidate - loss.target) ** 2, axis=0)
                                 <= budget + 1e-8 * np.maximum(1., budget)).all())
                if np.isfinite(after) and after < before and after <= before + self.armijo * alpha * derivative \
                        and feasible:
                    return alpha, dict(before=before, after=after, reason='accepted', feasible_upper=upper)
        reason = ('no_descent_direction' if derivative >= 0 else
                  'no_positive_feasible_step' if upper == 0 else 'line_search_rejected')
        return 0., dict(before=before, after=before, reason=reason, feasible_upper=upper)


class TrainingLoss:
    """Fixed-scale training-sample S (sum over the block's leads) plus the robust Fourier penalty."""

    def __init__(self, target, spectral_coefficient, spectral_scale):
        self.spectral_coefficient, self.spectral_scale = spectral_coefficient, spectral_scale
        self.target = np.array(target, dtype=np.float64, copy=True)
        if self.target.ndim != 2 or min(self.target.shape) < 1 or not np.isfinite(self.target).all():
            raise ValueError('Expected a finite training-target matrix')
        self.weights = np.column_stack([timse_coefficients(t) for t in self.target.T])
        self.initial_prediction = float(np.float32(self.target.mean()))
        initial = np.full_like(self.target, self.initial_prediction)
        self.scale = float(1. / self.coefficient(initial).mean())

    def terms(self, prediction):
        prediction = np.asarray(prediction, dtype=np.float64)
        if prediction.shape != self.target.shape or not np.isfinite(prediction).all():
            raise ValueError('Expected finite predictions matching the training targets')
        error = prediction - self.target
        mse = np.mean(error ** 2, axis=0)
        timse = np.sum(self.weights * error ** 2, axis=0)
        if not (np.isfinite(mse).all() and np.isfinite(timse).all()):
            raise ValueError('Nonfinite objective terms')
        return error, mse, timse, (mse * timse) ** .25

    def coefficient(self, prediction):
        """Per-cell majorisation weight of S: d S_h / d e = coefficient * e."""
        _, mse, timse, score = self.terms(prediction)
        if not ((mse > 0) & (timse > 0)).all():
            raise ValueError('MM coefficient requires positive MSE and TIMSE')
        return .5 * score * (1. / (len(self.target) * mse) + self.weights / timse)

    def spectral_value_gradient(self, error):
        z = np.fft.fft(error, axis=1, norm='ortho')
        norm = np.sqrt(1 + abs(z) ** 2 / self.spectral_scale ** 2)
        value = self.spectral_coefficient * self.spectral_scale ** 2 * np.sum(norm - 1)
        gradient = self.spectral_coefficient * np.fft.ifft(z / norm, axis=1, norm='ortho').real
        return float(value), gradient

    def value(self, prediction):
        error, mse, timse, score = self.terms(prediction)
        spectral, _ = self.spectral_value_gradient(error)
        if not ((mse > 0) & (timse > 0)).all():
            raise ValueError('S requires positive MSE and TIMSE')
        return float(self.scale * score.sum() + spectral)

    def gradient(self, prediction):
        error, _, _, _ = self.terms(prediction)
        _, spectral_gradient = self.spectral_value_gradient(error)
        return self.scale * self.coefficient(prediction) * error + spectral_gradient

    def curvature(self, prediction):
        return self.scale * self.coefficient(prediction) + self.spectral_coefficient

    def summary(self, prediction):
        _, mse, timse, score = self.terms(prediction)
        return dict(value=self.value(prediction), mean_s=float(score.mean()),
                    mse_by_lead=mse.tolist(), timse_by_lead=timse.tolist())


def feasible_upper(error, direction, budget):
    """Largest alpha in [0, 1] with mean_rows (error + alpha direction)^2 <= budget for every lead
    (exact quadratic roots; a current point above the budget beyond roundoff is an error)."""
    a = np.mean(direction ** 2, axis=0)
    b = 2 * np.mean(error * direction, axis=0)
    c = np.mean(error ** 2, axis=0) - budget
    if not all(np.isfinite(v).all() for v in (a, b, c)):
        raise ValueError('Nonfinite feasibility coefficients')
    if (c > 1e-8 * np.maximum(1., budget)).any():
        raise ValueError('Current predictor violates training MSE budget')
    c = np.minimum(c, 0.)
    upper = 1.
    for aa, bb, cc in zip(a, b, c):
        if aa == 0:
            bound = -cc / bb if bb > 0 else np.inf
        else:
            root = np.hypot(bb, 2 * np.sqrt(aa) * np.sqrt(-cc))
            # Stable positive root avoids cancellation when b is positive.
            bound = -2 * cc / (bb + root) if bb > 0 else (-bb + root) / (2 * aa)
        upper = min(upper, float(bound))
    return max(0., upper)


def block_forecast(reference, stages, alphas, xq, threads):
    """Inference from saved boosters: A900 forecast + sum of accepted alpha x stage forecast."""
    prediction = load_booster(reference, threads).inplace_predict(xq).astype(np.float64).reshape(len(xq), -1)
    for buffer, alpha in zip(stages, alphas):
        if alpha:
            prediction += alpha * load_booster(buffer, threads).inplace_predict(xq).astype(
                np.float64).reshape(prediction.shape)
    return prediction
