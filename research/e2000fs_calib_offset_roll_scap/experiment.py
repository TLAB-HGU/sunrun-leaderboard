"""e2000fs_calib_offset_roll_scap — A900 board recipe (point4 + spectral) + 180 direct-S continuation trees under a training-MSE cap, mean of seeds 42/43/44

The point4 + spectral e2000fs_calib_offset_roll recipe (git e6dde5e1a; drop_t035_v650 + v_lag24/48/72,
prior model + frozen per-lead offset, 900 rounds per 24-lead block) is trained first. Each of its
boosters is then continued with the training-sample competition score S = mean_h (MSE_h * TIMSE_h)^(1/4)
plus the same 0.1 x robust Fourier term: 6 stages x 30 proposal trees (stage seed = seed + stage),
S coefficients refreshed every round, and each stage accepted with an exact Armijo line search
(c = 1e-4, alpha in 1, 1/2, ..., 1/64, else 0) restricted to the step sizes that keep every lead's raw
training MSE <= 1.01 x the A900 booster's training MSE (exact quadratic feasibility interval). Source
runs: research/loss_mechanism_ablation_20261112/runs/scap_s4{2,3,4}.

This one file runs the whole experiment, from the raw table to the board forecast:
  1. data   data-v2/input/df.parquet -> causal per-lead feature forward fill over the
            train+query union -> training labels forward filled on target_time
            (query truth is never changed and never read before scoring)
  2. train  for each seed (42, 43, 44) and period (F1, F2, F3): a prior model on the
            reference window (origins + 144 h <= last 3 months) and a full model on all
            training origins; each is 3 lead blocks of 24 x (900 A900 rounds + 180 continuation
            proposal rounds), every number computed on training rows only
  3. offset frozen per-lead offset = 0.25 x mean residual of the continued prior model over
            the last 336 calibration origins (training data only)
  4. infer  A900 query forecast + sum of accepted alpha x stage forecast, clip [1, 3000],
            + offset, clip [1, 3000]
  5. mean3  fixed equal-weight mean of the three seed forecasts (no fitted weights)
  6. score  official scorer leaderboard/scorer/competition.py; labels are used only here

Outputs (in this folder unless --out is given):
  weights/F{1,2,3}_s{seed}.npz   one checkpoint per seed x period: {prior,full}_b{0..2} A900
                                 boosters, {prior,full}_b{0..2}_stage{0..5} continuation boosters
                                 (UBJSON bytes), accepted alphas and the frozen calibration
  results/s{seed}_*               predictions.parquet, score.json
  results/mean3_*                 predictions.parquet, score.json (the board forecast)
  logs/*.jsonl                   every step, one JSON object per line

Usage (GBM: CPU only):
  CUDA_VISIBLE_DEVICES= python experiment.py --threads 5
  CUDA_VISIBLE_DEVICES= python experiment.py --predict-only --out DIR   # inference from weights/
  CUDA_VISIBLE_DEVICES= python experiment.py --smoke --out DIR          # 2-round wiring check
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import platform
import socket
import subprocess
import time

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = 'e2000fs_calib_offset_roll_scap'
SEEDS = (42, 43, 44)
H = 72
CUTOFFS = {'F1': '2025-12-29', 'F2': '2026-03-29', 'F3': '2026-06-29'}
QUERY_COUNTS = {'F1': 2089, 'F2': 2113, 'F3': 2113}
METADATA = ('set', 'origin', 'h', 'target_time', 'period', 'fit_F1', 'fit_F2', 'fit_F3', 'y')
PARENT_CONSTANT = ['v_lag0', 'ace_speed_kms_last', 'ace_density_cm3_last', 'ace_density_cm3_m6',
                  'ace_temperature_k_last', 'ace_temperature_k_m24', 'ace_bt_nt_last',
                  'ace_bt_nt_m6', 'ace_by_gsm_nt_m24', 'ace_epam_p_channel3_last',
                  'ace_sis_gt10_flux_last', 'suvi_t035_top2_area',
                  'suvi_t035_meridian15_equatorial_area', 'suvi_t035_meridian30_equatorial_area',
                  'suvi_t055_meridian15_area', 'suvi_t055_meridian30_equatorial_area',
                  'suvi_t055_equatorial_area', 'img_age_h', 'enlil_n_pending', 'enlil_kp']
ADDED_CONSTANT = ['v_lag24', 'v_lag48', 'v_lag72']
CONSTANT = PARENT_CONSTANT + ADDED_CONSTANT
VARYING = ['rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed', 'strip_v450_logmed',
           'strip_v550_dark', 'strip_v750_logmed']
PARAMS = {'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7,
          'min_child_weight': 20, 'reg_lambda': 5, 'objective': 'reg:squarederror',
          'tree_method': 'hist', 'device': 'cpu', 'multi_strategy': 'multi_output_tree',
          'eval_metric': 'rmse', 'seed': 42, 'verbosity': 0}
ROUNDS = 900
BLOCK = 24
TAIL_THRESHOLD, TAIL_WEIGHT = 550., 4.
SPECTRAL_COEFFICIENT, SPECTRAL_SCALE = .1, 50.
PROPOSAL_ROUNDS, STAGE_ROUNDS = 180, 30
STEP_SIZES = (1., .5, .25, .125, .0625, .03125, .015625)
ARMIJO = 1e-4
OBJECTIVE = 'direct_s_spectral_continuation_mse_cap'
MSE_CAP = 1.01
ALPHA = .25
OFFSET_WINDOW = 336
PURGE_HOURS = 144
POLICY = 'training-only frozen calibration; official labels used only for scoring'
KEYS = ['origin_last_input_utc', 'horizon_hours']


class Logs:
    """Append-only JSON-lines logs, one file per stream (run, data, s42, s43, s44, mean3)."""

    def __init__(self, folder, run_id):
        self.folder, self.run_id = Path(folder), run_id
        self.folder.mkdir(parents=True, exist_ok=True)

    def __call__(self, stream, event, **fields):
        record = dict(ts=datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                      run_id=self.run_id, experiment=EXPERIMENT, event=event, **fields)
        line = json.dumps(record, allow_nan=False, default=str)
        with (self.folder / f'{stream}.jsonl').open('a') as handle:
            handle.write(line + '\n')
        print(line, flush=True)


def file_identity(path):
    stat = Path(path).stat()
    return dict(path=str(Path(path).resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


# 1. data -----------------------------------------------------------------------------------

def timeline(origin, horizon):
    """Representative rows, original-row -> unique (origin, h) indices, and h boundaries."""
    if len(origin) == 0:
        raise ValueError('The source table is empty')
    if np.isnat(origin).any() or pd.isna(horizon).any():
        raise ValueError('origin and h must not contain missing values')
    order = np.lexsort((origin, horizon))
    sorted_origin, sorted_h = origin[order], horizon[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = (sorted_origin[1:] != sorted_origin[:-1]) | (sorted_h[1:] != sorted_h[:-1])
    representatives = order[first]
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.cumsum(first, dtype=np.int64) - 1
    unique_h = horizon[representatives]
    boundaries = np.r_[0, np.flatnonzero(unique_h[1:] != unique_h[:-1]) + 1, len(unique_h)]
    return representatives, inverse, boundaries


def fill_feature(values, representatives, inverse, boundaries, name):
    """Validate repeated inputs, then forward-fill independently along each h timeline."""
    unique = values[representatives]
    repeated = unique[inverse]
    equal = (values == repeated) | (np.isnan(values) & np.isnan(repeated))
    if not equal.all():
        bad = int(np.flatnonzero(~equal)[0])
        raise ValueError(f'Conflicting duplicate (origin, h) values in {name!r}, source row {bad}')
    del repeated, equal
    before = int(np.isnan(values).sum())
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        segment = unique[start:stop]
        valid = ~np.isnan(segment)
        previous = np.maximum.accumulate(np.where(valid, np.arange(len(segment)), -1))
        take = previous >= 0
        segment[take] = segment[previous[take]]
    filled = unique[inverse]
    after = int(np.isnan(filled).sum())
    return filled, dict(missing_before=before, missing_after=after, filled=before - after)


def fill_training_labels(metadata):
    """y with training rows filled from the last observed target timestamp; query y unchanged."""
    time_axis = pd.to_datetime(metadata.target_time, utc=True)
    if time_axis.isna().any():
        raise ValueError('target_time must be present')
    timeline_frame = pd.DataFrame({'time': time_axis, 'y': metadata.y.to_numpy()})
    if timeline_frame.groupby('time').y.nunique(dropna=False).gt(1).any():
        raise ValueError('The same target timestamp has inconsistent source labels')
    timeline_frame = timeline_frame.drop_duplicates('time').sort_values('time').set_index('time')
    filled = timeline_frame.y.ffill()
    labels = metadata.y.to_numpy(copy=True)
    training = metadata['set'].eq('train').to_numpy()
    labels[training] = filled.reindex(pd.DatetimeIndex(time_axis[training])).to_numpy()
    original = metadata.y.to_numpy()
    if not np.array_equal(labels[~training], original[~training], equal_nan=True):
        raise RuntimeError('Query truth was modified')
    if not np.array_equal(labels[np.isfinite(original)], original[np.isfinite(original)], equal_nan=True):
        raise RuntimeError('Observed labels were modified')
    return labels


def load_table(path, logs):
    """The model table: metadata plus this experiment's features, causally filled."""
    began = time.perf_counter()
    features = CONSTANT + VARYING
    table = pd.read_parquet(path, columns=[*METADATA, *features])
    origin = pd.DatetimeIndex(table['origin']).asi8.view('datetime64[ns]')
    representatives, inverse, boundaries = timeline(origin, table['h'].to_numpy())
    stats = {}
    for name in features:
        table[name], stats[name] = fill_feature(table[name].to_numpy(), representatives, inverse,
                                                boundaries, name)
    training = table['set'].eq('train').to_numpy()
    missing = int(table['y'][training].isna().sum())
    table['y'] = fill_training_labels(table)
    logs('data', 'table_ready', source=file_identity(path), rows=len(table),
         unique_origin_h_keys=len(representatives), features=features,
         feature_ffill=stats, training_y_missing_before=missing,
         training_y_missing_after=int(np.isnan(table['y'].to_numpy()[training]).sum()),
         policy='feature ffill by h over sorted origin (train+query union, leading NaNs kept); '
                'training y ffill on sorted target_time; query y unchanged',
         seconds=round(time.perf_counter() - began, 3))
    return table


def load_grid(table, period, training):
    """Constant (origin) and varying (origin x 72 leads) arrays for training or query origins."""
    if training:
        mask = table[f'fit_{period}'].to_numpy()
    else:
        mask = (table['set'].eq('query') & table['period'].eq(period)).to_numpy()
    selected = table[mask]
    constant = selected.loc[selected['h'].eq(1).to_numpy(), ['origin', *CONSTANT]].sort_values(
        'origin', kind='stable')
    origins = pd.DatetimeIndex(pd.to_datetime(constant.origin, utc=True))
    if origins.has_duplicates:
        raise ValueError('duplicate input origins')
    columns = ['origin', 'h', *VARYING] + (['y'] if training else [])
    rows = selected[columns].sort_values(['origin', 'h'], kind='stable')
    if len(rows) != len(origins) * H or not np.array_equal(
            rows.h.to_numpy().reshape(-1, H), np.broadcast_to(np.arange(1, H + 1), (len(origins), H))):
        raise ValueError('incomplete origin/horizon grid')
    if not origins.equals(pd.DatetimeIndex(pd.to_datetime(rows.origin.iloc[::H], utc=True))):
        raise ValueError('constant and varying grids disagree')
    if training and not (origins + pd.Timedelta(hours=PURGE_HOURS)
                         <= pd.Timestamp(CUTOFFS[period], tz='UTC')).all():
        raise ValueError('training cutoff violated')
    grid = {'origins': origins, 'C': constant[CONSTANT].to_numpy(np.float32),
            'E': rows[VARYING].to_numpy(np.float32).reshape(len(origins), H, len(VARYING))}
    if np.isinf(grid['C']).any() or np.isinf(grid['E']).any():
        raise ValueError('Infinite feature values are not supported')
    return grid, (rows.y.to_numpy(np.float64).reshape(len(origins), H) if training else None)


def prior_split(origins):
    """Reference rows end 144 h before the last three months; those months calibrate the offset."""
    origins = pd.DatetimeIndex(origins)
    if not origins.is_monotonic_increasing or origins.has_duplicates:
        raise ValueError('Training origins must be unique and sorted')
    boundary = origins.max() - pd.DateOffset(months=3)
    calibration = np.flatnonzero(origins >= boundary)
    reference = np.flatnonzero(origins + pd.Timedelta(hours=PURGE_HOURS) <= boundary)
    if len(reference) < 2 or len(calibration) < 48:
        raise ValueError('Insufficient strictly prior reference/calibration support')
    return reference, calibration, boundary


def subset(grid, indices):
    return {key: value[indices] for key, value in grid.items()}


# 2. model ----------------------------------------------------------------------------------

def design(grid, start, size):
    """Parent constant features, the parent varying features of each lead, then the added
    constant features (column order of the published run)."""
    base = len(PARENT_CONSTANT)
    return np.hstack([grid['C'][:, :base],
                      grid['E'][:, start:start + size].reshape(len(grid['C']), size * grid['E'].shape[2]),
                      grid['C'][:, base:]]).astype(np.float32)


def make_objective(target):
    """Tail-weighted squared error (targets >= 550 km/s weigh 4, normalised per lead) plus a Huber-like
    spectral term on the 24-lead error (Hessian = weight + coefficient)."""
    if not np.isfinite(target).all():
        raise ValueError('Training targets must be finite')
    observed = np.isfinite(target)
    raw = np.where(target >= TAIL_THRESHOLD, TAIL_WEIGHT, 1.) * observed
    weights = raw / (raw.sum(0) / observed.sum(0))

    def objective(prediction, matrix):
        prediction = np.asarray(prediction).reshape(target.shape)
        error = np.where(observed, prediction - np.nan_to_num(target), 0.)
        z = np.fft.fft(error, axis=1, norm='ortho')
        rho = 1 / np.sqrt(1 + abs(z) ** 2 / 50. ** 2)
        gradient = (error * weights + SPECTRAL_COEFFICIENT * np.fft.ifft(
            z * rho, axis=1, norm='ortho').real) * observed
        hessian = (weights + SPECTRAL_COEFFICIENT) * observed
        return gradient.astype(np.float32), hessian.astype(np.float32)
    return objective


def load_booster(buffer, threads):
    booster = xgb.Booster(params={'nthread': threads, 'device': 'cpu'})
    booster.load_model(bytearray(np.asarray(buffer, np.uint8).tobytes()))
    return booster


def component_values(raw):
    return np.clip(raw, 1., 3000.)


def estimate_offset(prior, truth, smoke=False):
    if prior.shape != truth.shape or truth.ndim != 2 or truth.shape[1] != H:
        raise ValueError('Prior predictions and training truth require a shared 72-lead grid')
    if not np.isfinite(truth).all() or not np.isfinite(prior).all():
        raise ValueError('Nonfinite calibration values')
    rows = OFFSET_WINDOW
    if rows > len(truth) and not smoke:
        raise ValueError('Insufficient training calibration rows for frozen window')
    rows = min(rows, len(truth))
    residuals = truth[-rows:] - component_values(prior)[-rows:]
    return ALPHA * residuals.mean(0), rows


def apply_offset(raw, offset):
    corrected = np.clip(component_values(raw) + offset[None, :], 1., 3000.)
    if not np.isfinite(corrected).all():
        raise ValueError('Nonfinite final forecasts')
    return corrected


# 2b. continuation objective ---------------------------------------------------------------

def timse_coefficients(target):
    """Exact official finite-population TIMSE coefficients of one lead, summing to one."""
    target = np.asarray(target, dtype=np.float64)
    if target.ndim != 1 or len(target) < 2 or not np.isfinite(target).all():
        raise ValueError('TIMSE requires a finite nonempty training population')
    if np.ptp(target) == 0:
        raise ValueError('Constant training targets make TIMSE undefined')
    quantiles = np.quantile(target, np.arange(101) / 100, method='linear')
    order = np.argsort(target, kind='stable')
    positions = np.searchsorted(target[order], quantiles[:-1], side='left')
    increments = np.zeros(len(target), dtype=np.float64)
    np.add.at(increments, positions,
              np.diff(quantiles) / np.ptp(target) / (len(target) - positions))
    coefficients = np.empty(len(target), dtype=np.float64)
    coefficients[order] = np.cumsum(increments)
    return coefficients


class TrainingLoss:
    """Fixed-scale training-sample S (sum over the block's leads) plus the robust Fourier penalty."""

    def __init__(self, target):
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

    @staticmethod
    def spectral_value_gradient(error):
        z = np.fft.fft(error, axis=1, norm='ortho')
        norm = np.sqrt(1 + abs(z) ** 2 / SPECTRAL_SCALE ** 2)
        value = SPECTRAL_COEFFICIENT * SPECTRAL_SCALE ** 2 * np.sum(norm - 1)
        gradient = SPECTRAL_COEFFICIENT * np.fft.ifft(z / norm, axis=1, norm='ortho').real
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
        return self.scale * self.coefficient(prediction) + SPECTRAL_COEFFICIENT

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


def choose_step(loss, current, direction, budget):
    """Exact Armijo line search on the training objective over min(grid, feasible upper);
    every accepted step keeps each lead's raw training MSE within the budget."""
    current, direction = np.asarray(current), np.asarray(direction)
    before = loss.value(current)
    if direction.shape != current.shape or not np.isfinite(direction).all():
        raise ValueError('Invalid or nonfinite proposal direction')
    upper = feasible_upper(current - loss.target, direction, budget)
    derivative = float(np.sum(loss.gradient(current) * direction))
    if not np.isfinite(derivative):
        raise ValueError('Nonfinite directional derivative')
    if derivative < 0:
        for alpha in dict.fromkeys(min(step, upper) for step in STEP_SIZES):
            if alpha <= 0:
                continue
            candidate = current + alpha * direction
            after = loss.value(candidate)
            feasible = bool((np.mean((candidate - loss.target) ** 2, axis=0)
                             <= budget + 1e-8 * np.maximum(1., budget)).all())
            if np.isfinite(after) and after < before and after <= before + ARMIJO * alpha * derivative \
                    and feasible:
                return alpha, dict(before=before, after=after, reason='accepted', feasible_upper=upper)
    reason = ('no_descent_direction' if derivative >= 0 else
              'no_positive_feasible_step' if upper == 0 else 'line_search_rejected')
    return 0., dict(before=before, after=before, reason=reason, feasible_upper=upper)


def continue_block(reference, x, xq, y, seed, threads, proposal_rounds):
    """Continue one A900 booster on its training rows. Returns the query forecast (A900 + sum of
    accepted alpha x stage forecast), the stage boosters' UBJSON bytes and the stage records."""
    loss = TrainingLoss(y)
    current = reference.inplace_predict(x).astype(np.float64).reshape(y.shape)
    query = reference.inplace_predict(xq).astype(np.float64).reshape(len(xq), y.shape[1])
    budget = MSE_CAP * np.mean((current - y) ** 2, axis=0)
    settings = dict(PARAMS, device='cpu', nthread=threads, num_target=y.shape[1], base_score=0.)
    matrix = xgb.DMatrix(x, label=y.astype(np.float32), nthread=threads)
    initial = loss.summary(current)
    buffers, stages, used = [], [], 0
    while used < proposal_rounds:
        stage, rounds = len(stages), min(STAGE_ROUNDS, proposal_rounds - used)
        base = current.copy()

        def objective(delta, _matrix, base=base):
            actual = base + np.asarray(delta).reshape(y.shape)
            return loss.gradient(actual).astype(np.float32), loss.curvature(actual).astype(np.float32)

        settings['seed'] = seed + stage
        model = xgb.train(settings, matrix, num_boost_round=rounds, obj=objective)
        direction = model.inplace_predict(x).astype(np.float64).reshape(y.shape)
        alpha, decision = choose_step(loss, current, direction, budget)
        if alpha:
            current += alpha * direction
            query += alpha * model.inplace_predict(xq).astype(np.float64).reshape(len(xq), y.shape[1])
        used += rounds
        buffers.append(np.frombuffer(bytes(model.save_raw('ubj')), np.uint8))
        stages.append(dict(stage=stage, rounds=rounds, seed=settings['seed'], alpha=alpha, **decision,
                           training_mean_s=loss.summary(current)['mean_s'],
                   max_relative_budget_violation=float(np.maximum(
                       (np.mean((current - y) ** 2, axis=0) - budget) / budget, 0.).max()),
                           ))
    final = loss.summary(current)
    record = dict(fixed_block_scale=loss.scale, initial_prediction=loss.initial_prediction,
                  stages=stages, accepted_stages=sum(s['alpha'] > 0 for s in stages),
                  accepted_proposal_rounds=sum(s['rounds'] for s in stages if s['alpha'] > 0),
                  training_mean_s_initial=initial['mean_s'], training_mean_s_final=final['mean_s'],
                  training_mse_initial=initial['mse_by_lead'], training_mse_final=final['mse_by_lead'])
    return query, buffers, record


def block_forecast(reference, stages, alphas, xq, threads):
    """Inference from saved boosters: A900 forecast + sum of accepted alpha x stage forecast."""
    prediction = load_booster(reference, threads).inplace_predict(xq).astype(np.float64).reshape(len(xq), BLOCK)
    for buffer, alpha in zip(stages, alphas):
        if alpha:
            prediction += alpha * load_booster(buffer, threads).inplace_predict(xq).astype(
                np.float64).reshape(prediction.shape)
    return prediction


def fit_predict(train, query, target, seed, threads, rounds, proposal_rounds, logs, stream, period, stage,
                progress):
    """Three 24-lead blocks of A900 + continuation; returns forecasts, inference seconds and, per block,
    the A900 booster bytes, the stage booster bytes and the accepted alphas."""
    predictions = np.empty((len(query['origins']), H), np.float64)
    blocks, predict_seconds = [], 0.
    for block, start in enumerate(range(0, H, BLOCK)):
        began = time.perf_counter()
        x, xq = design(train, start, BLOCK), design(query, start, BLOCK)
        y = target[:, start:start + BLOCK]
        matrix = xgb.DMatrix(x, label=y.astype(np.float32), nthread=threads)
        params = dict(PARAMS, seed=seed, device='cpu', nthread=threads, num_target=BLOCK,
                      base_score=float(y.mean()))
        model = xgb.train(params, matrix, num_boost_round=rounds, obj=make_objective(y))
        reference = np.frombuffer(bytes(model.save_raw('ubj')), np.uint8)
        del model, matrix
        forecast, stages, record = continue_block(load_booster(reference, threads), x, xq, y.astype(np.float64),
                                                  seed, threads, proposal_rounds)
        alphas = [s['alpha'] for s in record['stages']]
        inference = time.perf_counter()
        replay = block_forecast(reference, stages, alphas, xq, threads)
        predict_seconds += time.perf_counter() - inference
        replay_error = float(np.max(np.abs(replay - forecast)))
        if replay_error != 0. or load_booster(reference, threads).num_boosted_rounds() != rounds:
            raise ValueError('Saved boosters did not exactly reproduce their forecasts/rounds')
        predictions[:, start:start + BLOCK] = forecast
        blocks.append(dict(reference=reference, stages=stages, alphas=alphas))
        progress['unit'] += 1
        logs(stream, 'fit', period=period, stage=stage, component=OBJECTIVE, block=block,
             horizon=start + BLOCK, rounds=rounds, proposal_rounds=proposal_rounds,
             train_rows=len(x), query_rows=len(xq), expanded_feature_count=x.shape[1],
             continuation=record, replay_max_abs_error=replay_error,
             fit_seconds=round(time.perf_counter() - began, 3), **progress)
        del x, xq
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite component forecasts')
    return predictions, predict_seconds, blocks


# 3. checkpoints ----------------------------------------------------------------------------

def save_checkpoint(path, prior, full, calibration):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays, alphas = {}, {}
    for stage, blocks in (('prior', prior), ('full', full)):
        for i, block in enumerate(blocks):
            arrays[f'{stage}_b{i}'] = block['reference']
            for k, buffer in enumerate(block['stages']):
                arrays[f'{stage}_b{i}_stage{k}'] = buffer
            alphas[f'{stage}_b{i}'] = block['alphas']
    arrays['alphas'] = np.frombuffer(json.dumps(alphas, allow_nan=False).encode(), np.uint8)
    arrays['calibration'] = np.frombuffer(json.dumps(calibration, allow_nan=False).encode(), np.uint8)
    temporary = path.with_name(path.stem + '.partial.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_checkpoint(path, stage):
    with np.load(path, allow_pickle=False) as arrays:
        alphas = json.loads(arrays['alphas'].tobytes().decode())
        blocks = []
        for i in range(H // BLOCK):
            name = f'{stage}_b{i}'
            blocks.append(dict(reference=arrays[name], alphas=alphas[name],
                               stages=[arrays[f'{name}_stage{k}'] for k in range(len(alphas[name]))]))
        calibration = json.loads(arrays['calibration'].tobytes().decode())
    return blocks, calibration


# 4. run ------------------------------------------------------------------------------------

def train_period(table, period, seed, args, logs):
    stream, began = f's{seed}', time.perf_counter()
    train, y = load_grid(table, period, True)
    query, _ = load_grid(table, period, False)
    if len(query['origins']) != QUERY_COUNTS[period]:
        raise ValueError('Unexpected official query grid')
    if not np.isfinite(y).all():
        raise ValueError('Prepared training targets must be finite')
    reference, calibration, boundary = prior_split(train['origins'])
    full = np.arange(len(y))
    if args.smoke:
        reference = reference[np.linspace(0, len(reference) - 1, min(len(reference), 128)).astype(int)]
        calibration = calibration[np.linspace(0, len(calibration) - 1, min(len(calibration), 64)).astype(int)]
        full = np.linspace(0, len(y) - 1, min(len(y), 192)).astype(int)
        query = subset(query, np.arange(min(len(query['origins']), 8)))
    rounds = 2 if args.smoke else ROUNDS
    proposal_rounds = 2 if args.smoke else PROPOSAL_ROUNDS
    progress = dict(unit=0, units=2 * (H // BLOCK))
    logs(stream, 'period_start', period=period, seed=seed, smoke=args.smoke,
         training_cutoff=CUTOFFS[period], maturity_hours=PURGE_HOURS,
         train_origins=len(full), train_start=str(train['origins'][full].min()),
         train_end=str(train['origins'][full].max()), query_origins=len(query['origins']),
         query_start=str(query['origins'].min()), query_end=str(query['origins'].max()), **progress)
    prior, _, prior_boosters = fit_predict(subset(train, reference), subset(train, calibration),
                                           y[reference], seed, args.threads, rounds, proposal_rounds, logs, stream,
                                           period, 'prior', progress)
    truth, cal_origins = y[calibration], train['origins'][calibration].asi8
    state = dict(calibration_rows=len(calibration), reference_rows=len(reference),
                 reference_start=str(train['origins'][reference].min()),
                 reference_end=str(train['origins'][reference].max()),
                 calibration_start=str(train['origins'][calibration].min()),
                 calibration_end=str(train['origins'][calibration].max()),
                 split_boundary=str(boundary), purge_hours=PURGE_HOURS)
    raw, seconds, full_boosters = fit_predict(subset(train, full), query, y[full], seed,
                                              args.threads, rounds, proposal_rounds, logs, stream, period, 'full',
                                              progress)
    offset, rows = estimate_offset(prior, truth, smoke=args.smoke)
    state.update(policy=POLICY, window=OFFSET_WINDOW, alpha=ALPHA, scope='horizon', applied=True,
                 offset=offset.tolist(), offset_rows=rows,
                 offset_start=str(pd.Timestamp(cal_origins[-rows], tz='UTC')),
                 offset_end=str(pd.Timestamp(cal_origins[-1], tz='UTC')))
    checkpoint = args.out / 'weights' / f'{period}_s{seed}.npz'
    save_checkpoint(checkpoint, prior_boosters, full_boosters, state)
    prediction = apply_offset(raw, offset)
    component = component_values(raw)
    logs(stream, 'period_complete', period=period, seed=seed, checkpoint=str(checkpoint),
         frozen_calibration=state, inference_seconds_per_fold=seconds / len(prediction),
         clipping=dict(component_cells=int(np.count_nonzero(raw != component)),
                       final_cells=int(np.count_nonzero(component + offset[None, :] != prediction))),
         end_to_end_seconds=round(time.perf_counter() - began, 3))
    return query['origins'], prediction


def predict_period(table, period, seed, args, logs):
    """Inference only: full-model boosters, alphas and the frozen offset from weights/F{p}_s{seed}.npz."""
    stream, began = f's{seed}', time.perf_counter()
    query, _ = load_grid(table, period, False)
    checkpoint = args.weights / f'{period}_s{seed}.npz'
    blocks, state = load_checkpoint(checkpoint, 'full')
    raw = np.empty((len(query['origins']), H), np.float64)
    for block, start in enumerate(range(0, H, BLOCK)):
        raw[:, start:start + BLOCK] = block_forecast(blocks[block]['reference'], blocks[block]['stages'],
                                                     blocks[block]['alphas'], design(query, start, BLOCK),
                                                     args.threads)
    prediction = apply_offset(raw, np.asarray(state['offset'], np.float64))
    logs(stream, 'predict_only', period=period, seed=seed, checkpoint=str(checkpoint),
         query_origins=len(prediction), offset=state['offset'],
         seconds=round(time.perf_counter() - began, 3))
    return query['origins'], prediction


def prediction_frame(parts):
    frames = []
    for period, (origins, pred) in parts.items():
        if origins.has_duplicates or not origins.is_monotonic_increasing or not np.isfinite(pred).all():
            raise ValueError(f'{period}: invalid prediction grid')
        frames.append(pd.DataFrame({
            KEYS[0]: np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), H),
            KEYS[1]: np.tile(np.arange(1, H + 1), len(origins)), 'pred_kms': pred.ravel()}))
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(KEYS).any():
        raise ValueError('Duplicate prediction keys')
    return frame


def mean3(frames):
    """Fixed equal-weight mean of the three seed forecasts on identical keys."""
    first = frames[0][KEYS]
    if any(not frame[KEYS].equals(first) for frame in frames):
        raise ValueError('All three seeds must have identical origin/horizon keys')
    matrix = np.stack([frame['pred_kms'].to_numpy(np.float64) for frame in frames])
    output = first.copy()
    output['pred_kms'] = matrix.mean(axis=0)
    return output


def score(frame, observations):
    spec = importlib.util.spec_from_file_location('competition', ROOT / 'leaderboard/scorer/competition.py')
    competition = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(competition)
    observed = competition.load_observations(observations)
    result = competition.score_competition(frame, competition.prepare_context(observed))
    if result['total']['n_replaced']:
        raise ValueError('Scorer replaced forecasts; the forecast grid is incomplete')
    return dict(competition=result, scorer_version=competition.VERSION,
                scorer=file_identity(spec.origin), observations=file_identity(observations))


def observations_path(path):
    if path is not None:
        return path
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download('tlabtlab/sunrun-lb-store', 'competition-observations.parquet',
                                repo_type='dataset'))


def write_results(folder, tag, frame, scores):
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / f'{tag}_predictions.parquet', index=False)
    if scores is not None:
        (folder / f'{tag}_score.json').write_text(json.dumps(scores, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', choices=[EXPERIMENT], default=EXPERIMENT)
    parser.add_argument('--data', type=Path, default=ROOT / 'data-v2/input/df.parquet')
    parser.add_argument('--observations', type=Path, help='default: the leaderboard dataset copy')
    parser.add_argument('--out', type=Path, default=HERE)
    parser.add_argument('--weights', type=Path, help='checkpoint folder for --predict-only (default: OUT/weights)')
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    parser.add_argument('--threads', type=int, default=5)
    parser.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--periods', nargs='+', choices=list(CUTOFFS), default=list(CUTOFFS))
    parser.add_argument('--predict-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('GBM runs are CPU only: start with CUDA_VISIBLE_DEVICES=""')
    if args.threads < 1:
        raise ValueError('threads must be positive')
    args.out = args.out.resolve()
    if args.smoke and (args.out == HERE or args.predict_only):
        raise ValueError('--smoke writes a throwaway run: give a separate --out and train')
    args.weights = (args.weights or args.out / 'weights').resolve()
    logs = Logs(args.out / 'logs', datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=False)
    logs('run', 'launch', mode='predict_only' if args.predict_only else ('smoke' if args.smoke else 'train'),
         host=socket.gethostname(), argv=vars(args), git_commit=git.stdout.strip(),
         versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                       xgboost=xgb.__version__),
         constant_features=CONSTANT, varying_features=VARYING,
         feature_count=len(CONSTANT) + len(VARYING), parameters=PARAMS, rounds=ROUNDS, block=BLOCK,
         tail=dict(threshold=TAIL_THRESHOLD, weight=TAIL_WEIGHT), spectral_coefficient=SPECTRAL_COEFFICIENT,
         continuation=dict(objective=OBJECTIVE, proposal_rounds=PROPOSAL_ROUNDS, stage_rounds=STAGE_ROUNDS,
                           step_sizes=STEP_SIZES, armijo=ARMIJO, mse_cap=MSE_CAP),
         alpha=ALPHA, offset_window=OFFSET_WINDOW, cutoffs=CUTOFFS, seeds=args.seeds)
    table = load_table(args.data, logs)
    observations = None if args.smoke else observations_path(args.observations)
    frames = {}
    for seed in args.seeds:
        parts = {}
        for period in args.periods:
            run = predict_period if args.predict_only else train_period
            parts[period] = run(table, period, seed, args, logs)
        frames[seed] = prediction_frame(parts)
        complete = not args.smoke and args.periods == list(CUTOFFS)
        scores = score(frames[seed], observations) if complete else None
        write_results(args.out / 'results', f's{seed}', frames[seed], scores)
        logs(f's{seed}', 'seed_complete', seed=seed, rows=len(frames[seed]),
             total_s=scores['competition']['total']['s'] if scores else None)
    if sorted(frames) == list(SEEDS):
        ensemble = mean3([frames[seed] for seed in SEEDS])
        scores = score(ensemble, observations) if observations is not None and args.periods == list(CUTOFFS) else None
        write_results(args.out / 'results', 'mean3', ensemble, scores)
        logs('mean3', 'mean3_complete', seeds=list(SEEDS), weights=[1 / 3] * 3, rows=len(ensemble),
             total_s=scores['competition']['total']['s'] if scores else None,
             periods={p: scores['competition'][p.lower()]['s'] for p in CUTOFFS} if scores else None)
    logs('run', 'complete', status='complete')


if __name__ == '__main__':
    main()
