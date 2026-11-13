"""The 24-lead-block XGBoost runner shared by every GBM board row.

For each seed (42, 43, 44) and period (F1, F2, F3): optionally a prior model on the reference window
(origins + 144 h <= last 3 months) whose calibration-window residuals give the frozen per-lead offset, and
a full model on all training origins; each model is three 24-lead multi-output XGBoost blocks with the
row's custom objective. Query forecast = full model, clip [1, 3000] (+ offset, clip [1, 3000]); board
forecast = fixed equal-weight mean of the three seeds; scored by the official scorer.

An experiment.py declares a GBMSpec (features, objective, calibration, optional block fitter) and calls
main(SPEC, __doc__, HERE). Outputs: see common.output_paths; checkpoints weights/F{n}/s{seed}.npz,
logs/{data,run,s42,s43,s44,mean3}.jsonl and logs/F{n}_s{seed}_metadata.json, results/{s42,s43,s44,mean3}_*.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xgboost as xgb

from . import calibration as cal
from .common import (CUTOFFS, DATA, H, QUERY_COUNTS, SEEDS, file_identity, git_commit, launch, mean3,
                     observations_path, output_paths, prediction_frame, run_logs, score, versions,
                     write_json, write_results)
from .table import PURGE_HOURS, load_grid, load_table, subset

PARAMS = {'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7,
          'min_child_weight': 20, 'reg_lambda': 5, 'objective': 'reg:squarederror',
          'tree_method': 'hist', 'device': 'cpu', 'multi_strategy': 'multi_output_tree',
          'eval_metric': 'rmse', 'seed': 42, 'verbosity': 0}
ROUNDS = 900
BLOCK = 24


@dataclass
class GBMSpec:
    experiment: str                      # submission ID (board row)
    constant: list                       # per-origin features
    varying: list                        # per-lead features
    objective: str                       # objective description for logs and metadata
    component: str                       # component label of the fit records
    make_objective: Callable             # (block target, period context) -> XGBoost objective
    parent_constant: list | None = None  # design order parent constant, leads, added constant (else constant, leads)
    calibrated: bool = True              # prior model + frozen offset; False = full model clipped only
    alpha: float = cal.ALPHA
    offset_window: int = cal.OFFSET_WINDOW
    prepare_period: Callable | None = None   # (table, period) -> context passed to make_objective
    context_name: str | None = None          # metadata key of that context
    keep_observed: bool = False              # keep unfilled y as y_observed in the table
    fit_block: Callable | None = None        # override of the per-block fit (see fit_block below)
    predict_block: Callable | None = None    # override of the per-block inference from a checkpoint
    launch: dict = field(default_factory=dict)
    threads: int = 4
    rounds: int = ROUNDS
    block: int = BLOCK
    params: dict = field(default_factory=lambda: dict(PARAMS))

    def design(self, grid, start, size):
        if self.parent_constant is None:
            return np.hstack([grid['C'], grid['E'][:, start:start + size].reshape(
                len(grid['C']), size * grid['E'].shape[2])]).astype(np.float32)
        base = len(self.parent_constant)
        return np.hstack([grid['C'][:, :base],
                          grid['E'][:, start:start + size].reshape(len(grid['C']), size * grid['E'].shape[2]),
                          grid['C'][:, base:]]).astype(np.float32)

    def feature_names(self, start, size):
        leads = [f'{name}_h{lead}' for lead in range(start + 1, start + size + 1) for name in self.varying]
        if self.parent_constant is None:
            return [*self.constant, *leads]
        base = len(self.parent_constant)
        return [*self.constant[:base], *leads, *self.constant[base:]]


def load_booster(buffer, threads):
    booster = xgb.Booster(params={'nthread': threads, 'device': 'cpu'})
    booster.load_model(bytearray(np.asarray(buffer, np.uint8).tobytes()))
    return booster


def fit_block(spec, x, xq, y, seed, threads, rounds, context, smoke):
    """One block booster; returns the query forecast, its UBJSON bytes, inference seconds and log fields."""
    matrix = xgb.DMatrix(x, label=y.astype(np.float32), nthread=threads)
    params = dict(spec.params, seed=seed, device='cpu', nthread=threads, num_target=spec.block,
                  base_score=float(y.mean()))
    model = xgb.train(params, matrix, num_boost_round=rounds, obj=spec.make_objective(y, context))
    buffer = np.frombuffer(bytes(model.save_raw('ubj')), np.uint8)
    inference = time.perf_counter()
    forecast = model.inplace_predict(xq)
    seconds = time.perf_counter() - inference
    replay = load_booster(buffer, threads)
    replay_error = float(np.max(np.abs(replay.inplace_predict(xq) - forecast)))
    if replay_error != 0. or replay.num_boosted_rounds() != rounds:
        raise ValueError('Saved booster did not exactly reproduce its forecasts/rounds')
    del model, replay, matrix
    return forecast, buffer, seconds, dict(replay_max_abs_error=replay_error)


def predict_block(artifact, xq, threads):
    return load_booster(artifact, threads).inplace_predict(xq)


def fit_predict(spec, train, query, target, context, seed, args, rounds, logs, stream, period, stage, progress):
    """Three 24-lead blocks; returns forecasts, inference seconds and the per-block artifacts."""
    fitter = spec.fit_block or fit_block
    predictions = np.empty((len(query['origins']), H), np.float64)
    blocks, predict_seconds = [], 0.
    for block, start in enumerate(range(0, H, spec.block)):
        began = time.perf_counter()
        x, xq = spec.design(train, start, spec.block), spec.design(query, start, spec.block)
        y = target[:, start:start + spec.block]
        forecast, artifact, seconds, extra = fitter(spec, x, xq, y, seed, args.threads, rounds, context, args.smoke)
        predictions[:, start:start + spec.block] = forecast
        predict_seconds += seconds
        blocks.append(artifact)
        progress['unit'] += 1
        logs(stream, 'fit', period=period, stage=stage, component=spec.component, block=block,
             horizon=start + spec.block, rounds=rounds, train_rows=len(x), query_rows=len(xq),
             expanded_feature_count=x.shape[1], **extra,
             fit_seconds=round(time.perf_counter() - began, 3), **progress)
        del x, xq
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite component forecasts')
    return predictions, predict_seconds, blocks


def save_checkpoint(path, stages, calibration=None):
    """{stage}_b{i} booster bytes ({stage}_b{i}_stage{k} + alphas for continued blocks) and the calibration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays, alphas = {}, {}
    for stage, blocks in stages.items():
        for i, block in enumerate(blocks):
            if isinstance(block, dict):
                arrays[f'{stage}_b{i}'] = block['reference']
                for k, buffer in enumerate(block['stages']):
                    arrays[f'{stage}_b{i}_stage{k}'] = buffer
                alphas[f'{stage}_b{i}'] = block['alphas']
            else:
                arrays[f'{stage}_b{i}'] = block
    if alphas:
        arrays['alphas'] = np.frombuffer(json.dumps(alphas, allow_nan=False).encode(), np.uint8)
    if calibration is not None:
        arrays['calibration'] = np.frombuffer(json.dumps(calibration, allow_nan=False).encode(), np.uint8)
    temporary = path.with_name(path.stem + '.partial.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_checkpoint(path, stage, blocks):
    with np.load(path, allow_pickle=False) as arrays:
        alphas = json.loads(arrays['alphas'].tobytes().decode()) if 'alphas' in arrays.files else None
        artifacts = []
        for i in range(blocks):
            name = f'{stage}_b{i}'
            if alphas is None:
                artifacts.append(arrays[name])
            else:
                artifacts.append(dict(reference=arrays[name], alphas=alphas[name],
                                      stages=[arrays[f'{name}_stage{k}'] for k in range(len(alphas[name]))]))
        state = json.loads(arrays['calibration'].tobytes().decode()) if 'calibration' in arrays.files else None
    return artifacts, state


def train_period(spec, table, period, seed, args, logs):
    stream, began = f's{seed}', time.perf_counter()
    train, y = load_grid(table, period, True, spec.constant, spec.varying)
    query, _ = load_grid(table, period, False, spec.constant, spec.varying)
    if len(query['origins']) != QUERY_COUNTS[period]:
        raise ValueError('Unexpected official query grid')
    if not np.isfinite(y).all():
        raise ValueError('Prepared training targets must be finite')
    if spec.calibrated:
        reference, calibration, boundary = cal.prior_split(train['origins'])
    full = np.arange(len(y))
    if args.smoke:
        if spec.calibrated:
            reference = reference[np.linspace(0, len(reference) - 1, min(len(reference), 128)).astype(int)]
            calibration = calibration[np.linspace(0, len(calibration) - 1, min(len(calibration), 64)).astype(int)]
        full = np.linspace(0, len(y) - 1, min(len(y), 192)).astype(int)
        query = subset(query, np.arange(min(len(query['origins']), 8)))
    rounds = 2 if args.smoke else spec.rounds
    progress = dict(unit=0, units=(2 if spec.calibrated else 1) * (H // spec.block))
    context = spec.prepare_period(table, period) if spec.prepare_period else None
    extra = {spec.context_name: context} if spec.context_name else {}
    logs(stream, 'period_start', period=period, seed=seed, smoke=args.smoke,
         training_cutoff=CUTOFFS[period], maturity_hours=PURGE_HOURS,
         train_origins=len(full), train_start=str(train['origins'][full].min()),
         train_end=str(train['origins'][full].max()), query_origins=len(query['origins']),
         query_start=str(query['origins'].min()), query_end=str(query['origins'].max()), **progress)
    if spec.calibrated:
        prior, _, prior_blocks = fit_predict(spec, subset(train, reference), subset(train, calibration),
                                             y[reference], context, seed, args, rounds, logs, stream,
                                             period, 'prior', progress)
        truth, cal_origins = y[calibration], train['origins'][calibration].asi8
        state = dict(calibration_rows=len(calibration), reference_rows=len(reference),
                     reference_start=str(train['origins'][reference].min()),
                     reference_end=str(train['origins'][reference].max()),
                     calibration_start=str(train['origins'][calibration].min()),
                     calibration_end=str(train['origins'][calibration].max()),
                     split_boundary=str(boundary), purge_hours=PURGE_HOURS)
    raw, seconds, full_blocks = fit_predict(spec, subset(train, full), query, y[full], context, seed, args,
                                            rounds, logs, stream, period, 'full', progress)
    checkpoint = args.weights / period / f's{seed}.npz'
    if spec.calibrated:
        offset, rows = cal.estimate_offset(prior, truth, spec.offset_window, spec.alpha, smoke=args.smoke)
        state.update(policy=cal.POLICY, **extra, window=spec.offset_window, alpha=spec.alpha, scope='horizon',
                     applied=True, offset=offset.tolist(), offset_rows=rows,
                     offset_start=str(pd.Timestamp(cal_origins[-rows], tz='UTC')),
                     offset_end=str(pd.Timestamp(cal_origins[-1], tz='UTC')))
        save_checkpoint(checkpoint, dict(prior=prior_blocks, full=full_blocks), state)
        prediction = cal.apply_offset(raw, offset)
        component = cal.component_values(raw)
        clipping = dict(component_cells=int(np.count_nonzero(raw != component)),
                        final_cells=int(np.count_nonzero(component + offset[None, :] != prediction)))
        calibration_record = dict(frozen_calibration=state)
    else:
        save_checkpoint(checkpoint, dict(full=full_blocks))
        prediction = cal.forecast_values(raw)
        clipping = dict(clipped_cells=int(np.count_nonzero(raw != prediction)))
        calibration_record = dict(calibration='none')
    metadata = dict(experiment=spec.experiment, period=period, seed=seed, smoke=args.smoke, rounds=rounds,
                    block=spec.block, objective=spec.objective, **extra, parameters=dict(spec.params, seed=seed),
                    feature_count=len(spec.constant) + len(spec.varying), constant_features=spec.constant,
                    varying_features=spec.varying,
                    expanded_feature_count=len(spec.constant) + spec.block * len(spec.varying),
                    expanded_features_by_block={str(block): spec.feature_names(start, spec.block)
                                                for block, start in enumerate(range(0, H, spec.block))},
                    training_cutoff=CUTOFFS[period], maturity_hours=PURGE_HOURS,
                    train_origins=len(full), train_start=str(train['origins'][full].min()),
                    train_end=str(train['origins'][full].max()), query_origins=len(query['origins']),
                    query_start=str(query['origins'].min()), query_end=str(query['origins'].max()),
                    inference_seconds_per_fold=seconds / len(prediction),
                    versions=versions(xgboost=xgb.__version__), input_file=file_identity(args.data),
                    policy=cal.POLICY if spec.calibrated else 'no calibration; official labels used only for scoring',
                    git_commit=git_commit(), **calibration_record)
    write_json(args.out / 'logs' / f'{period}_s{seed}_metadata.json', metadata)
    if not args.board:
        # campaign runs (one seed per --out): period completion for the sunrun TUI
        write_json(args.out / f'{period}_metadata.json', metadata)
    logs(stream, 'period_complete', period=period, seed=seed, checkpoint=str(checkpoint),
         **calibration_record, inference_seconds_per_fold=seconds / len(prediction), clipping=clipping,
         end_to_end_seconds=round(time.perf_counter() - began, 3))
    return query['origins'], prediction


def predict_period(spec, table, period, seed, args, logs):
    """Inference only: full-model boosters (and the frozen offset) from weights/F{p}/s{seed}.npz."""
    stream, began = f's{seed}', time.perf_counter()
    query, _ = load_grid(table, period, False, spec.constant, spec.varying)
    checkpoint = args.weights / period / f's{seed}.npz'
    blocks, state = load_checkpoint(checkpoint, 'full', H // spec.block)
    predictor = spec.predict_block or predict_block
    raw = np.empty((len(query['origins']), H), np.float64)
    inference_seconds = 0.
    for block, start in enumerate(range(0, H, spec.block)):
        features = spec.design(query, start, spec.block)
        tick = time.perf_counter()
        raw[:, start:start + spec.block] = predictor(blocks[block], features, args.threads)
        inference_seconds += time.perf_counter() - tick
    if spec.calibrated:
        prediction = cal.apply_offset(raw, np.asarray(state['offset'], np.float64))
    else:
        prediction = cal.forecast_values(raw)
    logs(stream, 'predict_only', period=period, seed=seed, checkpoint=str(checkpoint),
         query_origins=len(prediction), offset=state['offset'] if spec.calibrated else None,
         inference_seconds_per_fold=inference_seconds / len(prediction),
         seconds=round(time.perf_counter() - began, 3))
    return query['origins'], prediction


def main(spec, doc, here):
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', choices=[spec.experiment], default=spec.experiment)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--observations', type=Path, help='default: the leaderboard dataset copy')
    parser.add_argument('--out', type=Path, help='run folder (default: research/result/<experiment>)')
    parser.add_argument('--weights', type=Path,
                        help='checkpoint folder (default: <experiment folder>/weights, or OUT/weights with --out)')
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    parser.add_argument('--threads', type=int, default=spec.threads)
    parser.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--periods', nargs='+', choices=list(CUTOFFS), default=list(CUTOFFS))
    parser.add_argument('--predict-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('GBM runs are CPU only: start with CUDA_VISIBLE_DEVICES=""')
    if args.threads < 1:
        raise ValueError('threads must be positive')
    args.board = output_paths(args, here, spec.experiment)
    if args.smoke and (args.board or args.predict_only):
        raise ValueError('--smoke writes a throwaway run: give a separate --out and train')
    logs = run_logs(args, spec.experiment)
    launch(logs, args, mode='predict_only' if args.predict_only else ('smoke' if args.smoke else 'train'),
           versions=versions(xgboost=xgb.__version__),
           constant_features=spec.constant, varying_features=spec.varying,
           feature_count=len(spec.constant) + len(spec.varying), parameters=spec.params, rounds=spec.rounds,
           block=spec.block, objective=spec.objective, **spec.launch,
           **(dict(alpha=spec.alpha, offset_window=spec.offset_window) if spec.calibrated
              else dict(calibration='none')),
           cutoffs=CUTOFFS, seeds=args.seeds)
    table = load_table(args.data, spec.constant + spec.varying, logs, keep_observed=spec.keep_observed)
    observations = None if args.smoke else observations_path(args.observations)
    frames = {}
    for seed in args.seeds:
        parts = {}
        for period in args.periods:
            run = predict_period if args.predict_only else train_period
            parts[period] = run(spec, table, period, seed, args, logs)
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
