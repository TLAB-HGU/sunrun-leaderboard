"""Round-8 runner: A+B1 with the objective replaced by two separately trained components (s3_split_rmse_trmse).

Single architecture (CPU XGBoost, official truth enters scoring only):
  - 30 source features, 3 x 24-lead multi_output_tree boosters, 900 rounds, baseline.PARAMS, per component.
  - Component 'mse'  : plain squared error (gradient e, hessian 1); no tail weight, no spectral term.
  - Component 'timse': scorer TIMSE-weighted squared error. w_ih is the TIMSE tail weight of training target
                       y_ih (101 quantiles, side='left', computed once per lead from the training targets),
                       rescaled to mean one over the block's observed cells; gradient w*e, hessian w.
  - Combination: raw = (1 - lambda_h) * raw_mse + lambda_h * raw_timse per lead. lambda_h is chosen from
    0, .05, ..., 1 on the prior models' 3-month calibration holdout (status-0 unfilled training truth,
    h1-72) by the scorer's per-lead S of the clipped combination; ties keep the smaller lambda.
  - Frozen offset: alpha 0.25 x per-horizon mean residual of the combined prior on the last 336
    calibration-holdout rows (baseline.prior_split); component and final clip [1, 3000].
  - B1 short blend: h = 1..6 per-lead convex blend with causal persistence v_lag0, weights chosen on
    the combined prior holdout (status-0 truth) by the scorer's per-lead S; ties keep the larger model weight.
  - Diagnostics (not used for the decision): lambda = 0 (mse only) and lambda = 1 (timse only) forecasts,
    each with its own offset and B1 blend.
Every path except the objective and the combination is copied from research/offset_roll_ablation7_20261105/experiment.py.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import numpy as np
import pandas as pd
import xgboost as xgb
import baseline

EXPERIMENT = 's3_split_rmse_trmse'
SEEDS = (42, 43, 44)
ARCHITECTURE = dict(tail_mode='split_mse_timse', spectral=False, spectral_coefficient=0., offset=True,
                    offset_scope='horizon', offset_window=336, alpha=.25, block_size=24, rounds=900,
                    clip_component=True, clip_final=True, short_blend_leads=6)
POLICY = 'training-only frozen calibration; official labels used only for scoring'
BLOCK, ROUNDS, WINDOW, ALPHA, LEADS = 24, 900, 336, .25, 6
TRACE_EVERY = 100
TRACE_BINS = (0., 400., 550., 700., np.inf)
BLEND_GRID = np.round(np.arange(0., 1.0001, .05), 2)
STATUS0_TRUTH = 'train_status0_y_h1to72.parquet'
COMPONENTS = ('mse', 'timse')
LAMBDA_GRID = BLEND_GRID


def file_identity(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def experiment_name(seed):
    return f'{EXPERIMENT}_s{seed}'


def normalize_config(value):
    """Only the seed varies; the architecture keys, if present, must equal the fixed values."""
    unknown = set(value) - set(ARCHITECTURE) - {'seed', 'experiment'}
    if unknown:
        raise ValueError(f'Unknown experiment configuration keys: {sorted(unknown)}')
    for key, fixed in ARCHITECTURE.items():
        if key in value and value[key] != fixed:
            raise ValueError(f'{key} is fixed at {fixed!r} in {EXPERIMENT}')
    seed = value.get('seed', 42)
    if seed not in SEEDS:
        raise ValueError('seed is outside the preregistered seeds')
    config = dict(ARCHITECTURE, seed=seed)
    if 'experiment' in value:
        config['experiment'] = value['experiment']
    return config


def check_experiment(name, seed):
    if not re.fullmatch(rf'{EXPERIMENT}_s\d+', name) or name != experiment_name(seed):
        raise ValueError(f'--experiment must be {experiment_name(seed)}')


def feature_names():
    return baseline.selected_names(list(baseline.FEATURES))


def expanded_names(names, start, size):
    return list(names['constant']) + [f'{name}@h{h + 1}'
                                    for h in range(start, start + size)
                                    for name in names['varying']]


def design(grid, start, size):
    return np.hstack([grid['C'], grid['E'][:, start:start + size].reshape(
        len(grid['C']), size * grid['E'].shape[2])]).astype(np.float32)


def tail_weights(values):
    """Per-pair TIMSE weights of one lead: TIMSE = sum_i w_i e_i^2 (scorer quantile construction)."""
    n = len(values)
    order = np.argsort(values, kind='stable')
    quantiles = np.quantile(values, np.arange(101) / 100, method='linear')
    positions = np.searchsorted(values[order], quantiles[:-1], side='left')
    widths, span = np.diff(quantiles), float(quantiles[-1] - quantiles[0])
    if not span > 0:
        raise ValueError('constant truth makes TIMSE undefined')
    increments = np.zeros(n + 1)
    np.add.at(increments, positions, widths / (n - positions))
    weights = np.empty(n)
    weights[order] = np.cumsum(increments)[:n] / span
    return weights


def block_tail_weights(target, observed):
    weights = np.zeros(target.shape)
    for h in range(target.shape[1]):
        valid = observed[:, h]
        weights[valid, h] = tail_weights(target[valid, h])
    return weights


def block_terms(error, observed, weights):
    """Per-lead in-sample MSE_h, TIMSE_h, S_h (diagnostic trace only)."""
    squared = np.where(observed, error ** 2, 0.)
    mse = squared.sum(0) / observed.sum(0)
    timse = (weights * squared).sum(0)
    return mse, timse, (mse * timse) ** .25


def component_weights(target, observed, component):
    """Per-cell squared-error weight of one component (mean one over observed cells)."""
    if component == 'mse':
        return observed.astype(np.float64)
    if component != 'timse':
        raise ValueError(f'Unknown component {component!r}')
    weights = block_tail_weights(target, observed)
    return weights / (weights.sum() / observed.sum())


def make_objective(target, component, trace=None):
    if not np.isfinite(target).all():
        raise ValueError('Training targets must be finite')
    observed = np.isfinite(target)
    tail = block_tail_weights(target, observed)
    weights = component_weights(target, observed, component)
    hessian = weights.astype(np.float32)
    clean = np.nan_to_num(target)
    calls = [0]
    if trace is not None:
        bins = np.digitize(target, TRACE_BINS[1:-1])
        trace.append(dict(component=component, lead_weight=weights.sum(0).tolist(),
                          weight_by_speed={f'{TRACE_BINS[b]:.0f}+': float(weights[observed & (bins == b)].mean())
                                           for b in range(len(TRACE_BINS) - 1) if (observed & (bins == b)).any()}))

    def objective(prediction, matrix):
        prediction = np.asarray(prediction).reshape(target.shape)
        error = np.where(observed, prediction - clean, 0.)
        if trace is not None and calls[0] % TRACE_EVERY == 0:  # in-sample diagnostics only
            m, t, s = block_terms(error, observed, tail)
            trace.append(dict(round=calls[0], s_mean=float(s.mean()), mse_mean=float(m.mean()),
                              timse_mean=float(t.mean()), bias=float(error[observed].mean())))
        calls[0] += 1
        return (weights * error).astype(np.float32), hessian
    return objective


def fit_predict(train, query, target, seed, threads, out, period, stage, progress, component, smoke=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    predictions = np.empty((len(query['origins']), baseline.H), np.float64)
    predict_seconds = 0.
    for block, start in enumerate(range(0, baseline.H, BLOCK)):
        began = time.perf_counter()
        x, xq = design(train, start, BLOCK), design(query, start, BLOCK)
        y = target[:, start:start + BLOCK]
        matrix = xgb.DMatrix(x, label=y.astype(np.float32), nthread=threads)
        params = dict(baseline.PARAMS, seed=seed, device='cpu',
                      nthread=threads, num_target=BLOCK, base_score=float(y.mean()))
        trace = []
        model = xgb.train(params, matrix, num_boost_round=2 if smoke else ROUNDS,
                          obj=make_objective(y, component, trace))
        model.save_model(out / f'b{block}.ubj')
        write_json(out / f'b{block}_objective_trace.json', trace)
        inference = time.perf_counter()
        predictions[:, start:start + BLOCK] = model.inplace_predict(xq)
        predict_seconds += time.perf_counter() - inference
        progress['unit'] += 1
        print(json.dumps(dict(period=period, stage=stage, component=component,
                              block=block, horizon=start + BLOCK,
                              fit_seconds=round(time.perf_counter() - began, 3),
                              **progress)), flush=True)
        del model, matrix, x, xq
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite component forecasts')
    return predictions, predict_seconds


def component_values(raw):
    return np.clip(raw, 1., 3000.)


def estimate_offset(prior, truth, *, smoke=False):
    if prior.shape != truth.shape or truth.ndim != 2 or truth.shape[1] != baseline.H:
        raise ValueError('Prior predictions and training truth require a shared 72-lead grid')
    if not np.isfinite(truth).all() or not np.isfinite(prior).all():
        raise ValueError('Nonfinite calibration values')
    if WINDOW > len(truth) and not smoke:
        raise ValueError('Insufficient training calibration rows for frozen window')
    rows = min(WINDOW, len(truth))
    if rows == 0:
        raise ValueError('Empty calibration window')
    residuals = truth[-rows:] - component_values(prior)[-rows:]
    return ALPHA * residuals.mean(0), rows


def apply_offset(raw, offset):
    parent = component_values(raw)
    corrected = np.clip(parent + offset[None, :], 1., 3000.)
    parent = np.clip(parent, 1., 3000.)
    if not np.isfinite(corrected).all() or not np.isfinite(parent).all():
        raise ValueError('Nonfinite final forecasts')
    return corrected, parent


def lead_s(prediction, observed):
    """Scorer per-lead S = sqrt(RMSE*TRMSE) (same quantile/tail-MSE construction as competition.py)."""
    order = np.argsort(observed, kind='stable')
    quantiles = np.quantile(observed, np.arange(101) / 100, method='linear')
    positions = np.searchsorted(observed[order], quantiles[:-1], side='left')
    widths, span = np.diff(quantiles), float(quantiles[-1] - quantiles[0])
    if not span > 0:
        raise ValueError('constant truth makes TIMSE undefined')
    squared = (prediction - observed) ** 2
    suffix = np.cumsum(squared[order][::-1])[::-1]
    timse = float(np.dot(widths, suffix[positions] / (len(observed) - positions)) / span)
    return float(np.sqrt(np.sqrt(squared.mean()) * np.sqrt(timse)))


def origin_values(data, origins, column, *, training, period):
    """One value per origin (h == 1 row) from the run's input df, reindexed to the given origins."""
    filters = [(f'fit_{period}', '==', True)] if training else [('set', '==', 'query'), ('period', '==', period)]
    frame = pd.read_parquet(data, columns=['origin', column], filters=[*filters, ('h', '==', 1)])
    series = pd.Series(frame[column].to_numpy(np.float64), index=pd.DatetimeIndex(pd.to_datetime(frame.origin, utc=True)))
    values = series.reindex(pd.DatetimeIndex(origins, tz='UTC')).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f'{period}: {column} missing for some {"training" if training else "query"} origins')
    return values


def status0_truth(path, origins, leads):
    """Unfilled status-0 training truth (h = 1..leads) for the calibration origins; NaN where unobserved."""
    path = Path(path)
    frame = pd.read_parquet(path, columns=['origin', 'h', 'y'], filters=[('h', '<=', leads)])
    frame['origin'] = pd.to_datetime(frame.origin, utc=True)
    table = frame.pivot(index='origin', columns='h', values='y').reindex(
        index=pd.DatetimeIndex(origins, tz='UTC'), columns=range(1, leads + 1))
    return table.to_numpy(np.float64), file_identity(path)


def fit_short_blend(model, persistence, truth, leads, minimum=100):
    """Per-lead model weight on the calibration holdout; returns weights and the holdout S table."""
    weights, table = [], []
    for h in range(leads):
        valid = np.isfinite(truth[:, h])
        if valid.sum() < minimum:
            raise ValueError(f'h{h + 1}: too few status-0 calibration pairs ({int(valid.sum())})')
        observed = truth[valid, h]
        scores = [lead_s(w * model[valid, h] + (1 - w) * persistence[valid], observed) for w in BLEND_GRID]
        best = min(scores)
        weight = float(max(w for w, s in zip(BLEND_GRID, scores) if s <= best + 1e-12))
        weights.append(weight)
        table.append(dict(lead=h + 1, pairs=int(valid.sum()), weight=weight, s_model=scores[-1],
                          s_persistence=scores[0], s_blend=best))
    return np.array(weights), table


def apply_short_blend(prediction, persistence, weights):
    blended = prediction.copy()
    leads = len(weights)
    blended[:, :leads] = weights[None, :] * prediction[:, :leads] + (1 - weights[None, :]) * persistence[:, None]
    return np.clip(blended, 1., 3000.)


def combine(mse, timse, lambdas):
    return (1 - lambdas[None, :]) * mse + lambdas[None, :] * timse


def fit_lambda(mse, timse, truth, minimum=100):
    """Per-lead TIMSE-component weight on the calibration holdout; returns weights and the holdout S table."""
    lambdas, table = [], []
    for h in range(baseline.H):
        valid = np.isfinite(truth[:, h])
        if valid.sum() < minimum:
            raise ValueError(f'h{h + 1}: too few status-0 calibration pairs ({int(valid.sum())})')
        observed = truth[valid, h]
        scores = [lead_s(component_values((1 - w) * mse[valid, h] + w * timse[valid, h]), observed)
                  for w in LAMBDA_GRID]
        best = min(scores)
        weight = float(min(w for w, s in zip(LAMBDA_GRID, scores) if s <= best + 1e-12))
        lambdas.append(weight)
        table.append(dict(lead=h + 1, pairs=int(valid.sum()), weight=weight, s_mse=scores[0],
                          s_timse=scores[-1], s_combined=best))
    return np.array(lambdas), table


def finish(raw, prior, truth, cal_origins, cal_persistence, cal_truth, query_persistence, smoke):
    """Existing A+B1 tail of the pipeline: frozen offset, clip, B1 h1-6 blend (all fitted on training holdout)."""
    offset, rows = estimate_offset(prior, truth, smoke=smoke)
    prediction, no_offset = apply_offset(raw, offset)
    calibration_model = apply_offset(prior, offset)[0]
    weights, blend_table = fit_short_blend(calibration_model, cal_persistence, cal_truth[:, :LEADS], LEADS,
                                           minimum=10 if smoke else 100)
    return dict(offset=offset, rows=rows, prediction=apply_short_blend(prediction, query_persistence, weights),
                no_offset=no_offset, unblended=prediction, weights=weights, blend_table=blend_table)


def run_period(config, data, out, period, threads=1, smoke=False, identity=None, status0=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model_dir = out / f'{period}_models'
    model_dir.mkdir(exist_ok=True)
    began = time.perf_counter()
    names = feature_names()
    identity = identity or {}
    seed = config['seed']
    train, y = baseline.load_grid(data, period, names, True)
    query, query_truth = baseline.load_grid(data, period, names, False)
    if query_truth is not None:
        raise ValueError('Query loader must not return evaluation targets')
    if len(query['origins']) != baseline.QUERY_COUNTS[period]:
        raise ValueError('Unexpected official query grid')
    if not np.isfinite(y).all():
        raise ValueError('Prepared training targets must be finite')
    reference, calibration, boundary = baseline.prior_split(train['origins'])
    full = np.arange(len(y))
    if smoke:
        reference = reference[np.linspace(0, len(reference) - 1, min(len(reference), 128)).astype(int)]
        calibration = calibration[np.linspace(0, len(calibration) - 1, min(len(calibration), 64)).astype(int)]
        full = np.linspace(0, len(y) - 1, min(len(y), 192)).astype(int)
        query = baseline.subset(query, np.arange(min(len(query['origins']), 8)))
    progress = dict(unit=0, units=2 * len(COMPONENTS) * (baseline.H // BLOCK))
    priors, raws, seconds = {}, {}, 0.
    for name in COMPONENTS:
        priors[name], _ = fit_predict(baseline.subset(train, reference), baseline.subset(train, calibration),
                                      y[reference], seed, threads, model_dir / name / 'prior', period, 'prior',
                                      progress, name, smoke)
        raws[name], spent = fit_predict(baseline.subset(train, full), query, y[full], seed, threads,
                                        model_dir / name / 'full', period, 'full', progress, name, smoke)
        seconds += spent
    truth, cal_origins = y[calibration], train['origins'][calibration].asi8
    state = dict(calibration_rows=len(calibration), reference_rows=len(reference),
                 reference_start=str(train['origins'][reference].min()),
                 reference_end=str(train['origins'][reference].max()),
                 calibration_start=str(train['origins'][calibration].min()),
                 calibration_end=str(train['origins'][calibration].max()),
                 split_boundary=str(boundary), purge_hours=144)
    origins = query['origins'].asi8
    cal_persistence = origin_values(data, cal_origins, 'v_lag0', training=True, period=period)
    query_persistence = origin_values(data, origins, 'v_lag0', training=False, period=period)
    cal_truth, truth_identity = status0_truth(status0, cal_origins, baseline.H)
    latest_origin, cutoff = pd.Timestamp(int(np.max(cal_origins)), tz='UTC'), pd.Timestamp(baseline.CUTOFFS[period], tz='UTC')
    if latest_origin + pd.Timedelta(hours=LEADS) > cutoff - pd.Timedelta(hours=72):
        raise ValueError('Blend calibration truth crosses the training cutoff maturity')
    if latest_origin + pd.Timedelta(hours=baseline.H) > cutoff:
        raise ValueError('Lambda calibration truth crosses the training cutoff')
    lambdas, lambda_table = fit_lambda(priors['mse'], priors['timse'], cal_truth, minimum=10 if smoke else 100)
    prior = combine(priors['mse'], priors['timse'], lambdas)
    raw = combine(raws['mse'], raws['timse'], lambdas)
    print(json.dumps(dict(period=period, stage='combine', component='lambda', unit=1, units=1,
                          lambdas=lambdas.tolist())), flush=True)
    shared = (truth, cal_origins, cal_persistence, cal_truth, query_persistence, smoke)
    final = finish(raw, prior, *shared)
    diagnostics = {name: finish(raws[name], priors[name], *shared) for name in COMPONENTS}
    offset, rows, weights = final['offset'], final['rows'], final['weights']
    prediction, no_offset, unblended = final['prediction'], final['no_offset'], final['unblended']
    state.update(policy=POLICY, window=WINDOW, alpha=ALPHA, scope='horizon', applied=True,
                 offset=offset.tolist(), offset_estimator='mean', offset_condition='none',
                 offset_rows=rows, offset_start=str(pd.Timestamp(cal_origins[-rows], tz='UTC')),
                 offset_end=str(pd.Timestamp(cal_origins[-1], tz='UTC')))
    state.update(combination=dict(grid=LAMBDA_GRID.tolist(), lambdas=lambdas.tolist(), holdout=lambda_table,
                                  truth='status-0 unfilled training y h1-72', truth_identity=truth_identity,
                                  objective='scorer per-lead S of clip((1-l)*mse + l*timse) on calibration holdout',
                                  tie='smaller lambda'))
    np.savez_compressed(model_dir / 'prior_predictions.npz', origins=cal_origins, raw=prior,
                        component=component_values(prior), training_truth=truth,
                        raw_mse=priors['mse'], raw_timse=priors['timse'], lambdas=lambdas)
    state.update(short_blend=dict(leads=LEADS, grid=BLEND_GRID.tolist(), weights=weights.tolist(),
                                  holdout=final['blend_table'], calibration_rows=len(cal_origins),
                                  persistence='v_lag0 (input df, ACE label origin-1h, causal ffill)',
                                  truth='status-0 unfilled training y', truth_identity=truth_identity,
                                  objective='scorer per-lead S on calibration holdout'))
    state.update(diagnostics={name: dict(offset=d['offset'].tolist(), short_blend_weights=d['weights'].tolist())
                              for name, d in diagnostics.items()})
    np.savez_compressed(out / f'{period}_blend_arrays.npz', unblended=unblended, persistence=query_persistence,
                        weights=weights, origins=origins)
    print(json.dumps(dict(period=period, stage='short_blend', component='blend', unit=1, units=1,
                          weights=weights.tolist())), flush=True)
    write_json(model_dir / 'frozen_calibration.json', state)
    component = component_values(raw)
    realized = np.array(np.broadcast_to(offset[None, :], component.shape))
    preclip = component + realized
    np.savez_compressed(out / f'{period}_predictions.npz', pred=prediction, origins=origins)
    np.savez_compressed(out / f'{period}_no_offset_predictions.npz', pred=no_offset, origins=origins)
    np.savez_compressed(out / f'{period}_unblended_predictions.npz', pred=unblended, origins=origins)
    for name, d in diagnostics.items():
        np.savez_compressed(out / f'{period}_{name}_only_predictions.npz', pred=d['prediction'], origins=origins)
    np.savez_compressed(out / f'{period}_raw_predictions.npz', raw=raw, component=component,
                        preclip=preclip, offset=realized, origins=origins,
                        raw_mse=raws['mse'], raw_timse=raws['timse'], lambdas=lambdas)
    columns = {str(start): expanded_names(names, start, BLOCK) for start in range(0, baseline.H, BLOCK)}
    report = dict(experiment=config.get('experiment', out.name), period=period, smoke=smoke,
                  config=config, architecture=ARCHITECTURE, policy=POLICY, seed=seed, features=names,
                  feature_count=sum(map(len, names.values())), expanded_features=columns,
                  expanded_feature_count=len(next(iter(columns.values()))),
                  parameters=dict(baseline.PARAMS, seed=seed, nthread=threads),
                  objective='two components per 24-lead block: plain L2 (mse) and scorer-TIMSE-weighted L2 (timse); combined per lead by holdout lambda; no point4, no spectral term',
                  components=list(COMPONENTS), lambdas=lambdas.tolist(),
                  rounds=2 if smoke else ROUNDS, training_cutoff=baseline.CUTOFFS[period],
                  maturity_hours=144, train_origins=len(full), query_origins=len(prediction),
                  train_start=str(train['origins'][full].min()), train_end=str(train['origins'][full].max()),
                  query_start=str(pd.Timestamp(origins.min(), tz='UTC')),
                  query_end=str(pd.Timestamp(origins.max(), tz='UTC')),
                  frozen_state=state, inference_seconds_per_fold=seconds / len(prediction),
                  end_to_end_seconds=time.perf_counter() - began,
                  clipping=dict(component_cells=int(np.count_nonzero(raw != component)),
                                final_cells=int(np.count_nonzero(preclip != unblended))),
                  identity=identity,
                  versions=dict(python=platform.python_version(), numpy=np.__version__,
                                pandas=pd.__version__, xgboost=xgb.__version__))
    write_json(out / f'{period}_metadata.json', report)
    return report


def prediction_frame(out, suffix='predictions'):
    frames = []
    for period in baseline.CUTOFFS:
        with np.load(Path(out) / f'{period}_{suffix}.npz') as arrays:
            origins = pd.DatetimeIndex(arrays['origins'], tz='UTC')
            pred = arrays['pred']
        if len(origins) != baseline.QUERY_COUNTS[period] or pred.shape != (len(origins), baseline.H):
            raise ValueError(f'{period}: incomplete full prediction grid')
        if origins.has_duplicates or not origins.is_monotonic_increasing or not np.isfinite(pred).all():
            raise ValueError(f'{period}: invalid prediction grid')
        frames.append(pd.DataFrame(dict(origin_last_input_utc=np.repeat(
            origins.strftime('%Y-%m-%dT%H:%M:%SZ'), baseline.H),
            horizon_hours=np.tile(np.arange(1, baseline.H + 1), len(origins)), pred_kms=pred.ravel())))
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(['origin_last_input_utc', 'horizon_hours']).any():
        raise ValueError('Duplicate prediction keys')
    return frame


SCORE_FILES = {'predictions': 'local_score.json', 'no_offset_predictions': 'paired_no_offset_score.json',
               'unblended_predictions': 'paired_unblended_score.json',
               'mse_only_predictions': 'diagnostic_mse_only_score.json',
               'timse_only_predictions': 'diagnostic_timse_only_score.json'}


def score_outputs(out, observations):
    import competition
    out = Path(out)
    for period in baseline.CUTOFFS:
        if json.loads((out / f'{period}_metadata.json').read_text()).get('smoke'):
            raise ValueError('Smoke forecasts cannot be scored as official results')
    frames = {name: prediction_frame(out, name) for name in SCORE_FILES}
    for name, frame in frames.items():
        frame.to_parquet(out / f'{name}.parquet', index=False)
    observed = competition.load_observations(observations)
    context = competition.prepare_context(observed)
    identity = dict(observations=file_identity(observations), observation_rows=len(observed),
                    observation_start=str(observed.timestamp_utc.min()),
                    observation_end=str(observed.timestamp_utc.max()),
                    observation_columns=list(observed.columns),
                    scorer=file_identity(competition.__file__), scorer_version=competition.VERSION)
    scores = {}
    for name, frame in frames.items():
        score = competition.score_competition(frame, context)
        if score['total']['n_replaced']:
            raise ValueError('Scorer replaced forecasts; incomplete ablations are not accepted')
        result = dict(competition=score, **identity)
        write_json(out / SCORE_FILES[name], result)
        if name == 'predictions':
            write_json(out / 'submission.json', dict(result, local_only=True, published=False))
        scores[name] = result
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--status0', type=Path,
                        help=f'status-0 training truth (default: <campaign>/input/{STATUS0_TRUTH})')
    parser.add_argument('--experiment')
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--period', choices=list(baseline.CUTOFFS))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--score-only', action='store_true')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU GBM runs require CUDA_VISIBLE_DEVICES="" before startup')
    if args.threads < 1:
        raise ValueError('threads must be positive')
    config = normalize_config(json.loads(args.config.read_text()))
    config['experiment'] = args.experiment or experiment_name(config['seed'])
    check_experiment(config['experiment'], config['seed'])
    status0 = args.status0 or args.out.resolve().parent.parent / 'input' / STATUS0_TRUTH
    args.out.mkdir(parents=True, exist_ok=True)
    if args.score_only:
        score_outputs(args.out, args.observations)
        return
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
    identity = dict(data=file_identity(args.data), config=file_identity(args.config),
                    experiment=file_identity(__file__), baseline=file_identity(baseline.__file__),
                    status0=file_identity(status0), git_commit=git.stdout.strip())
    provenance = Path(__file__).with_name('provenance.json')
    if provenance.exists():
        identity['campaign_provenance'] = json.loads(provenance.read_text())
    write_json(args.out / 'resolved_config.json', config)
    for period in ([args.period] if args.period else list(baseline.CUTOFFS)):
        run_period(config, args.data, args.out, period, args.threads, args.smoke, identity, status0)
    if not args.smoke and not args.period:
        score_outputs(args.out, args.observations)


if __name__ == '__main__':
    main()
