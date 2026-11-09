"""Frozen additive-feature CPU runner; per-run entrypoints lock one configuration."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import numpy as np
import pandas as pd
import xgboost as xgb
import baseline

DEFAULTS = dict(add_features=[], tail_weight=4, spectral=True, offset=True,
                offset_scope='horizon', offset_window=336, alpha=.25, block_size=24,
                clip_component=True, clip_final=True, seed=42)
POLICY = 'training-only frozen calibration; official labels used only for scoring'
INVENTORY_PATH = Path(__file__).with_name('inventory.json')


def file_identity(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_inventory():
    return json.loads(INVENTORY_PATH.read_text())


def feature_names(config, inventory=None):
    names = {kind: list(values) for kind, values in baseline.PARENT_NAMES.items()}
    additions = config['add_features']
    if not isinstance(additions, list) or any(not isinstance(n, str) for n in additions):
        raise ValueError('add_features requires a list of actual source feature names')
    if len(set(additions)) != len(additions):
        raise ValueError('add_features requires distinct source feature names')
    parent = set(names['constant'] + names['varying'])
    if parent.intersection(additions):
        raise ValueError('An added feature already belongs to the parent')
    if set(baseline.LOCKED_REMOVALS).intersection(additions):
        raise ValueError('Locked removed features cannot be restored')
    if not additions:
        return names
    inventory = read_inventory() if inventory is None else inventory
    if inventory['parent_features'] != baseline.PARENT_NAMES:
        raise ValueError('Inventory parent features/order differ from the frozen parent')
    candidates = {row['name']: row for row in inventory['candidates']}
    if len(candidates) != len(inventory['candidates']):
        raise ValueError('Duplicate candidate names in inventory')
    for name in additions:
        row = candidates.get(name)
        if row is None:
            raise ValueError(f'Unknown added feature: {name}')
        if row.get('status') != 'eligible':
            raise ValueError(f'Added feature did not pass the frozen audit: {name}')
        if row['kind'] not in names:
            raise ValueError(f'Invalid constant/varying classification: {name}')
        names[row['kind']].append(name)
    return names


def normalize_config(value, inventory=None):
    unknown = set(value) - set(DEFAULTS) - {'experiment', 'rounds'}
    if unknown:
        raise ValueError(f'Unknown experiment configuration keys: {sorted(unknown)}')
    config = dict(DEFAULTS, **value)
    config['add_features'] = list(config['add_features']) if isinstance(config['add_features'], list) else config['add_features']
    for key, expected in DEFAULTS.items():
        if key not in ('seed', 'add_features') and (type(config[key]) is not type(expected) or config[key] != expected):
            raise ValueError(f'{key} is frozen at {expected!r}')
    if type(config['seed']) is not int or config['seed'] not in (42, 43, 44):
        raise ValueError('seed is outside the preregistered choices')
    if 'rounds' in config and (type(config['rounds']) is not int or config['rounds'] != 900):
        raise ValueError('Training rounds are fixed at 900')
    feature_names(config, inventory)
    return config


def expanded_names(names, start, size):
    base_c = len(baseline.PARENT_NAMES['constant'])
    base_e = len(baseline.PARENT_NAMES['varying'])
    return (list(names['constant'][:base_c]) +
            [f'{name}@h{h + 1}' for h in range(start, start + size)
             for name in names['varying'][:base_e]] +
            list(names['constant'][base_c:]) +
            [f'{name}@h{h + 1}' for h in range(start, start + size)
             for name in names['varying'][base_e:]])


def design(grid, start, size):
    base_c = len(baseline.PARENT_NAMES['constant'])
    base_e = len(baseline.PARENT_NAMES['varying'])
    n = len(grid['C'])
    if grid['C'].shape[1] < base_c or grid['E'].shape[2] < base_e:
        raise ValueError('Design is missing retained parent features')
    parent = [grid['C'][:, :base_c],
              grid['E'][:, start:start + size, :base_e].reshape(n, size * base_e)]
    extra = [grid['C'][:, base_c:],
             grid['E'][:, start:start + size, base_e:].reshape(n, size * (grid['E'].shape[2] - base_e))]
    return np.hstack(parent + extra).astype(np.float32)


def make_objective(target, config):
    if not np.isfinite(target).all():
        raise ValueError('Training targets must be finite')
    observed = np.isfinite(target)
    raw = np.where(target >= 550., float(config['tail_weight']), 1.) * observed
    weights = raw / (raw.sum(0) / observed.sum(0))

    def objective(prediction, matrix):
        prediction = np.asarray(prediction).reshape(target.shape)
        error = np.where(observed, prediction - np.nan_to_num(target), 0.)
        z = np.fft.fft(error, axis=1, norm='ortho')
        rho = 1 / np.sqrt(1 + abs(z) ** 2 / 50. ** 2)
        gradient = (error * weights + .1 * np.fft.ifft(
            z * rho, axis=1, norm='ortho').real) * observed
        hessian = (weights + .1) * observed
        return gradient.astype(np.float32), hessian.astype(np.float32)
    return objective


def fit_predict(train, query, target, config, threads, out, period, stage,
                progress, smoke=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    predictions = np.empty((len(query['origins']), baseline.H), np.float64)
    size = config['block_size']
    predict_seconds = 0.
    for block, start in enumerate(range(0, baseline.H, size)):
        began = time.perf_counter()
        x, xq = design(train, start, size), design(query, start, size)
        y = target[:, start:start + size]
        matrix = xgb.DMatrix(x, label=y.astype(np.float32), nthread=threads)
        params = dict(baseline.PARAMS, seed=config['seed'], device='cpu',
                      nthread=threads, num_target=size, base_score=float(y.mean()))
        model = xgb.train(params, matrix, num_boost_round=2 if smoke else 900,
                          obj=make_objective(y, config))
        model.save_model(out / f'b{block}.ubj')
        inference = time.perf_counter()
        predictions[:, start:start + size] = model.inplace_predict(xq)
        predict_seconds += time.perf_counter() - inference
        replay = xgb.Booster(params={'nthread': threads, 'device': 'cpu'})
        replay.load_model(out / f'b{block}.ubj')
        replay_error = float(np.max(np.abs(
            replay.inplace_predict(xq) - predictions[:, start:start + size])))
        if replay_error != 0. or replay.num_boosted_rounds() != (2 if smoke else 900):
            raise ValueError('Saved model did not exactly reproduce its forecasts/rounds')
        write_json(out / f'b{block}.replay.json', dict(
            status='passed', max_abs_error=replay_error,
            rounds=replay.num_boosted_rounds(), rows=len(xq),
            expanded_feature_count=xq.shape[1]))
        progress['unit'] += 1
        print(json.dumps(dict(period=period, stage=stage, component='spectral',
                              block=block, horizon=start + size,
                              fit_seconds=round(time.perf_counter() - began, 3),
                              **progress)), flush=True)
        del model, replay, matrix, x, xq
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite component forecasts')
    return predictions, predict_seconds


def component_values(raw, config):
    return np.clip(raw, 1., 3000.)


def estimate_offset(prior, truth, config, *, smoke=False):
    if prior.shape != truth.shape or truth.ndim != 2 or truth.shape[1] != baseline.H:
        raise ValueError('Prior predictions and training truth require a shared 72-lead grid')
    if not np.isfinite(truth).all() or not np.isfinite(prior).all():
        raise ValueError('Nonfinite calibration values')
    rows = config['offset_window']
    if rows > len(truth) and not smoke:
        raise ValueError('Insufficient training calibration rows for frozen window')
    rows = min(rows, len(truth))
    if rows == 0:
        raise ValueError('Empty calibration window')
    residuals = truth[-rows:] - component_values(prior, config)[-rows:]
    offset = config['alpha'] * residuals.mean(0)
    return offset, rows


def apply_offset(raw, offset, config):
    parent = component_values(raw, config)
    corrected = np.clip(parent + offset[None, :], 1., 3000.)
    parent = np.clip(parent, 1., 3000.)
    if not np.isfinite(corrected).all() or not np.isfinite(parent).all():
        raise ValueError('Nonfinite final forecasts')
    return corrected, parent


def run_period(config, data, out, period, threads=1, smoke=False, identity=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model_dir = out / f'{period}_models'
    model_dir.mkdir(exist_ok=True)
    began = time.perf_counter()
    config = normalize_config(config)
    names = feature_names(config)
    identity = identity or {}
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
    progress = dict(unit=0, units=2 * (baseline.H // config['block_size']))
    prior, _ = fit_predict(baseline.subset(train, reference),
                           baseline.subset(train, calibration), y[reference],
                           config, threads, model_dir / 'prior', period, 'prior', progress, smoke)
    truth, cal_origins = y[calibration], train['origins'][calibration].asi8
    state = dict(calibration_rows=len(calibration), reference_rows=len(reference),
                 reference_start=str(train['origins'][reference].min()),
                 reference_end=str(train['origins'][reference].max()),
                 calibration_start=str(train['origins'][calibration].min()),
                 calibration_end=str(train['origins'][calibration].max()),
                 split_boundary=str(boundary), purge_hours=144)
    raw, seconds = fit_predict(baseline.subset(train, full), query, y[full], config,
                               threads, model_dir / 'full', period, 'full', progress, smoke)
    origins = query['origins'].asi8
    train_rows = len(full)
    train_start, train_end = str(train['origins'][full].min()), str(train['origins'][full].max())
    offset, rows = estimate_offset(prior, truth, config, smoke=smoke)
    state.update(policy=POLICY, window=config['offset_window'], alpha=config['alpha'],
                 scope=config['offset_scope'], applied=config['offset'], offset=offset.tolist(),
                 offset_rows=rows, offset_start=str(pd.Timestamp(cal_origins[-rows], tz='UTC')),
                 offset_end=str(pd.Timestamp(cal_origins[-1], tz='UTC')))
    write_json(model_dir / 'frozen_calibration.json', state)
    np.savez_compressed(model_dir / 'prior_predictions.npz', origins=cal_origins, raw=prior,
                        spectral=component_values(prior, config), training_truth=truth)
    prediction, no_offset = apply_offset(raw, offset, config)
    component = component_values(raw, config)
    preclip = component + offset[None, :]
    np.savez_compressed(out / f'{period}_predictions.npz', pred=prediction, origins=origins)
    np.savez_compressed(out / f'{period}_no_offset_predictions.npz', pred=no_offset, origins=origins)
    np.savez_compressed(out / f'{period}_raw_predictions.npz', raw=raw, component=component,
                        preclip=preclip, offset=offset, origins=origins)
    columns = {str(start): expanded_names(names, start, config['block_size'])
               for start in range(0, baseline.H, config['block_size'])}
    report = dict(experiment=config.get('experiment', out.name), period=period, smoke=smoke,
                  config=config, policy=POLICY, seed=config['seed'], features=names,
                  feature_count=sum(map(len, names.values())),
                  parent_features=baseline.PARENT_NAMES, added_features=config['add_features'],
                  expanded_features=columns,
                  expanded_feature_count=len(next(iter(columns.values()))),
                  parameters=dict(baseline.PARAMS, seed=config['seed'], nthread=threads),
                  rounds=2 if smoke else 900, training_cutoff=baseline.CUTOFFS[period],
                  maturity_hours=144, train_origins=train_rows, query_origins=len(prediction),
                  train_start=train_start, train_end=train_end,
                  query_start=str(pd.Timestamp(origins.min(), tz='UTC')),
                  query_end=str(pd.Timestamp(origins.max(), tz='UTC')),
                  frozen_state=state, inference_seconds_per_fold=seconds / len(prediction),
                  end_to_end_seconds=time.perf_counter() - began,
                  clipping=dict(component_cells=int(np.count_nonzero(raw != component)),
                                final_cells=int(np.count_nonzero(preclip != prediction))),
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


def score_outputs(out, observations):
    import competition
    out = Path(out)
    for period in baseline.CUTOFFS:
        if json.loads((out / f'{period}_metadata.json').read_text()).get('smoke'):
            raise ValueError('Smoke forecasts cannot be scored as official results')
    frames = {name: prediction_frame(out, suffix) for name, suffix in
              [('predictions', 'predictions'), ('no_offset_predictions', 'no_offset_predictions')]}
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
        filename = 'local_score.json' if name == 'predictions' else 'paired_no_offset_score.json'
        write_json(out / filename, result)
        if name == 'predictions':
            write_json(out / 'submission.json', dict(result, local_only=True, published=False))
        scores[name] = result
    return scores


def main(argv=None, *, locked_experiment=None, locked_config=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=locked_config is None)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--experiment', choices=(locked_experiment,) if locked_experiment else None,
                        default=locked_experiment)
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--period', choices=list(baseline.CUTOFFS))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--score-only', action='store_true')
    args = parser.parse_args(argv)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU GBM runs require CUDA_VISIBLE_DEVICES="" before startup')
    if args.threads < 1:
        raise ValueError('threads must be positive')
    supplied = json.loads(args.config.read_text()) if args.config else locked_config
    config = normalize_config(supplied)
    if locked_config is not None:
        fixed = normalize_config(locked_config)
        for candidate in (config, fixed):
            if candidate.get('experiment', locked_experiment) != locked_experiment:
                raise ValueError('Configuration experiment differs from locked experiment')
        if {k: v for k, v in config.items() if k != 'experiment'} != {
                k: v for k, v in fixed.items() if k != 'experiment'}:
            raise ValueError('Configuration differs from locked experiment configuration')
    if args.experiment:
        config['experiment'] = args.experiment
    args.out.mkdir(parents=True, exist_ok=True)
    if args.score_only:
        score_outputs(args.out, args.observations)
        return
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
    identity = dict(data=file_identity(args.data), config=file_identity(args.config) if args.config else dict(locked=config),
                    inventory=file_identity(INVENTORY_PATH) if INVENTORY_PATH.exists() else None,
                    experiment=file_identity(__file__), baseline=file_identity(baseline.__file__),
                    git_commit=git.stdout.strip())
    provenance = Path(__file__).with_name('provenance.json')
    if provenance.exists():
        identity['campaign_provenance'] = json.loads(provenance.read_text())
    write_json(args.out / 'resolved_config.json', config)
    for period in ([args.period] if args.period else list(baseline.CUTOFFS)):
        run_period(config, args.data, args.out, period, args.threads, args.smoke, identity)
    if not args.smoke and not args.period:
        score_outputs(args.out, args.observations)
        write_json(args.out / 'complete.json', dict(status='complete', publish='not_required',
                   publish_reason='Preregistered follow-up ablation; report only, no publication.', experiment=config.get('experiment', args.out.name)))


if __name__ == '__main__':
    main()
