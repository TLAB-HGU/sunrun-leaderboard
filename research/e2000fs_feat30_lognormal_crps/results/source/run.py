"""Frozen current-HPO LightGBM distribution experiments, with local evaluation."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
import traceback

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HPO = ROOT / 'analysis/e2000fs_dirmo_24_feat30_lgbm_HPO'
FOLLOW = ROOT / 'analysis/e2000fs_feat30_followups'
METHODS = ('controls', 'lognormal_nll', 'lognormal_crps', 'smooth_wcrps', 'tailcal_crps')


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def array_save(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def run_directory(run_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('run-id must contain only letters, digits, underscore or hyphen')
    return HERE / 'runs' / run_id


def current_hpo():
    """Read only COMPLETE trials, including unfinished studies' current winners."""
    winners, counts = [], {}
    for index in range(1, 4):
        database = HPO / 'runs/main' / f'study{index}' / 'optuna.sqlite3'
        with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as conn:
            counts[str(index)] = dict(conn.execute('SELECT state, COUNT(*) FROM trials GROUP BY state'))
            rows = conn.execute("SELECT t.trial_id,t.number,v.value,t.datetime_complete "
                                "FROM trials t JOIN trial_values v ON v.trial_id=t.trial_id "
                                "WHERE t.state='COMPLETE' AND v.objective=0 AND v.value IS NOT NULL "
                                "ORDER BY v.value,t.number LIMIT 1").fetchall()
            if not rows:
                continue
            tid, number, value, completed_at = rows[0]
            params = {}
            for name, val, definition in conn.execute(
                    'SELECT param_name,param_value,distribution_json FROM trial_params WHERE trial_id=?', (tid,)):
                definition = json.loads(definition)
                params[name] = int(val) if definition['name'].startswith('Int') else val
            result_path = database.parent / 'trials' / f'{number:04}' / 'result.json'
            recorded = read(result_path)
            if recorded['params'] != params or recorded['value'] != value:
                raise ValueError('SQLite winner differs from its completed result receipt')
            winners.append(dict(study=index, trial=number, value=value, params=params,
                                completed_at=completed_at, result_path=str(result_path),
                                result_sha256=digest(result_path)))
    if not winners:
        raise RuntimeError('No completed HPO trial')
    best = min(winners, key=lambda w: (w['value'], w['study'], w['trial']))
    return dict(best=best, study_winners=winners, counts=counts,
                snapshot_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                policy='minimum internal MSE among currently COMPLETE trials; immutable snapshot, not final HPO winner')


def prepare(run_id, smoke=False):
    directory = run_directory(run_id)
    if (directory / 'manifest.json').exists():
        validate_manifest(directory, smoke)
        return read(directory / 'manifest.json')
    hpo = imported('dist_frozen_hpo', HPO / 'run.py')
    parent, _, origins, y, blocks = hpo.load_data()
    fit, val = hpo.temporal_split(origins)
    snapshot = current_hpo()
    parameters = hpo.full_params(snapshot['best']['params'])
    parameters['n_jobs'] = 2
    if smoke:
        parameters['n_estimators'] = 3
    source_paths = [HERE / 'run.py', HERE / 'methods.py', HERE / 'launch.py', HPO / 'run.py',
                    hpo.BASE / 'run.py', FOLLOW / 'evaluation.py']
    manifest = dict(experiment='e2000fs_feat30_lgbm_distribution', smoke=smoke,
                    hpo=snapshot, params=parameters, parent=parent,
                    methods=list(METHODS), selected_groups=parent['selected'],
                    source_sha256={str(p): digest(p) for p in source_paths},
                    inherited_hashes=parent['hashes'],
                    temporal=dict(fit_count=len(fit), validation_count=len(val),
                                  fit_last_target=str((origins[fit] + pd.Timedelta(hours=72)).max()),
                                  validation_first_origin=str(origins[val].min())),
                    target='absolute positive ACE speed km/s; no residual addition',
                    point_forecast='lognormal predictive mean, clipped to the common [1,3000] contract; raw mean also retained',
                    distribution_contract='continuous lognormal marginal per horizon; not a 72h joint distribution',
                    controls='unweighted L2 and factor4 block-event-weighted L2, both same frozen HPO',
                    method_settings=dict(smooth_wcrps=dict(floor=1., extra_weight=1., center_kms=550., width_kms=100.),
                                         tailcal_crps=dict(strength=1., note='smooth batch CDF calibration adaptation, not exact paper TMCB')),
                    selection='all fixed arms frozen here; no official-based candidate or hyperparameter selection',
                    comparisons='internal paired low MSE versus L2 and high MSE versus extreme L2; historical XGBoost reference separate',
                    posthoc_official_exposed=True,
                    publication='local evaluation only; no upload or parent promotion',
                    versions=dict(lightgbm=lgb.__version__, numpy=np.__version__, pandas=pd.__version__),
                    shape=dict(target=list(y.shape), blocks=[list(x.shape) for x in blocks]))
    write(directory / 'manifest.json', manifest)
    write(directory / 'hpo_snapshot.json', snapshot)
    print(json.dumps(dict(status='prepared', run_id=run_id, best=snapshot['best']), indent=2), flush=True)
    return manifest


def validate_manifest(directory, smoke):
    manifest = read(directory / 'manifest.json')
    if manifest['smoke'] != smoke:
        raise ValueError('Smoke/full identity mismatch')
    for path, expected in {**manifest['inherited_hashes'], **manifest['source_sha256']}.items():
        if not Path(path).is_file() or digest(path) != expected:
            raise ValueError(f'Frozen input/source changed: {path}')
    best = manifest['hpo']['best']
    if digest(best['result_path']) != best['result_sha256']:
        raise ValueError('Selected completed HPO trial receipt changed')
    return manifest


def status(directory, state, **kwargs):
    value = dict(status=state, pid=os.getpid(), utc=pd.Timestamp.now(tz='UTC').isoformat(), **kwargs)
    write(directory / 'status.json', value)
    print(json.dumps(value), flush=True)


def point_metrics(evaluation, y, prediction, origins):
    result = evaluation.metrics(y, prediction, origins)
    low = y < 550
    result['mse_lt550'] = float(np.mean((prediction[low] - y[low]) ** 2)) if low.any() else None
    result['n_pairs_lt550'] = int(low.sum())
    result['bias_lt550'] = float(np.mean(prediction[low] - y[low])) if low.any() else None
    return result


def block_extreme_weights(y, block):
    target = y[:, block * 24:(block + 1) * 24]
    event = (target.max(axis=1) >= 550) | ((target[:, 6:] - target[:, :-6]).max(axis=1) >= 100)
    weight = np.where(event, 4., 1.)
    return weight / weight.mean()


def controls_fit(x, y, xv, yv, params, weights, folder):
    history = {}
    model = lgb.LGBMRegressor(**params)
    model.fit(x, y, sample_weight=weights, eval_set=[(x, y), (xv, yv)],
              eval_names=['train', 'validation'], eval_metric='l2',
              callbacks=[lgb.record_evaluation(history)])
    temporary = folder / 'model.txt.tmp'
    model.booster_.save_model(str(temporary))
    restored = lgb.Booster(model_file=str(temporary))
    np.testing.assert_allclose(model.predict(xv), restored.predict(xv, num_threads=2), rtol=0, atol=0)
    temporary.replace(folder / 'model.txt')
    pd.DataFrame(dict(round=np.arange(1, len(history['train']['l2']) + 1),
                      train_mse=history['train']['l2'], validation_mse=history['validation']['l2'])).to_csv(folder / 'loss.csv', index=False)
    return restored


def train_stage(directory, method, stage, fit, val, origins, y, blocks, query, query_origins,
                params, manifest_hash, smoke):
    from methods import fit_distribution, load_distribution, predict_distribution
    arms = ('l2', 'extreme4') if method == 'controls' else (method,)
    outputs = {}
    for arm in arms:
        pred = np.empty((len(query_origins), 72))
        raw_mean, mu, sigma = (np.empty_like(pred) for _ in range(3))
        for horizon in range(72):
            root = directory / stage / arm / f'h{horizon + 1:02}'
            root.mkdir(parents=True, exist_ok=True)
            checkpoint = root / 'checkpoint.json'
            context = dict(horizon=horizon + 1, stage=stage, arm=arm, method=method,
                           fit_count=len(fit), fit_indices_sha256=hashlib.sha256(fit.tobytes()).hexdigest(),
                           last_target=str((origins[fit] + pd.Timedelta(hours=72)).max()))
            if checkpoint.exists():
                receipt = read(checkpoint)
                if receipt['manifest_sha256'] != manifest_hash:
                    raise ValueError('Checkpoint belongs to different experiment identity')
                if any(receipt.get(key) != value for key, value in context.items()):
                    raise ValueError('Checkpoint horizon/stage/training context mismatch')
                folder = root / receipt['attempt']
                for name, expected in receipt['sha256'].items():
                    if digest(folder / name) != expected:
                        raise ValueError(f'Checkpoint changed: {folder / name}')
            else:
                attempt = f'attempt{len(list(root.glob("attempt*"))):04}'
                folder = root / attempt
                folder.mkdir()
                block = horizon // 24
                status(directory, 'running', method=method, stage=stage, arm=arm, horizon=horizon + 1,
                       total_horizons=72, fit_rows=len(fit), validation_rows=len(val))
                if method == 'controls':
                    weights = block_extreme_weights(y[fit], block) if arm == 'extreme4' else None
                    controls_fit(blocks[block][fit], y[fit, horizon], blocks[block][val], y[val, horizon],
                                 params, weights, folder)
                else:
                    fit_distribution(blocks[block][fit], y[fit, horizon], blocks[block][val], y[val, horizon],
                                     params, method, folder, smoke=smoke)
                files = [p for p in folder.iterdir() if p.is_file() and not p.name.endswith('.tmp')]
                write(checkpoint, dict(manifest_sha256=manifest_hash, attempt=attempt,
                                       sha256={p.name: digest(p) for p in files},
                                       **context, validation_is_held_out=stage == 'internal'))
            if method == 'controls':
                prediction = lgb.Booster(model_file=str(folder / 'model.txt')).predict(query[horizon // 24], num_threads=2)
                pred[:, horizon] = np.clip(prediction, 1., 3000.)
            else:
                values = predict_distribution(load_distribution(folder), query[horizon // 24])
                raw_mean[:, horizon], mu[:, horizon], sigma[:, horizon] = values['mean'], values['mu'], values['sigma']
                pred[:, horizon] = np.clip(values['mean'], 1., 3000.)
            if not np.isfinite(pred[:, horizon]).all():
                raise ValueError('Nonfinite predictions')
        output = dict(pred=pred, origins=query_origins.asi8)
        if method != 'controls':
            output.update(raw_mean=raw_mean, mu=mu, sigma=sigma)
        array_save(directory / f'{stage}_{arm}.npz', **output)
        outputs[arm] = output
    return outputs


def distribution_metrics(y, output):
    from methods import distribution_scores
    values = distribution_scores(y.reshape(-1), output['mu'].reshape(-1), output['sigma'].reshape(-1))
    raw = output['raw_mean']
    values['raw_mean_mse'] = float(np.mean((raw - y) ** 2))
    values['clipped_mean_pairs'] = int(np.count_nonzero(raw != output['pred']))
    return values


def aggregate(directory):
    with (directory / 'summary.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        results = {m: read(directory / m / 'results.json') for m in METHODS if (directory / m / 'results.json').exists()}
        summary = dict(completed_methods=list(results), expected_methods=list(METHODS), results=results,
                       comparisons={}, posthoc_official_exposed=True)
        if 'controls' in results:
            for method, result in results.items():
                if method == 'controls':
                    continue
                comparisons = {}
                for stage in ('internal', 'combined'):
                    candidate = result[method][stage]
                    base = results['controls']['l2'][stage]
                    extreme = results['controls']['extreme4'][stage]
                    lo = candidate['mse_lt550'] - base['mse_lt550']
                    hi = candidate['mse_ge550'] - extreme['mse_ge550']
                    comparisons[stage] = dict(low_mse_difference_vs_l2=lo, high_mse_difference_vs_extreme4=hi,
                                               satisfies_both_point_estimates=lo <= 0 and hi <= 0,
                                               note='point estimates only; not a statistical noninferiority claim')
                summary['comparisons'][method] = comparisons
        write(directory / 'summary.json', summary)
        lines = ['# LightGBM continuous-distribution experiments', '',
                 'Frozen current-HPO parameters; official-exposed post-hoc diagnostics.', '',
                 '| Method | Stage | MSE | <550 MSE | >=550 MSE | CRPS |',
                 '|---|---|---:|---:|---:|---:|']
        for result in results.values():
            for arm, stages in result.items():
                for stage in ('internal', 'combined'):
                    row = stages[stage]
                    crps = row.get('distribution', {}).get('crps')
                    lines.append(f"| {arm} | {stage} | {row['mse']:.3f} | {row['mse_lt550']:.3f} | {row['mse_ge550']:.3f} | {crps if crps is not None else 'N/A'} |")
        (directory / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def execute(run_id, method, smoke=False):
    directory = run_directory(run_id)
    manifest = validate_manifest(directory, smoke)
    lane = directory / method
    lane.mkdir(exist_ok=True)
    with (lane / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            status(lane, 'running', method=method, phase='load_pinned_inputs')
            hpo = imported('dist_frozen_hpo', HPO / 'run.py')
            evaluation = imported('dist_frozen_evaluation', FOLLOW / 'evaluation.py')
            _, names, origins, y, blocks = hpo.load_data()
            fit, val = hpo.temporal_split(origins)
            if smoke:
                fit = fit[np.linspace(0, len(fit) - 1, min(256, len(fit))).astype(int)]
                val = val[np.linspace(0, len(val) - 1, min(24, len(val))).astype(int)]
            manifest_hash = digest(directory / 'manifest.json')
            stages = [('internal', fit, [x[val] for x in blocks], origins[val], y[val])]
            series = pd.read_parquet(hpo.CACHE / 'speed.parquet')['speed']
            for stage, grid, cutoff in [('f1', 'f1', '2026-02-26'), ('final', 'official', '2026-05-29')]:
                stage_fit = np.flatnonzero(origins + pd.Timedelta(hours=72) < pd.Timestamp(cutoff, tz='UTC'))
                if smoke:
                    stage_fit = stage_fit[np.linspace(0, len(stage_fit) - 1, min(256, len(stage_fit))).astype(int)]
                bundle = dict(np.load(hpo.CACHE / f'{grid}.npz'))
                oo = pd.DatetimeIndex(bundle['origins'], tz='UTC')
                query = [hpo.base.design(bundle, names, manifest['selected_groups'], b) for b in (0, 24, 48)]
                yy = hpo.base.targets(series, oo)
                if smoke:
                    ii = np.linspace(0, len(oo) - 1, 16).astype(int)
                    oo, yy, query = oo[ii], yy[ii], [x[ii] for x in query]
                stages.append((stage, stage_fit, query, oo, yy))
            results, all_outputs, truths, all_origins = {}, {}, {}, {}
            for stage, stage_fit, query, oo, yy in stages:
                outputs = train_stage(lane, method, stage, stage_fit, val, origins, y, blocks, query, oo,
                                      manifest['params'], manifest_hash, smoke)
                truths[stage], all_origins[stage] = yy, oo
                for arm, output in outputs.items():
                    result = point_metrics(evaluation, yy, output['pred'], oo)
                    if method != 'controls':
                        result['distribution'] = distribution_metrics(yy, output)
                    results.setdefault(arm, {})[stage] = result
                    all_outputs.setdefault(arm, {})[stage] = output
                    write(lane / 'progress_metrics.json', results)
            for arm, outputs in all_outputs.items():
                oo = all_origins['f1'].append(all_origins['final'])
                yy = np.vstack([truths[s] for s in ('f1', 'final')])
                combined = {key: np.vstack([outputs[s][key] for s in ('f1', 'final')])
                            for key in outputs['f1'] if key != 'origins'}
                combined['origins'] = oo.asi8
                result = point_metrics(evaluation, yy, combined['pred'], oo)
                if method != 'controls':
                    result['distribution'] = distribution_metrics(yy, combined)
                frame = evaluation.frame(oo, combined['pred'])
                if not smoke:
                    if len(frame) != 357840 or frame.duplicated(evaluation.KEY).any():
                        raise ValueError('Official prediction grid mismatch')
                    result['official'] = evaluation.official_score(frame, manifest['parent'])
                    np.testing.assert_allclose(result['mse'], result['official']['mse'], rtol=1e-12)
                frame.to_parquet(lane / f'{arm}_predictions.parquet', index=False)
                array_save(lane / f'combined_{arm}.npz', **combined)
                results[arm]['combined'] = result
            historical = FOLLOW / 'runs/full_20261028_v1/extreme/results.json'
            write(lane / 'historical_xgboost_reference.json', dict(source=str(historical), sha256=digest(historical),
                    reference=read(historical), note='Different estimator/hyperparameters; separate historical comparator, not controlled ablation'))
            write(lane / 'results.json', results)
            aggregate(directory)
            status(lane, 'completed', method=method, smoke=smoke, results=str(lane / 'results.json'))
        except Exception as error:
            status(lane, 'failed', method=method, error=str(error), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=('prepare', 'run', 'report'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.step == 'prepare':
        prepare(args.run_id, args.smoke)
    elif args.step == 'report':
        aggregate(run_directory(args.run_id))
    else:
        if not args.method:
            parser.error('--method is required for run')
        execute(args.run_id, args.method, args.smoke)
