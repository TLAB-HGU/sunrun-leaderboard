"""Frozen, CPU-only post-hoc feat30 research; no downloads or publication."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault('OMP_NUM_THREADS', '3')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / 'analysis/forecast_strategies_e2000/outputs/cache'
PARENT = ROOT / 'analysis/e2000fs_dirmo_24_feat30/logs/full_01'
METHODS = ('shape', 'wavelet', 'extreme', 'probtraj')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, (Path, pd.Timestamp)):
            return str(obj)
        raise TypeError(type(obj).__name__)
    temp.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + '\n')
    temp.replace(path)


def save_array(path, **arrays):
    temp = Path(str(path) + '.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    temp.replace(path)


def load_data():
    manifest = json.loads((PARENT / 'manifest.json').read_text())
    names = json.loads((CACHE / 'names.json').read_text())
    selected = manifest['selected']
    ci = [i for i, n in enumerate(names['constant']) if n in selected]
    ei = [i for i, n in enumerate(names['varying']) if n in selected]
    assert len(ci) == 21 and len(ei) == 9 and len(set(selected)) == 30
    series = pd.read_parquet(CACHE / 'speed.parquet').speed
    result = {}
    for stage in ('train', 'f1', 'official'):
        with np.load(CACHE / f'{stage}.npz') as a:
            origins = pd.to_datetime(a['origins'], utc=True)
            t = origins.values[:, None] + np.arange(1, 73).astype('timedelta64[h]')
            y = series.reindex(pd.DatetimeIndex(t.ravel(), tz='UTC')).to_numpy().reshape(-1, 72)
            result[stage] = dict(origins=origins, C=a['C'][:, ci], E=a['E'][:, :, ei], Y=y)
    return manifest, names, result


def design(bundle, block, augmented=False):
    from methods import augment_design
    e = bundle['E'][:, block * 24:(block + 1) * 24]
    if augmented:
        return augment_design(bundle['C'], e, include_wavelet=augmented != 'difference')
    return np.hstack([bundle['C'], e.reshape(len(e), -1)]).astype(np.float32)


def split(data, stage, smoke):
    origins = data['train']['origins']
    cutoff = {'select': '2025-12-01', 'f1': '2026-02-26', 'final': '2026-05-29'}[stage]
    fit = np.flatnonzero(origins + pd.Timedelta(hours=72) < pd.Timestamp(cutoff, tz='UTC'))
    valid = np.flatnonzero((origins >= pd.Timestamp('2025-12-04', tz='UTC')) &
                           (origins <= pd.Timestamp('2026-02-22T23:00:00Z')))
    if smoke:
        fit = fit[np.linspace(0, len(fit) - 1, min(128, len(fit))).astype(int)]
        valid = valid[np.linspace(0, len(valid) - 1, min(32, len(valid))).astype(int)]
    assert len(fit) and len(valid)
    assert (origins[fit] + pd.Timedelta(hours=72) < pd.Timestamp(cutoff, tz='UTC')).all()
    if stage == 'select':
        assert origins[fit].max() + pd.Timedelta(hours=72) < origins[valid].min()
    return fit, valid


def snapshot(smoke):
    parent = json.loads((PARENT / 'manifest.json').read_text())
    for path, expected in parent['hashes'].items():
        if not Path(path).is_file() or digest(path) != expected:
            raise RuntimeError(f'Frozen parent input changed or missing: {path}')
    paths = [*sorted(HERE.glob('*.py')), PARENT / 'manifest.json',
             *sorted(CACHE.glob('*.npz')), CACHE / 'names.json', CACHE / 'speed.parquet', CACHE / 'receipt.json',
             *sorted((PARENT / 'artifacts').glob('*.ubj'))]
    content = {str(p): digest(p) for p in paths}
    identity = hashlib.sha256(json.dumps(dict(hashes=content, smoke=smoke, threads=3,
                                              protocol='feat30-followups-v1'), sort_keys=True).encode()).hexdigest()
    return dict(identity=identity, hashes=content, smoke=smoke,
                source='frozen feat30; posthoc official exposed; not a fresh holdout',
                parent_mse=3249.842329310296, selected=parent['selected'],
                evaluation_revision=parent['evaluation_revision'], threads=3,
                target='absolute filled ACE speed km/s; no additive correction',
                seeds=[42], xgboost=xgb.__version__, python=sys.version)


def ensure_identity(directory, manifest):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'manifest.json'
    if path.exists():
        prior = json.loads(path.read_text())
        if prior != manifest:
            raise RuntimeError(f'Refusing incompatible resume: {directory}')
    else:
        write(path, manifest)


def log_status(directory, status, **extra):
    write(directory / 'status.json', dict(status=status, pid=os.getpid(), utc=pd.Timestamp.now(tz='UTC').isoformat(), **extra))
    print(json.dumps(dict(status=status, **extra)), flush=True)


def fit_models(directory, data, stage, arm, smoke):
    from methods import extreme_weights, shape_objective
    fit, valid = split(data, stage, smoke)
    models = []
    augmented = arm if arm in ('difference', 'wavelet') else False
    parent = json.loads((PARENT / 'manifest.json').read_text())
    for block in range(3):
        folder = directory / stage / arm
        folder.mkdir(parents=True, exist_ok=True)
        modelpath = folder / f'b{block:02}.ubj'
        receiptpath = modelpath.with_suffix('.receipt.json')
        X = design(data['train'], block, augmented)
        y = data['train']['Y'][:, block * 24:(block + 1) * 24].astype(np.float32)
        # Preserve parent's leading missing inputs; XGBoost routes NaNs natively.
        assert not np.isinf(X[fit]).any() and np.isfinite(y[fit]).all() and np.isfinite(y[valid]).all()
        if modelpath.exists() and receiptpath.exists():
            receipt = json.loads(receiptpath.read_text())
            if digest(modelpath) != receipt['sha256']:
                raise RuntimeError(f'Model checkpoint hash mismatch: {modelpath}')
            model = xgb.Booster(model_file=str(modelpath))
            model.set_param({'nthread': 3})
        else:
            params = dict(parent['params'])
            params.update(n_jobs=3, n_estimators=3 if smoke else 900,
                          base_score=float(y[fit].mean()), eval_metric='rmse')
            if arm.startswith('shape_'):
                params['objective'] = shape_objective(float(arm.split('_')[1]))
            model = xgb.XGBRegressor(**params)
            weights = extreme_weights(y[fit], float(arm.split('_')[1])) if arm.startswith('extreme_') else None
            print(f'{stage}/{arm}/b{block}: fit {len(fit)} rows, {X.shape[1]} columns', flush=True)
            model.fit(X[fit], y[fit], sample_weight=weights,
                      eval_set=[(X[fit], y[fit]), (X[valid], y[valid])], verbose=100)
            history = model.evals_result()
            pd.DataFrame(dict(round=np.arange(1, len(history['validation_0']['rmse']) + 1),
                              train_mse=np.square(history['validation_0']['rmse']),
                              diagnostic_validation_mse=np.square(history['validation_1']['rmse']),
                              validation_is_held_out=stage == 'select')).to_csv(folder / f'b{block:02}_loss.csv', index=False)
            booster = model.get_booster()
            temp = modelpath.with_name(modelpath.stem + '.tmp.ubj')
            booster.save_model(temp)
            reloaded = xgb.Booster(model_file=str(temp))
            reloaded.set_param({'nthread': 3})
            np.testing.assert_array_equal(booster.inplace_predict(X[valid]), reloaded.inplace_predict(X[valid]))
            temp.replace(modelpath)
            write(receiptpath, dict(sha256=digest(modelpath), fit_origins=len(fit), columns=X.shape[1],
                                    target_end_exclusive={'select': '2025-12-01', 'f1': '2026-02-26', 'final': '2026-05-29'}[stage],
                                    reload_parity=True, native_loss=True, validation_is_held_out=stage == 'select'))
            model = reloaded
        models.append(model)
    return models


def predict(models, bundle, arm):
    return np.clip(np.hstack([m.inplace_predict(design(bundle, b, arm if arm in ('difference', 'wavelet') else False))
                              for b, m in enumerate(models)]), 1., 3000.)


def take(bundle, indices):
    return {k: v[indices] for k, v in bundle.items()}


def frozen_models(stage):
    models = [xgb.Booster(model_file=str(PARENT / 'artifacts' / f'{stage}_b{b:02}.ubj')) for b in range(3)]
    for model in models:
        model.set_param({'nthread': 3})
    return models


def parent_parity(data):
    evidence = {}
    for stage, grid in [('f1', 'f1'), ('final', 'official')]:
        p = predict(frozen_models(stage), data[grid], 'baseline')
        saved = np.load(PARENT / 'artifacts' / f'{stage}_predictions.npy')
        np.testing.assert_array_equal(p, saved)
        evidence[stage] = dict(origins=len(p), max_abs_diff=0.)
    return evidence


def prepare(directory, smoke):
    from evaluation import metrics
    manifest = snapshot(smoke)
    ensure_identity(directory, manifest)
    log_status(directory, 'running', phase='prepare')
    _, _, data = load_data()
    parity = parent_parity(data)
    write(directory / 'parent_parity.json', parity)
    _, valid = split(data, 'select', smoke)
    validation = take(data['train'], valid)
    models = fit_models(directory, data, 'select', 'baseline', smoke)
    p = predict(models, validation, 'baseline')
    save_array(directory / 'internal_baseline.npz', pred=p, indices=valid)
    write(directory / 'internal_baseline_metrics.json', metrics(validation['Y'], p))
    log_status(directory, 'completed', phase='prepare', parent_parity=parity)


def wait_prepare(common, identity):
    while True:
        path = common / 'status.json'
        if path.exists():
            status = json.loads(path.read_text())
            if status['status'] == 'failed':
                raise RuntimeError(f'Preparation failed: {status}')
            if status['status'] == 'completed':
                if json.loads((common / 'manifest.json').read_text())['identity'] != identity:
                    raise RuntimeError('Preparation identity differs')
                return
        print('Waiting for verified shared internal baseline', flush=True)
        time.sleep(10)


def leaf_indices(models, bundle):
    # Column position is tree identity; numeric leaf IDs from different trees never mix.
    return np.hstack([m.predict(xgb.DMatrix(design(bundle, b), nthread=3), pred_leaf=True).astype(np.int32)
                      for b, m in enumerate(models)])


def probabilistic(directory, stage, models, fit_bundle, query_bundle):
    from probtraj import LeafTrajectoryDistribution, distribution_summary
    index = LeafTrajectoryDistribution().fit(leaf_indices(models, fit_bundle), fit_bundle['Y'])
    leaves = leaf_indices(models, query_bundle)
    means, records = [], []
    donors_dir = directory / stage / 'donors'
    donors_dir.mkdir(parents=True, exist_ok=True)
    for i, prediction in enumerate(index.iter_predictions(leaves, k=128, batch_size=8)):
        samples = fit_bundle['Y'][prediction['indices']]
        info = distribution_summary(samples, prediction['weights'], query_bundle['Y'][i])
        means.append(prediction['mean'])
        records.append(dict(origin=str(query_bundle['origins'][i]), fallback=bool(prediction['fallback']), **info))
        # Individual atomic files bound to top-level code/data/params manifest; retain complete donors on fallback.
        save_array(donors_dir / f'{i:05}.npz', indices=prediction['indices'], weights=prediction['weights'])
        if i % 128 == 0:
            log_status(directory, 'running', phase=f'{stage}/probtraj', completed_origins=i, total=len(leaves))
    write(directory / stage / 'distribution.json', records)
    return np.clip(np.asarray(means), 1., 3000.)


def run(directory, common, method, smoke):
    from evaluation import metrics, selection_score, selection_key, frame, official_score
    manifest = snapshot(smoke)
    manifest.update(method=method, arms={'shape': ['shape_0.25', 'shape_1.0'], 'wavelet': ['difference', 'wavelet'],
                                       'extreme': ['extreme_2.0', 'extreme_4.0'], 'probtraj': ['probtraj']}[method])
    ensure_identity(directory, manifest)
    log_status(directory, 'running', phase='waiting_for_baseline')
    wait_prepare(common, manifest['identity'])
    _, _, data = load_data()
    fit, valid = split(data, 'select', smoke)
    validation = take(data['train'], valid)
    base_metrics = json.loads((common / 'internal_baseline_metrics.json').read_text())
    choices = []
    for arm in manifest['arms']:
        log_status(directory, 'running', phase=f'internal/{arm}')
        if method == 'probtraj':
            models = fit_models(common, data, 'select', 'baseline', smoke)
            p = probabilistic(directory, 'select', models, take(data['train'], fit), validation)
        else:
            models = fit_models(directory, data, 'select', arm, smoke)
            p = predict(models, validation, arm)
        m = metrics(validation['Y'], p)
        score = selection_score(m, base_metrics)
        choices.append(dict(arm=arm, metrics=m, selection_score=score))
        save_array(directory / f'internal_{arm}.npz', pred=p, indices=valid)
    choices.sort(key=lambda x: selection_key(x['metrics'], base_metrics))
    selected = choices[0]['arm']
    write(directory / 'selection.json', dict(selected=selected, candidates=choices,
                                             rule='mean(parent-relative peak MAE, parent-relative positive 6h ramp MAE), then CSI then MSE',
                                             frozen_before_official=True))
    results, frames = {}, []
    for stage, grid in [('f1', 'f1'), ('final', 'official')]:
        log_status(directory, 'running', phase=f'{stage}/{selected}')
        if method == 'probtraj':
            models = frozen_models(stage)
            fit, _ = split(data, stage, smoke)
            bundle = data[grid] if not smoke else take(data[grid], np.arange(8))
            p = probabilistic(directory, stage, models, take(data['train'], fit), bundle)
        else:
            models = fit_models(directory, data, stage, selected, smoke)
            bundle = data[grid] if not smoke else take(data[grid], np.arange(8))
            p = predict(models, bundle, selected)
        save_array(directory / f'{stage}_predictions.npz', pred=p, origins=bundle['origins'].asi8)
        results[stage] = metrics(bundle['Y'], p)
        if method == 'probtraj':
            records = json.loads((directory / stage / 'distribution.json').read_text())
            results[stage]['probability'] = dict(
                origins=len(records), crps=float(np.mean([r['crps'] for r in records])),
                hourly_brier=float(np.mean([r['brier'] for r in records])),
                any550_brier=float(np.mean([r['any550_brier'] for r in records])),
                never550_probability=float(np.mean([r['never550_probability'] for r in records])),
                fallback_count=sum(r['fallback'] for r in records))
        frames.append(frame(bundle['origins'], p))
    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(directory / 'predictions.parquet', index=False)
    if not smoke:
        assert len(combined) == 357840 and not combined.duplicated(['origin_last_input_utc', 'horizon_hours']).any()
        results['official'] = official_score(combined, json.loads((PARENT / 'manifest.json').read_text()))
        combined_metrics(directory, data, results)
    if method == 'probtraj':
        stages = [results[s]['probability'] for s in ['f1', 'final']]
        results['probability_combined'] = {key: float(np.average([s[key] for s in stages], weights=[s['origins'] for s in stages]))
                                           for key in ['crps', 'hourly_brier', 'any550_brier', 'never550_probability']}
        results['probability_combined']['fallback_count'] = sum(s['fallback_count'] for s in stages)
    write(directory / 'results.json', results)
    from report import render
    render(directory, data, smoke)
    log_status(directory, 'completed', selected=selected, smoke=smoke)


def combined_metrics(directory, data, results):
    from evaluation import metrics
    ys = np.vstack([data['f1']['Y'], data['official']['Y']])
    pred = np.vstack([np.load(directory / f'{stage}_predictions.npz')['pred'] for stage in ('f1', 'final')])
    parents = np.vstack([np.load(PARENT / 'artifacts' / f'{stage}_predictions.npy') for stage in ('f1', 'final')])
    results['combined'] = metrics(ys, pred)
    results['parent'] = metrics(ys, parents)
    for key in ('peak_mae', 'ramp6_mae', 'mse'):
        if key in results['combined'] and key in results['parent']:
            results.setdefault('delta_percent', {})[key] = 100 * (results['combined'][key] / results['parent'][key] - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=['prepare', *METHODS])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        parser.error('run-id must be a directory name')
    common = HERE / 'runs' / args.run_id / 'common'
    directory = common if args.method == 'prepare' else HERE / 'runs' / args.run_id / args.method
    directory.mkdir(parents=True, exist_ok=True)
    try:
        if args.method == 'prepare':
            prepare(directory, args.smoke)
        else:
            run(directory, common, args.method, args.smoke)
    except BaseException as exc:
        traceback.print_exc()
        log_status(directory, 'failed', error=str(exc), exit_code=1)
        raise


if __name__ == '__main__':
    main()
