"""True DirMO-24: pointweight4 for h25-72, ordinary L2 for h1-24."""
import argparse
import csv
import fcntl
from functools import partial
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = HERE.name
CACHE = ROOT / 'analysis/forecast_strategies_e2000/outputs/cache'
PARENT = ROOT / 'analysis/e2000fs_feat30_pointweight/runs/full_20261028_v1/manifest.json'
HPO = ROOT / 'analysis/e2000fs_dirmo_24_feat30_lgbm_HPO/run.py'
EVALUATION = ROOT / 'analysis/e2000fs_feat30_followups/evaluation.py'
RECIPE = ROOT / 'analysis/e2000fs-visualizations/worktree/docs/visualizations/e2000fs_dirmo_24/feature_importance_receipt.json'


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')
    tmp.replace(path)


def weights(y, normalizer=None):
    y = np.asarray(y)
    if y.ndim != 2 or not np.isfinite(y).all():
        raise ValueError('Expected finite two-dimensional targets')
    w = np.where(y >= 550., 4., 1.)
    norm = w.mean(axis=0) if normalizer is None else np.asarray(normalizer)
    return (w / norm).astype(np.float32)


def block_weights(y, block, normalizer=None):
    if block not in (0, 1, 2):
        raise ValueError('Expected block index 0, 1, or 2')
    if block < 1:
        return np.ones_like(y, dtype=np.float32)
    return weights(y, normalizer)


def objective(pred, matrix, block=2):
    target = matrix.get_label().reshape(pred.shape)
    w = block_weights(target, block)
    return (pred - target) * w, w


class LossHistory(xgb.callback.TrainingCallback):
    def __init__(self, path):
        self.path = path

    def after_iteration(self, model, epoch, evals_log):
        with self.path.open('a', newline='') as stream:
            writer = csv.writer(stream)
            if epoch == 0:
                writer.writerow(['round', 'train_mse', 'validation_mse', 'train_weighted_mse', 'validation_weighted_mse'])
            writer.writerow([epoch + 1, evals_log['train']['rmse'][-1] ** 2,
                             evals_log['validation']['rmse'][-1] ** 2,
                             evals_log['train']['weighted_mse'][-1],
                             evals_log['validation']['weighted_mse'][-1]])
        return False


def execute(out, smoke=False):
    out.mkdir(parents=True, exist_ok=False)
    (out / 'models').mkdir()
    (out / 'losses').mkdir()
    write(out / 'status.json', dict(status='running', smoke=smoke, pid=os.getpid(), phase='load_frozen_inputs'))
    try:
        parent = read(PARENT)
        for p, sha in parent['input_sha256'].items():
            if digest(p) != sha:
                raise ValueError(f'Frozen parent input differs: {p}')
        hpo = imported('dirmo_point4_inputs', HPO)
        ev = imported('dirmo_point4_eval', EVALUATION)
        baseline, names, origins, y, blocks = hpo.load_data()
        fit, val = hpo.temporal_split(origins)
        recipe = read(RECIPE)['recipe'].copy()
        rounds = 2 if smoke else 900
        recipe.update(n_estimators=rounds, n_jobs=4, multi_strategy='multi_output_tree')
        params = {k: v for k, v in recipe.items() if k not in ('n_estimators', 'n_jobs', 'random_state')}
        params.update(nthread=4, seed=42, num_target=24)
        sources = [Path(__file__), HPO, Path(hpo.base.__file__), EVALUATION, RECIPE,
                   ROOT / 'analysis/forecast_strategies_e2000/data.py']
        sources += [HERE / n for n in ('execute.py', 'publish.py') if (HERE / n).is_file()]
        source_hashes = {str(p): digest(p) for p in sources}
        identity = dict(experiment=EXPERIMENT, smoke=smoke, selected=baseline['selected'],
                        source_sha256=source_hashes, input_identity=parent['input_sha256'],
                        recipe=dict(xgboost=recipe, rounds=rounds, architecture='three 24-output vector-leaf trees',
                                    loss='h1-24 ordinary L2; h25-72 per-target y>=550 weight4 else1 with training-horizon mean normalization',
                                    weighted_horizons=list(range(25, 73)),
                                    initialization='scalar unweighted training-target mean per block',
                                    seed=42, ace_lag_hours=0, maturity_hours=72),
                        xgboost_version=xgb.__version__, python=sys.version,
                        direct_parent='e2000fs_dirmo_24_feat30_pointweight4',
                        comparator='e2000fs_dirmo_24_feat30 (unweighted original DirMO)',
                        limitation='Top30 selection is official-exposed; no fresh holdout or official-score tuning',
                        objective_docs='https://github.com/dmlc/xgboost/blob/v2.1.4/demo/guide-python/multioutput_regression.py')
        write(out / 'manifest.json', identity)
        series = pd.read_parquet(CACHE / 'speed.parquet')['speed']
        stages = [('internal', fit, [x[val] for x in blocks], origins[val], y[val])]
        for stage, grid, cutoff in [('f1', 'f1', '2026-02-26'), ('final', 'official', '2026-05-29')]:
            train = np.flatnonzero(origins + pd.Timedelta(hours=72) < pd.Timestamp(cutoff, tz='UTC'))
            bundle = dict(np.load(CACHE / f'{grid}.npz'))
            oo = pd.DatetimeIndex(bundle['origins'], tz='UTC')
            query = [hpo.base.design(bundle, names, baseline['selected'], b) for b in (0, 24, 48)]
            stages.append((stage, train, query, oo, hpo.base.targets(series, oo)))
        results, outputs, timings, model_hashes = {}, {}, [], {}
        for stage, train, query, oo, yy in stages:
            if smoke:
                train = train[np.linspace(0, len(train)-1, min(128, len(train)), dtype=int)]
                oo, yy, query = oo[:16], yy[:16], [x[:16] for x in query]
            assert (origins[train] + pd.Timedelta(hours=72)).max() < oo.min()
            assert np.isfinite(y[train]).all() and np.isfinite(yy).all()
            print(f'{stage}: train={len(train)} eval={len(oo)} rounds={rounds}', flush=True)
            pred = np.empty_like(yy)
            for b in range(3):
                sl = slice(b*24, (b+1)*24)
                write(out / 'status.json', dict(status='running', smoke=smoke, stage=stage, block=b, rounds=rounds, pid=os.getpid()))
                target = y[train, sl].astype(np.float32)
                dt = xgb.DMatrix(blocks[b][train], label=target, nthread=4)
                dv = xgb.DMatrix(query[b], label=yy[:, sl].astype(np.float32), nthread=4)
                normalizer = np.where(target >= 550., 4., 1.).mean(axis=0)
                def metric(p, matrix):
                    actual = matrix.get_label().reshape(p.shape)
                    return 'weighted_mse', float(np.mean(block_weights(actual, b, normalizer) * (p-actual)**2))
                settings = dict(params, base_score=float(target.mean()))
                start = time.perf_counter()
                model = xgb.train(settings, dt, num_boost_round=rounds, obj=partial(objective, block=b),
                                  evals=[(dt, 'train'), (dv, 'validation')], custom_metric=metric,
                                  callbacks=[LossHistory(out / 'losses' / f'{stage}_b{b}.csv')], verbose_eval=25)
                path = out / 'models' / f'{stage}_b{b}.ubj'
                model.save_model(path)
                reload = xgb.Booster(model_file=path)
                begin = time.perf_counter()
                raw = model.inplace_predict(query[b])
                inference = time.perf_counter() - begin
                np.testing.assert_array_equal(raw, reload.inplace_predict(query[b]))
                assert raw.shape == (len(oo), 24)
                pred[:, sl] = np.clip(raw, 1., 3000.)
                model_hashes[str(path.relative_to(out))] = digest(path)
                if stage != 'internal':
                    timings.append(dict(stage=stage, block=b, origins=len(oo), seconds=inference))
                print(f'{stage} b{b}: vector output {raw.shape}, reload parity passed; elapsed={time.perf_counter()-start:.1f}s', flush=True)
            assert np.isfinite(pred).all()
            np.savez_compressed(out / f'{stage}_predictions.npz', pred=pred, origins=oo.asi8)
            results[stage] = ev.metrics(yy, pred, oo)
            write(out / 'results.json', results)
            outputs[stage] = (oo, yy, pred)
        oo = outputs['f1'][0].append(outputs['final'][0])
        yy = np.vstack([outputs[s][1] for s in ('f1', 'final')])
        pred = np.vstack([outputs[s][2] for s in ('f1', 'final')])
        frame = ev.frame(oo, pred)
        results['combined'] = ev.metrics(yy, pred, oo)
        if not smoke:
            assert len(frame) == 357840 and not frame.duplicated(ev.KEY).any()
            results['combined']['official'] = ev.official_score(frame, baseline)
            np.testing.assert_allclose(results['combined']['mse'], results['combined']['official']['mse'], rtol=1e-12)
        frame.to_parquet(out / 'predictions.parquet', index=False)
        write(out / 'results.json', results)
        seconds = sum(t['seconds'] for t in timings) / sum(len(outputs[s][0]) for s in ('f1', 'final'))
        write(out / 'timing.json', dict(inference_seconds_per_fold=seconds, measurements=timings))
        write(out / 'model_receipts.json', model_hashes)
        for p, sha in source_hashes.items():
            assert digest(p) == sha, f'Run source changed: {p}'
        write(out / 'status.json', dict(status='completed', smoke=smoke, pid=os.getpid(), results=str(out / 'results.json')))
        print('SMOKE PASSED' if smoke else 'TRAINING COMPLETE', flush=True)
    except BaseException:
        write(out / 'status.json', dict(status='failed', smoke=smoke, traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with (args.out.parent / (args.out.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args.out.resolve(), args.smoke)
