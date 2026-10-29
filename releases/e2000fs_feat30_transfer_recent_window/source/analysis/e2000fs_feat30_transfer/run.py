"""Frozen chronological LightGBM experiments; official evaluation is gate protected."""
import argparse
import fcntl
import hashlib
import importlib.util
from pathlib import Path
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('transfer_helpers', ROOT / 'analysis/e2000fs_feat30_lgbm_distribution/run.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
m = d.imported('transfer_methods', HERE / 'methods.py')
np, pd, lgb = d.np, d.pd, d.lgb
REFERENCE = ROOT / 'analysis/e2000fs_feat30_lgbm_distribution/runs/full_20261028_v1'
FOLDS = [('dev0', '2024-12-04', '2025-02-22'), ('dev1', '2025-03-04', '2025-05-22'),
         ('dev2', '2025-09-04', '2025-11-22'), ('confirmation', '2025-12-04', '2026-02-22')]


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if d.digest(path) != expected:
            raise ValueError(f'Frozen file changed: {path}')


def checkpoint(folder, context):
    receipt = d.read(folder / 'receipt.json')
    if receipt['context'] != context:
        raise ValueError('Checkpoint identity changed')
    verify_hashes({str(folder / name): digest for name, digest in receipt['sha256'].items()})
    return lgb.Booster(model_file=str(folder / 'model.txt'))


def wait_baseline(directory, source_hashes):
    while True:
        path = directory / 'status.json'
        if path.exists():
            state = d.read(path)['status']
            if state == 'failed':
                raise RuntimeError('Shared baseline failed')
            if state == 'completed':
                manifest = d.read(directory / 'manifest.json')
                if manifest['source_sha256'] != source_hashes:
                    raise ValueError('Baseline source identity differs')
                verify_hashes(d.read(directory / 'completion.json')['sha256'])
                return d.read(directory / 'results.json')
        time.sleep(2)


def execute(lane, smoke=False):
    root = HERE / 'runs' / ('smoke_20261028_v1' if smoke else 'full_20261028_v1')
    out = root / lane
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            d.status(out, 'running', phase='load_frozen_inputs')
            parent = d.read(REFERENCE / 'manifest.json')
            hpo = d.imported('transfer_hpo', d.HPO / 'run.py')
            ev = d.imported('transfer_evaluation', d.FOLLOW / 'evaluation.py')
            data_parent, names, origins, y, blocks = hpo.load_data()
            assert data_parent == parent['parent']
            params = dict(parent['params'], n_jobs=2)
            if smoke:
                params['n_estimators'] = 3
            paths = [Path(__file__), HERE / 'methods.py', HERE / 'PLAN.md', Path(d.__file__),
                     d.HPO / 'run.py', hpo.BASE / 'run.py', d.FOLLOW / 'evaluation.py',
                     REFERENCE / 'manifest.json', REFERENCE / 'controls/results.json']
            hashes = {str(p): d.digest(p) for p in paths}
            verify_hashes(parent['inherited_hashes'])
            manifest = dict(lane=lane, smoke=smoke, params=params, source_sha256=hashes,
                            input_sha256=parent['inherited_hashes'], folds=FOLDS,
                            confirmation='reused preofficial confirmation; not untouched',
                            selected_groups=parent['selected_groups'],
                            gate='all folds total/low/high nonworse; pooled total -1%, high -5%, low nonworse',
                            smoke_forces_stage_access=smoke, publication='none',
                            versions=dict(lightgbm=lgb.__version__, numpy=np.__version__, pandas=pd.__version__))
            # JSON serializes fold tuples to lists: compare the normalized representation.
            manifest['folds'] = [list(row) for row in FOLDS]
            if (out / 'manifest.json').exists():
                if d.read(out / 'manifest.json') != manifest:
                    raise ValueError('Run identity changed')
            else:
                d.write(out / 'manifest.json', manifest)
            identity = d.digest(out / 'manifest.json')
            baseline = None
            bundle = dict(np.load(hpo.CACHE / 'train.npz'))
            anchor = np.zeros(len(origins))
            mask = m.speed_mask(names, parent['selected_groups'])
            if lane == 'center':
                blocks, anchor = m.center(blocks, bundle['past'], mask)
            results, outputs = {}, {}

            def stage(name, cutoff, query, oo, yy, qa):
                train = m.mature_indices(origins, pd.Timestamp(cutoff, tz='UTC'))
                train, weights = m.training_policy(origins, train, lane)
                if smoke:
                    pick = np.linspace(0, len(train)-1, min(512, len(train)), dtype=int)
                    train = train[pick]
                    weights = None if weights is None else weights[pick] / weights[pick].mean()
                    oo, yy, qa = oo[:16], yy[:16], qa[:16]
                    query = [x[:16] for x in query]
                assert len(train) and (origins[train]+pd.Timedelta(hours=72)).max() < oo.min()
                pred = np.empty_like(yy)
                for horizon in range(72):
                    folder = out / name / f'h{horizon+1:02}'
                    folder.mkdir(parents=True, exist_ok=True)
                    context = dict(manifest_sha256=identity, lane=lane, stage=name, horizon=horizon+1,
                                   train_indices_sha256=hashlib.sha256(train.tobytes()).hexdigest())
                    if (folder / 'receipt.json').exists():
                        model = checkpoint(folder, context)
                    else:
                        d.status(out, 'running', stage=name, horizon=horizon+1)
                        x = blocks[horizon//24][train]
                        target = y[train, horizon] - anchor[train]
                        xv, yv = query[horizon//24], yy[:, horizon]-qa
                        if name in ('f1', 'final'):
                            xv, yv = x[:512], target[:512]
                        model = d.controls_fit(x, target, xv, yv,
                                               params, weights, folder)
                        d.write(folder / 'receipt.json', dict(context=context,
                                sha256={f:d.digest(folder/f) for f in ('model.txt', 'loss.csv')}))
                    pred[:, horizon] = np.clip(model.predict(query[horizon//24], num_threads=2)+qa, 1., 3000.)
                assert np.isfinite(pred).all()
                d.array_save(out / f'{name}_predictions.npz', pred=pred, origins=oo.asi8)
                results[name] = d.point_metrics(ev, yy, pred, oo)
                results[name]['training'] = dict(rows=len(train), high_target_count=int((y[train]>=550).sum()),
                    effective_sample_size=float(len(train) if weights is None else weights.sum()**2/(weights**2).sum()),
                    last_origin=str(origins[train].max()), last_target=str((origins[train]+pd.Timedelta(hours=72)).max()))
                outputs[name] = (oo, yy, pred)
                d.write(out / 'results.json', results)

            def pooled(stages):
                oo = outputs[stages[0]][0]
                for name in stages[1:]:
                    oo = oo.append(outputs[name][0])
                yy = np.vstack([outputs[name][1] for name in stages])
                pp = np.vstack([outputs[name][2] for name in stages])
                return d.point_metrics(ev, yy, pp, oo), oo, pp

            def finish(state):
                verify_hashes(hashes)
                d.write(out / 'results.json', results)
                (out / 'REPORT.md').write_text('# Transfer gate result\n\n' + state + '\n\n' +
                    ('SMOKE ONLY: forced stage access; no acceptance claim.\n\n' if smoke else '') +
                    '\n'.join(f'- {k}: {v}' for k, v in results.items()) + '\n')
                d.write(out / 'completion.json', dict(sha256={str(out/f):d.digest(out/f)
                        for f in ('manifest.json', 'results.json')}))
                d.status(out, state, smoke=smoke)

            for name, start, end in FOLDS:
                if name == 'confirmation' and lane != 'baseline':
                    d.status(out, 'running', phase='waiting_for_baseline')
                    baseline = wait_baseline(root / 'baseline', hashes)
                    passed = m.development_gate(results, baseline)
                    results['development_passed'] = passed
                    if not passed and not smoke:
                        finish('completed_rejected')
                        return
                ix = np.flatnonzero((origins >= pd.Timestamp(start, tz='UTC')) &
                                    (origins < pd.Timestamp(end, tz='UTC')+pd.Timedelta(days=1)))
                stage(name, start, [x[ix] for x in blocks], origins[ix], y[ix], anchor[ix])
                if name == 'dev2':
                    results['pooled'] = pooled(['dev0', 'dev1', 'dev2'])[0]
            if lane == 'baseline':
                finish('completed')
                return
            passed = m.gate(results['confirmation'], baseline['confirmation'], True)
            results['confirmation_passed'] = passed
            if not passed and not smoke:
                finish('completed_rejected')
                return
            # Official arrays and targets are opened only after both performance gates.
            series = pd.read_parquet(hpo.CACHE / 'speed.parquet')['speed']
            for name, grid, cutoff in [('f1', 'f1', '2026-02-26'), ('final', 'official', '2026-05-29')]:
                query_bundle = dict(np.load(hpo.CACHE / f'{grid}.npz'))
                oo = pd.DatetimeIndex(query_bundle['origins'], tz='UTC')
                query = [hpo.base.design(query_bundle, names, parent['selected_groups'], b) for b in (0,24,48)]
                qa = np.zeros(len(oo))
                if lane == 'center':
                    query, qa = m.center(query, query_bundle['past'], mask)
                stage(name, cutoff, query, oo, hpo.base.targets(series, oo), qa)
            results['combined'], oo, pp = pooled(['f1', 'final'])
            if not smoke:
                frame = ev.frame(oo, pp)
                assert len(frame) == 357840 and not frame.duplicated(ev.KEY).any()
                results['combined']['official'] = ev.official_score(frame, parent['parent'])
                np.testing.assert_allclose(results['combined']['mse'], results['combined']['official']['mse'], rtol=1e-12)
                frame.to_parquet(out / 'predictions.parquet', index=False)
                reference = d.read(REFERENCE / 'controls/results.json')['l2']['combined']
                results['transferred'] = m.gate(results['combined'], reference, True)
            finish('completed_smoke' if smoke else 'completed_evaluated')
        except Exception as error:
            d.status(out, 'failed', error=str(error), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane', required=True, choices=['baseline', 'center', 'recency', 'recent_window'])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    execute(args.lane, args.smoke)
