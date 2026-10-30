"""Pointweight4 direct-parent lanes; seasonal gates precede actual evaluation."""
import argparse
import fcntl
import hashlib
import importlib.util
from pathlib import Path
import time
import traceback

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('point4_baseline', HERE / 'baseline.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
d, np, pd, lgb = b.d, b.np, b.pd, b.lgb
m = d.imported('point4_methods', HERE / 'methods.py')
previous = d.imported('point4_validation', HERE.parent / 'e2000fs_feat30_transfer_v2/run.py')


def fit(folder, context, x, y, params, weights=None, teacher=None):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'receipt.json').exists():
        return b.checkpoint(folder, context)
    if teacher is None:
        model = d.controls_fit(x, y, x[:512], y[:512], params, weights, folder)
    else:
        before = m.topology(teacher)
        model = teacher.refit(x, y, decay_rate=.9, num_threads=2)
        assert model is not teacher and m.topology(model) == before
        model.save_model(str(folder / 'model.txt'))
        pd.DataFrame([dict(train_mse=float(np.mean((model.predict(x, num_threads=2)-y)**2)))]).to_csv(folder / 'loss.csv', index=False)
    restored = lgb.Booster(model_file=str(folder / 'model.txt'))
    np.testing.assert_array_equal(model.predict(x[:512], num_threads=2), restored.predict(x[:512], num_threads=2))
    d.write(folder / 'receipt.json', dict(context=context, sha256={f:d.digest(folder/f) for f in ('model.txt', 'loss.csv')}))
    return model


def execute(lane, smoke=False):
    root = HERE / 'runs' / ('smoke_20261029_v1' if smoke else 'full_20261029_v1')
    out, baseline_dir = root / lane, root / 'baseline'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            d.status(out, 'running', phase='waiting_for_baseline')
            while True:
                status = d.read(baseline_dir/'status.json')['status'] if (baseline_dir/'status.json').exists() else None
                if status == 'completed':
                    break
                if status == 'failed':
                    raise RuntimeError('Pointweight4 baseline failed')
                time.sleep(2)
            parent = d.read(b.REFERENCE/'manifest.json')
            b.verify_hashes(parent['source_sha256']); b.verify_hashes(parent['input_sha256'])
            hpo = d.imported('point4_hpo', d.HPO/'run.py')
            ev = d.imported('point4_ev', d.FOLLOW/'evaluation.py')
            data_parent, names, origins, y, blocks = hpo.load_data()
            distribution = d.read(b.p.REFERENCE/'manifest.json')
            assert data_parent == distribution['parent']
            params = dict(parent['params'], n_jobs=2)
            if smoke:
                params['n_estimators'] = 3
            baseline_manifest = d.read(baseline_dir/'manifest.json')
            assert baseline_manifest['smoke'] == smoke
            assert baseline_manifest['parent_experiment'] == parent['experiment']
            baseline, baseline_hashes = previous.baseline_load(baseline_dir, params, origins, y, ev, smoke)
            paths = [Path(__file__), HERE/'methods.py', HERE/'PLAN.md', HERE/'baseline.py', Path(previous.__file__), Path(previous.m.__file__), Path(d.__file__), d.HPO/'run.py', hpo.BASE/'run.py', d.FOLLOW/'evaluation.py', b.REFERENCE/'manifest.json', b.REFERENCE/'results.json']
            hashes = {str(p):d.digest(p) for p in paths}
            hashes.update(baseline_hashes)
            manifest = dict(lane=lane, smoke=smoke, params=params, source_sha256=hashes,
                input_sha256=parent['input_sha256'], folds=[list(f) for f in b.FOLDS],
                parent_experiment=parent['experiment'], selected_groups=distribution['selected_groups'],
                disclosure='All validation/official periods and Top30 selection previously exposed; no untouched test claim',
                method=dict(linear_leaf='train-only nan scaling; point4; linear_tree; serial; linear_lambda10',
                    leaf_refit='same-fold point4 teacher; unweighted refit decay .9; new Booster',
                    stable_retrain='in-sample same-fold teacher; target .25y+.75teacher; constant weight4')[lane],
                gate='each dev total/low/high nonworse; pooled and confirmation total -1%, low -5%, high nonworse',
                smoke_forces_stage_access=smoke, publication='none',
                versions=dict(lightgbm=lgb.__version__, numpy=np.__version__, pandas=pd.__version__))
            if (out/'manifest.json').exists():
                assert d.read(out/'manifest.json') == manifest, 'Run identity changed'
            else:
                d.write(out/'manifest.json', manifest)
            identity = d.digest(out/'manifest.json')
            results, outputs = {}, {}

            def stage(name, cutoff, query, oo, yy):
                train = b.mature_indices(origins, pd.Timestamp(cutoff, tz='UTC'))
                if smoke:
                    train = train[np.linspace(0,len(train)-1,min(512,len(train)),dtype=int)]
                    oo, yy, query = oo[:16], yy[:16], [x[:16] for x in query]
                assert len(train) and (origins[train]+pd.Timedelta(hours=72)).max() < oo.min()
                train_hash = hashlib.sha256(train.tobytes()).hexdigest()
                pred = np.empty_like(yy)
                for h in range(72):
                    d.status(out, 'running', stage=name, horizon=h+1)
                    folder = out/name/f'h{h+1:02}'
                    folder.mkdir(parents=True, exist_ok=True)
                    context = dict(manifest_sha256=identity, stage=name, horizon=h+1, train_indices_sha256=train_hash)
                    x, target, xq = blocks[h//24][train], y[train,h], query[h//24]
                    teacher = None
                    if lane != 'linear_leaf':
                        if name in {f[0] for f in b.FOLDS}:
                            source = baseline_dir/name/f'h{h+1:02}'
                            teacher = b.checkpoint(source, dict(context, manifest_sha256=d.digest(baseline_dir/'manifest.json')))
                        elif smoke:
                            source = folder/'smoke_teacher'
                            teacher = fit(source, context, x, target, params, b.point_weights(target))
                        else:
                            source = b.REFERENCE/name/f'h{h+1:02}'
                            receipt = d.read(source/'receipt.json')
                            assert receipt['train_count'] == len(train) and not receipt['validation_held_out']
                            assert receipt['sha256'] == d.digest(source/'model.txt')
                            teacher = lgb.Booster(model_file=str(source/'model.txt'))
                        teacher_hash = d.digest(source/'model.txt')
                        context = dict(context, teacher_sha256=teacher_hash)
                    if lane == 'linear_leaf':
                        mean, scale = m.scaling(x)
                        scale_path = folder/'scaling.npz'
                        if scale_path.exists():
                            with np.load(scale_path) as saved:
                                np.testing.assert_array_equal(saved['mean'],mean)
                                np.testing.assert_array_equal(saved['scale'],scale)
                        else:
                            d.array_save(scale_path, mean=mean, scale=scale)
                        context = dict(context, scaling_sha256=d.digest(scale_path))
                        x, xq = (x-mean)/scale, (xq-mean)/scale
                        model = fit(folder, context, x, target, dict(params, linear_tree=True, tree_learner='serial', linear_lambda=10), b.point_weights(target))
                    elif lane == 'leaf_refit':
                        model = fit(folder, context, x, target, params, teacher=teacher)
                    else:
                        pseudo, weights = m.stable_targets(target, teacher.predict(x,num_threads=2))
                        model = fit(folder, context, x, pseudo, params, weights)
                    if teacher is not None:
                        assert d.digest(source/'model.txt') == teacher_hash, 'Teacher mutated'
                    raw = model.predict(xq, num_threads=2)
                    restored = lgb.Booster(model_file=str(folder/'model.txt'))
                    np.testing.assert_array_equal(raw, restored.predict(xq, num_threads=2))
                    pred[:,h] = np.clip(raw,1.,3000.)
                assert np.isfinite(pred).all()
                d.array_save(out/f'{name}_predictions.npz', pred=pred, origins=oo.asi8)
                results[name] = d.point_metrics(ev, yy, pred, oo)
                results[name]['training'] = dict(rows=len(train), last_origin=str(origins[train].max()), last_target=str((origins[train]+pd.Timedelta(hours=72)).max()))
                outputs[name] = (oo,yy,pred)
                d.write(out/'results.json',results)

            def pooled(stages):
                oo = outputs[stages[0]][0].append([outputs[s][0] for s in stages[1:]])
                yy, pp = [np.vstack([outputs[s][i] for s in stages]) for i in (1,2)]
                return d.point_metrics(ev,yy,pp,oo),oo,pp

            def finish(status):
                b.verify_hashes(hashes); b.verify_hashes(parent['input_sha256'])
                d.write(out/'results.json',results)
                (out/'REPORT.md').write_text('# Pointweight4 '+lane+'\n\n'+status+'\n\n'+('SMOKE ONLY; forced stage access.\n' if smoke else '')+'Previously exposed periods; no untouched test claim.\n\n'+str(results)+'\n')
                artifacts = ['manifest.json','results.json']+[p.name for p in out.glob('*predictions.npz')]
                d.write(out/'completion.json',dict(sha256={str(out/f):d.digest(out/f) for f in artifacts}))
                d.status(out,status,smoke=smoke)

            for name,start,end in b.FOLDS:
                if name == 'confirmation':
                    results['development_passed'] = m.development_gate(results,baseline)
                    if not results['development_passed'] and not smoke:
                        finish('completed_rejected'); return
                ix = np.flatnonzero((origins>=pd.Timestamp(start,tz='UTC')) & (origins<pd.Timestamp(end,tz='UTC')+pd.Timedelta(days=1)))
                stage(name,start,[x[ix] for x in blocks],origins[ix],y[ix])
                if name == 'dev2':
                    results['pooled'] = pooled(['dev0','dev1','dev2'])[0]
            results['confirmation_passed'] = m.gate(results['confirmation'],baseline['confirmation'],True)
            if not results['confirmation_passed'] and not smoke:
                finish('completed_rejected'); return
            series = pd.read_parquet(hpo.CACHE/'speed.parquet')['speed']
            for name,grid,cutoff in [('f1','f1','2026-02-26'),('final','official','2026-05-29')]:
                bundle = dict(np.load(hpo.CACHE/f'{grid}.npz'))
                oo = pd.DatetimeIndex(bundle['origins'],tz='UTC')
                query = [hpo.base.design(bundle,names,distribution['selected_groups'],start) for start in (0,24,48)]
                stage(name,cutoff,query,oo,hpo.base.targets(series,oo))
            results['combined'],oo,pp = pooled(['f1','final'])
            if not smoke:
                frame = ev.frame(oo,pp)
                assert len(frame)==357840 and not frame.duplicated(ev.KEY).any()
                results['combined']['official'] = ev.official_score(frame,data_parent)
                np.testing.assert_allclose(results['combined']['mse'],results['combined']['official']['mse'],rtol=1e-12)
                frame.to_parquet(out/'predictions.parquet',index=False)
                results['transferred'] = bool(results['development_passed'] and results['confirmation_passed'] and m.gate(results['combined'],d.read(b.REFERENCE/'results.json')['combined'],True))
            finish('completed_smoke' if smoke else 'completed_evaluated')
        except Exception as error:
            d.status(out,'failed',error=str(error),traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane', required=True, choices=m.LANES)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    execute(args.lane,args.smoke)
