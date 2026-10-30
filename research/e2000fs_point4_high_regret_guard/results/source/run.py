"""Pointweight4 direct-parent lanes: F1 and F2 fits and official forecasts only.

User decision 2026-10-29: the study's internal seasonal gates (dev0-dev2 + confirmation) are removed; every lane
fits the F1 and F2 stages and is scored on the official periods. The v1 gated runs stay in runs/full_20261029_v1.
"""
import argparse
import fcntl
import hashlib
import importlib.util
from pathlib import Path
import traceback

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('point4_baseline', HERE.parent / 'e2000fs_point4_advance/baseline.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
d, np, pd, lgb = b.d, b.np, b.pd, b.lgb
m = d.imported('point4_methods', HERE / 'methods.py')


def fit(folder, context, x, y, params, weights=None, teacher=None, lane=None):
    folder.mkdir(parents=True, exist_ok=True)
    offset = float(np.mean(y)) if lane == 'high_regret_guard' else 0.
    context = dict(context, offset=offset)
    if (folder / 'receipt.json').exists():
        return b.checkpoint(folder, context), offset
    if lane == 'high_regret_guard':
        centered_teacher = teacher-offset
        high = y >= 550.
        def objective(target, prediction):
            return m.regret_derivatives(target, centered_teacher, prediction, high)
        model = lgb.LGBMRegressor(**dict(params, objective=objective))
        history = {}
        model.fit(x, y-offset, eval_set=[(x, y-offset)], eval_names=['train'],
                  eval_metric='l2', callbacks=[lgb.record_evaluation(history)])
        model = model.booster_
        model.save_model(str(folder/'model.txt'))
        pd.DataFrame(dict(round=np.arange(1,len(history['train']['l2'])+1),
                          train_mse=history['train']['l2'])).to_csv(folder/'loss.csv',index=False)
    else:
        if teacher is not None:
            y, weights = m.quadratic_targets(lane, y, teacher)
        model = d.controls_fit(x, y, x[:512], y[:512], params, weights, folder)
    d.write(folder/'offset.json', dict(offset=offset))
    restored = lgb.Booster(model_file=str(folder/'model.txt'))
    np.testing.assert_array_equal(model.predict(x[:512],num_threads=2)+offset,
                                  restored.predict(x[:512],num_threads=2)+d.read(folder/'offset.json')['offset'])
    d.write(folder/'receipt.json',dict(context=context,sha256={f:d.digest(folder/f) for f in ('model.txt','loss.csv','offset.json')}))
    return restored, offset


def execute(lane, smoke=False):
    root = HERE / 'runs' / ('smoke_20261029_v2_f1f2' if smoke else 'full_20261029_v2_f1f2')
    out = root / lane
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            d.status(out, 'running', phase='verify_parent')
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
            paths = [Path(__file__), HERE/'methods.py', HERE/'PLAN.md', Path(b.__file__), Path(d.__file__), d.HPO/'run.py', hpo.BASE/'run.py', d.FOLLOW/'evaluation.py', b.REFERENCE/'manifest.json', b.REFERENCE/'results.json']
            hashes = {str(p):d.digest(p) for p in paths}
            manifest = dict(lane=lane, smoke=smoke, params=params, source_sha256=hashes,
                input_sha256=parent['input_sha256'], stages=['f1','final'],
                parent_experiment=parent['experiment'], selected_groups=distribution['selected_groups'],
                disclosure='All official periods and Top30 selection previously exposed; no untouched test claim',
                method=m.DESCRIPTIONS[lane],
                gate='none: internal seasonal gates removed by user decision 2026-10-29; F1/F2 always evaluated',
                publication='leaderboard',
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
                    if smoke:
                        source = folder/'smoke_teacher'
                        teacher, _ = fit(source, context, x, target, params, b.point_weights(target))
                    else:
                        source = b.REFERENCE/name/f'h{h+1:02}'
                        receipt = d.read(source/'receipt.json')
                        assert receipt['train_count'] == len(train) and not receipt['validation_held_out']
                        assert receipt['sha256'] == d.digest(source/'model.txt')
                        teacher = lgb.Booster(model_file=str(source/'model.txt'))
                    teacher_hash = d.digest(source/'model.txt')
                    context = dict(context, teacher_sha256=teacher_hash)
                    model, offset = fit(folder, context, x, target, params,
                                        teacher=teacher.predict(x,num_threads=2),lane=lane)
                    if teacher is not None:
                        assert d.digest(source/'model.txt') == teacher_hash, 'Teacher mutated'
                    raw = model.predict(xq, num_threads=2)+offset
                    restored = lgb.Booster(model_file=str(folder/'model.txt'))
                    np.testing.assert_array_equal(raw, restored.predict(xq, num_threads=2)+d.read(folder/'offset.json')['offset'])
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
                (out/'REPORT.md').write_text('# Pointweight4 '+lane+'\n\n'+status+'\n\n'+('SMOKE ONLY.\n' if smoke else '')+'F1/F2 only; internal gates removed (user decision 2026-10-29). Previously exposed periods; no untouched test claim.\n\n'+str(results)+'\n')
                artifacts = ['manifest.json','results.json']+[p.name for p in out.glob('*predictions.npz')]+[p.name for p in out.glob('predictions.parquet')]
                d.write(out/'completion.json',dict(sha256={str(out/f):d.digest(out/f) for f in artifacts}))
                d.status(out,status,smoke=smoke)

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
