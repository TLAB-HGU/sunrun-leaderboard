"""Frozen train-only adaptations with fail-closed development/confirmation gates."""
import argparse
import fcntl
import hashlib
import importlib.util
from pathlib import Path
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('transfer2_helpers', ROOT / 'analysis/e2000fs_feat30_lgbm_distribution/run.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
m = d.imported('transfer2_methods', HERE / 'methods.py')
np, pd, lgb = d.np, d.pd, d.lgb
REFERENCE = ROOT / 'analysis/e2000fs_feat30_lgbm_distribution/runs/full_20261028_v1'
PREVIOUS = ROOT / 'analysis/e2000fs_feat30_transfer'
FOLDS = [('dev0', '2024-12-04', '2025-02-22'), ('dev1', '2025-03-04', '2025-05-22'),
         ('dev2', '2025-09-04', '2025-11-22'), ('confirmation', '2025-12-04', '2026-02-22')]


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if d.digest(path) != expected:
            raise ValueError(f'Frozen file changed: {path}')


def fit_checkpoint(folder, context, x, y, params, weights=None):
    """Training-only loss diagnostics; restored predictions must be exactly equal."""
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'receipt.json').exists():
        receipt = d.read(folder / 'receipt.json')
        if receipt['context'] != context:
            raise ValueError('Checkpoint identity changed')
        verify_hashes({str(folder / k): v for k, v in receipt['sha256'].items()})
        return lgb.Booster(model_file=str(folder / 'model.txt'))
    model = d.controls_fit(x, y, x[:512], y[:512], params, weights, folder)
    d.write(folder / 'receipt.json', dict(context=context,
            sha256={f: d.digest(folder / f) for f in ('model.txt', 'loss.csv')}))
    return model


def baseline_load(directory, params, origins, y, ev, smoke):
    if d.read(directory / 'status.json')['status'] != 'completed':
        raise ValueError('Reference baseline is incomplete')
    verify_hashes(d.read(directory / 'completion.json')['sha256'])
    manifest = d.read(directory / 'manifest.json')
    verify_hashes(manifest['source_sha256'])
    verify_hashes(manifest['input_sha256'])
    if manifest['params'] != params or manifest['folds'] != [list(f) for f in FOLDS]:
        raise ValueError('Baseline parameters/folds differ')
    metrics = d.read(directory / 'results.json')
    hashes = {str(directory / f): d.digest(directory / f)
              for f in ('manifest.json', 'results.json', 'completion.json')}
    all_origins, all_y, all_pred = [], [], []
    for name, start, end in FOLDS:
        ix = np.flatnonzero((origins >= pd.Timestamp(start, tz='UTC')) &
                            (origins < pd.Timestamp(end, tz='UTC') + pd.Timedelta(days=1)))
        if smoke:
            ix = ix[:16]
        path = directory / f'{name}_predictions.npz'
        hashes[str(path)] = d.digest(path)
        with np.load(path) as data:
            np.testing.assert_array_equal(data['origins'], origins[ix].asi8)
            pred = data['pred'].copy()
        checked = d.point_metrics(ev, y[ix], pred, origins[ix])
        for key in m.KEYS:
            if checked[key] is None or metrics[name][key] is None:
                assert checked[key] is metrics[name][key]
            else:
                np.testing.assert_allclose(checked[key], metrics[name][key], rtol=1e-12)
        if name != 'confirmation':
            all_origins.append(origins[ix]); all_y.append(y[ix]); all_pred.append(pred)
    pooled_origins = all_origins[0].append(all_origins[1:])
    checked = d.point_metrics(ev, np.vstack(all_y), np.vstack(all_pred), pooled_origins)
    for key in m.KEYS:
        if checked[key] is None or metrics['pooled'][key] is None:
            assert checked[key] is metrics['pooled'][key]
        else:
            np.testing.assert_allclose(checked[key], metrics['pooled'][key], rtol=1e-12)
    return metrics, hashes


def execute(lane, smoke=False):
    run_name = 'smoke_20261028_v1' if smoke else 'full_20261028_v1'
    out = HERE / 'runs' / run_name / lane
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            d.status(out, 'running', phase='load_frozen_inputs')
            parent = d.read(REFERENCE / 'manifest.json')
            hpo = d.imported('transfer2_hpo', d.HPO / 'run.py')
            ev = d.imported('transfer2_evaluation', d.FOLLOW / 'evaluation.py')
            data_parent, names, origins, y, blocks = hpo.load_data()
            assert data_parent == parent['parent']
            params = dict(parent['params'], n_jobs=2)
            if smoke:
                params['n_estimators'] = 3
            baseline, baseline_hashes = baseline_load(PREVIOUS / 'runs' / run_name / 'baseline',
                                                       params, origins, y, ev, smoke)
            paths = [Path(__file__), HERE / 'methods.py', HERE / 'PLAN.md', Path(d.__file__),
                     d.HPO / 'run.py', hpo.BASE / 'run.py', d.FOLLOW / 'evaluation.py',
                     REFERENCE / 'manifest.json', REFERENCE / 'controls/results.json']
            hashes = {str(p): d.digest(p) for p in paths}
            hashes.update(baseline_hashes)
            verify_hashes(parent['inherited_hashes'])
            manifest = dict(lane=lane, smoke=smoke, params=params, source_sha256=hashes,
                            input_sha256=parent['inherited_hashes'], folds=[list(f) for f in FOLDS],
                            confirmation='reused preofficial confirmation; not untouched',
                            selected_groups=parent['selected_groups'],
                            gate='all folds total/low/high nonworse; pooled total -1%, high -5%, low nonworse',
                            smoke_forces_stage_access=smoke, publication='none',
                            training_diagnostics='training rows only; no validation early stopping',
                            hyper_gain='a=v_lag0; b=sign(rec_mean-a)*clip(abs(delta),50,300); missing b=100; weights=b^2',
                            gaussian='train-only pilot100; squaredloss z; max(0.1,exp(-z^2/2)); mean normalized',
                            mc_calibration='latest20% train;72h purge; prefixbase400; head32/depth2/leaves4/min128/lr.03; alpha.25',
                            versions=dict(lightgbm=lgb.__version__, numpy=np.__version__, pandas=pd.__version__))
            if (out / 'manifest.json').exists():
                if d.read(out / 'manifest.json') != manifest:
                    raise ValueError('Run identity changed')
            else:
                d.write(out / 'manifest.json', manifest)
            identity = d.digest(out / 'manifest.json')
            bundle = dict(np.load(hpo.CACHE / 'train.npz'))
            anchor, gains, missing = m.decoder(bundle, names)
            results, outputs = {'decoder_missing_recurrence': missing}, {}

            def stage(name, cutoff, query, oo, yy, qa, qb):
                train = m.mature_indices(origins, pd.Timestamp(cutoff, tz='UTC'))
                prefix, calibration = m.calibration_split(origins, train) if lane == 'mc_calibration' else (train, None)
                if smoke:
                    def thin(ix):
                        return ix[np.linspace(0, len(ix)-1, min(512, len(ix)), dtype=int)]
                    train, prefix = thin(train), thin(prefix)
                    if calibration is not None:
                        calibration = thin(calibration)
                    oo, yy, qa, qb = oo[:16], yy[:16], qa[:16], qb[:16]
                    query = [x[:16] for x in query]
                assert len(train) and (origins[train]+pd.Timedelta(hours=72)).max() < oo.min()
                pred = np.empty_like(yy)
                base_pred = np.empty_like(yy) if lane == 'mc_calibration' else None
                stats = []
                for horizon in range(72):
                    folder = out / name / f'h{horizon+1:02}'
                    folder.mkdir(parents=True, exist_ok=True)
                    context = dict(manifest_sha256=identity, lane=lane, stage=name, horizon=horizon+1,
                                   train_indices_sha256=hashlib.sha256(train.tobytes()).hexdigest())
                    d.status(out, 'running', stage=name, horizon=horizon+1)
                    x, target, xq = blocks[horizon//24][train], y[train, horizon], query[horizon//24]
                    info = dict(rows=len(train))
                    if lane == 'hyper_gain':
                        b = gains[train, horizon]
                        weights = b*b
                        transformed = (target-anchor[train])/b
                        model = fit_checkpoint(folder / 'model', context, x, transformed, params, weights)
                        raw = qa + qb[:, horizon]*model.predict(xq, num_threads=2)
                        reload = lgb.Booster(model_file=str(folder / 'model/model.txt'))
                        np.testing.assert_array_equal(raw, qa+qb[:, horizon]*reload.predict(xq, num_threads=2))
                        info.update(weight_min=float(weights.min()), weight_max=float(weights.max()),
                                    effective_sample_size=float(weights.sum()**2/(weights**2).sum()))
                    elif lane == 'gaussian':
                        pilot_params = dict(params, n_estimators=2 if smoke else 100)
                        pilot = fit_checkpoint(folder / 'pilot', context, x, target, pilot_params)
                        weights, info_loss = m.gaussian_weights(target, pilot.predict(x, num_threads=2))
                        weights_path = folder / 'weights.npz'
                        if weights_path.exists():
                            with np.load(weights_path) as saved:
                                np.testing.assert_array_equal(saved['weights'], weights)
                                np.testing.assert_array_equal(saved['train_indices'], train)
                        else:
                            d.array_save(weights_path, weights=weights, train_indices=train)
                        model_context = dict(context, weights_sha256=d.digest(weights_path))
                        model = fit_checkpoint(folder / 'model', model_context, x, target, params, weights)
                        raw = model.predict(xq, num_threads=2)
                        info.update(info_loss, weights_sha256=d.digest(weights_path),
                                    effective_sample_size=float(weights.sum()**2/(weights**2).sum()),
                                    high_weight_mean=float(weights[target>=550].mean()) if (target>=550).any() else None)
                    else:
                        split_context = dict(context, prefix_sha256=hashlib.sha256(prefix.tobytes()).hexdigest(),
                                             calibration_sha256=hashlib.sha256(calibration.tobytes()).hexdigest())
                        base = fit_checkpoint(folder / 'base', split_context, blocks[horizon//24][prefix],
                                              y[prefix, horizon], params)
                        xc = blocks[horizon//24][calibration]
                        pc = base.predict(xc, num_threads=2)
                        head_params = dict(params, n_estimators=3 if smoke else 32, max_depth=2,
                                           num_leaves=4, min_child_samples=128, learning_rate=.03)
                        head = fit_checkpoint(folder / 'head', split_context, np.column_stack((xc, pc)),
                                              y[calibration,horizon]-pc, head_params)
                        pq = base.predict(xq, num_threads=2)
                        raw = pq + .25*head.predict(np.column_stack((xq,pq)), num_threads=2)
                        br = lgb.Booster(model_file=str(folder / 'base/model.txt'))
                        hr = lgb.Booster(model_file=str(folder / 'head/model.txt'))
                        pr = br.predict(xq, num_threads=2)
                        np.testing.assert_array_equal(raw, pr+.25*hr.predict(np.column_stack((xq,pr)), num_threads=2))
                        base_pred[:, horizon] = np.clip(pq,1.,3000.)
                        info.update(prefix_rows=len(prefix), calibration_rows=len(calibration),
                                    prefix_last_target=str((origins[prefix]+pd.Timedelta(hours=72)).max()),
                                    calibration_first_origin=str(origins[calibration].min()))
                    pred[:, horizon] = np.clip(raw, 1., 3000.)
                    stats.append(info)
                    d.write(folder / 'training.json', info)
                assert np.isfinite(pred).all()
                d.array_save(out / f'{name}_predictions.npz', pred=pred, origins=oo.asi8)
                results[name] = d.point_metrics(ev, yy, pred, oo)
                results[name]['training'] = dict(rows=len(train), high_target_count=int((y[train]>=550).sum()),
                    last_origin=str(origins[train].max()), last_target=str((origins[train]+pd.Timedelta(hours=72)).max()),
                    horizon_details=stats)
                if base_pred is not None:
                    results[name]['prefix_base_only'] = d.point_metrics(ev, yy, base_pred, oo)
                    d.array_save(out / f'{name}_prefix_base_predictions.npz', pred=base_pred, origins=oo.asi8)
                outputs[name] = (oo, yy, pred)
                d.write(out / 'results.json', results)

            def pooled(stages):
                oo = outputs[stages[0]][0].append([outputs[n][0] for n in stages[1:]])
                yy = np.vstack([outputs[n][1] for n in stages])
                pp = np.vstack([outputs[n][2] for n in stages])
                return d.point_metrics(ev, yy, pp, oo), oo, pp

            def finish(state):
                verify_hashes(hashes)
                verify_hashes(parent['inherited_hashes'])
                d.write(out / 'results.json', results)
                (out / 'REPORT.md').write_text('# Transfer gate result\n\n'+state+'\n\n'+
                    ('SMOKE ONLY: forced stage access; no acceptance claim.\n\n' if smoke else '')+
                    '\n'.join(f'- {k}: {v}' for k,v in results.items() if not isinstance(v,dict))+'\n')
                artifacts = ['manifest.json','results.json'] + [p.name for p in out.glob('*predictions.npz')]
                d.write(out / 'completion.json', dict(sha256={str(out/f):d.digest(out/f) for f in artifacts}))
                d.status(out,state,smoke=smoke)

            for name,start,end in FOLDS:
                if name == 'confirmation':
                    passed = m.development_gate(results,baseline)
                    results['development_passed'] = passed
                    if not passed and not smoke:
                        finish('completed_rejected')
                        return
                ix = np.flatnonzero((origins>=pd.Timestamp(start,tz='UTC')) &
                                    (origins<pd.Timestamp(end,tz='UTC')+pd.Timedelta(days=1)))
                stage(name,start,[x[ix] for x in blocks],origins[ix],y[ix],anchor[ix],gains[ix])
                if name == 'dev2':
                    results['pooled'] = pooled(['dev0','dev1','dev2'])[0]
            results['confirmation_passed'] = m.gate(results['confirmation'],baseline['confirmation'],True)
            if not results['confirmation_passed'] and not smoke:
                finish('completed_rejected')
                return
            # Actual evaluation arrays are inaccessible until both gates pass (except marked smoke).
            series = pd.read_parquet(hpo.CACHE / 'speed.parquet')['speed']
            for name,grid,cutoff in [('f1','f1','2026-02-26'),('final','official','2026-05-29')]:
                query_bundle = dict(np.load(hpo.CACHE / f'{grid}.npz'))
                oo = pd.DatetimeIndex(query_bundle['origins'],tz='UTC')
                query = [hpo.base.design(query_bundle,names,parent['selected_groups'],b) for b in (0,24,48)]
                qa,qb,_ = m.decoder(query_bundle,names)
                stage(name,cutoff,query,oo,hpo.base.targets(series,oo),qa,qb)
            results['combined'],oo,pp = pooled(['f1','final'])
            if not smoke:
                frame = ev.frame(oo,pp)
                assert len(frame)==357840 and not frame.duplicated(ev.KEY).any()
                results['combined']['official'] = ev.official_score(frame,parent['parent'])
                np.testing.assert_allclose(results['combined']['mse'],results['combined']['official']['mse'],rtol=1e-12)
                frame.to_parquet(out/'predictions.parquet',index=False)
                reference = d.read(REFERENCE/'controls/results.json')['l2']['combined']
                results['transferred'] = m.gate(results['combined'],reference,True)
            finish('completed_smoke' if smoke else 'completed_evaluated')
        except Exception as error:
            d.status(out,'failed',error=str(error),traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane',required=True,choices=m.LANES)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    execute(args.lane,args.smoke)
