"""Preregistered EDS-inspired complexity weighting; paired maturity-correct point4."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['OMP_NUM_THREADS'] = '3'
import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LANE = ROOT / 'research/tail-budget-20261029/lanes/complexity'
EXPERIMENT = HERE.name
PARENT_DIR = ROOT / 'analysis/e2000fs_dirmo_24_feat30_pointweight4'
spec = importlib.util.spec_from_file_location('complexity_exact_parent', PARENT_DIR / 'run.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
read, write, digest = p.read, p.write, p.digest
CONTEXT = ('v_lag0', 'ace_bt_nt_last', 'ace_density_cm3_last')
WINDOWS = [('D1', '2024-12-04', '2025-02-22'), ('D2', '2025-03-04', '2025-05-22'),
           ('D3', '2025-09-04', '2025-11-22'), ('C', '2025-12-04', '2026-02-22')]


def train_indices(origins, cutoff):
    return np.flatnonzero(origins + pd.Timedelta(hours=144) <= pd.Timestamp(cutoff))


def mass_weights(y, origins, raw, active):
    """Conserve parent mass in calendar-year / horizon / true-band groups."""
    base = p.weights(y)
    relative = np.ones(y.shape, dtype=np.float64)
    groups = []
    for year in np.unique(origins.year):
        for h in range(y.shape[1]):
            for high in (False, True):
                group = (origins.year == year) & ((y[:, h] >= 550) == high)
                chosen = group & active
                if chosen.sum() >= 32:
                    relative[chosen, h] = raw[chosen, h] / raw[chosen, h].mean()
                w = base[group, h].astype(float) * relative[group, h]
                groups.append(dict(year=int(year), horizon=h+1, high=high, n=int(group.sum()),
                                   active=int(chosen.sum()), parent_mass=float(base[group, h].astype(float).sum()),
                                   candidate_mass=float(w.sum()), ess=float(w.sum()**2/(w@w)) if len(w) else None))
    result = (base.astype(float) * relative).astype(np.float32)
    assert np.isfinite(result).all() and (result > 0).all()
    assert relative.min() >= .25-1e-12 and relative.max() <= 4+1e-12
    return result, dict(groups=groups, relative_min=float(relative.min()), relative_max=float(relative.max()),
                        ess_by_horizon=((result.astype(float).sum(axis=0)**2)/(result.astype(float)**2).sum(axis=0)).tolist(),
                        max_mass_relative_error=max((abs(g['candidate_mass']-g['parent_mass'])/max(g['parent_mass'],1) for g in groups), default=0.))


def density(previous, query):
    """Train-only octile histogram in three fixed causal context columns."""
    previous, query = np.asarray(previous, float), np.asarray(query, float)
    a, b = [], []
    for k in range(previous.shape[1]):
        finite = previous[:, k][np.isfinite(previous[:, k])]
        median = np.median(finite) if len(finite) else 0.
        historical = np.where(np.isfinite(previous[:, k]), previous[:, k], median)
        future = np.where(np.isfinite(query[:, k]), query[:, k], median)
        edges = np.unique(np.quantile(historical, np.arange(1, 8)/8))
        a.append(np.searchsorted(edges, historical))
        b.append(np.searchsorted(edges, future))
    ai = np.ravel_multi_index(tuple(a), (8, 8, 8))
    bi = np.ravel_multi_index(tuple(b), (8, 8, 8))
    counts = np.bincount(ai, minlength=512)
    return np.clip(np.sqrt(np.median(counts[counts > 0])/(counts[bi]+1)), .5, 2)


def objective_for(w):
    def objective(pred, matrix):
        target = matrix.get_label().reshape(pred.shape)
        return (pred-target)*w, w
    return objective


def forward_complexity(blocks, y, origins, context, params, rounds, out=None):
    raw = np.ones_like(y, dtype=float)
    active = np.zeros(len(y), dtype=bool)
    receipts = []
    started = time.perf_counter()
    quarters = np.linspace(0, len(y), 5, dtype=int)
    for fold in range(1, 4):
        val = np.arange(quarters[fold], quarters[fold+1])
        if not len(val):
            continue
        fit = train_indices(origins, origins[val[0]])
        assert not np.intersect1d(fit, val).size
        if len(fit) < 32:
            receipts.append(dict(fold=fold, neutral=True, fit_count=len(fit)))
            continue
        d = density(context[fit], context[val])
        prediction = np.empty((len(val), 72), dtype=np.float32)
        for b in range(3):
            sl = slice(24*b,24*(b+1))
            target = y[fit, sl].astype(np.float32)
            dt = xgb.DMatrix(blocks[b][fit], label=target, nthread=3)
            dv = xgb.DMatrix(blocks[b][val], label=y[val, sl], nthread=3)
            norm = np.where(target >= 550,4.,1.).mean(axis=0)
            def metric(pred, matrix):
                yy = matrix.get_label().reshape(pred.shape)
                return 'weighted_mse', float(np.mean(p.weights(yy, norm)*(pred-yy)**2))
            callbacks = [p.LossHistory(out / 'losses' / f'oof_fold{fold}_b{b}.csv')] if out else []
            print(f'OOF fold{fold} b{b}: train={len(fit)} query={len(val)} rounds={rounds}', flush=True)
            model = xgb.train(dict(params, base_score=float(target.mean())), dt,
                num_boost_round=rounds, obj=p.objective, evals=[(dt,'train'),(dv,'validation')],
                custom_metric=metric, callbacks=callbacks, verbose_eval=25 if out else False)
            prediction[:,sl] = np.clip(model.inplace_predict(blocks[b][val]),1,3000)
            if out:
                model.save_model(out / 'models' / f'oof_fold{fold}_b{b}.ubj')
        error = np.clip(np.abs(prediction-y[val])/100., 0, 3)
        raw[val] = np.clip(1+.25*(error*d[:,None]-1), .5, 2)
        active[val] = True
        receipts.append(dict(fold=fold, fit_count=len(fit), query_count=len(val),
            cutoff=str(origins[val[0]]), max_label_maturity=str((origins[fit]+pd.Timedelta(hours=144)).max()),
            train_index_sha256=__import__('hashlib').sha256(fit.tobytes()).hexdigest(),
            density_min=float(d.min()), density_max=float(d.max())))
    return raw, active, dict(seconds=time.perf_counter()-started, folds=receipts)


def metrics(ev, y, pred, origins):
    m = ev.metrics(y,pred,origins)
    error = pred-y
    low = y < 550
    m.update(mse_lt550=float(np.mean(error[low]**2)) if low.any() else None,
             bias=float(error.mean()), bias_lt550=float(error[low].mean()) if low.any() else None,
             n_pairs_lt550=int(low.sum()), high_fraction=float((~low).mean()))
    return m


def paired_summary(y, control, candidate, origins):
    """Paired 27-day origin-block bootstrap; no independent horizon claims."""
    delta = (candidate-y)**2-(control-y)**2
    block = ((origins.asi8-origins.asi8.min())//pd.Timedelta(days=27).value)
    sums = np.array([delta[block==k].sum() for k in np.unique(block)])
    counts = np.array([delta[block==k].size for k in np.unique(block)])
    rng = np.random.default_rng(42)
    sampled = rng.integers(0,len(sums),size=(1000,len(sums)))
    means = sums[sampled].sum(axis=1)/counts[sampled].sum(axis=1)
    return dict(delta_mse=float(delta.mean()), block_days=27, blocks=len(sums),
                bootstrap95=np.quantile(means,[.025,.975]).tolist())


def internal_decision(results):
    def good(key):
        a,b = results[key]['candidate'], results[key]['control']
        return a['mse'] < b['mse'] and a['mse_ge550'] is not None and b['mse_ge550'] is not None and a['mse_ge550'] <= b['mse_ge550']
    return dict(positive=bool(good('pooled_dev') and sum(good(s) for s in ('D1','D2','D3'))>=2 and good('C')),
                development_windows_positive=sum(good(s) for s in ('D1','D2','D3')),
                confirmation_positive=good('C'), official_unconditional=True)


def report(out, results, publication=None):
    lines = ['# e2000fs_budget_complexity', '',
             'EDS-inspired train-only OOF-error × causal-context density weighting; bounded and group-mass preserving. No simplex selection. Timing-correct point4 control uses identical origin+144h<=cutoff rows.', '',
             '| Stage | Control total | Candidate total | Delta | Control ≥550 | Candidate ≥550 | Delta |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for stage in ('D1','D2','D3','pooled_dev','C','f1','final','combined'):
        if stage not in results:
            continue
        c,a = results[stage]['control'], results[stage]['candidate']
        lines.append(f"| {stage} | {c['mse']:.4f} | {a['mse']:.4f} | {a['mse']-c['mse']:+.4f} | {c['mse_ge550']} | {a['mse_ge550']} | {a['mse_ge550']-c['mse_ge550'] if c['mse_ge550'] is not None and a['mse_ge550'] is not None else None} |")
    if 'decision' in results:
        lines += ['', 'Internal signal: '+json.dumps(results['decision'])]
    if 'outcome' in results:
        lines += ['', 'Official outcome: '+json.dumps(results['outcome']),
                  'Historical published parent (older maturity mask): total3556.1984 / ≥5505923.5239; not the paired control.']
    lines += ['', f'Artifacts: {out}', f'Manifest SHA256: {digest(out / "manifest.json")}',
              'Per-round losses: losses/*.csv (including auxiliary estimator); final loss/convergence: convergence.json.',
              'Hashes: manifest.json, model_receipts.json; grouped mass/ESS and auxiliary cost: weights/*.',
              'Submission receipt: '+(str(publication) if publication else 'pending; training/evaluation alone is not submission.'),
              'Limitations: previously exposed Top30 and evaluation periods; no fresh-holdout or future guarantee. Scientific outcome is separate from publication infrastructure status.']
    for path in (out/'REPORT.md', LANE/'REPORT.md'):
        path.write_text('\n'.join(lines)+'\n')


def execute(out, smoke=False):
    out.mkdir(parents=True, exist_ok=False)
    for sub in ('models','losses','weights'):
        (out/sub).mkdir()
    write(out/'status.json',dict(status='running',phase='load_frozen_inputs',smoke=smoke,pid=os.getpid()))
    try:
        parent = read(PARENT_DIR/'runs/full_v1/manifest.json')
        for name, sha in {**parent['source_sha256'], **parent['input_identity']}.items():
            if digest(name) != sha:
                raise ValueError(f'Frozen exact parent source/input differs: {name}')
        hpo = p.imported('complexity_inputs', p.HPO)
        ev = p.imported('complexity_eval', p.EVALUATION)
        baseline,names,origins,y,blocks = hpo.load_data()
        # Imported HPO module sets OMP4 at import: restore the lane's explicit limit.
        os.environ['OMP_NUM_THREADS'] = '3'
        bundle = dict(np.load(p.CACHE/'train.npz'))
        context = bundle['C'][:,[names['constant'].index(k) for k in CONTEXT]]
        recipe = parent['recipe']['xgboost'].copy()
        rounds = 2 if smoke else 900
        recipe.update(n_estimators=rounds,n_jobs=3)
        params = {k:v for k,v in recipe.items() if k not in ('n_estimators','n_jobs','random_state')}
        params.update(nthread=3,seed=42,num_target=24)
        sources = [HERE/n for n in ('run.py','execute.py','publish.py','test_run.py')]
        sources += [LANE/'PREREGISTRATION.md', p.HPO, Path(hpo.base.__file__),p.EVALUATION,
                    ROOT/'analysis/forecast_strategies_e2000/data.py',p.RECIPE,
                    ROOT/'analysis/forecast_strategies_kumo/publish.py']
        sources += [Path(k) for k in parent['source_sha256']]
        identity = dict(experiment=EXPERIMENT,smoke=smoke,selected=baseline['selected'],
            source_sha256={str(f):digest(f) for f in sources}, input_identity=parent['input_identity'],
            parent_manifest_sha256=digest(PARENT_DIR/'runs/full_v1/manifest.json'),
            recipe=dict(xgboost=recipe,rounds=rounds,mechanism='preregistered OOF error-density bounded weighting',
                        seed=42,ace_lag_hours=0,maturity_hours=72,vector_train_boundary='origin+144h<=cutoff'),
            direct_parent=PARENT_DIR.name,context=list(CONTEXT),xgboost_version=xgb.__version__,python=sys.version)
        write(out/'manifest.json',identity)
        series = pd.read_parquet(p.CACHE/'speed.parquet')['speed']
        stages = []
        for stage,start,end in WINDOWS:
            cutoff = pd.Timestamp(start,tz='UTC')
            val = np.flatnonzero((origins>=cutoff)&(origins<=pd.Timestamp(end+'T23:00:00Z')))
            stages.append((stage,cutoff,train_indices(origins,cutoff),[b[val] for b in blocks],origins[val],y[val]))
        for stage,grid,cutoff in [('f1','f1','2026-02-26'),('final','official','2026-05-29')]:
            cutoff = pd.Timestamp(cutoff,tz='UTC')
            query_bundle = dict(np.load(p.CACHE/f'{grid}.npz'))
            oo = pd.DatetimeIndex(query_bundle['origins'],tz='UTC')
            query = [hpo.base.design(query_bundle,names,baseline['selected'],b) for b in (0,24,48)]
            stages.append((stage,cutoff,train_indices(origins,cutoff),query,oo,hpo.base.targets(series,oo)))
        results,outputs,timings,convergence = {},{},[],{}
        for stage,cutoff,train,query,oo,yy in stages:
            if smoke:
                train = train[np.linspace(0,len(train)-1,min(128,len(train)),dtype=int)]
                oo,yy,query = oo[:16],yy[:16],[b[:16] for b in query]
            assert len(train) and len(oo) and (origins[train]+pd.Timedelta(hours=144)).max()<=cutoff
            stage_dir = out/'weights'/stage
            stage_dir.mkdir()
            (stage_dir/'models').mkdir()
            (stage_dir/'losses').mkdir()
            print(f'{stage}: train={len(train)} query={len(oo)} cutoff={cutoff} rounds={rounds}',flush=True)
            write(out/'status.json',dict(status='running',smoke=smoke,stage=stage,phase='forward_oof',pid=os.getpid()))
            raw,active,cost = forward_complexity([b[train] for b in blocks],y[train],origins[train],context[train],params,2 if smoke else 100,stage_dir)
            candidate_weights,trace = mass_weights(y[train],origins[train],raw,active)
            write(stage_dir/'trace.json',dict(**trace,auxiliary=cost,train_count=len(train),
                train_origin_sha256=__import__('hashlib').sha256(origins[train].asi8.tobytes()).hexdigest(),
                max_maturity=str((origins[train]+pd.Timedelta(hours=144)).max()),cutoff=str(cutoff)))
            np.savez_compressed(stage_dir/'weights.npz',raw=raw,active=active,weights=candidate_weights,train_indices=train)
            preds = {}
            for arm in ('control','candidate'):
                pred = np.empty_like(yy)
                for b in range(3):
                    sl = slice(24*b,24*(b+1))
                    tag = f'{arm}_{stage}_b{b}'
                    write(out/'status.json',dict(status='running',smoke=smoke,stage=stage,arm=arm,block=b,pid=os.getpid()))
                    target = y[train,sl].astype(np.float32)
                    dt = xgb.DMatrix(blocks[b][train],label=target,nthread=3)
                    dv = xgb.DMatrix(query[b],label=yy[:,sl],nthread=3)
                    w = p.weights(target) if arm=='control' else candidate_weights[:,sl]
                    norm = np.where(target>=550,4.,1.).mean(axis=0)
                    def metric(prediction,matrix):
                        actual = matrix.get_label().reshape(prediction.shape)
                        ww = w if matrix is dt else p.weights(actual,norm)
                        return 'weighted_mse',float(np.mean(ww*(prediction-actual)**2))
                    print(f'{tag}: training {rounds} rounds',flush=True)
                    model = xgb.train(dict(params,base_score=float(target.mean())),dt,num_boost_round=rounds,
                        obj=p.objective if arm=='control' else objective_for(w),
                        evals=[(dt,'train'),(dv,'validation')],custom_metric=metric,
                        callbacks=[p.LossHistory(out/'losses'/f'{tag}.csv')],verbose_eval=25)
                    model.save_model(out/'models'/f'{tag}.ubj')
                    begin = time.perf_counter()
                    prediction = model.inplace_predict(query[b])
                    seconds = time.perf_counter()-begin
                    np.testing.assert_array_equal(prediction,xgb.Booster(model_file=out/'models'/f'{tag}.ubj').inplace_predict(query[b]))
                    pred[:,sl] = np.clip(prediction,1,3000)
                    loss = pd.read_csv(out/'losses'/f'{tag}.csv')
                    convergence[tag] = loss.iloc[-1].to_dict()
                    write(out/'convergence.json',convergence)
                    if stage in ('f1','final') and arm=='candidate':
                        timings.append(dict(stage=stage,block=b,seconds=seconds,origins=len(oo)))
                assert np.isfinite(pred).all()
                np.savez_compressed(out/f'{arm}_{stage}_predictions.npz',pred=pred,origins=oo.asi8)
                preds[arm] = pred
            results[stage] = {arm:metrics(ev,yy,pred,oo) for arm,pred in preds.items()}
            high_share = float(np.mean(y[train]>=550))
            for arm in preds:
                m = results[stage][arm]
                m['mse_rw'] = high_share*m['mse_ge550']+(1-high_share)*m['mse_lt550'] if m['mse_ge550'] is not None and m['mse_lt550'] is not None else None
            results[stage]['paired'] = paired_summary(yy,preds['control'],preds['candidate'],oo)
            outputs[stage] = (oo,yy,preds)
            if stage=='C':
                dev_o = outputs['D1'][0].append([outputs[s][0] for s in ('D2','D3')])
                dev_y = np.vstack([outputs[s][1] for s in ('D1','D2','D3')])
                results['pooled_dev'] = {arm:metrics(ev,dev_y,np.vstack([outputs[s][2][arm] for s in ('D1','D2','D3')]),dev_o) for arm in preds}
                results['decision'] = internal_decision(results)
                write(out/'internal_decision.json',results['decision'])
            write(out/'paired_results.json',results)
            report(out,results)
        oo = outputs['f1'][0].append(outputs['final'][0])
        yy = np.vstack([outputs[s][1] for s in ('f1','final')])
        results['combined'] = {}
        for arm in ('control','candidate'):
            pred = np.vstack([outputs[s][2][arm] for s in ('f1','final')])
            frame = ev.frame(oo,pred)
            results['combined'][arm] = metrics(ev,yy,pred,oo)
            if not smoke:
                assert len(frame)==357840 and not frame.duplicated(ev.KEY).any()
                results['combined'][arm]['official'] = ev.official_score(frame,baseline)
                np.testing.assert_allclose(results['combined'][arm]['mse'],results['combined'][arm]['official']['mse'],rtol=1e-12)
            frame.to_parquet(out/('predictions.parquet' if arm=='candidate' else 'control_predictions.parquet'),index=False)
        a,c = results['combined']['candidate'],results['combined']['control']
        results['outcome'] = dict(target=bool(not smoke and a['mse']<3500 and a['mse_ge550']<6000),
            transferred=bool(not smoke and results['decision']['positive'] and a['mse']<c['mse'] and a['mse_ge550']<=c['mse_ge550']),smoke=smoke)
        write(out/'paired_results.json',results)
        # Compatibility interface for the archived publisher: candidate only.
        write(out/'results.json',{**{s:results[s]['candidate'] for s in ('f1','final','combined')},'internal_signal':results['decision']})
        seconds = sum(t['seconds'] for t in timings)/sum(len(outputs[s][0]) for s in ('f1','final'))
        write(out/'timing.json',dict(inference_seconds_per_fold=seconds,measurements=timings))
        write(out/'model_receipts.json',{str(f.relative_to(out)):digest(f) for f in sorted(out.rglob('*.ubj'))})
        for name,sha in identity['source_sha256'].items():
            assert digest(name)==sha,f'Run source changed: {name}'
        write(out/'status.json',dict(status='completed',smoke=smoke,pid=os.getpid()))
        report(out,results)
        print('SMOKE PASSED' if smoke else 'TRAINING COMPLETE; publication follows unconditionally',flush=True)
    except BaseException:
        write(out/'status.json',dict(status='failed',smoke=smoke,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with (args.out.parent/(args.out.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(args.out.resolve(),args.smoke)
