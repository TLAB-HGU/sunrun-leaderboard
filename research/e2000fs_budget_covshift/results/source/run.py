"""Frozen covariate-shift weighting; paired maturity-corrected point4 controls."""
import argparse
import csv
import fcntl
import hashlib
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
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LANE = ROOT / 'research/tail-budget-20261029/lanes/covshift'
EXPERIMENT = HERE.name
PARENT_DIR = ROOT / 'analysis/e2000fs_dirmo_24_feat30_pointweight4'
PARENT_MANIFEST = PARENT_DIR / 'runs/full_v1/manifest.json'
CACHE = ROOT / 'analysis/forecast_strategies_e2000/outputs/cache'


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parent = imported('covshift_parent', PARENT_DIR / 'run.py')
read, write, digest = parent.read, parent.write, parent.digest


def mature_rows(origins, cutoff):
    return np.flatnonzero(origins + pd.Timedelta(hours=144) <= cutoff)


def domain_rows(origins, cutoff):
    return np.flatnonzero(origins < cutoff)


def density_ratio(x, origins, cutoff, rounds=80):
    """Past-only blocked cross-fit of p_recent / p_all, with fold-specific priors."""
    idx = domain_rows(origins, cutoff)
    xx, oo = x[idx], origins[idx]
    recent = np.asarray(oo >= cutoff - pd.Timedelta(days=81))
    block = oo.asi8 // pd.Timedelta(days=27).value
    folds = block % 3
    # Embargo all contexts in the first/last 72h around a held block boundary.
    probabilities, raw = np.empty(len(idx)), np.empty(len(idx))
    receipts = []
    for fold in range(3):
        held = folds == fold
        if not held.any():
            raise ValueError('OOF fold is empty')
        held_blocks = np.unique(block[held])
        embargo = np.zeros(len(idx), dtype=bool)
        for value in held_blocks:
            start = value * pd.Timedelta(days=27).value
            end = (value + 1) * pd.Timedelta(days=27).value
            embargo |= (oo.asi8 >= start - pd.Timedelta(hours=72).value) & (oo.asi8 < end + pd.Timedelta(hours=72).value)
        source = np.flatnonzero(~embargo)
        target = np.flatnonzero(~embargo & recent)
        if not len(source) or not len(target):
            raise ValueError('Insufficient past source/recent contexts for blocked OOF')
        features = np.vstack([xx[source], xx[target]])
        labels = np.r_[np.zeros(len(source)), np.ones(len(target))]
        pi = len(target) / len(labels)
        model = xgb.train(dict(objective='binary:logistic', tree_method='hist', device='cpu',
                              max_depth=2, eta=.05, min_child_weight=20, reg_lambda=5,
                              subsample=1., colsample_bytree=1., seed=42, nthread=3, base_score=pi),
                          xgb.DMatrix(features, label=labels, nthread=3), num_boost_round=rounds)
        q = model.inplace_predict(xx[held]).astype(float)
        probabilities[held] = q
        raw[held] = q / np.maximum(1 - q, 1e-12) * len(source) / len(target)
        receipts.append(dict(fold=fold, source_count=len(source), target_count=len(target),
                             prior=pi, held_count=int(held.sum()), held_blocks=held_blocks.tolist()))
    ratio = np.clip(raw, .25, 4.)
    auc = roc_auc_score(np.r_[np.zeros(len(idx)), np.ones(recent.sum())],
                        np.r_[probabilities, probabilities[recent]])
    diag = dict(source_count=len(idx), target_count=int(recent.sum()), max_context=str(oo.max()),
                cutoff=str(cutoff), domain_auc=float(auc), folds=receipts,
                raw_quantiles=np.quantile(raw, [0,.01,.1,.5,.9,.99,1]).tolist(),
                clipped_quantiles=np.quantile(ratio, [0,.01,.1,.5,.9,.99,1]).tolist(),
                fraction_clip_low=float((raw < .25).mean()), fraction_clip_high=float((raw > 4).mean()),
                ess=float(ratio.sum()**2 / (ratio**2).sum()),
                recent_mean_ratio=float(ratio[recent].mean()), old_mean_ratio=float(ratio[~recent].mean()))
    return idx, ratio, diag, dict(origins_ns=oo.asi8, raw=raw, ratio=ratio, probability=probabilities, fold=folds)


def combined_weights(y, origins, ratio):
    base = parent.weights(y)
    if np.all(ratio == 1):
        return base.copy(), []
    result = base.astype(float) * np.asarray(ratio)[:, None]
    bands = np.digitize(y, [400., 550., 700.])
    eras = np.asarray(origins.year)
    receipt = []
    for era in np.unique(eras):
        for h in range(y.shape[1]):
            for band in range(4):
                mask = (eras == era) & (bands[:, h] == band)
                if not mask.any():
                    continue
                before = float(base[mask, h].sum(dtype=float))
                result[mask, h] *= before / result[mask, h].sum()
                after = float(result[mask, h].sum())
                np.testing.assert_allclose(before, after, rtol=1e-12)
                receipt.append(dict(era=int(era), horizon=h+1, speed_band=band, count=int(mask.sum()),
                                    parent_mass=before, candidate_mass=after))
    return result.astype(np.float32), receipt


def weighted_objective(weight):
    def objective(pred, matrix):
        y = matrix.get_label().reshape(pred.shape)
        return (pred-y)*weight, weight
    return objective


class History(parent.LossHistory):
    def after_iteration(self, model, epoch, evals_log):
        result = super().after_iteration(model, epoch, evals_log)
        if (epoch+1) % 25 == 0:
            write(self.path.parent.parent / 'progress.json', dict(loss_file=self.path.name, round=epoch+1,
                  pid=os.getpid(), utc=pd.Timestamp.now(tz='UTC').isoformat()))
        return result


def summary(y, p):
    err = p.astype(float)-y
    high = y >= 550
    result = dict(mse=float((err**2).mean()), n_pairs=int(y.size),
                  mse_by_band={f'h{i+1}-{i+24}':float((err[:,i:i+24]**2).mean()) for i in (0,24,48)})
    for tag, mask in [('ge550',high), ('lt550',~high)]:
        result.update({f'n_pairs_{tag}':int(mask.sum()), f'fraction_{tag}':float(mask.mean()),
                       f'mse_{tag}':float((err[mask]**2).mean()) if mask.any() else None,
                       f'bias_{tag}':float(err[mask].mean()) if mask.any() else None})
    return result


def sign(c, p):
    return c['mse'] < p['mse'] and c['mse_ge550'] is not None and c['mse_ge550'] <= p['mse_ge550']


def execute(out, smoke=False):
    out.mkdir(parents=True, exist_ok=False)
    for name in ('models','losses','diagnostics'):
        (out/name).mkdir()
    write(out/'status.json', dict(status='running', smoke=smoke, pid=os.getpid(), phase='hash_inputs'))
    try:
        frozen = read(PARENT_MANIFEST)
        for path, sha in {**frozen['source_sha256'], **frozen['input_identity']}.items():
            if digest(path) != sha:
                raise ValueError(f'Frozen parent source/input differs: {path}')
        hpo = imported('covshift_inputs', parent.HPO)
        ev = imported('covshift_eval', parent.EVALUATION)
        baseline, names, origins, y, blocks = hpo.load_data()
        os.environ['OMP_NUM_THREADS'] = '3'  # loader imports historically set 4; all models explicitly use3.
        assert baseline['selected'] == frozen['selected']
        rounds = 2 if smoke else 900
        recipe = dict(frozen['recipe']['xgboost'], n_estimators=rounds, n_jobs=3)
        params = {k:v for k,v in recipe.items() if k not in ('n_estimators','n_jobs','random_state')}
        params.update(nthread=3, seed=42, num_target=24)
        sources = list(HERE.glob('*.py')) + [LANE/'PREREGISTRATION.md']
        source_hashes = {**frozen['source_sha256'], **{str(p):digest(p) for p in sources}}
        manifest = dict(experiment=EXPERIMENT, smoke=smoke, selected=baseline['selected'],
                        source_sha256=source_hashes, parent_manifest_sha256=digest(PARENT_MANIFEST),
                        parent_source_sha256=frozen['source_sha256'], input_identity=frozen['input_identity'],
                        direct_parent=PARENT_DIR.name, historical_parent=dict(mse=3556.1984,mse_ge550=5923.5239,
                        timing='origin+72h<cutoff historical; not current target+72h maturity'),
                        recipe=dict(xgboost=recipe, rounds=rounds, maturity_hours=72, vector_origin_purge_hours=144,
                        ace_lag_hours=0, seed=42, classifier_rounds=80, recent_days=81, ratio_clip=[.25,4.],
                        mechanism='blockedOOF p_recent/p_all; era-year x horizon x true-band mass preservation'),
                        xgboost_version=xgb.__version__, python=sys.version)
        write(out/'manifest.json', manifest)
        series = pd.read_parquet(CACHE/'speed.parquet')['speed']
        stages = []
        for tag, start, end in [('D1','2024-12-04','2025-02-22'),('D2','2025-03-04','2025-05-22'),
                                 ('D3','2025-09-04','2025-11-22'),('C','2025-12-04','2026-02-22')]:
            cutoff = pd.Timestamp(start,tz='UTC')
            val = np.flatnonzero((origins>=cutoff)&(origins<=pd.Timestamp(end,tz='UTC')+pd.Timedelta(hours=23)))
            stages.append((tag,cutoff,[x[val] for x in blocks],origins[val],y[val]))
        for tag, grid, date in [('f1','f1','2026-02-26'),('final','official','2026-05-29')]:
            bundle = dict(np.load(CACHE/f'{grid}.npz'))
            oo = pd.DatetimeIndex(bundle['origins'],tz='UTC')
            stages.append((tag,pd.Timestamp(date,tz='UTC'),
                           [hpo.base.design(bundle,names,baseline['selected'],b) for b in (0,24,48)],oo,hpo.base.targets(series,oo)))
        results, outputs, timings, hashes = {}, {}, [], {}
        for tag,cutoff,query,oo,yy in stages:
            train = mature_rows(origins,cutoff)
            if smoke:
                train = train[np.linspace(0,len(train)-1,min(128,len(train)),dtype=int)]
                oo,yy,query = oo[:16],yy[:16],[x[:16] for x in query]
            assert (origins[train]+pd.Timedelta(hours=144)).max() <= cutoff
            assert np.isfinite(y[train]).all() and np.isfinite(yy).all()
            print(f'{tag}: domain fit; cutoff={cutoff}; mature train={len(train)} query={len(oo)}',flush=True)
            idx,ratio,diag,arrays = density_ratio(blocks[0],origins,cutoff,rounds=3 if smoke else 80)
            aligned = ratio[np.searchsorted(idx,train)]
            weight,mass = combined_weights(y[train],origins[train],aligned)
            base = parent.weights(y[train])
            diag.update(train_count=len(train), train_max_origin=str(origins[train].max()),
                        max_target_maturity=str((origins[train]+pd.Timedelta(hours=144)).max()),
                        train_index_sha256=hashlib.sha256(train.tobytes()).hexdigest(),
                        effective_ess_by_horizon=(weight.sum(axis=0)**2/(weight**2).sum(axis=0)).tolist(),
                        effective_multiplier_quantiles=np.quantile(weight/base,[0,.01,.5,.99,1]).tolist())
            write(out/'diagnostics'/f'{tag}_ratio.json',diag)
            write(out/'diagnostics'/f'{tag}_mass.json',mass)
            np.savez_compressed(out/'diagnostics'/f'{tag}_ratio.npz',**arrays,train_index=train)
            results[tag], outputs[tag] = {}, {}
            for arm in ('parent','candidate'):
                pred = np.empty_like(yy)
                for b in range(3):
                    sl = slice(b*24,(b+1)*24)
                    target = y[train,sl].astype(np.float32)
                    dt = xgb.DMatrix(blocks[b][train],label=target,nthread=3)
                    dv = xgb.DMatrix(query[b],label=yy[:,sl].astype(np.float32),nthread=3)
                    ww = (base if arm=='parent' else weight)[:,sl]
                    norm = np.where(target>=550,4.,1.).mean(axis=0)
                    def metric(p,matrix):
                        actual = matrix.get_label().reshape(p.shape)
                        return 'weighted_mse',float(np.mean(parent.weights(actual,norm)*(p-actual)**2))
                    write(out/'status.json',dict(status='running',smoke=smoke,stage=tag,arm=arm,block=b,pid=os.getpid()))
                    print(f'{tag} {arm} b{b}: TRAINING train={len(train)} rounds={rounds}',flush=True)
                    model = xgb.train(dict(params,base_score=float(target.mean())),dt,num_boost_round=rounds,
                        obj=parent.objective if arm=='parent' else weighted_objective(ww),
                        evals=[(dt,'train'),(dv,'validation')],custom_metric=metric,
                        callbacks=[History(out/'losses'/f'{tag}_{arm}_b{b}.csv')],verbose_eval=25)
                    path = out/'models'/f'{tag}_{arm}_b{b}.ubj'
                    model.save_model(path)
                    begin = time.perf_counter()
                    raw = model.inplace_predict(query[b])
                    elapsed = time.perf_counter()-begin
                    np.testing.assert_array_equal(raw,xgb.Booster(model_file=path).inplace_predict(query[b]))
                    pred[:,sl] = np.clip(raw,1.,3000.)
                    hashes[str(path.relative_to(out))] = digest(path)
                    if tag in ('f1','final') and arm=='candidate':
                        timings.append(dict(stage=tag,block=b,seconds=elapsed))
                assert np.isfinite(pred).all()
                outputs[tag][arm] = pred
                results[tag][arm] = summary(yy,pred)
                train_high = float((y[train]>=550).mean())
                results[tag][arm]['train_high_fraction'] = train_high
                results[tag][arm]['mse_rw'] = (train_high*results[tag][arm]['mse_ge550'] + (1-train_high)*results[tag][arm]['mse_lt550']) if (yy>=550).any() else None
                np.savez_compressed(out/f'{tag}_{arm}_predictions.npz',pred=pred,origins=oo.asi8)
            results[tag]['delta'] = {key:results[tag]['candidate'][key]-results[tag]['parent'][key] for key in ('mse','mse_ge550','mse_lt550') if results[tag]['parent'][key] is not None}
            # 27-day paired blocks, retaining complete origin trajectories.
            delta = (outputs[tag]['candidate']-yy)**2-(outputs[tag]['parent']-yy)**2
            pd.DataFrame(dict(origin=oo,block=oo.asi8//pd.Timedelta(days=27).value,delta_mse=delta.mean(axis=1))).groupby('block').agg(n_origins=('origin','size'),delta_mse=('delta_mse','mean')).to_csv(out/'diagnostics'/f'{tag}_paired_27day.csv')
            outputs[tag]['truth'],outputs[tag]['origins'] = yy,oo
            write(out/'results.json',results)
        for name,tags in [('development',['D1','D2','D3']),('combined',['f1','final'])]:
            actual = np.vstack([outputs[t]['truth'] for t in tags])
            results[name] = {}
            for arm in ('parent','candidate'):
                pred = np.vstack([outputs[t][arm] for t in tags])
                results[name][arm] = summary(actual,pred)
                if name=='combined':
                    oo = outputs['f1']['origins'].append(outputs['final']['origins'])
                    frame = ev.frame(oo,pred)
                    if not smoke:
                        assert len(frame)==357840 and not frame.duplicated(ev.KEY).any()
                        score = ev.official_score(frame,baseline)
                        results[name][arm]['official'] = score
                        np.testing.assert_allclose(results[name][arm]['mse'],score['mse'],rtol=1e-12)
                    frame.to_parquet(out/('predictions.parquet' if arm=='candidate' else 'parent_predictions.parquet'),index=False)
        internal = sign(results['development']['candidate'],results['development']['parent']) and sum(sign(results[t]['candidate'],results[t]['parent']) for t in ('D1','D2','D3'))>=2 and sign(results['C']['candidate'],results['C']['parent'])
        combined = results['combined']['candidate']
        results['decision'] = dict(internal_positive=bool(internal),official_always_run=True,
            target_achieved=bool(not smoke and combined['mse']<3500 and combined['mse_ge550']<6000),
            transferred=bool(not smoke and internal and sign(combined,results['combined']['parent'])))
        # Publication helper expects candidate scores directly at combined.official.
        if not smoke:
            results['combined']['official'] = combined['official']
        write(out/'results.json',results)
        write(out/'timing.json',dict(inference_seconds_per_fold=sum(t['seconds'] for t in timings)/sum(len(outputs[t]['origins']) for t in ('f1','final')),measurements=timings))
        write(out/'model_receipts.json',hashes)
        for path,sha in source_hashes.items():
            assert digest(path)==sha,f'Run source changed: {path}'
        write(out/'status.json',dict(status='completed',smoke=smoke,pid=os.getpid()))
        print('SMOKE PASSED' if smoke else 'TRAINING COMPLETE',flush=True)
    except BaseException:
        write(out/'status.json',dict(status='failed',smoke=smoke,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with (args.out.parent/(args.out.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(args.out.resolve(),args.smoke)
