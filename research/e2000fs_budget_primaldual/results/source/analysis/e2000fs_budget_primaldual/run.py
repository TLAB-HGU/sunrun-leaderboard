"""Single preregistered primal-dual risk-budget adaptation of vector DirMO-24."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['OMP_NUM_THREADS'] = '3'
import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RECORD = ROOT / 'research/tail-budget-20261029/lanes/primaldual'
EXPERIMENT = HERE.name
PARENT_DIR = ROOT / 'analysis/e2000fs_dirmo_24_feat30_pointweight4'
CACHE = ROOT / 'analysis/forecast_strategies_e2000/outputs/cache'


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parent = imported('primaldual_parent', PARENT_DIR / 'run.py')
write, read, digest = parent.write, parent.read, parent.digest


def train_indices(origins, cutoff):
    return np.flatnonzero(origins + pd.Timedelta(hours=144) <= pd.Timestamp(cutoff))


def calibration_indices(origins, cutoff):
    start = pd.Timestamp(cutoff) - pd.Timedelta(days=90)
    reference = train_indices(origins, start)
    calibration = np.flatnonzero((origins >= start) & (origins + pd.Timedelta(hours=144) <= pd.Timestamp(cutoff)))
    return reference, calibration


class RiskObjective:
    def __init__(self, target, budget):
        self.high = np.asarray(target) >= 550.
        self.p = float(self.high.mean())
        self.budget = float(budget)
        if not np.isfinite(self.budget) or self.budget < 1:
            raise ValueError('Budget must be finite and at least one')
        self.dual = 3. * self.p
        self.used = self.dual

    def weights(self):
        if not self.p:
            return np.ones(self.high.shape, dtype=np.float32)
        return ((1. + self.dual * self.high / self.p) / (1. + self.dual)).astype(np.float32)

    def __call__(self, pred, matrix):
        self.used = self.dual
        y = matrix.get_label().reshape(pred.shape)
        a = self.weights()
        return (pred - y) * a, a

    def update(self, high_risk):
        self.dual = float(np.clip(self.dual + .02 * (high_risk / self.budget - 1.), 0., 10.)) if self.p else 0.
        return self.dual


class History(xgb.callback.TrainingCallback):
    def __init__(self, path, objective=None):
        self.path, self.objective = path, objective

    def after_iteration(self, model, epoch, log):
        row = dict(round=epoch + 1)
        for split, metrics in log.items():
            for key, values in metrics.items():
                row[f'{split}_{key}'] = values[-1]
        if self.objective:
            obj = self.objective
            high = log['train']['high_mse'][-1]
            row.update(dual_used=obj.used, budget=obj.budget, empirical_feasible=high <= obj.budget,
                       dual_next=obj.update(high))
        with self.path.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if epoch == 0:
                writer.writeheader()
            writer.writerow(row)
        return False


def fit_model(out, key, x, y, q, yy, params, rounds, objective=None, calibration=None):
    dt = xgb.DMatrix(x, label=y, nthread=3)
    dv = xgb.DMatrix(q, label=yy, nthread=3)
    norm = np.where(y >= 550., 4., 1.).mean(axis=0)
    def metric(p, matrix):
        target = matrix.get_label().reshape(p.shape)
        error = (p - target) ** 2
        high = target >= 550.
        return [('mse', float(error.mean())),
                ('high_mse', float(error[high].mean()) if high.any() else 0.),
                ('point4_mse', float((parent.weights(target, norm) * error).mean()))]
    evals = [(dt, 'train'), (dv, 'validation')]
    if calibration is not None:
        cx, cy = calibration
        evals.append((xgb.DMatrix(cx, label=cy, nthread=3), 'calibration'))
    path = out / 'models' / f'{key}.ubj'
    begin = time.perf_counter()
    model = xgb.train(dict(params, base_score=float(y.mean())), dt, num_boost_round=rounds,
                      obj=objective if objective is not None else parent.objective,
                      evals=evals, custom_metric=metric, verbose_eval=100,
                      callbacks=[History(out / 'losses' / f'{key}.csv', objective)])
    model.save_model(path)
    start = time.perf_counter()
    raw = model.inplace_predict(q)
    seconds = time.perf_counter() - start
    np.testing.assert_array_equal(raw, xgb.Booster(model_file=path).inplace_predict(q))
    print(f'{key}: finished {rounds} rounds reload parity passed elapsed={time.perf_counter()-begin:.1f}s', flush=True)
    return np.clip(raw, 1., 3000.), seconds


def metrics(y, pred):
    high = y >= 550.
    error = pred - y
    return dict(mse=float(np.mean(error**2)), mse_ge550=float(np.mean(error[high]**2)) if high.any() else None,
                mse_lt550=float(np.mean(error[~high]**2)) if (~high).any() else None,
                n_pairs=y.size, n_pairs_ge550=int(high.sum()), high_fraction=float(high.mean()),
                bias=float(error.mean()), bias_ge550=float(error[high].mean()) if high.any() else None,
                mse_by_band={str(b):float(np.mean(error[:,b:b+24]**2)) for b in (0,24,48)})


def execute(out, smoke=False):
    out.mkdir(parents=True, exist_ok=False)
    for name in ('models', 'losses', 'indices'):
        (out / name).mkdir()
    frozen = read(RECORD / 'preregistration.json')
    for path, sha in frozen['parent_and_inputs'].items():
        assert digest(path) == sha, f'Frozen parent/source/input differs: {path}'
    hpo = imported('primaldual_inputs', parent.HPO)
    ev = imported('primaldual_evaluation', parent.EVALUATION)
    baseline, names, origins, y, blocks = hpo.load_data()
    os.environ['OMP_NUM_THREADS'] = '3'  # read-only imported loader sets its legacy default4
    recipe = read(parent.RECIPE)['recipe'].copy()
    rounds = 2 if smoke else 900
    recipe.update(n_estimators=rounds, n_jobs=3, multi_strategy='multi_output_tree')
    params = {k:v for k,v in recipe.items() if k not in ('n_estimators', 'n_jobs', 'random_state')}
    params.update(nthread=3, seed=42, num_target=24)
    sources = [HERE/n for n in ('run.py','execute.py','publish.py','test_run.py')]
    sources += [RECORD / 'PREREGISTRATION.md', RECORD / 'preregistration.json', parent.HPO,
                Path(hpo.base.__file__), parent.EVALUATION, parent.RECIPE, PARENT_DIR / 'run.py',
                ROOT/'analysis/forecast_strategies_e2000/data.py', ROOT/'analysis/forecast_strategies_kumo/publish.py']
    identity = dict(experiment=EXPERIMENT, smoke=smoke, selected=baseline['selected'],
                    source_sha256={str(p):digest(p) for p in sources},
                    input_identity=frozen['parent_and_inputs'],
                    recipe=dict(xgboost=recipe, rounds=rounds, seed=42, ace_lag_hours=0, maturity_hours=72,
                                vector_maturity_hours=144, loss='preregistered primal-dual risk budget',
                                budget_calibration_days=90, dual_step=.02, dual_clip=[0,10]),
                    direct_parent=PARENT_DIR.name, historical_published=[3556.1984,5923.5239],
                    historical_note='published comparison has different maturity; method paired to corrected control',
                    xgboost_version=xgb.__version__, python=sys.version)
    write(out / 'manifest.json', identity)
    stages=[]
    for name, start, end in [('D1','2024-12-04','2025-02-22'),('D2','2025-03-04','2025-05-22'),
                             ('D3','2025-09-04','2025-11-22'),('C','2025-12-04','2026-02-22')]:
        cutoff=pd.Timestamp(start,tz='UTC')
        idx=np.flatnonzero((origins>=cutoff)&(origins<=pd.Timestamp(end+'T23:00:00Z')))
        stages.append((name,cutoff,[x[idx] for x in blocks],origins[idx],y[idx]))
    series=pd.read_parquet(CACHE/'speed.parquet')['speed']
    for stage,grid,cutoff in [('f1','f1','2026-02-26'),('final','official','2026-05-29')]:
        bundle=dict(np.load(CACHE/f'{grid}.npz'))
        oo=pd.DatetimeIndex(bundle['origins'],tz='UTC')
        query=[hpo.base.design(bundle,names,baseline['selected'],b) for b in (0,24,48)]
        stages.append((stage,pd.Timestamp(cutoff,tz='UTC'),query,oo,hpo.base.targets(series,oo)))
    results, outputs, times = {}, {}, []
    for stage,cutoff,query,oo,yy in stages:
        train=train_indices(origins,cutoff)
        ref,cal=calibration_indices(origins,cutoff)
        if smoke:
            train,ref,cal=[a[np.linspace(0,len(a)-1,min(128,len(a)),dtype=int)] for a in (train,ref,cal)]
            oo,yy,query=oo[:8],yy[:8],[x[:8] for x in query]
        assert len(train) and len(ref) and len(cal) and len(oo)
        assert np.isfinite(y[train]).all() and np.isfinite(yy).all()
        np.savez_compressed(out/'indices'/f'{stage}.npz',train=train,reference=ref,calibration=cal,
                            train_origins=origins[train].asi8,reference_origins=origins[ref].asi8,
                            calibration_origins=origins[cal].asi8,cutoff=cutoff.value)
        print(f'{stage}: TRAINING train={len(train)} reference={len(ref)} calibration={len(cal)} query={len(oo)} rounds={rounds}',flush=True)
        predictions={arm:np.empty_like(yy) for arm in ('control','candidate')}
        budgets=[]
        for b in range(3):
            sl=slice(b*24,(b+1)*24)
            write(out/'status.json',dict(status='running',stage=stage,block=b,smoke=smoke,pid=os.getpid()))
            target=y[train,sl].astype(np.float32)
            common=(params,rounds)
            reference,_=fit_model(out,f'{stage}_reference_b{b}',blocks[b][ref],y[ref,sl].astype(np.float32),
                                  blocks[b][cal],y[cal,sl].astype(np.float32),*common)
            high=y[cal,sl]>=550.
            if not high.any():
                raise ValueError(f'No high labels in training calibration: {stage} block{b}')
            budget=max(1.,float(np.mean((reference-y[cal,sl])[high]**2)))
            budgets.append(dict(block=b,budget=budget,calibration_high_count=int(high.sum())))
            write(out/f'{stage}_budgets.json',budgets)
            for arm in ('control','candidate'):
                objective=RiskObjective(target,budget) if arm=='candidate' else None
                predictions[arm][:,sl],seconds=fit_model(out,f'{stage}_{arm}_b{b}',blocks[b][train],target,
                    query[b],yy[:,sl].astype(np.float32),*common,objective=objective,
                    calibration=(blocks[b][cal],y[cal,sl].astype(np.float32)))
                if stage in ('f1','final') and arm=='candidate':
                    times.append(seconds)
        results[stage]={arm:metrics(yy,p) for arm,p in predictions.items()}
        results[stage]['train_high_fraction']=float((y[train]>=550.).mean())
        for arm,m in [(a,results[stage][a]) for a in predictions]:
            p=results[stage]['train_high_fraction']
            m['mse_rw']=p*m['mse_ge550']+(1-p)*m['mse_lt550'] if m['mse_ge550'] is not None and m['mse_lt550'] is not None else None
            np.savez_compressed(out/f'{stage}_{arm}_predictions.npz',pred=predictions[arm],origins=oo.asi8)
        # Paired 27-day origin blocks; descriptive, no independence assumption for 72h cells.
        delta=(predictions['candidate']-yy)**2-(predictions['control']-yy)**2
        pd.DataFrame({'origin':oo,'delta_mse':delta.mean(axis=1)}).assign(
            block=((oo-oo.min()).total_seconds()//(27*86400)).astype(int)).groupby('block').agg(
                origins=('origin','size'),delta_mse=('delta_mse','mean')).to_csv(out/f'{stage}_paired_blocks.csv')
        outputs[stage]=(oo,yy,predictions)
        write(out/'paired_results.json',results)
    for aggregate,keys in [('pooled_development',('D1','D2','D3')),('combined',('f1','final'))]:
        oo=outputs[keys[0]][0].append([outputs[k][0] for k in keys[1:]])
        yy=np.vstack([outputs[k][1] for k in keys])
        results[aggregate]={}
        for arm in ('control','candidate'):
            pred=np.vstack([outputs[k][2][arm] for k in keys])
            results[aggregate][arm]=metrics(yy,pred)
            if aggregate=='combined':
                frame=ev.frame(oo,pred)
                if not smoke:
                    assert len(frame)==357840 and not frame.duplicated(ev.KEY).any()
                    results[aggregate][arm]['official']=ev.official_score(frame,baseline)
                    np.testing.assert_allclose(results[aggregate][arm]['mse'],results[aggregate][arm]['official']['mse'],rtol=1e-12)
                frame.to_parquet(out/('predictions.parquet' if arm=='candidate' else 'control_predictions.parquet'),index=False)
    write(out/'paired_results.json',results)
    # Compatible publication surface contains only candidate metrics and immutable official score.
    write(out/'results.json',{k:v['candidate'] for k,v in results.items()})
    write(out/'timing.json',dict(inference_seconds_per_fold=sum(times)/sum(len(outputs[k][0]) for k in ('f1','final'))))
    write(out/'model_receipts.json',{str(p.relative_to(out)):digest(p) for p in sorted((out/'models').glob('*.ubj'))})
    for p,sha in identity['source_sha256'].items():
        assert digest(p)==sha,f'Run source changed: {p}'
    write(out/'status.json',dict(status='completed',smoke=smoke,pid=os.getpid()))
    print('SMOKE PASSED' if smoke else 'TRAINING COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    execute(args.out.resolve(),args.smoke)
