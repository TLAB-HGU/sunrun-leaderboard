"""Fixed paired point4/FreDF-inspired direct24 experiment; no score gates."""
import argparse
import csv
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LANE = ROOT / 'research/tail-budget-20261029/lanes/spectral'
EXPERIMENT = HERE.name
PARENT_DIR = ROOT / 'analysis/e2000fs_dirmo_24_feat30_pointweight4'
COEFFICIENT = .10
SCALE = 50.


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parent = imported('frozen_point4_spectral', PARENT_DIR / 'run.py')
read, write, digest, weights = parent.read, parent.write, parent.digest, parent.weights


def spectral_loss(error, scale=SCALE):
    if scale <= 0:
        raise ValueError('positive smoothing scale required')
    z = np.fft.fft(error, axis=1, norm='ortho')
    # hypot avoids overflow; subtraction is benign at the chosen physical scale.
    return scale**2 * np.sum(np.sqrt(1 + abs(z)**2 / scale**2) - 1, axis=1)


def derivatives(pred, target, coefficient=COEFFICIENT):
    w = weights(target)
    error = pred - target
    if coefficient == 0:
        return error * w, w
    z = np.fft.fft(error, axis=1, norm='ortho')
    rho = 1 / np.sqrt(1 + abs(z)**2 / SCALE**2)
    grad = error * w + coefficient * np.fft.ifft(z * rho, axis=1, norm='ortho').real
    return grad.astype(np.float32), (w + coefficient).astype(np.float32)


def objective(pred, matrix):
    return derivatives(pred, matrix.get_label().reshape(pred.shape))


def mature_indices(origins, cutoff):
    return np.flatnonzero(origins + pd.Timedelta(hours=144) <= pd.Timestamp(cutoff))


def summary(y, pred, train_high_fraction):
    error = pred.astype(float) - y
    high = y >= 550
    mse = lambda mask: float(np.mean(error[mask]**2)) if mask.any() else None
    low, fast = mse(~high), mse(high)
    return dict(mse=float(np.mean(error**2)), mse_ge550=fast, mse_lt550=low,
                n_pairs=int(y.size), n_pairs_ge550=int(high.sum()),
                high_fraction=float(high.mean()), bias=float(error.mean()),
                bias_ge550=float(error[high].mean()) if high.any() else None,
                bias_lt550=float(error[~high].mean()) if (~high).any() else None,
                mse_rw=(1-train_high_fraction)*low + train_high_fraction*fast if low is not None and fast is not None else None,
                train_high_fraction=train_high_fraction,
                mse_by_band={f'h{b+1}-{b+24}': float(np.mean(error[:, b:b+24]**2)) for b in (0,24,48)})


def status(**kwargs):
    now = pd.Timestamp.now(tz='UTC').isoformat()
    write(LANE / 'progress.json', dict(time=now, pid=os.getpid(), **kwargs))
    (LANE / 'STATUS.md').write_text('# spectral\n\n' + json.dumps(dict(time=now, pid=os.getpid(), **kwargs), indent=2) + '\n\nOwned paths only; no other lane edits. Published parent historical: 3556.1984/5923.5239.\n')


class History(xgb.callback.TrainingCallback):
    def __init__(self, path, stage, arm, block):
        self.path, self.stage, self.arm, self.block = path, stage, arm, block
        self.last = 0

    def after_iteration(self, model, epoch, evals_log):
        row = {'round': epoch+1}
        for dataset, metrics in evals_log.items():
            for metric, values in metrics.items():
                row[f'{dataset}_{metric}'] = values[-1]
        with self.path.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if epoch == 0:
                writer.writeheader()
            writer.writerow(row)
        if time.monotonic()-self.last > 240:
            status(status='training', stage=self.stage, arm=self.arm, block=self.block, round=epoch+1)
            self.last = time.monotonic()
        return False


def execute(out, smoke=False):
    out.mkdir(parents=True, exist_ok=False)
    for name in ('models', 'losses', 'traces'):
        (out / name).mkdir()
    status(status='loading', out=str(out), smoke=smoke)
    frozen = read(PARENT_DIR / 'runs/full_v1/manifest.json')
    for mapping in ('source_sha256', 'input_identity'):
        for path, expected in frozen[mapping].items():
            if digest(path) != expected:
                raise ValueError(f'Frozen parent {mapping} changed: {path}')
    hpo = imported('spectral_inputs', parent.HPO)
    os.environ['OMP_NUM_THREADS'] = '3'  # imported historical loader sets 4
    ev = imported('spectral_evaluation', parent.EVALUATION)
    baseline, names, origins, y, blocks = hpo.load_data()
    assert baseline['selected'] == frozen['selected']
    rounds = 2 if smoke else 900
    recipe = dict(frozen['recipe']['xgboost'], n_jobs=3, n_estimators=rounds)
    params = {k:v for k,v in recipe.items() if k not in ('n_jobs','n_estimators','random_state')}
    params.update(nthread=3, seed=42, num_target=24)
    sources = [HERE / n for n in ('run.py','execute.py','publish.py','test_run.py')]
    sources += [LANE / 'PREREGISTRATION.md', PARENT_DIR / 'runs/full_v1/manifest.json']
    source_hashes = dict(frozen['source_sha256'])
    source_hashes.update({str(p):digest(p) for p in sources})
    manifest = dict(experiment=EXPERIMENT, smoke=smoke, selected=baseline['selected'],
                    source_sha256=source_hashes, input_identity=frozen['input_identity'],
                    direct_parent=PARENT_DIR.name, parent_manifest_sha256=digest(PARENT_DIR / 'runs/full_v1/manifest.json'),
                    recipe=dict(xgboost=recipe, rounds=rounds, lambda_spectral=COEFFICIENT,
                                smoothing_scale_kms=SCALE, fft='full complex unitary24',
                                diagonal_majorizer='parent weights + lambda', ace_lag_hours=0,
                                maturity_hours=72, train_mask='origin+144h<=cutoff'),
                    xgboost_version=xgb.__version__, python=sys.version,
                    historical_published_parent=dict(mse=3556.1984,mse_ge550=5923.5239,
                                                    limitation='different training maturity; not paired control'))
    write(out / 'manifest.json', manifest)
    series = pd.read_parquet(parent.CACHE / 'speed.parquet')['speed']
    stages = []
    for name, start, end in [('D1','2024-12-04','2025-02-22'),('D2','2025-03-04','2025-05-22'),('D3','2025-09-04','2025-11-22'),('C','2025-12-04','2026-02-22')]:
        cutoff = pd.Timestamp(start, tz='UTC')
        val = np.flatnonzero((origins >= cutoff) & (origins <= pd.Timestamp(end, tz='UTC')+pd.Timedelta(hours=23)))
        stages.append((name, cutoff, mature_indices(origins, cutoff), [x[val] for x in blocks], origins[val], y[val]))
    for stage, grid, date in [('f1','f1','2026-02-26'),('final','official','2026-05-29')]:
        cutoff = pd.Timestamp(date, tz='UTC')
        bundle = dict(np.load(parent.CACHE / f'{grid}.npz'))
        oo = pd.DatetimeIndex(bundle['origins'], tz='UTC')
        query = [hpo.base.design(bundle, names, baseline['selected'], b) for b in (0,24,48)]
        stages.append((stage, cutoff, mature_indices(origins, cutoff), query, oo, hpo.base.targets(series,oo)))
    results, outputs, timings, hashes = {}, {}, [], {}
    for stage, cutoff, train, query, oo, yy in stages:
        if smoke:
            train = train[np.linspace(0,len(train)-1,min(128,len(train)),dtype=int)]
            oo, yy, query = oo[:16], yy[:16], [x[:16] for x in query]
        assert len(train) and len(oo) and (origins[train]+pd.Timedelta(hours=144)).max()<=cutoff
        assert np.isfinite(y[train]).all() and np.isfinite(yy).all()
        train_receipt = dict(cutoff=str(cutoff), count=len(train), origins_sha256=hashlib.sha256(origins[train].asi8.tobytes()).hexdigest(),
                             targets_sha256=hashlib.sha256(y[train].tobytes()).hexdigest(),
                             max_origin=str(origins[train].max()), max_target_mature=str(origins[train].max()+pd.Timedelta(hours=144)),
                             query_origins_sha256=hashlib.sha256(oo.asi8.tobytes()).hexdigest())
        write(out / 'traces' / f'{stage}_rows.json', train_receipt)
        results[stage] = {}
        for arm in ('control','spectral'):
            print(f'{stage} {arm}: train={len(train)} eval={len(oo)} rounds={rounds}', flush=True)
            pred = np.empty_like(yy)
            for b in range(3):
                sl = slice(b*24,(b+1)*24)
                target = y[train,sl].astype(np.float32)
                dt = xgb.DMatrix(blocks[b][train], label=target, nthread=3)
                dv = xgb.DMatrix(query[b], label=yy[:,sl].astype(np.float32), nthread=3)
                normalizer = np.where(target>=550.,4.,1.).mean(axis=0)
                def metric(p, matrix):
                    actual = matrix.get_label().reshape(p.shape)
                    point = float(np.mean(weights(actual,normalizer)*(p-actual)**2))
                    spec = float(np.mean(spectral_loss(p-actual))/24)
                    return [('weighted_mse',point),('spectral_loss',spec),('joint_loss',.5*point+(.1 if arm=='spectral' else 0)*spec)]
                objective_fn = objective if arm=='spectral' else parent.objective
                settings = dict(params,base_score=float(target.mean()))
                status(status='training', stage=stage, arm=arm, block=b, round=0, smoke=smoke)
                model = xgb.train(settings,dt,num_boost_round=rounds,obj=objective_fn,
                    evals=[(dt,'train'),(dv,'validation')],custom_metric=metric,
                    callbacks=[History(out/'losses'/f'{stage}_{arm}_b{b}.csv',stage,arm,b)],verbose_eval=100)
                path=out/'models'/f'{stage}_{arm}_b{b}.ubj'
                model.save_model(path)
                reload=xgb.Booster(model_file=path)
                begin=time.perf_counter()
                raw=model.inplace_predict(query[b])
                elapsed=time.perf_counter()-begin
                np.testing.assert_array_equal(raw,reload.inplace_predict(query[b]))
                assert raw.shape==(len(oo),24)
                pred[:,sl]=np.clip(raw,1.,3000.)
                hashes[str(path.relative_to(out))]=digest(path)
                residual=model.predict(dt)-target
                rho=1/np.sqrt(1+abs(np.fft.fft(residual,axis=1,norm='ortho'))**2/SCALE**2)
                write(out/'traces'/f'{stage}_{arm}_b{b}.json',dict(normalizer=normalizer.tolist(),
                    rho_min=float(rho.min()),rho_mean=float(rho.mean()),rho_max=float(rho.max()),
                    spectral_hessian_eigenvalue_bound=[0,COEFFICIENT],majorizer_spectral_diagonal=COEFFICIENT,
                    reload_parity=True,train_rows=train_receipt,base_score=float(target.mean())))
                if stage in ('f1','final') and arm=='spectral':
                    timings.append(dict(stage=stage,block=b,seconds=elapsed))
                print(f'{stage} {arm} b{b}: saved; reload parity passed',flush=True)
            assert np.isfinite(pred).all()
            np.savez_compressed(out/f'{stage}_{arm}_predictions.npz',pred=pred,origins=oo.asi8)
            results[stage][arm]=summary(yy,pred,float(np.mean(y[train]>=550)))
            outputs[(stage,arm)]=(oo,yy,pred)
            write(out/'results.json',results)
        difference=(outputs[(stage,'spectral')][2]-yy)**2-(outputs[(stage,'control')][2]-yy)**2
        # Date blocks, 27 days; descriptive differences, no independent-horizon test.
        groups=(oo-oo.min()).days//27
        pd.DataFrame([dict(block=int(g),origins=int((groups==g).sum()),mean_loss_delta=float(difference[groups==g].mean())) for g in np.unique(groups)]).to_csv(out/f'{stage}_paired_27day.csv',index=False)
    for arm in ('control','spectral'):
        oo=outputs[('f1',arm)][0].append(outputs[('final',arm)][0])
        yy=np.vstack([outputs[(s,arm)][1] for s in ('f1','final')])
        pred=np.vstack([outputs[(s,arm)][2] for s in ('f1','final')])
        frame=ev.frame(oo,pred)
        results.setdefault('combined',{})[arm]=summary(yy,pred,0.)
        if not smoke:
            assert len(frame)==357840 and not frame.duplicated(ev.KEY).any()
            official=ev.official_score(frame,baseline)
            np.testing.assert_allclose(results['combined'][arm]['mse'],official['mse'],rtol=1e-12)
            results['combined'][arm]['official']=official
        frame.to_parquet(out/('predictions.parquet' if arm=='spectral' else 'control_predictions.parquet'),index=False)
    # Publication interface uses the single candidate's protected combined score.
    if not smoke:
        results['combined']['official']=results['combined']['spectral']['official']
    internal=[]
    for arm in ('control','spectral'):
        yy=np.vstack([outputs[(s,arm)][1] for s in ('D1','D2','D3')])
        pp=np.vstack([outputs[(s,arm)][2] for s in ('D1','D2','D3')])
        internal.append(summary(yy,pp,0.))
    results['pooled_development']=dict(zip(('control','spectral'),internal))
    def good(pair):
        a,b=pair['control'],pair['spectral']
        return (a['mse_ge550'] is not None and b['mse_ge550'] is not None
                and b['mse']<a['mse'] and b['mse_ge550']<=a['mse_ge550'])
    signal=good(results['pooled_development']) and sum(good(results[s]) for s in ('D1','D2','D3'))>=2 and good(results['C'])
    results['decision']=dict(internal_positive=signal,transferred=signal and good(results['combined']),
        target_achieved=not smoke and results['combined']['spectral']['mse']<3500 and results['combined']['spectral']['mse_ge550']<6000,
        internal_never_gates_official=True)
    write(out/'results.json',results)
    write(out/'timing.json',dict(inference_seconds_per_fold=sum(t['seconds'] for t in timings)/sum(len(outputs[(s,'spectral')][0]) for s in ('f1','final')),measurements=timings))
    write(out/'model_receipts.json',hashes)
    for path,expected in source_hashes.items():
        assert digest(path)==expected,f'Source changed: {path}'
    write(out/'status.json',dict(status='completed',smoke=smoke))
    status(status='smoke_passed' if smoke else 'training_complete',out=str(out))
    print('SMOKE PASSED' if smoke else 'TRAINING COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    try:
        execute(args.out.resolve(),args.smoke)
    except BaseException:
        status(status='failed_infrastructure',traceback=traceback.format_exc())
        raise
