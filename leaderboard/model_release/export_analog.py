"""Re-fit the three registered analog recipes and export fitted, portable state."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '6')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ['CUDA_VISIBLE_DEVICES'] = '1' if 'E0108' in sys.argv else ''
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_release.analog_runtime import freeze_pool, predict_residual, predict_branch

NAMES = {'E0421': 'xgb_d76_lead_moe_resid_dv', 'E0472': 'xgb_d76_lead_moe_resid_dv_enlil',
         'E0108': 'xgb_d46_xover_event_branch'}


def registered(experiment):
    path = next((ROOT/'scripts/seongeun'/experiment).glob('*.py'))
    spec = importlib.util.spec_from_file_location('registered_recipe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate()
    return module


def residual(experiment_id):
    enlil = experiment_id == 'E0472'
    sys.path.insert(0, str(ROOT/'experiments/d147_e0421_lean'))
    import lean as L
    if enlil:
        sys.path.insert(0, str(ROOT/'experiments/d148_enlil_live'))
        import lean_enlil as LE
        L.ia.enlil_runs, L.A.base_grid_for, L.feature_columns = LE.enlil_runs, LE.base_grid_for, LE.feature_columns
    cfg = {**L.DEFAULT, **({'dist': 'std_equal'} if enlil else {})}
    df = L.build_train({})
    cols = L.feature_columns(df, cfg)
    meta, block = L.ST.load_inputs()
    ace = L.A.filled_hourly()
    sp = L.SP
    origins = pd.date_range(sp.ORIGIN_START, sp.ORIGIN_END, freq='h', tz='UTC')
    oo = pd.date_range(sp.OOF_START_ISO, sp.OOF_END_ISO, freq=sp.OOF_FREQ)
    pool_queries = list(origins)+list(oo)
    if enlil:
        pool_queries += [pd.Timestamp(L.WINDOWS[0][2])]
    ct = L.A.cand_grid(pool_queries, sp.POOL_MAX_LOOKBACK_H)
    print('Fitting OOF models', flush=True)
    fs = L.fit_set(cfg, sp.BASE_DEPTH, sp.BASE_PARAMS, 0, df[df.target_t < pd.Timestamp(sp.OOF_TRAIN_END_ISO)], cols)
    print('Scoring OOF and candidate paths', flush=True)
    paths = lambda times: L.paths(cfg, fs, times, ace, meta, block, cols)
    cp = paths(ct)
    qb, qh = paths(oo), paths(oo-pd.Timedelta('12h'))
    scales = L.pool_scales(ace, ct, cp) if enlil else (1., 1., 1.)
    if enlil:
        dv = L.residual_multi(oo, ace, qb, qh, ct, cp, (15,), 'std_equal', scales)[15]
    else:
        dv = L.RL.residual_for_origins(oo, ace, qb, qh, ct, cp, 15)[0]
    key = pd.DataFrame({'origin': np.repeat(oo.to_numpy(), 72), 'h': np.tile(np.arange(1,73),len(oo))})
    key['origin'] = pd.to_datetime(key.origin, utc=True)
    y = key.merge(df[['origin','h','y']], on=['origin','h'], how='left').y.to_numpy(float)
    w = L.RL.fit_blend_weight_dv(qb.ravel(), dv, y, sp.BLEND_GRID)[0]
    print(f'Fitting final models; calibrated weight={w}', flush=True)
    full = L.fit_set(cfg, sp.BASE_DEPTH, sp.BASE_PARAMS, 0, df[df.target_t < L.CUTOFF], cols)
    state = {'cutoff': sp.CUTOFF_ISO, 'weight': float(w), 'weights': sp.WEIGHTS, 'scales': scales,
             'k':15, 'lookback_h': sp.POOL_MAX_LOOKBACK_H, 'include_enlil': enlil,
             'pool': freeze_pool(ct, cp, ace, L.CUTOFF)}
    bundle = {'kind':'residual_analog', 'models':{'oof':fs, 'full':full}, 'state':state, 'columns':cols}
    builder = lambda times: L.A.base_grid_for(times, ace, L.HR, meta, block, cols)
    # Independent original implementation comparison for all submitted origins.
    print('Checking original full-grid predictions', flush=True)
    q, hind = paths(origins), paths(origins-pd.Timedelta('12h'))
    if enlil:
        offdv = L.residual_multi(origins, ace, q, hind, ct, cp, (15,), 'std_equal', scales, cutoff=L.CUTOFF)[15]
    else:
        offdv = L.RL.residual_for_origins(origins, ace, q, hind, ct, cp, 15, cutoff=L.CUTOFF)[0]
    reference = L.RL.blend_dv(L.predict(cfg,full,builder(origins),cols),offdv.ravel(),w)
    predict = lambda b: predict_residual(b, origins, builder, ace)
    return bundle, origins, reference, predict


def branch():
    sys.path.insert(0, str(ROOT/'experiments/expos_runners'))
    import d46_official as D
    d = D.d46
    df = d.build_train(D.TRAIN_FREQ, d.ST.strip_columns())
    cols = d.ia.arms(df.columns)['existing_all']+d.ST.strip_columns()
    jointcols = cols+D.XO2.analog_columns()
    state, filled = D.XO2.build_state()
    scaler = D.XO2.scaler_from_predev(state,pd.Timestamp(d.PERIODS['dev'][0],tz='UTC'))
    print('Fitting D46 with exclusive GPU1 lease',flush=True)
    models = D.fit(df,cols,jointcols,state,filled,scaler)
    meta, block = d.ST.load_inputs()
    te = d.add_strips(d.base_grid(D.ORIGINS),meta,block)[0]
    keys, metric = D.XO2.split_config(D.XO2.PINNED_KEY)
    analog = D.XO2.analog_long(D.ORIGINS,state,filled,keys,metric,scaler)
    ref = D.predict(models,te,te.merge(analog,on=['origin','h'],how='left'),cols,jointcols)[0]
    from model_release.export_regular import cpu_models
    cpu_models(models, preserve_cuda_arithmetic=True)
    bundle = {'kind':'event_branch', 'models':models, 'columns':{'level':cols,'joint':jointcols},
              'state':{'scaler':scaler,'analog_keys':keys,'metric':metric,'cutoff':D.CUTOFF.isoformat(),'include_enlil':False,'training_device':'cuda','inference_device':'cpu','cuda_base_score_last':True}}
    return bundle,D.ORIGINS,ref,lambda b: predict_branch(b,te,analog)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('experiment_id',choices=NAMES)
    ap.add_argument('--output-dir',type=Path,default=ROOT.parent/'model-release-20261010')
    ap.add_argument('--reference',type=Path,required=True)
    args=ap.parse_args()
    if args.experiment_id == 'E0108':
        lockpath=ROOT/'store/ch-breakthrough-v2/gpu-leases/gpu1.lock'
        lockpath.parent.mkdir(parents=True,exist_ok=True)
        print('Waiting for GPU1 lease',flush=True)
        with lockpath.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            export(args)
    else:
        export(args)


def export(args):
    name=NAMES[args.experiment_id]
    script=registered(name)
    bundle, origins, expected, predict = branch() if args.experiment_id=='E0108' else residual(args.experiment_id)
    bundle.update(format_version=1,experiment=name,config=script.CONFIG,config_sha256=script.CONFIG_SHA256)
    print('Checking portable runtime predictions', flush=True)
    memory = predict(bundle)
    np.testing.assert_allclose(memory,expected,rtol=0,atol=1e-6)
    raw=pickle.dumps(bundle,protocol=pickle.HIGHEST_PROTOCOL)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    checkpoint=args.output_dir/f'{name}.unverified.pkl'
    checkpoint.write_bytes(raw)
    print('Checking serialized model predictions', flush=True)
    restored=predict(pickle.loads(raw))
    np.testing.assert_allclose(memory,restored,rtol=0,atol=1e-6)
    result=pd.DataFrame({'origin_last_input_utc':np.repeat(origins.to_numpy(),72),'horizon_hours':np.tile(np.arange(1,73),len(origins)),'pred_kms':memory})
    ref=pd.read_parquet(args.reference)
    keys=['origin_last_input_utc','horizon_hours']
    result[keys[0]]=pd.to_datetime(result[keys[0]],utc=True)
    ref[keys[0]]=pd.to_datetime(ref[keys[0]],utc=True)
    comparison=result.merge(ref[keys+['pred_kms']],on=keys,suffixes=('_new','_old'),validate='one_to_one')
    if len(comparison)!=2833*72: raise ValueError('Submission comparison is not the complete 2833x72 grid')
    delta=float(np.max(np.abs(comparison.pred_kms_new-comparison.pred_kms_old)))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    result.to_parquet(args.output_dir/f'{name}.predictions.parquet',index=False)
    report={'experiment':name,'config_sha256':script.CONFIG_SHA256,'recipe_runtime_max_abs':float(np.max(np.abs(memory-expected))),
            'pickle_max_abs':float(np.max(np.abs(memory-restored))),'submission_max_abs':delta,'rows':len(result),'passed':delta<=1e-3}
    (args.output_dir/f'{name}.verification.json').write_text(json.dumps(report,indent=2))
    if delta>1e-3: raise ValueError(f'Submission parity failed: {delta}; refusing final PKL publication')
    path=args.output_dir/f'{name}_seongeun.pkl'
    path.write_bytes(raw)
    checkpoint.unlink(missing_ok=True)
    print(json.dumps({**report,'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}),flush=True)

if __name__=='__main__':main()
