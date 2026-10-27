"""Frozen-recipe F1 cutoff fits; training only, publication eligibility separate."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import numpy as np
import pandas as pd
from experiments.cumulative_20261024.gate import locked
from experiments.e2000_parallel6 import launch as prior
from experiments.e2000_parallel6.model import LoggedMultiOutputRegressor, ModelConfig
from experiments.e2000_njobs_all.model import PooledRegressor, ModelConfig as PooledConfig
from experiments.e2000_njobs_all.runner import curves
from experiments.e2000_njobs_all.inputs import atomic_json, check_ref, digest, reference, sha, widen, target_matrix, prediction_frame

HERE=Path(__file__).resolve().parent
LB=HERE.parents[1]
PYTHON=LB/'.venv-fusion/bin/python'
MODULE='experiments.e2000_f1_release_v2.launch'
F1_CUTOFF=pd.Timestamp('2026-02-26T00:00:00Z')
F1_ORIGINS=pd.date_range('2026-02-28T23:00Z','2026-05-28T23:00Z',freq='h')


def code_hashes():
    from experiments.e2000 import prepare,source_proof,availability_rebuild
    paths=set(prior.code_hashes())|set(prior.baseline_launch.code_hashes())
    paths.update(str(p) for p in [HERE/'launch.py',HERE/'prepare.py',HERE/'__init__.py',
        LB.parent/'research/e2000/publication/f1-v2/supervise.py',
        Path(prepare.__file__),Path(source_proof.__file__),Path(availability_rebuild.__file__)])
    return {p:sha(p) for p in sorted(paths)}


def command(bundle,arm):
    threads='1' if arm=='direct' else '8'
    return shlex.join(['env',f'OMP_NUM_THREADS={threads}',f'OMP_THREAD_LIMIT={threads}','OMP_DYNAMIC=FALSE',
        'OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','NUMEXPR_NUM_THREADS=1',str(PYTHON),'-m',MODULE,
        'execute','--bundle',str(Path(bundle).resolve()),'--arm',arm])


def paths(spec,bundle,receipt):
    result=set(prior.registered_files(spec))|set(code_hashes())
    result.update(str(bundle/name) for name in ['inputs.json','intent.json'])
    result.update(r['path'] for r in receipt['selected_object_receipts'])
    result.update(receipt['code_hashes'])
    result.update(receipt[k]['path'] for k in ['features','metadata','source_mapping'])
    return sorted(result)


def register(spec_path,feature_receipt,bundle):
    spec=json.loads(Path(spec_path).read_text()); prior.preflight(spec)
    receipt=json.loads(Path(feature_receipt).read_text())
    if not receipt['passed'] or receipt['rows']!=153864 or receipt['origins']!=2137:
        raise ValueError('complete audited F1 features required')
    for r in receipt['selected_object_receipts']+[receipt[k] for k in ['features','metadata','source_mapping']]: check_ref(r)
    bundle=Path(bundle).resolve();bundle.mkdir(parents=True,exist_ok=False)
    atomic_json(bundle/'inputs.json',spec)
    intent={'input_sha256':digest(spec),'code_hashes':code_hashes(),'f1_feature_receipt':reference(feature_receipt),
            'cutoff':F1_CUTOFF.isoformat(),'seed':42,'direct_recipe':asdict(ModelConfig()),
            'pooled_recipe':asdict(PooledConfig()),'prior_f1_exposure':True,
            'publication_eligible':False,'tuning_cutoff_certified':False,
            'reason':'Current board adds F1. Refit unchanged user-fixed recipes before F1 cutoff; no HPO/early stopping. Prior internal validation overlaps F1; historical recipe tuning provenance unresolved.'}
    atomic_json(bundle/'intent.json',intent)
    core=prior.protected();ids={}
    for arm,ref in [('direct','E1987'),('pooled','E1988')]:
        event=core.register('D_E2000_MULTIOUTPUT_S42','codex-e2000-f1-extension',
            f'User-requested current-board F1 extension of {ref}: unchanged {arm} seed42 recipe, train targets strictly before 2026-02-26. '
            'No HPO or early stopping; native pre-cutoff overlapping diagnostic losses. Earlier F1 internal exposure disclosed; tuning-cutoff eligibility unresolved, no automatic upload.',
            command(bundle,arm),kind='reference',code=paths(spec,bundle,receipt)+[str(Path(feature_receipt).resolve())],reference=ref)
        ids[arm]=event['id'];atomic_json(bundle/'registrations.json',ids)
    return ids


def verify(bundle,arm):
    bundle=Path(bundle).resolve();spec=json.loads((bundle/'inputs.json').read_text())
    intent=json.loads((bundle/'intent.json').read_text());ids=json.loads((bundle/'registrations.json').read_text())
    core=prior.protected();entry=core.state()['experiments'][ids[arm]]
    if entry['cmd']!=command(bundle,arm) or entry.get('flags'):raise ValueError('registered command/flags mismatch')
    if intent['code_hashes']!=code_hashes() or intent['input_sha256']!=digest(spec) or intent['cutoff']!=F1_CUTOFF.isoformat():raise ValueError('frozen execution drift')
    receipt=json.loads(check_ref(intent['f1_feature_receipt']).read_text())
    for path in paths(spec,bundle,receipt)+[intent['f1_feature_receipt']['path']]:
        if entry['code'].get(path)!=sha(path):raise ValueError('registered file changed: '+path)
    prior.preflight(spec)
    return core,spec,intent,receipt,ids[arm]


def cutoff_masks(origins,observed):
    targets=origins.tz_convert('UTC').tz_localize(None).to_numpy(dtype='datetime64[ns]')[:,None]+np.arange(1,73,dtype='timedelta64[h]')[None,:]
    mask=observed & (targets<F1_CUTOFF.to_datetime64())
    diagnostic=mask & (targets >= (F1_CUTOFF-pd.Timedelta(days=28)).to_datetime64())
    if not mask.any(axis=0).all() or not diagnostic.any(axis=0).all():raise ValueError('every horizon requires pre-cutoff training/diagnostic labels')
    return mask,diagnostic


def fit_context(spec, receipt, features, origins):
    return {'code_hashes':code_hashes(),'data_hashes':{'input_envelope':digest(spec),'f1_features':receipt['features']['sha256']},
             'feature_names':list(features),'cutoff':F1_CUTOFF.isoformat(),'grid':{'origins':digest(origins.astype(str).tolist()),'horizons':72,'evaluation_period':'F1'},
             'source_availability':spec['source_receipts']+[reference(receipt['features']['path'])],'stage':'official'}


def execute(bundle,arm):
    core,spec,intent,receipt,eid=verify(bundle,arm);out=core.RUNS/eid
    if Path(os.environ.get('EXPOS_OUT','')).resolve()!=out.resolve():raise ValueError('ExpOS execution required')
    identity={'input_sha256':digest(spec),'feature_receipt':intent['f1_feature_receipt'],'code_hashes':code_hashes(),'arm':arm,'cutoff':F1_CUTOFF.isoformat()}
    with locked(out/'execution'):
        state=out/'execution.json'
        if state.exists():
            old=json.loads(state.read_text())
            if old['identity']!=identity:raise ValueError('checkpoint execution identity drift')
            if old['status']=='completed':
                for r in old['outputs']:check_ref(r)
                return old
        atomic_json(state,{'status':'running','identity':identity})
        try:
            features=json.loads(check_ref(spec['artifacts']['feature_names']).read_text())['features']
            origins,x=widen(pd.read_parquet(check_ref(spec['artifacts']['train_features'])),features)
            y,observed=target_matrix(pd.read_parquet(check_ref(spec['artifacts']['labels'])),origins)
            mask,diagnostic=cutoff_masks(origins,observed);keep=mask.any(axis=1)
            origins=origins[keep];x=x.loc[origins];y=y[keep];mask=mask[keep];diagnostic=diagnostic[keep]
            y=np.where(mask,y,np.nan)
            f1_origins,f1x=widen(pd.read_parquet(check_ref(receipt['features'])),features,expected_origins=F1_ORIGINS)
            if list(x)!=list(f1x):raise ValueError('F1 feature order mismatch')
            context=fit_context(spec,receipt,x.columns,f1_origins)
            make=lambda: LoggedMultiOutputRegressor(ModelConfig(),workers=6) if arm=='direct' else PooledRegressor(PooledConfig())
            va=diagnostic.any(axis=1)
            kwargs=dict(observed_mask=mask,validation=(x.to_numpy()[va],y[va],diagnostic[va]),output_dir=out/'f1',context=context,validation_role='overlapping_diagnostic')
            print(f'{arm} F1 fit: origins={len(x)}, observed_targets={mask.sum()}, cutoff={F1_CUTOFF}',flush=True)
            model=make().fit(x.to_numpy(),y,**kwargs)
            values=model.predict(f1x.to_numpy())
            restored=make().fit(x.to_numpy(),y,**kwargs)
            if not np.array_equal(values,restored.predict(f1x.to_numpy())):raise ValueError('checkpoint replay differs')
            frame=prediction_frame(f1_origins,values)
            if len(frame)!=153864 or frame[['origin_last_input_utc','horizon_hours']].duplicated().any() or not np.isfinite(frame.pred_kms).all():raise ValueError('invalid F1 grid')
            frame.to_parquet(out/'predictions_f1_s42.parquet',index=False);curves(out/'f1')
            atomic_json(out/'confirmation.json',{'status':'completed','arm':arm,'seed':42,'training_target_cutoff':F1_CUTOFF.isoformat(),
                'observed_training_targets':int(mask.sum()),'diagnostic_targets':int(diagnostic.sum()),'checkpoint_replay_exact':True,
                'prediction':reference(out/'predictions_f1_s42.parquet'),'publication_eligible':False,'tuning_cutoff_certified':False,
                'prior_f1_exposure':True,'native_loss_role':'overlapping precutoff diagnostic; not model selection'})
            outputs=[reference(p) for p in out.rglob('*') if p.is_file() and p!=state and p.suffix in ['.json','.csv','.png','.svg','.checkpoint','.parquet']]
            result={'status':'completed','identity':identity,'outputs':outputs};atomic_json(state,result);return result
        except BaseException as e:
            atomic_json(state,{'status':'failed','identity':identity,'error':str(e)});raise


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','execute']);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--inputs',type=Path);p.add_argument('--feature-receipt',type=Path);p.add_argument('--arm',choices=['direct','pooled'])
    a=p.parse_args();result=register(a.inputs,a.feature_receipt,a.bundle) if a.command=='register' else execute(a.bundle,a.arm)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
