"""Strict successor proof rejects forged CPU/future/cutoff/config evidence."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from experiments.mse20_exact_ridge import evaluate as ev
from experiments.mse20_exact_ridge import protocol as p


@pytest.fixture
def bundle(tmp_path,monkeypatch):
    def put(name,value):p.write_json(tmp_path/name,value);return p.file_ref(tmp_path/name)
    o=pd.date_range('2026-05-31T23:00Z',periods=2833,freq='h')
    folds=pd.DataFrame({'origin_last_input_utc':o.strftime('%Y-%m-%dT%H:%M:%SZ')});folds.to_parquet(tmp_path/'folds.parquet')
    frame=pd.DataFrame({'origin_last_input_utc':np.repeat(folds.iloc[:,0],72),'horizon_hours':np.tile(np.arange(1,73),2833),'pred_kms':400.})
    for name in ('predictions','gpu','cpu'):frame.to_parquet(tmp_path/f'{name}.parquet',index=False)
    config={'id':'test','target':'mr','alpha':.01,'history_days':None,'max_train_origins':6000,'seed':42}
    reg={'registered_utc':'2026-10-03T10:39:00Z','configs':[config]};registration=put('registered.json',reg)
    attempt=put('attempt.json',{'config':config,'registration_sha256':registration['sha256'],'started_utc':'2026-10-03T10:40:00Z'})
    from experiments.mse20_adaptive.neural import SCHEMA
    origins=pd.date_range('2026-05-01',periods=10,freq='6h',tz='UTC')
    np.savez(tmp_path/'model.npz',coef=np.zeros((1016,72)),intercept=np.zeros(72))
    model=p.file_ref(tmp_path/'model.npz')
    meta={'family':'exact_numeric_ridge','config':config,'cutoff':p.CUTOFF,'train_label_max':(origins.max()+pd.Timedelta(hours=72)).isoformat(),
          'training_origins':[o.isoformat() for o in origins],'n_train_origins':10,'train_origin_sha256':hashlib.sha256(origins.asi8.tobytes()).hexdigest(),
          'schema':SCHEMA,'flat_width':1016,'target_scale':100,'aux_size':680,
          'scaler':{'history_mean':0.,'history_scale':1.,'aux_mean':[0.]*340,'aux_scale':[1.]*340},
          'source_sha256':p.digest(Path(ev.__file__).with_name('ridge.py')),'model_sha256':model['sha256'],
          'solver':{'normal_equation_relative_residual':0.,'max_abs_training_residual_mean':0.},'fitted_utc':'2026-10-03T10:41:00Z'}
    metadata=put('metadata.json',meta);(tmp_path/'metadata.sha256').write_text(metadata['sha256'])
    selection=put('selection.json',{})
    recipe={'config':config,'frozen_utc':'2026-10-03T10:42:00Z','official_informed_selection':True,'confirmatory':False,
            'files':{'registration':registration,'model':model,'metadata':metadata,'attempt':attempt,'selection':selection}}
    recipe_ref=put('recipe.json',recipe)
    checks=[]
    for origin in o[[0,len(o)//2,-1]]:
        subset=p.canonical_predictions(frame.loc[pd.to_datetime(frame.iloc[:,0],utc=True)==origin],pd.DatetimeIndex([origin]))
        for mode in ('poison','drop'):
            name=f'check{len(checks)}.parquet';subset.to_parquet(tmp_path/name,index=False);ref=p.file_ref(tmp_path/name)
            checks.append({'origin':origin.isoformat(),'mode':mode,'horizons':72,'future_rows':{k:72 for k in ('speed','observed','base','physics','quality')},
                           'max_abs_diff_kms':0.,'full_subset_max_abs_diff_kms':0.,'files':{'baseline':ref,'altered':ref}})
    causal=put('causality.json',{'checks':checks})
    proof={'recipe_sha256':recipe_ref['sha256'],'candidate_sha256':p.digest(tmp_path/'predictions.parquet'),
           'started_utc':'2026-10-03T10:43:00Z','finished_utc':'2026-10-03T10:44:00Z','gpu':p.file_ref(tmp_path/'gpu.parquet'),'cpu':p.file_ref(tmp_path/'cpu.parquet'),
           'support':{'origins':2833,'pairs':203976},'gpu_max_abs_diff_kms':0.,'cpu_max_abs_diff_kms':0.}
    tests=put('tests.json',{'passed':True,'exit_code':0,'files':{'source':p.file_ref(ev.__file__)}})
    files={'recipe':recipe_ref,'model':model,'metadata':metadata,'attempt':attempt,'selection':selection,'registration':registration,
           'causality':causal,'replay':put('replay.json',proof),'tests':tests}
    files.update({name:p.file_ref(tmp_path/f'{name}.parquet') for name in ('predictions','gpu','cpu')})
    manifest={'files':files,'created_utc':'2026-10-03T10:45:00Z','official_informed_selection':True,'confirmatory':False};put('final-manifest.json',manifest)
    (tmp_path/'runtime.json').write_text(json.dumps({'search_stop_utc':'2026-10-03T10:56:29Z','optimization_stop_utc':'2026-10-03T13:11:29Z','deadline_utc':'2026-10-03T14:26:29Z'}))
    monkeypatch.setattr(p,'OLD',tmp_path);monkeypatch.setattr(p,'setup',lambda _:({}, {}, {'folds':tmp_path/'folds.parquet'}))
    monkeypatch.setattr(p,'verify_registration',lambda _:reg)
    return tmp_path,manifest,meta,proof,checks,recipe


def test_valid_full_structured_bundle(bundle):ev.verify(bundle[0])


@pytest.mark.parametrize('attack',['cpu','causal_values','vacuous','late_fit','future_labels','source','scaler','weights','bad_config','support'])
def test_forged_artifacts_fail_closed(bundle,attack):
    path,manifest,meta,proof,checks,recipe=bundle
    if attack=='cpu':
        data=pd.read_parquet(path/'cpu.parquet');data.pred_kms+=1;data.to_parquet(path/'cpu.parquet',index=False)
        manifest['files']['cpu']=p.file_ref(path/'cpu.parquet');proof['cpu']=manifest['files']['cpu']
    elif attack=='causal_values':checks[0]['max_abs_diff_kms']=.01
    elif attack=='vacuous':checks[0]['future_rows']['speed']=0
    elif attack=='late_fit':meta['fitted_utc']='2026-10-03T10:57:00Z'
    elif attack=='future_labels':meta['train_label_max']='2026-06-01T00:00Z'
    elif attack=='source':meta['source_sha256']='x'*64
    elif attack=='scaler':meta['scaler']['aux_scale'][0]=0
    elif attack=='weights':
        np.savez(path/'model.npz',coef=np.zeros((1016,71)),intercept=np.zeros(72));manifest['files']['model']=p.file_ref(path/'model.npz');meta['model_sha256']=manifest['files']['model']['sha256'];recipe['files']['model']=manifest['files']['model']
    elif attack=='bad_config':meta['config']['alpha']=2.
    elif attack=='support':proof['support']['pairs']=72
    p.write_json(path/'metadata.json',meta);manifest['files']['metadata']=p.file_ref(path/'metadata.json');recipe['files']['metadata']=manifest['files']['metadata'];(path/'metadata.sha256').write_text(manifest['files']['metadata']['sha256'])
    p.write_json(path/'causality.json',{'checks':checks});manifest['files']['causality']=p.file_ref(path/'causality.json')
    p.write_json(path/'recipe.json',recipe);manifest['files']['recipe']=p.file_ref(path/'recipe.json');proof['recipe_sha256']=manifest['files']['recipe']['sha256']
    p.write_json(path/'replay.json',proof);manifest['files']['replay']=p.file_ref(path/'replay.json');p.write_json(path/'final-manifest.json',manifest)
    with pytest.raises((ValueError,KeyError,AssertionError)):ev.verify(path)


def test_configuration_boundaries():
    c={'id':'x','target':'mr','alpha':.01,'history_days':None,'max_train_origins':6000,'seed':42};p.validate_config(c)
    c['alpha']=2
    with pytest.raises(ValueError):p.validate_config(c)
