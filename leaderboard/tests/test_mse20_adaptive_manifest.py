"""Full manifest verifier adversarial fixtures; synthetic official-size support."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.mse20_adaptive import evaluate as ev


def write(path,data):
    path.write_text(json.dumps(data))
    return ev.p.file_ref(path)


@pytest.fixture
def manifest_fixture(tmp_path,monkeypatch):
    def build(family='tree'):
        d=tmp_path/family;d.mkdir()
        origin=pd.date_range('2026-06-01',periods=2833,freq='h',tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
        folds=pd.DataFrame({'origin_last_input_utc':origin});folds.to_parquet(d/'folds.parquet')
        frame=pd.DataFrame({'origin_last_input_utc':np.repeat(origin,72),'horizon_hours':np.tile(np.arange(1,73),2833),'pred_kms':400.})
        frame.to_parquet(d/'pred.parquet');pred=ev.p.file_ref(d/'pred.parquet')
        write(d/'authorization.json',{'test':True})
        contract={'started_utc':'2026-10-03T06:26:29Z','deadline_utc':'2026-10-03T14:26:29Z'}
        monkeypatch.setattr(ev.p,'setup',lambda _: ({},contract,{'folds':d/'folds.parquet'}))
        source_files={str(i):ev.p.file_ref(path) for i,path in enumerate(sorted(ev._source_paths(family)))}
        config={'id':'candidate','bands':1}
        name='model.pt' if family=='neural' else 'model.ubj'
        (d/name).write_bytes(b'synthetic fitted artifact')
        model=ev.p.file_ref(d/name)
        meta={'family':{'tree':'adaptive_tree','neural':'adaptive_neural','reference':'combined_recurrence_xgboost'}[family],
              'config':config,'cutoff':ev.p.CUTOFF,'train_label_max':ev.p.CUTOFF,'n_train_origins':10,
              'model_files':[{'path':name,'sha256':model['sha256']}],'model_sha256':model['sha256']}
        if family=='tree':
            meta['source_sha256']={name:ev.p.digest(ev.p.ROOT/'experiments/mse20_adaptive'/name) for name in ('tree.py','features.py')}
        elif family=='neural':
            meta['source_sha256']=ev.p.digest(ev.p.ROOT/'experiments/mse20_adaptive/neural.py')
            ch=[f'ch_{tag}_{name}_{suffix}' for tag in ('t035','t045','t055') for name in ('area','top1_lon','top1_lat','top1_lon_width','equatorial_area') for suffix in ('lag0','lag24','change24','change72','mean24')]
            hist=[f'ch_{tag}_{name}_lag0' for tag in ('t035','t045','t055') for name in ('area','top1_lon','top1_lat','equatorial_area')]
            meta['schema']={'history_hours':168,'recurrence_days':[26,27,28],'ch_columns':ch,'ch_history_columns':hist,'history_lags':[48,72,96,119],
                            'normalization':'subtract issuance current speed from speed inputs only','inference_batch_size':256}
            width=216+len(ch)+len(hist)*4+1;meta['aux_size']=width*2
            meta['scaler']={'history_mean':0.,'history_scale':1.,'aux_mean':[0.]*width,'aux_scale':[1.]*width}
        metadata=write(d/'metadata.json',meta)
        selection=write(d/'selection.json',{'selected':'candidate'})
        component={'family':family,'id':'candidate','key':'candidate','weight':1.,'config':config,
                   'metadata':metadata,'model_files':[model],'source_files':source_files,'prediction':pred}
        recipe={'official_informed_selection':True,'confirmatory':False,'frozen_utc':'2026-10-03T10:00:00Z',
                'files':dict(source_files,selection=selection),'components':[component]}
        recipe_ref=write(d/'recipe.json',recipe)
        replay={'schema_version':1,'passed':True,'recipe_sha256':recipe_ref['sha256'],'candidate_sha256':pred['sha256'],
                'started_utc':'2026-10-03T10:01:00Z','finished_utc':'2026-10-03T10:02:00Z','support':{'origins':2833,'pairs':203976},
                'max_abs_diff_kms':0.,'files':{'replayed_predictions':pred,'component':pred},
                'components':[{'key':'candidate','family':family,'id':'candidate','metadata_sha256':metadata['sha256'],
                               'model_sha256':[model['sha256']],'prediction_sha256':pred['sha256'],'replayed_prediction':pred,'max_abs_diff_kms':0.}]}
        replay_ref=write(d/'replay.json',replay)
        summary=write(d/'tests.json',{'passed':True,'files':{'log':selection}})
        manifest={'schema_version':1,'authorization_sha256':ev.p.digest(d/'authorization.json'),'official_informed_selection':True,'confirmatory':False,
                  'created_utc':'2026-10-03T10:03:00Z','training_cutoff_utc':ev.p.CUTOFF,'training_label_max_utc':ev.p.CUTOFF,
                  'model_files':['model'],'transform_files':['metadata'] if family=='neural' else [],
                  'files':{'predictions':pred,'selection':selection,'test_summary':summary,'causality_summary':summary,
                           'recipe':recipe_ref,'replay_proof':replay_ref,'model':model,'metadata':metadata}}
        def save():
            component['metadata']=write(d/'metadata.json',meta)
            manifest['files']['metadata']=component['metadata']
            manifest['files']['recipe']=write(d/'recipe.json',recipe)
            replay['recipe_sha256']=manifest['files']['recipe']['sha256']
            replay['components'][0]['metadata_sha256']=component['metadata']['sha256']
            manifest['files']['replay_proof']=write(d/'replay.json',replay)
            write(d/'final-manifest.json',manifest)
        save()
        return d,manifest,recipe,replay,meta,save
    return build


@pytest.mark.parametrize('family',['tree','neural','reference'])
def test_valid_full_manifest(manifest_fixture,family):
    d,*_=manifest_fixture(family)
    manifest,recipe,files,paths,contract=ev.verify_manifest(d)
    assert recipe['components'][0]['family']==family


@pytest.mark.parametrize('mutation',['weight_only','extra_model','model_binding','config','cutoff','label_max','source_hash','source_closure',
                                    'prediction_hash','replay_candidate','replay_support','replay_models','weights','duplicate_key',
                                    'empty_summary','freeze_time','replay_time','missing_replay','blend','replay_actual','late_optimization_freeze'])
def test_manifest_tampering_rejected(manifest_fixture,mutation):
    d,m,r,proof,meta,save=manifest_fixture()
    c=r['components'][0]
    if mutation=='late_optimization_freeze':
        r['frozen_utc']='2026-10-03T13:12:00Z'
        proof['started_utc']='2026-10-03T13:13:00Z'
        proof['finished_utc']='2026-10-03T13:14:00Z'
        m['created_utc']='2026-10-03T13:15:00Z'
    if mutation=='weight_only':r['components']=[{'weight':1.}]
    elif mutation=='extra_model':
        (d/'other.ubj').write_bytes(b'unrelated');m['files']['extra']=ev.p.file_ref(d/'other.ubj');m['model_files'].append('extra')
    elif mutation=='model_binding':meta['model_files'][0]['sha256']='0'*64
    elif mutation=='config':meta['config']={'id':'different','bands':1}
    elif mutation=='cutoff':meta['cutoff']='2026-06-01T00:00:00Z'
    elif mutation=='label_max':meta['train_label_max']='2026-06-01T00:00:00Z'
    elif mutation=='source_hash':meta['source_sha256']['tree.py']='0'*64
    elif mutation=='source_closure':c['source_files']={}
    elif mutation=='prediction_hash':c['prediction']={'path':str(d/'pred.parquet'),'sha256':'0'*64}
    elif mutation=='replay_candidate':proof['candidate_sha256']='0'*64
    elif mutation=='replay_support':proof['support']['pairs']=72
    elif mutation=='replay_models':proof['components'][0]['model_sha256']=['0'*64]
    elif mutation=='weights':c['weight']=.9
    elif mutation=='duplicate_key':r['components'].append(dict(c));c['weight']=.5;r['components'][1]['weight']=.5
    elif mutation=='empty_summary':
        m['files']['test_summary']=write(d/'empty.json',{'passed':True,'files':{}})
    elif mutation=='freeze_time':r['frozen_utc']='2026-10-03T15:00:00Z'
    elif mutation=='replay_time':proof['started_utc']='2026-10-03T09:00:00Z'
    elif mutation=='missing_replay':proof['files']={}
    elif mutation in ('blend','replay_actual'):
        frame=pd.read_parquet(d/'pred.parquet');frame['pred_kms']=401.;frame.to_parquet(d/'altered.parquet')
        altered=ev.p.file_ref(d/'altered.parquet')
        if mutation=='blend':
            c['prediction']=altered;proof['components'][0]['prediction_sha256']=altered['sha256'];proof['components'][0]['replayed_prediction']=altered;proof['files']['component']=altered
        else:proof['files']['replayed_predictions']=altered
    save()
    with pytest.raises((ValueError,KeyError)):ev.verify_manifest(d)


@pytest.mark.parametrize('mutation',['missing_scaler','bad_scale','wrong_width','schema','missing_transform'])
def test_neural_transform_tamper(manifest_fixture,mutation):
    d,m,r,proof,meta,save=manifest_fixture('neural')
    if mutation=='missing_scaler':meta.pop('scaler')
    elif mutation=='bad_scale':meta['scaler']['aux_scale'][0]=0
    elif mutation=='wrong_width':meta['aux_size']=1
    elif mutation=='schema':meta['schema']['history_hours']=1
    else:m['transform_files']=[]
    save()
    with pytest.raises((ValueError,KeyError)):ev.verify_manifest(d)
