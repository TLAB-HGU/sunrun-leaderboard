import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.mse20_hss_shape import evaluate as e


def test_parent_import_is_torch_free():
    code='import sys; from experiments.mse20_hss_shape import evaluate; assert "torch" not in sys.modules'
    result=subprocess.run([str(e.p.ROOT/'.venv-fusion/bin/python'),'-c',code],cwd=e.p.ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def scoring_fixture(monkeypatch,tmp_path,mse=3200.,cell=3700.,tp=1):
    frame=pd.DataFrame({e.p.KEY[0]:['2026-06-01T00:00:00Z'],'horizon_hours':[1],'pred_kms':[400.]})
    ref=tmp_path/'ref.parquet';frame.to_parquet(ref)
    monkeypatch.setattr(e.old,'validate_frame',lambda x,*args:x)
    monkeypatch.setattr(e.old.scoring,'score',lambda *a:dict(mse=mse,n_pairs=203976,n_folds=2833,
        by_regime={key:dict(mse=cell,reliable=True) for key in e.old.BASELINE_CELLS}))
    calls=[]
    def unique(*args):
        matched=tp if not calls else 1;calls.append(1)
        return dict(tp=matched,fp=0,fn=23-matched,f1=2*matched/(23+matched))
    monkeypatch.setattr(e.d,'unique_event_metrics',unique)
    bench=dict(folds=None,truth=frame,naive=frame,paths={'reference':str(ref)},target=None,observed=None,origins=None,actual_events=None)
    return frame,bench


@pytest.mark.parametrize('mse,cell,tp,retain,status',[
    (3200,3700,1,True,'pass'),(3290,3700,2,True,'unmet'),
    (3200,3800,2,True,'unmet'),(3200,3700,0,True,'unmet')])
def test_paired_scientific_gates(monkeypatch,tmp_path,mse,cell,tp,retain,status):
    frame,bench=scoring_fixture(monkeypatch,tmp_path,mse,cell,tp)
    result=e.score_pair(frame,frame,bench,retain)
    assert result['status']==status
    assert set(result['gates'])=={'total_mse','unique_event_f1'} | (set(e.old.BASELINE_CELLS) if retain else set())
    assert 'precision' not in result['gates']


@pytest.mark.parametrize("policy",[None,False,1])
def test_explicit_cell_authorization_required(monkeypatch,tmp_path,policy):
    frame,bench=scoring_fixture(monkeypatch,tmp_path)
    with pytest.raises(ValueError,match='mandatory'):e.score_pair(frame,frame,bench,policy)


def metadata_fixture(tmp_path):
    pool=pd.date_range('2026-05-01',periods=2,freq='6h',tz='UTC')
    config={'epochs':1}; model=tmp_path/'model.pt';model.write_bytes(b'synthetic')
    names=['experiments/mse20_hss_shape/model.py','experiments/mse20_hss_arrival/diagnostics.py',
           'experiments/mse20_ch_dynamics/inputs.py','experiments/mse20_ch_dynamics/arrival.py',
           'experiments/mse20_ch_dynamics/neural.py']
    sources={name:e.p.file_ref(e.p.ROOT/name) for name in names}
    meta=dict(family='hss_shape_trapezoid',schema=dict(version=2,shape_definition={'decoder':'anchored'}),
        train_profile_sha256='1'*64,train_arrival_sha256='2'*64,train_shape_label_sha256='3'*64,config=config,model_sha256=e.p.digest(model),cutoff=e.p.CUTOFF,
        train_origins=[x.isoformat() for x in pool],train_origin_sha256=hashlib.sha256(pool.asi8.tobytes()).hexdigest(),
        train_label_max=(pool.max()+pd.Timedelta(hours=96)).isoformat(),train_profile_label_max=(pool.max()+pd.Timedelta(hours=72)).isoformat(),
        epochs=1,history=[{}],input_width=4,normalizer=dict(mean=[0.,0.],scale=[1.,1.],count=[2,2]),
        sources={str((e.p.ROOT/name).resolve()):sources[name]['sha256'] for name in names})
    def save():
        path=tmp_path/'metadata.json';path.write_text(json.dumps(meta));(tmp_path/'metadata.sha256').write_text(e.p.digest(path))
        return e.p.file_ref(path)
    return meta,save,config,sources,e.p.file_ref(model)


@pytest.mark.parametrize('tamper', ['none','future','source','scaler','model','shape_hash','shape_schema'])
def test_actual_metadata_cutoff_sources_and_scaler(tmp_path,tamper):
    meta,save,config,sources,model=metadata_fixture(tmp_path)
    if tamper=='shape_hash':meta['train_shape_label_sha256']=''
    if tamper=='shape_schema':meta['schema']['version']=1
    if tamper=='future':meta['train_label_max']='2026-06-01T00:00:00Z'
    if tamper=='source':meta['sources']={}
    if tamper=='scaler':meta['normalizer']['scale'][0]=0
    if tamper=='model':Path(model['path']).write_bytes(b'changed')
    if tamper=='none':assert e.verify_metadata(save(),model,config,e.p.CUTOFF,sources)==meta
    else:
        with pytest.raises(ValueError):e.verify_metadata(save(),model,config,e.p.CUTOFF,sources)


def test_cli_invalid_is_not_unmet(monkeypatch,capsys,tmp_path):
    def fail(*a):raise ValueError('missing proof')
    monkeypatch.setattr(e,'verify',fail)
    assert e.main(['--run-dir',str(tmp_path)])==2
    assert json.loads(capsys.readouterr().out)['status']=='invalid'


def test_empty_or_orphan_dev_ledger_fails_before_data_load(tmp_path):
    registration=dict(configs=list(range(4)),folds=list(range(14)))
    with pytest.raises(ValueError,match='cap'):e.verify_dev(tmp_path,registration,{}, {})


def test_snapshot_declared_delta_is_independently_checked():
    frame=pd.DataFrame({e.p.KEY[0]:[pd.Timestamp('2026-06-01',tz='UTC')],'horizon_hours':[1],'pred_kms':[400.]})
    changed=frame.copy();changed['pred_kms']+=.01
    with pytest.raises(ValueError,match='replay differs'):e.shared.delta(frame,changed)
    with pytest.raises(ValueError,match='recorded delta'):e.shared.declared_delta(0,.0005)


def replay_fixture(tmp_path,monkeypatch):
    origins=pd.date_range('2026-06-01',periods=3,freq='24h',tz='UTC')
    index=pd.date_range(origins.min(),origins.max()+pd.Timedelta(hours=72),freq='h')
    inventory={key:index for key in e.shared.RAW+e.shared.DERIVED}
    monkeypatch.setattr(e.shared,'verify_inventory',lambda *a:inventory)
    monkeypatch.setattr(e,'reconstruct_training',lambda *a:{'normalizer':{'mean':[0.]},'train_shape_label_sha256':'3'*64})
    def js(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value));return e.p.file_ref(path)
    def pred(name,subset=origins,offset=0):
        path=tmp_path/(name+'.parquet')
        pd.DataFrame({e.p.KEY[0]:np.repeat(subset,72),e.p.KEY[1]:np.tile(np.arange(1,73),len(subset)),
                      'pred_kms':np.full(len(subset)*72,400.+offset)}).to_parquet(path,index=False)
        return e.p.file_ref(path)
    meta=js('metadata.json',{});files={'model_metadata':meta}
    proof=dict(model_metadata=meta,frames_end_utc=index.max().isoformat(),source_files={},inventory={},checks=[],subset_checks=[],outputs={})
    proof['provenance']=js('provenance.json',dict(metadata=meta,source_files={},reconstructed={'normalizer':{'mean':[0.]},'train_shape_label_sha256':'3'*64}))
    source=e.p.file_ref(e.p.ROOT/'experiments/mse20_ch_dynamics/inputs.py')
    for mode in e.MODES:
        files[mode+'_predictions']=pred(mode)
        proof['outputs'][mode]=dict(candidate=files[mode+'_predictions'],gpu=pred(mode+'gpu'),cpu=pred(mode+'cpu'),max_abs_gpu=0.,max_abs_cpu=0.)
    for origin in origins:
        baseline={mode:pred(mode+str(origin.value),pd.DatetimeIndex([origin])) for mode in e.MODES}
        proof['subset_checks'].append(dict(origin=origin.isoformat(),baseline=baseline,max_abs_diff_kms=dict(mse=0.,peak=0.)))
        for kind,sourcekey in [('raw',s) for s in e.shared.RAW]+[('derived','joint')]:
            keys=(sourcekey,) if kind=='raw' else e.shared.DERIVED
            for action in ('poison','drop'):
                proof['checks'].append(dict(kind=kind,source=sourcekey,origin=origin.isoformat(),action=action,
                    mutation_counts={k:72 for k in keys},baseline=baseline,altered=baseline,max_abs_diff_kms=dict(mse=0.,peak=0.),
                    rebuild=dict(function='inputs.rebuild_frames',cache_bypass=True,source=source)))
    benchmark=dict(origins=origins,contract={},folds=None)
    return proof,files,benchmark,{'sources':{'experiments/mse20_ch_dynamics/inputs.py':source}}


@pytest.mark.parametrize('tamper',['none','count','coverage','cpu','normalizer','snapshot','shape_hash','missing_shape'])
def test_paired_replay_complete_and_tamper_failures(tmp_path,monkeypatch,tamper):
    proof,files,bench,registration=replay_fixture(tmp_path,monkeypatch)
    if tamper=='shape_hash':monkeypatch.setattr(e,'reconstruct_training',lambda *a:{'normalizer':{'mean':[0.]},'train_shape_label_sha256':'4'*64})
    if tamper=='missing_shape':monkeypatch.setattr(e,'reconstruct_training',lambda *a:{'normalizer':{'mean':[0.]}})
    if tamper=='count':proof['checks'][0]['mutation_counts']['ace']=0
    if tamper=='coverage':proof['checks'].pop()
    if tamper=='cpu':proof['outputs']['peak']['cpu']=proof['outputs']['peak']['gpu']
    if tamper=='normalizer':monkeypatch.setattr(e,'reconstruct_training',lambda *a:{'normalizer':{'mean':[1.]},'train_shape_label_sha256':'3'*64})
    if tamper=='snapshot':proof['checks'][0]['max_abs_diff_kms']['peak']=.0002
    def verify():return e.verify_replay(proof,files,bench,registration,{'normalizer':{'mean':[0.]},'train_shape_label_sha256':'3'*64},tmp_path)
    if tamper=='none':assert set(verify())=={'mse','peak'}
    else:
        with pytest.raises(ValueError):verify()


def dev_fixture(tmp_path,monkeypatch):
    configs=[{'id':str(i)} for i in range(4)]
    folds=[dict(id=str(i),start_utc='2025-01-01T00:00:00Z',end_utc='2025-02-01T00:00:00Z') for i in range(14)]
    registered=dict(configs=configs,folds=folds,registered_utc='2026-10-03T00:00:00Z',sources={})
    def save(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return e.p.file_ref(path)
    reg=save(tmp_path/'registered.json',registered)
    dummy=save(tmp_path/'dummy.json',{})
    support={'supported':True}
    monkeypatch.setattr(e.p,'load_frames',lambda *a:{})
    monkeypatch.setattr(e.dev,'source_support',lambda *a:support)
    monkeypatch.setattr(e,'verify_metadata',lambda *a:{})
    monkeypatch.setattr(e.shared,'predictions',lambda *a,**k:pd.DataFrame())
    metrics=dict(hss=dict(actual_count=1,predicted_count=1,tp=1,fp=0,fn=0,penalty_hours=0),
        modal_unique_events=dict(tp=1,fp=0,fn=0),squared_error_sum=1.,naive_squared_error_sum=2.,n_pairs=1)
    monkeypatch.setattr(e.dev,'score_pair',lambda *a:metrics)
    selection=dict(frozen_utc='2026-10-03T03:00:00Z',registration=reg,attempt_results={},
        selection_uses_official=False,previously_inspected_official_period=True,
        data_support=save(tmp_path/'support.json',dict(fold_support={f['id']:support for f in folds})))
    terminals=[];rows=[];config=configs[0];sha=e.dev.config_hash(config)
    for fold in folds:
        key=sha+'--'+fold['id'];path=tmp_path/'attempts'/(key+'.json')
        claim=save(path,dict(key=key,config=config,config_sha256=sha,fold=fold,
            claimed_utc='2026-10-03T01:00:00Z',registration=reg))
        pair=save(tmp_path/('pair'+fold['id']+'.json'),dict(fold=fold,config_sha256=sha,metadata=dummy,
            predictions=dict(mse=dummy,peak=dummy),saved_utc='2026-10-03T01:30:00Z'))
        result=dict(claim=claim,status='ok',finished_utc='2026-10-03T02:00:00Z',details=dict(
            support=support,metadata=dummy,model=dummy,pair=pair,predictions=dict(mse=dummy,peak=dummy),metrics=metrics))
        selection['attempt_results'][key]=save(tmp_path/'results'/(key+'.json'),result)
        terminals.append(result);rows.append(dict(result,config_sha256=sha,fold=fold))
    (tmp_path/'ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in terminals))
    selection.update(e.dev.rank_results(rows,configs,folds))
    return registered,selection,dict(search_stop_utc='2026-10-03T04:00:00Z')


@pytest.mark.parametrize('tamper',['none','orphan','ledger','ranking','official','unfinished'])
def test_complete_candidate_partial_budget_whole_dev_ledger(tmp_path,monkeypatch,tamper):
    reg,selection,runtime=dev_fixture(tmp_path,monkeypatch)
    if tamper=='orphan':(tmp_path/'results/orphan.json').write_text('{}')
    if tamper=='ledger':(tmp_path/'ledger.jsonl').write_text('')
    if tamper=='ranking':selection['selected_config']=reg['configs'][1]
    if tamper=='official':selection['selection_uses_official']=True
    if tamper=='unfinished':next((tmp_path/'results').glob('*.json')).unlink()
    if tamper=='none':assert e.verify_dev(tmp_path,reg,selection,runtime)==reg['configs'][0]
    else:
        with pytest.raises(ValueError):e.verify_dev(tmp_path,reg,selection,runtime)


def test_shape_grid_does_not_accept_old_six_config_registration(tmp_path):
    (tmp_path/'attempts').mkdir()
    (tmp_path/'attempts/one.json').write_text('{}')
    with pytest.raises(ValueError,match='grid/cap'):
        e.verify_dev(tmp_path,dict(configs=list(range(6)),folds=list(range(14))),{}, {})


def test_shape_claim_cap_is_56_even_when_all_claims_failed(tmp_path):
    (tmp_path/'attempts').mkdir()
    for number in range(57):(tmp_path/'attempts'/f'{number}.json').write_text('{}')
    with pytest.raises(ValueError,match='grid/cap'):
        e.verify_dev(tmp_path,dict(configs=list(range(4)),folds=list(range(14))),{}, {})


def test_shape_uses_frozen_event_definition_not_new_variant():
    from experiments.mse20_hss_arrival import diagnostics
    assert e.d is diagnostics


@pytest.mark.parametrize('route',['shape','original','missing','late','cycle','config','unknown'])
def test_global_preofficial_dispatch_and_fail_closed(monkeypatch,tmp_path,route):
    import sys
    import types
    from experiments import mse20_hss_shape as package
    from experiments.mse20_hss_arrival import evaluate as original
    helper=types.ModuleType('experiments.mse20_hss_shape.global_selection')
    monkeypatch.setitem(sys.modules,helper.__name__,helper)
    monkeypatch.setattr(package,'global_selection',helper,raising=False)
    monkeypatch.setattr(e.p,'deadline',lambda *a:100)
    monkeypatch.setattr(e.p,'verify_registration',lambda *a:{})
    monkeypatch.setattr(e.p,'setup',lambda *a:{'regime_cells_final_gate':True})
    monkeypatch.setattr(original.p,'setup',lambda *a:{'regime_cells_final_gate':True})
    config={'id':'selected'}
    monkeypatch.setattr(e.p,'read_json',lambda *a:{'claimed_utc':'2026-10-03T12:00:00Z','config':config})
    calls=[]
    monkeypatch.setattr(e,'_verify_shape',lambda *a:(calls.append('shape') or {'branch':'shape'}))
    monkeypatch.setattr(original,'verify',lambda *a:(calls.append('original') or {'branch':'original'}))
    def decision(run):
        assert Path(run)==e.p.RUN.resolve()
        if route=='missing':raise FileNotFoundError('missing global decision')
        return dict(selected_run=str(original.p.RUN if route=='original' else tmp_path if route=='unknown' else e.p.RUN),
                    cycle='original' if route in ('original','cycle') else 'shape',
                    selected_config={'id':'other'} if route=='config' else config,
                    frozen_utc='2026-10-03T13:00:00Z' if route=='late' else '2026-10-03T11:00:00Z')
    helper.verify_selection=decision
    if route in ('shape','original'):
        assert e.verify()=={'branch':route}
        assert calls==[route]
    else:
        with pytest.raises((ValueError,FileNotFoundError)):e.verify()
        assert calls==[]
