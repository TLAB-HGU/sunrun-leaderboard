"""New-cycle execution tests; old registered P1 code/artifacts remain read-only."""
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import types
import numpy as np
import pandas as pd
import pytest
from experiments.mse20_hss_state import dev,finalize as f,run as controller


def configs():
    return [dict(id=f'prior{strength}',bin_hours=6,slow_hours=168,epochs=20,max_train_origins=4096,prior_strength=strength)
            for strength in (0.,1.)]


def test_two_structural_configs_fourteen_same_folds_and28cap(tmp_path,monkeypatch):
    payload=dev.registration_payload()
    assert payload['max_attempts']==28 and len(payload['folds'])==14
    from experiments.mse20_hss_arrival import dev as old
    assert payload['folds']==old.registration_payload()['folds']
    registration=dict(configs=configs(),folds=payload['folds'],ranking=payload['ranking'])
    monkeypatch.setattr(dev.p,'verify_registration',lambda _:registration)
    assert dev.verify_registration(tmp_path)==registration
    registration['configs'][0]['bin_hours']=3
    with pytest.raises(ValueError,match='structure'):dev.verify_registration(tmp_path)


def test_claims_include_failed_attempts_and_never_retry(tmp_path):
    dev.p.write_new(tmp_path/'registered.json',{'synthetic':True})
    fold=dev.fold_grid()[0];config=configs()[0]
    path=dev.claim(tmp_path,config,fold)
    dev.terminal(tmp_path,path,'failed',{'synthetic':True})
    assert dev.claim(tmp_path,config,fold) is None
    for i in range(1,28):dev.p.write_new(tmp_path/'attempts'/f'synthetic{i}.json',{})
    with pytest.raises(ValueError,match='cap'):dev.claim(tmp_path,configs()[1],fold)
    assert len((tmp_path/'ledger.jsonl').read_text().splitlines())==1


def test_ranking_never_prefers_lower_mse_over_hss_timing():
    grid=configs()[:2];folds=dev.fold_grid()[:1]
    rows=[]
    for index,config in enumerate(grid):
        m=dict(hss=dict(actual_count=2,predicted_count=2,tp=1,fn=1,fp=1,penalty_hours=50-index),
               modal_unique_events=dict(tp=1,fp=0,fn=1),squared_error_sum=1+index,naive_squared_error_sum=100,n_pairs=72)
        rows.append(dict(config_sha256=dev.config_hash(config),fold=folds[0],status='ok',details={'metrics':m}))
    assert dev.rank_results(rows,grid,folds)['selected_config']==grid[1]
    for row in rows:row['details']['metrics']['hss']['tp']=0
    ranked=dev.rank_results(rows,grid,folds)
    assert ranked['selected_config']==grid[0] and ranked['fallback_dev_mse']


def test_watchdog_commands_use_new_namespace_and_clock():
    for stage,module in [('dev','dev'),('finalize','finalize'),('evaluate','evaluate')]:
        command=controller.command(Path('/tmp/new-state'),stage)
        assert 'experiments.mse20_hss_state.'+module in command
        assert not any('mse20_hss_arrival.' in item for item in command)
    assert str(controller.p.CLOCK) in controller.command(Path('/tmp/new-state'),'dev')


def test_watchdog_reserves_refit_and_releases_inference_to_finaldeadline(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(controller.p,'deadline',lambda *args:calls.append(args) or 100)
    controller.remaining(tmp_path,'dev');assert calls[-1][1:]==('search',1800)
    controller.remaining(tmp_path,'finalize');assert calls[-1][1:]==('search',)
    (tmp_path/'final-fit.json').write_text('{}')
    controller.remaining(tmp_path,'finalize');assert calls[-1][1:]==('final',)


def test_watchdog30second_cleanup_grace_and_birth_guard(monkeypatch):
    class Child:
        pid=123
        def __init__(self):self.waits=[]
        def poll(self):return None
        def wait(self,timeout):
            self.waits.append(timeout)
            if len(self.waits)==1:raise subprocess.TimeoutExpired('test',timeout)
    child=Child();signals=[]
    monkeypatch.setattr(controller.p,'process_birth',lambda _: 'birth')
    monkeypatch.setattr(controller.os,'killpg',lambda pid,sig:signals.append(sig))
    controller.stop_owned(child,'birth')
    assert child.waits==[30,10] and signals==[signal.SIGTERM,signal.SIGKILL]
    with pytest.raises(RuntimeError,match='identity'):controller.stop_owned(Child(),'different')


@pytest.fixture
def producer(tmp_path,monkeypatch):
    from experiments import mse20_hss_state as package
    index=pd.date_range('2026-05-01',periods=1800,freq='h',tz='UTC');x=np.arange(len(index))
    ace=pd.DataFrame({'filled_speed_kms':400+np.sin(x/10),'was_missing':0},index=index)
    ch=pd.DataFrame({col:.01 for col in f.inputs.CH_COLUMNS},index=index)
    raw=pd.DataFrame({'ace_density_cm3':5.,'ace_temperature_k':1e5,'ace_bt_nt':5.,'ace_speed_kms':ace.filled_speed_kms},index=index)
    frames=f.inputs.rebuild_frames(ace,ch,raw);origins=index[[900,1100,1400]]
    config=configs()[0];registration={'sources':{}}
    selection={'frozen_utc':'2026-10-03T10:00:00Z','selected_config':config,'selection_uses_official':False}
    f.p.write_new(tmp_path/'registered.json',registration);f.p.write_new(tmp_path/'dev-selection.json',selection)
    monkeypatch.setattr(f,'final_context',lambda run:(tmp_path,registration,selection,config,origins,{}))
    monkeypatch.setattr(f,'issued_frames',lambda *a:frames)
    monkeypatch.setattr(f.p,'load_frames',lambda *a:frames)
    monkeypatch.setattr(f.p,'verify_registration',lambda *a:registration)
    monkeypatch.setattr(f.p,'deadline',lambda *a,**k:10000)
    monkeypatch.setattr(f.p,'memory_admission',lambda *a,**k:contextlib.nullcontext())
    monkeypatch.setattr(f.p,'gpu_lease',lambda *a,**k:contextlib.nullcontext('cuda:0'))
    calls=[]
    def predict(frames,origins,directory,device):
        calls.append(('predict',device,len(origins)))
        pred=pd.DataFrame({f.p.KEY[0]:np.repeat(origins,72),'horizon_hours':np.tile(np.arange(1,73),len(origins)),
                           'pred_kms':np.repeat(frames['speed'].reindex(origins).to_numpy(),72)})
        return {'mse':pred,'peak':pred.copy()}
    def fit(frames,cutoff,origins,config,device,directory):
        calls.append(('fit',device,len(origins)))
        assert (tmp_path/'final-fit-claim.json').exists() and not (tmp_path/'prediction-pair.json').exists()
        assert origins.tolist()==[pd.Timestamp(f.p.CUTOFF)]
        directory.mkdir(parents=True);(directory/'model.pt').write_bytes(b'synthetic')
        meta=dict(config=config,cutoff=cutoff.isoformat(),train_label_max=cutoff.isoformat())
        (directory/'metadata.json').write_text(json.dumps(meta))
        return {},meta
    fake=types.SimpleNamespace(fit_predict=fit,predict_saved=predict,verify_training_provenance=lambda *a:{'shape_label_sha256':'synthetic','censor_mask_sha256':'synthetic'})
    monkeypatch.setattr(package,'model',fake,raising=False)
    return tmp_path,frames,origins,calls


def test_final_refit_once_and_paired24causality_checks(producer):
    run,frames,origins,calls=producer
    fit=f.fit_final(run);f.fit_final(run)
    assert sum(row[0]=='fit' for row in calls)==1
    assert fit['claim']==f.p.file_ref(run/'final-fit-claim.json')
    proof=f.replay(run)
    assert len(proof['checks'])==24 and len(proof['subset_checks'])==3
    assert sum(row['kind']=='raw' for row in proof['checks'])==18
    assert all(row['max_abs_diff_kms']=={'mse':0.,'peak':0.} for row in proof['checks'])
    assert f.p.read_json(proof['provenance']['path'])['reconstructed']['censor_mask_sha256']=='synthetic'
    with np.load(proof['inventory']['path']) as data:assert set(data.files)==set(f.RAW+f.DERIVED)


def test_interrupted_finalfit_claim_not_retried(producer):
    run,*_=producer
    f.p.write_new(run/'final-fit-claim.json',{'consumed':True})
    with pytest.raises(FileExistsError):f.fit_final(run)


def test_import_torch_free_and_shared_diagnostics_readonly():
    result=subprocess.run([str(f.p.ROOT/'.venv-fusion/bin/python'),'-c',
        'import sys;from experiments.mse20_hss_state import dev,finalize,run;assert "torch" not in sys.modules;assert dev.d.__name__=="experiments.mse20_hss_arrival.diagnostics"'],
        cwd=f.p.ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_dev_only_controller_freezes_without_final_fit_or_evaluation(tmp_path,monkeypatch):
    stages=[];frozen=[]
    monkeypatch.setattr(controller.p,'verify_registration',lambda _: {})
    monkeypatch.setattr(controller.p,'require_legacy_complete',lambda:None)
    monkeypatch.setattr(controller,'run_stage',lambda run,stage:stages.append(stage) or {'stage':stage,'exit_code':0})
    monkeypatch.setattr(dev,'recover_dead_claims',lambda _:None)
    monkeypatch.setattr(dev,'freeze',lambda run,clock:frozen.append(run))
    assert controller.supervise(tmp_path,stage='dev')==0
    assert stages==['dev'] and frozen==[tmp_path]
    result=json.loads(next(tmp_path.glob('controller-result-*.json')).read_text())
    assert result['status']=='dev_complete' and result['native_goal_mutated'] is False
    assert not (tmp_path/'final-fit-claim.json').exists()


def test_dev_only_cli_flag_is_routed_without_running_final_stages(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(controller,'supervise',lambda run,stage:calls.append((run,stage)) or 0)
    monkeypatch.setattr(controller.signal,'signal',lambda *args:None)
    assert controller.main(['--run-dir',str(tmp_path),'--stage','dev'])==0
    assert calls==[(tmp_path,'dev')]
    calls.clear()
    assert controller.main(['--run-dir',str(tmp_path)])==0
    assert calls==[(tmp_path,'all')]


def test_direct_user_gate_and_clock_override_unconfirmed_document(tmp_path,monkeypatch):
    from experiments.mse20_hss_state import protocol as p
    document=tmp_path/'alignment.md';document.write_text('Section8 says relaxed gates and extended clock')
    runtime=tmp_path/'runtime.json'
    p.write_new(runtime,dict(started_utc='2026-10-03T06:26:29.494380Z',
        search_stop_utc='2026-10-04T06:26:29.494380+00:00',optimization_stop_utc='2026-10-04T06:26:29.494380+00:00',
        deadline_utc='2026-10-04T07:41:29.494380+00:00'))
    directive=tmp_path/'directive.json'
    record=dict(regime_cells_final_gate=True,search_stop_utc='2026-10-04T06:26:29.494380Z',
                deadline_utc='2026-10-04T07:41:29.494380Z',unconfirmed_document_section8_applied=False)
    p.write_new(directive,record)
    auth=dict(user_document=p.file_ref(document),directive_record=p.file_ref(directive),
              peak_gate='unique_event_f1_nonregression_vs_48c28',total_mse_max=p.TOTAL_MAX,
              training_cutoff_utc=p.CUTOFF,official_for_selection=False,dual_output_same_model=True,
              additional_tree_hpo=False,simplex_blending=False,regime_cells_final_gate=True,
              runtime=p.file_ref(runtime),files={})
    monkeypatch.setattr(p,'USER_DOCUMENT',document);monkeypatch.setattr(p,'CLOCK',tmp_path)
    monkeypatch.setattr(p,'read_json',lambda path:auth if Path(path).name=='authorization.json' else json.loads(Path(path).read_text()))
    assert p.setup(tmp_path)==auth
    auth['regime_cells_final_gate']=False
    with pytest.raises(ValueError,match='boundary'):p.setup(tmp_path)
    auth['regime_cells_final_gate']=True
    record['search_stop_utc']='2026-10-06T06:26:29.494380Z'
    directive.write_text(json.dumps(record));auth['directive_record']=p.file_ref(directive)
    with pytest.raises(ValueError,match='authority'):p.setup(tmp_path)


def test_evaluator_routes_only_current_cycle_and_requires_caps(tmp_path,monkeypatch):
    from experiments.mse20_hss_state import evaluate as e
    seen=[]
    monkeypatch.setattr(e,'_verify_state',lambda run:seen.append(run) or {'status':'unmet'})
    assert e.verify(tmp_path)=={'status':'unmet'} and seen==[tmp_path]
    with pytest.raises(Exception,match='mandatory four'):
        e.score_pair(None,None,None,False)


def test_actual_cpu_model_metadata_matches_independent_evaluator(tmp_path):
    from experiments.mse20_hss_state import model as m,evaluate as e
    from test_mse20_hss_arrival_model import frames
    data=frames();cutoff=data['speed'].index[1040];origins=data['speed'].index[[1041]]
    config=m.candidate_configs()[0]
    _,metadata=m.fit_predict(data,cutoff,origins,config,'cpu',tmp_path/'model')
    refs=e.p.sources()
    actual=e.verify_metadata(e.p.file_ref(tmp_path/'model/metadata.json'),
                             e.p.file_ref(tmp_path/'model/model.pt'),config,cutoff,refs)
    assert actual==metadata
    actual['family']='wrong'
    path=tmp_path/'model/metadata.json';path.write_text(json.dumps(actual))
    (path.parent/'metadata.sha256').write_text(e.p.digest(path))
    with pytest.raises(Exception,match='same fitted configuration'):
        e.verify_metadata(e.p.file_ref(path),e.p.file_ref(path.parent/'model.pt'),config,cutoff,refs)
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.mse20_hss_state import evaluate as e


def test_parent_import_is_torch_free():
    code='import sys; from experiments.mse20_hss_state import evaluate; assert "torch" not in sys.modules'
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
    names=['experiments/mse20_hss_state/model.py','experiments/mse20_hss_arrival/diagnostics.py',
           'experiments/mse20_ch_dynamics/inputs.py','experiments/mse20_ch_dynamics/arrival.py',
           'experiments/mse20_ch_dynamics/neural.py','experiments/mse20_hss_shape/model.py',
           'experiments/mse20_four_regimes/protocol.py','experiments/ch_breakthrough_v2/common.py',
           'experiments/ch_breakthrough_v2/physical_quality.py','experiments/ch_breakthrough_v2/recurrence.py']
    sources={name:e.p.file_ref(e.p.ROOT/name) for name in names}
    meta=dict(family='hss_state_recurrence',schema=dict(version=3,shape_definition={'decoder':'anchored'}),
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
    registration=dict(configs=list(range(2)),folds=list(range(14)))
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
    configs=[{'id':str(i)} for i in range(2)]
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


def test_shape_claim_cap_is_28_even_when_all_claims_failed(tmp_path):
    (tmp_path/'attempts').mkdir()
    for number in range(29):(tmp_path/'attempts'/f'{number}.json').write_text('{}')
    with pytest.raises(ValueError,match='grid/cap'):
        e.verify_dev(tmp_path,dict(configs=list(range(4)),folds=list(range(14))),{}, {})


def test_shape_uses_frozen_event_definition_not_new_variant():
    from experiments.mse20_hss_arrival import diagnostics
    assert e.d is diagnostics
