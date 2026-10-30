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
from experiments.mse20_hss_phase import dev,finalize as f,run as controller


def configs():
    return [dict(id=name,gain=gain,epochs=20,max_train_origins=4096)
            for name,gain in [('shift_only',False),('shift_gain',True)]]


def test_two_structural_configs_fourteen_same_folds_and28cap(tmp_path,monkeypatch):
    payload=dev.registration_payload()
    assert payload['max_attempts']==28 and len(payload['folds'])==14
    from experiments.mse20_hss_arrival import dev as old
    assert payload['folds']==old.registration_payload()['folds']
    registration=dict(configs=configs(),folds=payload['folds'],ranking=payload['ranking'])
    monkeypatch.setattr(dev.p,'verify_registration',lambda _:registration)
    assert dev.verify_registration(tmp_path)==registration
    registration['configs'][0]['gain']=True
    with pytest.raises(ValueError,match='ablation'):dev.verify_registration(tmp_path)


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
        command=controller.command(Path('/tmp/new-phase'),stage)
        assert 'experiments.mse20_hss_phase.'+module in command
        assert not any('mse20_hss_arrival.' in item for item in command)
    assert str(controller.p.CLOCK) in controller.command(Path('/tmp/new-phase'),'dev')


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
    from experiments import mse20_hss_phase as package
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
    from experiments.mse20_hss_phase import selection_bridge
    monkeypatch.setattr(selection_bridge,'verify_selection',lambda run:dict(selected_run=str(run),selected_config=config))
    f.p.write_new(tmp_path/'global-dev-selection.json',{'synthetic':True})
    monkeypatch.setattr(f,'model_inventory',lambda directory:[f.p.file_ref(directory/'model.pt')])
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
    fake=types.SimpleNamespace(fit_predict=fit,predict_saved=predict,verify_training_provenance=lambda *a:{'train_assignment_sha256':'synthetic','train_label_hashes':{'soft_shift':'synthetic'}})
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
    assert f.p.read_json(proof['provenance']['path'])['reconstructed']['train_assignment_sha256']=='synthetic'
    with np.load(proof['inventory']['path']) as data:assert set(data.files)==set(f.RAW+f.DERIVED)


def test_interrupted_finalfit_claim_not_retried(producer):
    run,*_=producer
    f.p.write_new(run/'final-fit-claim.json',{'consumed':True})
    with pytest.raises(FileExistsError):f.fit_final(run)


def test_import_torch_free_and_shared_diagnostics_readonly():
    result=subprocess.run([str(f.p.ROOT/'.venv-fusion/bin/python'),'-c',
        'import sys;from experiments.mse20_hss_phase import dev,finalize,run;assert "torch" not in sys.modules;assert dev.d.__name__=="experiments.mse20_hss_arrival.diagnostics"'],
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


@pytest.fixture
def composite(tmp_path,monkeypatch):
    from experiments.mse20_hss_phase import model as m,evaluate as e
    from test_arrival_phase_model import frames,_fake_backbone
    data=frames(5900);cutoff=data['speed'].index[5700];origins=data['speed'].index[[5701,5702]]
    pool=pd.DatetimeIndex([cutoff-pd.Timedelta(days=120),cutoff-pd.Timedelta(days=30)])
    monkeypatch.setattr(m,'training_origins',lambda *args:pool)
    calls=_fake_backbone(monkeypatch)
    fake_fit=m.backbone.fit_predict
    def fit(*args):
        pair,meta=fake_fit(*args);directory=Path(args[-1])
        meta['source_sha256']={name:m.digest(Path(m.backbone.__file__).with_name(name)) for name in ('features.py','tree.py')}
        (directory/'metadata.json').write_text(json.dumps(meta))
        (directory/'metadata.sha256').write_text(m.digest(directory/'metadata.json'))
        return pair,meta
    monkeypatch.setattr(m.backbone,'fit_predict',fit)
    p=e.p;sources={name:p.file_ref(p.ROOT/name) for name in e.MODEL_SOURCE_NAMES}
    fold=dict(id='synthetic',start_utc=(cutoff+pd.Timedelta(hours=1)).isoformat(),end_utc=(cutoff+pd.Timedelta(days=30)).isoformat())
    registration=dict(registered_utc=p.utcnow(),configs=m.candidate_configs(),sources=sources,folds=[fold])
    p.write_new(tmp_path/'registered.json',registration)
    p.write_new(tmp_path/'attempts/head.json',dict(fold=fold,claimed_utc=p.utcnow()))
    config=m.candidate_configs()[0];directory=tmp_path/'outputs/model'
    pair,meta=m.fit_predict(data,cutoff,origins,config,'cpu',directory)
    return tmp_path,data,cutoff,origins,pair,meta,directory,config,sources,calls


def test_actual_composite_metadata_inventory_and_immutable_oof(composite):
    from experiments.mse20_hss_phase import evaluate as e
    root,data,cutoff,origins,pair,meta,directory,config,sources,calls=composite
    result=e.verify_metadata(e.p.file_ref(directory/'metadata.json'),e.p.file_ref(directory/'model.pt'),config,cutoff,sources)
    assert result==meta and len(calls)==3
    refs=f.model_inventory(directory)
    assert len(refs)==4 and [Path(ref['path']).suffix for ref in refs]==['.pt','.ubj','.ubj','.ubj']
    assert set(pair)=={'mse','peak','backbone'}


@pytest.mark.parametrize('tamper',['assignment','oof','native','source','normalizer','cutoff'])
def test_composite_tamper_fails_closed(composite,tamper):
    from experiments.mse20_hss_phase import evaluate as e
    root,data,cutoff,origins,pair,meta,directory,config,sources,calls=composite
    if tamper=='assignment':meta['train_assignment'][0]=1
    if tamper=='oof':Path(meta['oof_forecasts']['path']).write_bytes(b'changed')
    if tamper=='native':Path(meta['backbones'][0]['model_files'][-1]['path']).write_bytes(b'changed')
    if tamper=='source':meta['sources']={}
    if tamper=='normalizer':meta['normalizer']['scale'][0]=0
    if tamper=='cutoff':meta['train_label_max']='2026-06-01T00:00:00Z'
    path=directory/'metadata.json';path.write_text(json.dumps(meta));(directory/'metadata.sha256').write_text(e.p.digest(path))
    with pytest.raises(ValueError):
        e.verify_metadata(e.p.file_ref(path),e.p.file_ref(directory/'model.pt'),config,cutoff,sources)


def test_backbone_diagnostic_is_saved_and_scored_but_never_used_for_ranking(monkeypatch):
    calls=[]
    def score(pair,frames,fold):
        calls.append(pair)
        return {'mse':pair['mse']}
    monkeypatch.setattr(dev,'_score_pair',score)
    value=dev.score_pair({'mse':10,'peak':20,'backbone':30},None,None)
    assert value=={'mse':10,'uncorrected_backbone':{'mse':30}}
    assert calls[1]=={'mse':30,'peak':30}
    with pytest.raises(ValueError,match='backbone'):dev.score_pair({'mse':10,'peak':20},None,None)


def test_final_evaluator_rejects_removed_regime_gate():
    from experiments.mse20_hss_phase import evaluate as e
    with pytest.raises(ValueError,match='mandatory'):e.score_pair(None,None,None,False)


def test_all_shared_backbone_claims_count_and_require_terminal_or_dead_owner(composite,monkeypatch):
    from experiments.mse20_hss_phase import evaluate as e
    root,data,cutoff,origins,pair,meta,directory,config,sources,calls=composite
    monkeypatch.setattr(e.dev,'fold_origins',lambda fold:origins)
    registration=e.p.read_json(root/'registered.json')
    result=e.verify_backbone_inventory(root,registration)
    assert result['counts']=={'dev':3,'final':0} and len(result['claims'])==3
    Path(meta['backbones'][0]['result']['path']).unlink()
    with pytest.raises(ValueError,match='live nonterminal'):e.verify_backbone_inventory(root,registration)
    monkeypatch.setattr(e.p,'process_birth',lambda pid:None)
    result=e.verify_backbone_inventory(root,registration)
    assert result['counts']['dev']==3
    assert sum(item['status']=='interrupted_consumed' for item in result['claims'])==1


def test_shared_backbone_scope_tamper_rejected(composite,monkeypatch):
    from experiments.mse20_hss_phase import evaluate as e
    root,data,cutoff,origins,pair,meta,directory,config,sources,calls=composite
    monkeypatch.setattr(e.dev,'fold_origins',lambda fold:origins)
    claim=Path(meta['backbones'][0]['claim']['path']);value=e.p.read_json(claim)
    value['identity']['scope']='final';claim.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='scope/cutoff'):e.verify_backbone_inventory(root,e.p.read_json(root/'registered.json'))


@pytest.mark.parametrize('mse,cell,tp,status',[(3200.,3700.,1,'pass'),(3290.,3700.,2,'unmet'),
                                             (3200.,3800.,2,'unmet'),(3200.,3700.,0,'unmet')])
def test_all_six_scientific_gates_retained(tmp_path,monkeypatch,mse,cell,tp,status):
    from experiments.mse20_hss_phase import evaluate as e
    frame=pd.DataFrame({e.p.KEY[0]:['2026-06-01T00:00:00Z'],'horizon_hours':[1],'pred_kms':[400.]})
    ref=tmp_path/'ref.parquet';frame.to_parquet(ref)
    monkeypatch.setattr(e.old,'validate_frame',lambda frame,*args:frame)
    monkeypatch.setattr(e.old.scoring,'score',lambda *args:dict(mse=mse,n_pairs=203976,n_folds=2833,
        by_regime={key:dict(mse=cell,reliable=True) for key in e.old.BASELINE_CELLS}))
    calls=[]
    def unique(*args):
        count=tp if not calls else 1;calls.append(1)
        return dict(tp=count,fp=0,fn=23-count,f1=2*count/(23+count))
    monkeypatch.setattr(e.d,'unique_event_metrics',unique)
    benchmark=dict(folds=None,truth=frame,naive=frame,paths={'reference':str(ref)},target=None,
                   observed=None,origins=None,actual_events=None)
    result=e.score_pair(frame,frame,benchmark,True)
    assert result['status']==status
    assert set(result['gates'])=={'total_mse','unique_event_f1',*e.old.BASELINE_CELLS}


def test_final_refit_requires_global_dev_winner(producer,monkeypatch):
    from experiments.mse20_hss_phase import selection_bridge
    run,frames,origins,calls=producer
    monkeypatch.setattr(selection_bridge,'verify_selection',lambda root:dict(selected_run=str(root/'other'),selected_config={}))
    with pytest.raises(ValueError,match='global dev winner'):f.fit_final(run)
    assert not (run/'final-fit-claim.json').exists() and not calls
