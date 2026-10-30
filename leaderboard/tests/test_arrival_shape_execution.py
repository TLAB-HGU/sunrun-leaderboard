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
from experiments.mse20_hss_shape import dev,finalize as f,run as controller


def configs():
    return [dict(id=f'b{b}s{s}',bin_hours=b,slow_hours=s,epochs=20,max_train_origins=4096)
            for b in (6,12) for s in (72,168)]


def test_four_structural_configs_fourteen_same_folds_and56cap(tmp_path,monkeypatch):
    payload=dev.registration_payload()
    assert payload['max_attempts']==56 and len(payload['folds'])==14
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
    for i in range(1,56):dev.p.write_new(tmp_path/'attempts'/f'synthetic{i}.json',{})
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
        command=controller.command(Path('/tmp/new-shape'),stage)
        assert 'experiments.mse20_hss_shape.'+module in command
        assert not any('mse20_hss_arrival.' in item for item in command)
    assert str(controller.p.CLOCK) in controller.command(Path('/tmp/new-shape'),'dev')


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
    from experiments import mse20_hss_shape as package
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
        'import sys;from experiments.mse20_hss_shape import dev,finalize,run;assert "torch" not in sys.modules;assert dev.d.__name__=="experiments.mse20_hss_arrival.diagnostics"'],
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
