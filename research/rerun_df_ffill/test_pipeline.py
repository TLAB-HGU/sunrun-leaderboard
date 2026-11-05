"""Independent campaign integration checks; all processes/network writes mocked."""
import json
from pathlib import Path
from types import SimpleNamespace

import launch
import numpy as np
import pandas as pd
import publish
import pytest

HERE = Path(__file__).parent


@pytest.mark.parametrize('family',['strategies','foundation','direct','tcn','special'])
def test_launch_freezes_source_records_threads_and_passes_raw_data(tmp_path,monkeypatch,family):
    monkeypatch.setattr(launch,'HERE',tmp_path)
    (tmp_path/'manifest.json').write_text(json.dumps({'experiments':[{'experiment':'example','original_submission_utc':'old'}]}))
    (tmp_path/'input').mkdir(); (tmp_path/'input/df_ffill_trainy.parquet').write_bytes(b'input snapshot')
    original=(HERE/f'{family}.py').read_text()
    (tmp_path/f'{family}.py').write_text(original)
    calls=[]
    monkeypatch.setattr(launch.subprocess,'run',lambda command,**kwargs:calls.append(command) or SimpleNamespace(returncode=0))
    gpu = 0 if family in ('foundation', 'tcn') else None
    monkeypatch.setattr(launch, 'gpu0_available', lambda: True)
    monkeypatch.setattr(launch, 'GPU0_LEASE', tmp_path / 'gpu0.lock')
    launch.launch('example',family,'local',3,gpu=gpu)
    out=tmp_path/'runs/example'
    receipt=json.loads((out/'launch.json').read_text())
    script = (out / 'run.sh').read_text()
    assert ('CUDA_VISIBLE_DEVICES=0' if gpu == 0 else "CUDA_VISIBLE_DEVICES=") in script
    assert receipt['gpu'] == gpu
    assert receipt['threads']==3
    assert '--raw-data' in receipt['command']
    assert receipt['command'][1]==str(out/'experiment.py')
    assert receipt['input_identity']['bytes']==len(b'input snapshot')
    assert (out/'experiment.py').read_bytes()==(out/'runner_at_launch.py').read_bytes()
    assert calls[0][:2]==['tmux','new-session']
    monkeypatch.setattr(publish,'HERE',tmp_path)
    saved=publish.source_for('example',family)
    (tmp_path/f'{family}.py').write_text('later mutable source')
    assert publish.source_for('example',family)==saved
    with pytest.raises(FileExistsError,match='Nonempty'):
        launch.launch('example',family,'local',3,gpu=gpu)


def test_source_guard_rejects_changed_executable(tmp_path,monkeypatch):
    monkeypatch.setattr(publish,'HERE',tmp_path)
    out=tmp_path/'runs/example';out.mkdir(parents=True)
    (out/'experiment.py').write_text('changed')
    (out/'runner_at_launch.py').write_text('original')
    with pytest.raises(ValueError,match='changed'):
        publish.source_for('example','direct')


@pytest.fixture
def receipt(tmp_path,monkeypatch):
    monkeypatch.setattr(publish,'HERE',tmp_path)
    out=tmp_path/'runs/e2000_direct_s42';out.mkdir(parents=True)
    data=tmp_path/'input/df_ffill_trainy.parquet';data.parent.mkdir();data.write_bytes(b'new input')
    record={'family':'direct','training_target_ffill':True,'command':['python','runner','--data',str(data)],
            'input_identity':{'path':str(data),'bytes':data.stat().st_size,'mtime_ns':data.stat().st_mtime_ns}}
    (out/'launch.json').write_text(json.dumps(record));(out/'train.exit').write_text('0')
    (out/'model.ubj').write_bytes(b'model fixture')
    (out/'merge.json').write_text(json.dumps({'inference_seconds_per_fold':.01}))
    reports=[]
    for period in ('F1','F2','F3'):
        for h in range(1,73):
            report={'experiment':'e2000_direct_s42','period':period,'lead':h,'rounds':900,
                    'data':record['input_identity']}
            path=out/f'{period}_h{h:02d}.json';path.write_text(json.dumps(report));reports.append(path)
    origins=np.repeat(pd.date_range('2025-01-01',periods=6315,freq='h',tz='UTC'),72)
    lead=np.tile(np.arange(1,73),6315)
    pred=pd.DataFrame({'origin_last_input_utc':origins,'horizon_hours':lead,'pred_kms':400.})
    query=pd.DataFrame({'origin':origins,'h':lead,'period':'F1','set':'query'})
    monkeypatch.setattr(publish.pd,'read_parquet',lambda path,**kwargs:pred if Path(path).name=='predictions.parquet' else query)
    return out,reports,record


def test_complete_direct_receipt_validates(receipt):
    out,_reports,_=receipt
    got=publish.validate_run('e2000_direct_s42')
    assert got[0]==out and len(got[2])==216 and got[3]==.01


def test_old_feature_only_receipt_rejected(receipt):
    out,_,record=receipt
    record['training_target_ffill']=False
    (out/'launch.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='training-target-ffill'):
        publish.validate_run('e2000_direct_s42')


def test_smoke_rounds_without_smoke_flag_rejected(receipt):
    _,reports,_=receipt
    report=json.loads(reports[0].read_text());report['rounds']=2
    reports[0].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        publish.validate_run('e2000_direct_s42')


def test_missing_direct_horizon_report_rejected(receipt):
    _,reports,_=receipt
    reports[0].unlink()
    with pytest.raises(ValueError):
        publish.validate_run('e2000_direct_s42')


def test_changed_input_identity_in_fit_report_rejected(receipt):
    _,reports,_=receipt
    report=json.loads(reports[0].read_text());report['data']['bytes']+=1
    reports[0].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        publish.validate_run('e2000_direct_s42')


def test_source_publication_cas_retry_refreshes_parent_without_force(monkeypatch):
    import sys
    import urllib.error
    calls=[]
    attempt=[0]
    class API:
        def __init__(self,*args):
            pass
        def request(self,method,path,body=None):
            calls.append((method,path,body))
            if path=='git/blobs':
                return {'sha':'blob'}
            if path=='git/ref/heads/main':
                return {'object':{'sha':f'parent{attempt[0]}'}}
            if path.startswith('git/commits/'):
                return {'tree':{'sha':'tree'}}
            if path=='git/trees':
                return {'sha':'newtree'}
            if path=='git/commits':
                return {'sha':f'commit{attempt[0]}'}
            if method=='PATCH':
                assert body['force'] is False
                if attempt[0]==0:
                    attempt[0]+=1
                    raise urllib.error.HTTPError('https://mock',409,'CAS conflict',None,None)
                return {}
            raise AssertionError((method,path))
    monkeypatch.setitem(sys.modules,'metric_assets',SimpleNamespace(GitHubAPI=API))
    monkeypatch.setattr(publish.time,'sleep',lambda _:None)
    assert publish.publish_sources({'runner.py':b'print(1)'})=='commit1'
    parents=[body['parents'] for method,path,body in calls if path=='git/commits']
    assert parents==[['parent0'],['parent1']]


def test_controller_marks_lost_session_interrupted_without_submission(tmp_path,monkeypatch):
    import controller
    monkeypatch.setattr(controller,'HERE',tmp_path)
    (tmp_path/'registry.json').write_text('{}')
    monkeypatch.setattr(controller,'records',lambda:{'example':{'host':'local','tmux_session':'example'}})
    assert not hasattr(controller, 'remote_state')
    monkeypatch.setattr(controller,'memory_local',lambda:0)
    monkeypatch.setattr(controller.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=1))
    statuses=[]
    monkeypatch.setattr(controller,'status_update',lambda *args:statuses.append(args))
    runtime={}
    controller.tick(runtime)
    assert runtime['example']['state']=='interrupted'
    assert statuses[0][1]=='중단'


@pytest.fixture
def physics_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'HERE', tmp_path)
    (tmp_path / 'input').mkdir()
    (tmp_path / 'gong_cache').mkdir()
    (tmp_path / 'gong_cache/exit.code').write_text('0\n')
    origins = pd.date_range('2024-06-01', periods=4, freq='h', tz='UTC')
    pd.DataFrame({'origin': origins, 'h': 1}).to_parquet(tmp_path / 'input/df_ffill_trainy.parquet')
    output = tmp_path / 'input/huxt_features.parquet'
    columns = ['origin', 'huxt_max', 'huxt_argmax', 'huxt_h1', 'huxt_h24', 'huxt_h48', 'huxt_h72']
    frame = pd.DataFrame({name: 400. for name in columns[1:]}, index=range(4))
    frame['huxt_argmax'] = 24
    frame.insert(0, 'origin', origins)
    frame.to_parquet(output, index=False)
    metadata = {'stage': 'all', 'old_huxt_predictions_used': False, 'output_columns': columns,
                'output': str(output.resolve()), 'original_sources': ['original PFSS'], 'versions': {'huxt': '5.0.3'},
                'rows': 4, 'requested_origins': 4, 'first_origin': str(origins.min()),
                'last_origin': str(origins.max()), 'origin_start': str(origins.min()), 'origin_end': str(origins.max())}
    output.with_suffix('.metadata.json').write_text(json.dumps(metadata))
    return tmp_path, output, metadata


def test_physics_dependency_validates_complete_coverage_and_provenance(physics_ready):
    _, output, metadata = physics_ready
    receipt = launch.physics_dependency()
    assert receipt['required_origins'] == 4
    assert receipt['features']['bytes'] == output.stat().st_size
    assert receipt['provenance'] == metadata


@pytest.mark.parametrize('damage', ['running', 'failed', 'missing_origin', 'duplicate', 'nan', 'old_predictions', 'wrong_metadata'])
def test_physics_dependency_rejects_incomplete_or_stale_artifacts(physics_ready, damage):
    root, output, metadata = physics_ready
    frame = pd.read_parquet(output)
    if damage == 'running':
        (root / 'gong_cache/exit.code').unlink()
    elif damage == 'failed':
        (root / 'gong_cache/exit.code').write_text('1')
    elif damage == 'missing_origin':
        frame = frame.iloc[:-1]
    elif damage == 'duplicate':
        frame.loc[1, 'origin'] = frame.loc[0, 'origin']
    elif damage == 'nan':
        frame.loc[0, 'huxt_h72'] = np.nan
    elif damage == 'old_predictions':
        metadata['old_huxt_predictions_used'] = True
    else:
        metadata['rows'] = 3
    frame.to_parquet(output, index=False)
    output.with_suffix('.metadata.json').write_text(json.dumps(metadata))
    with pytest.raises((OSError, ValueError)):
        launch.physics_dependency()


def test_hsspeak_launch_passes_physics_and_records_provenance(physics_ready, monkeypatch):
    root, output, _ = physics_ready
    (root / 'manifest.json').write_text(json.dumps({'experiments': [
        {'experiment': 'e1604_hsspeak25', 'original_submission_utc': 'old'}]}))
    (root / 'special_hsspeak.py').write_text((HERE / 'special_hsspeak.py').read_text())
    calls = []
    monkeypatch.setattr(launch.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    with pytest.raises(ValueError, match='local host'):
        launch.launch('e1604_hsspeak25', 'special_hsspeak', 'golab2', 2)
    launch.launch('e1604_hsspeak25', 'special_hsspeak', 'local', 2)
    record = json.loads((root / 'runs/e1604_hsspeak25/launch.json').read_text())
    assert record['command'][record['command'].index('--huxt-features') + 1] == str(output)
    assert record['physics_input']['required_origins'] == 4
    assert record['gpu'] is None
    assert len(calls) == 1


def test_controller_keeps_hsspeak_queued_until_physics_ready(tmp_path, monkeypatch):
    import controller
    monkeypatch.setattr(controller, 'HERE', tmp_path)
    (tmp_path / 'registry.json').write_text(json.dumps({'e1604_hsspeak25': {'family': 'special_hsspeak', 'threads': 2}}))
    monkeypatch.setattr(controller, 'records', lambda: {})
    assert not hasattr(controller, 'remote_state')
    monkeypatch.setattr(controller, 'memory_local', lambda: 32)
    monkeypatch.setattr(launch, 'gpu0_available', lambda: False)
    monkeypatch.setattr(launch, 'physics_dependency', lambda: (_ for _ in ()).throw(ValueError('physics pending')))
    monkeypatch.setattr(launch, 'launch', lambda *a, **kw: pytest.fail('physics must finish first'))
    monkeypatch.setattr(controller, 'status_update', lambda *a: pytest.fail('must remain queued'))
    runtime = {}
    controller.tick(runtime)
    assert runtime['e1604_hsspeak25']['dependency_note'] == 'physics pending'


def test_hsspeak_resume_rejects_changed_physics_before_process_start(physics_ready):
    identity = launch.physics_dependency()
    identity['features']['mtime_ns'] += 1
    with pytest.raises(ValueError, match='changed since launch'):
        launch.resume({'experiment': 'e1604_hsspeak25', 'family': 'special_hsspeak', 'host': 'local',
                       'physics_input': identity})

@pytest.mark.parametrize('host,family,gpu',[('golab2','foundation',0),('local','foundation',None),('local','direct',0)])
def test_device_policy_rejects_other_execution(host,family,gpu):
    with pytest.raises(ValueError):
        launch.launch('example',family,host,2,gpu=gpu)


def test_controller_admits_large_recipe_on_local_cpu(tmp_path,monkeypatch):
    import controller
    monkeypatch.setattr(controller,'HERE',tmp_path)
    (tmp_path/'registry.json').write_text(json.dumps({'large':{'family':'direct','threads':2}}))
    out=tmp_path/'runs/large';out.mkdir(parents=True)
    monkeypatch.setattr(controller,'records',lambda:{})
    assert not hasattr(controller, 'remote_state')
    monkeypatch.setattr(controller,'memory_local',lambda:32)
    monkeypatch.setattr(controller,'status_update',lambda *a:None)
    calls=[]
    def fake_launch(name,family,host,threads,gpu=None):
        calls.append((host,gpu))
        (out/'launch.json').write_text(json.dumps({'host':host,'gpu':gpu}))
    monkeypatch.setattr(launch,'launch',fake_launch)
    runtime={}
    controller.tick(runtime)
    assert calls==[('local',None)]
    assert runtime['large']['state']=='running'


def test_decommissioned_remote_never_calls_subprocess(monkeypatch):
    monkeypatch.setattr(launch.subprocess,"run",lambda *a,**k:pytest.fail("Remote subprocess forbidden"))
    with pytest.raises(RuntimeError,match="decommissioned"):
        launch.remote("anything")


@pytest.mark.parametrize('available,expected', [(False, []), (True, [('dl1', 0)])])
def test_controller_dl_waits_for_gpu_and_only_admits_one(tmp_path, monkeypatch, available, expected):
    import controller
    monkeypatch.setattr(controller, 'HERE', tmp_path)
    (tmp_path / 'registry.json').write_text(json.dumps({n: {'family': 'foundation', 'threads': 2} for n in ('dl1', 'dl2')}))
    monkeypatch.setattr(controller, 'records', lambda: {})
    monkeypatch.setattr(controller, 'memory_local', lambda: 32)
    monkeypatch.setattr(launch, 'gpu0_available', lambda: available)
    monkeypatch.setattr(controller, 'status_update', lambda *a: None)
    calls = []
    def fake_launch(name, family, host, threads, gpu=None):
        calls.append((name, gpu))
        out = tmp_path / 'runs' / name
        out.mkdir(parents=True)
        (out / 'launch.json').write_text(json.dumps({'host': host, 'gpu': gpu}))
    monkeypatch.setattr(launch, 'launch', fake_launch)
    controller.tick({})
    assert calls == expected


def test_controller_counts_only_current_registry_and_preserves_failed_exit(tmp_path, monkeypatch):
    import controller
    monkeypatch.setattr(controller, 'HERE', tmp_path)
    (tmp_path / 'registry.json').write_text(json.dumps({'failed': {'family': 'strategies', 'threads': 2}}))
    out = tmp_path / 'runs' / 'failed'
    out.mkdir(parents=True)
    (out / 'train.exit').write_text('137')
    monkeypatch.setattr(controller, 'records', lambda: {'failed': {'host': 'local'}})
    monkeypatch.setattr(controller, 'memory_local', lambda: 32)
    monkeypatch.setattr(controller, 'status_update', lambda *a: None)
    monkeypatch.setattr(launch, 'launch', lambda *a, **kw: pytest.fail('failed runs require diagnosis'))
    runtime = {'removed': {'state': 'submitted'}}
    summary = controller.tick(runtime)
    assert summary['submitted'] == 0
    assert summary['failed'] == 1
    assert runtime['failed']['exit_code'] == 137


def test_gpu_admission_requires_free_memory_and_handles_missing_driver(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'GPU0_LEASE', tmp_path / 'gpu.lock')
    calls = []
    def result(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='4096, 0, 0')
    monkeypatch.setattr(launch.subprocess, 'run', result)
    assert not launch.gpu0_available()
    assert '--query-gpu=memory.free,memory.used,utilization.gpu' in calls[0]
    monkeypatch.setattr(launch.subprocess, 'run', lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()))
    assert not launch.gpu0_available()


def test_controller_reserves_peak_memory_across_ticks(tmp_path, monkeypatch):
    import controller
    monkeypatch.setattr(controller, 'HERE', tmp_path)
    monkeypatch.setattr(controller, 'memory_total', lambda: 32)
    registry = {'e2000fs_mimo': {'family': 'strategies'}, 'other': {'family': 'strategies'}}
    (tmp_path / 'registry.json').write_text(json.dumps(registry))
    monkeypatch.setattr(controller, 'records', lambda: {'e2000fs_mimo': {'host': 'local', 'tmux_session': 'mimo'}})
    monkeypatch.setattr(controller, 'memory_local', lambda: 32)
    monkeypatch.setattr(controller.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0))
    monkeypatch.setattr(controller, 'status_update', lambda *a: None)
    monkeypatch.setattr(launch, 'launch', lambda *a, **kw: pytest.fail('MIMO reserves its peak before RSS grows'))
    summary = controller.tick({})
    assert summary['running'] == 1


@pytest.mark.parametrize('family,gpu', [('strategies', None), ('foundation', 0)])
def test_resume_rebuilds_run_script_with_device_policy(tmp_path, monkeypatch, family, gpu):
    monkeypatch.setattr(launch, 'HERE', tmp_path)
    monkeypatch.setattr(launch, 'GPU0_LEASE', tmp_path / 'gpu.lock')
    monkeypatch.setattr(launch, 'gpu0_available', lambda: True)
    out = tmp_path / 'runs' / 'example'
    out.mkdir(parents=True)
    command = ['python', str(out / 'experiment.py')]
    if gpu is not None:
        command += ['--device', 'cuda:0']
    record = dict(experiment='example', host='local', family=family, output=str(out),
                  tmux_session='example', gpu=gpu, command=command,
                  input_identity=dict(path='data', bytes=1, mtime_ns=1))
    calls = []
    monkeypatch.setattr(launch.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    launch.resume(record)
    script = (out / 'run.sh').read_text()
    assert ('CUDA_VISIBLE_DEVICES=0' if gpu == 0 else 'CUDA_VISIBLE_DEVICES=') in script
    assert calls[-1][:2] == ['tmux', 'new-session']


def test_controller_limits_publication_to_keep_training_queue_responsive(tmp_path, monkeypatch):
    import controller
    monkeypatch.setattr(controller, 'HERE', tmp_path)
    (tmp_path / 'registry.json').write_text(json.dumps({n: {'family': 'direct'} for n in ('a', 'b')}))
    for name in ('a', 'b'):
        out = tmp_path / 'runs' / name
        out.mkdir(parents=True)
        (out / 'train.exit').write_text('0')
    monkeypatch.setattr(controller, 'records', lambda: {n: {'host': 'local'} for n in ('a', 'b')})
    monkeypatch.setattr(controller, 'memory_local', lambda: 0)
    calls = []
    monkeypatch.setattr(controller.subprocess, 'run', lambda command, **kw: calls.append(command) or SimpleNamespace(returncode=0))
    summary = controller.tick({})
    assert len(calls) == 1
    assert summary['submitted'] == 1
    assert summary['trained'] == 1
