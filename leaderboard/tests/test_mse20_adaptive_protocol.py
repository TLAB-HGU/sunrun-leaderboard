import json
from pathlib import Path
import fcntl

import pytest

from experiments.mse20_adaptive import protocol as p, evaluate as e
from experiments.mse20_four_regimes import evaluate as old


def metric(mse=4100, cell=5000, precision=1, recall=1/818, status='unmet'):
    return {'status':status,'official':{'mse':mse,'by_regime':{k:{'mse':cell} for k in old.BASELINE_CELLS}},
            'events':{'precision':precision,'recall':recall}}


def test_rank_full_goal_then_event_nonregression_then_worst_cell():
    assert p.rank(metric(3280,3700,status='pass'))<p.rank(metric(3270,3900))
    assert p.rank(metric(5000,7000))<p.rank(metric(3000,3500,precision=.99))
    assert p.rank(metric(4000,4000))<p.rank(metric(3900,6000))
    assert p.rank(metric(recall=None))[0]==2


def test_setup_is_not_goal_success(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'setup',lambda _:None)
    assert e.main(['--mode','setup','--run-dir',str(tmp_path)])==0
    proof=json.loads((tmp_path/'setup-verification.json').read_text())
    assert proof['objective_achieved'] is False and proof['status']=='setup_verified'


def test_missing_manifest_fails_closed_json(tmp_path):
    assert e.main(['--run-dir',str(tmp_path)])==2
    proof=json.loads((tmp_path/'acceptance.json').read_text())
    assert proof['objective_achieved'] is False and proof['status']=='indeterminate'


@pytest.mark.parametrize('status,code',[('pass',0),('unmet',1),('indeterminate',2)])
def test_amended_cli_exit_status(status,code,tmp_path,monkeypatch):
    monkeypatch.setattr(e,'evaluate',lambda _:{'status':status,'exit_code':code,'objective_achieved':status=='pass'})
    assert e.main(['--run-dir',str(tmp_path)])==code
    assert json.loads((tmp_path/'acceptance.json').read_text())['objective_achieved']==(status=='pass')


def test_lease_bounded_and_compatible_files(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path)
    monkeypatch.setattr(p,'deadline',lambda *args:1000)
    d=tmp_path/p.LEASE_DIR; d.mkdir(parents=True)
    handles=[(d/f'gpu{i}.lock').open('a+') for i in (0,1)]
    for h in handles: fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        with pytest.raises(TimeoutError,match='bounded30s'):
            with p.gpu_lease(timeout_seconds=.02): pass
    finally:
        for h in handles: h.close()


def test_lease_releases_after_capacity_and_error(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path)
    monkeypatch.setattr(p,'deadline',lambda *args:1000)
    monkeypatch.setattr(p.subprocess,'check_output',lambda *a,**k:'0,24564')
    with pytest.raises(RuntimeError,match='fit failure'):
        with p.gpu_lease(timeout_seconds=.02) as device:
            assert device=='cuda:0'
            raise RuntimeError('fit failure')
    with p.gpu_lease(timeout_seconds=.02) as device: assert device=='cuda:0'


def test_lease_rechecks_deadline_after_claim(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path)
    calls=0
    def check(*args):
        nonlocal calls
        calls+=1
        if calls>1: raise TimeoutError('search budget exhausted')
        return 1000
    monkeypatch.setattr(p,'deadline',check)
    with pytest.raises(TimeoutError):
        with p.gpu_lease(timeout_seconds=.02): pass
    d=tmp_path/p.LEASE_DIR
    for i in (0,1):
        with (d/f'gpu{i}.lock').open('a+') as handle: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)


def test_lease_probe_errors_propagate_and_capacity_retries(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path)
    monkeypatch.setattr(p,'deadline',lambda *args:1000)
    def fail(*a,**k): raise ValueError('primary probe failure')
    monkeypatch.setattr(p.subprocess,'check_output',fail)
    with pytest.raises(ValueError,match='primary probe failure'):
        with p.gpu_lease(timeout_seconds=.02): pass
    monkeypatch.setattr(p.subprocess,'check_output',lambda *a,**k:'12000,24564')
    with pytest.raises(TimeoutError,match='no safe shared'):
        with p.gpu_lease(timeout_seconds=.02): pass


def test_trial_tamper_and_fixed_cutoff(tmp_path):
    f=tmp_path/'weights'; f.write_text('weight')
    row={'status':'ok','files':{'model':p.file_ref(f)},'training_cutoff_utc':p.CUTOFF,
         'metadata':{'train_label_max':p.CUTOFF}}
    p.verify_trial(row)
    row['metadata']['train_label_max']='2026-06-01T00:00:00Z'
    with pytest.raises(ValueError,match='fixed cutoff'): p.verify_trial(row)
    row['metadata']['train_label_max']=p.CUTOFF; f.write_text('changed')
    with pytest.raises(ValueError,match='stale'): p.verify_trial(row)
