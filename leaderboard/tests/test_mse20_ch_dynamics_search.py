import signal
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.mse20_ch_dynamics import search as s
from experiments.mse20_ch_dynamics import protocol as p


def test_restart_preserves_exact_prefix_and_sequence():
    grid = {'tree': [dict(id=f't{i}', x=i) for i in range(36)]}
    reg = {'grid': grid}
    rows = [dict(family='tree', config=c, claimed_utc=f'2026-10-03T13:{i:02d}:00Z') for i,c in enumerate(grid['tree'][:18])]
    assert s.remaining_configs('tree', reg, rows) == grid['tree'][18:]
    assert grid['tree'][:18] + s.remaining_configs('tree', reg, rows) == grid['tree']
    rows[1]['config'] = grid['tree'][2]
    with pytest.raises(ValueError, match='prefix'):
        s.remaining_configs('tree', reg, rows)


def test_owned_timer_restores_on_timeout():
    previous = signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError):
        with s.owned_timeout(.01):
            signal.pause()
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


def test_owned_timer_does_not_steal_existing_timer():
    signal.setitimer(signal.ITIMER_REAL, 20)
    try:
        with pytest.raises(RuntimeError, match='another owner'):
            with s.owned_timeout(1):
                pass
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def test_pilot_can_force_early_stop_before_search_deadline(tmp_path, monkeypatch):
    p.write_new(tmp_path/'replay-pilots/x.json', dict(cpu_full_seconds=100,raw_rebuild_seconds=100,estimated_final_seconds=10000))
    monkeypatch.setattr(p, 'deadline', lambda run_dir, stage, reserve=0: 8000 if stage=='final' else 6000)
    with pytest.raises(TimeoutError, match='stop generation early'):
        s.fit_limit(tmp_path)


def test_claimed_admission_rejection_is_terminal_and_no_fit(tmp_path, monkeypatch):
    cfg = dict(id='synthetic')
    path = tmp_path/'attempts/tree__x.json'
    p.write_new(path, dict(family='tree',config=cfg,config_sha256='x'))
    monkeypatch.setattr(s, 'fit_limit', lambda *a: 1000)
    monkeypatch.setattr(p, 'claim', lambda *a: (path, p.read_json(path)))
    monkeypatch.setattr(p, 'verify_registration', lambda *a: {'sources': {}})
    monkeypatch.setattr(s, 'resource_estimate', lambda *a: (17*p.GIB, p.GIB, {}))
    monkeypatch.setattr(p, 'rss_bytes', lambda *a: p.GIB)
    result = s.run_attempt('tree', cfg, {}, None, tmp_path)
    assert result['status'] == 'admission_rejected'
    assert (tmp_path/'attempt-results/tree__x.json').exists()
    assert not (tmp_path/'models').exists()


def test_external_watchdog_terminates_only_owned_native_like_child(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'deadline', lambda *a: .1)
    log = tmp_path/'owned.log'
    with pytest.raises(subprocess.TimeoutExpired):
        s.run_owned([sys.executable,'-c','import time; print("started",flush=True); time.sleep(30)'],
                    log,tmp_path,'search')
    assert log.exists()


def test_external_watchdog_rejects_changed_birth(monkeypatch):
    child = subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'], start_new_session=True)
    birth=p.process_birth(child.pid)
    real=p.process_birth
    try:
        monkeypatch.setattr(p,'process_birth',lambda pid:'not-the-owned-birth')
        with pytest.raises(RuntimeError,match='identity changed'):
            s.stop_owned(child,birth)
        assert child.poll() is None
    finally:
        monkeypatch.setattr(p,'process_birth',real)
        s.stop_owned(child,birth)


def test_evaluator_receipt_loads_full_append_only_report(tmp_path):
    report_path=tmp_path/'acceptance-timestamp.json'
    p.write_new(report_path,dict(status='unmet',exit_code=1,objective_achieved=False,started_utc='a',finished_utc='b'))
    receipt_path=tmp_path/'receipt.json'
    p.write_new(receipt_path,dict(output=str(report_path),status='unmet',exit_code=1))
    report,ref=s.evaluator_receipt(receipt_path,1,tmp_path)
    assert report['objective_achieved'] is False
    assert ref==p.file_ref(report_path)
    with pytest.raises(ValueError,match='disagreement'):
        s.evaluator_receipt(receipt_path,0,tmp_path)


def test_fit_limit_keeps_ledger_termination_and_freeze_window(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'deadline',lambda run_dir,stage,reserve=0:5000 if stage=='search' else 9500)
    assert s.fit_limit(tmp_path)==4700
