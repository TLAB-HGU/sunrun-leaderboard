"""Controller boundary tests without live model processes or new fits."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.mse20_hss_arrival import run


def test_phase_clocks_preserve_final_refit_reserve(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(run.p,'deadline',lambda *args:calls.append(args) or 1)
    assert run.remaining(tmp_path,'dev')==1
    assert calls[-1]==(tmp_path,'search',1800)
    run.remaining(tmp_path,'finalize')
    assert calls[-1]==(tmp_path,'search')
    (tmp_path/'final-fit.json').write_text('{}')
    run.remaining(tmp_path,'finalize')
    assert calls[-1]==(tmp_path,'final')
    run.remaining(tmp_path,'evaluate')
    assert calls[-1]==(tmp_path,'final')


def test_owned_process_stop_fails_closed_on_pid_reuse(monkeypatch):
    class Child:
        pid=123
        def poll(self):return None
    monkeypatch.setattr(run.p,'process_birth',lambda pid:999)
    monkeypatch.setattr(run.os,'killpg',lambda *args:pytest.fail('must not signal reused pid'))
    with pytest.raises(RuntimeError,match='identity changed'):run.stop_owned(Child(),100)


def test_exit_before_cleanup_never_signals(monkeypatch):
    class Child:
        pid=123
        def poll(self):return 0
    monkeypatch.setattr(run.os,'killpg',lambda *args:pytest.fail('must not signal exited process'))
    run.stop_owned(Child(),100)


def test_fixed_commands_no_retry_or_new_config(tmp_path):
    dev=run.command(tmp_path,'dev')
    final=run.command(tmp_path,'finalize')
    evaluate=run.command(tmp_path,'evaluate')
    assert dev[-2:]==['--stage','run'] and str(run.p.CLOCK) in dev
    assert final[-2:]==['--stage','all']
    assert evaluate[0].endswith('.venv-fusion/bin/python')
    with pytest.raises(ValueError):run.command(tmp_path,'hpo')
