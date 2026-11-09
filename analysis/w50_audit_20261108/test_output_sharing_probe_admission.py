"""Synthetic resource/lifecycle checks; never inspect or start real jobs."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def admission(tmp_path, monkeypatch):
    path = Path(__file__).with_name('output_sharing_probe_admission_controller.py')
    spec = importlib.util.spec_from_file_location('sharing_admission_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'PROBE', tmp_path)
    monkeypatch.setattr(module, 'OUT', tmp_path / 'run01')
    monkeypatch.setattr(module, 'ADMISSION_LOCK', tmp_path / 'global.lock')
    monkeypatch.setattr(module.support, 'memory_gib', lambda: 20.)
    monkeypatch.setattr(module.support, 'busy_cpu_cores', lambda: 18.)
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **kw: pytest.fail('Real job started'))
    return module


def test_once_checks_full_global_budget_without_launch(admission, monkeypatch, capsys):
    monkeypatch.setattr(admission.host, 'workloads', lambda: [{'pid': i, 'threads': 4} for i in range(6)])
    assert admission.main(['--once']) == 0
    record = json.loads(capsys.readouterr().out)
    assert record['eligible'] is False and record['declared_threads'] == 24
    assert not admission.OUT.exists()


def test_once_eligible_still_never_launches(admission, monkeypatch, capsys):
    monkeypatch.setattr(admission.host, 'workloads', lambda: [{'pid': i, 'threads': 4} for i in range(5)])
    assert admission.main(['--once']) == 0
    assert json.loads(capsys.readouterr().out)['eligible'] is True
    assert not admission.OUT.exists()


def test_existing_launch_refuses_automatic_restart(admission, monkeypatch):
    admission.OUT.mkdir()
    (admission.OUT / 'launch.json').write_text('{}')
    monkeypatch.setattr(admission.host, 'workloads', lambda: pytest.fail('Admission checked for an existing launch'))
    with pytest.raises(ValueError, match='never restarted automatically'):
        admission.main([])
