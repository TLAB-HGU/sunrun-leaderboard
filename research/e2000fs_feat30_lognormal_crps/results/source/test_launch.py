"""Launcher contract tests: never start actual services or tmux sessions."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


spec = importlib.util.spec_from_file_location('distribution_launcher_tested', Path(__file__).with_name('launch.py'))
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class FakeRuntime:
    def __init__(self, failure=None):
        self.calls = []
        self.units = set()
        self.sessions = set()
        self.failure = failure

    def __call__(self, args, **kwargs):
        self.calls.append(args)
        assert kwargs == dict(capture_output=True, text=True)
        code, out, err = 0, '', ''
        if args[0] == 'systemctl':
            unit = args[3].removesuffix('.service')
            if unit in self.units:
                out = 'LoadState=loaded\nActiveState=active\nSubState=running\nMainPID=101\nExecMainStatus=0\nResult=success\n'
            else:
                out, code = 'LoadState=not-found\nActiveState=inactive\n', 1
        elif args[:2] == ['tmux', 'has-session']:
            code = 0 if args[-1].removeprefix('=') in self.sessions else 1
        elif args[0] == 'systemd-run':
            if self.failure == 'service':
                code, err = 1, 'synthetic service failure'
            else:
                self.units.add(next(a.split('=', 1)[1] for a in args if a.startswith('--unit=')))
        elif args[:2] == ['tmux', 'new-session']:
            if self.failure == 'tmux':
                code, err = 1, 'synthetic monitor failure'
            else:
                self.sessions.add(args[args.index('-s') + 1])
        else:
            pytest.fail(f'Unexpected command: {args}')
        return subprocess.CompletedProcess(args, code, out, err)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'HERE', tmp_path)
    directory = tmp_path / 'runs' / 'main_20261028'
    directory.mkdir(parents=True)
    (directory / 'manifest.json').write_text('{}')
    return directory


def test_all_lanes_have_durable_service_and_distinct_monitor(prepared, monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(launch.subprocess, 'run', runtime)
    result = launch.launch('main_20261028')
    assert result['status'] == 'started'
    assert len(runtime.units) == len(runtime.sessions) == 5
    services = [a for a in runtime.calls if a[0] == 'systemd-run']
    assert len(services) == 5
    for method, args in zip(launch.METHODS, services):
        assert args[-3:] == ['main_20261028', '--method', method]
        assert '--setenv=OMP_NUM_THREADS=2' in args
        assert '--property=StandardOutput=append:' + str(prepared / method / 'run.log') in args
        assert str(launch.PYTHON) in args
    assert all(e['returncode'] == e['tmux_returncode'] == 0 for e in result['lanes'])
    assert json.loads((prepared / 'launch.json').read_text()) == result
    before = len(runtime.calls)
    with pytest.raises(RuntimeError, match='Existing launch receipt'):
        launch.launch('main_20261028')
    assert len(runtime.calls) == before


@pytest.mark.parametrize('existing', ['service', 'session'])
def test_existing_resources_rejected_before_any_launch(prepared, monkeypatch, existing):
    runtime = FakeRuntime()
    unit, session = launch.names('main_20261028', 'smooth_wcrps')
    (runtime.units if existing == 'service' else runtime.sessions).add(unit if existing == 'service' else session)
    monkeypatch.setattr(launch.subprocess, 'run', runtime)
    with pytest.raises(RuntimeError, match='Existing'):
        launch.launch('main_20261028')
    assert not any(a[0] == 'systemd-run' for a in runtime.calls)
    assert not (prepared / 'launch.json').exists()


def test_failed_service_stops_once_and_records_foreground_fallback(prepared, monkeypatch):
    runtime = FakeRuntime('service')
    monkeypatch.setattr(launch.subprocess, 'run', runtime)
    with pytest.raises(RuntimeError, match='foreground command immediately'):
        launch.launch('main_20261028', smoke=True)
    receipt = json.loads((prepared / 'launch.json').read_text())
    assert receipt['status'] == 'launch_failed'
    assert receipt['lanes'][0]['returncode'] == 1
    assert receipt['lanes'][0]['foreground_command'][-1] == '--smoke'
    assert all(e['status'] == 'pending' for e in receipt['lanes'][1:])
    assert sum(a[0] == 'systemd-run' for a in runtime.calls) == 1
    assert not runtime.sessions


def test_failed_monitor_preserves_started_service(prepared, monkeypatch):
    runtime = FakeRuntime('tmux')
    monkeypatch.setattr(launch.subprocess, 'run', runtime)
    with pytest.raises(RuntimeError, match='Service started but tmux monitor failed'):
        launch.launch('main_20261028')
    assert len(runtime.units) == 1
    assert not any('stop' in a or 'kill-session' in a for a in runtime.calls)
    assert json.loads((prepared / 'launch.json').read_text())['lanes'][0]['tmux_returncode'] == 1


@pytest.mark.parametrize('run_id', ['../x', '.', 'a;b', 'a b', 'a$(id)', '/tmp/run', '', 'a.b'])
def test_invalid_identifiers_rejected(run_id):
    with pytest.raises(ValueError):
        launch.run_directory(run_id)


def test_status_is_read_only(prepared, monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(launch.subprocess, 'run', runtime)
    launch.launch('main_20261028')
    before = (prepared / 'launch.json').read_bytes()
    runtime.calls.clear()
    result = launch.status('main_20261028')
    assert len(result['lanes']) == 5
    assert all(e['tmux_exists'] for e in result['lanes'])
    assert not any(a[0] == 'systemd-run' or a[1] == 'new-session' for a in runtime.calls)
    assert (prepared / 'launch.json').read_bytes() == before


def test_missing_prepare_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'HERE', tmp_path)
    with pytest.raises(FileNotFoundError, match='Prepare the run first'):
        launch.launch('new_run')
