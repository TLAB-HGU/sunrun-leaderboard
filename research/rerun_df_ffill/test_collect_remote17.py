import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

import collect_remote17 as collector


def test_scheduler_does_not_launch_user_cancelled_jobs(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    import golab2_scheduler as scheduler
    names = ['e2000fs_calib5_detpool_all', 'e1604_hsspeak25']
    assignment = {'experiments': [{'experiment': name, 'family': 'pooling'} for name in names]}
    state = {name: {'state': 'cancelled_by_user'} for name in names}
    monkeypatch.setattr(scheduler, 'memory_gib', lambda: 30)
    monkeypatch.setattr(scheduler, 'launch', lambda *a: pytest.fail('Cancelled job must not launch'))
    scheduler.tick(tmp_path, assignment, datetime(2020, 1, 1, tzinfo=timezone.utc), state)
    assert all(e['state'] == 'cancelled_by_user' for e in state.values())


@pytest.mark.parametrize('via_manifest', [False, True])
def test_user_cancelled_jobs_are_not_collected_or_published(setup, monkeypatch, via_manifest):
    here, assignment, _ = setup
    names = [e['experiment'] for e in assignment['experiments']]
    if via_manifest:
        (here / 'manifest.json').write_text(json.dumps({'user_excluded_experiments': names}))
    monkeypatch.setattr(collector, 'read_remote_state', lambda _: {
        n: {'state': 'waiting' if via_manifest else 'cancelled_by_user'} for n in names})
    monkeypatch.setattr(collector, 'recover', lambda *a: pytest.fail('Cancelled run must not be collected'))
    monkeypatch.setattr(collector, 'publish_ready', lambda *a: pytest.fail('Cancelled run must not be published'))
    state = {}
    assert collector.tick(assignment, state) == 0
    assert all(v['phase'] == 'cancelled_by_user' for v in state.values())


@pytest.fixture
def setup(tmp_path, monkeypatch):
    here = tmp_path / 'research' / 'campaign'
    here.mkdir(parents=True)
    (here / 'runs').mkdir()
    (here / 'input').mkdir()
    prepared = here / 'input/df_ffill_trainy.parquet'
    prepared.write_bytes(b'prepared bytes')
    raw = tmp_path / 'data-v2/input/df.parquet'
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b'raw bytes')
    monkeypatch.setattr(collector, 'HERE', here)
    monkeypatch.setattr(collector, 'RECOVERY', here / 'recovery')
    monkeypatch.setattr(collector, 'STATE', here / 'collection.json')
    monkeypatch.setattr(collector.publish, 'HERE', here)
    assignment = {
        'remote_root': collector.REMOTE_ROOT,
        'experiments': [{'experiment': f'example{i}', 'family': 'advanced'} for i in range(17)],
        'input_identity': {**collector.identity(prepared), 'path': collector.REMOTE_ROOT + '/input/df_ffill_trainy.parquet'},
        'raw_identity': {**collector.identity(raw), 'path': collector.REMOTE_ROOT + '/raw/df.parquet'},
        'local_input_identity': collector.identity(prepared), 'local_raw_identity': collector.identity(raw),
    }
    original = here / 'original'
    original.mkdir()
    record = {'experiment': 'example0', 'family': 'advanced', 'host': 'golab2',
              'output': collector.REMOTE_ROOT + '/runs/example0',
              'input_identity': assignment['input_identity'],
              'command': ['python', 'experiment.py', '--data', assignment['input_identity']['path']]}
    (original / 'launch.json').write_text(json.dumps(record))
    (original / 'train.exit').write_text('0\n')
    (original / 'train.log').write_text('completed\n')
    for name in ('experiment.py', 'runner_at_launch.py'):
        (original / name).write_bytes(b'frozen source\n')
    (original / 'predictions.parquet').write_bytes(b'predictions')
    (original / 'model.ubj').write_bytes(b'model')
    (original / 'F1_metadata.json').write_text(json.dumps({'data': assignment['input_identity'], 'seed': 42}))
    return here, assignment, original


def fake_validator(name):
    out = collector.publish.HERE / 'runs' / name
    assert (out / 'predictions.parquet').read_bytes() == b'predictions'
    record = json.loads((out / 'launch.json').read_text())
    report = json.loads((out / 'F1_metadata.json').read_text())
    assert record['input_identity'] == collector.identity(collector.HERE / 'input/df_ffill_trainy.parquet')
    assert report['data'] == record['input_identity']
    assert report['seed'] == 42


def test_assignment_checks_exact_root_and_deployment_input(setup):
    here, assignment, _ = setup
    path = here / 'assignment.json'
    path.write_text(json.dumps(assignment))
    assert collector.load_assignment(path) == assignment
    (here / 'input/df_ffill_trainy.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Local input differs'):
        collector.load_assignment(path)


@pytest.mark.parametrize('placeholder', [False, True])
def test_install_preserves_original_and_exposes_only_validated_run(setup, monkeypatch, placeholder):
    here, assignment, original = setup
    before = (original / 'launch.json').read_bytes()
    out = here / 'runs/example0'
    if placeholder:
        out.mkdir()
        (out / 'remote-assignment.json').write_text(json.dumps({'experiment': 'example0', 'host': 'golab2', 'remote_root': collector.REMOTE_ROOT}))
    def validate(name):
        assert not (out / 'launch.json').exists()
        fake_validator(name)
    monkeypatch.setattr(collector.publish, 'validate_run', validate)
    collector.install('example0', original, assignment)
    assert (original / 'launch.json').read_bytes() == before
    for name in ('model.ubj', 'predictions.parquet', 'experiment.py', 'runner_at_launch.py'):
        assert (out / name).read_bytes() == (original / name).read_bytes()
    record = json.loads((out / 'launch.json').read_text())
    assert record['host'] == record['training_host'] == 'golab2'
    assert record['migration']['remote_root'] == collector.REMOTE_ROOT
    assert (collector.RECOVERY / 'example0.reservation').exists() == placeholder


def test_validation_failure_leaves_placeholder_and_original_untouched(setup, monkeypatch):
    here, assignment, original = setup
    out = here / 'runs/example0'
    out.mkdir()
    marker = {'experiment': 'example0', 'host': 'golab2', 'remote_root': collector.REMOTE_ROOT}
    (out / 'remote-assignment.json').write_text(json.dumps(marker))
    def invalid(name):
        raise ValueError('Missing full-run reports')
    monkeypatch.setattr(collector.publish, 'validate_run', invalid)
    with pytest.raises(ValueError, match='Missing full-run'):
        collector.install('example0', original, assignment)
    assert collector.is_placeholder(out, 'example0')
    assert not (out / 'launch.json').exists()
    assert (original / 'model.ubj').read_bytes() == b'model'


def test_existing_unrelated_execution_is_never_overwritten(setup):
    here, assignment, original = setup
    out = here / 'runs/example0'
    out.mkdir()
    value = {'experiment': 'example0', 'host': 'local'}
    (out / 'launch.json').write_text(json.dumps(value))
    with pytest.raises(ValueError, match='different execution'):
        collector.install('example0', original, assignment)
    assert json.loads((out / 'launch.json').read_text()) == value


def test_recovered_identity_mismatch_is_rejected(setup):
    _, assignment, original = setup
    record = json.loads((original / 'launch.json').read_text())
    record['input_identity']['mtime_ns'] += 1
    (original / 'launch.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='frozen assignment'):
        collector.install('example0', original, assignment)


def test_transfer_streams_to_file_and_rejects_path_traversal(setup, monkeypatch):
    _, assignment, _ = setup
    def run(command, **kwargs):
        assert kwargs['stdout'].fileno() >= 0
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        with tarfile.open(fileobj=kwargs['stdout'], mode='w:gz') as archive:
            member = tarfile.TarInfo('example0/../../escape')
            member.size = 1
            archive.addfile(member, io.BytesIO(b'x'))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(collector.subprocess, 'run', run)
    with pytest.raises(ValueError, match='archive member'):
        collector.recover('example0', assignment, {'state': 'trained_validated'})
    assert not (collector.RECOVERY / 'example0').exists()


def test_publish_receipt_wins_race_with_local_controller(setup, monkeypatch):
    here, assignment, original = setup
    monkeypatch.setattr(collector.publish, 'validate_run', fake_validator)
    out = collector.install('example0', original, assignment)
    def publish_elsewhere(command, **kwargs):
        (out / 'submission.json').write_text('{}')
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(collector.subprocess, 'run', publish_elsewhere)
    collector.publish_ready('example0')
    assert (out / 'submission.json').exists()


def test_remote_failures_preserved_and_all_terminal_returns_failure(setup, monkeypatch):
    _, assignment, original = setup
    assignment['experiments'] = [{'experiment': 'example0'}]
    monkeypatch.setattr(collector, 'read_remote_state', lambda a: {'example0': {'state': 'failed', 'reason': 'Training exited 1'}})
    monkeypatch.setattr(collector, 'recover', lambda *args: original)
    monkeypatch.setattr(collector, 'publish_ready', lambda n: pytest.fail('Failed training must never publish'))
    state = {}
    assert collector.tick(assignment, state) == 1
    assert state['example0']['phase'] == 'remote_failed'
    assert state['example0']['original'] == str(original)


def test_persistent_transfer_failure_has_bounded_retry(setup, monkeypatch):
    _, assignment, _ = setup
    assignment['experiments'] = [{'experiment': 'example0'}]
    monkeypatch.setattr(collector, 'read_remote_state', lambda a: {'example0': {'state': 'failed', 'reason': 'Launch failed before creating run'}})
    def unavailable(*args):
        raise RuntimeError('Remote run directory absent')
    monkeypatch.setattr(collector, 'recover', unavailable)
    state = {'example0': {'attempts': collector.MAX_ATTEMPTS - 1}}
    assert collector.tick(assignment, state) == 1
    assert state['example0']['phase'] == 'collection_failed'
