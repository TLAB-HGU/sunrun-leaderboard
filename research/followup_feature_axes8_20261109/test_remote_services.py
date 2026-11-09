"""No-network contract checks for frozen staging and collection safety."""
import io
import json
import subprocess
import tarfile
from unittest.mock import patch

import pytest

import remote_services as services


@pytest.fixture
def setup(tmp_path):
    for name in services.FILES:
        (tmp_path / name).write_text('frozen test source')
    (tmp_path / 'input').mkdir()
    for axis in range(1, 9):
        (tmp_path / f'input/axis{axis}.features.parquet').write_bytes(b'cache fixture')
    assignment = {'experiments': [dict(axis=axis, experiment=f'axis{axis}_s{seed}',
                                     config={'seed': seed, 'add_features': [f'feature{axis}']})
                                  for axis in range(1, 9) for seed in (42, 43, 44)],
                  'remote_observations': '/verified/observations.parquet'}
    return tmp_path, assignment


def test_payload_only_ready_axes_and_frozen_files(setup):
    root, assignment = setup
    (root / 'private.txt').write_text('not part of payload')
    data, receipt = services.payload(root, assignment, [1, 3])
    assert receipt['ready_axes'] == [1, 3]
    selected = receipt['assignment']
    assert len(selected['experiments']) == 6
    assert selected['observations'] == '/verified/observations.parquet'
    assert selected['max_jobs'] == 6
    assert all(r['threads'] == 4 and r['data'].startswith(services.REMOTE + '/input/') for r in selected['experiments'])
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        assert set(archive.getnames()) == set(services.FILES) | {
            'input/axis1.features.parquet', 'input/axis3.features.parquet', 'assignment.json', 'stage-request.json'}
    assert len(assignment['experiments']) == 24


def test_payload_rejects_missing_seed_and_unsafe_name(setup):
    root, assignment = setup
    with pytest.raises(ValueError, match='three locked seeds'):
        services.payload(root, dict(assignment, experiments=assignment['experiments'][1:]), [1])
    assignment['experiments'][0]['experiment'] = '../../escape'
    with pytest.raises(ValueError, match='Unsafe experiment'):
        services.payload(root, assignment, [1])


@pytest.mark.parametrize('member_name,kind', [('../escape', 'file'), ('/tmp/escape', 'file'),
                                             ('wrong/file', 'file'), ('safe/link', 'symlink')])
def test_collection_rejects_archive_traversal(tmp_path, member_name, kind):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        item = tarfile.TarInfo(member_name)
        if kind == 'symlink':
            item.type = tarfile.SYMTYPE
            item.linkname = '/etc/passwd'
        archive.addfile(item)
    with tarfile.open(fileobj=io.BytesIO(stream.getvalue()), mode='r:gz') as archive:
        with pytest.raises(ValueError, match='Unsafe remote artifact'):
            services._extract(archive, tmp_path, 'safe')
    assert not list(tmp_path.iterdir())


def test_stage_compiles_remote_code_and_never_launches(setup):
    root, assignment = setup
    def reply(code, **kwargs):
        compile(code, 'remote_stage', 'exec')
        assert kwargs['timeout'] == 1800
        assert 'CUDA_VISIBLE_DEVICES' in code
        assert 'sunrun-cpu-admission.lock' in code
        assert 'threads=1' in code
        assert "pa.set_cpu_count(1)" in code
        return subprocess.CompletedProcess([], 0, json.dumps({'root': services.REMOTE, 'resumed': False}).encode())
    with patch.object(services, 'remote', side_effect=reply) as remote:
        assert services.stage(root, assignment, [1, 2]) == services.REMOTE
        assert remote.call_count == 1
    assert (root / 'remote-prepared.json').is_file()


def test_existing_remote_destination_requires_matching_complete_receipt(setup, tmp_path):
    root, assignment = setup
    data, request = services.payload(root, assignment, [1])
    distant = tmp_path / 'remote'
    distant.mkdir()
    header = f'REMOTE={str(distant)!r}\nORIGINAL="unused"\nRPY="unused"\nPREFLIGHT_FILES=()\n'
    failed = subprocess.run(['python3', '-c', header + services._STAGE], input=data, capture_output=True)
    assert failed.returncode != 0
    assert b'no matching complete receipt' in failed.stderr
    (distant / 'frozen').write_text('unchanged')
    stat = (distant / 'frozen').stat()
    (distant / 'prepared-remote.json').write_text(json.dumps({
        'request': request, 'frozen_files': {'frozen': {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}}}))
    resumed = subprocess.run(['python3', '-c', header + services._STAGE], input=data, capture_output=True)
    assert resumed.returncode == 0, resumed.stderr.decode()
    assert json.loads(resumed.stdout)['resumed'] is True
    (distant / 'frozen').write_text('changed')
    changed = subprocess.run(['python3', '-c', header + services._STAGE], input=data, capture_output=True)
    assert changed.returncode != 0
    assert b'Frozen remote file changed' in changed.stderr


def test_remote_snapshot_checks_exact_process_argument():
    def reply(code, **kwargs):
        compile(code, 'snapshot', 'exec')
        assert 'split(bytes([0]))' in code
        return subprocess.CompletedProcess([], 0, b'{"state":{},"scheduler_alive":false}')
    with patch.object(services, 'remote', side_effect=reply):
        assert services.remote_snapshot() == {'state': {}, 'scheduler_alive': False}


def test_foreground_is_real_scheduler_and_records_exit(tmp_path):
    with patch.object(services.subprocess, 'run', return_value=subprocess.CompletedProcess([], 7)) as run:
        assert services.foreground_scheduler(tmp_path) == 7
    assert 'host_scheduler.py' in run.call_args.kwargs['input'].decode()
    assert 'CUDA_VISIBLE_DEVICES' in run.call_args.kwargs['input'].decode()
    assert (tmp_path / 'remote_scheduler_controller.exit').read_text().strip() == '7'
