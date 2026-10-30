import json
import os
from pathlib import Path

import pandas as pd
import pytest

from experiments.mse20_ch_dynamics import protocol as p


def fake_run(tmp_path, monkeypatch):
    stop = '2026-10-04T06:26:29Z'
    p.write_new(tmp_path/'runtime.json', {'search_stop_utc': stop, 'optimization_stop_utc': stop,
                                       'deadline_utc': '2026-10-04T07:41:29Z'})
    grid = {f: [dict(id=f'{f}{i}', value=i) for i in range(36)] for f in ('tree', 'neural')}
    registration = {'registered_utc': '2026-10-03T12:00:00Z', 'grid': grid, 'sources': {}}
    p.write_new(tmp_path/'registered.json', registration)
    monkeypatch.setattr(p, 'verify_registration', lambda run_dir, **kw: registration)
    monkeypatch.setattr(p, 'utcnow', lambda: '2026-10-03T13:00:00Z')
    return grid


def test_atomic_publish_exclusive_json(tmp_path):
    path = tmp_path/'nested/x.json'
    p.write_new(path, {'a': list(range(1000))})
    assert p.read_json(path)['a'] == list(range(1000))
    old = path.read_bytes()
    with pytest.raises(FileExistsError):
        p.write_new(path, {'bad': True})
    assert path.read_bytes() == old
    assert not list(path.parent.glob('*.tmp'))


def test_bad_json_never_publishes_partial(tmp_path):
    with pytest.raises(ValueError):
        p.write_new(tmp_path/'claim.json', {'bad': float('nan')})
    assert not (tmp_path/'claim.json').exists()


@pytest.mark.parametrize('stage,reserve,now,passes', [
    ('search', 900, '2026-10-04T06:11:28Z', True),
    ('search', 900, '2026-10-04T06:11:29Z', False),
    ('optimization', 0, '2026-10-04T06:26:29Z', False),
    ('final', 0, '2026-10-04T07:41:28Z', True),
    ('final', 0, '2026-10-04T07:41:29Z', False),
])
def test_deadline_boundaries(tmp_path, monkeypatch, stage, reserve, now, passes):
    fake_run(tmp_path, monkeypatch)
    if passes:
        assert p.deadline(tmp_path, stage, reserve, now) > reserve
    else:
        with pytest.raises(TimeoutError):
            p.deadline(tmp_path, stage, reserve, now)


def test_semantic_identity_does_not_reset_by_display_or_phase():
    config = dict(id='first', width=128, phase='phase1')
    assert p.semantic_digest('tree', config) == p.semantic_digest('tree', dict(config, id='second', phase='phase2'))
    assert p.semantic_digest('tree', config) != p.semantic_digest('neural', config)


def test_claim_order_uniqueness_and_count_failed(tmp_path, monkeypatch):
    grid = fake_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='next configuration'):
        p.claim('tree', grid['tree'][1], tmp_path)
    path, claim = p.claim('tree', grid['tree'][0], tmp_path)
    assert claim['owner_pid'] == os.getpid()
    assert claim['owner_birth'] == p.process_birth(os.getpid())
    p.terminal(path, 'admission_rejected', {'reason': 'budget'}, tmp_path)
    with pytest.raises(FileExistsError):
        p.claim('tree', grid['tree'][0], tmp_path)
    p.claim('tree', grid['tree'][1], tmp_path)
    assert len(p.verify_attempts(tmp_path)) == 2
    with pytest.raises(ValueError, match='nonterminal'):
        p.verify_attempts(tmp_path, require_terminal=True)


def test_all72_rejections_counted_and_restart_prefix(tmp_path, monkeypatch):
    grid = fake_run(tmp_path, monkeypatch)
    for family in ('tree', 'neural'):
        for config in grid[family]:
            path, _ = p.claim(family, config, tmp_path)
            p.terminal(path, 'failed', {'reason': 'controlled synthetic'}, tmp_path)
    assert len(p.verify_attempts(tmp_path, True)) == 72
    with pytest.raises(FileExistsError):
        p.claim('tree', grid['tree'][0], tmp_path)


def test_terminal_orphan_and_changed_registration(tmp_path, monkeypatch):
    grid = fake_run(tmp_path, monkeypatch)
    path, _ = p.claim('tree', grid['tree'][0], tmp_path)
    p.terminal(path, 'failed', {}, tmp_path)
    p.write_new(tmp_path/'attempt-results/orphan.json', {})
    with pytest.raises(ValueError, match='orphan'):
        p.verify_attempts(tmp_path, True)


def test_recover_only_confirmed_dead_birth(tmp_path, monkeypatch):
    grid = fake_run(tmp_path, monkeypatch)
    live, _ = p.claim('tree', grid['tree'][0], tmp_path)
    p.recover_dead_claims(tmp_path)
    assert not (tmp_path/'attempt-results'/live.name).exists()
    monkeypatch.setattr(p, 'process_birth', lambda pid: None)
    p.recover_dead_claims(tmp_path)
    assert p.read_json(tmp_path/'attempt-results'/live.name)['status'] == 'interrupted'
    p.recover_dead_claims(tmp_path)
    assert len(list((tmp_path/'attempt-results').glob('*.json'))) == 1


def test_joint_memory_reservation_and_release(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'rss_bytes', lambda pid=None: p.GIB)
    monkeypatch.setattr(p, 'available_bytes', lambda: 10*p.GIB)
    with p.memory_admission(4*p.GIB, tmp_path):
        assert len(p.read_json(tmp_path/'memory-reservations.json')) == 1
        with pytest.raises(RuntimeError, match='already holds'):
            with p.memory_admission(p.GIB, tmp_path):
                pass
    assert p.read_json(tmp_path/'memory-reservations.json') == {}
    with pytest.raises(MemoryError):
        with p.memory_admission(16*p.GIB, tmp_path):
            pass


def test_other_live_reservation_blocks_joint_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'rss_bytes', lambda pid=None: p.GIB)
    monkeypatch.setattr(p, 'available_bytes', lambda: 12*p.GIB)
    real_birth = p.process_birth
    monkeypatch.setattr(p, 'process_birth', lambda pid: 'other-birth' if pid == 123456 else real_birth(pid))
    p.write_new(tmp_path/'memory-reservations.json', {'other': {'pid':123456, 'birth':'other-birth', 'peak_bytes':8*p.GIB}})
    with pytest.raises(MemoryError, match='joint'):
        with p.memory_admission(3*p.GIB, tmp_path):
            pass


def test_environment_lock_is_live_not_saved_boolean(tmp_path, monkeypatch):
    p.write_new(tmp_path/'environment.json', {'model': {'version': 'a'}})
    monkeypatch.setattr(p, 'environment_snapshot', lambda: {'model': {'version': 'b'}})
    with pytest.raises(ValueError, match='versions changed'):
        p.verify_environment(tmp_path)


def test_torch_free_protocol_import_subprocess():
    import subprocess
    code = "import sys; from experiments.mse20_ch_dynamics import protocol; assert 'torch' not in sys.modules"
    subprocess.run([str(p.ROOT/'.venv-fusion/bin/python'), '-c', code], cwd=p.ROOT, check=True, timeout=30)


def test_gpu_deadline_checked_after_lease(tmp_path, monkeypatch):
    calls = []
    def deadline(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise TimeoutError('expired after lock')
    monkeypatch.setattr(p, 'deadline', deadline)
    with pytest.raises(TimeoutError, match='after lock'):
        with p.gpu_lease(tmp_path):
            pass
    assert len(calls) == 2


def test_unregistered_partition_denied_before_any_numeric_loading(tmp_path,monkeypatch):
    root=tmp_path/'repo'
    raw=root/'store/ch-breakthrough-v2/upload/raw-ace-snapshot/year=2026/part.parquet'
    raw.parent.mkdir(parents=True);raw.write_bytes(b'pinned')
    extra=raw.parent.parent/'year=9999/part.parquet'
    extra.parent.mkdir();extra.write_bytes(b'unregistered')
    monkeypatch.setattr(p,'ROOT',root)
    monkeypatch.setattr(p,'setup',lambda run_dir:({}, {'files':{'raw_year=2026':p.file_ref(raw)}},{},{}))
    monkeypatch.setattr(p.original,'load_frames',lambda end:pytest.fail('must not load unregistered data'))
    with pytest.raises(ValueError,match='partition inventory'):
        p.load_frames('2026-05-31T23:00:00Z',tmp_path)
