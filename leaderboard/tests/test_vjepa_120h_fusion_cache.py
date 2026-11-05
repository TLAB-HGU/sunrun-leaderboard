"""Immutable old-cache reuse and interrupted new-frame preparation regressions."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.vjepa_120h_fusion import cache


@pytest.fixture
def frame(tmp_path):
    raw = tmp_path/'raw.nc'
    raw.write_bytes(b'physical radiance test fixture')
    stat = raw.stat()
    row = SimpleNamespace(path=str(raw), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
        slot=pd.Timestamp('2025-01-01T00:00Z'), obs_end=pd.Timestamp('2024-12-31T23:59Z'))
    old, new = tmp_path/'old', tmp_path/'new'
    old.mkdir()
    new.mkdir()
    key = f'{raw.stem}_{row.size_bytes}_{row.mtime_ns}'
    np.save(old/(key+'.npy'), np.ones((384, 384), np.float32))
    record = {'key': key, 'raw_path': row.path,
        'bytes': row.size_bytes, 'mtime_ns': row.mtime_ns, 'source_slot': row.slot.isoformat(),
        'obs_end': row.obs_end.isoformat(),
        'geometry': {'valid_disk_fraction': 1.0}}
    return row, old, new, key, record


def prepare(frame, record):
    row, old, new, _, _ = frame
    return cache.prepare_frame(row, new, old, record, None, 'transform', 'code')


def test_reuse_is_verified_symlink_and_resumes_after_metadata_interruption(frame):
    _, old, new, key, record = frame
    original = (old/(key+'.npy')).read_bytes()
    assert prepare(frame, record) == record
    assert (new/(key+'.npy')).is_symlink()
    (new/(key+'.json')).unlink()
    assert prepare(frame, record) == record
    assert (new/(key+'.json')).is_file()
    assert (old/(key+'.npy')).read_bytes() == original


def test_reference_identity_mismatch_rejected(frame):
    record = dict(frame[-1], source_slot='2025-02-01T00:00:00+00:00')
    with pytest.raises(ValueError, match='timestamp identity'):
        prepare(frame, record)


def test_new_frame_recovers_orphan_and_then_reuses_local_checkpoint(frame, monkeypatch):
    _, old, new, key, _ = frame
    (new/(key+'.npy')).write_bytes(b'interrupted incomplete file')
    calls = []
    def preprocess(*args):
        calls.append(args)
        return np.ones((3, 384, 384), np.float32), None, {'valid_disk_fraction': 1.0}
    monkeypatch.setattr(cache, 'preprocess_frame', preprocess)
    record = prepare(frame, None)
    assert not (new/(key+'.npy')).is_symlink()
    assert prepare(frame, None) == record
    assert len(calls) == 1
    assert json.loads((new/(key+'.json')).read_text())['bytes'] == frame[-1]['bytes']


def test_foreign_symlink_is_not_overwritten(frame):
    _, old, new, key, record = frame
    other = old/'other.npy'
    np.save(other, np.zeros((384, 384), np.float32))
    (new/(key+'.npy')).symlink_to(other)
    with pytest.raises(ValueError, match='symlink target'):
        prepare(frame, record)
    with pytest.raises(ValueError, match='unrecognized frame symlink'):
        prepare(frame, None)
