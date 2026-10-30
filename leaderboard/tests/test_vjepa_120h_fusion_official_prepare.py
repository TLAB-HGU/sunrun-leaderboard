"""Official preparation reads only origins and safely combines immutable caches."""
import json

import numpy as np
import pandas as pd
import pytest

from experiments.vjepa_120h_fusion import official
from experiments.vjepa_120h_fusion.temporal import deduplicate_clips, plan_sources
from experiments.vjepa_frozen_fusion.encoder import sha256


def test_fold_origin_projection_never_reads_scoring_columns(tmp_path, monkeypatch):
    path = tmp_path/'folds.parquet'
    pd.DataFrame({'origin_last_input_utc': [official.CUTOFF, official.CUTOFF+pd.Timedelta(hours=1)],
                  'speed_kms': ['secret', 'secret'], 'regime': ['secret', 'secret']}).to_parquet(path)
    original = pd.read_parquet
    calls = []
    def read(*args, **kwargs):
        calls.append(kwargs.get('columns'))
        return original(*args, **kwargs)
    monkeypatch.setattr(pd, 'read_parquet', read)
    assert len(official.fold_origins(path)) == 2
    assert calls == [['origin_last_input_utc']]


def record(path, hour):
    slot = pd.Timestamp('2025-01-01T00:00Z')+pd.Timedelta(hours=hour)
    return {'raw_path': str(path.resolve()), 'source_slot': slot.isoformat(),
            'obs_end': (slot-pd.Timedelta(minutes=1)).isoformat(), 'key': str(hour),
            'raw_sha256': 'a', 'preprocessed_sha256': 'b'}


def test_two_reference_cache_lookup_uses_path_and_time_not_row_id(tmp_path):
    refs = [tmp_path/'historical', tmp_path/'official45']
    first, second, third = (record(tmp_path/f'raw{i}', i) for i in range(3))
    for ref, frames in zip(refs, ({'0': first, '1': second}, {'999': second, '0': third})):
        ref.mkdir()
        (ref/'frame-cache-manifest.json').write_text(json.dumps({'frames': frames}))
    index = official.reference_index(refs)
    assert len(index) == 3
    assert index[official.record_identity(second)][0] == refs[0]/'cache/frames'
    assert index[official.record_identity(third)][0] == refs[1]/'cache/frames'
    corrupted = dict(second, raw_sha256='changed')
    (refs[1]/'frame-cache-manifest.json').write_text(json.dumps({'frames': {'998': corrupted}}))
    with pytest.raises(ValueError, match='conflicting reference'):
        official.reference_index(refs)


def test_json_commit_recovery_and_symlink_protection(tmp_path):
    target = tmp_path/'old.npy'
    target.write_bytes(b'immutable')
    link = tmp_path/'new.npy'
    official.ensure_symlink(link, target)
    official.ensure_symlink(link, target)
    other = tmp_path/'other.npy'
    other.write_bytes(b'other')
    with pytest.raises(ValueError, match='target drift'):
        official.ensure_symlink(link, other)
    path = tmp_path/'protocol.json'
    official.ensure_json(path, {'identity': 'fixed'})
    official.ensure_json(path, {'identity': 'fixed'})
    with pytest.raises(ValueError, match='identity drift'):
        official.ensure_json(path, {'identity': 'changed'})
    assert target.read_bytes() == b'immutable'


def test_prepare_resumes_nonempty_run_and_recovers_protocol_commit(tmp_path, monkeypatch):
    root = tmp_path/'repo'
    root.mkdir()
    historical, old45, run = [tmp_path/name for name in ('history', 'old45', 'new')]
    for reference in (historical, old45):
        reference.mkdir()
        (reference/'frame-cache-manifest.json').write_text(json.dumps({'frames': {}}))
    (historical/'upstream').mkdir()
    (historical/'checkpoints').mkdir()
    (historical/'checkpoints'/official.old_official.CHECKPOINT).write_bytes(b'weights')
    (historical/'intensity.json').write_text('{}')
    (historical/'development_choice.json').write_text('{"arm":"pca32"}')
    (old45/'predictions.parquet').write_bytes(b'previous predictions')
    run.mkdir()
    (run/'handoff.md').write_text('preserve this checkpoint')
    fold = tmp_path/'folds.parquet'
    origin = official.CUTOFF+pd.Timedelta(hours=1)
    pd.DataFrame({'origin_last_input_utc': [origin], 'truth': ['not read']}).to_parquet(fold)
    monkeypatch.setattr(official, 'FOLDS_SHA', sha256(fold))
    script = root/'baseline.py'
    script.write_text('CONFIG = {}')
    config = {'hyperparameters': {'max_train_origins': 6000, 'seed': 42, 'xgboost': {'n_estimators': 100}},
              'data': {'input_sha256': {'ace': 'fixed'}, 'source_sha256': {}}}
    monkeypatch.setattr(official, 'baseline_config', lambda _: (config, script))
    monkeypatch.setattr(official, 'reference_hashes', lambda *args: {})
    frame = {'base': pd.DataFrame(index=pd.DatetimeIndex([origin])),
             'identities': {'ace': {'sha256': 'fixed'}}}
    monkeypatch.setattr(official, 'load_frames', lambda *args: frame)
    pool = pd.DatetimeIndex([official.CUTOFF-pd.Timedelta(hours=100)])
    monkeypatch.setattr(official.numeric, 'train_origins', lambda *args: pool)
    slots = pd.date_range(origin-pd.Timedelta(hours=220), periods=221, freq='h')
    inventory = pd.DataFrame({'slot': slots, 'obs_end': slots-pd.Timedelta(minutes=5),
                             'available': True, 'path': [str(tmp_path/f'raw{i}') for i in range(len(slots))]})
    monkeypatch.setattr(official.old_official, 'official_inventory', lambda *args: inventory)
    planned = pool.union(pd.DatetimeIndex([origin])).as_unit('ns').asi8
    _, ids, ends, ages, filled = plan_sources(inventory, planned)
    clips = deduplicate_clips(planned, ids, ends, ages, filled)
    count = len(np.unique(ids[ids >= 0]))
    monkeypatch.setattr(official, 'EXPECTED_COUNTS', {'origins': 2, 'train': 1, 'eval': 1,
        'unique_clips': len(clips['clip_ends']), 'required_frames': count, 'reused_frames': 0, 'new_frames': count})
    for name in ('encoder', 'images', 'numeric', 'official'):
        p = root/'experiments/vjepa_frozen_fusion'/f'{name}.py'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('# fixture')
    for name in ('protocol', 'tree'):
        p = root/'experiments/mse20_four_regimes'/f'{name}.py'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('# fixture')
    for relative in ('experiments/xgboost_suvi_fusion/features.py',
                     'store/ch-breakthrough-v2/upload/serialized/combined.parquet'):
        p = root/relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'fixture')
    result = official.prepare(fold, run, historical, old45, root)
    assert result['arm'] == 'pca32' and result['fixed_pca']['components'] == 32
    assert (run/'handoff.md').read_text() == 'preserve this checkpoint'
    assert pd.read_parquet(run/'fold-origins-only.parquet').columns.tolist() == ['origin_last_input_utc']
    old_hash = sha256(run/'planned_origins.npz')
    (run/'official_protocol.json').unlink()
    assert official.prepare(fold, run, historical, old45, root) == result
    assert sha256(run/'planned_origins.npz') == old_hash
    assert json.loads((run/'official_protocol.json').read_text()) == result
    (run/'intensity.json').write_text('{"corrupt":true}')
    with pytest.raises(ValueError, match='hash mismatch'):
        official.prepare(fold, run, historical, old45, root)


def test_gpu_and_merge_wrappers_always_use_absolute_run_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(official, 'verify_prepared', lambda path: calls.append(('verify', path)))
    monkeypatch.setattr(official.extraction, 'extract', lambda *args: calls.append(('extract', *args)))
    monkeypatch.setattr(official.extraction, 'merge', lambda path: calls.append(('merge', path)))
    official.extract('relative-run', 1, 4, True)
    official.merge('relative-run')
    assert all(call[1] == tmp_path/'relative-run' and call[1].is_absolute() for call in calls)
