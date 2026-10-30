"""Official adapter uses fold inputs and the frozen train-only video recipe."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.vjepa_frozen_fusion import images, numeric, official
from test_vjepa_frozen_fusion_numeric import config, embeddings, frames


def test_fold_reader_allows_cutoff_origin_and_reads_only_input_key(tmp_path, monkeypatch):
    path = tmp_path / 'folds.parquet'
    pd.DataFrame({'origin_last_input_utc': [official.CUTOFF, official.CUTOFF + pd.Timedelta(hours=1)],
                  'regime': ['never read', 'never read']}).to_parquet(path)
    original = pd.read_parquet
    reads = []
    def read(*args, **kwargs):
        reads.append(kwargs.get('columns'))
        return original(*args, **kwargs)
    monkeypatch.setattr(pd, 'read_parquet', read)
    assert len(official.fold_origins(path)) == 2
    assert reads == [['origin_last_input_utc']]
    pd.DataFrame({'origin_last_input_utc': [official.CUTOFF - pd.Timedelta(hours=1)]}).to_parquet(path)
    with pytest.raises(ValueError, match='training cutoff'):
        official.fold_origins(path)


def make_inventory(tmp_path):
    metadata = tmp_path / 'metadata/year=2026'
    metadata.mkdir(parents=True)
    raw = tmp_path / 'images'
    slots = pd.DatetimeIndex(['2026-05-31T22:00Z', '2026-05-31T23:00Z', '2026-06-01T00:00Z'])
    records = []
    for i, slot in enumerate(slots):
        end = slot - pd.Timedelta(minutes=5)
        filename = f'image{i}.fits'
        path = raw / 'g18' / end.strftime('%Y/%j') / filename.replace('.fits', '.nc')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture')
        records.append(dict(slot=slot, obs_end=end, sat='G18', channel='Fe195', status='ok',
                            filename=filename, rsun=1., crpix1=2., crpix2=2., crota=0.))
    pd.DataFrame(records).to_parquet(metadata / 'part.parquet')
    return metadata.parent, raw, slots


def test_official_inventory_matches_historical_rules_and_extends_without_relaxing_guard(tmp_path):
    metadata, raw, slots = make_inventory(tmp_path)
    old = images.inventory_sources(metadata, raw, end=slots[1])
    matching = official.official_inventory(slots[1], metadata, raw)
    pd.testing.assert_frame_equal(old, matching)
    extended = official.official_inventory(slots[2], metadata, raw)
    assert len(extended) == 3 and extended.available.all()
    assert (extended.obs_end <= extended.slot).all()
    with pytest.raises(ValueError, match='before official'):
        images.inventory_sources(metadata, raw, end=slots[2])
    with pytest.raises(ValueError, match='forbidden'):
        numeric.load_frames(end=slots[2])


def test_inventory_rejects_future_observation_and_invalid_geometry_is_missing(tmp_path):
    metadata, raw, slots = make_inventory(tmp_path)
    path = metadata / 'year=2026/part.parquet'
    table = pd.read_parquet(path)
    table.loc[1, 'rsun'] = 0.
    table.to_parquet(path)
    out = official.official_inventory(slots[-1], metadata, raw)
    assert out.loc[1, 'invalid_geometry'] and not out.loc[1, 'available']
    table.loc[2, 'obs_end'] = slots[-1] + pd.Timedelta(seconds=1)
    table.to_parquet(path)
    with pytest.raises(ValueError, match='exceeds source slot'):
        official.official_inventory(slots[-1], metadata, raw)


def test_saved_video_parity_future_invariance_and_real_per_fold_benchmark(tmp_path):
    f = frames()
    index = f['speed'].index
    cutoff, origins = index[900], index[[980, 990, 1000]]
    path = tmp_path / 'embeddings.npz'
    embeddings(path, index)
    model = tmp_path / 'model'
    pred, meta = numeric.fit_predict(f, cutoff, origins, config(), 'cpu', model, 'video', path)
    verified = official.verify_saved(f, origins, pred, model, path)
    assert verified['saved_prediction_parity']
    assert len(verified['future_numeric_checks']) == 6
    assert pd.Timestamp(meta['train_label_max']) <= cutoff
    with np.load(model / 'visual_transform.npz') as transform:
        assert transform['train_origins'].max() + 72 * 3600 * 10**9 <= cutoff.value
    timing = official.benchmark_saved(f, origins, model, path, repeats=2)
    assert timing['inference_seconds_per_fold'] > 0
    assert len(timing['samples_seconds']) == 6 and timing['n_origins'] == 3
    assert 'prediction matrix construction' in timing['excludes']


def test_prepare_isolated_train_plan_and_immutable_frame_links(tmp_path, monkeypatch):
    reference = tmp_path / 'reference'
    reference.mkdir()
    (reference / 'comparison_protocol.json').write_text(json.dumps({'config': {
        'id': 'champion', 'max_train_origins': 6000, 'params': {}, 'seed': 42}}))
    (reference / 'intensity.json').write_text('{}')
    cache = reference / 'cache/frames'
    cache.mkdir(parents=True)
    (cache / 'a.npy').write_bytes(b'image')
    (cache / 'a.json').write_text('{}')
    metadata, raw, slots = make_inventory(tmp_path)
    origins = pd.DatetimeIndex([official.CUTOFF, official.CUTOFF + pd.Timedelta(hours=1)])
    fold_path = tmp_path / 'folds.parquet'
    pd.DataFrame({'origin_last_input_utc': origins}).to_parquet(fold_path)
    pool = pd.date_range('2025-01-01', periods=16, freq='h', tz='UTC')
    monkeypatch.setattr(official, 'reference_identity', lambda _: {'frozen': 'hash'})
    monkeypatch.setattr(official, 'load_frames', lambda *_: {'base': pd.DataFrame(index=origins), 'identities': {}})
    monkeypatch.setattr(numeric, 'train_origins', lambda *_: pool)
    run = reference / 'leaderboard-upload/official-run'
    result = official.prepare(fold_path, run, reference, metadata_root=metadata, images_root=raw)
    assert result['n_folds'] == 2 and not result['official_truth_access']
    assert (run / 'cache/frames/a.npy').is_symlink()
    assert not (run / 'cache/frames').is_symlink()
    assert not (run / 'cache/embeddings').exists()
    with np.load(run / 'planned_origins.npz') as planned:
        np.testing.assert_array_equal(planned['train'], pool.asi8)
        np.testing.assert_array_equal(planned['eval'], origins.asi8)
    with pytest.raises(FileExistsError):
        official.prepare(fold_path, run, reference)


def test_reference_identity_resolves_arbitrary_checkout_name(tmp_path, monkeypatch):
    checkout = tmp_path / 'publication'
    module = checkout / 'experiments/frozen.py'
    module.parent.mkdir(parents=True)
    module.write_text('FROZEN = True\n')
    reference = tmp_path / 'portable-reference'
    reference.mkdir()
    source = '/old/location/leaderboard/experiments/frozen.py'
    maps = ('protected-source-hashes.json', 'comparison_code_identity.json', 'execution-source-hashes.json')
    for name in maps:
        (reference / name).write_text(json.dumps({source: official.protocol.digest(module)}))
    (reference / 'comparison_protocol.json').write_text('{}')
    (reference / 'intensity.json').write_text(json.dumps({'train_cutoff': '2025-07-31T23:00:00Z'}))
    checkpoint = reference / 'checkpoints' / official.CHECKPOINT
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'weights fixture')
    monkeypatch.setattr(official.protocol, 'ROOT', checkout)
    monkeypatch.setattr(official, 'CHECKPOINT_SHA', official.protocol.digest(checkpoint))
    monkeypatch.setattr(official.subprocess, 'check_output',
                        lambda args, **kwargs: official.UPSTREAM_SHA if 'rev-parse' in args else '')
    identities = official.reference_identity(reference)
    assert identities['leaderboard/experiments/frozen.py'] == official.protocol.digest(module)
    module.write_text('CHANGED = True\n')
    with pytest.raises(ValueError, match='frozen source changed'):
        official.reference_identity(reference)


def test_predict_writes_complete_canonical_grid_without_scoring_truth(tmp_path, monkeypatch):
    f = frames()
    f['identities'] = {'synthetic': 'fixed'}
    index = f['speed'].index
    cutoff, origins = index[900], index[[980, 990]]
    monkeypatch.setattr(official, 'CUTOFF', cutoff)
    monkeypatch.setattr(official, 'load_frames', lambda *_: f)
    monkeypatch.setattr(official, 'reference_identity', lambda _: {})
    folds = tmp_path / 'folds.parquet'
    pd.DataFrame({'origin_last_input_utc': origins}).to_parquet(folds)
    pool = numeric.train_origins(f, cutoff, config()['max_train_origins'])
    planned = pool.union(origins).sort_values()
    np.savez(tmp_path / 'planned_origins.npz', origins=planned.asi8, train=pool.asi8, eval=origins.asi8)
    embeddings(tmp_path / 'embeddings.npz', planned)
    (tmp_path / 'intensity.json').write_text('{}')
    (tmp_path / 'inventory.parquet').write_bytes(b'inventory fixture')
    manifest = dict(folds=official.protocol.file_ref(folds), config=config(), reference_run=str(tmp_path),
                    reference_identities={}, input_identities=f['identities'])
    for filename, field in [('inventory.parquet', 'inventory_sha256'),
                            ('planned_origins.npz', 'planned_origins_sha256'), ('intensity.json', 'intensity_sha256')]:
        manifest[field] = official.protocol.digest(tmp_path / filename)
    official.protocol.write_json(tmp_path / 'official_protocol.json', manifest)
    official.protocol.write_json(tmp_path / 'extraction.json', {
        'embeddings_sha256': official.protocol.digest(tmp_path / 'embeddings.npz')})
    result = official.predict(folds, tmp_path)
    output = pd.read_parquet(tmp_path / 'predictions.parquet')
    assert len(output) == 144 and output.origin_last_input_utc.str.endswith('Z').all()
    assert result['n_folds'] == 2 and result['n_predictions'] == 144
    assert not result['official_truth_access'] and not result['selection_or_hpo']
    assert result['inference_seconds_per_fold'] > 0
    # Restoring the completed model cannot silently switch inputs or refit it.
    metadata = tmp_path / 'model/metadata.json'
    original = json.loads(metadata.read_text())
    original['embedding_sha256'] = 'changed'
    metadata.write_text(json.dumps(original))
    with pytest.raises(ValueError, match='input identity drift'):
        official.predict(folds, tmp_path)
