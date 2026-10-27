from contextlib import contextmanager
from types import SimpleNamespace
import pandas as pd
import pytest
from . import availability_rebuild as ar


def test_ch_waits_for_exact_bytes(tmp_path):
    image = tmp_path / 'image.nc'
    image.write_bytes(b'fixture')
    meta = pd.DataFrame({'slot': pd.to_datetime(['2026-01-01T00:00Z']), 'status': ['ok'],
                         'obs_end': pd.to_datetime(['2025-12-31T23:59Z']),
                         's3_modified': pd.to_datetime(['2026-01-01T02:00Z']), 'nc_path': [str(image)]})
    assert ar.ch_metadata(meta).available_at.iloc[0] == pd.Timestamp('2026-01-01T02:00Z')


def test_ch_excludes_missing_and_empty_local_bodies(tmp_path):
    image = tmp_path / 'image.nc'
    image.write_bytes(b'fixture')
    empty = tmp_path / 'empty.nc'
    empty.touch()
    slots = pd.date_range('2026-01-01', periods=4, freq='h', tz='UTC')
    meta = pd.DataFrame({'slot': slots, 'status': ['ok']*4, 'obs_end': slots,
                         's3_modified': slots,
                         'nc_path': [str(image), None, str(empty), str(tmp_path/'missing.nc')]})
    result = ar.ch_metadata(meta)
    assert len(result) == 1 and result.slot.iloc[0] == slots[0]


def test_late_object_is_deferred_and_mismatched_bytes_rejected(monkeypatch):
    local = {'local_md5': 'a'*32, 'local_sha256': 'b'*64, 'local_size': 12}
    monkeypatch.setattr(ar, 'digests', lambda _: local)
    monkeypatch.setattr(ar.prepare, 'relocated_nc', lambda x: x)
    receipt = local | {'local_path': '/image.nc', 'remote_key': 'image.nc',
                       'earliest_origin': '2026-01-01T01:00Z', 'obs_end': '2025-12-31T23:59Z',
                       'head_headers': {'etag': 'a'*32, 'content-length': '12',
                                        'last-modified': 'Thu, 01 Jan 2026 02:00:00 GMT'}}
    meta = pd.DataFrame({'slot': pd.to_datetime(['2026-01-01T00:00Z']), 'status': ['ok'],
                         'obs_end': pd.to_datetime(['2025-12-31T23:59Z']),
                         's3_modified': pd.to_datetime(['2026-01-01T00:01Z']),
                         'nc_path': ['/image.nc'], 's3_key': ['image.fits']}, index=[7])
    result = ar.corrected_metadata(meta, [receipt])
    assert result.s3_modified.iloc[0] == pd.Timestamp('2026-01-01T02:00Z')
    assert result.original_slot_idx.iloc[0] == 7
    receipt['head_headers']['etag'] = 'c'*32
    with pytest.raises(ValueError, match='unverified object bytes'):
        ar.corrected_metadata(meta, [receipt])


def test_context_aligns_cache_and_restores(tmp_path, monkeypatch):
    slots = pd.date_range('2026-01-01', periods=3, freq='h', tz='UTC')
    meta = pd.DataFrame({'slot': slots[[0, 2]], 'status': ['ok', 'ok'],
                         'obs_end': slots[[0, 2]], 's3_modified': slots[[0, 2]]})
    path = tmp_path / 'meta.parquet'
    meta.to_parquet(path)
    original = lambda: (meta, pd.DataFrame({'value': [10, 20, 30]}, index=slots))
    old_ch = lambda: 'old'
    p = SimpleNamespace(ST=SimpleNamespace(load_inputs=original), L9=SimpleNamespace(load_meta=old_ch))
    @contextmanager
    def fake(_):
        yield p
    monkeypatch.setattr(ar.prepare, 'relocated_builder', fake)
    monkeypatch.setattr(ar.prepare, 'source_root', lambda: tmp_path)
    with pytest.raises(RuntimeError):
        with ar.causal_builder(path) as actual:
            corrected, block = actual.ST.load_inputs()
            assert block.value.tolist() == [10, 30]
            assert block.index.equals(pd.DatetimeIndex(corrected.slot))
            raise RuntimeError()
    assert p.ST.load_inputs is original and p.L9.load_meta is old_ch


def test_grid_loss_rejected():
    frame = pd.DataFrame({'origin': pd.to_datetime(['2026-01-01T00:00Z']*2), 'h': [1, 2], 'x': [1., 2.]})
    with pytest.raises(ValueError, match='changed output grid'):
        ar.changed_features(frame, frame.iloc[:1], ['h', 'x'])
    altered = frame.copy()
    altered.loc[1, 'x'] = float('nan')
    result = ar.changed_features(frame, altered, ['h', 'x'])
    assert result['changed_rows'] == 1 and result['by_feature'] == {'x': 1}
    storage_variant = frame.copy()
    storage_variant['h'] = storage_variant.h.astype('int16')
    assert ar.changed_features(frame, storage_variant, ['h', 'x'])['changed_rows'] == 0


def test_provisional_retains_unknown_candidates_for_closure(tmp_path, monkeypatch):
    slots = pd.date_range('2026-01-01', periods=2, freq='h', tz='UTC')
    meta = pd.DataFrame({'slot': slots, 'status': ['ok', 'ok'], 'obs_end': slots,
                         's3_modified': slots, 's3_key': ['one.fits', 'two.fits']})
    p = SimpleNamespace(ST=SimpleNamespace(load_inputs=lambda: (meta, None)))
    @contextmanager
    def fake(_):
        yield p
    monkeypatch.setattr(ar.prepare, 'relocated_builder', fake)
    monkeypatch.setattr(ar.prepare, 'source_root', lambda: tmp_path)
    result = ar.provisional_metadata(tmp_path)
    assert len(result) == 2 and not result.head_verified.any()
    assert (result.s3_modified == slots + pd.Timedelta(hours=1)).all()
