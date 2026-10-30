"""Regression checks for independent verification and diagnostic labeling."""
import json
import hashlib

import numpy as np
import pandas as pd
import pytest

from experiments.vjepa_120h_fusion import report, verify_run


def test_future_frames_preserve_past_and_input():
    index = pd.date_range('2025-01-01', periods=4, freq='h', tz='UTC')
    frames = {'base': pd.DataFrame({'x': np.arange(4.)}, index=index),
              'observed': pd.Series([True, False, True, False], index=index), 'identity': 'fixed'}
    for operation in ('poison', 'delete'):
        changed = verify_run.future_frames(frames, index[1], operation)
        pd.testing.assert_frame_equal(changed['base'].loc[:index[1]], frames['base'].loc[:index[1]])
        pd.testing.assert_series_equal(changed['observed'].loc[:index[1]], frames['observed'].loc[:index[1]])
        assert changed['identity'] == 'fixed'
        assert len(changed['base']) == (2 if operation == 'delete' else 4)
    assert frames['base'].iloc[-1, 0] == 3


def test_future_embedding_copy_changes_only_future(tmp_path):
    origins = pd.date_range('2025-01-01', periods=3, freq='h', tz='UTC')
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    metadata = {'quality': np.ones((3, 2)), 'source_end_ns': origins.asi8}
    path = verify_run.write_embedding_subset(tmp_path, origins, values, metadata, np.asarray([0, 2]), origins[0])
    actual = np.load(path)
    np.testing.assert_array_equal(actual[0], values[0])
    np.testing.assert_array_equal(actual[1], 900000.)
    np.testing.assert_array_equal(values[2], [8, 9, 10, 11])
    with np.load(path.with_name('embedding_metadata.npz')) as z:
        np.testing.assert_array_equal(z['origins'], origins.asi8[[0, 2]])
        np.testing.assert_array_equal(z['quality'][1], 900000.)


def test_reference_requires_hash_before_reading_metrics(tmp_path):
    path = tmp_path / 'report.json'
    path.write_text('{}')
    spec = {'reference_run': str(tmp_path), 'reference_identities': {
        'report.json': {'path': str(path), 'sha256': 'invalid'}}}
    with pytest.raises(ValueError, match='old reference changed'):
        report.reference_diagnostic(spec, {})


def test_protected_hash_change_fails_closed(tmp_path):
    protected = tmp_path / 'old.py'
    protected.write_text('changed')
    (tmp_path / 'protected-source-hashes.json').write_text(json.dumps({str(protected): 'old-hash'}))
    with pytest.raises(ValueError, match='protected hash changed'):
        verify_run.protected_identities(tmp_path)


def test_prediction_at_reads_canonical_string_origins():
    origin = pd.Timestamp('2025-01-01T00:00:00Z')
    saved = pd.DataFrame({'origin_last_input_utc': ['2025-01-01T00:00:00Z'] * 72,
                          'horizon_hours': np.arange(72, 0, -1), 'pred_kms': np.arange(72, 0, -1) + 300.})
    np.testing.assert_array_equal(verify_run.prediction_at(saved, origin), np.arange(1, 73) + 300.)
    with pytest.raises(AssertionError):
        verify_run.prediction_at(saved.iloc[:71], origin)


def test_extraction_identity_accepts_only_equivalent_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'input').write_text('data')
    (tmp_path / 'link').symlink_to(tmp_path / 'input')
    sources = {'input': 'filehash', 'plan_content_sha256': 'planhash'}
    identity = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    saved = {'identity': identity, 'sources': sources}
    current = {str(tmp_path / 'link'): 'filehash', 'plan_content_sha256': 'planhash'}
    assert verify_run.verify_extraction_identity(saved, current, identity) == identity
    for altered in ({str(tmp_path / 'link'): 'changedhash', 'plan_content_sha256': 'planhash'},
                    {str(tmp_path / 'other'): 'filehash', 'plan_content_sha256': 'planhash'},
                    {str(tmp_path / 'link'): 'filehash', 'plan_content_sha256': 'changedplan'}):
        with pytest.raises(ValueError, match='extraction code/input identity'):
            verify_run.verify_extraction_identity(saved, altered, identity)
    with pytest.raises(ValueError, match='stored extraction identity hash differs'):
        verify_run.verify_extraction_identity({'identity': 'wrong', 'sources': sources}, current, identity)
    with pytest.raises(ValueError, match='merged extraction identity differs'):
        verify_run.verify_extraction_identity(saved, current, 'wrong')
    with pytest.raises(ValueError, match='duplicate canonical'):
        verify_run.verify_extraction_identity(saved, {**current, 'input': 'filehash'}, identity)
