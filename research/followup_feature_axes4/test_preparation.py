"""No-training checks of follow-up preparation and predecessor release gates."""
import io
import json
import tarfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import analyze_results
import build_data
import prepare_controller
import remote_services


def test_exact_append_preserves_targets_metadata_nulls_and_nan(tmp_path):
    original = pa.table({'origin': pa.array(pd.date_range('2025-01-01', periods=4, tz='UTC')),
                         'h': [1] * 4, 'y': pa.array([1., float('nan'), None, 4.]),
                         'existing': [1, 2, 3, 4], 'set': ['train'] * 4})
    cache = original.select(['origin', 'h']).append_column('new_feature', pa.array([5., 6., 7., 8.]))
    pq.write_table(original, tmp_path / 'source.parquet', row_group_size=2)
    pq.write_table(cache, tmp_path / 'cache.parquet', row_group_size=2)
    proof = build_data.augment(tmp_path / 'source.parquet', tmp_path / 'cache.parquet', tmp_path / 'out.parquet')
    assert proof['original_columns_exact']
    assert build_data.equal_original(original, pq.read_table(tmp_path / 'out.parquet', columns=original.column_names))


def test_replacement_and_misaligned_keys_are_rejected(tmp_path):
    table = pa.table({'origin': [1], 'h': [1], 'existing': [2.]})
    pq.write_table(table, tmp_path / 'source.parquet')
    pq.write_table(table, tmp_path / 'cache.parquet')
    with pytest.raises(ValueError, match='replacement'):
        build_data.augment(tmp_path / 'source.parquet', tmp_path / 'cache.parquet', tmp_path / 'out.parquet')
    pq.write_table(pa.table({'origin': [2], 'h': [1], 'new': [3.]}), tmp_path / 'cache.parquet')
    with pytest.raises(ValueError, match='keys differ'):
        build_data.augment(tmp_path / 'source.parquet', tmp_path / 'cache.parquet', tmp_path / 'out.parquet')


def test_ch_cache_is_label_free_and_preserves_groups(tmp_path):
    frame = pd.DataFrame({'origin': pd.date_range('2025-01-01', periods=60, freq='h', tz='UTC'),
                          'h': 1, 'suvi_t045_meridian30_equatorial_area': np.arange(60.), 'y': np.inf})
    frame.to_parquet(tmp_path / 'source.parquet', row_group_size=20)
    proof = build_data.build_cache(tmp_path / 'source.parquet', tmp_path / 'cache.parquet', 4)
    values = pq.read_table(tmp_path / 'cache.parquet').to_pandas()
    assert proof['target_columns_read'] is False
    assert proof['row_groups'] == 3
    assert values.iloc[-1]['suvi_t045_meridian30_equatorial_d24'] == 24
    assert values.iloc[-1]['suvi_t045_meridian30_equatorial_d48'] == 48
    assert 'y' not in values


def test_minute_cache_uses_origin_aligned_values(tmp_path):
    origins = pd.date_range('2025-01-01', periods=2, freq='h', tz='UTC')
    frame = pd.DataFrame({'origin': origins.repeat(2), 'h': [1, 2] * 2, 'y': np.inf})
    frame.to_parquet(tmp_path / 'source.parquet', row_group_size=2)
    minutes = pd.DataFrame({'S': 0, 'spd': 400.}, index=pd.date_range(origins[0] - pd.Timedelta(hours=1), origins[-1], freq='min'))
    with patch.object(build_data.minute_features, 'load_minutes', return_value=minutes), patch.object(
            build_data, 'validate_minute_sources', return_value={'cache_root': str(build_data.ROOT.parents[1] / 'data-v2/raw/ACE/daily-txt/swepam')}):
        proof = build_data.build_cache(tmp_path / 'source.parquet', tmp_path / 'cache.parquet', 1)
    assert proof['finite_counts']['ace_valid_fraction60'] == 4
    values = pq.read_table(tmp_path / 'cache.parquet').to_pandas()
    assert values.ace_valid_fraction60.tolist() == [1.] * 4
    assert values.ace_last_valid_age60_min.tolist() == [4.] * 4


def _previous(root):
    recipes = [dict(axis=a, experiment=f'a{a}s{s}') for a in range(1, 9) for s in (42, 43, 44)]
    (root / 'assignment.global.json').write_text(json.dumps({'experiments': recipes}))
    state = {r['experiment']: {'state': 'trained_validated' if r['axis'] < 8 else 'blocked_source'} for r in recipes}
    (root / 'runtime.json').write_text(json.dumps(state))
    value = dict(complete=True, analysis_exit=0, completed_axes=list(range(1, 8)), blocked_axes=[8], campaign_complete=False)
    (root / 'ready-axes-complete.json').write_text(json.dumps(value))
    return value, state


def test_predecessor_gate_does_not_wait_blocked_cme(tmp_path):
    assert not prepare_controller.previous_ready(tmp_path)
    value, state = _previous(tmp_path)
    assert prepare_controller.previous_ready(tmp_path)
    state['a1s42']['state'] = 'running'
    (tmp_path / 'runtime.json').write_text(json.dumps(state))
    assert not prepare_controller.previous_ready(tmp_path)
    _, state = _previous(tmp_path)
    value['analysis_exit'] = 1
    (tmp_path / 'ready-axes-complete.json').write_text(json.dumps(value))
    assert not prepare_controller.previous_ready(tmp_path)


def _assignment():
    return {'experiments': [dict(axis=a, seed=s, experiment=f'a{a}s{s}', config={'seed': s, 'add_features': [f'f{a}']})
                            for a in range(1, 5) for s in (42, 43, 44)],
            'remote_observations': '/verified/observations.parquet'}


def test_frozen_family_is_four_with_twelve_unique_runs():
    assignment = _assignment()
    assert len(analyze_results.conditions_from(assignment)) == 4
    rows = [dict(axis=a, state='waiting') for a in range(1, 5)]
    analyze_results.adjust_family(rows)
    assert all(row['bh_family_size'] == 4 and row['q_bh'] == 1 for row in rows)
    assignment['experiments'].pop()
    with pytest.raises(ValueError, match='Exactly12'):
        analyze_results.conditions_from(assignment)


def test_remote_payload_isolated_and_complete(tmp_path):
    for name in remote_services.FILES:
        (tmp_path / name).write_text('frozen source fixture')
    (tmp_path / 'input').mkdir()
    for axis in range(1, 5):
        (tmp_path / f'input/axis{axis}.features.parquet').write_bytes(b'cache fixture')
        (tmp_path / f'input/axis{axis}.features.json').write_text('{}')
    data, receipt = remote_services.payload(tmp_path, _assignment(), [1, 2, 3, 4])
    assert len(receipt['assignment']['experiments']) == 12
    assert 'followup-feature-axes4' in remote_services.REMOTE
    assert all(r['data'].startswith(remote_services.REMOTE + '/input/') for r in receipt['assignment']['experiments'])
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        assert 'minute_features.py' in archive.getnames()
        assert 'temporal_features.py' in archive.getnames()
        assert 'basic_features.py' not in archive.getnames()
    with pytest.raises(ValueError, match='1..4'):
        remote_services.payload(tmp_path, _assignment(), [8])


def test_minute_manifest_rejects_drift_and_missing_required_day(tmp_path):
    source = tmp_path / '20250101.txt'
    source.write_text('source fixture')
    stat = source.stat()
    manifest = dict(cache_root=str(tmp_path), required_start='2025-01-01', required_end='2025-01-01',
                    files={str(source): dict(path=str(source), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)})
    (tmp_path / 'ace-source-manifest.json').write_text(json.dumps(manifest))
    assert build_data.validate_minute_sources(tmp_path)['files']
    with pytest.raises(ValueError, match='missing'):
        build_data.validate_minute_sources(tmp_path, '2025-01-01', '2025-01-02')
    source.write_text('changed source fixture')
    with pytest.raises(ValueError, match='changed'):
        build_data.validate_minute_sources(tmp_path)
