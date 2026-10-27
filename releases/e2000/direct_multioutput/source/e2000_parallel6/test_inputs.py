import json

import numpy as np
import pandas as pd
import pytest

from .inputs import (CUTOFF, HORIZONS, atomic_json, input_audit, prediction_frame,
                     reference, split_masks, target_matrix, widen)


def grid(origins):
    return pd.DataFrame({'origin': np.repeat(origins, 72), 'h': np.tile(HORIZONS, len(origins)),
                         'v_lag0': np.repeat(np.arange(len(origins)) + 400., 72),
                         'rec26': np.tile(np.arange(72) + 350., len(origins))})


def test_complete_features_independent_of_sparse_labels():
    origins = pd.date_range('2026-02-01', periods=2, freq='3h', tz='UTC')
    full = grid(origins)
    labels = full.iloc[[0, 74]].copy()
    labels['target_t'] = labels.origin + pd.to_timedelta(labels.h, unit='h')
    labels['y'] = [450., 500.]
    oo, x = widen(full.sample(frac=1, random_state=7), ['h', 'v_lag0', 'rec26'])
    y, mask = target_matrix(labels, oo)
    assert x.shape == (2, 73) and mask.sum() == 2
    assert y[0, 0] == 450 and y[1, 2] == 500
    assert x.iloc[0]['rec26_h72'] == 421
    assert x.notna().all().all()
    with pytest.raises(ValueError, match='complete feature grid'):
        widen(full.iloc[[0, 74]], ['h', 'v_lag0', 'rec26'])


def test_feature_order_nan_and_invariance():
    origins = pd.date_range('2026-02-01', periods=2, freq='3h', tz='UTC')
    full = grid(origins)
    full.loc[:71, 'v_lag0'] = np.nan
    oo, x = widen(full, ['v_lag0', 'rec26'])
    assert list(x)[:3] == ['v_lag0', 'rec26_h01', 'rec26_h02']
    assert np.isnan(x.iloc[0, 0])
    full.loc[5, 'v_lag0'] = 999
    with pytest.raises(ValueError, match='invariant'):
        widen(full, ['v_lag0', 'rec26'])


def test_cutoff_masks_and_purge():
    origins = pd.DatetimeIndex(['2026-02-01T00:00Z', '2026-02-27T00:00Z',
                                '2026-03-04T00:00Z', '2026-05-25T23:00Z'])
    tr, va = split_masks(origins, np.ones((4, 72), bool))
    assert tr[1, 46] and not tr[1, 47]
    assert va[2:].all() and not va[:2].any()
    assert not tr[2:].any()
    labels = grid(pd.DatetimeIndex(['2026-05-28T00:00Z']))
    labels['y'] = 400.
    labels['target_t'] = labels.origin + pd.to_timedelta(labels.h, unit='h')
    with pytest.raises(ValueError, match='official target'):
        target_matrix(labels, pd.DatetimeIndex(labels.origin.unique()))
    assert CUTOFF == pd.Timestamp('2026-05-29T00:00Z')


def test_reject_duplicate_and_infinite_inputs():
    full = grid(pd.date_range('2026-02-01', periods=1, tz='UTC'))
    with pytest.raises(ValueError, match='duplicate'):
        widen(pd.concat([full, full]), ['v_lag0'])
    full.loc[0, 'rec26'] = np.inf
    with pytest.raises(ValueError, match='infinite'):
        widen(full, ['rec26'])


def test_output_keeps_every_horizon():
    origins = pd.date_range('2026-05-31T23:00Z', '2026-09-26T23:00Z', freq='h')
    p = prediction_frame(origins, np.full((len(origins), 72), 400.))
    assert len(p) == 203976
    assert not p.duplicated(['origin_last_input_utc', 'horizon_hours']).any()
    assert p.iloc[-1].origin_last_input_utc == '2026-09-26T23:00:00Z'


def test_audit_does_not_trust_size_only_receipt(tmp_path):
    data = tmp_path / 'cache'
    data.write_bytes(b'local only')
    r = tmp_path / 'source.json'
    atomic_json(r, {'passed': True, 'causal': True, 'byte_bound': True,
                    **reference(data), 'features_covered': ['strip_v350_logmed'],
                    'limitations': ['file size comparison only']})
    report = input_audit({'artifacts': {}, 'source_receipts': [reference(r)]})
    assert not report['passed']
    assert any('exact local/remote content identity missing' in e for e in report['errors'])


def test_label_clock_and_mask_are_not_features():
    full = grid(pd.date_range('2026-02-01', periods=1, tz='UTC'))
    full['y'] = 400.
    with pytest.raises(ValueError, match='cannot be an input'):
        widen(full, ['y'])
    full['target_t'] = full.origin
    with pytest.raises(ValueError, match='clock'):
        target_matrix(full, pd.DatetimeIndex(full.origin.unique()))


def test_omitted_source_family_cannot_pass_audit(tmp_path):
    f = tmp_path / 'features.json'
    atomic_json(f, {'features': ['h', 'v_lag0', 'strip_v350_logmed']})
    r = tmp_path / 'source.json'
    cache = tmp_path / 'source-cache'
    cache.write_bytes(b'ACE')
    atomic_json(r, {'passed': True, 'causal': True, 'byte_bound': True,
                    **reference(cache), 'features_covered': ['v_*']})
    result = input_audit({'artifacts': {'feature_names': reference(f)},
                          'source_receipts': [reference(r)]})
    assert not result['passed']
    assert any('do not cover input features: strip_v350_logmed' in e for e in result['errors'])
