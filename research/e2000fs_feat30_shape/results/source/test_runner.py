"""Integration contracts for the frozen experiment driver."""
import numpy as np
import pandas as pd
import pytest

import run


def test_design_and_feature_poison():
    r = np.random.default_rng(42)
    bundle = dict(C=r.normal(size=(4, 21)), E=r.normal(size=(4, 72, 9)))
    assert run.design(bundle, 0).shape == (4, 237)
    assert run.design(bundle, 0, 'difference').shape == (4, 444)
    assert run.design(bundle, 0, 'wavelet').shape == (4, 660)
    original = run.design(bundle, 0, 'wavelet')
    bundle['E'][:, 24:] += 10000
    np.testing.assert_array_equal(original, run.design(bundle, 0, 'wavelet'))
    bundle['E'][:, 5] += 10000
    assert not np.array_equal(original, run.design(bundle, 0, 'wavelet'))


def test_preserve_leading_missing_inputs():
    bundle = dict(C=np.zeros((1, 21)), E=np.ones((1, 72, 9)))
    bundle['C'][0, 0] = np.nan
    bundle['E'][0, 0, 0] = np.nan
    for arm in [False, 'difference', 'wavelet']:
        x = run.design(bundle, 0, arm)
        assert np.isnan(x[0, 0]) and np.isnan(x[0, 21])
        assert not np.isinf(x).any()


def test_mature_cutoffs_and_disjoint_selection():
    origins = pd.date_range('2025-10-01', '2026-05-28', freq='h', tz='UTC')
    data = {'train': {'origins': origins}}
    for stage, cutoff in [('select', '2025-12-01'), ('f1', '2026-02-26'), ('final', '2026-05-29')]:
        fit, valid = run.split(data, stage, False)
        assert (origins[fit] + pd.Timedelta(hours=72) < pd.Timestamp(cutoff, tz='UTC')).all()
        if stage == 'select':
            assert origins[fit].max() + pd.Timedelta(hours=72) < origins[valid].min()


def test_identity_refuses_changed_sources(tmp_path):
    run.ensure_identity(tmp_path, {'identity': 'a', 'smoke': True})
    run.ensure_identity(tmp_path, {'identity': 'a', 'smoke': True})
    with pytest.raises(RuntimeError, match='incompatible'):
        run.ensure_identity(tmp_path, {'identity': 'b', 'smoke': True})


def test_atomic_arrays_and_json(tmp_path):
    run.save_array(tmp_path / 'a.npz', indices=np.arange(5))
    np.testing.assert_array_equal(np.load(tmp_path / 'a.npz')['indices'], np.arange(5))
    run.write(tmp_path / 'a.json', {'weights': np.array([.5, .5]), 'n': np.int64(2)})
    import json
    assert json.loads((tmp_path / 'a.json').read_text()) == {'weights': [.5, .5], 'n': 2}
