import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def probe():
    spec = importlib.util.spec_from_file_location('temporal_support_test', Path(__file__).with_name('temporal_support_probe.py'))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_exact_common_support_decomposition_and_permutation(probe):
    codes = np.repeat(np.r_[np.zeros(45), np.ones(45), np.zeros(60), np.ones(30)][:, None], 2, axis=1).astype(int)
    older, recent = np.arange(90), np.arange(90, 180)
    error = np.where(codes == 0, 2., 8.)
    error[recent] = np.where(codes[recent] == 0, 5., 10.)
    observed = np.ones_like(codes, bool)
    # An extra low-support stratum cannot enter the decomposition.
    codes[0, 0] = codes[90, 0] = 2
    result = probe.decomposition(error, observed, codes, older, recent)
    supported = codes != 2
    older_bias = error[older][supported[older]].mean()
    recent_bias = error[recent][supported[recent]].mean()
    parts = result['decomposition']
    assert parts['older_bias'] == pytest.approx(older_bias)
    assert parts['recent_bias'] == pytest.approx(recent_bias)
    assert parts['within'] + parts['composition'] == pytest.approx(recent_bias - older_bias)
    assert result['common_strata'] == 2
    assert result['strata'][2]['common_support'] is False
    assert result['coverage']['older']['pair_fraction'] == pytest.approx(179 / 180)
    permutation = np.random.default_rng(61).permutation(180)
    permuted = probe.decomposition(error[permutation], observed[permutation], codes[permutation],
                                   np.flatnonzero(permutation < 90), np.flatnonzero(permutation >= 90))
    assert permuted == result
    # Repeated leads cannot turn one origin into 30 independent-origin support.
    tiny = probe.decomposition(np.ones((2, 72)), np.ones((2, 72), bool), np.zeros((2, 72), int), [0], [1])
    assert tiny['decomposition'] is None
    assert tiny['strata'][0]['older']['origins'] == 1


def test_fixed_recency_tail_hessian_mass_and_ess_arithmetic(probe):
    origins = pd.date_range('2020-01-01', periods=4, freq='180D', tz='UTC')
    target = np.array([[800., 400.], [600., 800.], [400., 400.], [400., 600.]])
    observed = np.ones_like(target, bool)
    observed[1, 1] = False
    result = probe.weight_audit(origins, target, observed)
    raw = .1 + .9 * np.array([.125, .25, .5, 1.])
    r = raw / raw.mean()
    point = np.where(target >= 550, 4., 1.)
    point /= point.mean(axis=0)
    assert result['raw']['min'] == pytest.approx(raw.min())
    assert result['normalized']['max'] == pytest.approx(r.max())
    assert result['row_ess'] == pytest.approx(r.sum() ** 2 / (r ** 2).sum())
    full = result['classes']['all']
    np.testing.assert_allclose(full['baseline_point_mass_per_lead'], [4, 4])
    np.testing.assert_allclose(full['recency_point_mass_per_lead'], (r[:, None] * point).sum(axis=0))
    np.testing.assert_allclose(full['recency_hessian_mass_per_lead'], (r[:, None] * (point + .1)).sum(axis=0))
    assert full['flattened_pair_ess_recency_not_independent'] == pytest.approx(2 * result['row_ess'])
    rare = result['classes']['ge700']
    assert rare['origin_count_any_qualifying_lead'] == 2
    assert rare['row_ess_recency_any_qualifying_lead'] == pytest.approx(probe.ess(r[:2]))
    assert result['classes']['imputed']['recency_point_mass_per_lead'][1] == pytest.approx(r[1] * point[1, 1])
    for key in ('baseline_point_mass_per_lead', 'recency_point_mass_per_lead',
                'baseline_hessian_mass_per_lead', 'recency_hessian_mass_per_lead'):
        np.testing.assert_allclose(np.array(result['classes']['observed'][key]) + result['classes']['imputed'][key], full[key])
    permutation = [3, 1, 0, 2]
    permuted = probe.weight_audit(origins[permutation], target[permutation], observed[permutation])
    np.testing.assert_allclose(permuted['classes']['all']['recency_point_mass_per_lead'], full['recency_point_mass_per_lead'])


def test_raw_input_key_alignment_and_fixed_bin_boundaries(probe):
    origins = pd.date_range('2020-01-01', periods=5, freq='h', tz='UTC')
    rows = pd.DataFrame({'origin': origins, 'h': 1, 'v_lag0': [399., 400., 550., 700., np.nan],
                         'img_age_h': [1., np.nan, 1., 1., 1.]})
    aligned = probe.align_raw_inputs(rows.sample(frac=1, random_state=2), origins)
    np.testing.assert_array_equal(probe.speed_bin(aligned.v_lag0), [0, 1, 2, 3, 4])
    for invalid in [rows.iloc[:-1], pd.concat([rows.iloc[:-1], rows.iloc[[0]]]), rows.assign(h=2)]:
        with pytest.raises(ValueError, match='Raw input keys'):
            probe.align_raw_inputs(invalid, origins)
    codes = probe.strata_codes(aligned.v_lag0.to_numpy(), np.full((5, 24), 400.),
                                np.array([0., 1., np.nan, 0., 0.]), aligned.v_lag0.isna().to_numpy(),
                                aligned.img_age_h.isna().to_numpy())
    assert codes['speed_recurrence'][0, 0] == 1
    assert codes['plus_ch'][2, 0] == (2 * 5 + 1) * 3 + 2
    assert codes['plus_raw_missingness'][4, 0] == ((4 * 5 + 1) * 3 * 2 + 1) * 2
