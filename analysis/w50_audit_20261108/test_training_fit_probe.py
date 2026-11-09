import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def probe():
    spec = importlib.util.spec_from_file_location('training_probe_test', Path(__file__).with_name('training_fit_probe.py'))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_retained_loss_gradient_matches_actual_spectral_objective(probe):
    runner = probe.load_runner()
    _, core = runner.parent_modules()
    config = runner.fixed_config(core, 42, 'drop035_v650_s42')
    rng = np.random.default_rng(39)
    target = rng.uniform(300, 800, (9, 24))
    prediction = target + rng.normal(0, 80, target.shape)
    gradient, _ = core.make_objective(target, config)(prediction, None)
    epsilon = 1e-3
    for row, col in [(0, 0), (1, 3), (2, 17), (8, 23), (5, 11)]:
        plus, minus = prediction.copy(), prediction.copy()
        plus[row, col] += epsilon
        minus[row, col] -= epsilon
        numerical = (sum(x.sum() for x in probe.loss_arrays(plus, target))
                     - sum(x.sum() for x in probe.loss_arrays(minus, target))) / (2 * epsilon)
        assert numerical == pytest.approx(float(gradient[row, col]), rel=2e-6, abs=2e-5)
    point, spectral = probe.loss_arrays(target, target)
    assert not point.any()
    assert not spectral.any()


def test_original_training_mask_requires_exact_keys_and_observed_labels(probe):
    origins = pd.date_range('2024-01-01', periods=3, freq='h', tz='UTC')
    target = np.arange(216.).reshape(3, 72) + 300
    original = target.copy()
    original[0, 2] = original[2, 50] = np.nan
    rows = pd.DataFrame({'origin': np.repeat(origins, 72), 'h': np.tile(np.arange(1, 73), 3),
                         'y': original.ravel()})
    actual = probe.observed_mask(rows.sample(frac=1, random_state=1), origins, target, '2024-02-01')
    np.testing.assert_array_equal(actual, np.isfinite(original))
    changed = rows.copy()
    changed.loc[0, 'y'] += 1
    with pytest.raises(ValueError, match='labels changed'):
        probe.observed_mask(changed, origins, target, '2024-02-01')
    changed = rows.copy()
    changed.loc[0, 'origin'] += pd.Timedelta(hours=30)
    with pytest.raises(ValueError, match='keys differ'):
        probe.observed_mask(changed, origins, target, '2024-02-01')
    with pytest.raises(ValueError, match='missing or duplicate'):
        probe.observed_mask(pd.concat([rows.iloc[:-1], rows.iloc[[0]]]), origins, target, '2024-02-01')
    with pytest.raises(ValueError, match='144h cutoff'):
        probe.observed_mask(rows, origins, target, '2024-01-06')


def test_subset_point_loss_retains_full_population_tail_normalization(probe):
    runner = probe.load_runner()
    _, core = runner.parent_modules()
    config = runner.fixed_config(core, 42, 'drop035_v650_s42')
    target = np.full((4, 24), 400.)
    target[0] = 700.
    prediction = target + 10
    observed = np.ones_like(target, bool)
    observed[0, 0] = False
    report = probe.summarize(prediction, target, observed, {'full': np.arange(4), 'one_row': np.array([0])},
                             core.make_objective(target, config))
    expected = .5 * (4 / 1.75) * 10 ** 2
    assert report['one_row']['weighted_point_l2_per_pair'] == pytest.approx(expected)
    assert report['one_row']['label_groups']['imputed']['ge700']['pairs'] == 1
    assert report['one_row']['label_groups']['observed']['ge700']['pairs'] == 23
    assert report['one_row']['label_groups']['all']['lt400']['rmse'] is None
