import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('pointweight_run', Path(__file__).with_name('run.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def test_point_weight_does_not_spill_to_neighboring_low_targets():
    weights = r.point_weights(np.array([400., 549., 550., 700.]))
    np.testing.assert_allclose(weights / weights[0], [1., 1., 4., 4.])
    assert weights.mean() == 1.


def test_invalid_targets_rejected():
    with pytest.raises(ValueError):
        r.point_weights(np.array([np.nan]))


def test_acceptance_requires_both_regions():
    base = dict(mse_lt550=100., mse_ge550=500.)
    extreme = dict(mse_lt550=200., mse_ge550=300.)
    assert r.comparison(dict(mse_lt550=100., mse_ge550=300.), base, extreme)['satisfies_both']
    assert not r.comparison(dict(mse_lt550=99., mse_ge550=301.), base, extreme)['satisfies_both']
    assert not r.comparison(dict(mse_lt550=101., mse_ge550=299.), base, extreme)['satisfies_both']
