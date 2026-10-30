"""Contract tests for the D90 parametric event-profile target (synthetic only).

No real data, no fits, no registration, no GPU, no official reads.
Every case exercises the actual ``profspec`` constants and ``runner`` pure
helpers with fully synthetic numerics.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PKG_PARENT = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(PKG_PARENT))

from d90_profile import profspec as SP  # noqa: E402
from d90_profile import runner  # noqa: E402


def test_dual_tag_and_axes_prefix():
    assert SP.FAMILY_TAG == "[family=xgb]"
    assert SP.AXES_TAG == "[axes: target=event-profile]"
    assert SP.HYPOTHESIS.startswith("[family=xgb] [axes: target=event-profile]")


def test_hypothesis_cites_axes_target_experience_and_refs():
    for token in ("E0001", "E0003", "X0144", "X0125", "research/AXES.md",
                  "arrival-time + amplitude parametrisation"):
        assert token in SP.HYPOTHESIS
    assert SP.RECON_FORMULA in SP.HYPOTHESIS


def test_spec_check_passes_and_budget_holds():
    assert SP.check_spec()["formula"] == SP.RECON_FORMULA
    assert SP.PERIOD_JOBS * SP.FIT_NJOBS <= 10
    assert (SP.FREQ_PILOT, SP.FREQ_FULL) == ("6h", "3h")
    assert SP.SCRATCH_NAME == "worker-d90"
    assert "gpu0" in SP.GPU_REASON and "60s" in SP.GPU_REASON


def test_reconstruct_peak_and_symmetry():
    h = np.arange(1, 73)
    p = runner.reconstruct(400.0, 150.0, 36.0, 12.0, h)
    assert p.shape == (72,)
    assert np.isfinite(p).all()
    assert p[35] == pytest.approx(550.0)
    assert p[23] == pytest.approx(p[47])  # symmetric around t0=36
    assert p[35] > p[0] and p[35] > p[-1]


def test_reconstruct_signed_dip():
    h = np.arange(1, 73)
    p = runner.reconstruct(500.0, -120.0, 10.0, 6.0, h)
    assert p[9] == pytest.approx(380.0)
    assert (p <= 500.0).all()


def test_extract_params_recovers_synthetic_bump():
    h = np.arange(1, 73)
    y = 400.0 + 150.0 * np.exp(-0.5 * ((h - 30.0) / 8.0) ** 2)
    got = runner.extract_params(y, h, 400.0)
    assert got is not None
    amp, t0, sigma, n_obs = got
    assert n_obs == 72
    assert amp == pytest.approx(150.0, abs=1.0)
    assert t0 == pytest.approx(30.0)
    assert sigma == pytest.approx(19.0, abs=2.0)  # half-prominence width of the bump


def test_extract_params_signed_dip_and_quiet_default():
    h = np.arange(1, 73)
    dip = 500.0 - 120.0 * np.exp(-0.5 * ((h - 10.0) / 6.0) ** 2)
    amp, t0, sigma, _ = runner.extract_params(dip, h, 500.0)
    assert amp == pytest.approx(-120.0, abs=1.0)
    assert t0 == pytest.approx(10.0)
    flat = np.full(72, 402.0)
    amp, _, sigma, _ = runner.extract_params(flat, h, 400.0)
    assert abs(amp) < SP.AMP_QUIET and sigma == SP.WIDTH_DEFAULT


def test_extract_params_rejects_sparse_window():
    h = np.arange(1, 73)
    y = np.full(72, np.nan)
    y[:10] = 400.0
    assert runner.extract_params(y, h, 400.0) is None


def test_clip_params_bounds_and_nan_guards():
    a, t, s = runner.clip_params(999.0, -5.0, 0.5)
    assert (a, t, s) == (SP.AMP_HI, SP.T0_LO, SP.SIG_LO)
    a, t, s = runner.clip_params(np.nan, np.nan, np.nan)
    assert (a, t, s) == (0.0, 36.0, SP.WIDTH_DEFAULT)


def test_base_fallback_chain():
    assert runner.base_from(450.0, 470.0) == 450.0
    assert runner.base_from(np.nan, 470.0) == 470.0
    assert runner.base_from(np.nan, np.nan) == SP.BASE_FALLBACK


def test_clip_roundtrip_predict_params_shape():
    class _Head:
        def __init__(self, v):
            self.v = v

        def predict(self, X):
            return np.full(len(X), self.v)

    heads = {"amp": _Head(500.0), "t0": _Head(200.0), "sigma": _Head(0.1)}
    X = np.zeros((4, 3))
    a, t, s = runner.predict_params(heads, X)
    assert a.shape == t.shape == s.shape == (4,)
    assert (a == SP.AMP_HI).all() and (t == SP.T0_HI).all() and (s == SP.SIG_LO).all()
