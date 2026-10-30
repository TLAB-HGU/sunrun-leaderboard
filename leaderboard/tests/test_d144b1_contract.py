"""D144-B1 contract: pure-helper + static gates for the physics-axis run.

No real data, no fits, no registration, no official reads. Covers the B1
transforms (values, monotonicity, NaN handling), the E0368-recipe constants
(CPU, jobs budget, cutoff, grid, depth, seeds, single small group, scratch
isolation, axes tag), the stitch/blend helpers, and the runner's static
contract (output names, poison gate, forbidden-source absence).
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

PKG_PARENT = Path(__file__).resolve().parents[1] / "experiments" / "d144_axisB_physics"
sys.path.insert(0, str(PKG_PARENT))

import features_b1 as FB  # noqa: E402
import lib_b1 as DL  # noqa: E402
import spec as SP  # noqa: E402

RUN_B1_SRC = (PKG_PARENT / "run_b1.py").read_text()


def test_slog_known_values():
    out = FB.slog(np.array([-1.0, 0.0, 1.0, np.e - 1]))
    assert out[1] == 0.0
    assert out[2] == pytest.approx(np.log(2.0))
    assert out[0] == pytest.approx(-np.log(2.0))
    assert out[3] == pytest.approx(1.0)


def test_slog_monotone_and_sign_symmetric():
    x = np.linspace(-50, 1e6, 2001)
    y = FB.slog(x)
    assert np.all(np.isfinite(y))
    assert np.all(np.diff(y) > 0)
    assert np.allclose(y, -FB.slog(-x))


def test_asinh_scaled_known_values():
    assert FB.asinh_scaled(np.array([0.0]), 5.0)[0] == 0.0
    assert FB.asinh_scaled(np.array([5.0]), 5.0)[0] == pytest.approx(np.arcsinh(1.0))
    y = FB.asinh_scaled(np.linspace(-1e6, 1e6, 2001), 5.0e4)
    assert np.all(np.isfinite(y)) and np.all(np.diff(y) > 0)


def test_asinh_rejects_nonpositive_scale():
    with pytest.raises(ValueError):
        FB.asinh_scaled(np.array([1.0]), 0.0)
    with pytest.raises(ValueError):
        FB.asinh_scaled(np.array([1.0]), -2.0)


def test_transforms_preserve_nan():
    x = np.array([np.nan, 1.0, np.nan])
    assert np.isnan(FB.slog(x)[0]) and np.isnan(FB.slog(x)[2])
    assert np.isnan(FB.asinh_scaled(x, 1.0)[0])


def test_transforms_finite_on_skewed_ranges():
    rng = np.random.default_rng(7)
    dens = rng.lognormal(1.6, 0.7, 5000)  # cm-3-like
    temp = rng.lognormal(10.8, 0.6, 5000)  # K-like
    epam = rng.normal(2.5, 1.5, 5000)  # log10-flux-like, signed
    for v, s in ((dens, 5.0), (temp, 5.0e4), (epam, 1.0)):
        assert np.all(np.isfinite(FB.slog(v)))
        assert np.all(np.isfinite(FB.asinh_scaled(v, s)))


def test_b1_group_is_single_and_small():
    assert len(SP.B1_BASES) == 10
    assert SP.N_B1_ADDED == 20
    assert len(FB.b1_columns()) == 20
    assert len(set(FB.b1_columns())) == 20
    assert len(SP.B1_ASINH_SCALES) == len(SP.B1_BASES)
    assert all(s > 0 for s in SP.B1_ASINH_SCALES)


def test_recipe_matches_e0368():
    assert SP.REFERENCE == "E0368"
    assert SP.DEPTH == 4 and SP.BASE_PARAMS["max_depth"] == 4
    assert SP.DEVICE == "cpu" and SP.BASE_PARAMS["device"] == "cpu"
    assert SP.FIT_NJOBS <= 20 and SP.BASE_PARAMS["n_jobs"] == SP.FIT_NJOBS
    assert SP.BASE_PARAMS["n_estimators"] == 600
    assert SP.CUTOFF_ISO < "2026-05-29T00:00:01Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24


def test_seeds_really_vary_the_fit():
    assert tuple(SP.SEEDS) == (0, 1, 2) and len(set(SP.SEEDS)) == 3
    blocks = [DL.level_seeds(s) for s in SP.SEEDS]
    assert len({seed for b in blocks for seed in b}) == 9
    events = [DL.event_seed(s) for s in SP.SEEDS]
    assert len(set(events)) == 3
    for s, ev in zip(SP.SEEDS, events):
        assert ev in DL.level_seeds(s)


def test_hypothesis_has_axes_tag_and_single_group_claim():
    assert re.search(r"\[axes:\s*[^\]]+\]", SP.HYPOTHESIS, re.I)
    tags = SP.HYPOTHESIS.lower()
    assert "feat=physics-derived" in tags
    assert "exactly one added feature group" in tags


def test_scratch_isolation():
    assert SP.SCRATCH == "store/scratch/worker-t1b1"
    assert SP.OWNER == "worker-t1b1" and SP.DIRECTION == "D144"


def test_blend_weights_convex_and_pinned():
    DL.check_weights(SP.WEIGHTS)
    h = np.arange(1, 73)
    rng = np.random.default_rng(0)
    mem = {m: rng.normal(400, 100, 72) for m in DL.MEMBERS}
    out = DL.blend_per_lead(mem, SP.WEIGHTS, h)
    assert out.shape == (72,) and np.all(np.isfinite(out))
    lo, hi = min(v.min() for v in mem.values()), max(v.max() for v in mem.values())
    assert np.all((out >= lo) & (out <= hi))


def test_frame_schema():
    import pandas as pd

    te = pd.DataFrame({"origin": pd.date_range("2026-06-01", periods=2, freq="h", tz="UTC").repeat(72),
                       "h": np.tile(np.arange(1, 73), 2)})
    fr = DL.frame_predictions(te, np.full(len(te), 400.0))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert len(fr) == 144 and (fr["horizon_hours"].to_numpy() == np.tile(np.arange(1, 73), 2)).all()


def test_runner_writes_seed_files_and_poison_gate():
    assert "predictions_official_s{s}.parquet" in RUN_B1_SRC
    assert '"poison.json"' in RUN_B1_SRC
    assert 'refusing to present predictions as causal' in RUN_B1_SRC
    assert "ST.poison_test" in RUN_B1_SRC
    assert "--gpu" not in RUN_B1_SRC and "device" in RUN_B1_SRC


def test_runner_touches_no_forbidden_source():
    for banned in ("truth.parquet", "folds.parquet", "naive.parquet",
                   "official-predictions", "baseline-replay.parquet",
                   "official_check", "official-check",
                   "predictions_official_truth", "sunrun-lb-store",
                   "GONG", "ADAPT", "DONKI", "CME", "WSA"):
        assert banned not in RUN_B1_SRC, banned
    for mod in ("d141", "e0147_abl", "nn_", "expos_runners"):
        assert f"import {mod}" not in RUN_B1_SRC, mod
    assert "d144" in RUN_B1_SRC or "D144" in RUN_B1_SRC
