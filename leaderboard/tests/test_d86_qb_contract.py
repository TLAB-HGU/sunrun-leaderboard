"""D86 quantile-blend contract tests (synthetic, CPU-only).

No real data, no fits, no registration, no network, no official reads.
Exercises the actual qblend.py pure helpers plus runner hygiene (no member
prediction reads, scratch isolation, cuda/thread budget, dev-only alpha).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PKG_PARENT = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(PKG_PARENT / "d86_qblend"))
sys.path.insert(0, str(PKG_PARENT / "headroom_audit"))

import qblend as QB  # noqa: E402


def test_members_and_constants():
    assert tuple(QB.MEMBERS) == ("mean", "q90")
    assert int(QB.MEAN_SEED) == 1
    assert int(QB.Q90_SEED) == 1
    assert float(QB.Q90_ALPHA) == 0.9
    assert tuple(QB.ALPHA_GRID) == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert QB.check_members(QB.MEMBERS) == QB.MEMBERS
    with pytest.raises(ValueError):
        QB.check_members(("mean",))
    with pytest.raises(ValueError):
        QB.check_alpha(0.33)
    assert QB.check_alpha(0.5) == 0.5


def test_fit_params_budget():
    p = QB.fit_params()
    assert p.get("device") == "cuda"
    assert p.get("tree_method") == "hist"
    assert int(p.get("n_jobs")) <= 4
    assert int(QB.FIT_JOBS) == 3
    assert int(p.get("n_jobs")) * int(QB.FIT_JOBS) <= 20
    assert QB.SCRATCH_NAME == "worker-d86"
    pm, pq = QB.mean_params(), QB.q90_params()
    assert int(pm.get("random_state")) == 1
    assert int(pq.get("random_state")) == 1
    assert pq.get("objective") == "reg:quantileerror"
    assert float(pq.get("quantile_alpha")) == 0.9
    assert "quantile" not in str(pm.get("objective", "")).lower()


def test_blend_peak_preserving():
    mean = np.array([100.0, 200.0, 300.0])
    q90 = np.array([110.0, 190.0, 300.0])
    # alpha 0 recovers the pure mean line
    assert np.allclose(QB.blend_predictions(mean, q90, 0.0), mean)
    # lifts only upward: row0 -> 105 at 0.5, row1 stays at mean, row2 unchanged
    out = QB.blend_predictions(mean, q90, 0.5)
    assert np.allclose(out, [105.0, 200.0, 300.0])
    # alpha 1 with q90 above mean recovers q90; below mean stays at mean
    out1 = QB.blend_predictions(mean, q90, 1.0)
    assert np.allclose(out1, [110.0, 200.0, 300.0])
    # quiet rows (q90 == mean) never move at any alpha
    for a in QB.ALPHA_GRID:
        assert np.allclose(QB.blend_predictions(mean, mean, a), mean)
    with pytest.raises(ValueError):
        QB.blend_predictions(mean, q90[:2], 0.5)
    with pytest.raises(ValueError):
        QB.blend_predictions(np.array([1.0, np.nan]), np.array([1.0, 2.0]), 0.5)
    with pytest.raises(ValueError):
        QB.blend_predictions(mean, q90, 0.33)


def test_select_alpha_dev_only():
    rng = np.random.default_rng(0)
    y = rng.normal(400, 50, 500)
    # q90 helps on peaks: best alpha > 0
    mean = y - 20.0
    q90 = y + 5.0
    alpha, mses = QB.select_alpha(mean, q90, y)
    assert alpha in QB.ALPHA_GRID and alpha > 0.0
    assert mses[alpha] == min(mses.values())
    # q90 hurts everywhere (below mean): lift never triggers, tie -> alpha 0
    alpha0, _ = QB.select_alpha(y + 10.0, y - 30.0, y)
    assert alpha0 == 0.0
    # NaN labels are excluded, still selects
    yn = y.copy()
    yn[:400] = np.nan
    alpha_n, _ = QB.select_alpha(mean, q90, yn)
    assert alpha_n in QB.ALPHA_GRID
    with pytest.raises(ValueError):
        QB.select_alpha(mean, q90[:10], y)
    with pytest.raises(ValueError):
        QB.select_alpha(mean[:50], q90[:50], y[:50])


def test_runner_hygiene():
    runner = (PKG_PARENT / "d86_qblend" / "runner.py").read_text()
    qb_src = (PKG_PARENT / "d86_qblend" / "qblend.py").read_text()
    for src in (runner, qb_src):
        for bad in ("truth.parquet", "folds.parquet", "naive.parquet",
                    "official-predictions", "baseline-replay.parquet",
                    "official_check", "official-check", "predictions_official_truth",
                    "datasets--tlabtlab--sunrun-lb-store"):
            assert bad not in src
    # no member prediction file reads: any line mentioning predictions_ must be
    # a write path (write() call, docstring contract), never a read
    for line in runner.splitlines():
        if "predictions_" in line:
            low = line.lower()
            assert not any(k in low for k in ("read_parquet", "read_csv", "open(", "load(")), line
    assert runner.count("read_parquet") <= 4
    assert "worker-d86" in runner
    assert "store/scratch/worker-d86" in runner
    # alpha comes from dev only
    assert 'infos["dev"]["pred_mean"]' in runner
    assert "_dev_truth_for" in runner
    # selection rule documents dev-only + pre-rule
    assert "dev" in QB.SELECT_RULE and "selection/audit" in QB.SELECT_RULE
    # full conditions are preregistered before the pilot
    assert "poison" in QB.FULL_RULE and "E0001" in QB.FULL_RULE
