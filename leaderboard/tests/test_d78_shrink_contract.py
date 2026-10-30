"""D78 shrinkage-decomposition contract tests (synthetic, CPU-only).

No real data, no fits, no registration, no network, no official reads.
Exercises the actual shrink.py contract plus runner hygiene (explicit grid
flag, no member prediction reads, scratch isolation to worker-d78,
cuda/thread budget, dev-only weights via the D71 fixed pre-rule).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PKG_PARENT = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(PKG_PARENT / "d78_shrink"))
sys.path.insert(0, str(PKG_PARENT / "d71_stack"))
sys.path.insert(0, str(PKG_PARENT / "headroom_audit"))
sys.path.insert(0, str(PKG_PARENT / "d33_hsplit"))

import shrink as SH  # noqa: E402
import stack as STK  # noqa: E402


def test_grids_members_seeds_split():
    assert tuple(SH.GRIDS) == ("6h", "3h")
    assert SH.check_grid("6h") == "6h"
    assert SH.check_grid("3h") == "3h"
    with pytest.raises(ValueError):
        SH.check_grid("1h")
    assert tuple(SH.MEMBERS) == tuple(STK.MEMBERS) == ("e0003_line", "d33_hsplit", "d45_lvlmean")
    assert tuple(SH.LEVEL_SEEDS) == (0, 1, 2)
    assert int(SH.EVENT_SEED) == 1
    assert SH.check_split()["split_h"] == 24 == SH.SPLIT_H
    assert SH.check_members(SH.MEMBERS) == SH.MEMBERS
    assert SH.check_seeds((0, 1, 2)) == (0, 1, 2)
    assert SH.check_event_seed(1) == 1
    with pytest.raises(ValueError):
        SH.check_members(("e0003_line", "d33_hsplit"))
    with pytest.raises(ValueError):
        SH.check_seeds((0, 1, 3))
    with pytest.raises(ValueError):
        SH.check_event_seed(0)


def test_fit_params_budget_and_cuda():
    p = SH.fit_params()
    assert p.get("device") == "cuda"
    assert p.get("tree_method") == "hist"
    assert int(p.get("n_jobs")) <= 4
    assert int(SH.FIT_JOBS) == 3
    assert int(p.get("n_jobs")) * int(SH.FIT_JOBS) <= 20
    assert SH.SCRATCH_NAME == "worker-d78"
    # own params match the D71 fixed rule (only grid differs)
    q = STK.fit_params()
    assert p == q


def test_fixed_weight_rule_matches_d71():
    mses = {"e0003_line": 8000.0, "d33_hsplit": 7840.0, "d45_lvlmean": 7705.8}
    w = STK.inverse_mse_weights(mses)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["d45_lvlmean"] > w["d33_hsplit"] > w["e0003_line"]
    out = STK.stack_predictions(
        {"e0003_line": np.array([1.0, 2.0]), "d33_hsplit": np.array([3.0, 4.0]),
         "d45_lvlmean": np.array([5.0, 6.0])}, w)
    assert out.shape == (2,) and np.all(np.isfinite(out))


def test_runner_hygiene():
    runner = (PKG_PARENT / "d78_shrink" / "runner.py").read_text()
    shrink_src = (PKG_PARENT / "d78_shrink" / "shrink.py").read_text()
    for src in (runner, shrink_src):
        for bad in ("truth.parquet", "folds.parquet", "naive.parquet",
                    "official-predictions", "baseline-replay.parquet",
                    "official_check", "official-check", "predictions_official_truth",
                    "datasets--tlabtlab--sunrun-lb-store"):
            assert bad not in src
    # no member prediction file reads
    for line in runner.splitlines():
        if "predictions_" in line:
            low = line.lower()
            assert not any(k in low for k in ("read_parquet", "read_csv", "open(", "load(")), line
    assert runner.count("read_parquet") <= 4
    # scratch isolation to worker-d78 only
    assert "worker-d78" in runner
    assert "store/scratch/worker-d78" in runner
    assert "worker-d71" not in runner or "d71_stack" in runner  # only as read-only import path
    assert 'endswith("worker-d78")' in runner
    # explicit grid flag decouples expos kind from density
    assert "--grid" in runner
    assert 'SH.check_grid(freq)' in runner or "check_grid" in runner
    # hash-keyed feat cache + 3-period parallel + cuda budget
    assert "fcache_key" in runner and "featcache" in runner
    assert "ThreadPoolExecutor" in runner
    assert "device" in runner and "cuda" in runner
    # weights come from dev only via the D71 fixed pre-rule
    assert 'infos["dev"]["dev_mses"]' in runner
    assert "STK.inverse_mse_weights" in runner
    assert "_dev_truth_for" in runner
    # selection/full rules: dev-only + poison + E0001 + grid decomp
    assert "dev" in SH.SELECT_RULE and "selection/audit" in SH.SELECT_RULE
    assert "poison" in SH.FULL_RULE and "E0001" in SH.FULL_RULE
    assert "--grid 6h" in SH.SELECT_RULE and "--grid 3h" in SH.SELECT_RULE
    assert "gpu0" in SH.GPU_PLAN and "worker-d78" in SH.SELECT_RULE
