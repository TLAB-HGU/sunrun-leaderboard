"""D71 stacking contract tests (synthetic, CPU-only).

No real data, no fits, no registration, no network, no official reads.
Exercises the actual stack.py pure helpers plus runner hygiene (no member
prediction reads, scratch isolation, cuda/thread budget, dev-only weights).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PKG_PARENT = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(PKG_PARENT / "d71_stack"))
sys.path.insert(0, str(PKG_PARENT / "headroom_audit"))
sys.path.insert(0, str(PKG_PARENT / "d33_hsplit"))

import stack as STK  # noqa: E402


def test_members_and_split():
    assert tuple(STK.MEMBERS) == ("e0003_line", "d33_hsplit", "d45_lvlmean")
    assert tuple(STK.LEVEL_SEEDS) == (0, 1, 2)
    assert int(STK.EVENT_SEED) == 1
    assert STK.check_split()["split_h"] == 24
    assert STK.check_members(STK.MEMBERS) == STK.MEMBERS
    with pytest.raises(ValueError):
        STK.check_members(("e0003_line", "d33_hsplit"))


def test_fit_params_budget():
    p = STK.fit_params()
    assert p.get("device") == "cuda"
    assert p.get("tree_method") == "hist"
    assert int(p.get("n_jobs")) <= 4
    assert int(STK.FIT_JOBS) == 3
    assert int(p.get("n_jobs")) * int(STK.FIT_JOBS) <= 20


def test_inverse_weights_dev_only_precise():
    mses = {"e0003_line": 8000.0, "d33_hsplit": 7840.0, "d45_lvlmean": 7705.8}
    w = STK.inverse_mse_weights(mses)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(v >= 0 for v in w.values())
    # lower MSE -> higher weight
    assert w["d45_lvlmean"] > w["d33_hsplit"] > w["e0003_line"]
    # uniform input -> near-uniform output
    wu = STK.inverse_mse_weights({m: 100.0 for m in STK.MEMBERS})
    assert all(abs(v - 1 / 3) < 1e-9 for v in wu.values())
    # bad inputs abort
    with pytest.raises(ValueError):
        STK.inverse_mse_weights({"e0003_line": 1.0, "d33_hsplit": 2.0})
    with pytest.raises(ValueError):
        STK.inverse_mse_weights({m: 0.0 for m in STK.MEMBERS})


def test_stack_predictions_convex():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    c = np.array([7.0, 8.0, 9.0])
    preds = {"e0003_line": a, "d33_hsplit": b, "d45_lvlmean": c}
    w = {"e0003_line": 0.2, "d33_hsplit": 0.3, "d45_lvlmean": 0.5}
    out = STK.stack_predictions(preds, w)
    assert np.allclose(out, 0.2 * a + 0.3 * b + 0.5 * c)
    with pytest.raises(ValueError):
        STK.stack_predictions(preds, {"e0003_line": 0.5, "d33_hsplit": 0.5, "d45_lvlmean": 0.5})
    with pytest.raises(ValueError):
        STK.stack_predictions(
            {"e0003_line": a, "d33_hsplit": b[:2], "d45_lvlmean": c}, w)


def test_runner_hygiene():
    runner = (PKG_PARENT / "d71_stack" / "runner.py").read_text()
    stack_src = (PKG_PARENT / "d71_stack" / "stack.py").read_text()
    for src in (runner, stack_src):
        for bad in ("truth.parquet", "folds.parquet", "naive.parquet",
                    "official-predictions", "baseline-replay.parquet",
                    "official_check", "official-check", "predictions_official_truth",
                    "datasets--tlabtlab--sunrun-lb-store"):
            assert bad not in src
    # no member prediction file reads: any line mentioning predictions_ must be
    # a write path (write() call, docstring contract, or predictions_ key), never a read
    for line in runner.splitlines():
        if "predictions_" in line:
            low = line.lower()
            assert not any(k in low for k in ("read_parquet", "read_csv", "open(", "load(")), line
    assert runner.count("read_parquet") <= 4
    assert "worker-d71" in runner
    assert "store/scratch/worker-d71" in runner
    # weights come from dev only
    assert 'infos["dev"]["dev_mses"]' in runner
    assert "_dev_truth_for" in runner
    # selection rule documents dev-only + pre-rule
    assert "dev" in STK.SELECT_RULE and "selection/audit" in STK.SELECT_RULE