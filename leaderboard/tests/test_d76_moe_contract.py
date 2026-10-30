"""D76 per-lead MoE contract tests (synthetic, CPU-only).

No real data, no fits, no registration, no network, no official reads.
Exercises the actual moe.py pure helpers plus runner hygiene (no member
prediction reads, scratch isolation, cuda/thread budget, dev-only weights).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PKG_PARENT = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(PKG_PARENT / "d76_moe"))
sys.path.insert(0, str(PKG_PARENT / "headroom_audit"))
sys.path.insert(0, str(PKG_PARENT / "d33_hsplit"))

import moe as MOE  # noqa: E402


def test_members_leads_and_split():
    assert tuple(MOE.MEMBERS) == ("e0003_line", "d33_hsplit", "d45_lvlmean")
    assert tuple(MOE.LEADS) == ("1-6", "7-24", "25-48", "49-72")
    assert tuple(MOE.LEVEL_SEEDS) == (0, 1, 2)
    assert int(MOE.EVENT_SEED) == 1
    assert MOE.check_split()["split_h"] == 24
    assert MOE.check_partition()["n_horizons"] == 72
    assert MOE.check_members(MOE.MEMBERS) == MOE.MEMBERS
    assert MOE.check_leads(MOE.LEADS) == MOE.LEADS
    assert MOE.lead_of(1) == "1-6"
    assert MOE.lead_of(24) == "7-24"
    assert MOE.lead_of(25) == "25-48"
    assert MOE.lead_of(72) == "49-72"
    with pytest.raises(ValueError):
        MOE.lead_of(0)
    with pytest.raises(ValueError):
        MOE.lead_of(73)
    with pytest.raises(ValueError):
        MOE.check_members(("e0003_line", "d33_hsplit"))
    with pytest.raises(ValueError):
        MOE.check_leads(("1-6", "7-24"))


def test_fit_params_budget():
    p = MOE.fit_params()
    assert p.get("device") == "cuda"
    assert p.get("tree_method") == "hist"
    assert int(p.get("n_jobs")) <= 4
    assert int(MOE.FIT_JOBS) == 3
    assert int(p.get("n_jobs")) * int(MOE.FIT_JOBS) <= 20
    assert MOE.SCRATCH_NAME == "worker-d76"


def _lead_mses(short=(8000.0, 7840.0, 7705.8), long_v=8000.0):
    return {lead: {"e0003_line": short[0] if lead in ("1-6", "7-24") else long_v,
                   "d33_hsplit": short[1] if lead in ("1-6", "7-24") else long_v,
                   "d45_lvlmean": short[2] if lead in ("1-6", "7-24") else long_v}
            for lead in MOE.LEADS}


def test_per_lead_weights_dev_only_precise():
    w = MOE.inverse_mse_weights_per_lead(_lead_mses())
    for lead in MOE.LEADS:
        assert abs(sum(w[lead].values()) - 1.0) < 1e-9
        assert all(v >= 0 for v in w[lead].values())
    # lower MSE -> higher weight on short leads
    assert w["1-6"]["d45_lvlmean"] > w["1-6"]["d33_hsplit"] > w["1-6"]["e0003_line"]
    assert w["7-24"]["d45_lvlmean"] > w["7-24"]["d33_hsplit"] > w["7-24"]["e0003_line"]
    # identical members on long leads -> near-uniform output
    assert all(abs(v - 1 / 3) < 1e-9 for v in w["25-48"].values())
    # uniform input -> near-uniform output everywhere
    wu = MOE.inverse_mse_weights_per_lead(
        {lead: {m: 100.0 for m in MOE.MEMBERS} for lead in MOE.LEADS})
    for lead in MOE.LEADS:
        assert all(abs(v - 1 / 3) < 1e-9 for v in wu[lead].values())
    # per-lead independence: changing one lead must not move another
    base = _lead_mses()
    other = {lead: dict(d) for lead, d in base.items()}
    other["1-6"] = {"e0003_line": 7000.0, "d33_hsplit": 7840.0, "d45_lvlmean": 7705.8}
    w2 = MOE.inverse_mse_weights_per_lead(other)
    assert w2["7-24"] == pytest.approx(w["7-24"])
    assert w2["1-6"] != pytest.approx(w["1-6"])
    # bad inputs abort
    with pytest.raises(ValueError):
        MOE.inverse_mse_weights_per_lead(
            {lead: {"e0003_line": 1.0, "d33_hsplit": 2.0} for lead in MOE.LEADS})
    with pytest.raises(ValueError):
        MOE.inverse_mse_weights_per_lead(
            {lead: {m: 0.0 for m in MOE.MEMBERS} for lead in MOE.LEADS})
    with pytest.raises(ValueError):
        MOE.inverse_mse_weights_per_lead(
            {"1-6": {m: 1.0 for m in MOE.MEMBERS}})


def test_blend_per_lead_convex():
    h = np.array([1, 6, 7, 24, 25, 48, 49, 72])
    a = np.arange(8, dtype=float)
    b = a + 100.0
    c = a + 200.0
    preds = {"e0003_line": a, "d33_hsplit": b, "d45_lvlmean": c}
    weights = {lead: {"e0003_line": 0.2, "d33_hsplit": 0.3, "d45_lvlmean": 0.5}
               for lead in MOE.LEADS}
    out = MOE.blend_per_lead(preds, weights, h)
    assert np.allclose(out, 0.2 * a + 0.3 * b + 0.5 * c)
    # per-lead weights actually route: lead 1-6 picks member a only
    routed = {lead: ({"e0003_line": 1.0, "d33_hsplit": 0.0, "d45_lvlmean": 0.0}
                     if lead == "1-6" else
                     {"e0003_line": 0.0, "d33_hsplit": 0.0, "d45_lvlmean": 1.0})
              for lead in MOE.LEADS}
    out2 = MOE.blend_per_lead(preds, routed, h)
    assert np.allclose(out2[:2], a[:2])
    assert np.allclose(out2[2:], c[2:])
    with pytest.raises(ValueError):
        MOE.blend_per_lead(
            preds,
            {lead: {"e0003_line": 0.5, "d33_hsplit": 0.5, "d45_lvlmean": 0.5}
             for lead in MOE.LEADS}, h)
    with pytest.raises(ValueError):
        MOE.blend_per_lead(
            {"e0003_line": a, "d33_hsplit": b[:7], "d45_lvlmean": c}, weights, h)
    with pytest.raises(ValueError):
        MOE.blend_per_lead(preds, weights, h[:7])


def test_runner_hygiene():
    runner = (PKG_PARENT / "d76_moe" / "runner.py").read_text()
    moe_src = (PKG_PARENT / "d76_moe" / "moe.py").read_text()
    for src in (runner, moe_src):
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
    assert "worker-d76" in runner
    assert "store/scratch/worker-d76" in runner
    # weights come from dev only
    assert 'infos["dev"]["dev_lead_mses"]' in runner
    assert "_dev_truth_for" in runner
    # selection rule documents dev-only + pre-rule
    assert "dev" in MOE.SELECT_RULE and "selection/audit" in MOE.SELECT_RULE
    # full conditions are preregistered before the pilot
    assert "poison" in MOE.FULL_RULE and "E0001" in MOE.FULL_RULE
