"""D122 independent-seed-replicate contract: synthetic CPU-only checks, no data reads.

Covers the assignment acceptance surface without running the pipeline:
fresh-refit two-series discipline (no stored predictions), fresh rotation
seeds 10/11/12 disjoint from 0/1/2 with preregistered primary 10, frozen
hard switch h=24, preregistered dev-OOF grid sweep (9 points 0.05..0.85,
F1-floor-passing best-mean, blended = base + w*resid) per seed, w
distribution record, full-density key discipline, file budget, forbidden-
source hygiene, parallel/cache discipline, scratch isolation and the expos
output contract keys (both poison legs).
"""
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

PKG_PARENT = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(PKG_PARENT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "headroom_audit"))

import info_ablation as ia  # noqa: E402
from d122_seedrep import srspec as SPEC  # noqa: E402
from expos import core  # noqa: E402

D122 = Path(__file__).resolve().parents[1] / "experiments" / "d122_seedrep"
RUNNER_SRC = (D122 / "runner.py").read_text()
SPEC_SRC = (D122 / "srspec.py").read_text()

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay",
             "official_check", "official-check", "predictions_official_truth"]
COPY_BANS = ["def speed_features(", "def ace_features(", "def ch_features(",
             "def enlil_features(", "def build(", "def arms(", "def full_grid(",
             "def write("]
LATER_DATES = ("2025-12-04", "2026-02-25", "2026-03-04", "2026-05-28",
               "2026-05-29", "2026-05-31", "2026-09-26")


def test_dual_tag_and_citations():
    assert SPEC.FAMILY_TAG == "[family=nn]"
    assert SPEC.AXES_TAG.startswith("[axes:") and "val=" in SPEC.AXES_TAG
    assert "seed-replicates" in SPEC.AXES_TAG
    assert SPEC.HYPOTHESIS.startswith("[family=nn] [axes: val=seed-replicates]")
    for token in ("E0003", "E0102", "E0001", "E0193", "X0097", "research/AXES.md"):
        assert token in SPEC.HYPOTHESIS, f"hypothesis must cite {token}"
    assert "weights from dev-OOF only" in SPEC.HYPOTHESIS
    assert "no stored predictions" in SPEC.HYPOTHESIS
    assert "6h" in SPEC.HYPOTHESIS and "gpu0" in SPEC.HYPOTHESIS
    assert "screening" in SPEC.HYPOTHESIS
    assert SPEC.check_spec()["members"] == ["e0003-series", "e0102-series"]
    assert SPEC.check_spec()["seeds"] == [10, 11, 12]
    assert SPEC.check_spec()["experience"] == ["X0096", "X0097"]


def test_members_seeds_grid_and_floor():
    assert SPEC.MEMBERS == ("e0003-series", "e0102-series")
    assert not hasattr(SPEC, "EIDS"), "no stored experiment ids: members are refit, not loaded"
    assert SPEC.SEEDS == (10, 11, 12) and SPEC.PRIMARY_SEED == 10
    assert not (set(SPEC.SEEDS) & set(SPEC.PRIOR_SEEDS)), "rotation disjoint from measured 0/1/2"
    assert SPEC.SPLIT_H == 24
    assert SPEC.check_seeds() == SPEC.SEEDS
    with pytest.raises(ValueError):
        SPEC.check_seeds((0, 1, 2))
    with pytest.raises(ValueError):
        SPEC.check_seeds((10, 11))
    with pytest.raises(ValueError):
        SPEC.check_seeds((10, 11, 12, 13))
    assert SPEC.W_GRID == (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)
    assert (SPEC.W_MIN, SPEC.W_MAX) == (0.05, 0.90)
    assert SPEC.FLOOR_DEV_F1 == 0.55
    assert SPEC.check_grid() == SPEC.W_GRID
    assert SPEC.check_members(SPEC.MEMBERS) == SPEC.MEMBERS
    with pytest.raises(ValueError):
        SPEC.check_members(("e0003-series",))
    with pytest.raises(ValueError):
        SPEC.check_members(("e0003", "e0102"))
    w = SPEC.weights_for_w(0.35)
    assert abs(w["e0102-series"] - 0.35) < 1e-12 and abs(w["e0003-series"] - 0.65) < 1e-12
    assert abs(sum(w.values()) - 1.0) < 1e-12
    with pytest.raises(ValueError):
        SPEC.weights_for_w(0.0)
    with pytest.raises(ValueError):
        SPEC.weights_for_w(0.95)


def test_blend_predictions_residual_form():
    base = np.array([400.0, 400.0, 400.0])
    lift = np.array([500.0, 400.0, 300.0])
    w = {"e0003-series": 0.65, "e0102-series": 0.35}
    out = SPEC.blend_predictions({"e0003-series": base, "e0102-series": lift}, w)
    # blended = base + 0.35*(lift-base): bounded peak lift
    assert np.allclose(out, base + 0.35 * (lift - base))
    assert np.allclose(out, np.array([435.0, 400.0, 365.0]))
    with pytest.raises(ValueError):
        SPEC.blend_predictions({"e0003-series": base, "e0102-series": lift[:2]}, w)
    with pytest.raises(ValueError):
        SPEC.blend_predictions({"e0003-series": np.array([np.inf, 1.0, 2.0]),
                                "e0102-series": lift}, w)
    with pytest.raises(ValueError):
        SPEC.blend_predictions({"e0003-series": base, "e0102-series": lift},
                               {"e0003-series": 0.5, "e0102-series": 0.6})
    with pytest.raises(ValueError):
        SPEC.blend_predictions({"e0003-series": base, "e0102-series": lift,
                                "e0102": lift}, w)


def test_select_weight_floor_passing_best_mean():
    # floor 0.55: passing {0.05:7820/0.565, 0.15:7780/0.596, 0.25:7746/0.565, 0.35:7719/0.652}
    # best mean among passing is w=0.35 (lowest MSE)
    sweep = [
        {"w": 0.05, "dev_mse": 7820.0, "dev_f1": 0.565},
        {"w": 0.15, "dev_mse": 7780.0, "dev_f1": 0.596},
        {"w": 0.25, "dev_mse": 7746.0, "dev_f1": 0.565},
        {"w": 0.35, "dev_mse": 7719.0, "dev_f1": 0.652},
        {"w": 0.45, "dev_mse": 7699.0, "dev_f1": 0.533},
        {"w": 0.55, "dev_mse": 7685.0, "dev_f1": 0.444},
        {"w": 0.65, "dev_mse": 7678.0, "dev_f1": 0.500},
        {"w": 0.75, "dev_mse": 7677.0, "dev_f1": 0.489},
        {"w": 0.85, "dev_mse": 7684.0, "dev_f1": 0.478},
    ]
    sel = SPEC.select_weight(sweep)
    assert sel["w"] == 0.35 and sel["passed_floor"] is True
    assert abs(sel["weights"]["e0102-series"] - 0.35) < 1e-12
    # tie on MSE picks smallest w
    tied = [dict(r) for r in sweep]
    tied[0] = {"w": 0.05, "dev_mse": 7719.0, "dev_f1": 0.60}
    sel2 = SPEC.select_weight(tied)
    assert sel2["w"] == 0.05
    # floor miss picks lowest MSE with passed_floor=false
    miss = [{"w": g, "dev_mse": 7700.0 - g, "dev_f1": 0.30} for g in SPEC.W_GRID]
    sel3 = SPEC.select_weight(miss)
    assert sel3["passed_floor"] is False
    # grid order/coverage enforced (dev-only rule shape)
    with pytest.raises(ValueError):
        SPEC.select_weight(sweep[:8])
    with pytest.raises(ValueError):
        SPEC.select_weight([{"w": g, "dev_mse": 1.0, "dev_f1": 2.0} for g in SPEC.W_GRID])
    with pytest.raises(ValueError):
        SPEC.select_weight([{"w": 0.99, "dev_mse": 1.0, "dev_f1": 0.6} for _ in SPEC.W_GRID])


def test_no_forbidden_sources_or_copies_and_no_stored_predictions():
    for name, src in (("srspec.py", SPEC_SRC), ("runner.py", RUNNER_SRC)):
        for tok in FORBIDDEN:
            assert tok not in src, (name, tok)
        for s in LATER_DATES:
            assert s not in src, (name, f"later window {s}")
    for ban in COPY_BANS:
        assert ban not in RUNNER_SRC, f"runner must import, not copy: {ban}"
    assert core.scan(SPEC_SRC + "\n" + RUNNER_SRC + "\n" + SPEC.HYPOTHESIS) == []
    # D122 must never reuse saved predictions: no run-dir reads, no member loader
    assert "research/expos/runs" not in RUNNER_SRC
    assert "expos/runs" not in RUNNER_SRC
    assert "_load_member" not in RUNNER_SRC
    assert "_member_path" not in RUNNER_SRC
    assert "no stored predictions" in SPEC.SELECT_RULE


def test_parallel_cache_and_budget():
    assert "ThreadPoolExecutor" in RUNNER_SRC
    assert "fcache_key" in RUNNER_SRC and "featcache" in RUNNER_SRC
    assert "from existing_all import" in RUNNER_SRC and "write(" in RUNNER_SRC
    assert "SPEC.blend_predictions" in RUNNER_SRC
    assert "SPEC.select_weight" in RUNNER_SRC
    assert "unique_event_report" in RUNNER_SRC
    assert "dev_sweep" in RUNNER_SRC
    assert SPEC.PERIOD_JOBS == 3 and SPEC.FIT_NJOBS == 6
    assert SPEC.PERIOD_JOBS * SPEC.FIT_NJOBS <= 20
    assert SPEC.FREQ_PILOT == "6h" and SPEC.FREQ_FULL == "3h"
    assert SPEC.SCRATCH_NAME == "worker-d122"


def test_full_density_key_discipline_and_frozen_switch():
    assert "full-density" in SPEC_SRC or "full density" in SPEC_SRC
    assert "not full density" in RUNNER_SRC
    assert "72" in RUNNER_SRC  # every kept origin needs all 72 horizons
    assert "residual" in SPEC.SELECT_RULE and "sweep" in SPEC.SELECT_RULE
    assert SPEC.SPLIT_H == 24
    assert "check_split" in RUNNER_SRC
    assert "random_state" in RUNNER_SRC


def test_dev_only_selection_shape_and_no_held_write():
    assert "predictions_official" not in RUNNER_SRC
    assert "official" not in RUNNER_SRC.lower()
    assert "freeze" not in RUNNER_SRC
    assert "never read for weights" in SPEC.SELECT_RULE
    assert "selection/audit" in SPEC.SELECT_RULE
    assert "weight_source dev only" in SPEC.SELECT_RULE
    assert "primary seed 10" in SPEC.SELECT_RULE
    assert "grid" in SPEC.SELECT_RULE
    assert "72h purge" in SPEC.SELECT_RULE
    assert "full 3h" in SPEC.FULL_RULE and "no full run" in SPEC.FULL_RULE
    assert "0.41121495" in SPEC.FULL_RULE and "seed-repro" in SPEC.FULL_RULE
    assert "F1>=0.35" in SPEC.FULL_RULE


def test_scratch_isolation_and_owner():
    assert "store/scratch/worker-d122" in RUNNER_SRC
    assert "worker-d114" not in RUNNER_SRC and "worker-d121" not in RUNNER_SRC
    assert "worker-d123" not in RUNNER_SRC and "worker-d125" not in RUNNER_SRC
    assert SPEC.DIRECTION == "D122" and SPEC.OWNER == "worker-d122"
    assert "D122" in SPEC.GPU_REASON and "D123" in SPEC.GPU_REASON and "D125" in SPEC.GPU_REASON
    assert "gpu0" in SPEC.GPU_REASON and "gpu1" in SPEC.GPU_REASON and "CPU" in SPEC.GPU_REASON
    assert "60s" in SPEC.GPU_REASON


def test_output_contract_keys_present():
    for key in ["predictions_{dev,selection,audit}", "poison.json",
                "selection.json", "importance.json", '"passed"']:
        assert key in RUNNER_SRC, key
    assert "positive_control" in RUNNER_SRC and "pre_invariant" in RUNNER_SRC
    assert "blend_poison_check" in RUNNER_SRC
    assert "ST.poison_test" in RUNNER_SRC
    assert "sweep_table" in RUNNER_SRC and "selected_w" in RUNNER_SRC
    assert "w_distribution" in RUNNER_SRC or "w distribution" in RUNNER_SRC
    assert "per_seed" in RUNNER_SRC


def test_owned_dir_is_small():
    files = sorted(p.name for p in D122.iterdir() if p.suffix == ".py")
    assert len(files) <= 6, files
    assert {"srspec.py", "runner.py"} <= set(files)
