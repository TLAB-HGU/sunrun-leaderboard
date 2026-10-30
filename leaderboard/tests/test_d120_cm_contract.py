"""D120 capped-max contract: synthetic CPU-only checks, no data reads.

Covers the assignment acceptance surface without running the pipeline:
two-member discipline (E0163 excluded), preregistered dev-OOF cap sweep
(9 points 5..150 kms, F1-floor-passing best-mean, blended =
min(max(e0003,e0102),e0003+cap)), full-density key discipline, file
budget, forbidden-source hygiene, parallel/cache discipline, scratch
isolation and the expos output contract keys.
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
from d120_capmax import capspec as SPEC  # noqa: E402
from expos import core  # noqa: E402

D120 = Path(__file__).resolve().parents[1] / "experiments" / "d120_capmax"
RUNNER_SRC = (D120 / "runner.py").read_text()
SPEC_SRC = (D120 / "capspec.py").read_text()

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay",
             "official_check", "official-check", "predictions_official_truth"]
COPY_BANS = ["def speed_features(", "def ace_features(", "def ch_features(",
             "def enlil_features(", "def build(", "def arms(", "def full_grid(",
             "def write("]
LATER_DATES = ("2025-12-04", "2026-02-25", "2026-03-04", "2026-05-28",
               "2026-05-29", "2026-05-31", "2026-09-26")


def test_dual_tag_and_citations():
    assert SPEC.FAMILY_TAG == "[family=xgb]"
    assert SPEC.AXES_TAG.startswith("[axes:") and "ens=" in SPEC.AXES_TAG
    assert "capped-max" in SPEC.AXES_TAG
    assert SPEC.HYPOTHESIS.startswith("[family=xgb] [axes: ens=capped-max]")
    for token in ("E0003", "E0102", "E0001", "X0161", "X0164", "research/AXES.md"):
        assert token in SPEC.HYPOTHESIS, f"hypothesis must cite {token}"
    assert "weights from preofficial OOF only" in SPEC.HYPOTHESIS
    assert "min(max(E0003,E0102),E0003+cap)" in SPEC.HYPOTHESIS
    assert "E0102-E0003" in SPEC.HYPOTHESIS
    assert "6h" in SPEC.HYPOTHESIS and "gpu1" in SPEC.HYPOTHESIS
    assert "screening" in SPEC.HYPOTHESIS
    assert SPEC.check_spec()["members"] == ["e0003", "e0102"]
    assert SPEC.check_spec()["experience"] == ["X0161", "X0164"]


def test_members_grid_and_floor():
    assert SPEC.MEMBERS == ("e0003", "e0102")
    assert SPEC.EIDS == {"e0003": "E0003", "e0102": "E0102"}
    assert "e0163" not in SPEC.MEMBERS and "E0163" not in set(SPEC.EIDS.values())
    assert SPEC.CAP_GRID == (5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 65.0, 90.0, 150.0)
    assert (SPEC.CAP_MIN, SPEC.CAP_MAX) == (5.0, 150.0)
    assert SPEC.FLOOR_DEV_F1 == 0.55
    assert SPEC.check_grid() == SPEC.CAP_GRID
    assert SPEC.check_members(SPEC.MEMBERS) == SPEC.MEMBERS
    with pytest.raises(ValueError):
        SPEC.check_members(("e0003", "e0102", "e0163"))
    with pytest.raises(ValueError):
        SPEC.check_members(("e0003",))
    with pytest.raises(ValueError):
        SPEC.check_grid((5.0,) * 9)
    with pytest.raises(ValueError):
        SPEC.blend_predictions({"e0003": np.array([1.0]), "e0102": np.array([1.0])}, 4.9)
    with pytest.raises(ValueError):
        SPEC.blend_predictions({"e0003": np.array([1.0]), "e0102": np.array([1.0])}, 150.1)


def test_blend_predictions_capped_form():
    base = np.array([400.0, 400.0, 400.0])
    lift = np.array([500.0, 400.0, 300.0])
    out = SPEC.blend_predictions({"e0003": base, "e0102": lift}, 50.0)
    # min(max, base+50): max=[500,400,400], base+50=[450,450,450] -> [450,400,400]
    assert np.allclose(out, np.array([450.0, 400.0, 400.0]))
    big = SPEC.blend_predictions({"e0003": base, "e0102": lift}, 150.0)
    assert np.allclose(big, np.maximum(base, lift))
    with pytest.raises(ValueError):
        SPEC.blend_predictions({"e0003": base, "e0102": lift[:2]}, 50.0)
    with pytest.raises(ValueError):
        SPEC.blend_predictions({"e0003": np.array([np.inf, 1.0, 2.0]), "e0102": lift}, 50.0)
    with pytest.raises(ValueError):
        SPEC.blend_predictions({"e0003": base, "e0102": lift, "e0163": lift}, 50.0)


def test_select_cap_floor_passing_best_mean():
    # floor 0.55: passing {5:7820/0.565, 10:7780/0.596, 15:7746/0.565, 20:7719/0.652}
    # best mean among passing is cap=20 (lowest MSE)
    sweep = [
        {"cap": 5.0, "dev_mse": 7820.0, "dev_f1": 0.565},
        {"cap": 10.0, "dev_mse": 7780.0, "dev_f1": 0.596},
        {"cap": 15.0, "dev_mse": 7746.0, "dev_f1": 0.565},
        {"cap": 20.0, "dev_mse": 7719.0, "dev_f1": 0.652},
        {"cap": 30.0, "dev_mse": 7699.0, "dev_f1": 0.533},
        {"cap": 45.0, "dev_mse": 7685.0, "dev_f1": 0.444},
        {"cap": 65.0, "dev_mse": 7678.0, "dev_f1": 0.500},
        {"cap": 90.0, "dev_mse": 7677.0, "dev_f1": 0.489},
        {"cap": 150.0, "dev_mse": 7684.0, "dev_f1": 0.478},
    ]
    sel = SPEC.select_cap(sweep)
    assert sel["cap"] == 20.0 and sel["passed_floor"] is True
    # tie on MSE picks smallest cap
    tied = [dict(r) for r in sweep]
    tied[0] = {"cap": 5.0, "dev_mse": 7719.0, "dev_f1": 0.60}
    sel2 = SPEC.select_cap(tied)
    assert sel2["cap"] == 5.0
    # floor miss picks lowest MSE with passed_floor=false
    miss = [{"cap": g, "dev_mse": 7700.0 - g * 0.01, "dev_f1": 0.30} for g in SPEC.CAP_GRID]
    sel3 = SPEC.select_cap(miss)
    assert sel3["passed_floor"] is False
    # grid order/coverage enforced (dev-only rule shape)
    with pytest.raises(ValueError):
        SPEC.select_cap(sweep[:8])
    with pytest.raises(ValueError):
        SPEC.select_cap([{"cap": g, "dev_mse": 1.0, "dev_f1": 2.0} for g in SPEC.CAP_GRID])
    with pytest.raises(ValueError):
        SPEC.select_cap([{"cap": 999.0, "dev_mse": 1.0, "dev_f1": 0.6} for _ in SPEC.CAP_GRID])


def test_no_forbidden_sources_or_copies():
    for name, src in (("capspec.py", SPEC_SRC), ("runner.py", RUNNER_SRC)):
        for tok in FORBIDDEN:
            assert tok not in src, (name, tok)
        for s in LATER_DATES:
            assert s not in src, (name, f"later window {s}")
    for ban in COPY_BANS:
        assert ban not in RUNNER_SRC, f"runner must import, not copy: {ban}"
    assert core.scan(SPEC_SRC + "\n" + RUNNER_SRC + "\n" + SPEC.HYPOTHESIS) == []


def test_parallel_cache_and_budget():
    assert "ThreadPoolExecutor" in RUNNER_SRC
    assert "fcache_key" in RUNNER_SRC and "featcache" in RUNNER_SRC
    assert "from existing_all import" in RUNNER_SRC and "write(" in RUNNER_SRC
    assert "SPEC.blend_predictions" in RUNNER_SRC
    assert "SPEC.select_cap" in RUNNER_SRC
    assert "unique_event_report" in RUNNER_SRC
    assert SPEC.PERIOD_JOBS == 3 and SPEC.THREADS_PER == 1
    assert SPEC.PERIOD_JOBS * SPEC.THREADS_PER <= 20
    assert SPEC.FREQ_PILOT == "6h" and SPEC.FREQ_FULL == "3h"
    assert SPEC.SCRATCH_NAME == "worker-d120"


def test_full_density_key_discipline():
    assert "full-density" in SPEC_SRC or "full density" in SPEC_SRC
    assert "not full density" in RUNNER_SRC
    assert "72" in RUNNER_SRC  # every kept origin needs all 72 horizons
    assert "capped" in SPEC.SELECT_RULE and "sweep" in SPEC.SELECT_RULE


def test_dev_only_selection_shape_and_no_held_write():
    assert "predictions_official" not in RUNNER_SRC
    assert "official" not in RUNNER_SRC.lower()
    assert "freeze" not in RUNNER_SRC
    assert "never read for weights" in SPEC.SELECT_RULE
    assert "selection/audit" in SPEC.SELECT_RULE
    assert "weight_source dev only" in SPEC.SELECT_RULE
    assert "grid" in SPEC.SELECT_RULE
    assert "72h purge" in SPEC.SELECT_RULE
    assert "full 3h" in SPEC.FULL_RULE and "no full run" in SPEC.FULL_RULE
    assert "0.41121495" in SPEC.FULL_RULE and "E0102 3h density" in SPEC.FULL_RULE
    assert "F1>=0.35" in SPEC.FULL_RULE


def test_scratch_isolation_and_owner():
    assert "store/scratch/worker-d120" in RUNNER_SRC
    assert "worker-d119" not in RUNNER_SRC and "worker-d121" not in RUNNER_SRC
    assert "worker-d108" not in RUNNER_SRC
    assert SPEC.DIRECTION == "D120" and SPEC.OWNER == "worker-d120"
    assert "D120" in SPEC.GPU_REASON and "D119" in SPEC.GPU_REASON and "D121" in SPEC.GPU_REASON
    assert "gpu0" in SPEC.GPU_REASON and "gpu1" in SPEC.GPU_REASON and "CPU" in SPEC.GPU_REASON
    assert "60s" in SPEC.GPU_REASON


def test_output_contract_keys_present():
    for key in ["predictions_{dev,selection,audit}", "poison.json",
                "selection.json", "importance.json", '"passed"']:
        assert key in RUNNER_SRC, key
    assert "positive_control" in RUNNER_SRC and "pre_invariant" in RUNNER_SRC
    assert "blend_poison_check" in RUNNER_SRC
    assert "sweep_table" in RUNNER_SRC and "selected_cap" in RUNNER_SRC


def test_owned_dir_is_small():
    files = sorted(p.name for p in D120.iterdir() if p.suffix == ".py")
    assert len(files) <= 5, files
    assert {"capspec.py", "runner.py"} <= set(files)
