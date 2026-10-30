"""D117 peak-boost contract: synthetic CPU-only checks, no data reads.

Covers the assignment acceptance surface without running the pipeline:
single-base discipline (E0189 only), preregistered dev-OOF gain sweep
(9 points 1.00..1.60, F1-floor-passing best-mean, boosted = m_o +
g*max(0, p-m_o)), full-density key discipline, file budget,
forbidden-source hygiene, parallel/cache discipline, scratch isolation
and the expos output contract keys.
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
from d117_pkboost import pbspec as SPEC  # noqa: E402
from expos import core  # noqa: E402

D117 = Path(__file__).resolve().parents[1] / "experiments" / "d117_pkboost"
RUNNER_SRC = (D117 / "runner.py").read_text()
SPEC_SRC = (D117 / "pbspec.py").read_text()

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
    assert SPEC.AXES_TAG.startswith("[axes:") and "post=" in SPEC.AXES_TAG
    assert "peak-boost" in SPEC.AXES_TAG
    assert SPEC.HYPOTHESIS.startswith("[family=xgb] [axes: post=peak-boost]")
    for token in ("E0189", "E0001", "E0003", "X0170", "research/AXES.md"):
        assert token in SPEC.HYPOTHESIS, f"hypothesis must cite {token}"
    assert "preofficial OOF only" in SPEC.HYPOTHESIS
    assert "p + (g-1)*max(0, p-m_o)" in SPEC.HYPOTHESIS
    assert "6h" not in SPEC.HYPOTHESIS or "3h" in SPEC.HYPOTHESIS
    assert "3h" in SPEC.HYPOTHESIS and "CPU" in SPEC.HYPOTHESIS
    assert SPEC.check_spec()["base"] == "e0189"
    assert SPEC.check_spec()["experience"] == ["X0170"]


def test_base_grid_and_floor():
    assert SPEC.BASE == "e0189"
    assert SPEC.EIDS == {"e0189": "E0189"}
    assert SPEC.G_GRID == (1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.40, 1.50, 1.60)
    assert (SPEC.G_MIN, SPEC.G_MAX) == (1.00, 1.60)
    assert SPEC.FLOOR_DEV_F1 == 0.60
    assert SPEC.check_grid() == SPEC.G_GRID
    assert SPEC.check_base((SPEC.BASE,)) == (SPEC.BASE,)
    with pytest.raises(ValueError):
        SPEC.check_base(("e0003", "e0102"))
    with pytest.raises(ValueError):
        SPEC.check_base(("e0189", "e0003"))
    assert SPEC.check_gain(1.20) == 1.20
    with pytest.raises(ValueError):
        SPEC.check_gain(0.99)
    with pytest.raises(ValueError):
        SPEC.check_gain(1.61)


def test_boost_predictions_peak_form():
    pred = np.array([400.0, 500.0, 300.0, 400.0])
    origin = np.array(["o1", "o1", "o2", "o2"])
    # o1 mean 450: up 50*1.2 -> 510, valley-side 400 unchanged
    # o2 mean 350: up 50*1.2 -> 410, valley-side 300 unchanged
    out = SPEC.boost_predictions(pred, origin, 1.20)
    assert np.allclose(out, np.array([400.0, 510.0, 300.0, 410.0]))
    # g=1.0 is identity
    assert np.allclose(SPEC.boost_predictions(pred, origin, 1.00), pred)
    # uniform shift passes through (+100 in, +100 out with same deviations)
    assert np.allclose(SPEC.boost_predictions(pred + 100.0, origin, 1.40),
                       SPEC.boost_predictions(pred, origin, 1.40) + 100.0)
    with pytest.raises(ValueError):
        SPEC.boost_predictions(np.array([np.inf, 1.0]), np.array(["a", "b"]), 1.10)
    with pytest.raises(ValueError):
        SPEC.boost_predictions(pred, origin[:2], 1.10)
    with pytest.raises(ValueError):
        SPEC.boost_predictions(pred, origin, 0.90)


def test_select_gain_floor_passing_best_mean():
    sweep = [
        {"g": 1.00, "dev_mse": 7719.0, "dev_f1": 0.652},
        {"g": 1.05, "dev_mse": 7725.0, "dev_f1": 0.640},
        {"g": 1.10, "dev_mse": 7735.0, "dev_f1": 0.610},
        {"g": 1.15, "dev_mse": 7750.0, "dev_f1": 0.590},
        {"g": 1.20, "dev_mse": 7770.0, "dev_f1": 0.580},
        {"g": 1.30, "dev_mse": 7800.0, "dev_f1": 0.550},
        {"g": 1.40, "dev_mse": 7850.0, "dev_f1": 0.520},
        {"g": 1.50, "dev_mse": 7920.0, "dev_f1": 0.500},
        {"g": 1.60, "dev_mse": 8000.0, "dev_f1": 0.480},
    ]
    sel = SPEC.select_gain(sweep)
    assert sel["g"] == 1.00 and sel["passed_floor"] is True
    # tie on MSE picks smallest g
    tied = [dict(r) for r in sweep]
    tied[1] = {"g": 1.05, "dev_mse": 7719.0, "dev_f1": 0.640}
    sel2 = SPEC.select_gain(tied)
    assert sel2["g"] == 1.00
    # floor miss picks lowest MSE with passed_floor=false
    miss = [{"g": g, "dev_mse": 7700.0 + g, "dev_f1": 0.30} for g in SPEC.G_GRID]
    sel3 = SPEC.select_gain(miss)
    assert sel3["passed_floor"] is False
    with pytest.raises(ValueError):
        SPEC.select_gain(sweep[:8])
    with pytest.raises(ValueError):
        SPEC.select_gain([{"g": g, "dev_mse": 1.0, "dev_f1": 2.0} for g in SPEC.G_GRID])
    with pytest.raises(ValueError):
        SPEC.select_gain([{"g": 9.99, "dev_mse": 1.0, "dev_f1": 0.6} for _ in SPEC.G_GRID])


def test_no_forbidden_sources_or_copies():
    for name, src in (("pbspec.py", SPEC_SRC), ("runner.py", RUNNER_SRC)):
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
    assert "SPEC.boost_predictions" in RUNNER_SRC
    assert "SPEC.select_gain" in RUNNER_SRC
    assert "unique_event_report" in RUNNER_SRC
    assert SPEC.PERIOD_JOBS == 3 and SPEC.THREADS_PER == 1
    assert SPEC.PERIOD_JOBS * SPEC.THREADS_PER <= 20
    assert SPEC.FREQ_PILOT == "3h" and SPEC.FREQ_FULL == "3h"
    assert SPEC.SCRATCH_NAME == "worker-d117"


def test_full_density_key_discipline():
    assert "full-density" in SPEC_SRC or "full density" in SPEC_SRC
    assert "not full density" in RUNNER_SRC
    assert "72" in RUNNER_SRC  # every kept origin needs all 72 horizons
    assert "peak" in SPEC.SELECT_RULE and "sweep" in SPEC.SELECT_RULE


def test_dev_only_selection_shape_and_no_held_write():
    assert "predictions_official" not in RUNNER_SRC
    assert "official" not in RUNNER_SRC.lower()
    assert "freeze" not in RUNNER_SRC
    assert "never read for gains" in SPEC.SELECT_RULE
    assert "selection/audit" in SPEC.SELECT_RULE
    assert "gain_source dev only" in SPEC.SELECT_RULE
    assert "grid" in SPEC.SELECT_RULE
    assert "72h purge" in SPEC.SELECT_RULE
    assert "full 3h" in SPEC.FULL_RULE and "no full run" in SPEC.FULL_RULE
    assert "0.41121495" in SPEC.FULL_RULE and "E0189 3h density" in SPEC.FULL_RULE
    assert "F1>=0.35" in SPEC.FULL_RULE


def test_scratch_isolation_and_owner():
    assert "store/scratch/worker-d117" in RUNNER_SRC
    assert "worker-d114" not in RUNNER_SRC and "worker-d115" not in RUNNER_SRC
    assert "worker-d116" not in RUNNER_SRC and "worker-d118" not in RUNNER_SRC
    assert SPEC.DIRECTION == "D117" and SPEC.OWNER == "worker-d117"
    assert "D117" in SPEC.CPU_REASON and "D114" in SPEC.CPU_REASON and "D118" in SPEC.CPU_REASON
    assert "gpu0" in SPEC.CPU_REASON and "gpu1" in SPEC.CPU_REASON and "CPU" in SPEC.CPU_REASON
    assert "60s" in SPEC.CPU_REASON


def test_output_contract_keys_present():
    for key in ["predictions_{dev,selection,audit}", "poison.json",
                "selection.json", "importance.json", '"passed"']:
        assert key in RUNNER_SRC, key
    assert "positive_control" in RUNNER_SRC and "pre_invariant" in RUNNER_SRC
    assert "boost_poison_check" in RUNNER_SRC
    assert "sweep_table" in RUNNER_SRC and "selected_g" in RUNNER_SRC


def test_owned_dir_is_small():
    files = sorted(p.name for p in D117.iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"pbspec.py", "runner.py"} <= set(files)
