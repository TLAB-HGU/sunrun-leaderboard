"""D113 residual-boost contract: synthetic CPU-only checks, no data reads.

Covers the assignment acceptance surface without running the pipeline:
two-member discipline (E0163 excluded), fixed dev-OOF residual pre-rule
(inverse dev MSE, clipped + renormalized, blended = base + w*resid),
full-density key discipline, file budget, forbidden-source hygiene,
parallel/cache discipline, scratch isolation and the expos output
contract keys.
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
from d113_resb import rbspec as SPEC  # noqa: E402
from expos import core  # noqa: E402

D113 = Path(__file__).resolve().parents[1] / "experiments" / "d113_resb"
RUNNER_SRC = (D113 / "runner.py").read_text()
SPEC_SRC = (D113 / "rbspec.py").read_text()

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
    assert "residual-boost" in SPEC.AXES_TAG
    assert SPEC.HYPOTHESIS.startswith("[family=xgb] [axes: ens=residual-boost]")
    for token in ("E0003", "E0102", "E0001", "X0164", "research/AXES.md"):
        assert token in SPEC.HYPOTHESIS, f"hypothesis must cite {token}"
    assert "weights from preofficial OOF only" in SPEC.HYPOTHESIS
    assert "E0102-E0003" in SPEC.HYPOTHESIS
    assert "6h" in SPEC.HYPOTHESIS and "gpu1" in SPEC.HYPOTHESIS
    assert "screening" in SPEC.HYPOTHESIS
    assert SPEC.check_spec()["members"] == ["e0003", "e0102"]
    assert SPEC.check_spec()["experience"] == ["X0164"]


def test_members_and_weight_bounds():
    assert SPEC.MEMBERS == ("e0003", "e0102")
    assert SPEC.EIDS == {"e0003": "E0003", "e0102": "E0102"}
    assert "e0163" not in SPEC.MEMBERS and "E0163" not in set(SPEC.EIDS.values())
    assert (SPEC.W_MIN, SPEC.W_MAX) == (0.05, 0.90)
    assert SPEC.check_members(SPEC.MEMBERS) == SPEC.MEMBERS
    with pytest.raises(ValueError):
        SPEC.check_members(("e0003", "e0102", "e0163"))
    with pytest.raises(ValueError):
        SPEC.check_members(("e0003",))


def test_inverse_mse_weights_dev_only_prerule():
    # equal MSEs -> equal weights (pure mean == half-step residual)
    w = SPEC.inverse_mse_weights({"e0003": 100.0, "e0102": 100.0})
    assert abs(w["e0003"] - 0.5) < 1e-9 and abs(w["e0102"] - 0.5) < 1e-9
    # published dev MSEs: E0003 7843.04, E0102 7705.80 -> w_e0102 ~= 0.5044
    w2 = SPEC.inverse_mse_weights({"e0003": 7843.04, "e0102": 7705.80})
    assert abs(w2["e0102"] - 0.5044) < 1e-3
    assert abs(sum(w2.values()) - 1.0) < 1e-9
    assert all(0.05 - 1e-12 <= v <= 0.90 + 1e-12 for v in w2.values())
    # extreme ratio: clip engages then renormalizes to sum 1 (e0102 dominates)
    w3 = SPEC.inverse_mse_weights({"e0003": 1e9, "e0102": 1.0})
    assert abs(sum(w3.values()) - 1.0) < 1e-9 and w3["e0102"] > 0.9 > w3["e0003"]
    with pytest.raises(ValueError):
        SPEC.inverse_mse_weights({"e0003": 100.0, "e0102": 0.0})
    with pytest.raises(ValueError):
        SPEC.inverse_mse_weights({"e0003": float("nan"), "e0102": 100.0})
    with pytest.raises(ValueError):
        SPEC.inverse_mse_weights({"e0003": 100.0, "e0102": 100.0, "e0163": 100.0})


def test_blend_predictions_residual_form():
    base = np.array([400.0, 400.0, 400.0])
    lift = np.array([500.0, 400.0, 300.0])
    w = {"e0003": 0.5, "e0102": 0.5}
    out = SPEC.blend_predictions({"e0003": base, "e0102": lift}, w)
    # blended = base + 0.5*(lift-base): half the peak lift, bounded audit move
    assert np.allclose(out, np.array([450.0, 400.0, 350.0]))
    w2 = {"e0003": 0.4956, "e0102": 0.5044}
    out2 = SPEC.blend_predictions({"e0003": base, "e0102": lift}, w2)
    assert np.allclose(out2, base + 0.5044 * (lift - base))
    with pytest.raises(ValueError):
        SPEC.blend_predictions({"e0003": base, "e0102": lift[:2]}, w)
    with pytest.raises(ValueError):
        SPEC.blend_predictions({"e0003": np.array([np.inf, 1.0, 2.0]), "e0102": lift}, w)
    with pytest.raises(ValueError):
        SPEC.blend_predictions({"e0003": base, "e0102": lift}, {"e0003": 0.5, "e0102": 0.6})
    with pytest.raises(ValueError):
        SPEC.blend_predictions({"e0003": base, "e0102": lift, "e0163": lift}, w)


def test_no_forbidden_sources_or_copies():
    for name, src in (("rbspec.py", SPEC_SRC), ("runner.py", RUNNER_SRC)):
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
    assert "SPEC.inverse_mse_weights" in RUNNER_SRC
    assert "inverse" in RUNNER_SRC
    assert SPEC.PERIOD_JOBS == 3 and SPEC.THREADS_PER == 1
    assert SPEC.PERIOD_JOBS * SPEC.THREADS_PER <= 20
    assert SPEC.FREQ_PILOT == "6h" and SPEC.FREQ_FULL == "3h"
    assert SPEC.SCRATCH_NAME == "worker-d113"


def test_full_density_key_discipline():
    assert "full-density" in SPEC_SRC or "full density" in SPEC_SRC
    assert "not full density" in RUNNER_SRC
    assert "72" in RUNNER_SRC  # every kept origin needs all 72 horizons
    assert "residual" in SPEC.SELECT_RULE


def test_dev_only_selection_shape_and_no_held_write():
    assert "predictions_official" not in RUNNER_SRC
    assert "official" not in RUNNER_SRC.lower()
    assert "freeze" not in RUNNER_SRC
    assert "never read for weights" in SPEC.SELECT_RULE
    assert "selection/audit" in SPEC.SELECT_RULE
    assert "weight_source dev only" in SPEC.SELECT_RULE
    assert "no candidate search" in SPEC.SELECT_RULE
    assert "72h purge" in SPEC.SELECT_RULE
    assert "full 3h" in SPEC.FULL_RULE and "no full run" in SPEC.FULL_RULE
    assert "0.41121495" in SPEC.FULL_RULE and "E0102 3h density" in SPEC.FULL_RULE
    assert "F1>=0.35" in SPEC.FULL_RULE


def test_scratch_isolation_and_owner():
    assert "store/scratch/worker-d113" in RUNNER_SRC
    assert "worker-d108" not in RUNNER_SRC and "worker-d111" not in RUNNER_SRC
    assert "worker-d112" not in RUNNER_SRC
    assert SPEC.DIRECTION == "D113" and SPEC.OWNER == "worker-d113"
    assert "D111" in SPEC.GPU_REASON and "D113" in SPEC.GPU_REASON and "D112" in SPEC.GPU_REASON
    assert "gpu0" in SPEC.GPU_REASON and "gpu1" in SPEC.GPU_REASON and "CPU" in SPEC.GPU_REASON
    assert "60s" in SPEC.GPU_REASON


def test_output_contract_keys_present():
    for key in ["predictions_{dev,selection,audit}", "poison.json",
                "selection.json", "importance.json", '"passed"']:
        assert key in RUNNER_SRC, key
    assert "positive_control" in RUNNER_SRC and "pre_invariant" in RUNNER_SRC
    assert "blend_poison_check" in RUNNER_SRC


def test_owned_dir_is_small():
    files = sorted(p.name for p in D113.iterdir() if p.suffix == ".py")
    assert len(files) <= 5, files
    assert {"rbspec.py", "runner.py"} <= set(files)
