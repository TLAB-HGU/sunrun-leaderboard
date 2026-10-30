"""D101 OOF-blend contract: synthetic CPU-only checks, no data reads.

Covers the assignment acceptance surface without running the pipeline:
member discipline, dev-only inverse-MSE pre-rule precision, convex blend,
file budget, forbidden-source hygiene, parallel/cache discipline, scratch
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
from d101_oofb import blenspec as SPEC  # noqa: E402
from expos import core  # noqa: E402

D101 = Path(__file__).resolve().parents[1] / "experiments" / "d101_oofb"
RUNNER_SRC = (D101 / "runner.py").read_text()
SPEC_SRC = (D101 / "blenspec.py").read_text()

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
    assert "oof-blend" in SPEC.AXES_TAG
    assert SPEC.HYPOTHESIS.startswith("[family=xgb] [axes: ens=oof-blend]")
    for token in ("E0003", "E0102", "E0163", "E0001", "X0042", "research/AXES.md"):
        assert token in SPEC.HYPOTHESIS, f"hypothesis must cite {token}"
    assert "weights from preofficial OOF only" in SPEC.HYPOTHESIS
    assert "6h" in SPEC.HYPOTHESIS and "gpu1" in SPEC.HYPOTHESIS
    assert "screening" in SPEC.HYPOTHESIS
    assert SPEC.check_spec()["members"] == ["e0003", "e0102", "e0163"]


def test_members_and_weight_pre_rule():
    assert SPEC.MEMBERS == ("e0003", "e0102", "e0163")
    assert SPEC.EIDS == {"e0003": "E0003", "e0102": "E0102", "e0163": "E0163"}
    assert (SPEC.W_MIN, SPEC.W_MAX) == (0.05, 0.90)
    assert SPEC.check_members(SPEC.MEMBERS) == SPEC.MEMBERS
    with pytest.raises(ValueError):
        SPEC.check_members(("e0003", "e0102"))


def test_inverse_weights_dev_only_precise():
    mses = {"e0003": 7828.05, "e0102": 7663.73, "e0163": 8130.10}
    w = SPEC.inverse_mse_weights(mses)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(v >= 0 for v in w.values())
    assert w["e0102"] > w["e0003"] > w["e0163"]
    assert abs(w["e0003"] - 0.3351) < 0.002
    assert abs(w["e0102"] - 0.3423) < 0.002
    assert abs(w["e0163"] - 0.3226) < 0.002
    wu = SPEC.inverse_mse_weights({m: 100.0 for m in SPEC.MEMBERS})
    assert all(abs(v - 1 / 3) < 1e-9 for v in wu.values())
    with pytest.raises(ValueError):
        SPEC.inverse_mse_weights({"e0003": 1.0, "e0102": 2.0})
    with pytest.raises(ValueError):
        SPEC.inverse_mse_weights({m: 0.0 for m in SPEC.MEMBERS})


def test_blend_predictions_convex():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    c = np.array([7.0, 8.0, 9.0])
    preds = {"e0003": a, "e0102": b, "e0163": c}
    w = {"e0003": 0.3351, "e0102": 0.3423, "e0163": 0.3226}
    out = SPEC.blend_predictions(preds, w)
    assert np.allclose(out, 0.3351 * a + 0.3423 * b + 0.3226 * c)
    with pytest.raises(ValueError):
        SPEC.blend_predictions(preds, {"e0003": 0.5, "e0102": 0.5, "e0163": 0.5})
    with pytest.raises(ValueError):
        SPEC.blend_predictions({"e0003": a, "e0102": b[:2], "e0163": c}, w)


def test_no_forbidden_sources_or_copies():
    for name, src in (("blenspec.py", SPEC_SRC), ("runner.py", RUNNER_SRC)):
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
    assert "SPEC.inverse_mse_weights" in RUNNER_SRC
    assert "SPEC.blend_predictions" in RUNNER_SRC
    assert SPEC.PERIOD_JOBS == 3 and SPEC.THREADS_PER == 1
    assert SPEC.PERIOD_JOBS * SPEC.THREADS_PER <= 20
    assert SPEC.FREQ_PILOT == "6h" and SPEC.FREQ_FULL == "3h"
    assert SPEC.SCRATCH_NAME == "worker-d101"


def test_dev_only_selection_shape_and_no_held_write():
    assert "predictions_official" not in RUNNER_SRC
    assert "official" not in RUNNER_SRC.lower()
    assert "freeze" not in RUNNER_SRC
    assert "_dev_mses_on_intersection" in RUNNER_SRC
    assert "dev_mses" in RUNNER_SRC and "no selection/audit statistic used" in RUNNER_SRC
    assert "never read for weights" in SPEC.SELECT_RULE
    assert "selection/audit" in SPEC.SELECT_RULE
    assert "weight_source dev only" in SPEC.SELECT_RULE
    assert "no candidate search" in SPEC.SELECT_RULE
    assert "72h purge" in SPEC.SELECT_RULE
    assert "full 3h" in SPEC.FULL_RULE and "no full run" in SPEC.FULL_RULE
    assert "0.41121495" in SPEC.FULL_RULE and "seed-repro" in SPEC.FULL_RULE


def test_scratch_isolation_and_owner():
    assert "store/scratch/worker-d101" in RUNNER_SRC
    assert "worker-d99" not in RUNNER_SRC and "worker-d100" not in RUNNER_SRC
    assert SPEC.DIRECTION == "D101" and SPEC.OWNER == "worker-d101"
    assert "D99" in SPEC.GPU_REASON and "D101" in SPEC.GPU_REASON and "D100" in SPEC.GPU_REASON
    assert "gpu1" in SPEC.GPU_REASON and "60s" in SPEC.GPU_REASON


def test_output_contract_keys_present():
    for key in ["predictions_{dev,selection,audit}", "poison.json",
                "selection.json", "importance.json", '"passed"']:
        assert key in RUNNER_SRC, key
    assert "positive_control" in RUNNER_SRC and "pre_invariant" in RUNNER_SRC
    assert "blend_poison_check" in RUNNER_SRC
