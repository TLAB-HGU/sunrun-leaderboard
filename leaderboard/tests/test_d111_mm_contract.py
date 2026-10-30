"""D111 margin-max contract: synthetic CPU-only checks, no data reads.

Covers the assignment acceptance surface without running the pipeline:
two-member discipline, fixed conditional per-key max pre-rule with a
dev-only median-positive-gap margin (no fitted weight on any window),
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
from d111_margmax import mmspec as SPEC  # noqa: E402
from expos import core  # noqa: E402

D111 = Path(__file__).resolve().parents[1] / "experiments" / "d111_margmax"
RUNNER_SRC = (D111 / "runner.py").read_text()
SPEC_SRC = (D111 / "mmspec.py").read_text()

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
    assert "margin-max" in SPEC.AXES_TAG
    assert SPEC.HYPOTHESIS.startswith("[family=xgb] [axes: ens=margin-max]")
    for token in ("E0003", "E0102", "E0183", "E0001", "X0164", "research/AXES.md"):
        assert token in SPEC.HYPOTHESIS, f"hypothesis must cite {token}"
    assert "weights from preofficial OOF only" in SPEC.HYPOTHESIS
    assert "tau = median" in SPEC.HYPOTHESIS
    assert "6h" in SPEC.HYPOTHESIS and "gpu0" in SPEC.HYPOTHESIS
    assert "screening" in SPEC.HYPOTHESIS
    assert SPEC.check_spec()["members"] == ["e0003", "e0102"]
    assert SPEC.check_spec()["experience"] == ["X0164"]


def test_members_and_margin_pre_rule():
    assert SPEC.MEMBERS == ("e0003", "e0102")
    assert SPEC.EIDS == {"e0003": "E0003", "e0102": "E0102"}
    assert SPEC.check_members(SPEC.MEMBERS) == SPEC.MEMBERS
    with pytest.raises(ValueError):
        SPEC.check_members(("e0003", "e0102", "e0163"))
    with pytest.raises(ValueError):
        SPEC.check_members(("e0003",))
    assert "tau = median" in SPEC.MARGIN_FORMULA


def test_compute_margin_dev_only_median_positive():
    gap = np.array([-5.0, 2.0, 4.0, 6.0, 8.0])
    assert SPEC.compute_margin(gap) == pytest.approx(5.0)
    assert SPEC.compute_margin(np.array([-3.0, -1.0])) == 0.0
    with pytest.raises(ValueError):
        SPEC.compute_margin(np.array([1.0, np.inf]))
    with pytest.raises(ValueError):
        SPEC.compute_margin(np.array([]))


def test_margin_predictions_conditional_max():
    e3 = np.array([400.0, 400.0, 400.0, 400.0])
    e10 = np.array([405.0, 410.0, 400.0, 390.0])
    out = SPEC.margin_predictions({"e0003": e3, "e0102": e10}, 5.0)
    assert np.allclose(out, np.array([400.0, 410.0, 400.0, 400.0]))
    # tau=0 reduces to unconditional max
    assert np.allclose(SPEC.margin_predictions({"e0003": e3, "e0102": e10}, 0.0),
                       np.maximum(e3, e10))
    # large tau keeps base everywhere
    assert np.allclose(SPEC.margin_predictions({"e0003": e3, "e0102": e10}, 100.0), e3)
    with pytest.raises(ValueError):
        SPEC.margin_predictions({"e0003": e3, "e0102": e10}, -1.0)
    with pytest.raises(ValueError):
        SPEC.margin_predictions({"e0003": e3, "e0102": e10[:2]}, 5.0)
    with pytest.raises(ValueError):
        SPEC.margin_predictions({"e0003": e3, "e0102": e10, "e0163": e10}, 5.0)


def test_no_forbidden_sources_or_copies():
    for name, src in (("mmspec.py", SPEC_SRC), ("runner.py", RUNNER_SRC)):
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
    assert "SPEC.margin_predictions" in RUNNER_SRC and "SPEC.compute_margin" in RUNNER_SRC
    assert "inverse" not in RUNNER_SRC and "mse_weights" not in RUNNER_SRC
    assert SPEC.PERIOD_JOBS == 3 and SPEC.THREADS_PER == 1
    assert SPEC.PERIOD_JOBS * SPEC.THREADS_PER <= 20
    assert SPEC.FREQ_PILOT == "6h" and SPEC.FREQ_FULL == "3h"
    assert SPEC.SCRATCH_NAME == "worker-d111"


def test_full_density_key_discipline():
    assert "full-density" in SPEC_SRC or "full density" in SPEC_SRC
    assert "not full density" in RUNNER_SRC
    assert "72" in RUNNER_SRC  # every kept origin needs all 72 horizons
    assert "tau" in SPEC.SELECT_RULE


def test_dev_only_selection_shape_and_no_held_write():
    assert "predictions_official" not in RUNNER_SRC
    assert "official" not in RUNNER_SRC.lower()
    assert "freeze" not in RUNNER_SRC
    assert "never read for weights" in SPEC.SELECT_RULE
    assert "selection/audit" in SPEC.SELECT_RULE
    assert "weight_source" in SPEC.SELECT_RULE and "dev-only" in SPEC.SELECT_RULE
    assert "no candidate search" in SPEC.SELECT_RULE
    assert "72h purge" in SPEC.SELECT_RULE
    assert "full 3h" in SPEC.FULL_RULE and "no full run" in SPEC.FULL_RULE
    assert "0.41121495" in SPEC.FULL_RULE and "E0102 3h density" in SPEC.FULL_RULE
    assert "F1>=0.35" in SPEC.FULL_RULE


def test_scratch_isolation_and_owner():
    assert "store/scratch/worker-d111" in RUNNER_SRC
    assert "worker-d108" not in RUNNER_SRC and "worker-d112" not in RUNNER_SRC
    assert "worker-d113" not in RUNNER_SRC
    assert SPEC.DIRECTION == "D111" and SPEC.OWNER == "worker-d111"
    assert "D111" in SPEC.GPU_REASON and "D112" in SPEC.GPU_REASON and "D113" in SPEC.GPU_REASON
    assert "gpu0" in SPEC.GPU_REASON and "60s" in SPEC.GPU_REASON


def test_output_contract_keys_present():
    for key in ["predictions_{dev,selection,audit}", "poison.json",
                "selection.json", "importance.json", '"passed"']:
        assert key in RUNNER_SRC, key
    assert "positive_control" in RUNNER_SRC and "pre_invariant" in RUNNER_SRC
    assert "blend_poison_check" in RUNNER_SRC
