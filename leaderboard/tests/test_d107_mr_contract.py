"""D107 monotone-recal contract: synthetic CPU-only checks, no data reads.

Covers the assignment acceptance surface without running the pipeline:
single global monotone map discipline (dev-OOF-only fit, sel/audit
apply-only, one post axis), file budget, forbidden-source hygiene,
parallel/cache discipline, scratch isolation and the expos output contract.
The pilot run itself (registered separately) proves end-to-end behaviour
through expos run/evaluate/poison.
"""
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d107_monor"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
sys.path.insert(0, str(LB))

import monospec as SPEC  # noqa: E402
from expos import core  # noqa: E402

D107 = LB / "experiments" / "d107_monor"
RUNNER_SRC = (D107 / "runner.py").read_text()
SPEC_SRC = (D107 / "monospec.py").read_text()

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay",
             "official_check", "official-check", "predictions_official_truth"]
COPY_BANS = ["def speed_features(", "def ace_features(", "def ch_features(",
             "def enlil_features(", "def build(", "def arms(", "def full_grid(",
             "def write("]
LATER_DATES = ("2025-12-04", "2026-02-25", "2026-03-04", "2026-05-28",
               "2026-05-29", "2026-05-31", "2026-09-26")


def _synth(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(300.0, 800.0, n)
    obs = 0.9 * base + 40.0 + rng.normal(0, 15.0, n)
    return base, obs


def test_dual_tag_and_citations():
    assert SPEC.FAMILY_TAG == "[family=xgb]"
    assert SPEC.AXES_TAG == "[axes: post=monotone-recal]"
    assert SPEC.HYPOTHESIS.startswith("[family=xgb] [axes: post=monotone-recal]")
    for token in ("E0003", "E0001", "E0139", "X0127", "E0143", "X0132",
                  "research/AXES.md", "X0157", "X0158"):
        assert token in SPEC.HYPOTHESIS, f"hypothesis must cite {token}"
    assert "dev-OOF" in SPEC.HYPOTHESIS
    assert "selection/audit" in SPEC.HYPOTHESIS
    assert "6h" in SPEC.HYPOTHESIS and "3h" in SPEC.HYPOTHESIS
    assert "screening" in SPEC.HYPOTHESIS
    assert "No freeze" in SPEC.HYPOTHESIS
    assert "official-score" in SPEC.HYPOTHESIS
    assert "cpu" in SPEC.HYPOTHESIS.lower() or "CPU" in SPEC.HYPOTHESIS
    assert "60s" in SPEC.HYPOTHESIS
    assert SPEC.check_spec()["experience"] == ["X0127", "X0132"]
    assert SPEC.DIRECTION == "D107" and SPEC.OWNER == "worker-d107"


def test_single_global_map_monotone_clip_deterministic():
    base, obs = _synth()
    m = SPEC.fit_global_map(base, obs)
    assert len(m["x"]) > 0 and len(m["x"]) == len(m["y"])
    assert bool((np.diff(m["y"]) >= 0).all()), "fitted values must be non-decreasing"
    out = SPEC.apply_global_map(base, m)
    assert np.isfinite(out).all()
    # monotone function of base: sorting by base sorts the output
    order = np.argsort(base, kind="stable")
    assert bool((np.diff(out[order]) >= -1e-9).all())
    # clip outside the OOF range
    lo = SPEC.apply_global_map(np.array([1.0]), m)[0]
    hi = SPEC.apply_global_map(np.array([5000.0]), m)[0]
    assert lo == m["y"][0] and hi == m["y"][-1]
    # determinism guard: frozen map is a pure function of the base vector
    assert np.array_equal(out, SPEC.apply_global_map(base, m))
    # empty map stays identity (abstain path)
    ident = SPEC.apply_global_map(base, {"x": np.array([]), "y": np.array([])})
    assert np.array_equal(ident, base)
    with pytest.raises(ValueError):
        SPEC.fit_global_map(base[:50], obs[:50])
    with pytest.raises(ValueError):
        SPEC.apply_global_map(np.array([np.inf, 1.0]), m)


def test_global_gate_mse_and_f1():
    assert SPEC.select_apply(100.0, 90.0, 0.40, 0.42) is True
    assert SPEC.select_apply(100.0, 110.0, 0.40, 0.42) is False  # MSE regress
    assert SPEC.select_apply(100.0, 90.0, 0.40, 0.39) is False  # F1 regress
    assert SPEC.select_apply(100.0, 90.0, 0.40, None) is False  # fail closed
    assert SPEC.select_apply(100.0, 90.0, None, 0.42) is False  # fail closed


def test_no_forbidden_sources_or_copies():
    for name, src in (("monospec.py", SPEC_SRC), ("runner.py", RUNNER_SRC)):
        for tok in FORBIDDEN:
            assert tok not in src, (name, tok)
        for s in LATER_DATES:
            assert s not in src, (name, f"later window {s}")
    for ban in COPY_BANS:
        assert ban not in RUNNER_SRC, f"runner must import, not copy: {ban}"
    assert "predictions_official" not in RUNNER_SRC
    assert "sample_weight" not in RUNNER_SRC
    assert core.scan(SPEC_SRC + "\n" + RUNNER_SRC + "\n" + SPEC.HYPOTHESIS) == []


def test_parallel_cache_and_budget():
    assert "ThreadPoolExecutor" in RUNNER_SRC
    assert "fcache_key" in RUNNER_SRC and "featcache" in RUNNER_SRC
    assert "from existing_all import" in RUNNER_SRC and "write(" in RUNNER_SRC
    assert "SPEC.apply_global_map" in RUNNER_SRC and "SPEC.fit_global_map" in RUNNER_SRC
    assert SPEC.PERIOD_JOBS == 3 and SPEC.THREADS_PER == 1
    assert SPEC.PERIOD_JOBS * SPEC.THREADS_PER <= 20
    assert SPEC.FREQ_PILOT == "6h" and SPEC.FREQ_FULL == "3h"
    assert SPEC.SCRATCH_NAME == "worker-d107"
    assert SPEC.OOF_FOLDS == 3 and SPEC.PURGE_ORIGIN_H == 72
    assert SPEC.MIN_OOF_ROWS == 1000


def test_dev_only_selection_shape_and_full_rule():
    assert "freeze" not in RUNNER_SRC.lower()
    assert "feeds any fitted value" in SPEC.SELECT_RULE
    assert "selection/audit" in SPEC.SELECT_RULE
    assert "apply-only" in SPEC.SELECT_RULE
    assert "dev-OOF" in SPEC.SELECT_RULE
    assert "post=monotone-recal" in SPEC.SELECT_RULE
    assert "72h" in SPEC.SELECT_RULE
    assert "full 3h" in SPEC.FULL_RULE and "no full run" in SPEC.FULL_RULE
    assert "0.41121495" in SPEC.FULL_RULE and "seed-repro" in SPEC.FULL_RULE
    assert "F1>=0.30" in SPEC.FULL_RULE and "E0001" in SPEC.FULL_RULE
    assert "E0003" in SPEC.FULL_RULE and "rc==0" in SPEC.FULL_RULE
    assert "poison" in SPEC.FULL_RULE


def test_scratch_isolation_and_owner():
    assert "store/scratch/worker-d107" in RUNNER_SRC
    others = [l for l in RUNNER_SRC.splitlines() if "worker-" in l and "worker-d107" not in l]
    assert others == [], f"scratch leak: {others[:3]}"
    assert "worker-d107" in SPEC_SRC
    assert SPEC.DIRECTION == "D107" and SPEC.OWNER == "worker-d107"
    assert "D105" in SPEC.GPU_REASON and "D106" in SPEC.GPU_REASON and "D107" in SPEC.GPU_REASON
    assert "CPU" in SPEC.GPU_REASON and "60s" in SPEC.GPU_REASON and "20" in SPEC.GPU_REASON
    assert SPEC.check_members(("e0003",)) == ("e0003",)
    with pytest.raises(ValueError):
        SPEC.check_members(("e0003", "e0102"))


def test_output_contract_keys_present():
    for key in ["predictions_{dev,selection,audit}", "poison.json",
                "selection.json", "importance.json", '"passed"']:
        assert key in RUNNER_SRC, key
    assert "positive_control" in RUNNER_SRC and "pre_invariant" in RUNNER_SRC
    assert "recal_poison_check" in RUNNER_SRC
    assert "SystemExit" in RUNNER_SRC  # fail-closed on poison/non-finite


def test_owned_dir_is_small():
    files = sorted(p.name for p in D107.iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"monospec.py", "runner.py"} <= set(files)
