"""D88 level-regime contract: synthetic CPU-only checks, no data reads.

Covers the assignment acceptance surface without running the pipeline:
regime purity/causality on synthetic frames, single-definition discipline
(no candidate search, dev-only selection shape), file budget,
forbidden-source hygiene, cuda/parallel/cache discipline, scratch isolation
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
from d88_regime import regspec as SP  # noqa: E402
from expos import core  # noqa: E402

D88 = Path(__file__).resolve().parents[1] / "experiments" / "d88_regime"
RUNNER_SRC = (D88 / "runner.py").read_text()
SPEC_SRC = (D88 / "regspec.py").read_text()

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay",
             "official_check", "official-check", "predictions_official_truth"]
COPY_BANS = ["def speed_features(", "def ace_features(", "def ch_features(",
             "def enlil_features(", "def build(", "def arms(", "def full_grid(",
             "def write("]
# Selection/audit window boundaries must never appear in preregistered code.
LATER_DATES = ("2025-12-04", "2026-02-25", "2026-03-04", "2026-05-28",
               "2026-05-29", "2026-05-31", "2026-09-26")


def _frame(rows):
    return pd.DataFrame(rows, columns=["origin", SP.V0_COL])


def test_dual_tag_and_experience_citation():
    assert SP.FAMILY_TAG == "[family=xgb]"
    assert SP.AXES_TAG.startswith("[axes:") and "data=" in SP.AXES_TAG
    assert "regime-specific" in SP.AXES_TAG
    assert SP.HYPOTHESIS.startswith("[family=xgb] [axes: data=regime-specific]")
    for token in ("E0001", "E0003", "X0008", "research/AXES.md"):
        assert token in SP.HYPOTHESIS, f"hypothesis must cite {token}"
    assert "regime-specific models" in SP.HYPOTHESIS
    assert "6h" in SP.HYPOTHESIS and "gpu1" in SP.HYPOTHESIS
    assert "freeze" in SP.HYPOTHESIS.lower() and "official-score" in SP.HYPOTHESIS
    assert "screening" in SP.HYPOTHESIS
    assert SP.check_spec()["experience"] == ["X0008", "X0053"]


def test_regime_preregistered_single_definition():
    assert SP.REGIMES == ("slow", "mid", "fast")
    assert (SP.SLOW_MAX, SP.FAST_MIN) == (420.0, 520.0)
    assert SP.V0_COL == "v_lag0" and SP.REGIME_COL == "regime"
    assert SP.MIN_ROWS == 2000
    assert SP.SEED == 0 and SP.PURGE_H == 72
    assert SP.FIT_NJOBS == 2 and SP.PERIOD_JOBS == 3
    assert SP.FREQ_PILOT == "6h" and SP.FREQ_FULL == "3h"
    assert SP.SCRATCH_NAME == "worker-d88"
    # single definition: no candidate search anywhere
    assert "CANDIDATES" not in SPEC_SRC and "pick_winner" not in SPEC_SRC
    assert "oof_select" not in RUNNER_SRC and "CANDIDATES" not in RUNNER_SRC
    assert "def oof_select" not in RUNNER_SRC


def test_assign_slow_mid_fast():
    df = _frame([("2025-01-01", 300.0),   # slow
                 ("2025-01-02", 419.9),   # slow (boundary)
                 ("2025-01-03", 420.0),   # mid (boundary)
                 ("2025-01-04", 470.0),   # mid
                 ("2025-01-05", 519.9),   # mid (boundary)
                 ("2025-01-06", 520.0),   # fast (boundary)
                 ("2025-01-07", 700.0)])  # fast
    assert list(SP.assign(df)) == ["slow", "slow", "mid", "mid", "mid", "fast", "fast"]


def test_assign_nan_safe_quiet_mid_and_horizon_blind():
    df = _frame([("2025-01-01", float("nan")),
                 ("2025-01-02", 700.0)])
    df["h"] = [1, 72]
    df["y"] = [999.0, 999.0]
    # NaN v0 -> mid; horizon/target columns cannot change the assignment
    assert list(SP.assign(df)) == ["mid", "fast"]
    df2 = df.copy()
    df2["h"] = [72, 1]
    df2["y"] = [1.0, 1.0]
    assert list(SP.assign(df2)) == list(SP.assign(df))


def test_assign_uses_origin_row_only():
    df = _frame([(f"2025-01-0{d}", v) for d, v in
                 enumerate([300.0, 470.0, 700.0, 419.9, 520.0], start=1)])
    fwd = list(SP.assign(df))
    rev = list(SP.assign(df.iloc[::-1]))[::-1]
    assert fwd == rev
    assert set(fwd) <= set(SP.REGIMES)


def test_no_forbidden_sources_or_copies():
    for name, src in (("regspec.py", SPEC_SRC), ("runner.py", RUNNER_SRC)):
        for tok in FORBIDDEN:
            assert tok not in src, (name, tok)
        for s in LATER_DATES:
            assert s not in src, (name, f"later window {s}")
    for ban in COPY_BANS:
        assert ban not in RUNNER_SRC, f"runner must import, not copy: {ban}"
    assert core.scan(SPEC_SRC + "\n" + RUNNER_SRC + "\n" + SP.HYPOTHESIS) == []


def test_cuda_parallel_cache_and_single_axis_params():
    assert SP.fit_params("cuda")["device"] == "cuda" and "SP.fit_params" in RUNNER_SRC
    assert "ThreadPoolExecutor" in RUNNER_SRC
    assert "fcache_key" in RUNNER_SRC and "featcache" in RUNNER_SRC
    assert "set(probe) == set(ia.PARAMS)" in RUNNER_SRC
    assert "fitted_choice" in RUNNER_SRC
    assert SP.PERIOD_JOBS * SP.FIT_NJOBS <= 10
    p = SP.fit_params("cuda")
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k in ("n_jobs", "random_state"):
            continue
        assert p[k] == v, f"param {k} differs from E0001 recipe"
    assert p["n_jobs"] == 2 and p["random_state"] == 0


def test_no_selection_fed_choice_and_no_official_write():
    assert "predictions_official" not in RUNNER_SRC
    assert "official" not in RUNNER_SRC.lower()
    assert "freeze" not in RUNNER_SRC
    assert "X0008" in RUNNER_SRC
    assert "dev only" in SP.SELECT_RULE or "dev-only" in SP.SELECT_RULE or "dev\n" in SP.SELECT_RULE
    assert "selection/audit" in SP.SELECT_RULE and "never read" in SP.SELECT_RULE


def test_scratch_isolation_and_owner():
    assert "store/scratch/worker-d88" in RUNNER_SRC
    assert "worker-d87" not in RUNNER_SRC and "worker-d89" not in RUNNER_SRC
    assert SP.DIRECTION == "D88" and SP.OWNER == "worker-d88"
    assert "gpu1" in SP.GPU_REASON and "D87" in SP.GPU_REASON and "D89" in SP.GPU_REASON


def test_output_contract_keys_present():
    for key in ["predictions_{dev,selection,audit}", "poison.json",
                "selection.json", "importance.json", '"passed"']:
        assert key in RUNNER_SRC, key
    assert "from existing_all import" in RUNNER_SRC and "write(te," in RUNNER_SRC
    assert "route_predict" in RUNNER_SRC and "fit_regime_heads" in RUNNER_SRC
    assert "positive_control" in RUNNER_SRC and "regime_invariant" in RUNNER_SRC
