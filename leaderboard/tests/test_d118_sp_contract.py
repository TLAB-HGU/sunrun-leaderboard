"""D118 speed-persist contract: synthetic CPU-only checks, no data reads.

Covers the assignment acceptance surface without running the pipeline:
dual-tag hypothesis with AXES.md feat-section citation, frozen trailing
persistence math (known-answer + bounds), trailing-only truncation
invariance on synthetic series, unused-first verification (levels/means/
stds/lags only, longwin/wavelet/subband/imf-persist/shock distinct,
repo-wide spdp_ grep empty), frozen spec (threads/devices/freq/purge),
forbidden-source hygiene, 3-period parallel + hash-keyed cache discipline,
scratch isolation and the expos output contract keys.
"""
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

LB = Path(__file__).resolve().parents[1]
D118 = LB / "experiments" / "d118_spdpersist"
sys.path.insert(0, str(D118))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import spdspec as SPEC  # noqa: E402
import spdpfeatures as SP  # noqa: E402
from expos import core  # noqa: E402

RUNNER_SRC = (D118 / "runner.py").read_text()
SPEC_SRC = (D118 / "spdspec.py").read_text()
FEAT_SRC = (D118 / "spdpfeatures.py").read_text()

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
    assert SPEC.AXES_TAG.startswith("[axes:") and "feat=" in SPEC.AXES_TAG
    assert "speed-persist" in SPEC.AXES_TAG
    assert SPEC.HYPOTHESIS.startswith("[family=xgb] [axes: feat=speed-persist]")
    for token in ("E0001", "X0002", "X0008", "X0043", "research/AXES.md"):
        assert token in SPEC.HYPOTHESIS, f"hypothesis must cite {token}"
    assert SPEC.AXES_QUOTE == "multi-scale rolling stats (3h..27d)"
    assert SPEC.AXES_QUOTE in SPEC.HYPOTHESIS
    assert "spdp_" in SPEC.HYPOTHESIS and "causal" in SPEC.HYPOTHESIS
    assert "gpu1" in SPEC.HYPOTHESIS and "60s" in SPEC.HYPOTHESIS
    assert "screening" in SPEC.HYPOTHESIS
    assert "No freeze" in SPEC.HYPOTHESIS and "official-score" in SPEC.HYPOTHESIS
    assert SPEC.check_spec()["experience"] == ["X0002", "X0008", "X0043"]


def test_frozen_persistence_constants():
    assert SPEC.RANGE_WINS == (24, 72) and SPEC.STABLE_WINS == (24, 72)
    assert SPEC.STABLE_TOL == 30.0 and SPEC.JUMP_LEVEL == 80.0
    assert SPEC.AGE_CAP == 168.0 and SPEC.AGE_LOOKBACK == 168
    assert SPEC.TREND_WIN == 72 and SPEC.JUMP_WIN == 24 and SPEC.UPFRAC_WIN == 24
    assert len(SPEC.SPDP_COLS) == 8 and len(set(SPEC.SPDP_COLS)) == 8
    assert all(c.startswith("spdp_") for c in SPEC.SPDP_COLS)
    assert SP.spdp_columns() == list(SPEC.SPDP_COLS)


def _synthetic_speed(n=400, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    return pd.Series(400.0 + rng.normal(0, 8, n), index=idx)


def test_known_answer():
    # 70 x 400.0 then a hand-checkable tail; origin = last hour (v0 = 500).
    tail = [400.0, 400.0, 410.0, 420.0, 420.0, 420.0, 500.0, 500.0, 500.0, 500.0]
    vals = [400.0] * 70 + tail
    idx = pd.date_range("2025-01-01", periods=len(vals), freq="h", tz="UTC")
    speed = pd.Series(vals, index=idx)
    got = SP.spdp_features(pd.DatetimeIndex([idx[-1]]), speed=speed).iloc[0]
    assert got["spdp_range24"] == pytest.approx(100.0)
    assert got["spdp_range72"] == pytest.approx(100.0)
    assert got["spdp_stable24"] == pytest.approx(4.0 / 24.0)
    assert got["spdp_stable72"] == pytest.approx(4.0 / 72.0)
    assert got["spdp_trend72"] == pytest.approx(500.0 - (62 * 400.0 + 4470.0) / 72.0)
    assert got["spdp_jump24"] == pytest.approx(80.0)
    assert got["spdp_age80"] == pytest.approx(3.0)
    assert got["spdp_upfrac24"] == pytest.approx(3.0 / 24.0)


def test_truncation_invariance_and_row_contract():
    speed = _synthetic_speed()
    origin = speed.index[300]
    full = SP.spdp_features(pd.DatetimeIndex([origin]), speed=speed)
    assert list(full.columns) == list(SPEC.SPDP_COLS) and len(full) == 1
    row = full.iloc[0]
    assert np.isfinite(row.to_numpy(dtype=float)).all()
    assert row["spdp_range24"] >= 0.0 and row["spdp_range72"] >= row["spdp_range24"] - 1e-9
    assert 0.0 <= row["spdp_stable24"] <= 1.0 and 0.0 <= row["spdp_stable72"] <= 1.0
    assert 0.0 <= row["spdp_upfrac24"] <= 1.0
    assert 0.0 <= row["spdp_age80"] <= 168.0 and row["spdp_jump24"] >= 0.0
    # truncating the series at the origin leaves the origin row bitwise identical.
    trunc = SP.spdp_features(pd.DatetimeIndex([origin]), speed=speed.iloc[:301])
    assert np.allclose(full.to_numpy(float), trunc.to_numpy(float), equal_nan=True)
    # the dataset-start edge never raises: single-point stables are 1.0, the
    # other six columns are NaN (documented warmup semantics, XGB-missing).
    early = SP.spdp_features(pd.DatetimeIndex([speed.index[0]]), speed=speed)
    assert early["spdp_stable24"].iloc[0] == 1.0 and early["spdp_stable72"].iloc[0] == 1.0
    assert early.drop(columns=["spdp_stable24", "spdp_stable72"]).isna().to_numpy().all()
    # pure-math split agrees with the loader path on coverage helper input.
    cov = SP.spdp_coverage([origin], speed)
    assert cov == {"n": 1, "covered": 1, "frac": 1.0}


def test_unused_vs_levels_and_repo():
    for src_name in ("headroom.py", "info_ablation.py"):
        src = (LB / "experiments" / "headroom_audit" / src_name).read_text()
        assert "spdp_" not in src, src_name
    base = (LB / "experiments" / "expos_runners" / "existing_all.py").read_text()
    assert "spdp_" not in base, "existing_all must not carry persist columns"
    for rel in ("d40_longwin/longwin.py", "d74_wavelet/wavspec.py",
                "d110_subb/sbspec.py", "d95_imf/imfspec.py",
                "d7_shock_precursor/shock.py"):
        src = (LB / "experiments" / rel).read_text()
        assert "spdp_" not in src, rel
        for c in SPEC.SPDP_COLS:
            assert c not in src, (rel, c)
    hits = []
    for p in (LB / "experiments").rglob("*.py"):
        if "d118_spdpersist" in p.parts or "test_d118" in p.name:
            continue
        t = p.read_text(errors="ignore")
        if "spdp_" in t:
            hits.append(str(p))
    assert hits == [], f"spdp_ columns already used: {hits}"
    # the English axis phrase may be cross-referenced by the sibling triple
    # leg (D117 names the same gpu split); novelty is the column namespace,
    # so phrase hits must define no spdp_ column.
    for p in (LB / "experiments").rglob("*.py"):
        if "d118_spdpersist" in p.parts or "test_d118" in p.name:
            continue
        t = p.read_text(errors="ignore")
        if "speed-persist" in t:
            assert "spdp_" not in t, f"speed-persist implementation outside D118: {p}"


def test_frozen_spec_budget_and_device():
    spec = SPEC.check_spec()
    assert spec["n_spdp"] == 8 and spec["family"] == "[family=xgb]"
    assert spec["axes"] == "[axes: feat=speed-persist]"
    assert spec["threads"] == 6 and SPEC.PERIOD_JOBS * SPEC.FIT_NJOBS <= 10
    assert SPEC.SEED == 0 and SPEC.N_ESTIMATORS == 600 and SPEC.PURGE_HOURS == 72
    assert SPEC.FREQ_PILOT == "6h" and SPEC.FREQ_FULL == "3h"
    assert SPEC.resolve_device() == "cpu"  # no lease env in test -> foreground fallback
    assert "D114" in SPEC.GPU_REASON and "D118" in SPEC.GPU_REASON and "D117" in SPEC.GPU_REASON
    assert "gpu0" in SPEC.GPU_REASON and "gpu1" in SPEC.GPU_REASON and "CPU" in SPEC.GPU_REASON
    assert "60s" in SPEC.GPU_REASON and "EXPOS_GPUS" in SPEC.GPU_REASON
    p = SPEC.fit_params()
    assert p["n_jobs"] <= SPEC.FIT_NJOBS and p["tree_method"] == "hist"
    assert p.get("objective", "reg:squarederror") == "reg:squarederror"
    assert "tweedie_variance_power" not in p


def test_no_forbidden_sources_or_copies():
    for name, src in (("spdspec.py", SPEC_SRC), ("spdpfeatures.py", FEAT_SRC),
                      ("runner.py", RUNNER_SRC)):
        for tok in FORBIDDEN:
            assert tok not in src, (name, tok)
        for s in LATER_DATES:
            assert s not in src, (name, f"later window {s}")
    for ban in COPY_BANS:
        assert ban not in RUNNER_SRC, f"runner must import, not copy: {ban}"
    assert "import info_ablation" in RUNNER_SRC and "from existing_all import" in RUNNER_SRC
    assert core.scan(SPEC_SRC + "\n" + FEAT_SRC + "\n" + RUNNER_SRC + "\n"
                     + SPEC.HYPOTHESIS) == []


def test_parallel_cache_and_budget():
    assert "ThreadPoolExecutor" in RUNNER_SRC
    assert "max_workers=PS.PERIOD_JOBS" in RUNNER_SRC
    assert "fcache_key" in RUNNER_SRC and "featcache" in RUNNER_SRC
    assert "raw_ace" in RUNNER_SRC and "_origins_sha" in RUNNER_SRC
    assert "write(" in RUNNER_SRC and "SP.add_spdp" in RUNNER_SRC
    assert '"passed"' in RUNNER_SRC and "positive_control" in RUNNER_SRC
    assert "exact_index_validated" in RUNNER_SRC and "prediction_invariant" in RUNNER_SRC
    assert "nan_mask_validated" in RUNNER_SRC
    assert "EXPOS_SMOKE" in RUNNER_SRC
    assert "PS.resolve_device" in RUNNER_SRC and "device=device" in RUNNER_SRC
    assert "independent_of_E0003" in RUNNER_SRC


def test_full_rule_and_selection_discipline():
    assert "predictions_official" not in RUNNER_SRC
    assert "official" not in RUNNER_SRC.lower()
    assert "freeze" not in RUNNER_SRC
    assert "no selection/audit statistic" in SPEC.SELECT_RULE
    assert "dev/OOF only" in SPEC.SELECT_RULE
    assert "72h purge" in SPEC.SELECT_RULE
    assert ">=1/3" in SPEC.FULL_RULE and "F1>=0.30" in SPEC.FULL_RULE
    assert "screening only" in SPEC.FULL_RULE and "no full run" in SPEC.FULL_RULE
    assert "3h density" in SPEC.FULL_RULE
    assert "no freeze" in SPEC.FULL_RULE and "official-score" in SPEC.FULL_RULE


def test_scratch_isolation_and_owner():
    assert "store/scratch/worker-d118" in RUNNER_SRC
    assert "worker-d114" not in RUNNER_SRC and "worker-d117" not in RUNNER_SRC
    assert "worker-d116" not in RUNNER_SRC
    assert SPEC.DIRECTION == "D118" and SPEC.OWNER == "worker-d118"


def test_output_contract_keys_present():
    for key in ["predictions_{dev,selection,audit}", "poison.json",
                "selection.json", "importance.json", "manifest.json", '"passed"']:
        assert key in RUNNER_SRC, key
    assert "n_spdp_cols" in RUNNER_SRC


def test_owned_dir_is_small():
    files = sorted(p.name for p in D118.iterdir() if p.suffix == ".py")
    assert len(files) <= 5, files
    assert {"spdspec.py", "spdpfeatures.py", "runner.py"} <= set(files)
