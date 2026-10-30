"""D116 shock-normal contract: synthetic CPU-only checks, no data reads.

Covers the assignment acceptance surface without running the pipeline:
dual-tag hypothesis with AXES.md feat-section citation, frozen coplanarity
math (known-answer + fallback + properties), trailing-only truncation
invariance on synthetic series, unused-first verification (D7 magnitudes /
scalar levels / d95 persistence disjoint, repo-wide name grep empty),
frozen spec (threads/devices/freq/purge), forbidden-source hygiene,
3-period parallel + hash-keyed cache discipline, scratch isolation and the
expos output contract keys.
"""
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

LB = Path(__file__).resolve().parents[1]
D116 = LB / "experiments" / "d116_shnorm"
sys.path.insert(0, str(D116))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
sys.path.insert(0, str(LB / "experiments" / "d7_shock_precursor"))

import snspec as SPEC  # noqa: E402
import shnfeatures as SH  # noqa: E402
import shock as D7  # noqa: E402  (read-only: name-disjointness check)
from expos import core  # noqa: E402

RUNNER_SRC = (D116 / "runner.py").read_text()
SPEC_SRC = (D116 / "snspec.py").read_text()
FEAT_SRC = (D116 / "shnfeatures.py").read_text()

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
    assert "shock-normal" in SPEC.AXES_TAG
    assert SPEC.HYPOTHESIS.startswith("[family=xgb] [axes: feat=shock-normal]")
    for token in ("E0001", "X0008", "X0002", "X0043", "research/AXES.md"):
        assert token in SPEC.HYPOTHESIS, f"hypothesis must cite {token}"
    assert SPEC.AXES_QUOTE == "change-point scores"
    assert SPEC.AXES_QUOTE in SPEC.HYPOTHESIS
    assert "D7" in SPEC.HYPOTHESIS and "coplanarity" in SPEC.HYPOTHESIS
    assert "gpu1" in SPEC.HYPOTHESIS and "60s" in SPEC.HYPOTHESIS
    assert "screening" in SPEC.HYPOTHESIS
    assert "No freeze" in SPEC.HYPOTHESIS and "official-score" in SPEC.HYPOTHESIS
    assert SPEC.check_spec()["experience"] == ["X0008", "X0002", "X0043"]


def test_frozen_geometry_windows():
    assert (SPEC.SCAN_HOURS, SPEC.DOWNSTREAM_HOURS) == (72, 2)
    assert (SPEC.UPSTREAM_HOURS, SPEC.PRIOR_MEAN_HOURS) == (4, 3)
    assert len(SPEC.SHN_COLS) == 9 and len(set(SPEC.SHN_COLS)) == 9
    assert all(c.startswith("shn_") for c in SPEC.SHN_COLS)
    assert SH.shn_columns() == list(SPEC.SHN_COLS)


def test_coplanar_known_answer():
    # 90-degree rotation: Bu=(0,5,0) upstream, Bd=(0,0,5) downstream.
    # cr = Bd x Bu = (-25,0,0); d = (0,-5,5); n = cr x d = (0,125,125) -> (0,1,1)/sqrt2.
    n, ok = SH.coplanar_normal(np.array([0.0, 5.0, 0.0]), np.array([0.0, 0.0, 5.0]))
    assert ok is True
    assert np.allclose(n, np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0))
    assert abs(np.linalg.norm(n) - 1.0) < 1e-12 and n[0] >= 0.0


def test_coplanar_fallback_and_sign():
    # pure compression, no rotation: degenerate -> radial fallback, ok False.
    n, ok = SH.coplanar_normal(np.array([5.0, 0.0, 0.0]), np.array([8.0, 0.0, 0.0]))
    assert ok is False
    assert np.allclose(n, np.array([1.0, 0.0, 0.0]))
    # NaN input is also degenerate, never raises.
    n2, ok2 = SH.coplanar_normal(np.array([np.nan, 0.0, 0.0]), np.array([8.0, 0.0, 0.0]))
    assert ok2 is False and np.allclose(n2, np.array([1.0, 0.0, 0.0]))
    # sunward sign convention holds on seeded random vectors.
    rng = np.random.default_rng(0)
    for _ in range(50):
        bu, bd = rng.normal(0, 5, 3), rng.normal(0, 5, 3)
        n3, _ = SH.coplanar_normal(bu, bd)
        assert abs(np.linalg.norm(n3) - 1.0) < 1e-9 and n3[0] >= 0.0


def _synthetic_frame(n=400, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    b = rng.normal(0, 4, (n, 3))
    b[200, :] += np.array([6.0, -4.0, 3.0])  # one sharp discontinuity
    df = pd.DataFrame(b, index=idx, columns=SH.BVEC)
    df["timestamp_utc"] = df.index
    return df


def test_truncation_invariance_and_row_contract():
    frame = _synthetic_frame()
    origin = frame.index[300]
    full = SH.normals_at(frame.index, frame[SH.BVEC].to_numpy(float),
                         pd.DatetimeIndex([origin]))
    assert list(full.columns) == list(SPEC.SHN_COLS) and len(full) == 1
    row = full.iloc[0]
    assert np.isfinite(row[["shn_theta_bn", "shn_lon", "shn_lat", "shn_cone",
                            "shn_tau_age", "shn_rot"]]).all()
    assert 0.0 <= row["shn_theta_bn"] <= 180.0
    assert -180.0 <= row["shn_lon"] <= 180.0 and -90.0 <= row["shn_lat"] <= 90.0
    assert 0.0 <= row["shn_cone"] <= 180.0 and row["shn_coplanar_ok"] in (0.0, 1.0)
    # truncating the frame at the origin leaves the origin row bitwise identical.
    cut = frame.iloc[:301]
    trunc = SH.normals_at(cut.index, cut[SH.BVEC].to_numpy(float),
                          pd.DatetimeIndex([origin]))
    assert np.allclose(full.to_numpy(float), trunc.to_numpy(float), equal_nan=True)
    # early-history origins with no 76h lookback are NaN rows, never raise.
    early = SH.normals_at(frame.index, frame[SH.BVEC].to_numpy(float),
                           pd.DatetimeIndex([frame.index[10]]))
    assert early.isna().to_numpy().all()


def test_unused_vs_d7_levels_and_repo():
    assert not set(SPEC.SHN_COLS) & set(D7.SHOCK_COLS), "orientation must be new vs D7 magnitudes"
    assert not set(SPEC.SHN_COLS) & {"shock_" + c for c in SPEC.SHN_COLS}
    for src_name in ("headroom.py", "info_ablation.py"):
        src = (LB / "experiments" / "headroom_audit" / src_name).read_text()
        assert "shn_" not in src and "theta_bn" not in src and "coplanar" not in src, src_name
    base = (LB / "experiments" / "expos_runners" / "existing_all.py").read_text()
    assert "shn_" not in base and "theta_bn" not in base, "existing_all must not carry normals"
    hits = []
    for p in (LB / "experiments").rglob("*.py"):
        if "d116_shnorm" in p.parts or "test_d116" in p.name:
            continue
        t = p.read_text(errors="ignore")
        if "theta_bn" in t or "coplanar" in t or "shn_" in t or "shock_normal" in t:
            hits.append(str(p))
    assert hits == [], f"shock-normal already used: {hits}"


def test_frozen_spec_budget_and_device():
    spec = SPEC.check_spec()
    assert spec["n_shn"] == 9 and spec["family"] == "[family=xgb]"
    assert spec["axes"] == "[axes: feat=shock-normal]"
    assert spec["threads"] == 6 and SPEC.PERIOD_JOBS * SPEC.FIT_NJOBS <= 10
    assert SPEC.SEED == 0 and SPEC.N_ESTIMATORS == 600 and SPEC.PURGE_HOURS == 72
    assert SPEC.FREQ_PILOT == "6h" and SPEC.FREQ_FULL == "3h"
    assert SPEC.resolve_device() == "cpu"  # no lease env in test -> foreground fallback
    assert "D114" in SPEC.GPU_REASON and "D116" in SPEC.GPU_REASON and "D115" in SPEC.GPU_REASON
    assert "gpu0" in SPEC.GPU_REASON and "gpu1" in SPEC.GPU_REASON and "CPU" in SPEC.GPU_REASON
    assert "60s" in SPEC.GPU_REASON and "EXPOS_GPUS" in SPEC.GPU_REASON
    p = SPEC.fit_params()
    assert p["n_jobs"] <= SPEC.FIT_NJOBS and p["tree_method"] == "hist"
    assert p.get("objective", "reg:squarederror") == "reg:squarederror"
    assert "tweedie_variance_power" not in p


def test_no_forbidden_sources_or_copies():
    for name, src in (("snspec.py", SPEC_SRC), ("shnfeatures.py", FEAT_SRC),
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
    assert "write(" in RUNNER_SRC and "SH.add_shn" in RUNNER_SRC
    assert '"passed"' in RUNNER_SRC and "positive_control" in RUNNER_SRC
    assert "pre_invariant" not in RUNNER_SRC  # d116 uses exact_index/nan_mask/prediction_invariant keys
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
    assert "store/scratch/worker-d116" in RUNNER_SRC
    assert "worker-d114" not in RUNNER_SRC and "worker-d115" not in RUNNER_SRC
    assert "worker-d113" not in RUNNER_SRC
    assert SPEC.DIRECTION == "D116" and SPEC.OWNER == "worker-d116"


def test_output_contract_keys_present():
    for key in ["predictions_{dev,selection,audit}", "poison.json",
                "selection.json", "importance.json", "manifest.json", '"passed"']:
        assert key in RUNNER_SRC, key
    assert "n_shn_cols" in RUNNER_SRC


def test_owned_dir_is_small():
    files = sorted(p.name for p in D116.iterdir() if p.suffix == ".py")
    assert len(files) <= 5, files
    assert {"snspec.py", "shnfeatures.py", "runner.py"} <= set(files)
