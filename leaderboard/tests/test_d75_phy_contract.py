"""D75 derived-physics contract: preregistered constants, causality, single-axis."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1] / "experiments" / "d75_physics"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "headroom_audit"))
import phyfeatures as PF  # noqa: E402
import physpec as PS  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_double_tag_front():
    assert PS.FAMILY_TAG == "[family=xgb]"
    assert PS.AXES_TAG.startswith("[axes:") and "feat=physics" in PS.AXES_TAG
    assert PS.HYPOTHESIS.startswith("[family=xgb] [axes:")
    for token in ("E0001", "E0003", "AXES.md", "physics", "X0104"):
        assert token in PS.HYPOTHESIS, f"hypothesis must cite {token}"
    for token in ("Alfven", "dynamic pressure", "plasma beta"):
        assert token in PS.HYPOTHESIS, f"hypothesis must define {token}"
    assert "freeze" in PS.HYPOTHESIS.lower() and "official-score" in PS.HYPOTHESIS
    assert "6h" in PS.HYPOTHESIS and "gpu1" in PS.HYPOTHESIS and "cuda" in PS.HYPOTHESIS
    assert PS.check_spec()["n_phy"] == 10


def test_constants_frozen_and_params_e0003():
    assert (PS.MU0, PS.MP, PS.KB) == (1.25663706127e-6, 1.67262192369e-27, 1.380649e-23)
    assert PS.FFILL_LIMIT == 6 and PS.MEAN_WIN == 24 and PS.MISS_WIN == 192
    assert PS.PURGE_H == 72 and PS.SEED == 0
    assert len(PS.PHY_COLS) == 10 and len(set(PS.PHY_COLS)) == 10
    assert PS.FULL_RULE and PS.SELECT_RULE
    p = PS.fit_params()
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["tree_method"] == "hist"
    assert p["device"] == "cuda", "GPU build: device=cuda required"
    assert p["n_jobs"] == PS.FIT_NJOBS == 4


def test_threads_budget_and_gpu():
    assert PS.FIT_NJOBS == 4 and PS.PERIOD_JOBS == 3 and PS.GPU_ID == 1
    assert PS.PERIOD_JOBS * PS.FIT_NJOBS <= 20
    assert "worker-d75" in PS.GPU_REASON and "gpu1" in PS.GPU_REASON
    assert "cuda" in PS.GPU_REASON
    runner = (PKG / "runner.py").read_text()
    assert "cuda" in runner and "--prefer-gpu" not in runner  # lease flags live in the expos cmd, not code
    assert "CPU-only" not in runner and "CPU pilot" not in runner


def test_no_forbidden_strings_in_code():
    from expos import core
    blob = "".join((PKG / f).read_text() for f in ("runner.py", "physpec.py", "phyfeatures.py"))
    assert core.scan(blob) == [], f"forbidden path strings in D75 code: {core.scan(blob)}"
    assert core.scan(PS.HYPOTHESIS) == []


def _synth_frame(v=400.0, n=5.0, t=1e5, b=5.0, periods=400, miss_val=0.0):
    idx = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    frame = pd.DataFrame({"ace_speed_kms": v, "ace_density_cm3": n,
                          "ace_temperature_k": t, "ace_bt_nt": b}, index=idx)
    miss = pd.Series(miss_val, index=idx)
    return idx, frame, miss


def test_flat_series_matches_hand_computed_si_values():
    idx, frame, miss = _synth_frame()
    per = PF.origin_physics(idx[200:210], frame, miss)
    assert list(per.columns) == PS.PHY_COLS
    # hand-computed SI references for V=400km/s n=5/cm3 T=1e5K B=5nT
    assert np.allclose(per["phy_ma"].to_numpy(), 8.20, atol=0.05)
    assert np.allclose(per["phy_pdyn"].to_numpy(), 1.338, atol=0.02)
    assert np.allclose(per["phy_beta"].to_numpy(), 0.694, atol=0.02)
    assert np.allclose(per["phy_log_ma"].to_numpy(), np.log10(per["phy_ma"].to_numpy()))
    assert np.allclose(per["phy_ma_m24"].to_numpy(), per["phy_ma"].to_numpy())
    assert np.allclose(per["phy_miss192"].to_numpy(), 0.0, atol=1e-12)


def test_monotone_response_and_future_never_used():
    idx, frame, miss = _synth_frame()
    frame2 = frame.copy()
    frame2.loc[idx[300]:, "ace_speed_kms"] += 100.0  # V step at hour 300
    per_pre = PF.origin_physics(idx[200:210], frame2, miss)
    assert np.allclose(per_pre["phy_ma"].to_numpy(), 8.20, atol=0.05), "future leak into pre-step origins"
    per_post = PF.origin_physics(idx[300:302], frame2, miss)
    assert (per_post["phy_ma"].to_numpy() > 8.20).all(), "MA must rise with V"
    assert (per_post["phy_pdyn"].to_numpy() > 1.338).all(), "Pdyn must rise with V"
    # B step raises vA and lowers MA, lowers beta
    frame3 = frame.copy()
    frame3.loc[idx[300]:, "ace_bt_nt"] += 5.0
    per_b = PF.origin_physics(idx[300:302], frame3, miss)
    assert (per_b["phy_ma"].to_numpy() < 8.20).all()
    assert (per_b["phy_beta"].to_numpy() < 0.694).all()


def test_add_physics_joins_long_table():
    idx, frame, miss = _synth_frame()
    df = pd.DataFrame({"origin": list(idx[200:202]) * 2, "h": [1, 2, 1, 2]})
    out = PF.add_physics(df, frame, miss)
    for c in PS.PHY_COLS:
        assert c in out.columns
    assert len(out) == 4
    assert np.allclose(out["phy_ma"].to_numpy(), 8.20, atol=0.05)


def test_phy_poison_pre_invariant_and_positive_control():
    idx = pd.date_range("2025-01-01", periods=600, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"ace_speed_kms": 400.0 + rng.normal(0, 20, len(idx)),
                          "ace_density_cm3": 5.0 + rng.normal(0, 1, len(idx)).clip(min=0.5),
                          "ace_temperature_k": 1e5 + rng.normal(0, 2e4, len(idx)).clip(min=1e4),
                          "ace_bt_nt": 5.0 + rng.normal(0, 1, len(idx)).clip(min=0.5)}, index=idx)
    miss = pd.Series(0.0, index=idx)
    origins = idx[100:300]
    res = PF.phy_poison_test(origins, frame, miss, "2025-01-09T00:00:00Z")
    assert res["passed"] is True and res["pre_invariant"] is True and len(res["moved_columns"]) > 0


def test_full_rule_prespecified_and_report_only():
    assert "full 3h" in PS.FULL_RULE and "E0001" in PS.FULL_RULE
    assert "F1" in PS.FULL_RULE and "0.30" in PS.FULL_RULE
    assert "no full run" in PS.FULL_RULE
    assert "freeze" in PS.FULL_RULE.lower() and "official-score" in PS.FULL_RULE
    assert "E0003" in PS.FULL_RULE and "seed-repro" in PS.FULL_RULE
    assert "poison" in PS.FULL_RULE and "1/3" in PS.FULL_RULE


def test_owned_files_and_no_vendored_strips():
    files = sorted(f.name for f in PKG.glob("*.py"))
    assert len(files) <= 5, files
    assert {"physpec.py", "runner.py", "phyfeatures.py"} <= set(files)
    runner = (PKG / "runner.py").read_text()
    assert "import strips" in runner or "import ST" in runner
    assert "import physpec" in runner or "import PS" in runner
    assert "store/scratch/worker-d75" in runner
    for other in ("worker-d69", "worker-d70", "worker-d71", "worker-d72", "worker-d73",
                  "worker-d74", "worker-d59", "worker-d60", "worker-d61", "worker-d45", "worker-d33"):
        assert other not in runner
    for f in PKG.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"


def test_runner_physics_poison_gpu_cache_and_smoke():
    runner = (PKG / "runner.py").read_text()
    assert "PF.add_physics" in runner
    assert "PF.phy_poison_test" in runner and '"passed"' in runner
    assert "SystemExit" in runner
    assert "ThreadPoolExecutor" in runner
    assert "max_workers=PS.PERIOD_JOBS" in runner
    assert "ST.build_manifest" in runner
    assert "ST.poison_test" in runner
    assert "phy_features.json" in runner
    assert "fcache" in runner.lower() and "featcache" in runner
    assert "EXPOS_SMOKE" in runner
    assert "design_cols" in runner and "dict.fromkeys" in runner
    assert "X0104" in runner
    assert "device" in runner and "cuda" in runner
    assert "write(te, pred, name)" in runner
    assert "official" not in runner.lower(), "runner must not touch the frozen window"
