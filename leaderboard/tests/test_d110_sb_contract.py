"""D110 Fourier-subband contract: preregistered constants, causality, single-axis."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1] / "experiments" / "d110_subb"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "headroom_audit"))
import sbfeatures as SF  # noqa: E402
import sbspec as SB  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_double_tag_front():
    assert SB.FAMILY_TAG == "[family=xgb]"
    assert SB.AXES_TAG.startswith("[axes:") and "feat=subband" in SB.AXES_TAG
    assert SB.HYPOTHESIS.startswith("[family=xgb] [axes:")
    for token in ("E0001", "E0003", "AXES.md", "subband", "X0128", "Fourier"):
        assert token in SB.HYPOTHESIS, f"hypothesis must cite {token}"
    assert "X0104" in SB.HYPOTHESIS
    assert "freeze" in SB.HYPOTHESIS.lower() and "official-score" in SB.HYPOTHESIS
    assert "6h" in SB.HYPOTHESIS and "gpu1" in SB.HYPOTHESIS
    assert SB.check_spec()["n_sb"] == 8


def test_constants_frozen_and_params_e0003():
    assert SB.WIN == 192 and SB.MISS_WIN == 192 and SB.MIN_FINITE == 48
    assert SB.BAND_EDGES_H == (48, 12, 3)
    assert SB.PURGE_H == 72 and SB.SEED == 0
    assert len(SB.SB_COLS) == 8 and len(set(SB.SB_COLS)) == 8
    assert SB.FULL_RULE and SB.SELECT_RULE
    p = SB.fit_params("cuda")
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["tree_method"] == "hist"
    assert str(p["device"]).startswith("cuda"), "GPU device required"


def test_threads_budget_and_gpu():
    assert SB.FIT_NJOBS == 2 and SB.PERIOD_JOBS == 3
    assert SB.PERIOD_JOBS * SB.FIT_NJOBS <= 20
    assert "gpu1" in SB.GPU_REASON and "worker-d110" in SB.GPU_REASON
    for name in ("sbspec.py", "runner.py", "sbfeatures.py"):
        text = (PKG / name).read_text()
        assert "datasets--tlabtlab" not in text
    runner = (PKG / "runner.py").read_text()
    assert "cuda" in runner
    assert SB.device_for(0).startswith("cuda")


def test_no_forbidden_strings_in_code():
    from expos import core
    blob = "".join((PKG / f).read_text() for f in ("runner.py", "sbspec.py", "sbfeatures.py"))
    assert core.scan(blob) == [], f"forbidden path strings in D110 code: {core.scan(blob)}"
    assert core.scan(SB.HYPOTHESIS) == []


def _synth_series():
    idx = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
    speed = pd.Series(400.0, index=idx)  # flat baseline
    miss = pd.Series(0.0, index=idx)
    return idx, speed, miss


def test_flat_series_zero_var_and_fractions():
    idx, speed, miss = _synth_series()
    per = SF.origin_subband(idx[200:210], speed, miss)
    assert list(per.columns) == SB.SB_COLS
    assert np.allclose(per["sb_var192"].to_numpy(), 0.0, atol=1e-9)
    for c in ("sb_low", "sb_mid", "sb_high", "sb_vhigh"):
        assert np.allclose(per[c].to_numpy(), 0.0, atol=1e-9)
    assert np.allclose(per["sb_miss192"].to_numpy(), 0.0, atol=1e-12)


def test_step_series_future_never_used_and_fractions_sum():
    idx, speed, miss = _synth_series()
    speed2 = speed.copy()
    speed2.loc[idx[300]:] += 100.0  # step at hour 300
    per_pre = SF.origin_subband(idx[200:210], speed2, miss)
    assert np.allclose(per_pre["sb_var192"].to_numpy(), 0.0, atol=1e-9), "future leak into variance"
    for c in ("sb_low", "sb_mid", "sb_high", "sb_vhigh"):
        assert np.allclose(per_pre[c].to_numpy(), 0.0, atol=1e-9), f"future leak at {c}"
    per_post = SF.origin_subband(idx[300:302], speed2, miss)
    assert float(per_post["sb_var192"].iloc[0]) > 1.0
    fr = per_post[["sb_low", "sb_mid", "sb_high", "sb_vhigh"]].to_numpy()
    assert np.allclose(fr.sum(1), 1.0, atol=1e-6)
    assert per_post["sb_high"].iloc[0] + per_post["sb_vhigh"].iloc[0] > 0.05


def test_add_subband_joins_long_table():
    idx, speed, miss = _synth_series()
    df = pd.DataFrame({"origin": list(idx[200:202]) * 2, "h": [1, 2, 1, 2]})
    out = SF.add_subband(df, speed, miss)
    for c in SB.SB_COLS:
        assert c in out.columns
    assert len(out) == 4


def test_sb_poison_pre_invariant_and_positive_control():
    idx = pd.date_range("2025-01-01", periods=600, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    speed = pd.Series(400.0 + rng.normal(0, 20, len(idx)), index=idx)
    miss = pd.Series(0.0, index=idx)
    origins = idx[100:300]
    res = SF.sb_poison_test(origins, speed, miss, "2025-01-09T00:00:00Z")
    assert res["passed"] is True and res["pre_invariant"] is True and len(res["moved_columns"]) > 0


def test_full_rule_prespecified_and_report_only():
    assert "full 3h" in SB.FULL_RULE and "E0001" in SB.FULL_RULE
    assert "F1" in SB.FULL_RULE and "0.30" in SB.FULL_RULE
    assert "no full run" in SB.FULL_RULE
    assert "freeze" in SB.FULL_RULE.lower() and "official-score" in SB.FULL_RULE
    assert "E0003" in SB.FULL_RULE and "seed-repro" in SB.FULL_RULE
    assert "poison" in SB.FULL_RULE and "1/3" in SB.FULL_RULE
    assert "0.41121495" in SB.FULL_RULE


def test_owned_files_and_no_vendored_strips():
    files = sorted(f.name for f in PKG.glob("*.py"))
    assert len(files) <= 5, files
    assert {"sbspec.py", "runner.py", "sbfeatures.py"} <= set(files)
    runner = (PKG / "runner.py").read_text()
    assert "import strips" in runner or "import ST" in runner
    assert "import sbspec" in runner or "import SB" in runner
    assert "store/scratch/worker-d110" in runner
    for other in ("worker-d108", "worker-d109", "worker-d74", "worker-d75",
                  "worker-d104", "worker-d100", "worker-d105"):
        assert other not in runner
    for f in PKG.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"


def test_runner_subband_poison_gpu_cache_and_smoke():
    runner = (PKG / "runner.py").read_text()
    assert "SF.add_subband" in runner
    assert "SF.sb_poison_test" in runner and '"passed"' in runner
    assert "SystemExit" in runner
    assert "ThreadPoolExecutor" in runner
    assert "max_workers=SB.PERIOD_JOBS" in runner
    assert "ST.build_manifest" in runner
    assert "ST.poison_test" in runner
    assert "sb_features.json" in runner
    assert "fcache" in runner.lower() and "featcache" in runner
    assert "EXPOS_SMOKE" in runner
    assert "design_cols" in runner and "dict.fromkeys" in runner
    assert "X0128" in runner
    assert "write(te, pred, name)" in runner
    assert "official" not in runner.lower(), "runner must not touch the frozen window"
