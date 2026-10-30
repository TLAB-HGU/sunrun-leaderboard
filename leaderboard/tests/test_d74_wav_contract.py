"""D74 ACE-wavelet contract: preregistered constants, causality, single-axis."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1] / "experiments" / "d74_wavelet"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "headroom_audit"))
import wavfeatures as WF  # noqa: E402
import wavspec as WS  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_double_tag_front():
    assert WS.FAMILY_TAG == "[family=xgb]"
    assert WS.AXES_TAG.startswith("[axes:") and "feat=wavelet" in WS.AXES_TAG
    assert WS.HYPOTHESIS.startswith("[family=xgb] [axes:")
    for token in ("E0001", "E0003", "AXES.md", "wavelet", "X0104"):
        assert token in WS.HYPOTHESIS, f"hypothesis must cite {token}"
    assert "freeze" in WS.HYPOTHESIS.lower() and "official-score" in WS.HYPOTHESIS
    assert "6h" in WS.HYPOTHESIS and "CPU" in WS.HYPOTHESIS
    assert WS.check_spec()["n_wav"] == 15


def test_constants_frozen_and_params_e0003():
    assert WS.WAV_SCALES == (3, 6, 12, 24, 48, 96)
    assert WS.WIN_MAX == 192 and WS.MISS_WIN == 192
    assert WS.PURGE_H == 72 and WS.SEED == 0
    assert len(WS.WAV_COLS) == 15 and len(set(WS.WAV_COLS)) == 15
    assert WS.FULL_RULE and WS.SELECT_RULE
    p = WS.fit_params()
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["tree_method"] == "hist"
    assert "device" not in p, "CPU-only: no device key"


def test_threads_budget_and_cpu():
    assert WS.FIT_NJOBS == 2 and WS.PERIOD_JOBS == 3
    assert WS.PERIOD_JOBS * WS.FIT_NJOBS <= 20
    assert "CPU" in WS.CPU_REASON and "worker-d74" in WS.CPU_REASON
    for name in ("wavspec.py", "runner.py", "wavfeatures.py"):
        text = (PKG / name).read_text()
        assert "cuda" not in text, f"{name} must stay CPU-only"
    runner = (PKG / "runner.py").read_text()
    assert "--gpu" not in runner


def test_no_forbidden_strings_in_code():
    from expos import core
    blob = "".join((PKG / f).read_text() for f in ("runner.py", "wavspec.py", "wavfeatures.py"))
    assert core.scan(blob) == [], f"forbidden path strings in D74 code: {core.scan(blob)}"
    assert core.scan(WS.HYPOTHESIS) == []


def _synth_series():
    idx = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
    speed = pd.Series(400.0, index=idx)  # flat baseline
    miss = pd.Series(0.0, index=idx)
    return idx, speed, miss


def test_flat_series_zero_detail_and_energy():
    idx, speed, miss = _synth_series()
    per = WF.origin_wavelet(idx[200:210], speed, miss)
    assert list(per.columns) == WS.WAV_COLS
    for s in WS.WAV_SCALES:
        assert np.allclose(per[f"wav_d{s:03d}"].to_numpy(), 0.0, atol=1e-9)
        assert np.allclose(per[f"wav_e{s:03d}"].to_numpy(), 0.0, atol=1e-12)
    assert np.allclose(per["wav_std024"].to_numpy(), 0.0, atol=1e-9)
    assert np.allclose(per["wav_miss192"].to_numpy(), 0.0, atol=1e-12)


def test_step_series_moves_details_and_future_never_used():
    idx, speed, miss = _synth_series()
    speed2 = speed.copy()
    speed2.loc[idx[300]:] += 100.0  # step at hour 300
    # origins strictly before the step see none of it
    per_pre = WF.origin_wavelet(idx[200:210], speed2, miss)
    for s in WS.WAV_SCALES:
        assert np.allclose(per_pre[f"wav_d{s:03d}"].to_numpy(), 0.0, atol=1e-9), f"future leak at scale {s}"
    # origins just after the step carry nonzero detail/energy at small scales
    per_post = WF.origin_wavelet(idx[300:302], speed2, miss)
    assert abs(per_post["wav_d003"].iloc[0]) > 1.0
    assert per_post["wav_e003"].iloc[0] > 1.0
    assert per_post["wav_e003"].iloc[0] == per_post["wav_d003"].iloc[0] ** 2
    # energy relation holds everywhere it is finite
    per_all = WF.origin_wavelet(idx[290:320], speed2, miss)
    for s in WS.WAV_SCALES:
        d = per_all[f"wav_d{s:03d}"].to_numpy()
        e = per_all[f"wav_e{s:03d}"].to_numpy()
        assert np.allclose(e[np.isfinite(d)], d[np.isfinite(d)] ** 2)


def test_add_wavelet_joins_long_table():
    idx, speed, miss = _synth_series()
    df = pd.DataFrame({"origin": list(idx[200:202]) * 2, "h": [1, 2, 1, 2]})
    out = WF.add_wavelet(df, speed, miss)
    for c in WS.WAV_COLS:
        assert c in out.columns
    assert len(out) == 4


def test_wav_poison_pre_invariant_and_positive_control():
    idx = pd.date_range("2025-01-01", periods=600, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    speed = pd.Series(400.0 + rng.normal(0, 20, len(idx)), index=idx)
    miss = pd.Series(0.0, index=idx)
    origins = idx[100:300]
    res = WF.wav_poison_test(origins, speed, miss, "2025-01-09T00:00:00Z")
    assert res["passed"] is True and res["pre_invariant"] is True and len(res["moved_columns"]) > 0


def test_full_rule_prespecified_and_report_only():
    assert "full 3h" in WS.FULL_RULE and "E0001" in WS.FULL_RULE
    assert "F1" in WS.FULL_RULE and "0.30" in WS.FULL_RULE
    assert "no full run" in WS.FULL_RULE
    assert "freeze" in WS.FULL_RULE.lower() and "official-score" in WS.FULL_RULE
    assert "E0003" in WS.FULL_RULE and "seed-repro" in WS.FULL_RULE
    assert "poison" in WS.FULL_RULE and "1/3" in WS.FULL_RULE


def test_owned_files_and_no_vendored_strips():
    files = sorted(f.name for f in PKG.glob("*.py"))
    assert len(files) <= 5, files
    assert {"wavspec.py", "runner.py", "wavfeatures.py"} <= set(files)
    runner = (PKG / "runner.py").read_text()
    assert "import strips" in runner or "import ST" in runner
    assert "import wavspec" in runner or "import WS" in runner
    assert "store/scratch/worker-d74" in runner
    for other in ("worker-d69", "worker-d70", "worker-d71", "worker-d72", "worker-d73",
                  "worker-d59", "worker-d60", "worker-d61", "worker-d45", "worker-d33"):
        assert other not in runner
    for f in PKG.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"


def test_runner_wavelet_poison_cpu_cache_and_smoke():
    runner = (PKG / "runner.py").read_text()
    assert "WF.add_wavelet" in runner
    assert "WF.wav_poison_test" in runner and '"passed"' in runner
    assert "SystemExit" in runner
    assert "ThreadPoolExecutor" in runner
    assert "max_workers=WS.PERIOD_JOBS" in runner
    assert "ST.build_manifest" in runner
    assert "ST.poison_test" in runner
    assert "wav_features.json" in runner
    assert "fcache" in runner.lower() and "featcache" in runner
    assert "EXPOS_SMOKE" in runner
    assert "design_cols" in runner and "dict.fromkeys" in runner
    assert "X0104" in runner
    assert "write(te, pred, name)" in runner
    assert "official" not in runner.lower(), "runner must not touch the frozen window"
