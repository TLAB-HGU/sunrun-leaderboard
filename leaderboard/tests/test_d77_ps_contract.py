"""D77 peak-preserving smoothing contract: preregistered spec, causality, single-axis."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1] / "experiments" / "d77_peaksmooth"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "headroom_audit"))
import psmooth as SM  # noqa: E402
import psspec as PS  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W", "def origin_physics", "def add_physics"]


def test_double_tag_front():
    assert PS.FAMILY_TAG == "[family=xgb]"
    assert PS.AXES_TAG.startswith("[axes:") and "post=peak-smooth" in PS.AXES_TAG
    assert PS.HYPOTHESIS.startswith("[family=xgb] [axes:")
    for token in ("E0001", "E0003", "AXES.md", "peak", "X0077"):
        assert token in PS.HYPOTHESIS, f"hypothesis must cite {token}"
    assert "freeze" in PS.HYPOTHESIS.lower() and "official-score" in PS.HYPOTHESIS
    assert "6h" in PS.HYPOTHESIS and "gpu1" in PS.HYPOTHESIS and "cuda" in PS.HYPOTHESIS
    assert PS.check_spec()["windows"] == [3, 5]


def test_constants_frozen_and_params_e0001():
    assert PS.WINDOW_CANDIDATES == (3, 5)
    assert PS.PROM_KMS == 50.0
    assert PS.PURGE_H == 72 and PS.SEED == 0
    assert PS.FULL_RULE and PS.SELECT_RULE
    p = PS.fit_params()
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0001 recipe"
    assert p["tree_method"] == "hist"
    assert p["device"] == "cuda", "GPU build: device=cuda required"
    assert p["n_jobs"] == PS.FIT_NJOBS == 4


def test_threads_budget_and_gpu():
    assert PS.FIT_NJOBS == 4 and PS.PERIOD_JOBS == 3 and PS.GPU_ID == 1
    assert PS.PERIOD_JOBS * PS.FIT_NJOBS <= 20
    assert "worker-d77" in PS.GPU_REASON and "gpu1" in PS.GPU_REASON
    assert "cuda" in PS.GPU_REASON
    runner = (PKG / "runner.py").read_text()
    assert "cuda" in runner and "--prefer-gpu" not in runner  # lease flags live in the expos cmd, not code
    assert "CPU-only" not in runner and "CPU pilot" not in runner


def test_no_forbidden_strings_in_code():
    from expos import core
    blob = "".join((PKG / f).read_text() for f in ("runner.py", "psspec.py", "psmooth.py"))
    assert core.scan(blob) == [], f"forbidden path strings in D77 code: {core.scan(blob)}"
    assert core.scan(PS.HYPOTHESIS) == []


def test_centered_mean_hand_values():
    row = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(SM.centered_mean(row, 3), [1.5, 2.0, 3.0, 4.0, 4.5])
    assert np.allclose(SM.centered_mean(row, 5), [2.0, 2.5, 3.0, 3.5, 4.0])


def test_peak_restore_keeps_spike_but_smooths_jitter():
    base = np.full(72, 400.0)
    base[30] = 620.0  # HSS-like spike, prom 220 >> 50
    out = SM.smooth_origin_matrix(base.reshape(1, 72), 5, 50.0).ravel()
    assert out[30] == 620.0, "peak must survive smoothing (X0077)"
    assert out[0] == 400.0 and out[10] == 400.0, "flat regions stay flat"
    rng = np.random.default_rng(0)
    jitter = 400.0 + rng.normal(0, 8, 72)
    sm = SM.smooth_origin_matrix(jitter.reshape(1, 72), 5, 50.0).ravel()
    assert np.std(sm - 400.0) < np.std(jitter - 400.0), "jitter must shrink"


def test_rows_independent_and_future_never_used():
    rng = np.random.default_rng(1)
    m = 400.0 + rng.normal(0, 10, (4, 72))
    m[1, 40] = 700.0
    out = SM.smooth_origin_matrix(m, 5, 50.0)
    solo = SM.smooth_origin_matrix(m[:1], 5, 50.0)
    assert np.array_equal(out[:1], solo), "one origin must not affect another"
    assert out[1, 40] == 700.0, "peak in row 2 survives regardless of other rows"


def test_select_window_dev_only_precise():
    assert SM.select_window_dev_only({3: 8000.0, 5: 7900.0}) == 5
    assert SM.select_window_dev_only({3: 7900.0, 5: 8000.0}) == 3
    assert SM.select_window_dev_only({3: 100.0, 5: 100.0}) == 3, "ties keep smaller window"
    import pytest
    with pytest.raises(ValueError):
        SM.select_window_dev_only({})
    with pytest.raises(ValueError):
        SM.select_window_dev_only({3: 0.0, 5: 1.0})
    with pytest.raises(ValueError):
        SM.select_window_dev_only({4: 1.0, 5: 2.0})


def test_smooth_poison_pre_invariant_and_positive_control():
    import runner as RN  # noqa: E402  (imports pure poison helper only)
    idx = pd.date_range("2025-01-01", periods=72, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    raw = 400.0 + rng.normal(0, 10, (len(idx), 72))
    res = RN.smooth_poison(idx, raw, 5)
    assert res["passed"] is True and res["pre_invariant"] is True and res["moved"] is True


def test_full_rule_prespecified_and_report_only():
    assert "full 3h" in PS.FULL_RULE and "E0001" in PS.FULL_RULE
    assert "F1" in PS.FULL_RULE and "0.30" in PS.FULL_RULE
    assert "no full run" in PS.FULL_RULE
    assert "freeze" in PS.FULL_RULE.lower() and "official-score" in PS.FULL_RULE
    assert "E0003" in PS.FULL_RULE and "seed-repro" in PS.FULL_RULE
    assert "poison" in PS.FULL_RULE and "1/3" in PS.FULL_RULE


def test_owned_files_and_no_vendored_code():
    files = sorted(f.name for f in PKG.glob("*.py"))
    assert len(files) <= 4, files
    assert {"psspec.py", "runner.py", "psmooth.py"} <= set(files)
    runner = (PKG / "runner.py").read_text()
    assert "store/scratch/worker-d77" in runner
    for other in ("worker-d71", "worker-d75", "worker-d76", "worker-d69", "worker-d70",
                  "worker-d72", "worker-d73", "worker-d74", "worker-d59", "worker-d60",
                  "worker-d45", "worker-d33"):
        assert other not in runner
    for f in PKG.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"


def test_runner_post_poison_gpu_cache_and_smoke():
    runner = (PKG / "runner.py").read_text()
    assert "SM.smooth_origin_matrix" in runner
    assert "SM.select_window_dev_only" in runner
    assert "smooth_poison" in runner and "ace_poison" in runner and '"passed"' in runner
    assert "SystemExit" in runner
    assert "ThreadPoolExecutor" in runner
    assert "max_workers=PS.PERIOD_JOBS" in runner
    assert "psmooth.json" in runner
    assert "availability_manifest.json" in runner
    assert "fcache" in runner.lower() and "featcache" in runner
    assert "EXPOS_SMOKE" in runner
    assert "design_cols" in runner and "dict.fromkeys" in runner
    assert "X0077" in runner
    assert "device" in runner and "cuda" in runner
    assert "write(infos[n]" in runner
    assert "_dev_truth_for" in runner
    assert "official" not in runner.lower(), "runner must not touch the frozen window"
