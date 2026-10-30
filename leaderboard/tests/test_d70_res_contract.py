"""D70 residual-target contract: XGB on E0003-line residual, target axis only.

Fast: no data reads, no heavy fits (numpy synthetic only, tiny checks).
"""

import sys
from pathlib import Path

import numpy as np

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d70_residual"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import spec as SP  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_dual_tag_and_experience_citation():
    assert SP.FAMILY_TAG == "[family=xgb]"
    assert SP.AXES_TAG.startswith("[axes:") and "target=" in SP.AXES_TAG
    assert "residual" in SP.AXES_TAG
    assert SP.HYPOTHESIS.startswith("[family=xgb] [axes:")
    for token in ("E0124", "X0111", "E0001"):
        assert token in SP.HYPOTHESIS, f"hypothesis must cite {token}"
    assert "Residual to persistence or to the E0003 forecast" in SP.HYPOTHESIS
    assert "freeze" in SP.HYPOTHESIS.lower() and "official-score" in SP.HYPOTHESIS
    assert "6h" in SP.HYPOTHESIS and "CPU" in SP.HYPOTHESIS


def test_target_only_change_params_and_cols():
    p = SP.fit_params()
    assert set(p) == set(ia.PARAMS)
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["tree_method"] == "hist"
    assert "device" not in p, "CPU-only: no device key"
    assert p["n_jobs"] == SP.FIT_NJOBS == 2
    assert SP.PERIOD_JOBS * SP.FIT_NJOBS <= 20
    assert SP.check_spec()["threads"] == SP.PERIOD_JOBS * SP.FIT_NJOBS


def test_cpu_only_no_gpu_markers():
    for name in ("spec.py", "runner.py"):
        text = (LB / "experiments" / "d70_residual" / name).read_text()
        assert "cuda" not in text, f"{name} must stay CPU-only"
    assert "CPU" in SP.CPU_REASON and "worker-d70" in SP.CPU_REASON
    runner = (LB / "experiments" / "d70_residual" / "runner.py").read_text()
    assert "--gpu" not in runner


def test_oof_split_deterministic_and_covering():
    import pandas as pd
    n = 100
    rng = np.random.default_rng(0)
    origins = pd.date_range("2025-08-04", periods=50, freq="6h", tz="UTC")
    df = pd.DataFrame({"origin": rng.choice(origins, n),
                       "target_t": rng.choice(origins, n), "y": rng.normal(400, 50, n)})
    m1, m2 = SP.oof_halves(df), SP.oof_halves(df)
    assert (m1 == m2).all(), "OOF split must be deterministic"
    assert m1.any() and (~m1).any(), "both halves non-empty"
    first = df.loc[m1, "origin"].max()
    second = df.loc[~m1, "origin"].min()
    assert first <= second, "halves split by origin order"


def test_combine_adds_base_and_correction():
    base = np.array([400.0, 500.0, np.nan])
    corr = np.array([10.0, -20.0, 5.0])
    out = SP.combine(base[:2], corr[:2])
    assert np.allclose(out, [410.0, 480.0])
    try:
        SP.combine(base, corr)
    except AssertionError:
        pass
    else:
        raise AssertionError("combine must reject non-finite input")


def test_full_rule_prespecified_and_report_only():
    assert "full 3h" in SP.FULL_RULE and "E0001" in SP.FULL_RULE
    assert "F1" in SP.FULL_RULE and "0.30" in SP.FULL_RULE
    assert "no full run" in SP.FULL_RULE
    assert "freeze" in SP.FULL_RULE.lower() and "official-score" in SP.FULL_RULE
    assert "E0003" in SP.FULL_RULE and "seed-repro" in SP.FULL_RULE
    assert "poison" in SP.FULL_RULE and "1/3" in SP.FULL_RULE


def test_owned_files_and_no_vendored_strips():
    pkg = LB / "experiments" / "d70_residual"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 4, files
    assert {"spec.py", "runner.py"} <= set(files)
    runner = (pkg / "runner.py").read_text()
    assert "import strips" in runner or "import ST" in runner
    assert "import spec" in runner or "import SP" in runner
    assert "store/scratch/worker-d70" in runner
    assert "worker-d68" not in runner
    for other in ("worker-d59", "worker-d60", "worker-d61", "worker-d63",
                  "worker-d64", "worker-d65", "worker-d66", "worker-d69",
                  "worker-d73", "worker-d45", "worker-d33"):
        assert other not in runner
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"


def test_runner_residual_poison_cpu_cache_and_smoke():
    runner = (LB / "experiments" / "d70_residual" / "runner.py").read_text()
    assert "oof_base" in runner
    assert "resid = y - base_oof" in runner or "y - base_oof" in runner
    assert "SP.combine" in runner
    assert "residual_table" in runner
    assert "ST.poison_test" in runner and '"passed"' in runner
    assert "SystemExit" in runner
    assert "ThreadPoolExecutor" in runner
    assert "max_workers=SP.PERIOD_JOBS" in runner
    assert "ST.build_manifest" in runner
    assert "fcache" in runner.lower() and "featcache" in runner
    assert "EXPOS_SMOKE" in runner
    assert "design_cols" in runner and "dict.fromkeys" in runner
    assert "E0124" in runner and "X0111" in runner
    assert "write(te, pred, name)" in runner
    assert "for n in PERIODS" in runner or "for n, f in futs" in runner
    assert "official" not in runner.lower(), "runner must not touch the frozen window"
