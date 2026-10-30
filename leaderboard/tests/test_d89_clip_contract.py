"""D89 monotone-clip post-rule contract: post axis only, CPU.

Fast: no data reads, no heavy fits (synthetic clip math + tiny XGB only).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d89_clip"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import clipspec as SP  # noqa: E402
import info_ablation as ia  # noqa: E402
from expos import core  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay",
             "official_check", "official-check", "predictions_official_truth"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]
# Selection/audit window boundaries must never appear in preregistered code.
LATER_DATES = ("2025-12-04", "2026-02-25", "2026-03-04", "2026-05-28",
               "2026-05-29", "2026-05-31", "2026-09-26")


def test_dual_tag_and_experience_citation():
    assert SP.FAMILY_TAG == "[family=xgb]"
    assert SP.AXES_TAG.startswith("[axes:") and "post=" in SP.AXES_TAG
    assert "clip-monotone" in SP.AXES_TAG
    assert SP.HYPOTHESIS.startswith("[family=xgb] [axes: post=clip-monotone]")
    for token in ("E0001", "E0003", "X0127", "X0132", "X0134", "research/AXES.md"):
        assert token in SP.HYPOTHESIS, f"hypothesis must cite {token}"
    assert "monotone or clipping rules" in SP.HYPOTHESIS
    assert "dev-period OOF MSE only" in SP.HYPOTHESIS
    assert "6h" in SP.HYPOTHESIS and "CPU" in SP.HYPOTHESIS
    assert "freeze" in SP.HYPOTHESIS.lower() and "official-score" in SP.HYPOTHESIS
    assert "screening" in SP.HYPOTHESIS
    assert SP.check_spec()["experience"] == ["X0127", "X0132", "X0134"]


def test_clip_grid_preregistered():
    assert SP.CLIP_GRID[0] == (None, None) and len(SP.CLIP_GRID) == 6
    assert SP.LO_GRID == (None, 250.0) and SP.HI_GRID == (None, 900.0, 1100.0)
    assert SP.SEED == 0 and SP.PURGE_H == 72
    assert SP.clip_key(None, None) == "None-None"
    assert SP.check_pair(None, 900.0) == (None, 900.0)
    import pytest
    with pytest.raises(ValueError):
        SP.check_pair(0.0, 10.0)
    for src in ("clipspec.py", "runner.py"):
        text = (LB / "experiments" / "d89_clip" / src).read_text()
        for s in LATER_DATES:
            assert s not in text, f"{src} must not reference later window {s}"
        for s in FORBIDDEN:
            assert s not in text, f"{src} contains forbidden {s!r}"
    assert core.scan((LB / "experiments" / "d89_clip" / "clipspec.py").read_text()
                     + "\n" + (LB / "experiments" / "d89_clip" / "runner.py").read_text()
                     + "\n" + SP.HYPOTHESIS) == []


def test_clip_monotone_properties():
    raw = np.array([200.0, 300.0, 500.0, 950.0, 1300.0])
    ident = SP.clip_predictions(raw, None, None)
    assert np.array_equal(ident, raw), "(None,None) must recover the raw base"
    capped = SP.clip_predictions(raw, 250.0, 900.0)
    assert np.array_equal(capped, np.array([250.0, 300.0, 500.0, 900.0, 900.0]))
    # Order-preserving: ranks never flip.
    assert (np.argsort(np.argsort(capped)) == np.argsort(np.argsort(raw))).all()
    # In-bounds peaks bitwise untouched.
    assert capped[2] == raw[2]
    # One-sided caps work.
    assert np.array_equal(SP.clip_predictions(raw, None, 900.0)[-2:], [900.0, 900.0])
    assert np.array_equal(SP.clip_predictions(raw, 250.0, None)[:2], [250.0, 300.0])
    import pytest
    with pytest.raises(ValueError):
        SP.clip_predictions([1.0, np.nan], None, 900.0)
    with pytest.raises(ValueError):
        SP.clip_predictions(raw, 900.0, 250.0)


def _synthetic_dev(seed=0):
    rng = np.random.default_rng(seed)
    y = 400.0 + rng.normal(scale=40.0, size=500)
    raw = y + rng.normal(scale=30.0, size=500)
    raw[::50] = 2500.0  # absurd tails the cap should tame
    return raw, y


def test_select_clip_dev_only_no_clip_wins_ties():
    raw, y = _synthetic_dev()
    (lo, hi), mses = SP.select_clip(raw, y)
    assert (lo, hi) in SP.CLIP_GRID
    assert len(mses) == 6 and mses["None-None"] > mses[SP.clip_key(lo, hi)]
    # Tie-break: identical vectors keep (None,None) so the clip never loses on dev.
    raw2 = np.full(300, 400.0)
    y2 = np.full(300, 400.0)
    best2, _ = SP.select_clip(raw2, y2)
    assert best2 == (None, None)
    # NaN labels are excluded, not fitted.
    yn = y.copy()
    yn[:50] = np.nan
    bestn, _ = SP.select_clip(raw, yn)
    assert bestn in SP.CLIP_GRID
    import pytest
    with pytest.raises(ValueError):
        SP.select_clip(raw[:50], y[:50])
    with pytest.raises(ValueError):
        SP.select_clip(raw, y[:100])


def test_fit_params_single_axis_post_only():
    p = SP.fit_params()
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k in ("n_jobs", "random_state"):
            continue
        assert p[k] == v, f"param {k} differs from E0001 recipe"
    assert p["n_jobs"] == SP.FIT_NJOBS == 2
    assert p["random_state"] == 0 and p["device"] == "cpu"
    assert SP.device_for(0) == "cpu" and SP.device_for(1) == "cpu"
    assert SP.PERIOD_JOBS * SP.FIT_NJOBS <= 20


def test_cpu_reason_assignment_split():
    assert "D89" in SP.GPU_REASON and "CPU" in SP.GPU_REASON
    assert "D87" in SP.GPU_REASON and "D88" in SP.GPU_REASON
    assert "gpu0" in SP.GPU_REASON and "gpu1" in SP.GPU_REASON
    assert "60s" in SP.GPU_REASON and "worker-d89" in SP.GPU_REASON
    assert "n_jobs" in SP.GPU_REASON


def test_rules_preregistered_report_only():
    assert "dev/OOF" in SP.SELECT_RULE and "selection/audit" in SP.SELECT_RULE
    assert "never read" in SP.SELECT_RULE and "72h purge" in SP.SELECT_RULE
    assert "X0127" in SP.SELECT_RULE and "X0132" in SP.SELECT_RULE
    assert "full 3h" in SP.FULL_RULE and "E0001" in SP.FULL_RULE
    assert "F1>=0.30" in SP.FULL_RULE and "no full run" in SP.FULL_RULE
    assert "E0003" in SP.FULL_RULE and "seed-repro" in SP.FULL_RULE
    assert "screening" in SP.FULL_RULE
    assert "freeze" in SP.FULL_RULE.lower() and "official-score" in SP.FULL_RULE


def test_clipped_tiny_fit_cpu():
    import xgboost as xgb

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"h": np.tile(np.arange(1, 73, 3), 9)[:n].astype(float),
                       "v_lag0": rng.normal(400, 30, n)})
    df["y"] = df["v_lag0"] + rng.normal(scale=5.0, size=n)
    cols = ["h", "v_lag0"]
    m = xgb.XGBRegressor(**{**SP.fit_params(), "n_estimators": 10}).fit(df[cols], df["y"])
    pred = m.predict(df[cols])
    out = SP.clip_predictions(pred, 250.0, 900.0)
    assert np.isfinite(out).all() and len(out) == len(df)
    assert (out >= 250.0).all() and (out <= 900.0).all()


def test_owned_files_scratch_no_vendor():
    pkg = LB / "experiments" / "d89_clip"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 4, files
    assert {"clipspec.py", "runner.py"} <= set(files)
    runner = (pkg / "runner.py").read_text()
    assert "store/scratch/worker-d89" in runner
    for other in ("worker-d83", "worker-d84", "worker-d86", "worker-d87", "worker-d88",
                  "worker-d81", "worker-d82", "worker-d85", "worker-d71", "worker-d76"):
        assert other not in runner
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"
    assert "from existing_all import full_grid" in runner
    assert "from existing_all import write" in runner
    assert 'ia.arms(df.columns)["existing_all"]' in runner
    assert "clip_predictions" in runner and "select_clip" in runner


def test_runner_post_poison_cache_smoke_strings():
    runner = (LB / "experiments" / "d89_clip" / "runner.py").read_text()
    assert "clip_predictions" in runner and "select_clip" in runner
    assert "CLIP_GRID" in runner
    assert "predictions_{dev,selection,audit}" in runner
    assert "poison_check" in runner and '"passed"' in runner
    assert "positive_control" in runner and "SystemExit" in runner
    assert "ThreadPoolExecutor" in runner and '"cpu"' in runner
    assert "fcache" in runner.lower() and "featcache" in runner
    assert "EXPOS_SMOKE" in runner
    assert "X0127" in runner and "X0132" in runner and "E0003" in runner and "E0001" in runner
    assert "72h" in runner
