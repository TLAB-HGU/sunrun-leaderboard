"""D79 stability-selection contract: seed x period stable top-35, dev-only.

Fast: no data reads, no heavy fits (numpy synthetic only, tiny XGB).
"""

import sys
from pathlib import Path

import numpy as np

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d79_stabsel"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import stabspec as SP  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_dual_tag_and_experience_citation():
    assert SP.FAMILY_TAG == "[family=xgb]"
    assert SP.AXES_TAG.startswith("[axes:") and "sel=" in SP.AXES_TAG
    assert "stability-selection" in SP.AXES_TAG
    assert SP.HYPOTHESIS.startswith("[family=xgb] [axes:")
    for token in ("E0114", "X0105", "E0001"):
        assert token in SP.HYPOTHESIS, f"hypothesis must cite {token}"
    assert "dev-OOF" in SP.HYPOTHESIS or "dev-only" in SP.HYPOTHESIS
    assert "freeze" in SP.HYPOTHESIS.lower() and "official-score" in SP.HYPOTHESIS
    assert "6h" in SP.HYPOTHESIS and "gpu1" in SP.HYPOTHESIS


def test_stability_rule_fixed_and_dev_only():
    assert SP.TOP_K == 35 and SP.PER_TRIAL_K == 35
    assert SP.PERM_REPEATS == 3
    assert SP.SEEDS == (0, 1, 2) and SP.N_TRIALS == 6
    assert SP.check_spec()["top_k"] == 35
    assert "dev-OOF" in SP.SELECT_RULE and "selection/audit" in SP.SELECT_RULE
    assert "never read" in SP.SELECT_RULE
    assert "E0114" in SP.SELECT_RULE and "X0105" in SP.SELECT_RULE
    assert "stability" in SP.SELECT_RULE and "72h purge" in SP.SELECT_RULE
    assert "permutation" in SP.SELECT_RULE
    import pandas as pd
    assert pd.Timestamp(SP.DEV_LO_UTC) < pd.Timestamp(SP.VAL_A_LO_UTC) < pd.Timestamp(SP.VAL_B_LO_UTC)
    assert pd.Timestamp(SP.VAL_B_LO_UTC) < pd.Timestamp(SP.DEV_END_UTC)
    n = 200
    rng = np.random.default_rng(0)
    origins = pd.date_range("2025-08-04", "2025-11-27 23:00", freq="6h", tz="UTC")
    df = pd.DataFrame({"origin": rng.choice(origins, n),
                       "target_t": rng.choice(origins, n), "y": rng.normal(400, 50, n)})
    df = df.sort_values("origin").reset_index(drop=True)
    for name, lo, hi in SP.SPLITS:
        tr, va = SP.split_for(df, lo, hi)
        assert (va["origin"] >= pd.Timestamp(SP.DEV_LO_UTC)).all()
        assert (va["origin"] <= pd.Timestamp(SP.DEV_END_UTC)).all()
        assert (tr["target_t"] < pd.Timestamp(lo) - pd.Timedelta(hours=72)).all()
    import pytest
    with pytest.raises(AssertionError):
        SP.split_for(df, "2025-12-04T00:00:00Z", "2026-02-25T23:00:00Z")  # selection window banned


def _synthetic_reg(n=300, seed=0, n_noise=5):
    import pandas as pd

    rng = np.random.default_rng(seed)
    strong = rng.standard_normal(n) * 50 + 400
    df = pd.DataFrame({"h": rng.integers(1, 73, n).astype(float),
                       "strong": strong,
                       **{f"noise{i}": rng.standard_normal(n) * 50 + 400 for i in range(n_noise)}})
    df["y"] = 400.0 + 0.8 * df["strong"] + 0.3 * df["h"] + rng.standard_normal(n) * 5.0
    return df


def test_permutation_and_stability_deterministic():
    import xgboost as xgb

    cols = ["h", "strong", "noise0", "noise1", "noise2", "noise3", "noise4"]
    df = _synthetic_reg()
    m = xgb.XGBRegressor(n_estimators=30, max_depth=3, learning_rate=0.1,
                         subsample=0.9, colsample_bytree=0.9, min_child_weight=5,
                         reg_lambda=5, tree_method="hist", n_jobs=2, random_state=0)
    m.fit(df[cols], df["y"])
    base, d1 = SP.permutation_deltas(m, df, cols, seed=1)
    base2, d2 = SP.permutation_deltas(m, df, cols, seed=1)
    assert np.isfinite(base) and base > 0
    assert d1 == d2, "permutation deltas must be deterministic"
    assert d1["strong"] > 0 and d1["h"] > 0
    assert d1["strong"] > max(d1[f"noise{i}"] for i in range(5))
    tops = [SP.trial_topk(d1, {c: 1.0 for c in cols}, top_k=3) for _ in range(6)]
    deltas = [dict(d1) for _ in range(6)]
    gains = [{c: 1.0 for c in cols} for _ in range(6)]
    # stable signal: same trial repeated -> S=1 for its members
    sub, stab, md, mg = SP.stability_select(tops, deltas, gains, top_k=3)
    assert len(sub) == 3 and "h" in sub and "strong" in sub
    assert stab["strong"] == 1.0
    # unstable challenger: one trial swaps in noise -> stability separates them
    tops2 = list(tops)
    tops2[0] = ["h", "noise0", "noise1"]
    sub2, stab2, _, _ = SP.stability_select(tops2, deltas, gains, top_k=3)
    assert stab2["strong"] < 1.0 and stab2["strong"] > stab2["noise0"]
    assert "strong" in sub2 and "noise0" not in sub2
    assert SP.stability_select(tops, deltas, gains, top_k=3)[0] == sub  # deterministic


def test_trial_topk_keeps_h_and_check_subset_contract():
    base = ["h", "a", "b", "c", "d"] + [f"x{i}" for i in range(100)]
    d = {c: float(i) for i, c in enumerate(base)}
    g = {c: 1.0 for c in base}
    top = SP.trial_topk(d, g)
    assert len(top) == 35 and "h" in top
    good = ["h"] + [f"x{i}" for i in range(34)]
    assert SP.check_subset(good, base) == good
    import pytest
    with pytest.raises(AssertionError):
        SP.check_subset(["a"] + [f"x{i}" for i in range(34)], base)  # missing h
    with pytest.raises(AssertionError):
        SP.check_subset(["h"] + [f"x{i}" for i in range(33)], base)  # wrong size
    with pytest.raises(AssertionError):
        SP.check_subset(["h"] + [f"x{i}" for i in range(33)] + ["nope"], base)  # outside base


def test_fit_params_match_e0003_recipe_infra_only():
    p = SP.fit_params("cuda:1", seed=1)
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k in ("n_jobs", "random_state"):
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["tree_method"] == "hist"
    assert str(p["device"]).startswith("cuda")
    assert p["n_jobs"] == SP.FIT_NJOBS == 3
    assert p["random_state"] == 1
    assert SP.PERIOD_JOBS * SP.FIT_NJOBS <= 10


def test_devices_round_robin_visible_gpus(monkeypatch):
    monkeypatch.delenv("EXPOS_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert SP.device_for(0) == "cuda"
    monkeypatch.setenv("EXPOS_GPUS", "1")
    assert SP.device_for(0) == "cuda"
    monkeypatch.setenv("EXPOS_GPUS", "0,1")
    assert SP.device_for(0) == "cuda:0"
    assert SP.device_for(1) == "cuda:1"
    assert SP.device_for(2) == "cuda:0"


def test_full_rule_prespecified_and_report_only():
    assert "full 3h" in SP.FULL_RULE and "E0001" in SP.FULL_RULE
    assert "F1" in SP.FULL_RULE and "0.30" in SP.FULL_RULE
    assert "no full run" in SP.FULL_RULE
    assert "freeze" in SP.FULL_RULE.lower() and "official-score" in SP.FULL_RULE
    assert "E0003" in SP.FULL_RULE and "seed-repro" in SP.FULL_RULE


def test_gpu_reason_assigns_gpu1_and_cpu_for_d80():
    assert "gpu1" in SP.GPU_REASON and "D79" in SP.GPU_REASON
    assert "D78" in SP.GPU_REASON and "D80" in SP.GPU_REASON
    assert "CPU" in SP.GPU_REASON


def test_owned_files_and_no_vendored_strips():
    pkg = LB / "experiments" / "d79_stabsel"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 5, files
    assert {"stabspec.py", "runner.py"} <= set(files)
    runner = (pkg / "runner.py").read_text()
    assert "import strips" in runner or "import ST" in runner
    assert "import stabspec" in runner or "import SP" in runner
    assert "store/scratch/worker-d79" in runner
    assert "worker-d78" not in runner and "worker-d80" not in runner
    for other in ("worker-d59", "worker-d60", "worker-d61", "worker-d67",
                  "worker-d68", "worker-d45", "worker-d33"):
        assert other not in runner
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"


def test_runner_stability_poison_gpu_cache_and_smoke():
    runner = (LB / "experiments" / "d79_stabsel" / "runner.py").read_text()
    assert "dev_oof_stability" in runner
    assert "permutation_deltas" in runner and "stability_select" in runner
    assert "trial_topk" in runner and "check_subset" in runner
    assert "ST.poison_test" in runner and '"passed"' in runner
    assert "SystemExit" in runner
    assert "ThreadPoolExecutor" in runner
    assert "device" in runner and "cuda" in runner
    assert "for i, n in enumerate(names)" in runner
    assert "ST.build_manifest" in runner
    assert "fcache" in runner.lower() and "featcache" in runner
    assert "EXPOS_SMOKE" in runner
    assert "design_cols" in runner and "dict.fromkeys" in runner
    assert "E0114" in runner and "X0105" in runner
