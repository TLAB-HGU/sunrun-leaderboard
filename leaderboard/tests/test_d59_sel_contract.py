"""D59 sel-subset contract: dev-OOF permutation top-35, E0003-identical recipe.

Fast: no data reads, no heavy fits (numpy/sklearn synthetic only, tiny XGB).
"""

import sys
from pathlib import Path

import numpy as np

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d59_selsubset"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import selspec as SP  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_sel_rule_fixed_and_dev_only():
    assert SP.TOP_K == 35
    assert SP.PERM_REPEATS == 5
    assert SP.PURGE_H == 72 and SP.SEED == 0
    assert SP.check_spec()["top_k"] == 35
    assert "dev-OOF" in SP.SELECT_RULE and "selection/audit" in SP.SELECT_RULE
    assert "X0104" in SP.SELECT_RULE and "AXES.md" in SP.SELECT_RULE
    assert "permutation" in SP.SELECT_RULE and "72h purge" in SP.SELECT_RULE
    assert "prediction" in SP.SELECT_RULE and "never read" in SP.SELECT_RULE
    import pandas as pd
    assert pd.Timestamp(SP.DEV_LO_UTC) < pd.Timestamp(SP.VAL_LO_UTC) < pd.Timestamp(SP.DEV_END_UTC)


def test_fit_params_match_e0003_recipe_infra_only():
    p = SP.fit_params("cuda:1")
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["tree_method"] == "hist"
    assert str(p["device"]).startswith("cuda")
    assert p["n_jobs"] == SP.FIT_NJOBS == 3
    assert SP.PERIOD_JOBS * SP.FIT_NJOBS <= 10


def test_devices_round_robin_visible_gpus(monkeypatch):
    monkeypatch.delenv("EXPOS_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert SP.device_for(0) == "cuda"
    monkeypatch.setenv("EXPOS_GPUS", "0,1")
    assert SP.device_for(0) == "cuda:0"
    assert SP.device_for(1) == "cuda:1"
    assert SP.device_for(2) == "cuda:0"


def test_full_rule_prespecified_and_report_only():
    assert "full 3h" in SP.FULL_RULE and "E0001" in SP.FULL_RULE
    assert "F1" in SP.FULL_RULE and "0.30" in SP.FULL_RULE
    assert "no full run" in SP.FULL_RULE
    assert "freeze" in SP.FULL_RULE.lower() and "official" in SP.FULL_RULE.lower()


def _synthetic_reg(n=300, seed=0, n_noise=5):
    import pandas as pd

    rng = np.random.default_rng(seed)
    strong = rng.standard_normal(n) * 50 + 400
    df = pd.DataFrame({"h": rng.integers(1, 73, n).astype(float),
                       "strong": strong,
                       **{f"noise{i}": rng.standard_normal(n) * 50 + 400 for i in range(n_noise)}})
    df["y"] = 400.0 + 0.8 * df["strong"] + 0.3 * df["h"] + rng.standard_normal(n) * 5.0
    return df


def test_permutation_selects_signal_and_is_deterministic():
    import xgboost as xgb

    cols = ["h", "strong", "noise0", "noise1", "noise2", "noise3", "noise4"]
    df = _synthetic_reg()
    m = xgb.XGBRegressor(n_estimators=30, max_depth=3, learning_rate=0.1,
                         subsample=0.9, colsample_bytree=0.9, min_child_weight=5,
                         reg_lambda=5, tree_method="hist", n_jobs=2, random_state=0)
    m.fit(df[cols], df["y"])
    base, d1 = SP.permutation_deltas(m, df, cols)
    base2, d2 = SP.permutation_deltas(m, df, cols)
    assert np.isfinite(base) and base > 0
    assert d1 == d2, "permutation deltas must be deterministic"
    assert d1["strong"] > 0 and d1["h"] > 0
    # signal columns outrank pure noise on this synthetic frame
    assert d1["strong"] > max(d1[f"noise{i}"] for i in range(5))
    sub = SP.select_subset(d1, {c: 1.0 for c in cols}, top_k=3)
    assert len(sub) == 3 and "h" in sub and "strong" in sub
    full = SP.select_subset(d1, {c: float(i) for i, c in enumerate(cols)}, top_k=3)
    assert full == sub or set(full) == set(sub) or "strong" in full
    # tie-break is deterministic: same inputs, same ranked order
    assert SP.select_subset(d1, {c: 1.0 for c in cols}, top_k=3) == sub


def test_check_subset_contract():
    base = ["h", "a", "b", "c", "d"] + [f"x{i}" for i in range(100)]
    good = ["h"] + [f"x{i}" for i in range(34)]
    assert SP.check_subset(good, base) == good
    import pytest
    with pytest.raises(AssertionError):
        SP.check_subset(["a"] + [f"x{i}" for i in range(34)], base)  # missing h
    with pytest.raises(AssertionError):
        SP.check_subset(["h"] + [f"x{i}" for i in range(33)], base)  # wrong size
    with pytest.raises(AssertionError):
        SP.check_subset(["h"] + [f"x{i}" for i in range(33)] + ["nope"], base)  # outside base


def test_owned_files_and_no_vendored_strips():
    pkg = LB / "experiments" / "d59_selsubset"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 5, files
    assert {"selspec.py", "runner.py"} <= set(files)
    runner = (pkg / "runner.py").read_text()
    assert "import strips" in runner or "import ST" in runner
    assert "import selspec" in runner or "import SP" in runner
    assert "store/scratch/worker-d59" in runner
    for other in ("worker-d60", "worker-d61", "worker-d43", "worker-d33",
                  "worker-d22", "worker-d23", "worker-d21", "worker-d45"):
        assert other not in runner
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"


def test_runner_sel_poison_gpu_cache_and_smoke():
    runner = (LB / "experiments" / "d59_selsubset" / "runner.py").read_text()
    assert "dev_oof_select" in runner
    assert "permutation_deltas" in runner and "select_subset" in runner
    assert "check_subset" in runner
    assert "ST.poison_test" in runner and '"passed"' in runner
    assert "SystemExit" in runner
    assert "ThreadPoolExecutor" in runner
    assert "device" in runner and "cuda" in runner
    assert "for i, n in enumerate(names)" in runner
    assert "ST.build_manifest" in runner
    assert "fcache" in runner.lower() and "featcache" in runner
    assert "EXPOS_SMOKE" in runner
    assert "design_cols" in runner and "dict.fromkeys" in runner
    assert "X0104" in runner or "AXES.md" in runner
