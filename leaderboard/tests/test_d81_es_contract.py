"""D81 early-stop-rolling contract: dev-only patience, single opt axis, gpu0.

Fast: no data reads, no heavy fits (synthetic XGB only, tiny).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d81_estop"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import esspec as SP  # noqa: E402
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
    assert SP.AXES_TAG.startswith("[axes:") and "opt=" in SP.AXES_TAG
    assert "early-stop-rolling" in SP.AXES_TAG
    assert SP.HYPOTHESIS.startswith("[family=xgb] [axes: opt=early-stop-rolling]")
    for token in ("E0116", "X0107", "E0001"):
        assert token in SP.HYPOTHESIS, f"hypothesis must cite {token}"
    assert "dev origins" in SP.HYPOTHESIS and "no selection/audit" in SP.HYPOTHESIS
    assert "patience 50" in SP.HYPOTHESIS
    assert "6h" in SP.HYPOTHESIS and "gpu0" in SP.HYPOTHESIS
    assert "freeze" in SP.HYPOTHESIS.lower() and "official-score" in SP.HYPOTHESIS
    assert "screening" in SP.HYPOTHESIS
    assert SP.check_spec()["experience"] == ["E0116", "X0107"]


def test_folds_patience_preregistered_dev_only():
    assert SP.PATIENCE == 50 and SP.MAX_ESTIMATORS == 2000
    assert (SP.BEST_MIN, SP.BEST_MAX) == (100, 1500)
    assert SP.SEED == 0 and SP.PURGE_H == 72
    assert [f["name"] for f in SP.FOLDS] == ["F1", "F2", "F3"]
    SP.check_folds_inside_dev()
    assert SP.check_spec()["folds"] == ["F1", "F2", "F3"]
    for src in ("esspec.py", "runner.py"):
        text = (LB / "experiments" / "d81_estop" / src).read_text()
        for s in LATER_DATES:
            assert s not in text, f"{src} must not reference later window {s}"
        for s in FORBIDDEN:
            assert s not in text, f"{src} contains forbidden {s!r}"
    assert core.scan((LB / "experiments" / "d81_estop" / "esspec.py").read_text()
                     + "\n" + (LB / "experiments" / "d81_estop" / "runner.py").read_text()
                     + "\n" + SP.HYPOTHESIS) == []


def test_fold_frames_dev_only_guard():
    n = 200
    rng = np.random.default_rng(0)
    origins = pd.date_range("2025-08-04", "2025-11-27 23:00", freq="6h", tz="UTC")
    df = pd.DataFrame({"origin": rng.choice(origins, n),
                       "target_t": rng.choice(origins, n), "y": rng.normal(400, 50, n)})
    df = df.sort_values("origin").reset_index(drop=True)
    for f in SP.FOLDS:
        tr, va = SP.fold_frames(df, f)
        assert (va["origin"] >= pd.Timestamp(SP.DEV[0])).all()
        assert (va["origin"] <= pd.Timestamp(SP.DEV[1])).all()
        assert (tr["target_t"] < pd.Timestamp(f["train_before"]) - pd.Timedelta(hours=72)).all()
    import pytest
    bad = dict(SP.FOLDS[0])
    bad["val_lo"] = "2025-12-04T00:00:00Z"
    bad["val_hi"] = "2026-02-25T23:00:00Z"
    with pytest.raises(AssertionError):
        SP.fold_frames(df, bad)


def test_best_n_median_clip_deterministic():
    assert SP.best_n_from_folds([199, 299, 399]) == 300
    assert SP.best_n_from_folds([0, 0, 0]) == SP.BEST_MIN
    assert SP.best_n_from_folds([1999, 1999, 1999]) == SP.BEST_MAX
    assert SP.best_n_from_folds([99, 500, 900]) == 501
    assert SP.best_n_from_folds([199, 299, 399]) == SP.best_n_from_folds([399, 199, 299])


def test_fit_params_single_axis_opt_only():
    p = SP.fit_params("cuda", 437)
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k in ("n_estimators", "n_jobs", "random_state"):
            continue
        assert p[k] == v, f"param {k} differs from E0001 recipe"
    assert p["n_estimators"] == 437 and p["n_jobs"] == SP.FIT_NJOBS == 2
    assert p["random_state"] == 0 and p["device"] == "cuda"
    assert SP.PERIOD_JOBS * SP.FIT_NJOBS <= 10
    fp = SP.fold_fit_params("cuda")
    assert fp["n_estimators"] == SP.MAX_ESTIMATORS == 2000
    assert fp["early_stopping_rounds"] == SP.PATIENCE == 50
    refit = SP.fit_params("cuda", 437)
    assert "early_stopping_rounds" not in refit, "refits use frozen best_n only"


def test_devices_and_gpu_reason():
    assert SP.device_for(0) == "cuda"
    import os
    old = (os.environ.get("EXPOS_GPUS"), os.environ.get("CUDA_VISIBLE_DEVICES"))
    try:
        os.environ.pop("EXPOS_GPUS", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        assert SP.device_for(0) == "cuda"
        os.environ["EXPOS_GPUS"] = "0,1"
        assert SP.device_for(0) == "cuda:0" and SP.device_for(1) == "cuda:1"
    finally:
        if old[0] is not None:
            os.environ["EXPOS_GPUS"] = old[0]
        else:
            os.environ.pop("EXPOS_GPUS", None)
        if old[1] is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = old[1]
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    assert "gpu0" in SP.GPU_REASON and "D81" in SP.GPU_REASON
    assert "D85" in SP.GPU_REASON and "D82" in SP.GPU_REASON and "CPU" in SP.GPU_REASON
    assert "60s" in SP.GPU_REASON or "60" in SP.GPU_REASON


def test_rules_preregistered_report_only():
    assert "dev/OOF" in SP.SELECT_RULE and "selection/audit" in SP.SELECT_RULE
    assert "never read" in SP.SELECT_RULE and "72h purge" in SP.SELECT_RULE
    assert "E0116" in SP.SELECT_RULE and "X0107" in SP.SELECT_RULE
    assert "median" in SP.SELECT_RULE and "patience" in SP.SELECT_RULE.lower()
    assert "full 3h" in SP.FULL_RULE and "E0001" in SP.FULL_RULE
    assert "F1>=0.30" in SP.FULL_RULE and "no full run" in SP.FULL_RULE
    assert "E0003" in SP.FULL_RULE and "seed-repro" in SP.FULL_RULE
    assert "screening" in SP.FULL_RULE
    assert "freeze" in SP.FULL_RULE.lower() and "official-score" in SP.FULL_RULE


def _synthetic(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    df = pd.DataFrame({"h": rng.integers(1, 73, n).astype(float),
                       "a": x[:, 0], "b": x[:, 1], "c": x[:, 2], "d": x[:, 3]})
    df["y"] = 400.0 + 2.0 * df["a"] + 0.3 * df["h"] + rng.normal(scale=2.0, size=n)
    return df


def test_earlystop_patience_deterministic_synthetic():
    import xgboost as xgb

    cols = ["h", "a", "b", "c", "d"]
    df = _synthetic()
    tr, va = df.iloc[:200], df.iloc[200:]
    kw = dict(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.9,
              colsample_bytree=0.9, min_child_weight=5, reg_lambda=5,
              tree_method="hist", n_jobs=2, random_state=0,
              early_stopping_rounds=10)
    m1 = xgb.XGBRegressor(**kw).fit(tr[cols], tr["y"], eval_set=[(va[cols], va["y"])],
                                    verbose=False)
    m2 = xgb.XGBRegressor(**kw).fit(tr[cols], tr["y"], eval_set=[(va[cols], va["y"])],
                                    verbose=False)
    assert m1.best_iteration == m2.best_iteration, "early stopping must be deterministic"
    assert 0 <= int(m1.best_iteration) < 200
    assert float(m1.best_score) > 0 and np.isfinite(float(m1.best_score))
    curve = m1.evals_result()["validation_0"]["rmse"]
    assert len(curve) == int(m1.best_iteration) + 1 + 10 or len(curve) <= 200
    frozen = SP.best_n_from_folds([int(m1.best_iteration)] * 3)
    assert SP.BEST_MIN <= frozen <= SP.BEST_MAX


def test_owned_files_scratch_no_vendor():
    pkg = LB / "experiments" / "d81_estop"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 4, files
    assert {"esspec.py", "runner.py"} <= set(files)
    runner = (pkg / "runner.py").read_text()
    assert "store/scratch/worker-d81" in runner
    assert "worker-d85" not in runner and "worker-d82" not in runner
    for other in ("worker-d79", "worker-d80", "worker-d78", "worker-d61", "worker-d33"):
        assert other not in runner
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"
    assert "import esspec" in runner or "import SP" in runner
    assert "from existing_all import full_grid" in runner
    assert "ia.arms(df.columns)[\"existing_all\"]" in runner


def test_runner_earlystop_poison_cache_smoke_strings():
    runner = (LB / "experiments" / "d81_estop" / "runner.py").read_text()
    assert "early_stopping_rounds" in runner and "best_iteration" in runner
    assert "best_n_from_folds" in runner and "fold_frames" in runner
    assert "median" in runner
    assert "predictions_{dev,selection,audit}" in runner
    assert "poison_check" in runner and '"passed"' in runner
    assert "positive_control" in runner and "SystemExit" in runner
    assert "ThreadPoolExecutor" in runner and "cuda" in runner
    assert "device_for" in runner and "fcache" in runner.lower() and "featcache" in runner
    assert "EXPOS_SMOKE" in runner
    assert "E0116" in runner and "X0107" in runner
    assert "72h" in runner
