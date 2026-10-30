"""D125 XGBRF contract: dual tags, extreme-preserving single head, CPU twin, isolation.

Fast static + unit checks only: no data reads, no fits beyond one small
synthetic CPU fit, no held-window contact. The pilot run itself (registered
separately) proves end-to-end behaviour through expos run/evaluate/poison.
"""
import sys
from pathlib import Path

import numpy as np

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d125_xgbrf"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
sys.path.insert(0, str(LB))

import xrspec as SPEC  # noqa: E402
import info_ablation as ia  # noqa: E402
from expos import core  # noqa: E402

RUNNER_SRC = (LB / "experiments" / "d125_xgbrf" / "runner.py").read_text()
SPEC_SRC = (LB / "experiments" / "d125_xgbrf" / "xrspec.py").read_text()

COPY_BANS = ["def speed_features(", "def ace_features(", "def ch_features(",
             "def enlil_features(", "def build(", "def arms(", "def full_grid(",
             "def write("]
LATER_DATES = ("2025-12-04", "2026-02-25", "2026-03-04", "2026-05-28",
               "2026-05-29", "2026-05-31", "2026-09-26")


def test_hypothesis_dual_tag_and_experience():
    assert SPEC.HYPOTHESIS.startswith("[family=rf] [axes: model=xgbrf]")
    assert SPEC.FAMILY_TAG == "[family=rf]"
    assert SPEC.AXES_TAG.startswith("[axes:") and "model=" in SPEC.AXES_TAG
    assert "xgbrf" in SPEC.AXES_TAG
    assert SPEC.EXPERIENCE == ["X0109", "X0037"]
    assert "X0109" in SPEC.HYPOTHESIS and "X0037" in SPEC.HYPOTHESIS
    assert "E0117" in SPEC.HYPOTHESIS  # replicated pilot id stated
    assert SPEC.DIRECTION == "D125" and SPEC.OWNER == "worker-d125"
    assert "no freeze" in SPEC.HYPOTHESIS.lower()
    assert "6h" in SPEC.HYPOTHESIS  # pilot grid stated
    assert "screening" in SPEC.HYPOTHESIS  # 6h F1 is screening only
    assert "held window" in SPEC.HYPOTHESIS


def test_extreme_preserving_single_head_params():
    p = SPEC.fit_params()
    assert SPEC.ESTIMATOR == "XGBRFRegressor"
    assert p["max_depth"] >= 10 and p["min_child_weight"] <= 2
    assert p["n_estimators"] == 600 and p["tree_method"] == "hist"
    assert p["device"] == "cpu" and p["random_state"] == 0
    assert p["n_jobs"] <= 2
    assert p.get("objective", "reg:squarederror") == "reg:squarederror"
    import xgboost as xgb
    m = xgb.XGBRFRegressor(**p)
    assert type(m).__name__ == "XGBRFRegressor"


def test_single_axis_model_only():
    p = SPEC.fit_params()
    # infra-identical where the axis does not touch the recipe
    assert p["n_estimators"] == ia.PARAMS["n_estimators"] == 600
    assert p["subsample"] == ia.PARAMS["subsample"] == 0.8
    assert p["reg_lambda"] == ia.PARAMS["reg_lambda"]
    assert p["tree_method"] == ia.PARAMS["tree_method"] == "hist"
    assert p["random_state"] == ia.PARAMS["random_state"] == 0
    # mechanism change only: forest leaves + forest column sampling, no boosting rate
    assert p["max_depth"] == 12 and p["min_child_weight"] == 1
    assert "colsample_bynode" in p and "learning_rate" not in p
    assert "sample_weight" not in RUNNER_SRC
    assert "EPAM_COLS" not in RUNNER_SRC and "epam" not in RUNNER_SRC.lower()
    assert "blend" not in RUNNER_SRC.lower()
    assert "EVENT_THRESHOLD" not in RUNNER_SRC
    assert RUNNER_SRC.count(".fit(") == 1, "exactly one head fit (single head, no refit)"
    assert 'ia.arms(df.columns)["existing_all"]' in RUNNER_SRC  # no new columns
    assert SPEC.check_spec()["axes"] == "[axes: model=xgbrf]"


def test_e0001_twin_rows_purge_budget():
    assert SPEC.PURGE_HOURS == 72
    assert SPEC.FREQ_PILOT == "6h" and SPEC.FREQ_FULL == "3h"
    assert SPEC.N_ESTIMATORS == 600 and SPEC.SEED == 0
    assert "SPEC.PURGE_HOURS" in RUNNER_SRC
    assert "build_train" in RUNNER_SRC and "test_grid" in RUNNER_SRC
    assert "design_cols" in RUNNER_SRC


def test_cpu_budget_no_accelerator():
    assert SPEC.PERIOD_JOBS * SPEC.FIT_NJOBS <= 10
    assert SPEC.PERIOD_JOBS == 3, "three periods run in parallel"
    assert SPEC.FIT_NJOBS <= 2
    assert SPEC.fit_params()["device"] == "cpu"
    for pat in ("gpu_lease", "--gpu", "prefer-gpu", "EXPOS_GPUS",
                "CUDA_VISIBLE_DEVICES", "device_for"):
        assert pat not in RUNNER_SRC, pat
        assert pat not in SPEC_SRC, pat
    assert "CPU" in RUNNER_SRC
    assert "D122" in SPEC.CPU_REASON and "D123" in SPEC.CPU_REASON
    assert "gpu0" in SPEC.CPU_REASON and "gpu1" in SPEC.CPU_REASON
    assert "60s" in SPEC.CPU_REASON and "foreground" in SPEC.CPU_REASON
    assert "lease" in SPEC.CPU_REASON


def test_scratch_isolation():
    assert "store/scratch/worker-d125" in RUNNER_SRC
    others = [l for l in RUNNER_SRC.splitlines() if "worker-" in l and "worker-d125" not in l]
    assert others == [], f"scratch leak: {others[:3]}"
    assert "featcache" in RUNNER_SRC and "fcache_key" in RUNNER_SRC
    assert SPEC.SCRATCH_NAME == "worker-d125"


def test_readonly_no_vendored_copies():
    for s in COPY_BANS:
        assert s not in RUNNER_SRC, f"runner must import, not copy: {s!r}"
        assert s not in SPEC_SRC, f"spec must not vendor: {s!r}"
    assert "from existing_all import" in RUNNER_SRC


def test_no_forbidden_sources_or_later_windows():
    for name, src in (("xrspec.py", SPEC_SRC), ("runner.py", RUNNER_SRC)):
        for tok in ("datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
                    "naive.parquet", "official-predictions", "baseline-replay",
                    "official_check", "official-check", "predictions_official_truth"):
            assert tok not in src, (name, tok)
        for s in LATER_DATES:
            assert s not in src, (name, f"later window {s}")
    assert core.scan(SPEC_SRC + "\n" + RUNNER_SRC + "\n" + SPEC.HYPOTHESIS) == []


def test_no_foreign_experiment_reads():
    assert "research/expos/runs/E" not in RUNNER_SRC
    assert "RUNS /" not in RUNNER_SRC and "RUNS/" not in RUNNER_SRC
    assert "E0003" not in RUNNER_SRC and "E0117" not in RUNNER_SRC


def test_full_rule_preregistered_no_held_claim():
    assert "rc==0" in SPEC.FULL_RULE and "poison" in SPEC.FULL_RULE and "E0001" in SPEC.FULL_RULE
    assert "F1>=0.30" in SPEC.FULL_RULE and "no freeze" in SPEC.HYPOTHESIS.lower()
    assert "6h F1" in SPEC.FULL_RULE  # 6h F1 is screening only
    assert "3h" in SPEC.FULL_RULE  # F1 claimed only at 3h density
    assert "no freeze" in SPEC_SRC.lower()
    assert "official_score(" not in RUNNER_SRC and "core.freeze" not in RUNNER_SRC
    assert "official-score" in SPEC.FULL_RULE
    assert "E0003" in SPEC.FULL_RULE and "seed-repro" in SPEC.FULL_RULE


def test_predictions_three_kinds_and_poison():
    assert "predictions_{dev,selection,audit}" in RUNNER_SRC  # output contract
    assert "write(te, pred, name)" in RUNNER_SRC  # per-period full-grid write
    assert "predictions_official" not in RUNNER_SRC
    assert "official" not in RUNNER_SRC.lower()
    assert "freeze" not in RUNNER_SRC.lower()
    assert "test_grid" in RUNNER_SRC and "prediction_invariant" in RUNNER_SRC
    assert "poison_check" in RUNNER_SRC and "positive_control" in RUNNER_SRC
    assert "passed" in RUNNER_SRC and "poison.json" in RUNNER_SRC
    assert "SystemExit" in RUNNER_SRC  # fail-closed on poison/non-finite


def test_parallel_cache_and_budget():
    assert "ThreadPoolExecutor" in RUNNER_SRC
    assert "fcache_key" in RUNNER_SRC and "featcache" in RUNNER_SRC
    assert SPEC.PERIOD_JOBS == 3
    assert SPEC.PERIOD_JOBS * SPEC.FIT_NJOBS <= 20
    assert SPEC.FREQ_PILOT == "6h" and SPEC.FREQ_FULL == "3h"


def test_dev_only_selection_shape():
    assert "no selection/audit statistic" in SPEC.SELECT_RULE
    assert "no fitted choice" in SPEC.SELECT_RULE.lower()
    assert "72h purge" in SPEC.SELECT_RULE
    assert "fitted_choice" in RUNNER_SRC
    assert "rule record" in SPEC.SELECT_RULE
    assert "no full run" in SPEC.FULL_RULE.lower() or "no full run" in SPEC.FULL_RULE


def test_owned_dir_is_small():
    files = sorted(p.name for p in (LB / "experiments" / "d125_xgbrf").iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"xrspec.py", "runner.py"} <= set(files)


def test_xgbrf_unit_fit_smoke():
    """Extreme-preserving XGBRF head splits on synthetic data on CPU."""
    import xgboost as xgb

    rng = np.random.default_rng(7)
    X = rng.standard_normal((288, 6))
    y = 400.0 + 30.0 * X[:, 0] + rng.standard_normal(288) * 10.0
    p = SPEC.fit_params()
    m = xgb.XGBRFRegressor(**dict(p, n_estimators=20)).fit(X, y)
    pred = np.asarray(m.predict(X), dtype=float)
    assert np.isfinite(pred).all() and float(np.std(pred)) > 0
    dump = m.get_booster().get_dump(dump_format="text")
    assert sum(t.count("[f") for t in dump) > 0, "trees must split"
