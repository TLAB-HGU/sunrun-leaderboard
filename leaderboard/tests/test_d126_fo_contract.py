"""D126 forest contract: dual tags, extreme-preserving single head, CPU twin, isolation.

Fast static + unit checks only: no data reads, no fits beyond one small
synthetic CPU fit, no held-window contact. The pilot run itself (registered
separately) proves end-to-end behaviour through expos run/evaluate/poison.
"""
import sys
from pathlib import Path

import numpy as np

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d126_forest"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
sys.path.insert(0, str(LB))

import fospec as SPEC  # noqa: E402
import info_ablation as ia  # noqa: E402
from expos import core  # noqa: E402

RUNNER_SRC = (LB / "experiments" / "d126_forest" / "runner.py").read_text()
SPEC_SRC = (LB / "experiments" / "d126_forest" / "fospec.py").read_text()

COPY_BANS = ["def speed_features(", "def ace_features(", "def ch_features(",
             "def enlil_features(", "def build(", "def arms(", "def full_grid(",
             "def write("]
LATER_DATES = ("2025-12-04", "2026-02-25", "2026-03-04", "2026-05-28",
               "2026-05-29", "2026-05-31", "2026-09-26")


def test_hypothesis_dual_tag_and_experience():
    assert SPEC.HYPOTHESIS.startswith("[family=rf] [axes: model=forest]")
    assert SPEC.FAMILY_TAG == "[family=rf]"
    assert SPEC.AXES_TAG.startswith("[axes:") and "model=" in SPEC.AXES_TAG
    assert "forest" in SPEC.AXES_TAG
    assert SPEC.EXPERIENCE == ["X0112", "X0109"]
    assert "X0112" in SPEC.HYPOTHESIS and "X0109" in SPEC.HYPOTHESIS
    assert "E0123" in SPEC.HYPOTHESIS  # replicated forest pilot id stated
    assert SPEC.DIRECTION == "D126" and SPEC.OWNER == "worker-d126"
    assert "no freeze" in SPEC.HYPOTHESIS.lower()
    assert "6h" in SPEC.HYPOTHESIS  # pilot grid stated
    assert "screening" in SPEC.HYPOTHESIS  # 6h F1 is screening only
    assert "held window" in SPEC.HYPOTHESIS


def test_extreme_preserving_single_head_params():
    p = SPEC.fit_params()
    assert SPEC.ESTIMATOR == "RandomForestRegressor"
    assert p["max_depth"] >= 10 and p["min_samples_leaf"] == 1
    assert p["max_features"] == 1.0 and p["bootstrap"] is True
    assert p["n_estimators"] == 300 and p["criterion"] == "squared_error"
    assert p["random_state"] == 0 and p["n_jobs"] <= 2
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(**p)
    assert type(m).__name__ == "RandomForestRegressor"


def test_single_axis_model_only():
    p = SPEC.fit_params()
    assert p["random_state"] == ia.PARAMS["random_state"] == 0
    assert p["criterion"] == "squared_error"  # loss axis untouched
    assert "learning_rate" not in p and "colsample_bynode" not in p
    assert "sample_weight" not in RUNNER_SRC
    assert "EPAM_COLS" not in RUNNER_SRC and "epam" not in RUNNER_SRC.lower()
    assert "blend" not in RUNNER_SRC.lower()
    assert "EVENT_THRESHOLD" not in RUNNER_SRC
    assert RUNNER_SRC.count(".fit(") == 1, "exactly one head fit (single head, no refit)"
    assert 'ia.arms(df.columns)["existing_all"]' in RUNNER_SRC  # no new columns
    assert SPEC.check_spec()["axes"] == "[axes: model=forest]"


def test_e0001_twin_rows_purge_budget():
    assert SPEC.PURGE_HOURS == 72
    assert SPEC.FREQ_PILOT == "6h" and SPEC.FREQ_FULL == "3h"
    assert SPEC.N_ESTIMATORS == 300 and SPEC.SEED == 0
    assert "SPEC.PURGE_HOURS" in RUNNER_SRC
    assert "build_train" in RUNNER_SRC and "test_grid" in RUNNER_SRC
    assert "design_cols" in RUNNER_SRC
    assert "check_train_memory" in RUNNER_SRC


def test_cpu_budget_no_accelerator():
    assert SPEC.PERIOD_JOBS * SPEC.FIT_NJOBS <= 10
    assert SPEC.PERIOD_JOBS == 3, "three periods run in parallel"
    assert SPEC.FIT_NJOBS <= 2
    assert "device" not in SPEC.fit_params()  # sklearn is CPU-only, no device knob
    for pat in ("gpu_lease", "--gpu", "prefer-gpu", "EXPOS_GPUS",
                "CUDA_VISIBLE_DEVICES", "device_for"):
        assert pat not in RUNNER_SRC, pat
        assert pat not in SPEC_SRC, pat
    assert "CPU" in RUNNER_SRC
    assert "D124" in SPEC.CPU_REASON and "D127" in SPEC.CPU_REASON
    assert "gpu0" in SPEC.CPU_REASON and "gpu1" in SPEC.CPU_REASON
    assert "60s" in SPEC.CPU_REASON and "foreground" in SPEC.CPU_REASON
    assert "lease" in SPEC.CPU_REASON


def test_scratch_isolation():
    assert "store/scratch/worker-d126" in RUNNER_SRC
    others = [l for l in RUNNER_SRC.splitlines() if "worker-" in l and "worker-d126" not in l]
    assert others == [], f"scratch leak: {others[:3]}"
    assert "featcache" in RUNNER_SRC and "fcache_key" in RUNNER_SRC
    assert SPEC.SCRATCH_NAME == "worker-d126"


def test_readonly_no_vendored_copies():
    for s in COPY_BANS:
        assert s not in RUNNER_SRC, f"runner must import, not copy: {s!r}"
        assert s not in SPEC_SRC, f"spec must not vendor: {s!r}"
    assert "from existing_all import" in RUNNER_SRC


def test_no_forbidden_sources_or_later_windows():
    for name, src in (("fospec.py", SPEC_SRC), ("runner.py", RUNNER_SRC)):
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
    assert "E0123" not in RUNNER_SRC and "E0199" not in RUNNER_SRC


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
    files = sorted(p.name for p in (LB / "experiments" / "d126_forest").iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"fospec.py", "runner.py"} <= set(files)


def test_forest_unit_fit_smoke():
    """Extreme-preserving RandomForest head splits on synthetic data on CPU."""
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(7)
    X = rng.standard_normal((288, 6))
    y = 400.0 + 30.0 * X[:, 0] + rng.standard_normal(288) * 10.0
    p = SPEC.fit_params()
    m = RandomForestRegressor(**dict(p, n_estimators=20)).fit(X, y)
    pred = np.asarray(m.predict(X), dtype=float)
    assert np.isfinite(pred).all() and float(np.std(pred)) > 0
    assert m.estimators_[0].tree_.node_count > 1, "trees must split"
    assert abs(float(np.sum(m.feature_importances_)) - 1.0) < 1e-6
