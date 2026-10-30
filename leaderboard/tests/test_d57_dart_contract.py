"""D57 dart contract: hypothesis tags, single-axis dart dropout, twin spec, isolation.

Covers the assignment: [family=gbm] [axes: opt=dart-dropout] leading dual tag,
E0006/X0037 citation, dart-dropout as the single preregistered variant
(monotone untouched, depth 6 unchanged), E0001-identical rows/purge/budget,
GPU-1 three-period parallelism within the shared n_jobs budget, scratch
isolation to worker-d57, no forbidden paths, preregistered full rule, three
prediction kinds plus real poison, no freeze/official-score.
"""
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d57_dart"))
sys.path.insert(0, str(LB))

import dart as DA  # noqa: E402
from expos import core  # noqa: E402

RUNNER_SRC = (LB / "experiments" / "d57_dart" / "runner.py").read_text()
DART_SRC = (LB / "experiments" / "d57_dart" / "dart.py").read_text()


def test_hypothesis_dual_tag_and_experience():
    assert DA.HYPOTHESIS.startswith("[family=gbm] [axes: opt=dart-dropout]")
    assert "E0006" in DA.HYPOTHESIS and "X0037" in DA.HYPOTHESIS
    assert DA.EXPERIENCE == "X0037"
    assert DA.DIRECTION == "D57" and DA.OWNER == "worker-d57"
    assert "monotone" in DA.HYPOTHESIS  # rejected candidates named
    assert "6h" in DA.HYPOTHESIS  # pilot grid stated


def test_single_axis_dart_dropout_only():
    p = DA.fit_params("cuda:0")
    assert p["booster"] == "dart"
    assert p["rate_drop"] == 0.10 and p["skip_drop"] == 0.50
    assert p["sample_type"] == "uniform" and p["normalize_type"] == "tree"
    # E0001 tree structure untouched: the axis is opt only
    assert p["max_depth"] == 6 and p["min_child_weight"] == 20
    assert p["reg_lambda"] == 5.0 and p["n_estimators"] == 600
    assert p["learning_rate"] == 0.05 and p["subsample"] == 0.8
    assert p["colsample_bytree"] == 0.7
    assert "monotone_constraints" not in p
    assert "monotone_constraints" not in RUNNER_SRC
    assert type(__import__("xgboost").XGBRegressor(**p)).__name__ == "XGBRegressor"


def test_e0001_twin_rows_purge_budget():
    assert DA.PURGE_HOURS == 72
    assert DA.FREQ_PILOT == "6h" and DA.FREQ_FULL == "3h"
    assert DA.N_ESTIMATORS == 600 and DA.SEED == 0
    assert DA.check_twin()["booster"] == "dart"
    assert DA.check_twin()["experience"] == "X0037"


def test_gpu_budget_and_devices(monkeypatch):
    assert DA.PERIOD_JOBS * DA.FIT_NJOBS <= 10
    assert DA.FIT_NJOBS <= 2
    monkeypatch.setenv("EXPOS_GPUS", "1")
    assert DA.device_for(0) == "cuda"
    monkeypatch.setenv("EXPOS_GPUS", "0,1")
    assert DA.device_for(1) == "cuda:1" and DA.device_for(2) == "cuda:0"


def test_scratch_isolation():
    assert "store/scratch/worker-d57" in RUNNER_SRC
    others = [l for l in RUNNER_SRC.splitlines() if "worker-" in l and "worker-d57" not in l]
    assert others == [], f"scratch leak: {others[:3]}"
    assert "featcache" in RUNNER_SRC and "fcache_key" in RUNNER_SRC


def test_no_forbidden_paths_in_registered_code():
    assert core.scan(RUNNER_SRC + "\n" + DART_SRC) == []
    assert core.scan(DA.HYPOTHESIS) == []


def test_full_rule_preregistered_no_freeze_claim():
    assert "rc==0" in DA.FULL_RULE and "poison" in DA.FULL_RULE and "E0001" in DA.FULL_RULE
    assert "F1>=0.35" in DA.FULL_RULE and "No freeze" in DA.HYPOTHESIS
    assert "No freeze" in DART_SRC
    assert "official_score(" not in RUNNER_SRC and "core.freeze" not in RUNNER_SRC
    assert "official-score" in DA.HYPOTHESIS or "official_score" in DA.HYPOTHESIS


def test_predictions_three_kinds_and_poison():
    assert "predictions_{dev,selection,audit}" in RUNNER_SRC  # output contract
    assert "write(te, pred, name)" in RUNNER_SRC  # per-period full-grid write
    assert "predictions_official" not in RUNNER_SRC
    assert "poison_check" in RUNNER_SRC and "positive_control" in RUNNER_SRC
    assert "passed" in RUNNER_SRC and "poison.json" in RUNNER_SRC


def test_owned_dir_is_small():
    files = sorted(p.name for p in (LB / "experiments" / "d57_dart").iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"dart.py", "runner.py"} <= set(files)
