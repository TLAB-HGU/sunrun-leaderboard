"""D63 lr-schedule contract: hypothesis tags, fixed decay schedule, twin spec, isolation.

Covers the assignment: [family=xgb] [axes: opt=lr-schedule] leading dual tag,
E0116/X0107 citation, fixed 600x0.05+600x0.02+400x0.008 decay schedule with more
trees as the single preregistered variant (no search), E0001-identical
rows/purge/budget, gpu0 three-period parallelism within the shared n_jobs
budget, scratch isolation to worker-d63, no forbidden paths, preregistered
full rule, three prediction kinds plus real poison, no freeze/official-score.
"""
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d63_lrsched"))
sys.path.insert(0, str(LB))

import lrsched as LR  # noqa: E402
from expos import core  # noqa: E402

RUNNER_SRC = (LB / "experiments" / "d63_lrsched" / "runner.py").read_text()
SPEC_SRC = (LB / "experiments" / "d63_lrsched" / "lrsched.py").read_text()


def test_hypothesis_dual_tag_and_experience():
    assert LR.HYPOTHESIS.startswith("[family=xgb] [axes: opt=lr-schedule]")
    assert "E0116" in LR.HYPOTHESIS and "X0107" in LR.HYPOTHESIS
    assert LR.EXPERIENCE == "X0107"
    assert LR.DIRECTION == "D63" and LR.OWNER == "worker-d63"
    assert "600x0.05" in LR.HYPOTHESIS and "600x0.02" in LR.HYPOTHESIS and "400x0.008" in LR.HYPOTHESIS
    assert "6h" in LR.HYPOTHESIS  # pilot grid stated


def test_schedule_preregistered_fixed_no_search():
    assert LR.SCHEDULE == ((600, 0.05), (600, 0.02), (400, 0.008))
    assert LR.TOTAL_TREES == 1600 == sum(n for n, _ in LR.SCHEDULE)
    assert LR.N_ESTIMATORS == 1600
    assert "Optuna" not in SPEC_SRC and "trial" not in SPEC_SRC.lower().replace("train", "")
    assert "search" not in SPEC_SRC.lower() or "never searched" in SPEC_SRC.lower()
    # staged continuation is the mechanism, not a search loop over lr values
    assert "xgb_model" in RUNNER_SRC
    assert "SMOKE_SCHEDULE" in SPEC_SRC
    assert LR.check_twin()["total_trees"] == 1600


def test_single_axis_lr_schedule_only():
    p = LR.stage_params("cuda", 600, 0.05)
    assert p["learning_rate"] == 0.05 and p["n_estimators"] == 600
    assert p["max_depth"] == 6 and p["min_child_weight"] == 20
    assert p["reg_lambda"] == 5.0 and p["subsample"] == 0.8
    assert p["colsample_bytree"] == 0.7
    assert p["tree_method"] == "hist" and p["random_state"] == 0
    assert "objective" not in p and "booster" not in p
    assert "monotone_constraints" not in p
    assert "monotone_constraints" not in RUNNER_SRC
    assert "sample_weight" not in RUNNER_SRC
    assert type(__import__("xgboost").XGBRegressor(**p)).__name__ == "XGBRegressor"


def test_e0001_twin_rows_purge_budget():
    assert LR.PURGE_HOURS == 72
    assert LR.FREQ_PILOT == "6h" and LR.FREQ_FULL == "3h"
    assert LR.N_ESTIMATORS == 1600 and LR.SEED == 0
    assert LR.BASE_LEARNING_RATE == 0.05
    assert LR.check_twin()["purge_h"] == 72
    assert LR.check_twin()["experience"] == "X0107"


def test_gpu_budget_and_devices(monkeypatch):
    assert LR.PERIOD_JOBS * LR.FIT_NJOBS <= 10
    assert LR.FIT_NJOBS <= 2
    assert LR.PERIOD_JOBS == 3
    monkeypatch.setenv("EXPOS_GPUS", "0")
    assert LR.device_for(0) == "cuda"
    monkeypatch.setenv("EXPOS_GPUS", "0,1")
    assert LR.device_for(1) == "cuda:1" and LR.device_for(2) == "cuda:0"
    assert "gpu0" in LR.GPU_REASON and "gpu1" in LR.GPU_REASON


def test_scratch_isolation():
    assert "store/scratch/worker-d63" in RUNNER_SRC
    others = [l for l in RUNNER_SRC.splitlines() if "worker-" in l and "worker-d63" not in l]
    assert others == [], f"scratch leak: {others[:3]}"
    assert "featcache" in RUNNER_SRC and "fcache_key" in RUNNER_SRC


def test_no_forbidden_paths_in_registered_code():
    assert core.scan(RUNNER_SRC + "\n" + SPEC_SRC) == []
    assert core.scan(LR.HYPOTHESIS) == []
    assert core.scan(LR.FULL_RULE) == []


def test_full_rule_preregistered_no_freeze_claim():
    assert "rc==0" in LR.FULL_RULE and "poison" in LR.FULL_RULE and "E0001" in LR.FULL_RULE
    assert "F1>=0.35" in LR.FULL_RULE and "No freeze" in LR.HYPOTHESIS
    assert "No freeze" in SPEC_SRC
    assert "official_score(" not in RUNNER_SRC and "core.freeze" not in RUNNER_SRC
    assert "official-score" in LR.HYPOTHESIS or "official_score" in LR.HYPOTHESIS
    assert "official-score" in LR.FULL_RULE or "official_score" in LR.FULL_RULE


def test_predictions_three_kinds_and_poison():
    assert "predictions_{dev,selection,audit}" in RUNNER_SRC  # output contract
    assert "write(te, pred, name)" in RUNNER_SRC  # per-period full-grid write
    assert "predictions_official" not in RUNNER_SRC
    assert "poison_check" in RUNNER_SRC and "positive_control" in RUNNER_SRC
    assert "passed" in RUNNER_SRC and "poison.json" in RUNNER_SRC
    assert "SystemExit" in RUNNER_SRC  # fail-closed on poison/non-finite


def test_owned_dir_is_small():
    files = sorted(p.name for p in (LB / "experiments" / "d63_lrsched").iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"lrsched.py", "runner.py"} <= set(files)
