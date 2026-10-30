"""D55 RF contract: hypothesis tags, extreme-preserving single head, twin spec, isolation.

Fast static + unit checks only: no data reads, no fits, no GPU, no official
contact. The pilot run itself (registered separately) proves end-to-end
behaviour through expos run/evaluate/poison.
"""
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d55_rf"))
sys.path.insert(0, str(LB))

import rf as RF  # noqa: E402
from expos import core  # noqa: E402

RUNNER_SRC = (LB / "experiments" / "d55_rf" / "runner.py").read_text()
RF_SRC = (LB / "experiments" / "d55_rf" / "rf.py").read_text()


def test_hypothesis_dual_tag_and_experience():
    assert RF.HYPOTHESIS.startswith("[family=rf] [axes: model=rf]")
    assert RF.EXPERIENCE == "X0037" and "X0037" in RF.HYPOTHESIS
    assert RF.DIRECTION == "D55" and RF.OWNER == "worker-d55"


def test_extreme_preserving_single_head_params():
    p = RF.fit_params("cuda:0")
    assert RF.ESTIMATOR == "XGBRFRegressor"
    assert p["max_depth"] >= 10 and p["min_child_weight"] <= 2
    assert p["n_estimators"] == 600 and p["tree_method"] == "hist"
    assert p["device"] == "cuda:0" and p["random_state"] == 0
    assert p["n_jobs"] <= 2
    import xgboost as xgb
    m = xgb.XGBRFRegressor(**p)
    assert type(m).__name__ == "XGBRFRegressor"


def test_e0001_twin_rows_features_purge():
    assert RF.PURGE_HOURS == 72
    assert RF.FREQ_PILOT == "6h" and RF.FREQ_FULL == "3h"
    assert 'ia.arms(df.columns)["existing_all"]' in RUNNER_SRC
    assert "RF.PURGE_HOURS" in RUNNER_SRC
    assert RUNNER_SRC.count(".fit(") == 1, "exactly one head fit (single head, no refit)"
    assert RF.check_twin()["estimator"] == "XGBRFRegressor"


def test_gpu_budget_and_devices(monkeypatch):
    assert RF.PERIOD_JOBS * RF.FIT_NJOBS <= 10
    monkeypatch.delenv("EXPOS_GPUS", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert RF.device_for(0) == "cuda" and RF.device_for(2) == "cuda"
    monkeypatch.setenv("EXPOS_GPUS", "0,1")
    assert RF.device_for(1) == "cuda:1" and RF.device_for(2) == "cuda:0"


def test_scratch_isolation():
    assert "store/scratch/worker-d55" in RUNNER_SRC
    others = [l for l in RUNNER_SRC.splitlines() if "worker-" in l and "worker-d55" not in l]
    assert others == [], f"scratch leak: {others[:3]}"


def test_no_forbidden_paths_in_registered_code():
    assert core.scan(RUNNER_SRC + "\n" + RF_SRC) == []
    assert core.scan(RF.HYPOTHESIS) == []


def test_full_rule_preregistered_no_freeze_claim():
    assert "rc==0" in RF.FULL_RULE and "poison" in RF.FULL_RULE and "E0001" in RF.FULL_RULE
    assert "no" in RF.FULL_RULE and "freeze" in RF.FULL_RULE.lower()
    assert "selection.json is a rule record" in RF.SELECT_RULE or "rule record" in RF.SELECT_RULE
    assert "No freeze" in RF_SRC
