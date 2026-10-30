"""D56 ET contract: hypothesis tags, extreme-preserving single head, twin spec, isolation.

Fast static + unit checks only: no data reads, no fits, no GPU, no official
contact. The pilot run itself (registered separately) proves end-to-end
behaviour through expos run/evaluate/poison.
"""
import sys
from pathlib import Path

import pytest

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d56_extratrees"))
sys.path.insert(0, str(LB))

import et as ETS  # noqa: E402
from expos import core  # noqa: E402

RUNNER_SRC = (LB / "experiments" / "d56_extratrees" / "runner.py").read_text()
ET_SRC = (LB / "experiments" / "d56_extratrees" / "et.py").read_text()


def test_hypothesis_dual_tag_and_experience():
    assert ETS.HYPOTHESIS.startswith("[family=rf] [axes: model=extratrees]")
    assert ETS.EXPERIENCE == "X0109" and "X0109" in ETS.HYPOTHESIS
    assert "E0117" in ETS.HYPOTHESIS
    assert ETS.DIRECTION == "D56" and ETS.OWNER == "worker-d56"


def test_extreme_preserving_single_head_params():
    p = ETS.fit_params()
    assert ETS.ESTIMATOR == "ExtraTreesRegressor"
    assert p["max_depth"] >= 12 and p["min_samples_leaf"] == 1
    assert p["min_samples_split"] == 2 and p["max_features"] == 1.0
    assert p["bootstrap"] is False and p["criterion"] == "squared_error"
    assert p["n_estimators"] == 300 and p["random_state"] == 0
    assert p["n_jobs"] <= 4 and "device" not in p and "cuda" not in str(p).lower()
    from sklearn.ensemble import ExtraTreesRegressor
    m = ExtraTreesRegressor(**p)
    assert type(m).__name__ == "ExtraTreesRegressor"


def test_row_memory_guard():
    est = ETS.check_train_memory(286776, 88)
    assert est == int(286776 * 88 * 8 * 3.0)
    with pytest.raises(SystemExit):
        ETS.check_train_memory(286776, 88, cap=1)


def test_e0001_twin_rows_features_purge():
    assert ETS.PURGE_HOURS == 72
    assert ETS.FREQ_PILOT == "6h" and ETS.FREQ_FULL == "3h"
    assert 'ia.arms(df.columns)["existing_all"]' in RUNNER_SRC
    assert "ETS.PURGE_HOURS" in RUNNER_SRC
    assert RUNNER_SRC.count(".fit(") == 1, "exactly one head fit (single head, no refit)"
    assert ETS.check_twin()["estimator"] == "ExtraTreesRegressor"


def test_cpu_budget_no_gpu():
    assert ETS.PERIOD_JOBS * ETS.FIT_NJOBS <= 20
    assert ETS.PERIOD_JOBS == 3, "three periods run in parallel"
    blob = RUNNER_SRC + "\n" + ET_SRC
    for pat in ("cuda", "gpu_lease", "--gpu", "prefer-gpu", "EXPOS_GPUS",
                "CUDA_VISIBLE_DEVICES", "XGBRFRegressor", "tree_method"):
        assert pat not in blob, pat
    assert "CPU" in RUNNER_SRC  # CPU-only guard documented


def test_scratch_isolation():
    assert "store/scratch/worker-d56" in RUNNER_SRC
    others = [l for l in RUNNER_SRC.splitlines() if "worker-" in l and "worker-d56" not in l]
    assert others == [], f"scratch leak: {others[:3]}"


def test_no_forbidden_paths_in_registered_code():
    assert core.scan(RUNNER_SRC + "\n" + ET_SRC) == []
    assert core.scan(ETS.HYPOTHESIS) == []


def test_full_rule_preregistered_no_freeze_claim():
    assert "rc==0" in ETS.FULL_RULE and "poison" in ETS.FULL_RULE and "E0001" in ETS.FULL_RULE
    assert "no" in ETS.FULL_RULE and "freeze" in ETS.FULL_RULE.lower()
    assert "official-score" in ETS.FULL_RULE
    assert "rule record" in ETS.SELECT_RULE
    assert "No freeze" in ET_SRC
