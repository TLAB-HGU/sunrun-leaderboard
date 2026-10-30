"""D85 CatBoost contract: hypothesis tags, single-axis ordered engine, twin spec, isolation.

Fast: no data reads, no fits (imports only; CatBoost import is lazy inside
the runner's .venv-gbm+cblib interpreter, never required here).
"""
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d85_catboost"))
sys.path.insert(0, str(LB))

import cb as CB  # noqa: E402
from expos import core  # noqa: E402

RUNNER_SRC = (LB / "experiments" / "d85_catboost" / "runner.py").read_text()
CB_SRC = (LB / "experiments" / "d85_catboost" / "cb.py").read_text()


def test_hypothesis_dual_tag_and_experience():
    assert CB.HYPOTHESIS.startswith("[family=gbm] [axes: model=catboost]")
    assert "X0115" in CB.HYPOTHESIS and "E0127" in CB.HYPOTHESIS
    assert ".venv-gbm" in CB.HYPOTHESIS
    assert "6h" in CB.HYPOTHESIS  # pilot grid stated


def test_single_axis_ordered_catboost_only():
    gp, cp = CB.fit_params("gpu"), CB.fit_params("cpu")
    assert gp["boosting_type"] == "Ordered" and cp["boosting_type"] == "Ordered"
    assert gp["iterations"] == 600 and gp["depth"] == 6 and gp["learning_rate"] == 0.05
    assert gp["task_type"] == "GPU" and cp["task_type"] == "CPU"
    assert gp["allow_writing_files"] is False
    assert "Ordered" in CB.ENGINE_REASON or "ordered" in CB.ENGINE_REASON.lower()
    assert "CatBoostRegressor" in RUNNER_SRC
    assert "sample_weight" not in RUNNER_SRC
    assert "no blending" in RUNNER_SRC and "no averaging" in RUNNER_SRC


def test_e0001_twin_rows_purge_budget():
    assert CB.PURGE_HOURS == 72
    assert CB.FREQ_PILOT == "6h" and CB.FREQ_FULL == "3h"
    assert CB.check_twin()["experience"] == "X0115"
    assert "existing_all" in RUNNER_SRC
    assert "strips" not in RUNNER_SRC.lower()  # E0001 features only, no D1 strips


def test_env_isolation_scratch_cblib_venv_untouched():
    assert (LB / ".venv-gbm" / "bin" / "python").is_file()
    assert (LB / "store" / "scratch" / "worker-d85" / "cblib" / "catboost").is_dir()
    log = (LB / "experiments" / "d85_catboost" / "install.log").read_text()
    assert "catboost==1.2.10" in log and "untouched" in log
    for old in (".venv-fusion", ".venv-mse20", ".venv-dinov3", ".venv-vjepa", ".venv-poly"):
        assert old not in RUNNER_SRC and old not in CB_SRC
    assert 'CBLIB = SCRATCH / "cblib"' in RUNNER_SRC and "worker-d85" in RUNNER_SRC


def test_gpu_budget_and_threads():
    assert CB.PERIOD_JOBS * CB.FIT_NJOBS <= 10
    assert CB.FIT_NJOBS <= 2
    assert "ThreadPoolExecutor" in RUNNER_SRC
    assert "prefer-gpu 1" in CB.ENGINE_REASON or "prefer-gpu" in CB.ENGINE_REASON


def test_scratch_isolation():
    assert "store/scratch/worker-d85" in RUNNER_SRC
    for other in ("worker-d81", "worker-d82", "worker-d58", "worker-d54", "worker-d62"):
        assert other not in RUNNER_SRC and other not in CB_SRC
    assert "featcache" in RUNNER_SRC and "fcache_key" in RUNNER_SRC


def test_no_forbidden_paths_in_registered_code():
    assert core.scan(RUNNER_SRC + "\n" + CB_SRC) == []
    assert core.scan(CB.HYPOTHESIS) == []


def test_full_rule_preregistered_no_freeze_claim():
    assert "rc==0" in CB.FULL_RULE and "poison" in CB.FULL_RULE and "E0001" in CB.FULL_RULE
    assert "no freeze/official-score" in CB.FULL_RULE.lower()
    assert "official-score" in CB.HYPOTHESIS or "official_score" in CB.HYPOTHESIS


def test_predictions_three_kinds_and_poison():
    assert "predictions_{dev,selection,audit}" in RUNNER_SRC  # output contract
    assert "selection.json" in RUNNER_SRC and "importance.json" in RUNNER_SRC
    assert "passed" in RUNNER_SRC and "poison.json" in RUNNER_SRC


def test_owned_dir_is_small():
    files = sorted(p.name for p in (LB / "experiments" / "d85_catboost").iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"cb.py", "runner.py"} <= set(files)
