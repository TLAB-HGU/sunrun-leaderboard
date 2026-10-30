"""D58 LightGBM contract: hypothesis tags, single-axis leaf-wise engine, twin spec, isolation.

Fast: no data reads, no fits (imports only; LightGBM import is lazy inside
the runner's .venv-gbm interpreter, never required here).
"""
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d58_newgbm"))
sys.path.insert(0, str(LB))

import gbm as GB  # noqa: E402
from expos import core  # noqa: E402

RUNNER_SRC = (LB / "experiments" / "d58_newgbm" / "runner.py").read_text()
GBM_SRC = (LB / "experiments" / "d58_newgbm" / "gbm.py").read_text()


def test_hypothesis_dual_tag_and_experience():
    assert GB.HYPOTHESIS.startswith("[family=gbm] [axes: model=lgbm-catboost]")
    assert "X0037" in GB.HYPOTHESIS and "E0006" in GB.HYPOTHESIS
    assert ".venv-gbm" in GB.HYPOTHESIS
    assert "6h" in GB.HYPOTHESIS  # pilot grid stated


def test_single_axis_lightgbm_only():
    p = GB.fit_params()
    assert p["num_leaves"] == 31 and p["max_depth"] == 6
    assert p["n_estimators"] == 600 and p["learning_rate"] == 0.05
    assert p["device"] == "cpu"  # No OpenCL device -> foreground, reason recorded
    assert "No OpenCL device" in GB.ENGINE_REASON
    assert "LGBMRegressor" in RUNNER_SRC
    assert "sample_weight" not in RUNNER_SRC
    assert "no blending" in RUNNER_SRC and "no averaging" in RUNNER_SRC


def test_e0001_twin_rows_purge_budget():
    assert GB.PURGE_HOURS == 72
    assert GB.FREQ_PILOT == "6h" and GB.FREQ_FULL == "3h"
    assert GB.check_twin()["experience"] == "X0037"
    assert "existing_all" in RUNNER_SRC
    assert "strips" not in RUNNER_SRC.lower()  # E0001 features only, no D1 strips


def test_env_isolation_new_venv_gbm_only():
    assert (LB / ".venv-gbm" / "bin" / "python").is_file()
    log = (LB / "experiments" / "d58_newgbm" / "install.log").read_text()
    assert "lightgbm==4.6.0" in log and "xgboost==2.1.4" in log
    for old in (".venv-fusion", ".venv-mse20", ".venv-dinov3", ".venv-vjepa", ".venv-poly"):
        assert old not in RUNNER_SRC and old not in GBM_SRC


def test_gpu_budget_and_threads():
    assert GB.PERIOD_JOBS * GB.FIT_NJOBS <= 10
    assert GB.FIT_NJOBS <= 2
    assert "ThreadPoolExecutor" in RUNNER_SRC


def test_scratch_isolation():
    assert "store/scratch/worker-d58" in RUNNER_SRC
    for other in ("worker-d54", "worker-d62", "worker-d57", "worker-d53", "worker-d56"):
        assert other not in RUNNER_SRC and other not in GBM_SRC
    assert "featcache" in RUNNER_SRC and "fcache_key" in RUNNER_SRC


def test_no_forbidden_paths_in_registered_code():
    assert core.scan(RUNNER_SRC + "\n" + GBM_SRC) == []
    assert core.scan(GB.HYPOTHESIS) == []


def test_full_rule_preregistered_no_freeze_claim():
    assert "rc==0" in GB.FULL_RULE and "poison" in GB.FULL_RULE and "E0001" in GB.FULL_RULE
    assert "no freeze/official-score" in GB.FULL_RULE.lower()
    assert "official-score" in GB.HYPOTHESIS or "official_score" in GB.HYPOTHESIS


def test_predictions_three_kinds_and_poison():
    assert "predictions_{dev,selection,audit}" in RUNNER_SRC  # output contract
    assert "selection.json" in RUNNER_SRC and "importance.json" in RUNNER_SRC
    assert "passed" in RUNNER_SRC and "poison.json" in RUNNER_SRC


def test_owned_dir_is_small():
    files = sorted(p.name for p in (LB / "experiments" / "d58_newgbm").iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"gbm.py", "runner.py"} <= set(files)
