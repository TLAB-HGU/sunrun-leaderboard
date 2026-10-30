"""D64 recency-weight contract: hypothesis tags, single-axis data change, isolation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d64_recency"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
sys.path.insert(0, str(LB))

import recency as RC  # noqa: E402
import info_ablation as ia  # noqa: E402
from expos import core  # noqa: E402

RUNNER_SRC = (LB / "experiments" / "d64_recency" / "runner.py").read_text()
SPEC_SRC = (LB / "experiments" / "d64_recency" / "recency.py").read_text()


def test_hypothesis_dual_tag_and_experience():
    assert RC.HYPOTHESIS.startswith("[family=xgb] [axes: data=recency-weight]")
    assert "X0009" in RC.HYPOTHESIS
    assert "X0003" in RC.HYPOTHESIS and "X0104" in RC.HYPOTHESIS
    assert RC.EXPERIENCE[0] == "X0009"
    assert "365" in RC.HYPOTHESIS  # preregistered decay rule stated
    assert "freeze" in RC.HYPOTHESIS and "official-score" in RC.HYPOTHESIS


def test_single_axis_recency_weight_only():
    p = RC.fit_params()
    base = dict(ia.PARAMS)
    assert p["n_estimators"] == base["n_estimators"] == 600
    assert p["max_depth"] == base["max_depth"]
    assert p["learning_rate"] == base["learning_rate"]
    assert p["tree_method"] == base["tree_method"]
    assert str(p["device"]).startswith("cuda")
    assert "sample_weight" not in p  # weights are per-row data, not params
    assert type(__import__("xgboost").XGBRegressor(**p)).__name__ == "XGBRegressor"


def test_e0001_twin_rows_purge_budget():
    assert RC.PURGE_HOURS == 72
    assert RC.FREQ_PILOT == "6h" and RC.FREQ_FULL == "3h"
    assert RC.SEED == 0 and RC.N_ESTIMATORS == 600
    assert RC.PERIOD_JOBS * RC.FIT_NJOBS <= 20
    assert RC.check_twin()["half_life_days"] == 365.0


def test_weight_rule_causal_preregistered():
    cutoff = pd.Timestamp("2025-08-04", tz="UTC")
    t = pd.DatetimeIndex([cutoff, cutoff - pd.Timedelta(days=365),
                          cutoff - pd.Timedelta(days=730)])
    w = RC.recency_weights(t, cutoff)
    assert w[0] == 1.0  # newest row has full weight
    assert abs(w[1] - 0.5) < 1e-12  # half-life property
    assert abs(w[2] - 0.25) < 1e-12
    assert bool((np.diff(w) < 0).all())  # monotone in age
    assert bool(np.isfinite(w).all()) and bool((w > 0).all())
    # causal: weight depends only on age vs cutoff, shifting both changes nothing
    w2 = RC.recency_weights(t - pd.Timedelta(days=7), cutoff - pd.Timedelta(days=7))
    assert np.allclose(w, w2)
    assert RC.effective_sample_size(np.ones(100)) == 100.0


def test_gpu_budget_no_oversubscribe():
    assert RC.PERIOD_JOBS * RC.FIT_NJOBS <= 20
    assert "gpu1" in RC.GPU_REASON and "prefer-gpu 1" in RC.GPU_REASON
    assert RC.device_for(0) == "cuda"  # single visible GPU under --gpu lease


def test_scratch_isolation():
    assert "store/scratch/worker-d64" in RUNNER_SRC
    assert "worker-d62" not in RUNNER_SRC and "worker-d63" not in RUNNER_SRC
    assert "featcache" in RUNNER_SRC and "fcache_key" in RUNNER_SRC


def test_no_forbidden_paths_in_registered_code():
    assert core.scan(RUNNER_SRC + "\n" + SPEC_SRC) == []
    assert core.scan(RC.HYPOTHESIS) == []


def test_full_rule_preregistered_no_freeze_claim():
    assert "rc==0" in RC.FULL_RULE and "poison" in RC.FULL_RULE and "E0001" in RC.FULL_RULE
    assert "dev" in RC.FULL_RULE  # dev/OOF-only selection signal
    assert "freeze" in RC.FULL_RULE and "official-score" in RC.FULL_RULE


def test_predictions_three_kinds_and_poison():
    assert "predictions_{dev,selection,audit}" in RUNNER_SRC  # output contract
    assert "poison.json" in RUNNER_SRC and "positive_control" in RUNNER_SRC
    assert '"passed"' in RUNNER_SRC
    assert "sample_weight" in RUNNER_SRC
    assert "SystemExit" in RUNNER_SRC  # fail-closed on poison/non-finite


def test_owned_dir_is_small():
    files = sorted(p.name for p in (LB / "experiments" / "d64_recency").iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"recency.py", "runner.py"} <= set(files)
