import numpy as np
import optuna
import pandas as pd
import pytest

from experiments.xgboost_suvi_fusion.ch_experiment import final_gate, freeze_candidates
from experiments.xgboost_suvi_fusion.ch_hpo import (
    ARMS,
    _predict_task,
    checkpoint,
    completed_candidates,
    create_studies,
    select_columns,
    shortlist,
)
from experiments.xgboost_suvi_fusion.modeling import sample_weights


def test_ablation_feature_isolation_and_capped_weights():
    columns = ["ace_lag_0", "ch_t035_area", "ch_t045_area", "ch_t055_area", "suvi_old"]
    assert select_columns(columns, "ch_only") == ["ch_t045_area"]
    assert select_columns(columns, "ace_only") == ["ace_lag_0"]
    assert select_columns(columns, "ace_ch", .35) == ["ace_lag_0", "ch_t035_area"]
    np.testing.assert_equal(sample_weights(np.array([700., 500.]), np.array([400., 450.]), 3, 3, 5), [5, 1])
    with pytest.raises(ValueError, match="missing"):
        select_columns(["ace_lag_0"], "ch_only")


def test_sampler_resume_and_interrupted_trial_not_complete(tmp_path):
    ref = {"mse": 100, "peak_macro_f1": .2, "peak_bias": -100, "by_horizon": {24: {"f1": .2}}}
    studies = create_studies(tmp_path, "test", ref)
    trial = studies["ace_only"].ask()
    trial.suggest_float("x", 0, 1)
    checkpoint(tmp_path/"samplers.pkl", studies)
    reopened = create_studies(tmp_path, "test", ref)
    assert reopened["ace_only"].trials[0].state == optuna.trial.TrialState.FAIL
    assert completed_candidates(reopened)[1] == {a: [] for a in ARMS}
    expected = studies["ch_only"].sampler._rng.rng.get_state()[1]
    actual = reopened["ch_only"].sampler._rng.rng.get_state()[1]
    np.testing.assert_array_equal(expected, actual)


def test_resume_mismatch_fails_closed(tmp_path):
    studies = create_studies(tmp_path, "test", {})
    studies["ace_only"].ask()
    with pytest.raises(RuntimeError, match="checkpoint"):
        create_studies(tmp_path, "test", {})


def test_shortlist_and_selection_do_not_use_holdout():
    ref = {"mse": 100, "peak_macro_f1": .2, "peak_bias": -100}
    candidates = [{"number": i, "arm": "ch_only", "metrics": {"mse": 10+i,
                   "peak_macro_f1": f, "peak_bias": -50}} for i, f in enumerate([.1, .4, .3, .2])]
    assert [c["number"] for c in shortlist(candidates, ref)] == [0, 1, 2]
    rows = [{"metrics": c["metrics"], "events": {"passed": c["number"] != 0}} for c in candidates]
    frozen, selected = freeze_candidates(rows, candidates)
    assert frozen["ch_only"]["number"] == 1
    assert selected == "ch_only"


def test_new_50percent_gate():
    candidate = {"official": {"mse": 50.01}, "events": {"passed": True}}
    assert not final_gate(candidate, 100)
    candidate["official"]["mse"] = 50
    assert final_gate(candidate, 100)
    candidate["events"]["passed"] = False
    assert not final_gate(candidate, 100)


def test_real_worker_split_and_output(tmp_path):
    import time
    times = pd.date_range("2025-07-20", periods=24*28, freq="h", tz="UTC")
    speed = 550+80*np.sin(np.arange(len(times))/12)
    ace = pd.DataFrame({"timestamp_utc": times, "filled_speed_kms": speed, "was_missing": 0})
    frame = pd.DataFrame({"ace_lag_0": speed, "ch_t045_area": .02}, index=times)
    ace.to_parquet(tmp_path/"ace.parquet")
    frame.to_parquet(tmp_path/"features.parquet")
    task = {"arm": "ch_only", "threshold": .45, "device": "cpu",
            "features": str(tmp_path/"features.parquet"), "ace": str(tmp_path/"ace.parquet"),
            "params": {"n_estimators": 3, "max_depth": 2}, "peak_weight": 1, "rise_weight": 1,
            "cutoff": "2025-07-31T23:00:00Z", "start": "2025-08-01", "end": "2025-08-16T23:00:00Z",
            "horizons": (24, 48, 72), "deadline": time.time()+60,
            "prediction_output": str(tmp_path/"pred.parquet")}
    result = _predict_task(task)
    assert result["feature_count"] == 1
    assert result["metrics"]["mse"] > 0
    p = pd.read_parquet(tmp_path/"pred.parquet")
    assert p.origin_last_input_utc.min() >= pd.Timestamp("2025-08-01", tz="UTC")
    assert (p.origin_last_input_utc+pd.to_timedelta(p.horizon_hours, unit="h")).max() <= pd.Timestamp(task["end"])


@pytest.mark.parametrize("arms", [ARMS, ("ace_only",), ("ch_only",), ("ace_ch",)])
def test_two_device_coordinator_balance_and_resume(tmp_path, monkeypatch, arms):
    import time
    from concurrent.futures import Future

    from experiments.xgboost_suvi_fusion import ch_hpo as h
    calls = []
    class Executor:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, worker, task):
            calls.append(task)
            future = Future()
            future.set_result({"metrics": {"mse": 10., "peak_macro_f1": .5, "peak_bias": -20.},
                               "rounds": {horizon: 3 for horizon in h.SCREEN_HORIZONS}})
            return future
    monkeypatch.setattr(h, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(h, "resources_available", lambda devices: True)
    paths = {arm: tmp_path/f"{arm}.parquet" for arm in ARMS}
    for path in paths.values():
        path.touch()
    ref = {"mse": 100., "peak_macro_f1": .2, "peak_bias": -100.}
    result = h.run_screen(paths, tmp_path/"ace.parquet", tmp_path/"studies", ref,
                          time.time()+60, ("cpu:0", "cpu:1"), max_trials=2, arms=arms)
    assert result["counts"] == {a: 2 for a in arms}
    assert [c["arm"] for c in calls] == list(arms)*2
    assert [c["device"] for c in calls] == ["cpu:0", "cpu:1"]*len(arms)
    for arm in arms:
        first = next(c for c in calls if c["arm"] == arm)
        assert first["peak_weight"] == first["rise_weight"] == 1
    calls.clear()
    reopened = h.run_screen(paths, tmp_path/"ace.parquet", tmp_path/"studies", ref,
                            time.time()+60, ("cpu:0", "cpu:1"), max_trials=2, arms=arms)
    assert calls == []
    assert reopened["counts"] == result["counts"]
    if arms != ARMS:
        with pytest.raises(ValueError, match="feature-set changed"):
            create_studies(tmp_path/"studies", "ch-v1", ref)
