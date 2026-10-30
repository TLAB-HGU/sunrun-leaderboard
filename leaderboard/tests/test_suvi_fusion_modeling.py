import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.xgboost_suvi_fusion.modeling import (
    DirectForecaster,
    ensure_device,
    horizon_training_data,
    passes_gate,
    prediction_metrics,
    sample_weights,
    select_feasible_trial,
    split_origins,
)
from experiments.xgboost_suvi_fusion.tune import run_study


def sample():
    index = pd.date_range("2025-01-01", periods=240, freq="h", tz="UTC")
    target = pd.Series(450 + 100 * np.sin(np.arange(240) / 20), index=index)
    return pd.DataFrame({"last": target, "lag1": target.shift(1)}, index=index), target


def test_cpu_device_preflight_is_a_noop():
    assert ensure_device("cpu") is None


def test_label_cutoff_uses_target_time_and_exact_timestamp():
    x, y = sample()
    train, labels = horizon_training_data(x, y, 72, y.index[150])
    assert train.index[-1] == y.index[78]
    assert labels[-1] == y.iloc[150]
    changed = y.copy()
    changed.iloc[151:] = 9999
    _, labels2 = horizon_training_data(x, changed, 72, y.index[150])
    np.testing.assert_array_equal(labels, labels2)
    sparse = y.drop(y.index[73])
    tx, _ = horizon_training_data(x, sparse, 72, y.index[150])
    assert y.index[1] not in tx.index


def test_event_weights_and_complete_split():
    np.testing.assert_array_equal(sample_weights(np.array([650, 550, 650]),
                                                np.array([600, 400, 400]), 2, 3), [2, 3, 6])
    x, _ = sample()
    selected = split_origins(x.index, x.index[10], x.index[100])
    assert selected[0] == x.index[10]
    assert selected[-1] == x.index[28]


def test_observed_metrics_exclude_imputed_high_speed():
    index = pd.date_range("2025-01-01", periods=100, freq="h", tz="UTC")
    target = pd.Series(400., index=index)
    target.iloc[[24, 48, 72]] = 650
    pred = pd.DataFrame({"origin_last_input_utc": [index[0]] * 3,
                         "horizon_hours": [24, 48, 72], "pred_kms": [650., 650., 650.]})
    observed = pd.Series(True, index=index)
    assert prediction_metrics(pred, target, observed)["peak_macro_f1"] == 1
    observed.iloc[48] = False
    result = prediction_metrics(pred, target, observed)
    assert result["peak_macro_f1"] is None
    assert result["by_horizon"][48]["peak_count"] == 0


def test_feasibility_requires_all_gates_and_stable_ties():
    reference = {"mse": 100., "mse_observed": 100., "peak_macro_f1": .2, "peak_bias": -50.}
    good = {"mse": 90., "mse_observed": 90., "peak_macro_f1": .25, "peak_bias": -39.}
    assert passes_gate(good, reference, require_observed=True)
    assert not passes_gate(dict(good, peak_bias=None), reference)
    assert not passes_gate(dict(good, mse_observed=101), reference, require_observed=True)
    trials = [{"number": 5, "metrics": good}, {"number": 2, "metrics": good}]
    assert select_feasible_trial(trials, reference)["number"] == 2


def test_cpu_direct_models_reproduce_and_emit_72_horizons():
    pytest.importorskip("xgboost")
    x, y = sample()
    params = {"n_estimators": 2, "max_depth": 2}
    first = DirectForecaster(params).fit(x, y, y.index[150])
    second = DirectForecaster(params).fit(x, y, y.index[150])
    p = first.predict(x.iloc[[151, 152]])
    assert len(p) == 144
    assert p.groupby("origin_last_input_utc").horizon_hours.nunique().eq(72).all()
    pd.testing.assert_frame_equal(p, second.predict(x.iloc[[151, 152]]))


def test_validation_cannot_overlap_training():
    pytest.importorskip("xgboost")
    x, y = sample()
    with pytest.raises(ValueError, match="Validation must follow"):
        DirectForecaster({"n_estimators": 2}, horizons=[1]).fit(
            x, y, y.index[150], validation_features=x.iloc[100:160], validation_end=y.index[-1])


def test_early_stopping_rounds_can_be_reused_without_validation():
    pytest.importorskip("xgboost")
    x, y = sample()
    model = DirectForecaster({"n_estimators": 4}, horizons=[1, 2], early_stopping_rounds=2)
    model.fit(x.iloc[:120], y, y.index[119], validation_features=x.iloc[120:150],
              validation_end=y.index[160])
    rounds = model.best_rounds()
    refit = DirectForecaster(horizons=[1, 2]).fit(x, y, y.index[160], rounds_by_horizon=rounds)
    assert refit.best_rounds() == rounds


def test_hpo_rejects_holdout_access():
    pytest.importorskip("optuna")
    x, y = sample()
    with pytest.raises(ValueError, match="holdout"):
        run_study(x, y, validation_end="2026-03-01")


def test_sequential_sqlite_study_records_diagnostics_and_resumes(tmp_path):
    optuna = pytest.importorskip("optuna")
    pytest.importorskip("xgboost")
    x, y = sample()
    kwargs = {"storage": f"sqlite:///{tmp_path / 'study.db'}", "n_trials": 1,
              "n_estimators": 2, "train_end": y.index[99], "validation_start": y.index[100],
              "validation_end": y.index[-1]}
    study = run_study(x, y, **kwargs)
    assert len(study.trials) == 1
    assert isinstance(study.pruner, optuna.pruners.NopPruner)
    assert len(study.trials[0].user_attrs["rounds_by_horizon"]) == 72
    assert "rise_recall" in study.trials[0].user_attrs["metrics"]["by_horizon"]["24"]
    again = run_study(x, y, **dict(kwargs, n_trials=2))
    assert len(again.trials) == 2
    assert again.trials[0].params != again.trials[1].params
