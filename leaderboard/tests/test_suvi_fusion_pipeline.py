import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.xgboost_suvi_fusion import report, run


def fixture_data():
    index = pd.date_range("2026-06-01", periods=74, freq="h", tz="UTC")
    series = pd.DataFrame({"timestamp_utc": index, "filled_speed_kms": 700.0, "was_missing": 0})
    series.loc[0, "filled_speed_kms"] = 400.0
    folds = pd.DataFrame({"origin_last_input_utc": [index[0]]})
    prediction = pd.DataFrame({"origin_last_input_utc": index[0], "horizon_hours": range(1, 73), "pred_kms": 700.0})
    return series, folds, prediction


def test_observed_peak_and_rise_metrics_and_gates():
    series, folds, candidate = fixture_data()
    reference = candidate.assign(pred_kms=400.0)
    result = report.evaluate_predictions({"fusion": candidate, "ace_only": reference, "leader": reference}, series, folds)
    assert result["eligible_for_submission"]
    metrics = result["models"]["fusion"]
    assert metrics["official"]["mse"] == 0
    assert metrics["peaks"]["by_horizon"]["24"]["rise"]["recall"] == 1
    series.loc[24, "was_missing"] = 1
    result = report.evaluate_predictions({"fusion": candidate, "leader": reference}, series, folds)
    assert not result["gates"]["leader"]["passed"]
    assert result["models"]["fusion"]["peaks"]["by_horizon"]["24"]["peak_count"] == 0


def test_invalid_predictions_fail_before_scoring():
    series, folds, candidate = fixture_data()
    with pytest.raises(ValueError, match="required rows missing"):
        report.evaluate_predictions({"fusion": candidate.iloc[:71]}, series, folds)


def test_ace_fill_is_forward_only_and_tracks_missingness(tmp_path):
    path = tmp_path / "ace.csv"
    pd.DataFrame({"timestamp_utc": ["2025-01-01T00:00Z", "2025-01-01T02:00Z"],
                  "ace_speed_kms": [400, 600]}).to_csv(path, index=False)
    result = run.load_ace([str(path)])
    assert result.filled_speed_kms.tolist() == [400, 400, 600]
    assert result.was_missing.tolist() == [0, 1, 0]


def test_evaluate_cli_writes_report_with_manifest(tmp_path):
    series, folds, pred = fixture_data()
    for name, frame in (("ace", series), ("folds", folds), ("pred", pred)):
        frame.to_parquet(tmp_path / f"{name}.parquet")
    out = tmp_path / "result.json"
    assert run.main(["--mode", "evaluate", "--ace", str(tmp_path / "ace.parquet"),
                     "--folds", str(tmp_path / "folds.parquet"), "--predictions", str(tmp_path / "pred.parquet"),
                     "--output", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["models"]["fusion"]["official"]["n_pairs"] == 72
    assert not result["eligible_for_submission"]
    assert len(result["input_manifest"]) == 3


def test_final_holdout_failure_never_runs_leaderboard(monkeypatch, tmp_path):
    from experiments.xgboost_suvi_fusion import modeling
    index = pd.date_range("2026-02-01", "2026-05-31T23:00", freq="h", tz="UTC")
    ace = pd.DataFrame({"timestamp_utc": index, "filled_speed_kms": 700.0, "was_missing": 0})
    frame = pd.DataFrame({"ace_lag_0": 700.0, "suvi_disk": 1.0}, index=index)
    cutoffs = []

    class Forecaster:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, features, target, cutoff, **kwargs):
            cutoffs.append(cutoff)

        def predict(self, features):
            return pd.DataFrame({"origin_last_input_utc": np.repeat(features.index, 72),
                                 "horizon_hours": np.tile(np.arange(1, 73), len(features)), "pred_kms": 400.0})

    monkeypatch.setattr(modeling, "DirectForecaster", Forecaster)
    ace_path = tmp_path / "ace.parquet"
    suvi_path = tmp_path / "suvi.parquet"
    cache_path = tmp_path / "cache.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(ace_path)
    pd.DataFrame({"value": [1]}).to_parquet(suvi_path)
    pd.DataFrame({"value": [1]}).to_parquet(cache_path)
    config = {"data": {"median_fit_cutoff": "2025-07-31T23:00:00Z", "train_end_utc": "2026-05-31T23:00:00Z",
                       "leaderboard_revision": "bcf5d5417d99eaa331ddca40581d48953e09a58e",
                       "ace": [run.file_identity(ace_path)],
                       "suvi_metadata": run.file_identity(suvi_path),
                       "suvi_feature_cache": run.file_identity(cache_path)},
              "hyperparameters": {"estimator": {}, "peak_weight": 1, "rise_weight": 1,
                                  "rounds_by_horizon": {str(i): 1 for i in range(1, 73)}}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"config": config}))
    output = tmp_path / "prediction.parquet"
    args = run.parser().parse_args([
        "--mode", "final", "--ace", str(ace_path), "--suvi-meta", str(suvi_path),
        "--feature-cache", str(cache_path), "--config", str(path), "--output", str(output),
        "--device", "cpu",
    ])
    run.run_model(args, ace, frame)
    assert cutoffs == ["2026-02-28T23:00:00Z"] * 2
    result = json.loads(output.with_suffix(".report.json").read_text())
    assert result["status"] == "holdout_gate_failed"
    assert not result["eligible_for_submission"]
    assert not output.exists()
    assert not output.with_suffix(".submission.json").exists()
