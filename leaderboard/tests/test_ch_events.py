import numpy as np
import pandas as pd
import pytest

from experiments.xgboost_suvi_fusion.events import (
    event_metrics,
    find_events,
    match_events,
)


def test_plateau_merge_and_boundary_censor():
    y = np.full(72, 400.)
    y[0:3] = 650
    y[20:23] = 700
    y[29:32] = 700  # six quiet samples: merge, earliest maximum wins
    y[70:] = 650
    events, censored = find_events(y)
    assert censored == 2
    assert events == [{"peak": 20, "height": 700., "start": 20, "end": 31}]


def test_maximum_matching_not_greedy_and_tolerance():
    actual = [{"peak": 10}, {"peak": 30}]
    predicted = [{"peak": 21}, {"peak": 0}]
    assert set(match_events(actual, predicted)) == {(0, 1), (1, 0)}
    assert match_events([{"peak": 20}], [{"peak": 32}]) == [(0, 0)]
    assert match_events([{"peak": 20}], [{"peak": 33}]) == []


def data():
    times = pd.date_range("2026-03-01", periods=24*18, freq="h", tz="UTC")
    y = pd.Series(400., index=times)
    for day in range(18):
        y.iloc[day*24+10:day*24+13] = [650, 700, 650]
    origins = times[::24][:-3]
    p = pd.concat([pd.DataFrame({"origin_last_input_utc": o, "horizon_hours": np.arange(1, 73),
                                "pred_kms": y.reindex(pd.date_range(o+pd.Timedelta(hours=1), periods=72, freq="h")).to_numpy()})
                   for o in origins], ignore_index=True)
    return p, y, pd.Series(True, index=times)


def test_event_success_unique_counts_and_bootstrap():
    p, y, observed = data()
    result = event_metrics(p, y, observed)
    assert result["passed"]
    assert result["precision"] == result["recall"] == 1
    assert result["height_mae_kms"] == result["time_mae_hours"] == 0
    assert result["unique_observed_events"] < result["tp"]
    assert result["ci95_7day_blocks"]["precision"] == [1., 1.]


def test_missing_windows_and_insufficient_events():
    p, y, observed = data()
    first = p.origin_last_input_utc.iloc[0]
    small = p[p.origin_last_input_utc.eq(first)]
    result = event_metrics(small, y, observed, missing_policy="strict")
    assert not result["passed"]
    assert result["unique_observed_events"] == 3
    observed.iloc[20] = False
    result = event_metrics(small, y, observed, missing_policy="strict")
    assert result["missing_windows"] == 1
    assert result["evaluated_windows"] == 0
    assert result["precision"] is None
    assert not result["passed"]


def test_user_forward_fill_policy_is_causal_and_reports_imputation():
    p, y, observed = data()
    observed.iloc[11] = False
    y.iloc[11] = 99999  # invalid/unobserved value must never create a peak
    result = event_metrics(p, y, observed)
    assert result["evaluated_windows"] == result["requested_windows"]
    assert result["imputed_pairs"] == 1
    assert result["missing_policy"] == "forward_fill"
    assert result["height_mae_kms"] < 100
    assert result["unique_observed_peak_timestamps"] == result["unique_observed_events"]


def test_duplicate_and_incomplete_predictions_rejected():
    p, y, observed = data()
    with pytest.raises(ValueError, match="duplicate"):
        event_metrics(pd.concat([p, p.iloc[:1]]), y, observed)
    with pytest.raises(ValueError, match="complete"):
        event_metrics(p.drop(index=0), y, observed)
