import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scorer"))
from scoring import HORIZON, KEY, _events, score  # noqa: E402


def plateau(start, end, base=400.0, high=650.0):
    """72h series at `base` with hours start..end (0-based) at `high` and a single peak in the middle."""
    v = np.full(HORIZON, base)
    v[start:end + 1] = high
    v[(start + end) // 2] = high + 50
    return v


def frames(truth_windows, pred_windows, periods):
    origins = [f"2026-0{3 + i}-01T00:00:00Z" for i in range(len(truth_windows))]
    folds = pd.DataFrame({"origin_last_input_utc": origins, "regime": "low_low_low", "period": periods})
    keys = {"origin_last_input_utc": np.repeat(origins, HORIZON), "horizon_hours": np.tile(np.arange(1, HORIZON + 1), len(origins))}
    truth = pd.DataFrame({**keys, "target_kms": np.concatenate(truth_windows), "was_missing": 0})
    pred = pd.DataFrame({**keys, "pred_kms": np.concatenate(pred_windows)})
    return pred, truth, folds


def test_events_merge_short_gaps_and_censor_window_edges():
    v = plateau(10, 20)
    v[23:25] = 600.0  # 2 quiet hours after hour 20: merged into the first event
    v[33:38] = 600.0  # 8 quiet hours after hour 24: a separate event, peak at its first maximum
    assert _events(v) == [15, 33]
    assert _events(plateau(0, 8)) == [] and _events(plateau(60, 71)) == []  # cut by the window: extent unknown


def test_shifted_hit_gives_csi_one_and_signed_timing():
    pred, truth, folds = frames([plateau(20, 30)], [plateau(23, 33)], ["F2"])
    r = score(pred, truth, folds)
    assert (r["events"][""]["tp"], r["events"][""]["fp"], r["events"][""]["fn"]) == (1, 0, 0)
    assert r["csi"] == 1.0 and r["timing_mae_h"] == pytest.approx(3) and r["timing_bias_h"] == pytest.approx(3)


def test_miss_false_alarm_and_late_beyond_tolerance():
    flat = np.full(HORIZON, 400.0)
    pred, truth, folds = frames([plateau(20, 30), flat, plateau(10, 14)],
                                [flat, plateau(40, 50), plateau(40, 44)], ["F1", "F1", "F2"])
    r = score(pred, truth, folds)
    ev = r["events"]
    assert (ev["_f1"]["tp"], ev["_f1"]["fp"], ev["_f1"]["fn"]) == (0, 1, 1)  # one miss, one false alarm
    assert (ev["_f2"]["tp"], ev["_f2"]["fp"], ev["_f2"]["fn"]) == (0, 1, 1)  # 30h late: unmatched both ways
    assert r["csi"] == 0.0 and r["timing_mae_h"] is None
    assert ev[""]["pod"] == 0.0 and ev[""]["far"] == 1.0 and ev[""]["n_windows"] == 3


def test_period_split_and_no_event_windows():
    flat = np.full(HORIZON, 400.0)
    pred, truth, folds = frames([plateau(20, 30), flat], [plateau(18, 28), flat], ["F1", "F2"])
    r = score(pred, truth, folds)
    assert r["csi_f1"] == 1.0 and r["timing_bias_h_f1"] == pytest.approx(-2)  # 2h early
    assert r["csi_f2"] is None and r["timing_mae_h_f2"] is None  # no event anywhere in F2
    assert r["csi"] == 1.0
