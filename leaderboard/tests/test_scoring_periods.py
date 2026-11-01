import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scorer"))
from scoring import HORIZON, KEY, score  # noqa: E402


def frames(periods):
    """One origin per period, 72h apart; targets alternate 500/600 so half are >= 550."""
    origins = [f"2026-0{1 + 3 * i}-01T00:00:00Z" for i in range(len(periods))]
    folds = pd.DataFrame({"origin_last_input_utc": origins, "regime": "low_low_low", "period": periods})
    h = np.arange(1, HORIZON + 1)
    truth = pd.DataFrame({"origin_last_input_utc": np.repeat(origins, HORIZON), "horizon_hours": np.tile(h, len(origins)),
                          "target_kms": np.tile(np.where(h % 2, 500.0, 600.0), len(origins)), "was_missing": 0})
    return folds, truth


def test_period_and_high_speed_mse_match_hand_computation():
    folds, truth = frames(["F1", "F2"])
    err = np.where(truth["origin_last_input_utc"] == folds["origin_last_input_utc"][0], 10.0, 20.0)  # F1 off by 10, F2 by 20
    err = np.where(truth["target_kms"] >= 550, err * 3, err)  # >=550 targets miss by 3x as much
    pred = truth[KEY].assign(pred_kms=truth["target_kms"] + err)
    r = score(pred, truth, folds)
    assert r["mse_f1"] == pytest.approx((100 + 900) / 2)
    assert r["mse_f2"] == pytest.approx((400 + 3600) / 2)
    assert r["mse"] == pytest.approx((500 + 2000) / 2)
    assert r["mse_f1_ge550"] == pytest.approx(900)
    assert r["mse_f2_ge550"] == pytest.approx(3600)
    assert r["mse_ge550"] == pytest.approx((900 + 3600) / 2)
    assert (r["n_pairs_f1"], r["n_pairs_f1_ge550"], r["n_pairs_ge550"]) == (72, 36, 72)
    assert r["by_regime"]["low_low_low"]["n_folds"] == 1  # regime rows count F2 + F3 folds only
    assert r["mse_f3"] is None and r["n_pairs_f3"] == 0
    assert (r["coverage_f1"], r["coverage_f2"], r["coverage_f3"]) == (1.0, 1.0, None)  # no F3 folds in this fixture


def test_three_periods_and_regime_rows_use_f2_plus_f3():
    folds, truth = frames(["F1", "F2", "F3"])
    shift = {folds["origin_last_input_utc"][0]: 10.0, folds["origin_last_input_utc"][1]: 20.0,
             folds["origin_last_input_utc"][2]: 30.0}
    pred = truth[KEY].assign(pred_kms=truth["target_kms"] + truth["origin_last_input_utc"].map(shift))
    r = score(pred, truth, folds)
    assert (r["mse_f1"], r["mse_f2"], r["mse_f3"]) == pytest.approx((100, 400, 900))
    assert r["mse"] == pytest.approx((100 + 400 + 900) / 3)
    assert (r["n_pairs_f1"], r["n_pairs_f2"], r["n_pairs_f3"]) == (72, 72, 72)
    assert r["by_regime"]["low_low_low"]["n_folds"] == 2  # F2 + F3, not F1
    assert r["coverage_f1"] == r["coverage_f2"] == r["coverage_f3"] == 1.0


def test_partial_coverage_is_reported_and_skill_uses_the_covered_folds():
    folds, truth = frames(["F1", "F2", "F3"])
    first = folds["origin_last_input_utc"][0]
    pred = truth[truth["origin_last_input_utc"] != first][KEY].reset_index(drop=True)  # no predictions for F1
    pred = pred.assign(pred_kms=truth.loc[truth["origin_last_input_utc"] != first, "target_kms"].to_numpy() + 10.0)
    naive = truth[KEY].assign(pred_kms=truth["target_kms"] + 20.0)
    r = score(pred, truth, folds, naive)
    assert r["coverage_f1"] == 0.0 and r["coverage_f2"] == r["coverage_f3"] == 1.0
    assert r["mse_f1"] is None and r["mse"] == pytest.approx(100)
    assert r["skill_vs_naive"] == pytest.approx(1 - 100 / 400)  # naive scored only on the folds the model covers


def test_folds_without_period_are_scored_as_f2():
    folds, truth = frames(["F2", "F2"])
    pred = truth[KEY].assign(pred_kms=truth["target_kms"] + 10.0)
    r = score(pred, truth, folds.drop(columns="period"))
    assert r["mse"] == r["mse_f2"] == pytest.approx(100)
    assert r["mse_f1"] is None and r["mse_f1_ge550"] is None and r["n_pairs_f1"] == 0
