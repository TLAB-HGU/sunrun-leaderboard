import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from scoring import HORIZON, build_truth, n_blocks, score, validate  # noqa: E402


@pytest.fixture
def setup():
    ts = pd.date_range("2026-08-25", periods=24 * 12, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    series = pd.DataFrame({"timestamp_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "filled_speed_kms": 400 + rng.normal(0, 30, len(ts)),
                           "was_missing": (np.arange(len(ts)) % 5 == 0).astype(int)})
    origins = pd.date_range("2026-08-27", periods=30, freq="h", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    regimes = ["low_low_low"] * 5 + ["high_low_high"] * 25
    folds = pd.DataFrame({"origin_last_input_utc": origins, "regime": regimes})
    return series, folds, build_truth(series, folds)


def test_perfect_prediction_is_zero(setup):
    _, folds, truth = setup
    pred = truth.rename(columns={"target_kms": "pred_kms"})
    assert validate(pred, folds) == []
    r = score(pred, truth, folds)
    assert r["mse"] == 0 and r["n_pairs"] == 30 * HORIZON


def test_constant_error(setup):
    _, folds, truth = setup
    pred = truth.assign(pred_kms=truth["target_kms"] + 10).drop(columns="target_kms")
    r = score(pred, truth, folds)
    assert r["mse"] == pytest.approx(100) and r["by_regime"]["low_low_low"]["mse"] == pytest.approx(100)
    assert r["by_regime"]["high_high_high"]["n_folds"] == 0 and not r["by_regime"]["high_high_high"]["reliable"]
    # 25 consecutive hourly origins are one 72h block -> not enough independent blocks to report
    assert r["by_regime"]["high_low_high"]["n_blocks"] == 1 and not r["by_regime"]["high_low_high"]["reliable"]
    assert r["regime_balanced_mse"] is None


def test_rejects_missing_rows_nan_and_duplicates(setup):
    _, folds, truth = setup
    good = truth.rename(columns={"target_kms": "pred_kms"})
    assert any("missing" in e for e in validate(good.iloc[1:], folds))
    bad = good.copy(); bad.loc[0, "pred_kms"] = np.nan
    assert any("NaN" in e for e in validate(bad, folds))
    assert any("duplicate" in e for e in validate(pd.concat([good, good.iloc[:1]]), folds))
    assert any("missing columns" in e for e in validate(good.drop(columns="pred_kms"), folds))


def test_skill_vs_naive(setup):
    series, folds, truth = setup
    last = series.set_index("timestamp_utc")["filled_speed_kms"]
    naive = truth[["origin_last_input_utc", "horizon_hours"]].copy()
    naive["pred_kms"] = naive["origin_last_input_utc"].map(last).to_numpy()
    assert score(naive, truth, folds, naive)["skill_vs_naive"] == pytest.approx(0)


def test_n_blocks():
    o = pd.date_range("2026-06-01", periods=300, freq="h", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    assert n_blocks(o) == 5  # 300h / 72h -> blocks start at 0,72,144,216,288
    assert n_blocks([]) == 0


def test_mse_observed_masks_filled_targets(setup):
    _, folds, truth = setup
    pred = truth.assign(pred_kms=truth["target_kms"] + np.where(truth["was_missing"] == 1, 50.0, 0.0)).drop(columns="target_kms")
    r = score(pred, truth, folds)
    assert r["mse_observed"] == pytest.approx(0) and r["mse"] > 0


def test_paired_ci_is_deterministic_and_detects_better_model(setup):
    series, folds, truth = setup
    last = series.set_index("timestamp_utc")["filled_speed_kms"]
    naive = truth[["origin_last_input_utc", "horizon_hours"]].copy()
    naive["pred_kms"] = naive["origin_last_input_utc"].map(last).to_numpy()
    perfect = truth.rename(columns={"target_kms": "pred_kms"})
    a, b = score(perfect, truth, folds, naive)["vs_naive"], score(perfect, truth, folds, naive)["vs_naive"]
    assert a == b and a["beats_reference"] and a["ci95"][1] < 0
