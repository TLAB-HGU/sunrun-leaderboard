"""Causal-input, original-recipe and missing-target checks for strategy reruns."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

PATH = Path(__file__).with_name("strategies.py")
SPEC = importlib.util.spec_from_file_location("rerun_strategies", PATH)
S = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = S
SPEC.loader.exec_module(S)


def toy(n=100, seed=7):
    rng = np.random.default_rng(seed)
    inp = S.Inputs(rng.normal(size=(n, 3)).astype(np.float32),
                   rng.normal(size=(n, 72, 2)).astype(np.float32),
                   rng.normal(400, 30, size=(n, 24)).astype(np.float32))
    y = rng.normal(400, 30, size=(n, 72)).astype(np.float32)
    return inp, y


def test_feature_list_matches_frozen_original_and_mapping():
    source = Path(__file__).resolve().parents[2] / "analysis/A0014/outputs/cache/names.json"
    names = json.loads(source.read_text())
    assert S.CONSTANT == names["constant"]
    assert S.VARYING == names["varying"]
    assert len(S.CONSTANT) + len(S.VARYING) + 1 == 111
    assert S.mapped("ch_ch_t055_meridian30_equatorial_area") == "suvi_t055_meridian30_equatorial_area"


def test_stale_uses_last_observation_time_and_no_future():
    t = pd.date_range("2024-01-01", periods=12, freq="h", tz="UTC")
    ages = np.full(12, np.nan)
    ages[1] = 0.5
    ages[10] = 0.25
    raw = pd.DataFrame({"origin": t, "img_age_h": ages})
    stale = S.stale_series(raw)
    assert stale.iloc[0] == 1
    assert (stale.iloc[1:7] == 0).all()
    assert (stale.iloc[7:10] == 1).all()
    assert stale.iloc[10] == 0
    poisoned = raw.copy()
    poisoned.loc[poisoned.origin > t[8], "img_age_h"] = 0.0
    pd.testing.assert_series_equal(stale.iloc[:9], S.stale_series(poisoned).iloc[:9])
    pd.testing.assert_series_equal(stale, S.stale_series(pd.concat([raw, raw.iloc[2:5]], ignore_index=True)))
    bad = pd.concat([raw, pd.DataFrame({"origin": [t[1]], "img_age_h": [0.2]})], ignore_index=True)
    with pytest.raises(ValueError, match="disagree"):
        S.stale_series(bad)


def test_past_window_preserves_new_origin_minus_one_observation():
    t = pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC")
    snapshots = pd.Series(np.arange(50, dtype=float), index=t)
    history = S.past_window(snapshots, t[[30]])
    np.testing.assert_equal(history, np.arange(7, 31, dtype=np.float32)[None, :])
    poisoned = snapshots.copy()
    poisoned.loc[t[31]:] = 10000
    np.testing.assert_equal(history, S.past_window(poisoned, t[[30]]))


def test_teacher_fill_does_not_change_targets_or_use_later_labels():
    inp, y = toy(2)
    y[:, :3] = np.nan
    y[:, 4] = np.nan
    before = y.copy()
    trajectory = S.training_trajectory(inp, y)
    np.testing.assert_equal(y, before)
    np.testing.assert_equal(trajectory[:, 24:27], np.repeat(inp.past[:, -1:], 3, axis=1))
    np.testing.assert_equal(trajectory[:, 28], y[:, 3])
    other = y.copy()
    other[:, 10:] = 9999
    np.testing.assert_equal(trajectory[:, :34], S.training_trajectory(inp, other)[:, :34])


def test_original_design_order_and_width():
    inp, y = toy(4)
    trajectory = S.training_trajectory(inp, y)
    for kind in ("RecMO", "DirMO", "DirRecMO"):
        actual = S.design(kind, inp, trajectory, 13, 18)
        if kind == "RecMO":
            expected = np.hstack([inp.ctx, trajectory[:, 12:36], inp.E[:, 12:18].reshape(4, -1), np.full((4, 1), 13)])
        elif kind == "DirMO":
            expected = np.hstack([inp.ctx, inp.past, inp.E[:, 12:18].reshape(4, -1)])
        else:
            expected = np.hstack([inp.ctx, inp.past, y[:, :12], inp.E[:, 12:18].reshape(4, -1)])
        np.testing.assert_equal(actual, expected.astype(np.float32))


def test_masked_gradient_ignores_missing_targets():
    observed = np.array([[True, False], [False, True]])
    matrix = xgb.DMatrix(np.ones((2, 2)), label=np.array([[1., 0.], [0., 4.]]))
    gradient, hessian = S.masked_l2(observed)(np.array([[3., 8000.], [9000., 3.]]), matrix)
    np.testing.assert_equal(gradient, [[2., 0.], [0., -1.]])
    np.testing.assert_equal(hessian, observed.astype(np.float32))


def test_finite_targets_match_original_xgboost_recipe():
    inp, y = toy(120)
    X = S.design("DirMO", inp, S.training_trajectory(inp, y), 1, 6)
    labels = y[:, :6]
    original = xgb.XGBRegressor(n_estimators=3, max_depth=3, learning_rate=0.03,
                                subsample=0.8, colsample_bytree=0.7, min_child_weight=20,
                                reg_lambda=5, random_state=42, objective="reg:squarederror",
                                tree_method="hist", device="cpu", eval_metric="rmse",
                                multi_strategy="multi_output_tree", n_jobs=1)
    original.fit(X, labels)
    ported = S.fit_booster(X, labels, 3, 1, None, "finite", [])
    np.testing.assert_allclose(ported.inplace_predict(X), original.predict(X), atol=1e-4, rtol=0)


@pytest.mark.parametrize("experiment", sorted(S.STRATEGIES))
def test_all_strategies_fit_missing_targets_and_predict_without_query_truth(experiment):
    inp, y = toy(100)
    y[::7, ::3] = np.nan
    model = S.build(experiment, rounds=1, threads=1)
    origins = pd.date_range("2024-01-01", periods=100, freq="6h", tz="UTC")
    if isinstance(model, S.Rectify):
        model.fit(inp, y, origins)
    else:
        model.fit(inp, y)
    prediction = model.predict(inp.take(slice(0, 5)))
    assert prediction.shape == (5, 72)
    assert np.isfinite(prediction).all()
    y[:] = 99999
    np.testing.assert_array_equal(prediction, model.predict(inp.take(slice(0, 5))))


def test_purged_oof_original_gap():
    origins = pd.date_range("2024-01-01", periods=500, freq="h", tz="UTC")
    seen = []
    for training, held in S.purged_folds(origins):
        assert not np.intersect1d(training, held).size
        assert ((origins[training] < origins[held[0]] - pd.Timedelta(hours=72)) |
                (origins[training] > origins[held[-1]] + pd.Timedelta(hours=72))).all()
        seen.extend(held)
    assert sorted(seen) == list(range(len(origins)))


def test_models_and_observed_loss_are_persisted(tmp_path):
    inp, y = toy(40)
    y[0, :3] = np.nan
    model = S.Block("DirMO", 24, rounds=2, output=tmp_path,
                    context_names=["a", "b", "c"], varying_names=["u", "v"]).fit(inp, y)
    assert len(list(tmp_path.glob("*.ubj"))) == 3
    assert len(list(tmp_path.glob("*_loss.csv"))) == 3
    metadata = json.loads((tmp_path / "fit_b00.json").read_text())
    assert metadata["observed_targets"] == 40 * 24 - 3
    assert metadata["feature_count"] == len(metadata["feature_names"])
    loaded = xgb.Booster()
    loaded.load_model(tmp_path / "fit_b00.ubj")
    X = S.design("DirMO", inp, S.training_trajectory(inp, y), 1, 24)
    np.testing.assert_array_equal(loaded.inplace_predict(X), model.models[0].inplace_predict(X))


def test_grid_loader_retains_missing_targets_and_rejects_future_fit(tmp_path):
    origins = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    columns = {"origin": np.repeat(origins, 72), "h": np.tile(np.arange(1, 73), 2),
               "fit_F1": True, "y": np.arange(144, dtype=np.float32) + 400}
    for i, name in enumerate([S.mapped(c) for c in S.CONSTANT if c != "img_stale"] + S.VARYING):
        columns[name] = float(i)
    frame = pd.DataFrame(columns)
    frame.loc[5, "y"] = np.nan
    path = tmp_path / "df.parquet"
    frame.to_parquet(path, index=False)
    speeds = pd.Series([400., 410.], index=origins)
    stale = pd.Series([1., 0.], index=origins)
    observed_origins, inp, y = S.load_grid(path, "F1", True, speeds, stale)
    assert observed_origins.equals(origins)
    assert inp.ctx.shape == (2, 90) and inp.E.shape == (2, 72, 20)
    assert np.isnan(y[0, 5]) and y[0, 4] == 404
    np.testing.assert_array_equal(inp.ctx[:, S.CONSTANT.index("img_stale")], [1, 0])
    frame["origin"] = frame.origin + pd.Timedelta(days=363)
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="maturity"):
        S.load_grid(path, "F1", True, speeds, stale)
