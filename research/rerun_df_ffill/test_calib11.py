"""Checks numerical preservation, missing-label exclusion and the end-to-end training path."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from calib11 import (
    BLOCK,
    PARAMS,
    PERIODS,
    RECIPES,
    VARYING,
    H,
    constants,
    feature_names,
    fit,
    masked_objective,
    merge,
    mse_timse,
    normalized_weights,
    sera_weights,
    table,
    timse_weights,
    train_period,
)

ROOT = Path(__file__).resolve().parents[2]


def original(experiment):
    archived = Path(__file__).resolve().parent / "source_archive" / f"{experiment}.py"
    source = archived if archived.exists() else ROOT / "research" / experiment / "experiment.py"
    spec = importlib.util.spec_from_file_location(experiment, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("experiment", RECIPES)
def test_frozen_features_rounds_and_booster_parameters_match_original(experiment):
    old = original(experiment)
    assert constants(experiment) == [x.replace("ch_ch_", "suvi_") for x in old.CONSTANT]
    assert VARYING == old.VARYING
    expected_rounds = getattr(old, "ROUNDS_BY_PERIOD", {p: old.ROUNDS for p in PERIODS})
    assert RECIPES[experiment][3] == tuple(expected_rounds.values())
    for key, value in PARAMS.items():
        assert old.PARAMS[key] == value
    assert len(feature_names(experiment, 0)) == (235 if experiment.endswith("noenlil") else 237)


def test_finite_data_metric_weights_and_metrics_match_original():
    old = original("e2000fs_calib11_metricgn")
    rng = np.random.default_rng(18)
    y = rng.uniform(250, 800, (121, 4))
    pred = y + rng.normal(0, 50, y.shape)
    np.testing.assert_allclose(timse_weights(y), old.timse_weights(y), rtol=1e-13)
    actual, expected = mse_timse(pred, y), old.mse_timse(pred, y)
    np.testing.assert_allclose(actual, expected, rtol=1e-13)


def test_missing_targets_excluded_from_quantiles_metrics_and_normalization():
    y = np.array([[300., np.nan], [np.nan, 400.], [500., 500.], [700., 800.]])
    pred = np.full_like(y, 450.)
    actual_mse, actual_timse = mse_timse(pred, y)
    w = timse_weights(y)
    assert (w[~np.isfinite(y)] == 0).all()
    for lead in range(2):
        mask = np.isfinite(y[:, lead])
        expected = mse_timse(pred[mask, lead, None], y[mask, lead, None])
        np.testing.assert_allclose([actual_mse[lead], actual_timse[lead]], np.asarray(expected).ravel())
    sera = sera_weights(y)
    for normalization in ("mean", "median"):
        normalized = normalized_weights(sera, np.isfinite(y), normalization)
        assert (normalized[~np.isfinite(y)] == 0).all()
        reduce = np.mean if normalization == "mean" else np.median
        for lead in range(2):
            np.testing.assert_allclose(reduce(normalized[np.isfinite(y[:, lead]), lead]), 1.)


def test_missing_targets_have_exactly_zero_gradient_and_hessian():
    weights = np.array([[1., 0.], [2., 3.]], dtype=np.float32)
    class Matrix:
        def get_label(self):
            return np.array([300., 0., 400., 500.])
    grad, hess = masked_objective(weights)(np.full((2, 2), 450.), Matrix())
    assert grad[0, 1] == hess[0, 1] == 0
    np.testing.assert_equal(grad, [[150., 0.], [100., -150.]])


def test_fit_ignores_all_missing_origin_and_persists_real_model(tmp_path):
    rng = np.random.default_rng(9)
    x = rng.normal(size=(80, 5)).astype(np.float32)
    y = 400 + rng.normal(size=(80, BLOCK)) * 20
    y[0] = np.nan
    y[1, 3] = np.nan
    path = tmp_path / "model.ubj"
    model, report = fit(x, y, np.ones_like(y), "mean", 2, 1, path)
    assert path.is_file() and path.with_suffix(".loss.json").is_file()
    assert report["dropped_no_target"] == 1
    assert report["train_origins"] == 79
    assert np.isfinite(model.inplace_predict(x)).all()
    # An entirely unobserved origin must have no effect on the fitted tree.
    other, _ = fit(x[1:], y[1:], np.ones_like(y[1:]), "mean", 2, 1, tmp_path / "other.ubj")
    np.testing.assert_array_equal(model.inplace_predict(x), other.inplace_predict(x))


def synthetic_table(experiment):
    rng = np.random.default_rng(4)
    train_origins = pd.date_range("2025-01-01", periods=160, freq="h", tz="UTC")
    query_origins = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    origins = train_origins.append(query_origins)
    n = len(origins)
    data = {"set": np.repeat(["train"] * len(train_origins) + ["query"] * len(query_origins), H),
            "origin": np.repeat(origins, H), "h": np.tile(np.arange(1, H + 1), n),
            "period": np.repeat([None] * len(train_origins) + ["F1"] * len(query_origins), H)}
    for period in PERIODS:
        data[f"fit_{period}"] = np.repeat(np.arange(n) < len(train_origins), H)
    for col in constants(experiment):
        data[col] = np.repeat(rng.uniform(1, 20, n), H)
    for col in VARYING:
        data[col] = rng.uniform(1, 20, n * H)
    data["y"] = rng.uniform(300, 700, n * H)
    frame = pd.DataFrame(data)
    frame.loc[frame.index % 7 == 0, "y"] = np.nan
    frame.loc[frame["set"].eq("query"), "y"] = np.nan
    return frame


@pytest.mark.parametrize("experiment", ["e2000fs_calib11b_metricwt", "e2000fs_calib11g_serarounds2"])
def test_period_smoke_missing_targets_and_query_target_poisoning(tmp_path, monkeypatch, experiment):
    monkeypatch.setitem(PERIODS, "F1", 4)
    frame = synthetic_table(experiment)
    source = tmp_path / "df.parquet"
    frame.to_parquet(source, index=False)
    out = tmp_path / "first"
    out.mkdir()
    train_period(source, experiment, "F1", [0], 2, 1, out)
    with np.load(out / "F1_b0.npz") as z:
        expected = z["pred"].copy()
    assert expected.shape == (4, 24)
    assert np.isfinite(expected).all()
    frame.loc[frame["set"].eq("query"), "y"] = 1e12
    frame.to_parquet(source, index=False)
    second = tmp_path / "second"
    second.mkdir()
    train_period(source, experiment, "F1", [0], 2, 1, second)
    with np.load(second / "F1_b0.npz") as z:
        np.testing.assert_array_equal(z["pred"], expected)


def test_table_rejects_duplicate_origin_grid():
    experiment = "e2000fs_calib11g_serarounds2"
    frame = synthetic_table(experiment).iloc[:H].copy()
    duplicate = pd.concat([frame, frame], ignore_index=True)
    with pytest.raises(ValueError, match="grid"):
        table(duplicate, experiment)


def test_merge_rejects_smoke_artifacts(tmp_path, monkeypatch):
    monkeypatch.setitem(PERIODS, "F1", 1)
    experiment = "e2000fs_calib11g_serarounds2"
    for block in range(3):
        np.savez(tmp_path / f"F1_b{block}.npz", pred=np.full((1, BLOCK), 400.),
                 origins=pd.date_range("2026-01-01", periods=1, tz="UTC").asi8)
        report = {"experiment": experiment, "period": "F1", "block": block, "rounds": 2,
                  "features": feature_names(experiment, block)}
        (tmp_path / f"F1_b{block}.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="smoke"):
        merge(tmp_path)
