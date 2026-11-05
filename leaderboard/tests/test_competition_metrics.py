import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "leaderboard" / "scorer"))
from competition import (LEADS, VERSION, align_predictions, load_observations,
                         prepare_context, score_competition)


@pytest.fixture
def observations():
    times = pd.date_range("2025-12-30", "2026-01-22", freq="h", tz="UTC")
    steps = np.arange(len(times))
    values = 400 + 70 * np.sin(steps / 19) + steps / 9
    values[steps % 17 == 3] = np.nan
    return pd.DataFrame({"timestamp_utc": times, "speed_kms": values})


@pytest.fixture
def context(observations):
    return prepare_context(observations, {
        "F1": ("2026-01-01", "2026-01-07"),
        "F2": ("2026-01-08", "2026-01-14"),
        "F3": ("2026-01-15", "2026-01-21"),
    })


def predictions(context, values=None):
    if values is None:
        values = np.nan_to_num(context["truth"], nan=420) + np.arange(72)[None, :] / 3 + 2
    return pd.DataFrame({"origin_last_input_utc": np.repeat(context["origins"], 72),
                         "horizon_hours": np.tile(LEADS, len(context["origins"])),
                         "pred_kms": values.ravel()})


def notebook_scorer():
    notebook = json.loads((ROOT / "metric" / "metric.ipynb").read_text())
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "def score_predictions(" in source:
            function = next(node for node in ast.parse(source).body
                            if isinstance(node, ast.FunctionDef) and node.name == "score_predictions")
            namespace = {"np": np, "pd": pd, "LEADS": LEADS}
            exec(compile(ast.Module(body=[function], type_ignores=[]), "metric/metric.ipynb", "exec"), namespace)
            return namespace["score_predictions"]
    raise AssertionError("reference notebook scorer not found")


def test_matches_notebook_for_each_scope_and_every_lead(context):
    pred = predictions(context)
    pred.loc[::119, "pred_kms"] = np.nan
    pred.loc[::137, "pred_kms"] = -1
    pred = pred.drop(index=pred.index[::151])
    got = score_competition(pred, context)
    score_reference = notebook_scorer()
    aligned = align_predictions(pred, context)
    assert got["version"] == VERSION
    for name, scope in context["scopes"].items():
        selected = scope["selected"]
        truth = context["truth"][selected]
        reference = score_reference(aligned[selected], {
            "truth": truth,
            "persistence": context["persistence"][selected],
            "quantiles": pd.DataFrame([
                np.quantile(values[np.isfinite(values)], np.arange(101) / 100, method="linear")
                for values in truth.T], index=LEADS),
        }, "test")
        expected = reference["by_lead"]
        actual = pd.DataFrame(got[name]["by_lead"]).set_index("lead_hours")
        for key, reference_key in [("s", "score_kms"), ("timse", "timse_kms2"),
                                   ("rmse", "rmse_kms"), ("trmse", "trmse_kms"),
                                   ("bias", "bias_kms"), ("n_valid", "n_valid"),
                                   ("n_replaced", "n_replaced")]:
            np.testing.assert_allclose(actual[key], expected[reference_key], rtol=1e-13)
        assert got[name]["s"] == pytest.approx(reference["summary"]["final_score_kms"])
        assert got[name]["rmse"] == pytest.approx(reference["summary"]["mean_rmse_kms"])
        assert got[name]["bias"] == pytest.approx(reference["summary"]["bias_kms"])
        assert got[name]["n_pairs"] == reference["summary"]["n_pairs"]


def test_total_pools_within_lead_and_weights_leads_equally(context):
    error = np.where(context["period"] == "F1", 1, np.where(context["period"] == "F2", 20, 80))
    values = np.nan_to_num(context["truth"], nan=450) + error[:, None] * (1 + LEADS[None, :] / 50)
    result = score_competition(predictions(context, values), context)
    total = result["total"]
    assert total["s"] != pytest.approx(np.mean([result[p]["s"] for p in ("f1", "f2", "f3")]))
    assert len({row["n_valid"] for row in total["by_lead"]}) > 1
    pooled_rmse = np.sqrt(np.nanmean((values - context["truth"]) ** 2))
    assert total["rmse"] != pytest.approx(pooled_rmse)
    assert total["rmse"] == pytest.approx(np.mean([r["rmse"] for r in total["by_lead"]]))
    assert total["timse"] == pytest.approx(np.mean([r["timse"] for r in total["by_lead"]]))


def test_persistence_uses_last_observation_at_or_before_origin(observations, context):
    origin = context["origins"][0]
    observations.loc[observations.timestamp_utc == origin, "speed_kms"] = np.nan
    observations.loc[observations.timestamp_utc == origin + pd.Timedelta(hours=1), "speed_kms"] = 9999
    rebuilt = prepare_context(observations, {
        "F1": ("2026-01-01", "2026-01-07"), "F2": ("2026-01-08", "2026-01-14"),
        "F3": ("2026-01-15", "2026-01-21"),
    })
    before = observations.loc[observations.timestamp_utc <= origin, "speed_kms"].dropna().iloc[-1]
    assert rebuilt["persistence"][0] == before
    empty = predictions(rebuilt).iloc[:0]
    result = score_competition(empty, rebuilt)
    assert result["total"]["replacement_fraction"] == 1
    assert result["total"]["n_pairs"] == int(np.isfinite(rebuilt["truth"]).sum())


def test_official_grid_and_boundaries():
    times = pd.date_range("2025-12-30", "2026-10-01", freq="h", tz="UTC")
    context = prepare_context(pd.DataFrame({"timestamp_utc": times,
                                           "speed_kms": 400 + np.arange(len(times)) % 199}))
    assert len(context["origins"]) == 1052
    assert set(context["origins"].hour) == {0, 6, 12, 18}
    assert [int((context["period"] == p).sum()) for p in ("F1", "F2", "F3")] == [348, 352, 352]
    assert context["origins"][0] == pd.Timestamp("2026-01-01", tz="UTC")
    assert context["origins"][-1] == pd.Timestamp("2026-09-26T18:00:00Z")


def test_hourly_and_boundary_extras_are_ignored_but_still_validated(context):
    pred = predictions(context)
    extra = pred.iloc[[0]].copy()
    extra["origin_last_input_utc"] = pd.Timestamp("2026-01-06T01:00:00Z")
    assert score_competition(pd.concat([pred, extra]), context) == score_competition(pred, context)
    with pytest.raises(ValueError, match="duplicate"):
        score_competition(pd.concat([pred, extra, extra]), context)


@pytest.mark.parametrize("column,value,match", [
    ("horizon_hours", 0, "integers"), ("horizon_hours", 1.5, "integers"),
    ("origin_last_input_utc", "2026-01-01T00:30:00Z", "hourly"),
    ("origin_last_input_utc", "2025-01-01T00:00:00Z", "hourly"),
])
def test_rejects_invalid_keys(context, column, value, match):
    pred = predictions(context)
    pred[column] = pred[column].astype(object)
    pred.loc[0, column] = value
    with pytest.raises(ValueError, match=match):
        score_competition(pred, context)


def test_rejects_undefined_truth_and_missing_persistence(observations):
    periods = {"F1": ("2026-01-01", "2026-01-07"), "F2": ("2026-01-08", "2026-01-14"),
               "F3": ("2026-01-15", "2026-01-21")}
    with pytest.raises(ValueError, match="constant"):
        prepare_context(observations.assign(speed_kms=400), periods)
    missing = observations.copy()
    missing.loc[missing.timestamp_utc <= pd.Timestamp("2026-01-01", tz="UTC"), "speed_kms"] = np.nan
    with pytest.raises(ValueError, match="persistence"):
        prepare_context(missing, periods)
    missing = observations.copy()
    missing.loc[missing.timestamp_utc.dt.hour % 6 == 1, "speed_kms"] = np.nan
    with pytest.raises(ValueError, match="no valid truth"):
        prepare_context(missing, periods)


def test_duplicate_quantiles_and_overflow(context):
    times = pd.date_range("2025-12-30", "2026-10-01", freq="h", tz="UTC")
    tied = prepare_context(pd.DataFrame({"timestamp_utc": times,
                                        "speed_kms": np.where(np.arange(len(times)) % 7 == 0, 600, 400)}))
    result = score_competition(predictions(tied, tied["truth"] + 10), tied)
    assert result["total"]["timse"] == pytest.approx(100)
    assert result["total"]["s"] == pytest.approx(10)
    with pytest.raises(FloatingPointError):
        score_competition(predictions(context).assign(pred_kms=1e308), context)


def test_raw_observations_mask_status_and_invalid_values(tmp_path):
    folder = tmp_path / "year=2026"
    folder.mkdir()
    pd.DataFrame({"timestamp_utc": pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC"),
                  "ace_speed_kms": [400, 500, -9999, np.inf, np.nan],
                  "ace_swepam_status": [0, 1, 0, 0, 0]}).to_parquet(folder / "part.parquet")
    loaded = load_observations(tmp_path)
    assert loaded.speed_kms.iloc[0] == 400
    assert loaded.speed_kms.iloc[1:].isna().all()
    canonical = tmp_path / "observations.parquet"
    loaded.to_parquet(canonical)
    pd.testing.assert_frame_equal(load_observations(canonical), loaded)
