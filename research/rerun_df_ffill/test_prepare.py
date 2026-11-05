"""Causality and preservation checks for the rerun's feature-only ffill snapshot."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from prepare import METADATA, prepare


def source_frame():
    # Deliberately unordered, with train/query duplicates and different h values.
    origins = pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02", "2026-01-01",
                              "2026-01-02", "2026-01-03", "2026-01-02"], utc=True)
    horizons = np.array([1, 2, 1, 1, 2, 2, 1], dtype=np.int16)
    return pd.DataFrame({
        "set": pd.Categorical(["query", "train", "train", "train", "query", "query", "query"]),
        "origin": origins, "h": horizons,
        "target_time": origins + pd.to_timedelta(horizons, unit="h"),
        "period": pd.Categorical(["F1", None, None, None, "F1", None, "F1"]),
        "fit_F1": [False, True, True, True, False, False, False],
        "fit_F2": [False, True, True, True, False, False, False],
        "fit_F3": [False, True, True, True, False, False, False],
        "y": [np.nan, 30.0, np.nan, 10.0, np.nan, 60.0, np.nan],
        "x": np.array([np.nan, np.nan, np.nan, 10, 200, np.nan, np.nan], dtype=np.float32),
        "empty": np.full(7, np.nan, dtype=np.float32),
        "other": np.array([300, 2, np.nan, 1, 20, np.nan, np.nan], dtype=np.float32),
    })


def run(tmp_path, frame, name="run", **kwargs):
    directory = tmp_path / name
    directory.mkdir()
    source = directory / "source.parquet"
    frame.to_parquet(source, index=False, row_group_size=2)
    (directory / "MANIFEST.json").write_text(json.dumps({"built_utc": "2026-11-02T12:09:07Z"}))
    out, report = directory / "filled.parquet", directory / "report.json"
    result = prepare(source, out, report, column_chunk=2, row_batch=2, **kwargs)
    return pd.read_parquet(out), result, source, out, report


def test_ffill_respects_horizon_and_preserves_metadata(tmp_path):
    frame = source_frame()
    filled, report, source, out, _ = run(tmp_path, frame)
    np.testing.assert_allclose(filled.x, [10, np.nan, 10, 10, 200, 200, 10], equal_nan=True)
    assert filled["empty"].isna().all()
    pd.testing.assert_frame_equal(filled[list(METADATA)], frame[list(METADATA)])
    pd.testing.assert_frame_equal(pd.read_parquet(source), frame)
    assert pq.read_schema(source).equals(pq.read_schema(out), check_metadata=True)
    assert report["duplicate_rows_validated"] == 1
    assert report["missing_by_feature"]["x"] == {
        "missing_before": 5, "missing_after": 1, "filled": 4, "all_missing": False}
    assert report["source"]["manifest_built_utc"] == "2026-11-02T12:09:07Z"
    assert all(report["validation"].values())


def test_future_perturbation_does_not_change_past(tmp_path):
    frame = source_frame()
    baseline, *_ = run(tmp_path, frame, "base")
    changed = frame.copy()
    future = changed.origin > pd.Timestamp("2026-01-02", tz="UTC")
    changed.loc[future, "x"] = 999999
    poisoned, *_ = run(tmp_path, changed, "poisoned")
    pd.testing.assert_frame_equal(baseline.loc[~future], poisoned.loc[~future])
    assert not baseline.loc[future, "x"].equals(poisoned.loc[future, "x"])


def test_conflicting_train_query_input_is_rejected(tmp_path):
    frame = source_frame()
    frame.loc[6, "x"] = 12
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        run(tmp_path, frame)
    assert not (tmp_path / "run/filled.parquet").exists()
    assert not (tmp_path / "run/report.json").exists()


def test_selected_features_do_not_change_others(tmp_path):
    frame = source_frame()
    filled, report, *_ = run(tmp_path, frame, features=["x"])
    pd.testing.assert_series_equal(filled.other, frame.other)
    assert report["filled_feature_names"] == ["x"]
    assert report["feature_names"] == ["x", "empty", "other"]


def test_existing_output_and_protected_columns_are_rejected(tmp_path):
    _, _, source, out, report = run(tmp_path, source_frame())
    with pytest.raises(FileExistsError):
        prepare(source, out, report)
    with pytest.raises(ValueError, match="protected"):
        prepare(source, tmp_path / "bad.parquet", tmp_path / "bad.json", features=["y"])


def test_cli_smoke(tmp_path):
    source = tmp_path / "source.parquet"
    source_frame().to_parquet(source, index=False)
    out, report = tmp_path / "out.parquet", tmp_path / "report.json"
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("prepare.py")),
                             "--source", str(source), "--out", str(out), "--report", str(report),
                             "--column-chunk", "1", "--row-batch", "3"],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout.splitlines()[-1])["status"] == "complete"
    assert pq.read_table(out).num_rows == 7
