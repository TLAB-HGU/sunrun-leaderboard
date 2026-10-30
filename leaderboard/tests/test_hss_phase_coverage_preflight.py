"""Synthetic-only preflight tests; never loads scientific data."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/hss_phase_coverage_preflight.py"
SPEC = importlib.util.spec_from_file_location("hss_phase_coverage_preflight", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

HOLE = "2024-06-17T18:00:00+00:00"


def synthetic_frames():
    idx = pd.date_range("2024-06-15T00:00:00Z", periods=144, freq="h", tz="UTC")
    speed = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    base = pd.DataFrame({"f": np.arange(len(idx), dtype=float)}, index=idx)
    base = base.drop(pd.Timestamp(HOLE))
    base.loc[idx[10], "f"] = np.nan
    base.loc[idx[11], "f"] = np.inf
    return {"speed": speed, "base": base}, idx


def synthetic_folds(idx):
    return [{"id": "synth", "start_utc": idx[48].isoformat(), "end_utc": idx[-1].isoformat()}]


def origins_fn(fold):
    return pd.date_range(fold["start_utc"], pd.Timestamp(fold["end_utc"]) - pd.Timedelta(hours=73), freq="6h")


def test_frame_coverage_reports_nan_inf_and_hourly_gap():
    frames, _ = synthetic_frames()
    cov = MOD.frame_coverage(frames)
    assert cov["base"]["n_nan_rows"] == 1
    assert cov["base"]["n_inf_rows"] == 1
    assert cov["base"]["n_missing_hourly"] == 1
    assert HOLE in cov["base"]["first_missing_hourly"]
    assert cov["speed"]["n_missing_hourly"] == 0


def test_deployment_coverage_keeps_every_missing_with_prev_base_lag():
    frames, idx = synthetic_frames()
    folds = synthetic_folds(idx)
    deploy = MOD.deployment_coverage(frames, folds, origins_fn)
    assert deploy["n_issuances"] == sum(v["n_origins"] for v in deploy["per_fold"].values())
    assert deploy["n_missing"] == len(deploy["missing"]) >= 1
    hole_rows = [m for m in deploy["missing"] if m["issuance_utc"] == pd.Timestamp(HOLE).isoformat()]
    assert len(hole_rows) == 1
    row = hole_rows[0]
    assert row["in_base"] is False and row["prev_base_utc"] == pd.Timestamp("2024-06-17T17:00:00+00:00").isoformat()
    assert row["lag_hours"] == pytest.approx(1.0)


def test_hole_analysis_explicit_fields():
    frames, idx = synthetic_frames()
    hole = MOD.analyze_hole(frames, synthetic_folds(idx), origins_fn)
    assert hole["issuance_utc"] == pd.Timestamp(HOLE).isoformat()
    assert hole["in_base"] is False and hole["in_speed"] is True
    assert hole["lag_hours"] == pytest.approx(1.0)
    assert hole["n_window_missing_base_48h"] >= 1


def test_training_oof_audit_never_fits_and_captures_errors():
    frames, idx = synthetic_frames()
    folds = synthetic_folds(idx)
    calls = []

    def fake_pool(bounded, cutoff, config=None):
        calls.append("pool")
        return pd.DatetimeIndex([pd.Timestamp("2024-06-14T00:00:00Z")])

    def fake_cal(pool, cutoff):
        return None, np.zeros(len(pool), dtype=int)

    out = MOD.training_oof_audit(frames, folds, fake_pool, fake_cal, None)
    assert out["synth"]["ok"] is True and calls == ["pool"]

    def boom(bounded, cutoff, config=None):
        raise ValueError("no strictly OOF full96h correction origins")

    out = MOD.training_oof_audit(frames, folds, boom, None, None)
    assert out["synth"]["ok"] is False and "no strictly OOF" in out["synth"]["error"]


def test_output_inside_run_dir_refused_and_synthetic_cli_writes_json(tmp_path):
    run_dir = tmp_path / "frozen"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="outside"):
        MOD._refuse_inside_run_dir(run_dir, run_dir / "out.json")
    out = tmp_path / "fresh" / "preflight.json"
    MOD.run_preflight(run_dir, out, synthetic=True)
    report = json.loads(out.read_text())
    assert report["no_model_fit"] is True
    assert report["hole_2024_06_17T18Z"]["issuance_utc"] == pd.Timestamp(HOLE).isoformat()
    assert report["n_issuances"] == report["n_missing"] + sum(
        v["n_origins"] - v["n_missing_base"] for v in report["fold_counts"].values()
    )
