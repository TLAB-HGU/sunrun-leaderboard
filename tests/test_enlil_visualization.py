import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/visualize_enlil_validation.py"
SPEC = importlib.util.spec_from_file_location("enlil_visualization", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def replay(tmp_path):
    origin = pd.Timestamp("2026-09-30T23:00Z")
    pred = pd.DataFrame({MODULE.ORIGIN: [origin] * 72,
                         "horizon_hours": range(1, 73), "pred_kms": np.arange(72) + 400.})
    history = pd.DataFrame({"timestamp_utc": pd.date_range(origin - pd.Timedelta(hours=167), periods=240, freq="h"),
                            "filled_speed_kms": np.arange(240) + 300., "was_missing": 0})
    args = SimpleNamespace(reference_month="2026-10", experiment="E0472",
                           predictions=tmp_path / "pred.parquet", history=tmp_path / "history.parquet",
                           folds=tmp_path / "folds.parquet", truth=tmp_path / "truth.parquet",
                           output_dir=tmp_path / "output", model_file=None,
                           width=1000, height=500, frame_ms=100, limit=None)
    pred.to_parquet(args.predictions)
    history.to_parquet(args.history)
    return args, pred, history


def test_reference_alignment_and_render_preserve_official_manifest(replay):
    args, _, history = replay
    args.model_file = args.predictions.parent / "model.pkl"
    args.model_file.write_bytes(b"frozen model")
    args.provenance = args.predictions.parent / "provenance.json"
    receipt = {"model_sha256": MODULE.sha256(args.model_file),
               "predictions_sha256": MODULE.sha256(args.predictions)}
    args.provenance.write_text(json.dumps(receipt))
    data = MODULE.load_reference_data(args)
    np.testing.assert_array_equal(data[2][0], history.filled_speed_kms[:168])
    np.testing.assert_array_equal(data[4][0], history.filled_speed_kms[168:])
    assert data[-1] == (200., 800.)
    args.output_dir.mkdir()
    official = args.output_dir / "manifest.json"
    official.write_text('existing official manifest\n')
    MODULE.render(args, data)
    assert official.read_text() == 'existing official manifest\n'
    manifest = json.loads((args.output_dir / "manifest_2026-10.json").read_text())
    assert manifest["verification"]["all_gifs_decoded"]
    assert manifest["outputs"][0]["frames"] == 1
    assert "expected_mse" not in manifest["verification"]
    assert set(manifest["sources"]) == {"predictions", "history", "model_file", "provenance"}
    assert manifest["provenance"] == receipt
    assert manifest["sources"]["provenance"]["sha256"] == MODULE.sha256(args.provenance)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        MODULE.render(args, data)


@pytest.mark.parametrize("change", ["missing", "duplicate", "invalid", "nonfinite"])
def test_reference_rejects_bad_forecasts(replay, change):
    args, pred, _ = replay
    if change == "missing":
        pred = pred.iloc[:-1]
    elif change == "duplicate":
        pred = pd.concat([pred, pred.iloc[:1]])
    elif change == "invalid":
        pred.loc[0, "horizon_hours"] = 0
    else:
        pred.loc[0, "pred_kms"] = np.nan
    pred.to_parquet(args.predictions)
    with pytest.raises(ValueError, match="exactly one row|Non-finite"):
        MODULE.load_reference_data(args)


@pytest.mark.parametrize("row", [0, 239])
def test_reference_rejects_missing_input_or_actual(replay, row):
    args, _, history = replay
    history.drop(index=row).to_parquet(args.history)
    with pytest.raises(ValueError, match="missing required input or future actual"):
        MODULE.load_reference_data(args)


def test_reference_month_and_expanded_scale(replay):
    args, pred, _ = replay
    args.reference_month = "2026-09"
    with pytest.raises(ValueError, match="outside requested month"):
        MODULE.load_reference_data(args)
    args.reference_month = "2026-10"
    pred.loc[0, "pred_kms"] = 900.
    pred.to_parquet(args.predictions)
    assert MODULE.load_reference_data(args)[-1] == (200., 950.)


def test_official_fold_guard_is_still_required(replay):
    args, pred, _ = replay
    pred[[MODULE.ORIGIN]].iloc[:1].to_parquet(args.folds)
    pred.rename(columns={"pred_kms": "target_kms"}).assign(was_missing=0).to_parquet(args.truth)
    with pytest.raises(ValueError, match="2,833 unique folds"):
        MODULE.load_data(args)


@pytest.mark.parametrize("key", ["model_sha256", "predictions_sha256"])
def test_reference_provenance_hashes_are_checked(replay, key):
    args, _, _ = replay
    args.model_file = args.predictions.parent / "model.pkl"
    args.model_file.write_bytes(b"frozen model")
    args.provenance = args.predictions.parent / "provenance.json"
    receipt = {"model_sha256": MODULE.sha256(args.model_file),
               "predictions_sha256": MODULE.sha256(args.predictions)}
    receipt[key] = "wrong"
    args.provenance.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Provenance .* hash mismatch"):
        MODULE.render(args, MODULE.load_reference_data(args))
    assert not args.output_dir.exists()
