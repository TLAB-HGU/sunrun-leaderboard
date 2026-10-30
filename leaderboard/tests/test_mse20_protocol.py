import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from experiments.mse20_four_regimes import protocol as p
from experiments.mse20_four_regimes.search import pick_rows


def test_split_purge_and_coverage():
    index = pd.date_range("2025-01-01", "2026-06-01", freq="h", tz="UTC")
    origins = p.split_origins(index, "development")
    assert len(origins) == 464
    assert origins.min() > pd.Timestamp(p.SPLITS["development"]["cutoff"]) + pd.Timedelta(hours=72)
    with pytest.raises(ValueError, match="missing origins"):
        p.split_origins(index.drop(origins[0]), "development")


def test_deadline_persists(tmp_path):
    runtime = {"deadline_utc": (pd.Timestamp.now(tz="UTC") + timedelta(seconds=60)).isoformat()}
    p.write_json(tmp_path / "runtime.json", runtime)
    assert 0 < p.check_deadline(tmp_path) <= 60
    with pytest.raises(TimeoutError):
        p.check_deadline(tmp_path, reserve_seconds=61)
    assert json.loads((tmp_path / "runtime.json").read_text()) == runtime


def test_slice_drops_future_poison():
    idx = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    frames = {"speed": pd.Series([1, 2, 3, 999999], index=idx),
              "ace": pd.DataFrame({"timestamp_utc": idx, "speed": [1, 2, 3, 999999]})}
    limited = p.slice_frames(frames, idx[2])
    assert len(limited["speed"]) == len(limited["ace"]) == 3
    assert limited["speed"].max() == 3


def test_undefined_event_metrics_not_feasible():
    assert not p.feasible({"precision": None, "recall": 1}, {"precision": 1, "recall": 1})
    assert not p.feasible({"precision": .9, "recall": 1}, {"precision": 1, "recall": .1})


def test_selection_does_not_compensate_precision_with_recall():
    ref = {"precision": 1, "recall": .1}
    bad = {"status": "ok", "config": {"id": "bad"}, "metrics": {"mse": 1, "precision": .9, "recall": 1}}
    good = {"status": "ok", "config": {"id": "good"}, "metrics": {"mse": 2, "precision": 1, "recall": .1}}
    assert pick_rows([bad, good], ref) == [good]


def test_prediction_serialization_and_blend():
    origin = pd.Timestamp("2025-01-01", tz="UTC")
    a = pd.DataFrame({p.KEY[0]: [origin] * 72, p.KEY[1]: np.arange(1, 73), "pred_kms": 400.})
    b = a.assign(pred_kms=500.)
    merged = p.blend([a, b], [.25, .75])
    assert (merged.pred_kms == 475).all()
    assert merged[p.KEY[0]].iloc[0] == "2025-01-01T00:00:00Z"
    with pytest.raises(ValueError):
        p.canonical_predictions(a.iloc[:-1])


def test_missing_observation_mask_rejected():
    origin = pd.Timestamp("2025-01-01", tz="UTC")
    times = pd.date_range(origin + pd.Timedelta(hours=1), periods=72, freq="h")
    pred = pd.DataFrame({p.KEY[0]: [origin] * 72, p.KEY[1]: np.arange(1, 73), "pred_kms": 400.})
    frames = {"speed": pd.Series(400., index=times), "observed": pd.Series(True, index=times[:-1])}
    with pytest.raises(ValueError, match="observation mask"):
        p.score_development(pred, frames, pd.DatetimeIndex([origin]))


def test_stale_trial_rejected(tmp_path):
    from experiments.mse20_four_regimes.search import verify_trial
    file = tmp_path / "source.txt"
    file.write_text("original")
    ref = p.file_ref(file)
    record = {"status": "ok", **{k: ref for k in ("source", "protocol", "orchestrator", "split", "prediction", "model_metadata")},
              "fitted_artifacts": {"model": ref}}
    verify_trial(record)
    file.write_text("mutated")
    with pytest.raises(ValueError, match="stale artifact"):
        verify_trial(record)
