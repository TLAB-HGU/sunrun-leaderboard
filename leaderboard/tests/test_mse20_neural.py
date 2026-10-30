"""Synthetic-only adapter checks; no real-data training or official targets."""
import copy
import os

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from experiments.mse20_four_regimes import neural


def synthetic_frames():
    index = pd.date_range("2025-06-20", periods=2500, freq="h", tz="UTC")
    values = 420 + 30 * np.sin(np.arange(len(index)) / 25)
    observed = pd.Series(True, index=index)
    observed.iloc[::31] = False
    base = pd.DataFrame({col: 0.2 + np.sin(np.arange(len(index)) / 50) * 0.1
                         for col in neural.CH_COLUMNS}, index=index)
    base.iloc[::29, 0] = np.nan
    return {"speed": pd.Series(values, index=index), "observed": observed, "base": base}


def config():
    return {**neural.candidate_grid()[0], "max_epochs": 2, "max_train_origins": 24, "batch_size": 8}


def test_preregistered_grid():
    grid = neural.candidate_grid()
    assert len(grid) == 4
    assert {c["width"] for c in grid} == {32, 64}
    assert {c["loss"] for c in grid} == {"mse", "light_peak"}
    assert all(c["seed"] == 42 and c["max_epochs"] <= 60 and c["patience"] <= 6 for c in grid)


def test_fit_backward_saved_roundtrip_and_future_causality(tmp_path):
    frames = synthetic_frames()
    cutoff = frames["speed"].index[850]
    origins = frames["speed"].index[[950, 970]]
    device = os.environ.get("MSE20_TEST_DEVICE", "cpu")
    predictions, metadata = neural.fit_predict(frames, cutoff, origins, config(), device, tmp_path / "model", fixed_epochs=2)
    assert len(predictions) == 144 and np.isfinite(predictions.pred_kms).all()
    assert metadata["selected_epochs"] == 2
    assert pd.Timestamp(metadata["train_label_max"]) <= cutoff
    assert metadata["history_hours"] == 168 and metadata["recurrence_days"] == [26, 27, 28]
    restored = neural.predict_saved(frames, origins, tmp_path / "model", device)
    np.testing.assert_allclose(predictions.pred_kms, restored.pred_kms, rtol=0, atol=1e-5)
    # Network took actual gradient steps, rather than returning baseline unchanged.
    baseline = neural._inputs(frames, origins)[-1]
    assert np.max(np.abs(predictions.pred_kms.to_numpy().reshape(-1, 72) - baseline)) > 1e-4
    before = neural.predict_saved(frames, origins[:1], tmp_path / "model", device)
    deleted = {key: value.loc[:origins[0]].copy() for key, value in frames.items()}
    after = neural.predict_saved(deleted, origins[:1], tmp_path / "model", device)
    np.testing.assert_array_equal(before.pred_kms, after.pred_kms)
    poisoned = copy.deepcopy(frames)
    for key, value in poisoned.items():
        value.loc[value.index > origins[0]] = False if key == "observed" else 999999
    poisoned["base"]["future_regime"] = 999999
    after = neural.predict_saved(poisoned, origins[:1], tmp_path / "model", device)
    np.testing.assert_array_equal(before.pred_kms, after.pred_kms)
    with (tmp_path / "model/model.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        neural.predict_saved(frames, origins, tmp_path / "model", device)


def test_train_only_scaler_and_model_parameters(tmp_path):
    frames = synthetic_frames()
    cutoff = frames["speed"].index[850]
    origins = frames["speed"].index[[950]]
    _, original = neural.fit_predict(frames, cutoff, origins, config(), "cpu", tmp_path / "original", fixed_epochs=1)
    changed = copy.deepcopy(frames)
    for key in ("speed", "base"):
        changed[key].loc[changed[key].index > cutoff] = 12345
    _, poisoned = neural.fit_predict(changed, cutoff, origins, config(), "cpu", tmp_path / "poisoned", fixed_epochs=1)
    assert original["scaler"] == poisoned["scaler"]
    one = torch.load(tmp_path / "original/model.pt", weights_only=True)
    two = torch.load(tmp_path / "poisoned/model.pt", weights_only=True)
    assert all(torch.equal(one[key], two[key]) for key in one)


def test_missing_masks_and_explicit_allowlist():
    frames = synthetic_frames()
    origins = frames["speed"].index[[950]]
    raw = neural._inputs(frames, origins)
    assert (raw[1] == 0).any()
    scale = neural._scaler(raw)
    transformed = neural._transform(raw, scale)
    assert (transformed[0][:, 0, :][raw[1] == 0] == 0).all()
    frames["base"]["future_regime"] = 123456
    alternate = neural._inputs(frames, origins)
    for left, right in zip(raw, alternate):
        np.testing.assert_array_equal(left, right)


def test_epoch_and_split_guards(tmp_path):
    frames = synthetic_frames()
    origins = frames["speed"].index[[1200]]
    with pytest.raises(ValueError, match="development cutoff"):
        neural.fit_predict(frames, frames["speed"].index[850], origins, config(), "cpu", tmp_path / "bad")
    with pytest.raises(ValueError, match="cannot consume validation"):
        neural.fit_predict(frames, frames["speed"].index[850], origins, config(), "cpu", tmp_path / "bad", validation_origins=origins, fixed_epochs=1)
    with pytest.raises(ValueError, match="outside purged"):
        neural.fit_predict(frames, "2025-07-31T23:00Z", origins[:1] - pd.Timedelta(days=20), config(), "cpu", tmp_path / "bad", validation_origins=origins[:1] - pd.Timedelta(days=20))
    predictions, meta = neural.fit_predict(frames, "2025-07-31T23:00Z", origins, config(), "cpu", tmp_path / "dev", validation_origins=origins)
    assert len(predictions) == 72 and meta["epoch_rule"] == "development_early_stopping"
    assert 1 <= meta["selected_epochs"] <= 2
    with pytest.raises(FileExistsError):
        neural.fit_predict(frames, "2025-07-31T23:00Z", origins, config(), "cpu", tmp_path / "dev", validation_origins=origins)


@pytest.mark.parametrize("width", [32, 64])
def test_prediction_independent_of_requested_batch_shape(width):
    """Full464, partial last batch, and three-origin replay agree numerically."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    device = os.environ.get("MSE20_TEST_DEVICE", "cpu")
    aux_size = (216 + len(neural.CH_COLUMNS)) * 2
    model = neural.TemporalResidual(width, aux_size).to(device)
    with torch.no_grad():
        # Exercise the nonzero trained-head path without any real-data fit.
        model.head[-1].weight.normal_(0, 0.1)
    rng = np.random.default_rng(42)
    data = (rng.normal(size=(464, 2, 168)).astype(np.float32),
            rng.normal(size=(464, aux_size)).astype(np.float32),
            np.full((464, 72), 450.0))
    full = neural._predict(model, data, device)
    ix = np.array([0, 255, 463])
    subset = neural._predict(model, tuple(value[ix] for value in data), device)
    np.testing.assert_allclose(full[ix], subset, rtol=0, atol=1e-5)
