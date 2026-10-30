"""Synthetic-only tree adapter regressions; never load official truth."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.ch_breakthrough_v2.common import build_features
from experiments.ch_breakthrough_v2.physical_quality import (
    build_ch_quality_frame, build_physics_frame,
)
from experiments.ch_breakthrough_v2.recurrence import TREC_COLUMNS
from experiments.mse20_four_regimes import tree


def _sources():
    index = pd.date_range("2024-01-01", periods=1100, freq="h", tz="UTC")
    speed = 420 + 40 * np.sin(np.arange(len(index)) / 40)
    ace = pd.DataFrame({"filled_speed_kms": speed, "was_missing": 0}, index=index)
    ch = pd.DataFrame({"ch_t035_area": 0.03 + 0.001 * np.sin(np.arange(len(index)))}, index=index)
    raw = pd.DataFrame({"ace_density_cm3": 5., "ace_temperature_k": 100000.,
                        "ace_bt_nt": 5., "ace_speed_kms": speed}, index=index)
    return ace, ch, raw


def _frames(ace, ch, raw):
    return {"base": build_features(ace, ch),
            "physics": build_physics_frame(raw, ace.index),
            "quality": build_ch_quality_frame(ch, ace.index),
            "speed": ace["filled_speed_kms"].copy()}


@pytest.fixture
def fitted(tmp_path):
    sources = _sources()
    frames = _frames(*sources)
    cutoff = sources[0].index[850]
    origins = sources[0].index[[950, 951]]
    config = {"id": "synthetic", "params": {"n_estimators": 3, "max_depth": 2},
              "max_train_origins": 12, "seed": 42}
    preds, meta = tree.fit_predict(frames, cutoff, origins, config, "cpu", tmp_path)
    return sources, frames, cutoff, origins, config, preds, meta, tmp_path


def test_candidate_grid_fixed_bounded_and_champion():
    grid = tree.candidate_grid()
    assert grid == tree.candidate_grid()
    assert len(grid) <= 12
    assert len({x["id"] for x in grid}) == len(grid)
    assert sum(x["max_train_origins"] == 12000 for x in grid) == 1
    assert all(x["max_train_origins"] in (6000, 12000) and x["seed"] == 42 for x in grid)
    path = Path(tree.__file__).resolve().parents[2] / "store/ch-breakthrough-v2/upload/preparation.json"
    hp = json.loads(path.read_text())[0]["config"]["hyperparameters"]
    assert grid[0]["params"] == hp["xgboost"]
    assert grid[0]["max_train_origins"] == hp["max_train_origins"]


def test_roundtrip_full_schema_training_cutoff(fitted):
    _, frames, cutoff, origins, _, preds, meta, path = fitted
    assert len(preds) == 144
    assert np.isfinite(preds["pred_kms"]).all()
    assert pd.Timestamp(meta["train_label_max"]) <= cutoff
    assert meta["n_train_origins"] == 12
    assert set(TREC_COLUMNS) <= set(meta["columns"])
    assert meta["base_columns"] == list(tree._canonical_columns())
    pd.testing.assert_frame_equal(preds, tree.predict_saved(frames, origins, path, "cpu"), check_exact=True)


def test_perturb_and_delete_future_sources_leave_features_predictions_identical(fitted):
    sources, frames, _, origins, _, _, _, path = fitted
    origin = origins[:1]
    expected = tree.predict_saved(frames, origin, path, "cpu")
    for truncate in (False, True):
        altered = []
        for source in sources:
            source = source.copy()
            if truncate:
                source = source.loc[:origin[0]]
            else:
                source.loc[source.index > origin[0], :] = 999999.
            altered.append(source)
        changed = _frames(*altered)
        pd.testing.assert_frame_equal(tree._features(frames).loc[:origin[0]],
                                      tree._features(changed).loc[:origin[0]])
        pd.testing.assert_frame_equal(expected, tree.predict_saved(changed, origin, path, "cpu"), check_exact=True)


def test_future_labels_do_not_change_fit(fitted, tmp_path):
    _, frames, cutoff, origins, config, preds, meta, _ = fitted
    changed = copy.deepcopy(frames)
    # Poison after cutoff but preserve issuance history at prediction origins.
    # Model bytes, not forecast anchors, must remain identical.
    changed["speed"].loc[changed["speed"].index > cutoff] = 1900.
    out = tmp_path / "poisoned"
    _, new_meta = tree.fit_predict(changed, cutoff, origins, config, "cpu", out)
    assert new_meta["model_sha256"] == meta["model_sha256"]
    assert new_meta["train_origin_sha256"] == meta["train_origin_sha256"]
    pd.testing.assert_frame_equal(preds, tree.predict_saved(frames, origins, out, "cpu"), check_exact=True)


def test_future_regime_not_in_feature_allowlist(fitted):
    _, frames, _, origins, _, _, _, path = fitted
    changed = copy.deepcopy(frames)
    changed["physics"]["future_regime"] = 1.
    with pytest.raises(ValueError, match="allowlist"):
        tree.predict_saved(changed, origins, path, "cpu")


def test_saved_model_tamper_rejected(fitted):
    _, frames, _, origins, _, _, _, path = fitted
    with (path / "model.ubj").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        tree.predict_saved(frames, origins, path, "cpu")


def test_existing_artifacts_and_bad_origins_rejected(fitted):
    _, frames, cutoff, origins, config, _, _, path = fitted
    with pytest.raises(FileExistsError):
        tree.fit_predict(frames, cutoff, origins, config, "cpu", path)
    with pytest.raises(ValueError, match="unique"):
        tree.predict_saved(frames, origins.repeat(2), path, "cpu")


def test_matches_untouched_combined_forecaster(fitted):
    from experiments.ch_breakthrough_v2.recurrence import RecurrenceResidualForecaster

    _, frames, cutoff, origins, config, preds, _, _ = fitted
    fore = RecurrenceResidualForecaster(params=config["params"], device="cpu",
                                        seed=config["seed"], use_target=True)
    fore.fit(tree._features(frames), frames["speed"], cutoff,
             max_origins=config["max_train_origins"])
    reference = fore.predict(tree._features(frames), frames["speed"], origins)
    pd.testing.assert_frame_equal(preds, reference, check_exact=True)
