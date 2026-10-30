"""Synthetic-only target-relative numerical CH alignment regressions."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.ch_breakthrough_v2.common import CH_TAGS, CH_BASE_COLUMNS
from experiments.ch_breakthrough_v2.recurrence import RecurrenceResidualForecaster
from experiments.mse20_four_regimes import tree, tree_alignment as alignment
from test_mse20_tree import _sources, _frames


@pytest.fixture
def frames():
    ace, ch, raw = _sources()
    for k, tag in enumerate(CH_TAGS):
        for j, col in enumerate(CH_BASE_COLUMNS):
            ch[f"ch_{tag}_{col}"] = (k + 1) * 10 + j + np.arange(len(ch)) / 100
    return _frames(ace, ch, raw)


@pytest.fixture
def fitted(frames, tmp_path):
    cutoff = frames["speed"].index[850]
    origins = frames["speed"].index[[950, 960]]
    config = {"id": "synthetic", "params": {"n_estimators": 3, "max_depth": 2},
              "max_train_origins": 12, "seed": 42}
    preds, meta = alignment.fit_predict(frames, cutoff, origins, config, "cpu", tmp_path)
    return cutoff, origins, config, preds, meta, tmp_path


def test_exact_grid():
    first, second = alignment.candidate_grid()
    assert first["params"] == tree.candidate_grid()[0]["params"]
    assert second["params"] == {**first["params"], "max_depth": 4, "reg_lambda": 5.0}
    assert all(c["seed"] == 42 and c["max_train_origins"] == 6000 for c in (first, second))
    assert first["id"] != second["id"]
    assert len(alignment.CH_COLUMNS) == 15


def test_all_source_times_and_parent_design_unchanged(frames):
    features = alignment._features(frames)
    origins = features.index[[950, 960]]
    fore = alignment.AlignmentResidualForecaster()
    _, design, anchor = fore._build_predict_matrix(features, origins, frames["speed"])
    _, base, expected_anchor = RecurrenceResidualForecaster()._build_predict_matrix(features, origins, frames["speed"])
    pd.testing.assert_frame_equal(design[base.columns], base)
    np.testing.assert_array_equal(anchor, expected_anchor)
    for i, origin in enumerate(origins):
        for horizon in range(1, 73):
            for delay in alignment.DELAYS:
                source = origin + pd.Timedelta(hours=horizon - delay)
                assert source <= origin
                for col in alignment.CH_COLUMNS:
                    assert design.loc[i * 72 + horizon - 1, f"align{delay}h_{col}"] == features.loc[source, col]
    col = f"align72h_{alignment.CH_COLUMNS[0]}"
    assert design.loc[71, col] == features.loc[origins[0], alignment.CH_COLUMNS[0]]
    assert design.loc[0, col] != design.loc[71, col]


def test_missing_rows_and_cells_not_filled(frames):
    features = alignment._features(frames)
    origin = features.index[950]
    missing = origin - pd.Timedelta(hours=71)
    features = features.drop(missing)
    col = alignment.CH_COLUMNS[0]
    features.loc[origin, col] = np.nan
    _, design, _ = alignment.AlignmentResidualForecaster()._build_predict_matrix(features, [origin], frames["speed"])
    for column in alignment.CH_COLUMNS:
        assert np.isnan(design.loc[0, f"align72h_{column}"])
        assert design.loc[0, f"align72h_{column}_available"] == 0
    assert np.isnan(design.loc[71, f"align72h_{col}"])
    assert design.loc[71, f"align72h_{col}_available"] == 0


def test_ubj_full_subset_parity_and_cutoff(frames, fitted):
    cutoff, origins, _, preds, meta, path = fitted
    assert meta["family"] == "tree_alignment"
    assert pd.Timestamp(meta["train_label_max"]) <= cutoff
    assert meta["columns"] == list(alignment._design_columns())
    assert meta["base_columns"] == list(alignment._canonical_columns())
    pd.testing.assert_frame_equal(preds, alignment.predict_saved(frames, origins, path, "cpu"), check_exact=True)
    subset = alignment.predict_saved(frames, origins[1:], path, "cpu")
    pd.testing.assert_frame_equal(preds.iloc[72:].reset_index(drop=True), subset, check_exact=True)


@pytest.mark.parametrize("drop", [False, True])
def test_future_poison_and_drop_all_horizons(frames, fitted, drop):
    _, origins, _, _, _, path = fitted
    for origin in origins:
        changed = copy.deepcopy(frames)
        for key, frame in changed.items():
            if drop:
                changed[key] = frame.loc[:origin]
            else:
                frame.loc[frame.index > origin] = 999999.
        pd.testing.assert_frame_equal(alignment.predict_saved(frames, [origin], path, "cpu"),
                                      alignment.predict_saved(changed, [origin], path, "cpu"), check_exact=True)


def test_post_cutoff_labels_and_features_do_not_change_model(frames, fitted, tmp_path):
    cutoff, origins, config, _, meta, _ = fitted
    changed = copy.deepcopy(frames)
    for frame in changed.values():
        frame.loc[frame.index > cutoff] = 1900.
    _, poisoned = alignment.fit_predict(changed, cutoff, origins, config, "cpu", tmp_path / "poison")
    assert poisoned["model_sha256"] == meta["model_sha256"]
    assert poisoned["train_origin_sha256"] == meta["train_origin_sha256"]


@pytest.mark.parametrize("field", ["columns", "base_columns", "delays", "subclass_identity", "config", "train_label_max"])
def test_tampered_metadata_rejected(frames, fitted, field):
    _, origins, _, _, meta, path = fitted
    meta[field] = "2099-01-01T00:00:00Z" if field == "train_label_max" else []
    (path / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        alignment.predict_saved(frames, origins, path, "cpu")


def test_model_tamper_rejected(frames, fitted):
    _, origins, _, _, _, path = fitted
    with (path / "model.ubj").open("ab") as out:
        out.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        alignment.predict_saved(frames, origins, path, "cpu")


def test_fixed_schema_ignores_injected_future_columns(frames):
    expected = alignment._features(frames)
    for frame in (frames["base"], frames["physics"], frames["quality"]):
        frame["future_target"] = 1.
    pd.testing.assert_frame_equal(expected, alignment._features(frames))
    frames["base"] = frames["base"].drop(columns=alignment.CH_COLUMNS[0])
    with pytest.raises(ValueError, match="schema"):
        alignment._features(frames)
