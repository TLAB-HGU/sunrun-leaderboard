"""Tests for experiments.ch_breakthrough_v2.recurrence (solar-rotation lane)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.ch_breakthrough_v2 import recurrence
from experiments.ch_breakthrough_v2.common import (
    HORIZONS,
    PooledResidualForecaster,
    as_speed_series,
    utc,
)


def _synthetic(n=1200, seed=7, missing_at=()):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    speed = (400 + 30 * np.sin(np.arange(n) / 24 * 2 * np.pi)
             + rng.normal(0, 5, n))
    was_missing = np.zeros(n, dtype=int)
    for pos in missing_at:
        if 0 <= pos < n:
            was_missing[pos] = 1
    ace = pd.DataFrame({"timestamp_utc": idx, "filled_speed_kms": speed,
                        "was_missing": was_missing})
    cols = {}
    for tag in ("t035", "t045", "t055"):
        cols[f"ch_{tag}_area"] = np.abs(rng.normal(0.02, 0.005, n))
        cols[f"ch_{tag}_top1_lon"] = rng.normal(0, 20, n)
        cols[f"ch_{tag}_top1_lat"] = rng.normal(0, 15, n)
        cols[f"ch_{tag}_top1_lon_width"] = np.abs(rng.normal(20, 5, n))
        cols[f"ch_{tag}_equatorial_area"] = np.abs(rng.normal(0.01, 0.003, n))
    ch = pd.DataFrame(cols, index=idx)
    ch.index.name = "slot"
    return ace, ch


def test_recurrence_imports_common_foundation():
    assert recurrence._common.PooledResidualForecaster is PooledResidualForecaster
    for name in ("build_features", "purged_split", "subsample_origins",
                 "score_forecast", "write_manifest", "file_identity",
                 "check_cutoff"):
        assert callable(getattr(recurrence._common, name)), name
    for name in ("build_recurrence_features", "horizon_target_features",
                 "target_source_time", "RecurrenceResidualForecaster",
                 "arm_columns", "arm_use_target", "train", "evaluate",
                 "train_forecaster", "evaluate_forecaster", "compare_arms",
                 "run_experiment"):
        assert callable(getattr(recurrence, name)), name
    assert set(recurrence.ARMS) == {"recent-ACE", "origin-recurrence",
                                    "target-recurrence",
                                    "target-recurrence+CH"}


def test_origin_control_sources_leq_issuance_26_27_28d():
    ace, ch = _synthetic()
    full = recurrence.build_recurrence_features(ace, ch)
    origin = utc("2024-02-15T00:00:00Z")
    assert origin in full.index
    row = full.loc[origin]
    speed = as_speed_series(ace)
    grid = speed.reindex(pd.date_range(speed.index.min(), speed.index.max(),
                                       freq="h"))
    for day, lag in zip((26, 27, 28), (624, 648, 672)):
        # origin-only control: source is o - lag, never target-relative
        expected = float(grid.loc[origin - pd.Timedelta(hours=lag)])
        assert row[f"orec{day}d_lag0"] == np.float32(expected)
        window = grid.loc[origin - pd.Timedelta(hours=lag + 12):
                          origin - pd.Timedelta(hours=lag - 12)]
        assert abs(float(row[f"orec{day}d_mean_pm12h"])
                   - float(window.mean())) < 1e-3
        # no target-relative columns live per-origin under a false name
        assert f"rec{day}d_lag0" not in full.columns
    # perturbing every source timestamp after issuance leaves the row unchanged
    future_ace = ace.copy()
    future_ace.loc[future_ace["timestamp_utc"] > origin,
                   "filled_speed_kms"] *= 3.0
    future_ch = ch.copy()
    future_ch.loc[future_ch.index > origin] *= 5.0
    altered = recurrence.build_recurrence_features(future_ace, future_ch)
    pd.testing.assert_series_equal(full.loc[origin], altered.loc[origin],
                                   check_names=False)


def test_target_relative_alignment_at_horizons_1_24_72():
    ace, ch = _synthetic()
    speed = as_speed_series(ace)
    grid = speed.reindex(pd.date_range(speed.index.min(), speed.index.max(),
                                       freq="h"))
    origin = pd.DatetimeIndex(utc(["2024-02-15T00:00:00Z"]))
    inter = recurrence.horizon_target_features(speed, origin)
    for day in (26, 27, 28):
        for horizon in (1, 24, 72):
            col = horizon - 1  # 0-based position in the (1, 72) matrix
            source = origin[0] + pd.Timedelta(hours=horizon) \
                - pd.Timedelta(hours=day * 24)
            assert recurrence.target_source_time(
                origin[0], horizon, day) == source
            # sources are known past at issuance for every horizon 1..72
            assert source <= origin[0]
            assert inter[f"trec{day}d"][0, col] == float(grid.loc[source])
            assert inter[f"trec{day}d_missing"][0, col] == 0.0
            current = float(grid.loc[origin[0]])
            assert inter[f"trec{day}d_diff"][0, col] == \
                np.float64(current - float(grid.loc[source]))
            assert inter[f"trec{day}d_src_age_h"][0, col] == day * 24 - horizon


def test_target_invariance_to_future_perturbation():
    ace, ch = _synthetic()
    full = recurrence.build_recurrence_features(ace, ch)
    speed = as_speed_series(ace)
    cols = recurrence.arm_columns(full, "target-recurrence")
    cutoff = utc("2024-02-05T00:00:00Z")
    fore = recurrence.train_forecaster(
        full[cols], speed, cutoff, params={"n_estimators": 5, "max_depth": 2},
        device="cpu", max_train_origins=40, use_target=True)
    origins = pd.date_range("2024-02-12", periods=2, freq="h", tz="UTC")
    preds = recurrence.evaluate(fore, full[cols], speed, origins)
    # perturb every sample strictly after the latest evaluated origin
    bumped_ace = ace.copy()
    bumped_ace.loc[bumped_ace["timestamp_utc"] > origins.max(),
                   "filled_speed_kms"] *= 3.0
    bumped_ch = ch.copy()
    bumped_ch.loc[bumped_ch.index > origins.max()] *= 5.0
    bumped_full = recurrence.build_recurrence_features(bumped_ace, bumped_ch)
    bumped_speed = as_speed_series(bumped_ace)
    bumped_preds = recurrence.evaluate(fore, bumped_full[cols], bumped_speed,
                                       origins)
    pd.testing.assert_frame_equal(preds, bumped_preds)


def test_horizon_specific_design_not_repeated_72x():
    ace, ch = _synthetic()
    speed = as_speed_series(ace)
    origin = pd.DatetimeIndex(utc(["2024-02-15T00:00:00Z"]))
    inter = recurrence.horizon_target_features(speed, origin)
    # target-relative values genuinely vary across the 72 horizons
    for day in (26, 27, 28):
        assert np.std(inter[f"trec{day}d"][0]) > 1e-6
        assert np.std(inter[f"trec{day}d_src_age_h"][0]) > 1e-6
    full = recurrence.build_recurrence_features(ace, ch)
    cols = recurrence.arm_columns(full, "target-recurrence")
    cutoff = utc("2024-02-05T00:00:00Z")
    fore = recurrence.train_forecaster(
        full[cols], speed, cutoff, params={"n_estimators": 5, "max_depth": 2},
        device="cpu", max_train_origins=40, use_target=True)
    assert any(c.startswith("trec") for c in fore.columns)
    assert "horizon_hours" in fore.columns
    assert not any(c.startswith("orec") for c in fore.columns)


def test_genuine_missingness_for_recurrent_sources():
    origin = utc("2024-02-15T00:00:00Z")
    idx = pd.date_range("2024-01-01", periods=1200, freq="h", tz="UTC")
    pos = int((origin - idx[0]).total_seconds() // 3600)
    # genuinely missing hours at the 27-day origin return and at one
    # target-relative source (origin + 24h - 27d)
    ace, ch = _synthetic(missing_at=(pos - 648, pos + 24 - 648))
    full = recurrence.build_recurrence_features(ace, ch)
    row = full.loc[origin]
    assert float(row["orec27d_missing_pm24h"]) >= 1.0
    speed = as_speed_series(ace)
    inter = recurrence.horizon_target_features(
        speed, pd.DatetimeIndex(utc([origin])))
    # horizon 24 sources the perturbed hour -> flag 1; horizon 1 does not
    assert inter["trec27d_missing"][0, 24 - 1] == 1.0
    assert inter["trec27d_missing"][0, 1 - 1] == 0.0


def test_extension_label_present_in_manifest_extra_and_readme():
    assert "beyond-trailing-5d" in recurrence.EXTENSION_LABEL
    assert "26/27/28" in recurrence.EXTENSION_LABEL
    assert recurrence.EXTENSION_LABEL in recurrence.README
    assert "BEYOND" in recurrence.README or "beyond" in recurrence.README
    assert "target-relative" in recurrence.README
    assert "origin" in recurrence.README.lower()
    import json
    import tempfile
    ace, ch = _synthetic()
    full = recurrence.build_recurrence_features(ace, ch)
    cols = recurrence.arm_columns(full, "target-recurrence+CH")
    speed = as_speed_series(ace)
    cutoff = utc("2024-02-05T00:00:00Z")
    fore = recurrence.train_forecaster(full[cols], speed, cutoff,
                                       params={"n_estimators": 5,
                                               "max_depth": 2},
                                       device="cpu", max_train_origins=40,
                                       use_target=True)
    with tempfile.TemporaryDirectory() as tmp:
        manifest = recurrence.write_manifest(
            Path(tmp) / "m.json", mode="recurrence-target-recurrence+CH",
            train_cutoff=cutoff, params={"n_estimators": 5},
            tree_counts={"n_estimators": 5}, columns=list(fore.columns),
            input_identities={}, seed=42,
            inference_seconds_per_origin=0.01,
            extra={"extension": recurrence.EXTENSION_LABEL})
        assert manifest["extra"]["extension"] == recurrence.EXTENSION_LABEL
        saved = json.loads((Path(tmp) / "m.json").read_text())
        assert saved["extra"]["extension"] == recurrence.EXTENSION_LABEL
        assert "common" in saved["code_hashes"]


def test_predict_covers_horizons_1_to_72():
    ace, ch = _synthetic()
    full = recurrence.build_recurrence_features(ace, ch)
    cols = recurrence.arm_columns(full, "origin-recurrence")
    assert not any(c.startswith("ch_") for c in cols)
    assert any(c.startswith("orec27d_") for c in cols)
    assert not any(c.startswith("trec") for c in cols)
    speed = as_speed_series(ace)
    cutoff = utc("2024-02-05T00:00:00Z")
    fore = recurrence.train(
        full[cols], speed, cutoff,
        params={"n_estimators": 5, "max_depth": 2}, device="cpu",
        max_train_origins=40, use_target=False)
    origins = pd.date_range("2024-02-12", periods=2, freq="h", tz="UTC")
    preds = recurrence.evaluate(fore, full[cols], speed, origins)
    assert set(preds["horizon_hours"]) == set(HORIZONS) == set(range(1, 73))
    assert len(preds) == 2 * 72
    assert np.isfinite(preds["pred_kms"]).all()


def test_matched_params_three_arm_comparison():
    ace, ch = _synthetic()
    full = recurrence.build_recurrence_features(ace, ch)
    speed = as_speed_series(ace)
    cutoff = utc("2024-02-05T00:00:00Z")
    origins = pd.date_range("2024-02-12", periods=2, freq="h", tz="UTC")
    params = {"n_estimators": 5, "max_depth": 2}
    out = recurrence.compare_arms(full, speed, cutoff, origins, params=params,
                                  device="cpu", max_train_origins=40)
    assert set(out["arms"]) == {"recent-ACE", "origin-recurrence",
                                "target-recurrence", "target-recurrence+CH"}
    assert out["params"] == params
    for arm, res in out["arms"].items():
        assert set(res["predictions"]["horizon_hours"]) == set(range(1, 73))
    assert (out["origins"] == pd.DatetimeIndex(utc(origins))).all()
    recent = set(out["arms"]["recent-ACE"]["columns"])
    origin = set(out["arms"]["origin-recurrence"]["columns"])
    target = out["arms"]["target-recurrence"]
    target_ch = out["arms"]["target-recurrence+CH"]
    assert not any(c.startswith("orec") for c in recent)
    assert not any(c.startswith("ch_") for c in recent)
    assert any(c.startswith("orec") for c in origin)
    assert not any(c.startswith("ch_") for c in origin)
    assert out["arms"]["origin-recurrence"]["use_target"] is False
    assert target["use_target"] is True
    assert any(c == "trec27d" for c in target["forecaster"].columns)
    assert not any(c.startswith("ch_") for c in target["columns"])
    assert any(c.startswith("ch_") for c in target_ch["columns"])
    assert any(c == "trec27d" for c in target_ch["forecaster"].columns)


def test_no_inf_reaches_xgboost_design():
    ace, ch = _synthetic()
    for include_ch in (True, False):
        full = recurrence.build_recurrence_features(ace, ch,
                                                    include_ch=include_ch)
        assert not np.isinf(full.to_numpy(dtype=float)).any()
    speed = as_speed_series(ace)
    origins = pd.DatetimeIndex(utc(["2024-02-15T00:00:00Z"]))
    inter = recurrence.horizon_target_features(speed, origins)
    for name, mat in inter.items():
        assert not np.isinf(np.asarray(mat, dtype=float)).any(), name


def test_eval_origins_purged_72h_after_cutoff():
    ace, ch = _synthetic(n=2500)
    full = recurrence.build_recurrence_features(ace, ch)
    cutoff = utc("2024-03-01T00:00:00Z")
    kept, sampling = recurrence.eval_origins_for_split(
        full.index, cutoff, "2024-03-01T00:00:00Z", "2024-04-15T00:00:00Z",
        step="6h")
    assert (kept.min() - cutoff) >= pd.Timedelta(hours=72)
    assert sampling["step"] == "6h"
    assert sampling["kept"] == len(kept)


def test_manifest_carries_required_identity_fields():
    import json
    import tempfile
    ace, ch = _synthetic()
    full = recurrence.build_recurrence_features(ace, ch)
    cols = recurrence.arm_columns(full, "target-recurrence")
    speed = as_speed_series(ace)
    cutoff = utc("2024-02-05T00:00:00Z")
    fore = recurrence.train_forecaster(
        full[cols], speed, cutoff, params={"n_estimators": 5, "max_depth": 2},
        device="cpu", max_train_origins=40, use_target=True)
    origins = pd.date_range("2024-02-12", periods=2, freq="h", tz="UTC")
    sampling = {"step": "hourly", "requested": 2, "kept": 2, "seed": 42}
    with tempfile.TemporaryDirectory() as tmp:
        res = recurrence.finish_arm_split(
            arm="target-recurrence", split_name="dev-A", full_features=full,
            speed=speed, forecaster=fore, cutoff=cutoff, fit_seconds=0.1,
            eval_origins=pd.DatetimeIndex(utc(origins)), sampling=sampling,
            params={"n_estimators": 5, "max_depth": 2}, device="cpu",
            device_info={"lease": "cpu-fallback"}, seed=42,
            max_train_origins=40, outdir=tmp,
            input_identities={"ace": {"sha256": "x", "path": "p", "bytes": 1,
                                              "files": 1}})
        manifest = json.loads(
            (Path(tmp) / "target-recurrence__dev-A.manifest.json")
            .read_text())
        assert "common" in manifest["code_hashes"]
        assert "caller" in manifest["code_hashes"]
        assert manifest["train_cutoff"] == cutoff.isoformat()
        assert manifest["params"]["n_estimators"] == 5
        assert manifest["tree_counts"]["n_estimators"] == 5
        assert manifest["columns"] == list(fore.columns)
        assert manifest["input_identities"]["ace"]["sha256"] == "x"
        assert manifest["sampling"]["kept"] == 2
        assert manifest["seed"] == 42
        assert manifest["inference_seconds_per_origin"] >= 0
        assert "versions" in manifest
        assert manifest["prediction"]["sha256"] == \
            manifest["extra"]["prediction_sha256"]
        assert manifest["extra"]["extension"] == recurrence.EXTENSION_LABEL
        assert manifest["extra"]["arm"] == "target-recurrence"
        assert manifest["extra"]["use_target"] is True
        assert res["parquet"].endswith(".parquet")
