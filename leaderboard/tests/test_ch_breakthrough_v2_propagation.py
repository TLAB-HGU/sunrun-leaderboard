"""Focused tests for experiments.ch_breakthrough_v2.propagation."""
import inspect
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from experiments.ch_breakthrough_v2 import propagation as prop
from experiments.ch_breakthrough_v2.common import (
    HORIZON_HOURS,
    PooledResidualForecaster,
    as_speed_series,
    build_features,
    mean_reversion_predict,
    score_forecast,
    observed_mask,
    utc,
)


def _synthetic(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    speed = 400 + 40 * np.sin(np.arange(n) / 48.0) + rng.normal(0, 8, n)
    ace = pd.DataFrame({"timestamp_utc": index,
                        "filled_speed_kms": speed,
                        "was_missing": np.zeros(n, dtype=int)})
    ch = {}
    for tag in ("t035", "t045", "t055"):
        ch[f"ch_{tag}_area"] = 0.02 + 0.005 * np.sin(np.arange(n) / 72.0)
        ch[f"ch_{tag}_top1_lon"] = 20 * np.sin(np.arange(n) / 100.0)
        ch[f"ch_{tag}_top1_lat"] = 10 * np.cos(np.arange(n) / 120.0)
        ch[f"ch_{tag}_top1_lon_width"] = 25.0 + rng.normal(0, 1, n)
        ch[f"ch_{tag}_equatorial_area"] = 0.012 + 0.002 * np.sin(np.arange(n) / 80.0)
        ch[f"ch_{tag}_top1_area"] = 0.008 + 0.001 * np.sin(np.arange(n) / 90.0)
        ch[f"ch_{tag}_top1_lat_width"] = 18.0 + rng.normal(0, 1, n)
        ch[f"ch_{tag}_meridian15_area"] = 0.006
        ch[f"ch_{tag}_meridian30_area"] = 0.010
    ch_frame = pd.DataFrame(ch, index=pd.DatetimeIndex(index, name="slot"))
    return ace, ch_frame


def test_imports_frozen_common():
    src = inspect.getsource(prop)
    assert "ch_breakthrough_v2.common" in src
    for name in ("build_features", "mean_reversion_predict", "purged_split",
                 "subsample_origins", "score_forecast", "write_manifest",
                 "file_identity", "acquire_gpu_lease", "ensure_device"):
        assert name in src, f"propagation must use common.{name}"
    assert prop.HUXT_STATUS.startswith("not_installed")
def test_screening_subsample_cap_recorded():
    origins = pd.date_range("2025-01-05", periods=500, freq="h", tz="UTC")
    kept, sampling = prop._subsample_with_cap(
        origins, step="6h", max_origins=20, seed=42)
    assert len(kept) == 20
    assert sampling["step"] == "6h"
    assert sampling["requested"] == 500
    assert sampling["kept"] == 20
    assert sampling["cap_applied"] is True
    assert (kept == kept.floor("6h")).all()


def test_no_future_speed_used():
    ace, ch = _synthetic()
    feat0, _ = prop.build_propagation_features(ace, ch, b0_raw=pd.Series(dtype=float))
    origin = feat0.index[len(feat0) // 2]
    # Perturb only strictly-future speed and CH geometry.
    ace_future = ace.copy()
    mask = pd.DatetimeIndex(utc(ace_future["timestamp_utc"])) > origin
    ace_future.loc[mask, "filled_speed_kms"] *= 1.5
    ch_future = ch.copy()
    ch_future.loc[ch_future.index > origin, "ch_t045_top1_lon"] += 40.0
    ch_future.loc[ch_future.index > origin, "ch_t045_top1_lat"] += 30.0
    feat1, _ = prop.build_propagation_features(
        ace_future, ch_future, b0_raw=pd.Series(dtype=float))
    row0 = feat0.loc[[origin]].reset_index(drop=True)
    row1 = feat1.loc[[origin]].reset_index(drop=True)
    pd.testing.assert_frame_equal(row0, row1)
    # Transit scenario columns are finite and driven by the past speed.
    assert np.isfinite(feat0["prop_transit_past_h"].to_numpy()).all()
    assert (feat0["prop_transit_past_h"] > 0).all()


def test_b0_source_recorded():
    ace, ch = _synthetic(n=800)
    _, info_empty = prop.build_propagation_features(
        ace, ch, b0_raw=pd.Series(dtype=float))
    assert info_empty["b0_source"]["fallback_constant"] == info_empty["b0_source"]["origins"]
    assert info_empty["b0_source"]["suvi_metadata"] == 0
    slots = pd.DatetimeIndex(utc(ch.index))
    b0 = pd.Series(-5.0 + np.zeros(len(slots)), index=slots, name="solar_b0")
    feat, info = prop.build_propagation_features(ace, ch, b0_raw=b0)
    assert info["b0_source"]["suvi_metadata"] == info["b0_source"]["origins"]
    assert (feat["prop_b0_source_fallback"] == 0).all()
    # Observer-relative latitude honours the B0 correction.
    lat0 = feat["prop_obs_rel_lat"] + feat["prop_b0_deg"]
    assert np.isfinite(lat0.to_numpy()).all()


def test_horizons_complete():
    ace, ch = _synthetic()
    speed = as_speed_series(ace)
    base = build_features(ace, ch)
    feat, _ = prop.build_propagation_features(
        ace, ch, b0_raw=pd.Series(dtype=float))
    cutoff = utc("2024-02-10T23:00:00Z")
    origins = pd.DatetimeIndex([utc("2024-02-15T00:00:00Z"),
                                utc("2024-02-15T06:00:00Z")])
    generic = PooledResidualForecaster(
        params={"n_estimators": 5, "max_depth": 2}, device="cpu")
    generic.fit(base, speed, cutoff, max_origins=60)
    transit = prop.TransitResidualForecaster(
        params={"n_estimators": 5, "max_depth": 2}, device="cpu")
    transit.fit(base, feat, speed, cutoff, max_origins=60)
    for preds in (generic.predict(base, speed, origins),
                  transit.predict(base, feat, speed, origins)):
        assert list(preds["horizon_hours"].unique()) == list(range(1, 73))
        assert len(preds) == len(origins) * HORIZON_HOURS
        assert np.isfinite(preds["pred_kms"].to_numpy()).all()


def test_generic_vs_transit_comparison_present(tmp_path):
    ace, ch = _synthetic()
    speed = as_speed_series(ace)
    base = build_features(ace, ch)
    feat, info = prop.build_propagation_features(
        ace, ch, b0_raw=pd.Series(dtype=float))
    cutoff = utc("2024-02-10T23:00:00Z")
    origins = pd.DatetimeIndex([utc("2024-02-15T00:00:00Z"),
                                utc("2024-02-15T12:00:00Z")])
    observed = observed_mask(speed, speed.index)
    params = {"n_estimators": 5, "max_depth": 2}
    generic = PooledResidualForecaster(params=params, device="cpu").fit(
        base, speed, cutoff, max_origins=60)
    transit = prop.TransitResidualForecaster(params=params, device="cpu").fit(
        base, feat, speed, cutoff, max_origins=60)
    preds_g = generic.predict(base, speed, origins)
    preds_t = transit.predict(base, feat, speed, origins)
    mse_g = score_forecast(preds_g, speed, observed, origins=origins)["mse"]
    mse_t = score_forecast(preds_t, speed, observed, origins=origins)["mse"]
    comparison = {"generic_mse": mse_g, "transit_mse": mse_t,
                  "transit_gain_vs_generic": (mse_g - mse_t) / mse_g,
                  "generic_n_columns": len(generic.columns),
                  "transit_n_columns": len(transit.columns),
                  "b0_source": info["b0_source"]}
    assert len(transit.columns) > len(generic.columns)
    assert np.isfinite(comparison["transit_gain_vs_generic"])
    assert comparison["b0_source"]["origins"] == len(feat)
    assert "fallback_constant" in comparison["b0_source"]

def test_finite_or_nan_maps_inf_without_dropping_rows():
    frame = pd.DataFrame({"a": [1.0, np.inf, -np.inf, np.nan],
                          "b": [0.0, 1.0, 2.0, 3.0]})
    out = prop._finite_or_nan(frame)
    assert len(out) == len(frame)
    assert list(out.columns) == ["a", "b"]
    assert not np.isinf(out.to_numpy(dtype=float)).any()
    assert np.isnan(out.loc[1, "a"]) and np.isnan(out.loc[2, "a"])
    assert out.loc[0, "a"] == 1.0 and np.isnan(out.loc[3, "a"])


def test_quality_zero_area_no_inf():
    ace, ch = _synthetic(n=900)
    # Zero total area with nonzero meridian areas forces nonzero/0 divisions
    # in meridian/equatorial fractions; zero lat-width with nonzero lon-width
    # forces inf in aspect. Quality handling must yield NaN, never inf, with
    # no rows dropped.
    zero_idx = ch.index[100:160]
    ch.loc[zero_idx, "ch_t045_area"] = 0.0
    ch.loc[zero_idx, "ch_t045_top1_lat_width"] = 0.0
    feat, _ = prop.build_propagation_features(
        ace, ch, b0_raw=pd.Series(dtype=float))
    assert len(feat) == len(feat.index.unique())
    vals = feat.to_numpy(dtype=float)
    assert not np.isinf(vals).any()
    for col in ("prop_meridian15_frac", "prop_meridian30_frac",
                "prop_equatorial_frac", "prop_aspect"):
        affected = feat.loc[zero_idx.intersection(feat.index), col].to_numpy(
            dtype=float)
        assert not np.isinf(affected).any()
    # Pooled design matrix also carries no inf into XGBoost.
    speed = as_speed_series(ace)
    base = build_features(ace, ch)
    transit = prop.TransitResidualForecaster(
        params={"n_estimators": 5, "max_depth": 2}, device="cpu")
    cutoff = utc("2024-01-28T23:00:00Z")
    origins = feat.index[(feat.index > cutoff + pd.Timedelta(hours=72))][:4]
    _, big, _, _ = transit._design(base, feat, origins, speed)
    assert not np.isinf(big.to_numpy(dtype=float)).any()


def test_predict_future_perturbation_identical():
    ace, ch = _synthetic()
    speed = as_speed_series(ace)
    base = build_features(ace, ch)
    feat, _ = prop.build_propagation_features(
        ace, ch, b0_raw=pd.Series(dtype=float))
    cutoff = utc("2024-02-10T23:00:00Z")
    origins = pd.DatetimeIndex([utc("2024-02-15T00:00:00Z"),
                                utc("2024-02-15T06:00:00Z")])
    params = {"n_estimators": 5, "max_depth": 2}
    generic = PooledResidualForecaster(params=params, device="cpu").fit(
        base, speed, cutoff, max_origins=60)
    transit = prop.TransitResidualForecaster(params=params, device="cpu").fit(
        base, feat, speed, cutoff, max_origins=60)
    preds_g0 = generic.predict(base, speed, origins)
    preds_t0 = transit.predict(base, feat, speed, origins)
    # Perturb only strictly-future truth (post-origin target hours).
    speed_future = speed.copy()
    future_mask = speed_future.index > origins.max()
    speed_future.loc[future_mask] = speed_future.loc[future_mask] * 1.5 + 100.0
    preds_g1 = generic.predict(base, speed_future, origins)
    preds_t1 = transit.predict(base, feat, speed_future, origins)
    pd.testing.assert_frame_equal(preds_g0, preds_g1)
    pd.testing.assert_frame_equal(preds_t0, preds_t1)


def test_horizon_interactions_causal_no_future_speed():
    ace, ch = _synthetic()
    feat0, _ = prop.build_propagation_features(
        ace, ch, b0_raw=pd.Series(dtype=float))
    origin = feat0.index[len(feat0) // 2]
    ace_future = ace.copy()
    mask = pd.DatetimeIndex(utc(ace_future["timestamp_utc"])) > origin
    ace_future.loc[mask, "filled_speed_kms"] *= 2.0
    feat1, _ = prop.build_propagation_features(
        ace_future, ch, b0_raw=pd.Series(dtype=float))
    origins = pd.DatetimeIndex([origin])
    inter0 = prop.horizon_interactions(feat0, origins)
    inter1 = prop.horizon_interactions(feat1, origins)
    for key in ("prop_rot_dist_h", "prop_rot_weight_h",
                "prop_transit_gauss_past_h", "prop_conn_h"):
        np.testing.assert_array_equal(inter0[key], inter1[key])
