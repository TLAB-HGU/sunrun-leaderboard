"""Lane tests for experiments.ch_breakthrough_v2.physical_quality.

Covers: physics-frame causality (+ valid Pdyn derivation), the
vintage_verified=0 limitation record, CH-quality causality/bounds, complete
1..72 horizons with all three arms present, and header/mask delimitation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.ch_breakthrough_v2.common import (  # noqa: E402
    HORIZON_HOURS,
    HORIZONS,
    PooledResidualForecaster,
    as_speed_series,
    build_features,
    observed_mask,
    utc,
)
from experiments.ch_breakthrough_v2.physical_quality import (  # noqa: E402
    ARMS,
    PARAMS,
    PDYN_FACTOR,
    arm_columns,
    build_ch_quality_frame,
    build_physics_frame,
    header_conventions_report,
    vintage_record,
)


def _synthetic(n=900, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    speed = 400 + 30 * np.sin(np.arange(n) / 24 * 2 * np.pi) + rng.normal(0, 5, n)
    ace = pd.DataFrame({"timestamp_utc": idx, "filled_speed_kms": speed,
                        "was_missing": np.zeros(n, dtype=int)})
    cols = {}
    for tag in ("t035", "t045", "t055"):
        cols[f"ch_{tag}_area"] = np.abs(rng.normal(0.02, 0.005, n))
        cols[f"ch_{tag}_top1_lon"] = rng.normal(0, 20, n)
        cols[f"ch_{tag}_top1_lat"] = rng.normal(0, 15, n)
        cols[f"ch_{tag}_top1_lon_width"] = np.abs(rng.normal(20, 5, n))
        cols[f"ch_{tag}_equatorial_area"] = np.abs(rng.normal(0.01, 0.003, n))
        cols[f"ch_{tag}_valid_fraction"] = np.clip(rng.normal(0.99, 0.02, n), 0, 1)
        cols[f"ch_{tag}_count"] = rng.integers(0, 5, n).astype(float)
        cols[f"ch_{tag}_max_area"] = np.abs(rng.normal(0.01, 0.003, n))
    ch = pd.DataFrame(cols, index=idx)
    ch.index.name = "slot"
    raw = pd.DataFrame({
        "point_in_time_vintage_verified": np.zeros(n, dtype=int),
        "ace_density_cm3": np.abs(rng.normal(5, 2, n)),
        "ace_temperature_k": np.abs(rng.normal(1e5, 2e4, n)),
        "ace_bt_nt": np.abs(rng.normal(5, 2, n)),
        "ace_speed_kms": speed + rng.normal(0, 3, n),
        "ace_swepam_status": np.zeros(n),
        "ace_mag_status": np.zeros(n),
    }, index=idx)
    raw.index.name = "timestamp_utc"
    raw.loc[raw.index[100], "ace_density_cm3"] = np.nan
    return ace, ch, raw


def test_physics_columns_causal_and_pdyn_valid():
    _, _, raw = _synthetic()
    t = utc("2024-01-20T00:00:00Z")
    future = raw.copy()
    future.loc[future.index > t, ["ace_density_cm3", "ace_temperature_k",
                                 "ace_bt_nt", "ace_speed_kms"]] *= 7.0
    origins = raw.index[raw.index <= t + pd.Timedelta(hours=5)]
    base = build_physics_frame(raw, origins)
    altered = build_physics_frame(future, origins)
    pd.testing.assert_frame_equal(base.loc[:t], altered.loc[:t])
    assert (base.loc[:t].index == origins[origins <= t]).all()
    # Valid derived dynamic pressure matches the documented formula.
    row = base.loc[t]
    n = raw.loc[t, "ace_density_cm3"]
    v = raw.loc[t, "ace_speed_kms"]
    assert row["phys_pdyn_lag0"] == np.float32(PDYN_FACTOR * n * v ** 2)
    # Missing density propagates to NaN pressure (native XGBoost missingness).
    bad_t = raw.index[100]
    assert np.isnan(base.loc[bad_t, "phys_pdyn_lag0"])
    assert base.loc[bad_t, "phys_dens_missing24"] >= 1
    # Short causal gradients only use past samples.
    assert set(HORIZONS) == set(range(1, 73))  # horizons constant sanity
    assert "phys_dens_diff_1" in base.columns and "phys_bt_diff_6" in base.columns


def test_vintage_limitation_recorded_never_verified():
    _, _, raw = _synthetic()
    record = vintage_record(raw)
    assert record["verified"] is False
    assert record["unique_values"] == [0]
    assert "NEVER" in record["limitation"] or "never" in record["limitation"].lower()
    assert "0" in record["limitation"]
    assert "verified" in record["column"]


def test_ch_quality_causal_bounded_and_ranked():
    _, ch, _ = _synthetic()
    t = utc("2024-01-20T00:00:00Z")
    ch_future = ch.copy()
    ch_future.loc[ch_future.index > t] *= 4.0
    origins = ch.index[ch.index <= t + pd.Timedelta(hours=5)]
    base = build_ch_quality_frame(ch, origins)
    altered = build_ch_quality_frame(ch_future, origins)
    pd.testing.assert_frame_equal(base.loc[:t], altered.loc[:t])
    assert (base["chq_rank_ok"].dropna().isin([0.0, 1.0])).all()
    for col in ("chq_agree_045_035", "chq_agree_055_045", "chq_agree_055_035"):
        assert (base[col].dropna() > 0).all()
    assert (base["chq_t045_vf_lag0"].dropna().le(1.0)).all()
    assert (base["chq_t035_area_absdiff_1"].dropna() >= 0).all()


def test_zero_area_masks_yield_nan_not_inf():
    # Real ch-hourly has zero-area rows; agreement ratios must be NaN, never
    # inf (XGBoost rejects inf inputs outright).
    ace, ch, raw = _synthetic()
    ch.loc[ch.index[50], ["ch_t035_area", "ch_t045_area", "ch_t055_area"]] = 0.0
    quality = build_ch_quality_frame(ch, ch.index)
    physics = build_physics_frame(raw, ch.index)
    for frame in (quality, physics):
        assert not np.isinf(frame.to_numpy(dtype=float)).any()
    row = quality.loc[ch.index[50]]
    assert np.isnan(row["chq_agree_045_035"]) or row["chq_agree_045_035"] >= 0

def test_horizons_complete_and_three_arm_comparison_present():
    ace, ch, raw = _synthetic(n=1600)
    speed = as_speed_series(ace)
    base = build_features(ace, ch, None)
    origins = base.index
    physics = build_physics_frame(raw, origins)
    quality = build_ch_quality_frame(ch, origins)
    cols = {arm: arm_columns(base, physics, quality, arm) for arm in ARMS}
    assert set(cols) == set(ARMS)
    assert len(cols["physical_ace"]) > len(cols["speed_only"])
    assert len(cols["physical_ace_ch_quality"]) > len(cols["physical_ace"])
    assert all(c.startswith("ace_") for c in cols["speed_only"])
    assert any(c.startswith("phys_") for c in cols["physical_ace"])
    assert any(c.startswith("chq_") for c in cols["physical_ace_ch_quality"])
    cutoff = utc(origins[int(len(origins) * 0.85)].isoformat())
    eval_origins = origins[(origins > cutoff + pd.Timedelta(hours=72))][:3]
    observed = observed_mask(speed, speed.index)
    assert len(observed) > 0
    for arm in ARMS:
        arm_features = pd.concat(
            [base[[c for c in cols[arm] if c in base.columns]],
             physics[[c for c in cols[arm] if c in physics.columns]],
             quality[[c for c in cols[arm] if c in quality.columns]]],
            axis=1)[cols[arm]]
        fore = PooledResidualForecaster(
            params={"n_estimators": 5, "max_depth": 2}, device="cpu", seed=0)
        fore.fit(arm_features, speed, cutoff, max_origins=30)
        preds = fore.predict(arm_features, speed, eval_origins)
        assert len(preds) == len(eval_origins) * HORIZON_HOURS
        assert sorted(preds["horizon_hours"].unique().tolist()) == list(HORIZONS)
        assert preds["pred_kms"].notna().all()


def test_header_report_delimits_without_wcs_claims():
    report = header_conventions_report()
    assert set(("CRPIX1", "CRPIX2", "RSUN")).issubset(report["consumed_keys"])
    assert report["wcs_sign_scale_error_asserted"] is False
    assert "re-extraction" in report["delimitation"] or "re-extraction" in str(report)
    assert len(report["masks_dir_listing"]) > 0
    assert report["masks_are_rendered_overlays_only"] is True


def test_observed_mask_uses_genuine_missingness_not_finite_fill():
    # Forward-filled finite speeds with was_missing==1 must stay unobserved;
    # a finite-only mask would wrongly mark them observed.
    ace, _, _ = _synthetic(n=800)
    ace.loc[10:19, "was_missing"] = 1  # finite filled values, genuinely missing
    ace.loc[30, "filled_speed_kms"] = -5.0  # invalid observation, flag clear
    speed = as_speed_series(ace)
    mask = observed_mask(speed, speed.index)
    assert mask.iloc[10:20].tolist() == [False] * 10
    assert bool(mask.iloc[30]) is False
    assert bool(mask.iloc[0]) is True and bool(mask.iloc[21]) is True


def test_predictions_invariant_to_future_truth_perturbation():
    # Prediction must not need future truth: corrupting or deleting every
    # speed sample after the last eval origin leaves predictions identical.
    ace, ch, raw = _synthetic(n=1600)
    speed = as_speed_series(ace)
    base = build_features(ace, ch, None)
    origins = base.index
    physics = build_physics_frame(raw, origins)
    quality = build_ch_quality_frame(ch, origins)
    cols = arm_columns(base, physics, quality, "physical_ace_ch_quality")
    arm_features = pd.concat(
        [base[[c for c in cols if c in base.columns]],
         physics[[c for c in cols if c in physics.columns]],
         quality[[c for c in cols if c in quality.columns]]],
        axis=1)[cols]
    cutoff = utc(origins[int(len(origins) * 0.8)].isoformat())
    eval_origins = origins[(origins > cutoff + pd.Timedelta(hours=72))][:3]
    fore = PooledResidualForecaster(
        params={"n_estimators": 5, "max_depth": 2}, device="cpu", seed=0)
    fore.fit(arm_features, speed, cutoff, max_origins=30)
    expected = fore.predict(arm_features, speed, eval_origins)
    last = eval_origins.max()
    corrupted = speed.copy()
    corrupted.loc[corrupted.index > last] *= 7.0
    pd.testing.assert_frame_equal(
        fore.predict(arm_features, corrupted, eval_origins), expected)
    dropped = speed.loc[speed.index <= last]
    pd.testing.assert_frame_equal(
        fore.predict(arm_features, dropped, eval_origins), expected)


def test_matched_params_identical_across_arms():
    # No per-arm tuning: every arm fits the same PARAMS object contents.
    ace, ch, raw = _synthetic(n=1600)
    speed = as_speed_series(ace)
    base = build_features(ace, ch, None)
    origins = base.index
    physics = build_physics_frame(raw, origins)
    quality = build_ch_quality_frame(ch, origins)
    cutoff = utc(origins[int(len(origins) * 0.8)].isoformat())
    fitted = {}
    for arm in ARMS:
        cols = arm_columns(base, physics, quality, arm)
        arm_features = pd.concat(
            [base[[c for c in cols if c in base.columns]],
             physics[[c for c in cols if c in physics.columns]],
             quality[[c for c in cols if c in quality.columns]]],
            axis=1)[cols]
        fore = PooledResidualForecaster(
            params=dict(PARAMS), device="cpu", seed=0)
        fore.fit(arm_features, speed, cutoff, max_origins=30)
        fitted[arm] = dict(fore.params)
    assert fitted["speed_only"] == fitted["physical_ace"] == \
        fitted["physical_ace_ch_quality"] == dict(PARAMS)
