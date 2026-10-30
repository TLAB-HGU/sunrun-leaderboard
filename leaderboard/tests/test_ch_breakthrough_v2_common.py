"""Focused contract tests for experiments.ch_breakthrough_v2.common."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.ch_breakthrough_v2.common import (
    HORIZON_HOURS,
    PooledResidualForecaster,
    as_speed_series,
    build_features,
    check_cutoff,
    code_hash,
    code_hashes,
    file_identity,
    mean_reversion_predict,
    observed_mask,
    purged_split,
    runtime_versions,
    subsample_origins,
    utc,
    validate_manifest_identities,
    write_manifest,
)


def _synthetic(n=600, seed=0):
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
    ch = pd.DataFrame(cols, index=idx)
    ch.index.name = "slot"
    return ace, ch


def test_causality_no_future_source_times():
    ace, ch = _synthetic()
    t = utc("2024-01-20T00:00:00Z")
    future = ace.copy()
    future.loc[future["timestamp_utc"] > t, "filled_speed_kms"] *= 3.0
    ch_future = ch.copy()
    ch_future.loc[ch_future.index > t] *= 5.0
    base = build_features(ace, ch)
    altered = build_features(future, ch_future)
    row_base = base.loc[:t].iloc[[-1]]
    row_alt = altered.loc[:t].iloc[[-1]]
    assert row_base.index[0] == t
    pd.testing.assert_frame_equal(row_base, row_alt)


def test_purge_at_split_and_cutoff():
    origins = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    cutoff = utc("2024-01-05T00:00:00Z")
    train, test = purged_split(origins, cutoff)
    assert ((train + pd.Timedelta(hours=72)) <= cutoff).all()
    assert (test.min() - cutoff) >= pd.Timedelta(hours=72)
    assert len(train.intersection(test)) == 0
    try:
        check_cutoff(train, pd.DatetimeIndex([cutoff + pd.Timedelta(hours=1)]),
                     cutoff)
    except ValueError:
        pass
    else:
        raise AssertionError("label past cutoff must raise")


def test_mean_reversion_reproduces_formula():
    n = 648 + 10
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    speed = pd.Series(400.0, index=idx)
    speed.iloc[-1] = 500.0  # last value only deviates
    origins = idx[-5:]
    preds = mean_reversion_predict(speed, origins)
    assert set(preds["horizon_hours"]) == set(range(1, 73))
    mean = speed.rolling(648, min_periods=648).mean().loc[origins].to_numpy()
    last = speed.loc[origins].to_numpy()
    horizons = np.arange(1, 73)
    expected = mean[:, None] + (last - mean)[:, None] * np.exp(-horizons / 48.0)
    got = preds.sort_values(["origin_last_input_utc", "horizon_hours"])
    np.testing.assert_allclose(got["pred_kms"].to_numpy(), expected.ravel(),
                               rtol=1e-10)


def test_manifest_identity_fields_and_validation(tmp_path):
    ace = tmp_path / "ace.parquet"
    ace.write_bytes(b"fake-bytes")
    manifest_path = tmp_path / "m.manifest.json"
    manifest = write_manifest(
        manifest_path, mode="smoke-control", train_cutoff="2024-12-31T23:00:00Z",
        params={"n_estimators": 10}, tree_counts={"n_estimators": 10},
        columns=["a", "b"], input_identities={"ace": file_identity(ace)},
        seed=42, inference_seconds_per_origin=0.01,
        sampling={"step": "24h", "kept": 3})
    for field in ("train_cutoff", "params", "tree_counts", "columns",
                  "input_identities", "code_hash", "seed",
                  "inference_seconds_per_origin"):
        assert field in manifest, field
    assert manifest["input_identities"]["ace"]["sha256"] == file_identity(ace)["sha256"]
    assert validate_manifest_identities(manifest_path, {"ace": file_identity(ace)})
    ace.write_bytes(b"tampered")
    try:
        validate_manifest_identities(manifest_path, {"ace": file_identity(ace)})
    except ValueError:
        pass
    else:
        raise AssertionError("tampered input must fail validation")


def test_pooled_forecaster_covers_all_horizons_nan_safe():
    ace, ch = _synthetic(n=1600)
    ace.loc[700:710, "filled_speed_kms"] = np.nan  # raw gap survives load path
    ace["filled_speed_kms"] = ace["filled_speed_kms"].ffill()
    ch.iloc[700:710, :] = np.nan  # CH outage -> NaN native features
    features = build_features(ace, ch)
    assert features.isna().any().any()  # NaN stays native, not imputed away
    assert "ace_missing_72" in features.columns
    assert "ch_t035_missing24" in features.columns
    speed = as_speed_series(ace)
    cutoff = utc("2024-02-20T00:00:00Z")
    fore = PooledResidualForecaster(
        params={"n_estimators": 5, "max_depth": 2}, device="cpu")
    fore.fit(features, speed, cutoff, max_origins=60)
    origins = pd.date_range("2024-02-25", periods=3, freq="h", tz="UTC")
    preds = fore.predict(features, speed, origins)
    assert set(preds["horizon_hours"]) == set(range(1, HORIZON_HOURS + 1))
    assert np.isfinite(preds["pred_kms"]).all()
    assert len(preds) == 3 * HORIZON_HOURS


def test_build_features_compact_not_expanded():
    ace, ch = _synthetic()
    features = build_features(ace, ch)
    assert isinstance(features.index, pd.DatetimeIndex) and features.index.tz is not None
    assert len(features.columns) < 200  # compact, not the 5911-column expansion
    assert any(c.startswith("ace_") for c in features.columns)
    assert any(c.startswith("ch_t035_") for c in features.columns)
    assert "ch_agree_ratio_055_035" in features.columns


def test_ace_missing_and_age_from_genuine_was_missing():
    index = pd.date_range("2024-01-01", periods=800, freq="h", tz="UTC")
    missing = np.zeros(len(index), dtype=int)
    missing[750:753] = 1
    ace = pd.DataFrame({
        "timestamp_utc": index,
        "filled_speed_kms": np.full(len(index), 400.0),
        "was_missing": missing,
    })
    ch = pd.DataFrame({"ch_t035_area": np.full(len(index), 0.01)}, index=index)
    features = build_features(ace, ch)
    assert features.loc[index[750], "ace_age_hours"] == 1
    assert features.loc[index[751], "ace_age_hours"] == 2
    assert features.loc[index[752], "ace_age_hours"] == 3
    assert features.loc[index[753], "ace_age_hours"] == 0
    assert features.loc[index[756], "ace_missing_72"] == 3
    assert features.loc[index[753], "ace_is_missing"] == 0
    assert features.loc[index[750], "ace_is_missing"] == 1


def test_ace_missing_counts_invalid_speed_even_when_flag_clear():
    index = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    ace = pd.DataFrame({
        "timestamp_utc": index,
        "filled_speed_kms": np.full(len(index), 400.0),
        "was_missing": np.zeros(len(index), dtype=int),
    })
    ace.loc[100, "filled_speed_kms"] = np.nan
    ace.loc[101, "filled_speed_kms"] = -5.0
    ch = pd.DataFrame({"ch_t035_area": np.full(len(index), 0.01)}, index=index)
    features = build_features(ace, ch)
    # Both invalid hours count even though was_missing==0.
    assert features.loc[index[101], "ace_missing_72"] >= 2
    assert features.loc[index[101], "ace_age_hours"] >= 1


def test_ace_features_use_only_sources_leq_origin():
    index = pd.date_range("2024-01-01", periods=300, freq="h", tz="UTC")
    ace = pd.DataFrame({
        "timestamp_utc": index,
        "filled_speed_kms": np.full(len(index), 400.0),
        "was_missing": np.zeros(len(index), dtype=int),
    })
    ch = pd.DataFrame({"ch_t035_area": np.full(len(index), 0.01)}, index=index)
    t = index[200]
    future = ace.copy()
    future.loc[future["timestamp_utc"] > t, "was_missing"] = 1
    future.loc[future["timestamp_utc"] > t, "filled_speed_kms"] = 900.0
    base = build_features(ace, ch)
    altered = build_features(future, ch)
    pd.testing.assert_frame_equal(base.loc[:t].iloc[[-1]], altered.loc[:t].iloc[[-1]])


def test_observed_mask_preserves_was_missing_and_explicit_override():
    index = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    ace = pd.DataFrame({
        "timestamp_utc": index,
        "filled_speed_kms": np.full(len(index), 400.0),
        "was_missing": [0, 0, 1, 1, 0, 0, 1, 0],
    })
    speed = as_speed_series(ace)
    assert "observed" in speed.attrs
    observed = observed_mask(speed, index)
    assert observed.tolist() == [True, True, False, False, True, True, False, True]
    # Existing lane call shape (no explicit mask) already returns genuine flags.
    again = observed_mask(speed, speed.index)
    assert again.tolist() == observed.tolist()
    # Explicit Series override wins over preserved metadata.
    manual = pd.Series([True] * len(index), index=index)
    assert observed_mask(speed, index, observed=manual).all()
    # Explicit array + observed_index wins as well.
    arr = np.array([False] * len(index))
    assert not observed_mask(speed, index, observed=arr).any()
    # Bare Series without metadata falls back to finiteness.
    bare = pd.Series(np.full(len(index), 400.0), index=index)
    assert observed_mask(bare, index).all()
    bare_nan = pd.Series([400.0, np.nan] + [400.0] * 6, index=index)
    assert observed_mask(bare_nan, index).tolist() == [True, False] + [True] * 6


def test_observed_only_mse_differs_when_imputed_present():
    from experiments.ch_breakthrough_v2.common import score_forecast
    index = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    was_missing = np.array([0] * 100 + [1] * 20 + [0] * 80)
    # Genuinely observed hours sit at 400; imputed (filled) hours at 600 so
    # the observed-only error differs from the full-window error.
    filled = np.where(was_missing == 1, 600.0, 400.0)
    ace = pd.DataFrame({
        "timestamp_utc": index,
        "filled_speed_kms": filled,
        "was_missing": was_missing,
    })
    speed = as_speed_series(ace)
    observed = observed_mask(speed, speed.index)
    assert 0.0 < (1.0 - observed.mean()) < 1.0
    origins = index[110:112]
    preds = pd.DataFrame({
        "origin_last_input_utc": np.repeat(origins, 72),
        "horizon_hours": np.tile(np.arange(1, 73), len(origins)),
        "pred_kms": np.full(2 * 72, 900.0),
    })
    out = score_forecast(preds, speed, observed, origins=origins)
    assert out["imputed_fraction"] is not None and out["imputed_fraction"] > 0
    assert out["mse_observed"] != out["mse"]


def test_official_imputed_fraction_matches_pinned_truth():
    ace_path = Path("store/ch-v1/ace.parquet")
    if not ace_path.exists():
        return
    import pandas as pd
    ace_raw = pd.read_parquet(ace_path)
    from experiments.ch_breakthrough_v2.common import load_ace_hourly
    ace = load_ace_hourly([str(ace_path)])
    speed = as_speed_series(ace)
    observed = observed_mask(speed, speed.index)
    origins = pd.DatetimeIndex(pd.to_datetime(
        ["2026-05-31T23:00:00Z"] , utc=True)).append(
        pd.date_range("2026-06-01T00:00:00Z", "2026-09-26T23:00:00Z", freq="h", tz="UTC"))
    assert len(origins) == 2833
    times = origins.repeat(72) + pd.to_timedelta(
        np.tile(np.arange(1, 73), len(origins)), unit="h")
    times = pd.DatetimeIndex(times)
    frac = 1.0 - observed.reindex(times).fillna(False).mean()
    assert abs(float(frac) - 0.19829783896144645) < 1e-12


def test_capped_sampling_preserves_utc_and_reproducible():
    origins = pd.date_range("2024-01-01", periods=30 * 24, freq="h", tz="UTC")
    sampled, policy = subsample_origins(origins, step="daily", max_origins=12, seed=42)
    repeated, _ = subsample_origins(origins, step="daily", max_origins=12, seed=42)
    assert len(sampled) == 12
    assert str(sampled.tz) == "UTC"
    assert sampled.equals(repeated)
    assert sampled.is_unique and sampled.is_monotonic_increasing
    assert (sampled.hour == 0).all()
    assert policy["kept"] == 12
    other, _ = subsample_origins(origins, step="daily", max_origins=12, seed=7)
    assert len(other) == 12
    # Uncapped path returns the full daily grid.
    full, full_policy = subsample_origins(origins, step="daily", max_origins=None, seed=42)
    assert len(full) == 30 and full_policy["kept"] == 30


def test_predict_ignores_future_truth_perturbation_and_absence():
    ace, ch = _synthetic(n=1600)
    features = build_features(ace, ch)
    speed = as_speed_series(ace)
    cutoff = utc("2024-02-20T00:00:00Z")
    fore = PooledResidualForecaster(
        params={"n_estimators": 5, "max_depth": 2}, device="cpu")
    fore.fit(features, speed, cutoff, max_origins=60)
    origins = pd.date_range("2024-02-25", periods=3, freq="h", tz="UTC")
    base_preds = fore.predict(features, speed, origins)
    # Perturb only post-origin truth; predictions must not move.
    perturbed = speed.copy()
    future_times = perturbed.index[perturbed.index > origins.max()]
    perturbed.loc[future_times] = perturbed.loc[future_times] * 5.0 + 200.0
    pert_preds = fore.predict(features, perturbed, origins)
    pd.testing.assert_frame_equal(base_preds, pert_preds)
    # Absence of post-origin truth: truncate speed at the last origin.
    truncated = speed.loc[:origins.max()].copy()
    truncated.attrs["observed"] = speed.attrs["observed"].loc[:origins.max()]
    truncated.attrs["observed_index"] = truncated.attrs["observed"].index[
        truncated.attrs["observed"].to_numpy(dtype=bool)]
    trunc_preds = fore.predict(features, truncated, origins)
    pd.testing.assert_frame_equal(base_preds, trunc_preds)


def test_manifest_provenance_and_stale_code_invalidates(tmp_path):
    ace = tmp_path / "ace.parquet"
    ace.write_bytes(b"fake-bytes")
    pred = tmp_path / "pred.parquet"
    pred.write_bytes(b"fake-preds")
    manifest_path = tmp_path / "m.manifest.json"
    manifest = write_manifest(
        manifest_path, mode="smoke-control", train_cutoff="2024-12-31T23:00:00Z",
        params={"n_estimators": 10}, tree_counts={"n_estimators": 10},
        columns=["a", "b"], input_identities={"ace": file_identity(ace)},
        seed=42, inference_seconds_per_origin=0.01,
        sampling={"step": "24h", "kept": 3},
        caller_file=__file__, actual_device="cpu",
        prediction_path=pred)
    assert "code_hashes" in manifest and "common" in manifest["code_hashes"]
    assert "caller" in manifest["code_hashes"]
    assert "versions" in manifest and "pandas" in manifest["versions"]
    assert manifest["actual_device"] == "cpu"
    assert manifest["prediction"]["sha256"] == file_identity(pred)["sha256"]
    assert manifest["code_hash"] == code_hash()
    assert code_hashes()["common"]["sha256"] == code_hash()
    assert "pandas" in runtime_versions()
    assert validate_manifest_identities(manifest_path, {"ace": file_identity(ace)})
    try:
        validate_manifest_identities(
            manifest_path, {"ace": file_identity(ace)},
            current_code_hash="0" * 64)
    except ValueError:
        pass
    else:
        raise AssertionError("stale code hash must invalidate the cache")
    tampered = dict(manifest)
    tampered["code_hash"] = "0" * 64
    try:
        validate_manifest_identities(tampered, {"ace": file_identity(ace)})
    except ValueError:
        pass
    else:
        raise AssertionError("tampered code hash must fail validation")


def test_forecaster_records_actual_device_and_cutoff():
    ace, ch = _synthetic(n=1600)
    features = build_features(ace, ch)
    speed = as_speed_series(ace)
    cutoff = utc("2024-02-20T00:00:00Z")
    fore = PooledResidualForecaster(
        params={"n_estimators": 5, "max_depth": 2}, device="cpu")
    fore.fit(features, speed, cutoff, max_origins=60)
    assert fore.cutoff == cutoff
    assert fore.actual_device is not None and "cpu" in str(fore.actual_device).lower()
    assert list(fore.columns[-2:]) == ["horizon_hours", "horizon_scaled"]
