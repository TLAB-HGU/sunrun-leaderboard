import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("suvi_features", Path(__file__).parents[1] /
    "experiments/xgboost_suvi_fusion/features.py")
F = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(F)


def test_disk_quality_mask_and_orientation():
    rad = np.arange(121, dtype=float).reshape(11, 11)
    quality = np.zeros_like(rad)
    header = {"CRPIX1": 6, "CRPIX2": 6, "RSUN": 5, "CDELT1": 1, "CDELT2": 1}
    base = F.extract_frame_features(rad, quality, header)
    # Actual SUVI RSUN already describes a radius in pixels, despite CDELT=2.5.
    assert F.extract_frame_features(rad, quality, dict(header, CDELT1=2.5, CDELT2=2.5)) == base
    rad[0, 0] = 1e20
    assert F.extract_frame_features(rad, quality, header) == base
    quality[5, 5] = 1
    masked = F.extract_frame_features(rad, quality, header)
    rad[5, 5] = 1e20
    assert F.extract_frame_features(rad, quality, header) == masked
    rotated = F.extract_frame_features(np.rot90(rad, 2), np.rot90(quality, 2),
        dict(header, PC1_1=-1, PC1_2=0, PC2_1=0, PC2_2=-1))
    np.testing.assert_allclose(list(masked.values()), list(rotated.values()), equal_nan=True)


def fixtures():
    times = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
    ace = pd.DataFrame({"timestamp_utc": times, "filled_speed_kms": np.arange(400)+300.})
    images = pd.DataFrame({"brightness": np.arange(400, dtype=float)}, index=times)
    return times, ace, images


def test_future_perturbation_and_training_medians():
    times, ace, images = fixtures()
    median = F.fit_image_medians(images, times[150])
    expected = F.build_origin_features(ace, images, times[[200, 201]], median)
    images.loc[times[202]:] = 1e10
    ace.loc[202:, "filled_speed_kms"] = -1e10
    pd.testing.assert_series_equal(median, F.fit_image_medians(images, times[150]))
    pd.testing.assert_frame_equal(expected, F.build_origin_features(ace, images, times[[200, 201]], median))
    assert expected.iloc[0].ace_lag_119 == 381
    assert expected.iloc[0].ace_lag_0 == 500


def test_fill_is_limited_and_never_reaches_before_input_window():
    times, ace, images = fixtures()
    median = pd.Series({"brightness": -5.})
    images.loc[times[81]:times[87]] = np.nan
    features = F.build_origin_features(ace, images, times[[200]], median)
    assert features.iloc[0].suvi_brightness_lag_119 == -5
    images.loc[times[195]:times[200]] = np.nan
    features = F.build_origin_features(ace, images, times[[200]], median)
    assert features.iloc[0].suvi_brightness_lag_0 == 194
    images.loc[times[194]] = np.nan
    features = F.build_origin_features(ace, images, times[[200]], median)
    assert features.iloc[0].suvi_brightness_lag_0 == -5
    assert features.iloc[0].suvi_brightness_age_hours == 7


def test_cache_rejects_future_observation_before_loading(tmp_path):
    metadata = pd.DataFrame([{"slot": "2025-01-01T00:00Z", "obs_end": "2025-01-01T01:00Z",
        "filename": "fake.nc", "status": "ok"}])
    with pytest.raises(ValueError, match="after"):
        F.build_hourly_cache(metadata, tmp_path)


def test_cache_treats_known_upstream_nc_absence_as_missing(tmp_path):
    metadata = pd.DataFrame([{
        "slot": "2025-01-01T00:00Z",
        "obs_end": "2024-12-31T23:59Z",
        "filename": "missing.fits",
        "nc_path": None,
        "status": "ok",
    }])
    result = F.build_hourly_cache(metadata, tmp_path)
    assert result.isna().all(axis=None)


def test_cache_treats_known_invalid_geometry_as_missing(tmp_path):
    metadata = pd.DataFrame([{
        "slot": "2025-01-01T00:00Z",
        "obs_end": "2024-12-31T23:59Z",
        "filename": "corrupt.fits",
        "nc_path": "does-not-need-to-exist.nc",
        "status": "ok",
        "rsun": -999,
    }])
    result = F.build_hourly_cache(metadata, tmp_path)
    assert result.isna().all(axis=None)


def test_cache_is_deterministic_and_reused(tmp_path):
    frame = tmp_path / "fake.nc"
    frame.touch()
    metadata = pd.DataFrame([{"slot": "2025-01-01T00:00Z", "obs_end": "2024-12-31T23:59Z",
        "filename": "fake.nc", "nc_path": str(frame), "status": "ok"}])
    calls = []
    def loader(path):
        calls.append(path)
        return np.ones((11, 11)), np.zeros((11, 11)), {"CRPIX1": 6, "CRPIX2": 6, "RSUN": 5, "CDELT1": 1, "CDELT2": 1}
    cache = tmp_path / "features.parquet"
    first = F.build_hourly_cache(metadata, tmp_path, cache, loader)
    manifest = cache.with_suffix(".parquet.json").read_bytes()
    second = F.build_hourly_cache(metadata, tmp_path, cache, loader)
    pd.testing.assert_frame_equal(first, second)
    assert len(calls) == 1
    assert cache.with_suffix(".parquet.json").read_bytes() == manifest


def test_cache_resolves_staged_nc_path_under_image_root(tmp_path):
    image = tmp_path / "data/g18/2025/001/frame.nc"
    image.parent.mkdir(parents=True)
    image.touch()
    metadata = pd.DataFrame([{
        "slot": "2025-01-01T00:00Z",
        "obs_end": "2024-12-31T23:59Z",
        "filename": "frame.fits",
        "nc_path": "data/g18/2025/001/frame.nc",
        "status": "ok",
    }])

    def loader(path):
        assert path == image
        return np.ones((3, 3)), np.zeros((3, 3)), {"CRPIX1": 2, "CRPIX2": 2, "RSUN": 3}

    assert len(F.build_hourly_cache(metadata, tmp_path, frame_loader=loader)) == 1


def test_missing_ace_history_rejected():
    times, ace, images = fixtures()
    with pytest.raises(ValueError, match="120 finite"):
        F.build_origin_features(ace, images, times[[100]], pd.Series({"brightness": 1.}))


def test_netcdf_physical_decode(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "packed.nc"
    with h5py.File(path, "w") as nc:
        rad = nc.create_dataset("RAD", data=np.array([[2, -99], [4, 8]], dtype=np.int16))
        rad.attrs["scale_factor"] = .5
        rad.attrs["add_offset"] = 3.
        rad.attrs["_FillValue"] = -99
        nc.create_dataset("DQF", data=np.zeros((2, 2)))
        nc.create_dataset("RSUN", data=960.)
    rad, _dqf, header = F.read_suvi_frame(path)
    np.testing.assert_allclose(rad, [[4., np.nan], [5., 7.]], equal_nan=True)
    assert header["RSUN"] == 960
