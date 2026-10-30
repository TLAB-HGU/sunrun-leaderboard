import numpy as np
import pandas as pd
import pytest

from experiments.xgboost_suvi_fusion.coronal_holes import (
    build_ch_hourly_cache,
    build_ch_origin_features,
    extract_ch_frame,
    geometry,
)


def image():
    rad = np.ones((101, 101))
    header = {"CRPIX1": 51, "CRPIX2": 51, "RSUN": 45}
    return rad, np.zeros_like(rad), header


def test_explicit_area_position_and_thresholds():
    rad, dqf, h = image()
    rad[46:55, 66:75] = .4
    result = extract_ch_frame(rad, dqf, h)
    assert result["ch_t035_area"] == 0
    assert result["ch_t045_count"] == 1
    _, lat, lon, area = geometry(rad.shape, h)
    mask = rad == .4
    assert result["ch_t045_area"] == pytest.approx(area[mask].sum())
    assert result["ch_t045_top1_lon"] == pytest.approx(np.average(lon[mask], weights=area[mask]))
    assert result["ch_t045_top1_lat"] == pytest.approx(np.average(lat[mask], weights=area[mask]))
    assert result["ch_t045_top1_lon"] > 20
    assert sum(v for k, v in result.items() if k.startswith("ch_t045_lon")) == pytest.approx(result["ch_t045_area"])


def test_orientation_quality_and_disk_exclusion():
    rad, dqf, h = image()
    rad[45:55, 65:75] = .1
    base = extract_ch_frame(rad, dqf, h)
    rotated = extract_ch_frame(np.rot90(rad, 2), np.rot90(dqf, 2),
                              dict(h, PC1_1=-1, PC1_2=0, PC2_1=0, PC2_2=-1))
    np.testing.assert_allclose(list(base.values()), list(rotated.values()), atol=1e-12)
    rad[:8] = 0
    assert extract_ch_frame(rad, dqf, h) == base
    dqf[45:55, 65:75] = 1
    assert extract_ch_frame(rad, dqf, h)["ch_t045_area"] == 0


def test_no_hole_is_zero_but_missing_frame_is_nan():
    rad, dqf, h = image()
    result = extract_ch_frame(rad, dqf, h)
    assert result["ch_t045_count"] == 0
    assert result["ch_t045_top1_lat"] == 0
    missing = extract_ch_frame(np.full(rad.shape, np.nan), dqf, h)
    assert all(np.isnan(v) for v in missing.values())


def test_ch_causality_window_boundary_and_no_ace():
    times = pd.date_range("2025-01-01", periods=300, freq="h", tz="UTC")
    hourly = pd.DataFrame({"ch_t045_area": np.arange(300.)}, index=times)
    hourly.loc[times[81]:times[87]] = np.nan
    base = build_ch_origin_features(hourly, times[[200]], times[150])
    assert base.iloc[0].ch_t045_area_lag119 == hourly.loc[:times[150]].median().iloc[0]
    assert not any(c.startswith("ace_") for c in base)
    hourly.loc[times[201]:] = 1e10
    pd.testing.assert_frame_equal(base, build_ch_origin_features(hourly, times[[200]], times[150]))
    hourly.loc[times[194]:times[200]] = np.nan
    result = build_ch_origin_features(hourly, times[[200]], times[150])
    assert result.iloc[0].ch_t045_area_lag0 == hourly.loc[:times[150]].median().iloc[0]
    assert result.iloc[0].ch_t045_area_age == 7


def test_ch_missing_cache_has_stable_schema(tmp_path):
    meta = pd.DataFrame([{"slot": "2025-01-01T00:00Z", "obs_end": "2024-12-31T23:59Z",
                          "filename": "absent.nc", "nc_path": None, "status": "ok"}])
    frame = build_ch_hourly_cache(meta, tmp_path, tmp_path/"ch.parquet", workers=1)
    assert frame.isna().all(axis=None)
    assert "ch_t045_top1_lon" in frame


def test_partial_cache_resumes_only_unfinished_frames(tmp_path):
    import json

    from experiments.xgboost_suvi_fusion.features import build_hourly_cache
    paths = [tmp_path/"one.nc", tmp_path/"two.nc"]
    for path in paths:
        path.touch()
    meta = pd.DataFrame([{"slot": f"2025-01-01T0{i}:00Z", "obs_end": "2024-12-31T23:59Z",
                          "filename": path.name, "nc_path": str(path), "status": "ok"}
                         for i, path in enumerate(paths)])
    calls = []
    def loader(path):
        calls.append(path)
        return image()
    cache = tmp_path/"ch.parquet"
    def build():
        return build_hourly_cache(meta, tmp_path, cache, loader,
                                 extractor=extract_ch_frame, extractor_version="test-ch")
    full = build()
    manifest = json.loads(cache.with_suffix(".parquet.json").read_text())
    full.iloc[:1].to_parquet(cache.with_suffix(".parquet.partial"))
    cache.with_suffix(".parquet.partial.json").write_text(json.dumps(manifest))
    cache.unlink()  # test fixture only; force the actual partial-cache branch
    calls.clear()
    resumed = build()
    assert calls == [paths[1]]
    pd.testing.assert_frame_equal(full, resumed)
