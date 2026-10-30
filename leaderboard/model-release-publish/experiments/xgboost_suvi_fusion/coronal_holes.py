"""Causal, explicitly measured coronal-hole candidate features from SUVI Fe195."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from .features import _hourly, build_hourly_cache, fit_image_medians

THRESHOLDS = (0.35, 0.45, 0.55)
VERSION = "ch-v1-t035-045-055-opening-closing3-minarea0005"
LAGS = (0, 6, 12, 24, 48, 72, 96, 119)


def prefix(threshold):
    return f"ch_t{round(threshold * 100):03d}_"


def geometry(shape, header):
    h = {str(k).upper(): float(np.asarray(v).item()) for k, v in header.items()
         if str(k).upper() in {"CRPIX1", "CRPIX2", "RSUN", "CROTA", "SOLAR_B0",
                              "PC1_1", "PC1_2", "PC2_1", "PC2_2"}}
    rsun = h["RSUN"]
    if not np.isfinite(rsun) or rsun <= 0:
        raise ValueError("RSUN must be positive")
    y, x = np.indices(shape, dtype=float)
    x -= h["CRPIX1"] - 1
    y -= h["CRPIX2"] - 1
    theta = np.deg2rad(h.get("CROTA", 0))
    matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    if all(k in h for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2")):
        matrix = np.array([[h["PC1_1"], h["PC1_2"]], [h["PC2_1"], h["PC2_2"]]])
    xx = (matrix[0, 0]*x + matrix[0, 1]*y) / rsun
    yy = (matrix[1, 0]*x + matrix[1, 1]*y) / rsun
    disk = xx**2 + yy**2 <= .98**2
    z = np.sqrt(np.maximum(1e-8, 1-xx**2-yy**2))
    b0 = np.deg2rad(h.get("SOLAR_B0", 0))
    lat = np.rad2deg(np.arcsin(np.clip(yy*np.cos(b0)+z*np.sin(b0), -1, 1)))
    lon = np.rad2deg(np.arctan2(xx, z*np.cos(b0)-yy*np.sin(b0)))
    area = np.where(disk, 1 / (z*2*np.pi*rsun**2), 0)
    return disk, lat, lon, area


def candidate_mask(rad, dqf, header, threshold):
    return _candidate_from_geometry(rad, dqf, geometry(rad.shape, header), threshold)


def _candidate_from_geometry(rad, dqf, coordinates, threshold):
    disk, lat, lon, area = coordinates
    valid = disk & (dqf == 0) & np.isfinite(rad) & (rad >= 0)
    reference = np.median(rad[valid]) if valid.any() else np.nan
    dark = valid & (rad < threshold*reference)
    structure = np.ones((3, 3), dtype=bool)
    dark = ndimage.binary_closing(ndimage.binary_opening(dark, structure), structure) & valid
    labels, count = ndimage.label(dark, structure)
    totals = np.bincount(labels.ravel(), weights=area.ravel(), minlength=count+1)
    keep = totals >= .0005
    keep[0] = False
    mask = keep[labels]
    labels, count = ndimage.label(mask, structure)
    return mask, labels, count, valid, lat, lon, area


def extract_ch_frame(rad, dqf, header):
    rad, dqf = np.asarray(rad, dtype=float), np.asarray(dqf)
    if rad.ndim != 2 or rad.shape != dqf.shape:
        raise ValueError("RAD and DQF must have the same 2D shape")
    # SUVI frames include a large unused off-disk border. Crop a conservative
    # bounding square without resampling, and retain exactly the original WCS.
    # The least matrix singular value also handles non-rotation PC transforms.
    h = dict(header)
    rsun = float(np.asarray(h["RSUN"]).item())
    if all(key in h for key in ("PC1_1", "PC1_2", "PC2_1", "PC2_2")):
        matrix = np.array([[float(h["PC1_1"]), float(h["PC1_2"])],
                           [float(h["PC2_1"]), float(h["PC2_2"])]])
        scale = np.linalg.svd(matrix, compute_uv=False).min()
        if scale <= 0:
            raise ValueError("singular SUVI PC matrix")
    else:
        scale = 1.
    radius = rsun/scale+2
    cx, cy = float(h["CRPIX1"])-1, float(h["CRPIX2"])-1
    x0, x1 = max(0, int(np.floor(cx-radius))), min(rad.shape[1], int(np.ceil(cx+radius))+1)
    y0, y1 = max(0, int(np.floor(cy-radius))), min(rad.shape[0], int(np.ceil(cy+radius))+1)
    rad, dqf = rad[y0:y1, x0:x1], dqf[y0:y1, x0:x1]
    h["CRPIX1"], h["CRPIX2"] = cx+1-x0, cy+1-y0
    coordinates = geometry(rad.shape, h)
    out = {}
    for threshold in THRESHOLDS:
        mask, labels, count, valid, lat, lon, area = _candidate_from_geometry(rad, dqf, coordinates, threshold)
        result = {"area": float(area[mask].sum()), "count": float(count),
                  "valid_fraction": float(valid.sum() / max(1, (area > 0).sum()))}
        objects = []
        # Slices bound component work; avoid scanning the whole image for each object.
        for label, bounds in enumerate(ndimage.find_objects(labels), 1):
            if bounds is None:
                continue
            inside = labels[bounds] == label
            weights = area[bounds][inside]
            la, lo = lat[bounds][inside], lon[bounds][inside]
            objects.append((float(weights.sum()), float(np.average(la, weights=weights)),
                            float(np.average(lo, weights=weights)), float(np.ptp(lo)), float(np.ptp(la))))
        objects.sort(key=lambda row: (-row[0], row[1], row[2]))
        result["max_area"] = objects[0][0] if objects else 0.
        for i in range(3):
            values = objects[i] if i < len(objects) else (0.,)*5
            result.update({f"top{i+1}_{name}": value for name, value in
                           zip(("area", "lat", "lon", "lon_width", "lat_width"), values)})
        for width in (15, 30):
            result[f"meridian{width}_area"] = float(area[mask & (np.abs(lon) <= width)].sum())
            result[f"meridian{width}_equatorial_area"] = float(
                area[mask & (np.abs(lon) <= width) & (np.abs(lat) <= 30)].sum())
        result["equatorial_area"] = float(area[mask & (np.abs(lat) <= 30)].sum())
        longitude = np.clip(((lon+90)/15).astype(int), 0, 11)
        latitude = np.clip(((lat+90)/60).astype(int), 0, 2)
        totals = np.bincount((longitude*3+latitude)[mask], weights=area[mask], minlength=36)
        result.update({f"lon{i:02d}_lat{j}_area": float(totals[i*3+j])
                       for i in range(12) for j in range(3)})
        if not valid.any():
            result = {key: np.nan for key in result}
        out.update({prefix(threshold)+key: value for key, value in result.items()})
    return out


def build_ch_hourly_cache(metadata, images_root, cache_path, workers=12):
    return build_hourly_cache(metadata, images_root, cache_path, workers=workers,
                              extractor=extract_ch_frame, extractor_version=VERSION)


def build_ch_origin_features(hourly, origins, cutoff):
    hourly = _hourly(hourly)
    medians = fit_image_medians(hourly, cutoff)
    origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True), name="origin_last_input_utc")
    if origins.empty or not origins.equals(origins.floor("h")):
        raise ValueError("origins must be nonempty hourly timestamps")
    grid = pd.date_range(origins.min()-pd.Timedelta(hours=119), origins.max(), freq="h")
    positions = grid.get_indexer(origins)-119
    time = np.arange(120)[None, :]
    output = {}
    for col in hourly:
        values = np.lib.stride_tricks.sliding_window_view(
            hourly[col].reindex(grid).to_numpy(dtype=float), 120)[positions]
        good = np.isfinite(values)
        last = np.maximum.accumulate(np.where(good, time, -1), axis=1)
        gathered = np.take_along_axis(values, np.maximum(last, 0), axis=1)
        filled = np.where((last >= 0) & (time-last <= 6), gathered, medians[col])
        for lag in LAGS:
            output[f"{col}_lag{lag}"] = filled[:, -1-lag]
            if lag:
                output[f"{col}_change{lag}"] = filled[:, -1] - filled[:, -1-lag]
        output[f"{col}_missing120"] = (~good).sum(1)
        output[f"{col}_age"] = 119-last[:, -1]
        for width in (24, 72, 120):
            v = filled[:, -width:]
            t = np.arange(width)-(width-1)/2
            for name, data in (("mean", v.mean(1)), ("std", v.std(1)), ("min", v.min(1)),
                               ("max", v.max(1)), ("slope", (v@t)/(t@t))):
                output[f"{col}_{name}{width}"] = data
    return pd.DataFrame(output, index=origins, dtype=np.float32)
