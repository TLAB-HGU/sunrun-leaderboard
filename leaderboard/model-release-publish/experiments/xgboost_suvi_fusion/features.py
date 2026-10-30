"""Deterministic SUVI disk summaries and strictly trailing 120-hour features.

RAD supplied to ``extract_frame_features`` is already physical radiance. The
NetCDF reader explicitly decodes packing once; image interpolation is never used.
"""
from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW = 120
CACHE_VERSION = 2


def read_suvi_frame(path):
    """Read GOES SUVI NetCDF using lazy h5py (no automatic scale decoding)."""
    import h5py

    with h5py.File(path, "r") as nc:
        var = nc["RAD"]
        raw = np.asarray(var[...])
        rad = raw.astype(np.float64)
        invalid = ~np.isfinite(rad)
        for name in ("_FillValue", "missing_value"):
            if name in var.attrs:
                invalid |= raw == np.asarray(var.attrs[name]).reshape(-1)[0]
        rad = rad * float(np.asarray(var.attrs.get("scale_factor", 1)).item())
        rad += float(np.asarray(var.attrs.get("add_offset", 0)).item())
        rad[invalid] = np.nan
        header = dict(nc.attrs)
        for key in ("CRPIX1", "CRPIX2", "RSUN", "CROTA", "SOLAR_B0", "CDELT1", "CDELT2",
                    "PC1_1", "PC1_2", "PC2_1", "PC2_2"):
            if key in nc:
                header[key] = np.asarray(nc[key][...]).item()
        return rad.squeeze(), np.asarray(nc["DQF"][...]).squeeze(), header


def extract_frame_features(rad, dqf, header):
    """Summarize physical log1p radiance in the disk and heliographic cells.

    SUVI L1b RSUN is in pixels (not arcseconds). FITS CRPIX is 1-based.
    Longitude/latitude are observer-relative with optional solar B0 correction.
    """
    rad, dqf = np.asarray(rad, dtype=float), np.asarray(dqf)
    if rad.ndim != 2 or rad.shape != dqf.shape:
        raise ValueError("RAD and DQF must have the same 2-D shape")
    h = {str(k).upper(): v for k, v in header.items()}
    def number(key, default=None):
        value = h.get(key, default)
        if value is None:
            raise ValueError(f"missing SUVI geometry: {key}")
        return float(np.asarray(value).item())
    yy, xx = np.indices(rad.shape, dtype=float)
    xx -= number("CRPIX1") - 1
    yy -= number("CRPIX2") - 1
    theta = np.deg2rad(number("CROTA", 0))
    matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    if all(k in h for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2")):
        matrix = np.array([[number("PC1_1"), number("PC1_2")],
                           [number("PC2_1"), number("PC2_2")]])
    rsun = number("RSUN")
    if not np.isfinite(rsun) or rsun <= 0:
        raise ValueError("RSUN must be positive")
    x = (matrix[0, 0] * xx + matrix[0, 1] * yy) / rsun
    y = (matrix[1, 0] * xx + matrix[1, 1] * yy) / rsun
    disk = x*x + y*y <= 0.98**2
    valid = disk & (dqf == 0) & np.isfinite(rad) & (rad >= 0)
    z = np.sqrt(np.maximum(0, 1 - x*x - y*y))
    b0 = np.deg2rad(number("SOLAR_B0", 0))
    lat = np.arcsin(np.clip(y*np.cos(b0) + z*np.sin(b0), -1, 1))
    lon = np.arctan2(x, z*np.cos(b0) - y*np.sin(b0))
    longitude = np.clip(np.floor((lon + np.pi/2) / np.pi * 12), 0, 11).astype(int)
    latitude = np.clip(np.floor((lat + np.pi/2) / np.pi * 3), 0, 2).astype(int)
    reference = np.median(rad[valid]) if valid.any() else np.nan
    result = {}
    for name, region in [("disk", disk)] + [
        (f"lon{i:02d}_lat{j}", disk & (longitude == i) & (latitude == j))
        for i in range(12) for j in range(3)
    ]:
        values = rad[valid & region]
        log = np.log1p(values)
        result[f"{name}_log_median"] = float(np.median(log)) if len(log) else np.nan
        result[f"{name}_log_q10"] = float(np.quantile(log, .1)) if len(log) else np.nan
        result[f"{name}_dark_fraction"] = float(np.mean(values < .5*reference)) if len(log) else np.nan
        result[f"{name}_valid_fraction"] = float(len(values) / region.sum()) if region.any() else 0.
    return result


def _hourly(frame):
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    if frame.index.has_duplicates or not frame.index.equals(frame.index.floor("h")):
        raise ValueError("timestamps must be unique hourly boundaries")
    return frame.sort_index()


def _extract_path(path, extractor=extract_frame_features):
    if not path:
        return {}
    try:
        return extractor(*read_suvi_frame(path))
    except Exception as exc:
        raise RuntimeError(f"failed to extract SUVI frame {path}: {exc}") from exc


def build_hourly_cache(metadata, images_root, cache_path=None, frame_loader=read_suvi_frame,
                       workers=1, *, extractor=extract_frame_features, extractor_version=None):
    """Extract one row per metadata slot and preserve known upstream gaps.

    Cache identity includes metadata and file sizes/mtimes. Corrupt frames and
    future observations raise rather than silently converting to missing data.
    A selected observation with a null ``nc_path`` is an explicit upstream gap;
    a non-null path missing locally remains an integrity error.
    """
    rows = metadata.copy()
    rows["slot"] = pd.to_datetime(rows["slot"], utc=True)
    rows = rows.sort_values("slot")
    _hourly(rows.set_index("slot"))
    paths = []
    root = Path(images_root)
    for row in rows.to_dict("records"):
        if row.get("status", "ok") != "ok":
            paths.append(None)
            continue
        rsun = row.get("rsun")
        if rsun is not None and (not np.isfinite(rsun) or rsun <= 0):
            paths.append(None)
            continue
        obs = pd.Timestamp(row["obs_end"])
        obs = obs.tz_localize("UTC") if obs.tzinfo is None else obs.tz_convert("UTC")
        if obs > row["slot"]:
            raise ValueError("SUVI observation is after its feature slot")
        filename = str(row["filename"]).replace(".fits", ".nc")
        candidates = [root / str(row.get("sat", "g18")).lower() / obs.strftime("%Y/%j") / filename,
                      root / obs.strftime("%Y/%j") / filename]
        if pd.notna(row.get("nc_path")):
            staged = Path(row["nc_path"])
            candidates.append(staged if staged.is_absolute() else root / staged)
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            # A null nc_path records a selected FITS observation whose NetCDF twin
            # was unavailable upstream. It is a genuine missing image slot. A
            # non-null path disappearing locally is an integrity error.
            if pd.isna(row.get("nc_path")):
                paths.append(None)
                continue
            raise FileNotFoundError(f"Missing SUVI frame for {row['slot']}: {filename}")
        paths.append(path)
    identity = {"version": CACHE_VERSION, "metadata": rows.to_json(date_format="iso", orient="split"),
                "files": [(str(p), p.stat().st_size, p.stat().st_mtime_ns) if p else None for p in paths]}
    if extractor_version is not None:
        identity["extractor_version"] = extractor_version
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_path) if cache_path else None
    manifest = cache.with_suffix(cache.suffix + ".json") if cache else None
    if cache and cache.exists() and manifest.exists() and json.loads(manifest.read_text())["source_sha256"] == digest:
        return _hourly(pd.read_parquet(cache))
    if workers < 1:
        raise ValueError("workers must be positive")
    columns = list(extractor(np.ones((3, 3)), np.zeros((3, 3)),
                   {"CRPIX1": 2, "CRPIX2": 2, "RSUN": 3}))
    partial_cache = cache.with_suffix(cache.suffix+".partial") if cache and extractor_version else None
    partial_manifest = cache.with_suffix(cache.suffix+".partial.json") if partial_cache else None
    records = []
    if partial_cache and partial_cache.exists() and partial_manifest.exists():
        info = json.loads(partial_manifest.read_text())
        if info.get("source_sha256") == digest:
            saved = pd.read_parquet(partial_cache)
            if list(saved.columns) != columns or not saved.index.equals(pd.DatetimeIndex(rows.slot[:len(saved)], name="slot")):
                raise ValueError("partial cache schema/timestamps do not match")
            records = saved.to_dict("records")
            print(f"SUVI frames: resumed {len(records)}/{len(paths)}", file=sys.stderr, flush=True)
    if workers > 1 and frame_loader is read_suvi_frame:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            offset = len(records)
            for completed, record in enumerate(executor.map(partial(_extract_path, extractor=extractor), paths[offset:], chunksize=8), offset+1):
                records.append(record)
                if completed % 1000 == 0 or completed == len(paths):
                    print(f"SUVI frames: {completed}/{len(paths)}", file=sys.stderr, flush=True)
                    if partial_cache:
                        saved = pd.DataFrame(records, index=pd.DatetimeIndex(rows.slot[:completed], name="slot"), columns=columns)
                        temp = partial_cache.with_suffix(".tmp")
                        saved.to_parquet(temp)
                        temp.replace(partial_cache)
                        partial_manifest.write_text(json.dumps({"source_sha256": digest, "rows": completed}))
    else:
        records += [extractor(*frame_loader(p)) if p else {} for p in paths[len(records):]]
    # Stable schema even if the supplied metadata contains only missing slots.
    result = pd.DataFrame(records, index=pd.DatetimeIndex(rows.slot, name="slot"), columns=columns)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache)
        manifest.write_text(json.dumps({"source_sha256": digest, "version": CACHE_VERSION}, sort_keys=True))
    return result


def fit_image_medians(hourly, cutoff):
    """Learn imputation values exclusively from the specified training prefix."""
    hourly = _hourly(hourly)
    cutoff = pd.to_datetime(cutoff, utc=True)
    if hourly.loc[:cutoff].empty:
        raise ValueError("no image slots at or before training cutoff")
    medians = hourly.loc[:cutoff].median().fillna(0.)
    medians.attrs["training_cutoff"] = cutoff.isoformat()
    return medians


def build_origin_features(ace, hourly, origins, image_medians):
    """Build float32 features from exactly origin-119h through origin inclusive.

    Image fill runs separately inside each window, so even a lagged imputed
    value never reaches outside its origin's five-day input window.
    """
    ace = _hourly(ace.set_index("timestamp_utc") if "timestamp_utc" in ace else ace)
    hourly = _hourly(hourly)
    origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True), name="origin_last_input_utc")
    if len(origins) == 0:
        raise ValueError("origins must not be empty")
    if not origins.equals(origins.floor("h")):
        raise ValueError("origins must be hourly boundaries")
    grid = pd.date_range(origins.min()-pd.Timedelta(hours=119), origins.max(), freq="h")
    positions = grid.get_indexer(origins) - 119
    def windows(values):
        return np.lib.stride_tricks.sliding_window_view(values, WINDOW)[positions]
    speed = windows(ace["filled_speed_kms"].reindex(grid).to_numpy(dtype=float))
    if not np.isfinite(speed).all():
        raise ValueError("ACE needs 120 finite hourly speed values at every origin")
    output = {f"ace_lag_{lag}": speed[:, -1-lag] for lag in range(120)}
    for lag in (1, 6, 12, 24, 48, 72, 119):
        output[f"ace_diff_{lag}"] = speed[:, -1] - speed[:, -1-lag]
    def summaries(prefix, values, widths):
        for width in widths:
            v = values[:, -width:]
            t = np.arange(width) - (width-1)/2
            for name, data in (("mean", v.mean(1)), ("std", v.std(1)), ("min", v.min(1)),
                               ("slope", (v @ t)/(t @ t))):
                output[f"{prefix}_{name}_{width}"] = data
    summaries("ace", speed, (6, 12, 24, 48, 72, 120))
    if set(hourly.columns) != set(image_medians.index):
        raise ValueError("image medians must match hourly feature columns")
    time = np.arange(120)[None, :]
    for column in hourly.columns:
        values = windows(hourly[column].reindex(grid).to_numpy(dtype=float))
        good = np.isfinite(values)
        last = np.maximum.accumulate(np.where(good, time, -1), axis=1)
        gathered = np.take_along_axis(values, np.maximum(last, 0), axis=1)
        filled = np.where((last >= 0) & (time-last <= 6), gathered, float(image_medians[column]))
        prefix = f"suvi_{column}"
        for lag in (0, 24, 48, 72, 96, 119):
            output[f"{prefix}_lag_{lag}"] = filled[:, -1-lag]
        output[f"{prefix}_missing_120"] = (~good).sum(1)
        output[f"{prefix}_age_hours"] = 119-last[:, -1]
        summaries(prefix, filled, (24, 120))
    return pd.DataFrame(output, index=origins, dtype=np.float32)
