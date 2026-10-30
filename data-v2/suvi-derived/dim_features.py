"""Causal low-order coronal dimming summaries from GOES-18 SUVI Fe195."""
from __future__ import annotations

import io
import hashlib
import json
import os
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage

CONFIG = {
    "threshold_ratio": 0.65,
    "lookback_hours": 18,
    "reference_lag_hours": 24,
    "reference_tolerance_minutes": 90,
    "minimum_reference_frames": 3,
    "minimum_consecutive_dimming_frames": 2,
    "disk_radius_fraction": 0.95,
    "epsilon_radiance": 1e-6,
    "dqf_required": 0,
    "grid_latitude_degrees": 1.0,
    "grid_longitude_degrees": 1.0,
    "rotation_sidereal": [14.38, -2.96, -0.48],
    "rotation_synodic_subtract": 0.9856,
    "connected_component_structure": "8-connected, periodic longitude",
}
FEATURES = (
    "dim_area_frac", "dim_largest_frac", "dim_mean_logratio", "dim_p10_logratio",
    "dim_area_slope_6h", "dim_depth_slope_6h", "dim_n_components",
    "dim_centroid_lon", "dim_centroid_lat", "dim_east_west_balance",
    "dim_onset_age_h", "dim_peak_growth_age_h", "dim_missing",
)
LAT_CENTERS = np.arange(-89.5, 90.0, 1.0, dtype=np.float64)
LAT_AREA = np.cos(np.deg2rad(LAT_CENTERS)).astype(np.float32)
EPOCH = datetime(2022, 1, 1, tzinfo=timezone.utc)
STRUCTURE_8 = np.ones((3, 3), dtype=bool)


class UnusableSUVIFrame(ValueError):
    """A provenance-valid source frame without physical solar-disk geometry."""


class FrameCatalog(list):
    """Sorted frame rows with precomputed time/provenance arrays for origin queries."""

    def __init__(self, records):
        super().__init__(records)
        self.obs_end_ns = np.asarray([r["obs_end_ns"] for r in records], dtype=np.int64)
        self.last_modified_ns = np.asarray([r["last_modified_ns"] for r in records], dtype=np.int64)
        self.byte_equal = np.asarray([r["byte_equal"] for r in records], dtype=bool)
        self.source_usable = np.asarray([r.get("source_usable", r["byte_equal"]) for r in records], dtype=bool)


def _as_utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    return value.to_pydatetime().replace(tzinfo=timezone.utc) if value.tzinfo is None else value.to_pydatetime().astimezone(timezone.utc)


def _rotation_rate(latitude_degrees):
    s2 = np.sin(np.deg2rad(latitude_degrees)) ** 2
    sidereal = 14.38 - 2.96 * s2 - 0.48 * s2 * s2
    return sidereal - 0.9856


def extract_carrington_map(path, obs_end):
    """Read a local NetCDF image and map valid r<=0.95R radiance to 1-degree CAR bins."""
    import h5py

    source = io.BytesIO(path) if isinstance(path, (bytes, bytearray)) else Path(path)
    path_label = "<image-bytes>" if isinstance(path, (bytes, bytearray)) else str(path)
    with h5py.File(source, "r") as nc:
        rad_var, dqf_var = nc["RAD"], nc["DQF"]
        header = dict(nc.attrs)
        for key in ("CRPIX1", "CRPIX2", "RSUN", "CROTA", "SOLAR_B0", "PC1_1", "PC1_2", "PC2_1", "PC2_2"):
            if key in nc:
                header[key] = np.asarray(nc[key][...]).item()
        h = {str(k).upper(): v for k, v in header.items()}

        def number(key, default=None):
            value = h.get(key, default)
            if value is None:
                raise ValueError(f"missing SUVI geometry {key}: {path_label}")
            return float(np.asarray(value).item())

        matrix = np.array([
            [number("PC1_1", np.cos(np.deg2rad(number("CROTA", 0)))),
             number("PC1_2", -np.sin(np.deg2rad(number("CROTA", 0))))],
            [number("PC2_1", np.sin(np.deg2rad(number("CROTA", 0)))),
             number("PC2_2", np.cos(np.deg2rad(number("CROTA", 0))))],
        ], dtype=np.float64)
        singular = np.linalg.svd(matrix, compute_uv=False).min()
        rsun = number("RSUN")
        if rsun == -999.0:
            raise UnusableSUVIFrame(f"invalid SUVI geometry: RSUN sentinel -999: {path_label}")
        if not np.isfinite(rsun) or rsun <= 0 or singular <= 0:
            raise ValueError(f"invalid SUVI geometry: {path_label}")
        cx, cy = number("CRPIX1") - 1.0, number("CRPIX2") - 1.0
        radius = rsun / singular + 2.0
        height, width = rad_var.shape
        x0 = max(0, int(np.floor(cx - radius)))
        x1 = min(width, int(np.ceil(cx + radius)) + 1)
        y0 = max(0, int(np.floor(cy - radius)))
        y1 = min(height, int(np.ceil(cy + radius)) + 1)
        raw = np.asarray(rad_var[y0:y1, x0:x1])
        dqf = np.asarray(dqf_var[y0:y1, x0:x1])
        attrs = dict(rad_var.attrs)

    rad = raw.astype(np.float64)
    invalid = ~np.isfinite(rad)
    for name in ("_FillValue", "missing_value"):
        if name in attrs:
            invalid |= raw == np.asarray(attrs[name]).reshape(-1)[0]
    rad *= float(np.asarray(attrs.get("scale_factor", 1)).item())
    rad += float(np.asarray(attrs.get("add_offset", 0)).item())
    rad[invalid] = np.nan

    yy, xx = np.indices(rad.shape, dtype=np.float64)
    xx += x0 - cx
    yy += y0 - cy
    x = (matrix[0, 0] * xx + matrix[0, 1] * yy) / rsun
    y = (matrix[1, 0] * xx + matrix[1, 1] * yy) / rsun
    r2 = x * x + y * y
    disk = r2 <= CONFIG["disk_radius_fraction"] ** 2
    z = np.sqrt(np.maximum(1e-12, 1.0 - r2))
    b0 = np.deg2rad(number("SOLAR_B0", 0.0))
    latitude = np.arcsin(np.clip(y * np.cos(b0) + z * np.sin(b0), -1.0, 1.0))
    longitude = np.arctan2(x, z * np.cos(b0) - y * np.sin(b0))
    lat_deg = np.rad2deg(latitude)
    lon_deg = np.rad2deg(longitude)
    obs_end = _as_utc(obs_end)
    elapsed = (obs_end - EPOCH).total_seconds() / 86400.0
    carrington_lon = np.mod(lon_deg + _rotation_rate(lat_deg) * elapsed, 360.0)
    valid = disk & (dqf == CONFIG["dqf_required"]) & np.isfinite(rad) & (rad >= 0.0)
    if not valid.any():
        return np.full((180, 360), np.nan, dtype=np.float32)

    lat_index = np.clip(np.floor(lat_deg[valid] + 90.0).astype(np.int32), 0, 179)
    lon_index = np.floor(carrington_lon[valid]).astype(np.int32) % 360
    flat = lat_index * 360 + lon_index
    surface_weight = 1.0 / z[valid]
    sums = np.bincount(flat, weights=rad[valid] * surface_weight, minlength=180 * 360)
    weights = np.bincount(flat, weights=surface_weight, minlength=180 * 360)
    result = np.full(180 * 360, np.nan, dtype=np.float32)
    np.divide(sums, weights, out=result, where=weights > 0)
    return result.reshape(180, 360)


def _periodic_labels(mask):
    """Label 8-connected regions while joining the longitude seam."""
    labels, count = ndimage.label(mask, structure=STRUCTURE_8)
    parent = np.arange(count + 1, dtype=np.int32)

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def join(left, right):
        if left and right:
            a, b = find(int(left)), find(int(right))
            if a != b:
                parent[b] = a

    for row in range(mask.shape[0]):
        for delta in (-1, 0, 1):
            other = row + delta
            if 0 <= other < mask.shape[0]:
                join(labels[row, 0], labels[other, -1])
    roots = np.arange(count + 1, dtype=np.int32)
    for value in range(1, count + 1):
        roots[value] = find(value)
    return roots[labels]


def _wrap_degrees(values):
    return (values + 180.0) % 360.0 - 180.0


def _median_with_min_count(stack, minimum):
    finite = np.isfinite(stack)
    counts = finite.sum(axis=0)
    ordered = np.sort(np.where(finite, stack, np.inf), axis=0)
    lower = np.maximum(0, (counts - 1) // 2)
    upper = np.maximum(0, counts // 2)
    lo = np.take_along_axis(ordered, lower[None, ...], axis=0)[0]
    hi = np.take_along_axis(ordered, upper[None, ...], axis=0)[0]
    median = ((lo + hi) * 0.5).astype(np.float32, copy=False)
    median[counts < minimum] = np.nan
    return median, counts


def current_reference_groups(origin, catalog):
    """Return causal current frames and their admissible 24-hour reference groups."""
    origin = _as_utc(origin)
    origin_ns = np.datetime64(origin.replace(tzinfo=None), "ns").astype(np.int64)
    obs_ns = catalog.obs_end_ns if isinstance(catalog, FrameCatalog) else np.asarray([r["obs_end_ns"] for r in catalog], dtype=np.int64)
    mod_ns = catalog.last_modified_ns if isinstance(catalog, FrameCatalog) else np.asarray([r["last_modified_ns"] for r in catalog], dtype=np.int64)
    source_usable = catalog.source_usable if isinstance(catalog, FrameCatalog) else np.asarray(
        [r.get("source_usable", r["byte_equal"]) for r in catalog], dtype=bool)
    eligible = (obs_ns + int(3600 * 1e9) <= origin_ns) & (mod_ns <= origin_ns) & source_usable
    eligible_indices = np.flatnonzero(eligible)
    current_indices = [i for i in eligible_indices if obs_ns[i] >= origin_ns - int(18 * 3600 * 1e9)]
    tolerance = int(90 * 60 * 1e9)
    lag = int(24 * 3600 * 1e9)
    groups = []
    for i in current_indices:
        center = obs_ns[i] - lag
        lo = np.searchsorted(obs_ns, center - tolerance, side="left")
        hi = np.searchsorted(obs_ns, center + tolerance, side="right")
        ref_indices = [j for j in range(lo, hi) if eligible[j]]
        if len(ref_indices) >= CONFIG["minimum_reference_frames"]:
            groups.append((int(i), [int(j) for j in ref_indices]))
    return groups


def _build_frame_product(current_index, ref_indices, catalog, get_map):
    row = catalog[current_index]
    current = get_map(row["frame_id"])
    reference_stack = np.stack([get_map(catalog[j]["frame_id"]) for j in ref_indices]).astype(np.float32)
    reference, ref_counts = _median_with_min_count(reference_stack, CONFIG["minimum_reference_frames"])
    valid = (np.isfinite(current) & np.isfinite(reference) &
             (ref_counts >= CONFIG["minimum_reference_frames"]) & (reference > 0.0))
    ratio = np.zeros_like(current, dtype=np.float32)
    ratio[valid] = (current[valid] + CONFIG["epsilon_radiance"]) / (
        reference[valid] + CONFIG["epsilon_radiance"])
    log_ratio = np.zeros_like(current, dtype=np.float32)
    log_ratio[valid] = np.log(ratio[valid])
    candidate = valid & (ratio <= CONFIG["threshold_ratio"]) & (log_ratio <= 0.0)
    return {"valid": valid, "log_ratio": log_ratio, "candidate": candidate,
            "labels": _periodic_labels(candidate)}


class FrameProductCache:
    """Hash-keyed per-current-frame ratio and component cache with a small RAM LRU."""

    def __init__(self, catalog, get_map, root, code_hash, lru_size=128):
        self.catalog = catalog
        self.get_map = get_map
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.code_hash = code_hash
        self.lru_size = lru_size
        self.memory = OrderedDict()
        self.used_keys = set()
        self.generated_count = 0
        self.loaded_count = 0

    def _key(self, current_index, ref_indices):
        row = self.catalog[current_index]
        sources = [row] + [self.catalog[index] for index in ref_indices]
        payload = {
            "cache_version": 1,
            "code_config_sha256": self.code_hash,
            "config": CONFIG,
            "frame_ids": [source["frame_id"] for source in sources],
            "map_sha256": [source["map_sha256"] for source in sources],
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(raw).hexdigest()

    def get(self, current_index, ref_indices):
        key = self._key(current_index, ref_indices)
        self.used_keys.add(key)
        product = self.memory.pop(key, None)
        if product is not None:
            self.memory[key] = product
            return product
        path = self.root / f"{key}.npz"
        if path.is_file():
            with np.load(path) as saved:
                product = {name: saved[name].copy() for name in ("valid", "log_ratio", "candidate", "labels")}
            self.loaded_count += 1
        else:
            product = _build_frame_product(current_index, ref_indices, self.catalog, self.get_map)
            temp = path.with_suffix(f".tmp.{os.getpid()}.npz")
            np.savez(temp, **product)
            temp.replace(path)
            self.generated_count += 1
        self.memory[key] = product
        while len(self.memory) > self.lru_size:
            self.memory.popitem(last=False)
        return product

    def receipt(self):
        keys = sorted(self.used_keys)
        digest = hashlib.sha256("\n".join(keys).encode()).hexdigest()
        return {"cache_root": str(self.root), "unique_product_count": len(keys),
                "generated_count": self.generated_count, "loaded_from_disk_count": self.loaded_count,
                "lru_size": self.lru_size, "input_key_manifest_sha256": digest}


def _origin_arrays(origin, catalog, get_map, product_cache=None):
    origin = _as_utc(origin)
    origin_pd = np.datetime64(origin.replace(tzinfo=None), "ns")
    obs_ns = catalog.obs_end_ns if isinstance(catalog, FrameCatalog) else np.asarray([r["obs_end_ns"] for r in catalog], dtype=np.int64)
    groups = current_reference_groups(origin, catalog)
    selected_ids = set()
    frames = []
    previous_mask = None
    previous_time_ns = None
    for i, ref_indices in groups:
        row = catalog[i]
        product = (product_cache.get(i, ref_indices) if product_cache is not None else
                   _build_frame_product(i, ref_indices, catalog, get_map))
        valid = product["valid"]
        if valid.sum() < 100:
            previous_mask, previous_time_ns = None, None
            continue
        log_ratio = product["log_ratio"]
        candidate = product["candidate"]
        labels = product["labels"]
        persistent = np.zeros_like(candidate)
        gap_hours = None if previous_time_ns is None else (obs_ns[i] - previous_time_ns) / 3.6e12
        if previous_mask is not None and gap_hours <= 1.5:
            overlap_labels = np.unique(labels[candidate & previous_mask])
            overlap_labels = overlap_labels[overlap_labels > 0]
            if len(overlap_labels):
                persistent = candidate & np.isin(labels, overlap_labels)
        previous_mask = candidate
        previous_time_ns = obs_ns[i]
        selected_ids.add(row["frame_id"])
        selected_ids.update(catalog[j]["frame_id"] for j in ref_indices)
        frames.append({"time_ns": obs_ns[i], "valid": valid, "log_ratio": log_ratio,
                       "candidate": candidate, "persistent": persistent,
                       "labels": labels, "row": row})

    empty = {name: 0.0 for name in FEATURES}
    empty["dim_onset_age_h"] = 18.0
    empty["dim_peak_growth_age_h"] = 18.0
    empty["dim_missing"] = 1.0
    if len(frames) < CONFIG["minimum_consecutive_dimming_frames"]:
        return empty, sorted(selected_ids), {"valid_pairs": len(frames), "active": False}

    latest = frames[-1]
    latest_valid = latest["valid"]
    area_grid = np.broadcast_to(LAT_AREA[:, None], (180, 360))
    valid_area = float(area_grid[latest_valid].sum())
    if valid_area <= 0:
        return empty, sorted(selected_ids), {"valid_pairs": len(frames), "active": False}

    area_series, depth_series, active_times = [], [], []
    component_areas = []
    for frame in frames:
        mask = frame["persistent"]
        area = float(area_grid[mask].sum() / max(1e-12, float(area_grid[frame["valid"]].sum()))) if mask.any() else 0.0
        depth = float(-np.mean(frame["log_ratio"][mask])) if mask.any() else 0.0
        area_series.append((frame["time_ns"], area))
        depth_series.append((frame["time_ns"], depth))
        if mask.any():
            active_times.append(frame["time_ns"])
            labels = frame["labels"]
            ids = np.unique(labels[mask])
            ids = ids[ids > 0]
            sizes = np.asarray([float(area_grid[mask & (labels == label)].sum()) for label in ids])
            component_areas.append((frame["time_ns"], sizes))

    latest_mask = latest["persistent"]
    latest_log = latest["log_ratio"]
    latest_labels = latest["labels"]
    if latest_mask.any():
        ids = np.unique(latest_labels[latest_mask])
        ids = ids[ids > 0]
        sizes = np.asarray([float(area_grid[latest_mask & (latest_labels == label)].sum()) for label in ids])
        area_fraction = float(area_grid[latest_mask].sum() / valid_area)
        largest_fraction = float(sizes.max() / valid_area) if len(sizes) else 0.0
        selected_logs = latest_log[latest_mask].astype(np.float64)
        rows, cols = np.nonzero(latest_mask)
        lat = LAT_CENTERS[rows]
        carr_lon = cols.astype(np.float64) + 0.5
        origin_days = (origin - EPOCH).total_seconds() / 86400.0
        lon_obs = _wrap_degrees(carr_lon - _rotation_rate(lat) * origin_days)
        weights = area_grid[rows, cols].astype(np.float64)
        lon_angle = np.deg2rad(lon_obs)
        centroid_lon = float(np.rad2deg(np.arctan2(np.sum(weights * np.sin(lon_angle)),
                                                     np.sum(weights * np.cos(lon_angle)))))
        centroid_lat = float(np.average(lat, weights=weights))
        east_area = float(weights[lon_obs < 0].sum())
        west_area = float(weights[lon_obs >= 0].sum())
        east_west = (east_area - west_area) / max(1e-12, east_area + west_area)
        n_components = float(len(ids))
        mean_log = float(np.mean(selected_logs))
        p10_log = float(np.quantile(selected_logs, 0.1))
    else:
        sizes = np.asarray([], dtype=np.float64)
        area_fraction = largest_fraction = mean_log = p10_log = 0.0
        centroid_lon = centroid_lat = east_west = n_components = 0.0

    cutoff_ns = (origin_pd - np.timedelta64(6, "h")).astype(np.int64)
    recent_area = [(t, v) for t, v in area_series if t >= cutoff_ns]
    recent_depth = [(t, v) for t, v in depth_series if t >= cutoff_ns]
    def slope(points):
        if len(points) < 2:
            return 0.0
        x = np.asarray([(t - origin_pd.astype(np.int64)) / 3.6e12 for t, _ in points], dtype=np.float64)
        y = np.asarray([v for _, v in points], dtype=np.float64)
        return float(np.polyfit(x, y, 1)[0])

    if active_times:
        onset_age = float(np.clip((origin_pd.astype(np.int64) - min(active_times)) / 3.6e12, 0.0, 18.0))
        peak_time, _ = max(area_series, key=lambda item: item[1])
        peak_age = float(np.clip((origin_pd.astype(np.int64) - peak_time) / 3.6e12, 0.0, 18.0))
    else:
        onset_age = peak_age = 18.0
    latest_age_hours = (origin_pd.astype(np.int64) - latest["time_ns"]) / 3.6e12
    missing = float(latest_age_hours > 2.0)
    features = {
        "dim_area_frac": area_fraction,
        "dim_largest_frac": largest_fraction,
        "dim_mean_logratio": mean_log,
        "dim_p10_logratio": p10_log,
        "dim_area_slope_6h": slope(recent_area),
        "dim_depth_slope_6h": slope(recent_depth),
        "dim_n_components": n_components,
        "dim_centroid_lon": centroid_lon,
        "dim_centroid_lat": centroid_lat,
        "dim_east_west_balance": east_west,
        "dim_onset_age_h": onset_age,
        "dim_peak_growth_age_h": peak_age,
        "dim_missing": missing,
    }
    return features, sorted(selected_ids), {"valid_pairs": len(frames), "active": bool(active_times),
                                           "latest_source_age_hours": float(latest_age_hours)}

