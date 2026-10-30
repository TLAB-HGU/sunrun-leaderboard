"""Causal PFSS Earth-footpoint OCB topology from hash-pinned GONG maps."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path('/home/t-lab01/sunrun')
LB = ROOT / "leaderboard"
STORE_ROOT = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store')
HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "gong_manifest.json"
PRECOMPUTE = HERE / "precompute.py"
VENDOR = LB / "experiments/q78_wsa_smoke/vendor"
STORE = STORE_ROOT / "solar-inc-gongocb72-20261014-r7-cache"

OCB_COLUMNS = (
    "footpoint_ocb_dist_deg",
    "open_flux_frac_5deg",
    "open_flux_frac_10deg",
    "ocb_length_local",
    "ocb_curvature",
    "nearest_open_patch_area",
    "nearest_open_patch_width",
    "footpoint_expansion_factor",
    "footpoint_lat",
    "footpoint_lon",
    "map_age_h",
    "ocb_missing",
)

OCB_CONFIG = {
    "source_surface_radius_solar_radii": 2.5,
    "pfss_solver": "pfsspy 1.2.0 potential-field source-surface solver (vendored q78 source)",
    "pfss_n_r": 15,
    "input_grid_latitude": 18,
    "input_grid_longitude": 36,
    "input_downsample_factor": 10,
    "availability_lag_hours": 2.0,
    "availability_rule": "max(observation_time + 2 h conservative latency, HTTP Last-Modified)",
    "earth_radial_speed_kms": 400.0,
    "solar_synodic_rotation_deg_per_day": 13.1988,
    "footpoint_patch_spacing_deg": 5.0,
    "footpoint_patch_half_width_deg": 25.0,
    "local_geometry_radius_deg": 10.0,
    "fieldline_tracer": "pfsspy.FortranTracer(step_size=0.1,max_steps=60000), PythonTracer fallback for incomplete Earth lines; incomplete local cells are missing; choose first connected origin per map as grid reference, mark all origins missing when none connect",
    "local_trace_failure_policy": "invalid fieldline endpoints mark local topology metrics missing without Python retry",
    "fieldline_step_size": 0.1,
    "fieldline_max_steps": 60000,
    "fallback_atol": 1.0e-4,
    "fallback_rtol": 1.0e-4,
    "endpoint_tolerance_solar_radii": 0.05,
    "cache_dir": str(STORE),
    "origin_feature_start_utc": "2022-10-01T00:00:00Z",
    "origin_feature_end_utc": "2026-09-30T23:00:00Z",
}

HOUR_NS = 3_600_000_000_000
DAY_NS = 24 * HOUR_NS
RSUN_KM = 6.957e5
AU_KM = 1.495978707e8
L0_ANCHOR_UTC = pd.Timestamp("2025-09-15T12:14:00Z")
L0_ANCHOR_CR = 2302.23358172
CARRINGTON_RATE_DEG_PER_DAY = 360.0 / 27.2753


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _timestamp_ns(value) -> int:
    stamp = pd.Timestamp(value)
    if stamp.tz is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.value


def _datetime_ns(values) -> np.ndarray:
    """Return datetime values as nanoseconds across pandas datetime resolutions."""
    index = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    return np.asarray(index.as_unit("ns").asi8, dtype=np.int64)


def _http_time_ns(value: str) -> int:
    dt = parsedate_to_datetime(value)
    return pd.Timestamp(dt).tz_convert("UTC").value


def _availability_ns(row) -> int:
    obs_ns = _timestamp_ns(row["observation_time_utc"])
    available = max(obs_ns + 2 * HOUR_NS, _http_time_ns(row["last_modified"]))
    if "available_at_utc" in row and _timestamp_ns(row["available_at_utc"]) != available:
        raise ValueError(f"GONG availability receipt mismatch: {row['file']}")
    return available


def _bundle_hash() -> str:
    pieces = []
    for path in (Path(__file__), PRECOMPUTE, HERE / "config.json"):
        pieces.append(path.name.encode() + b"\0" + path.read_bytes())
    return _file_digest(b"\0".join(pieces))


def _config_hash(config) -> str:
    return _file_digest(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())


def _cache_key(source_sha256: str, code_sha256: str, config_sha256: str) -> str:
    return _file_digest(f"{code_sha256}:{config_sha256}:{source_sha256}".encode("ascii"))


def anchor_l0(times) -> np.ndarray:
    times = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    dt_days = (_datetime_ns(times) - L0_ANCHOR_UTC.value) / DAY_NS
    cr = L0_ANCHOR_CR + dt_days / 27.2753
    return (360.0 * (1.0 - np.mod(cr, 1.0))) % 360.0


def earth_source_surface_longitude(origin, config) -> float:
    origin = pd.Timestamp(origin)
    l0 = float(anchor_l0(pd.DatetimeIndex([origin]))[0])
    travel_days = ((AU_KM - float(config["source_surface_radius_solar_radii"]) * RSUN_KM)
                   / float(config["earth_radial_speed_kms"]) / 86400.0)
    return float((l0 + CARRINGTON_RATE_DEG_PER_DAY * travel_days) % 360.0)


def _load_pfss_context(row, config, *, code_sha256, config_sha256, cache_dir, reference_origins):
    """Build one fixed-Rss PFSS solution and a local photospheric open/closed grid."""
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    import sunpy.map
    from scipy.ndimage import map_coordinates
    import pfsspy
    from pfsspy.tracing import FortranTracer, PythonTracer

    source_path = Path(row["path"])
    actual_sha = sha256_file(source_path)
    if actual_sha != row["sha256"]:
        raise ValueError(f"GONG map SHA256 mismatch: {row['file']}")

    raw_map = sunpy.map.Map(str(source_path))
    raw = np.asarray(raw_map.data, dtype=np.float64)
    nlat = int(config["input_grid_latitude"])
    nlon = int(config["input_grid_longitude"])
    factor = int(config["input_downsample_factor"])
    if raw.shape != (nlat * factor, nlon * factor):
        raise ValueError(f"unexpected GONG map shape {raw.shape}: {row['file']}")
    br = raw.reshape(nlat, factor, nlon, factor).mean((1, 3))
    date_obs = pd.Timestamp(row["observation_time_utc"]).to_pydatetime()
    header = pfsspy.utils.carr_cea_wcs_header(
        date_obs, br.T.shape, map_center_longitude=float(raw_map.meta["crval1"]) * u.deg)
    balanced_br = br - float(np.nanmean(br))
    magnetogram = sunpy.map.Map(balanced_br, header)
    solution = pfsspy.pfss(pfsspy.Input(
        magnetogram, int(config["pfss_n_r"]),
        float(config["source_surface_radius_solar_radii"])))
    tracer = FortranTracer(max_steps=int(config["fieldline_max_steps"]),
                           step_size=float(config["fieldline_step_size"]))
    rss = float(config["source_surface_radius_solar_radii"])
    tolerance = float(config["endpoint_tolerance_solar_radii"])

    reference_origins = [
        pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")
        for value in reference_origins
    ]
    if not reference_origins:
        raise ValueError("a GONG map task needs at least one candidate reference origin")

    source_sha = row["sha256"]
    key = _cache_key(source_sha, code_sha256, config_sha256)
    cache_dir = Path(cache_dir) / "maps"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.npz"
    cached = None
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as values:
                if (str(values["cache_key"].item()) == key
                        and str(values["source_sha256"].item()) == source_sha
                        and str(values["code_sha256"].item()) == code_sha256
                        and str(values["config_sha256"].item()) == config_sha256
                        and str(values["reference_origin_utc"].item()) in reference_origins):
                    cached = {name: np.asarray(values[name]) for name in (
                        "open_mask", "valid_mask", "expansion", "abs_br", "grid_x", "grid_y")}
                    ref_lat = float(values["reference_footpoint_lat"].item())
                    ref_lon = float(values["reference_footpoint_lon"].item())
                    reference_expansion = float(values["reference_expansion"].item())
                    repair_count = int(values["grid_repair_count"].item())
        except (OSError, KeyError, ValueError):
            cached = None

    if cached is None:
        reference_line = None
        source_fallback = 0
        reference_origin = None
        for candidate_origin in reference_origins:
            try:
                reference_line, source_fallback = _trace_source_line(
                    solution, tracer, candidate_origin, config, rss, tolerance,
                    PythonTracer, SkyCoord, u)
            except RuntimeError as exc:
                if "Earth source-surface line did not reach the photosphere" not in str(exc):
                    raise
                continue
            reference_origin = candidate_origin
            break
        if reference_line is None:
            return {
                "row": row,
                "solution": solution,
                "tracer": tracer,
                "reference_missing": True,
                "repair_count": 0,
                "cache_path": None,
            }
        ref_lat, ref_lon = _line_footpoint(reference_line, u)
        reference_expansion = float(reference_line.expansion_factor)
        spacing = float(config["footpoint_patch_spacing_deg"])
        half = float(config["footpoint_patch_half_width_deg"])
        offsets = np.arange(-half, half + spacing * 0.5, spacing, dtype=float)
        grid_x, grid_y = np.meshgrid(offsets, offsets)
        cosine = max(0.2, math.cos(math.radians(ref_lat)))
        grid_lats = ref_lat + grid_y.ravel()
        grid_lons = (ref_lon + grid_x.ravel() / cosine) % 360.0
        seeds = SkyCoord(grid_lons * u.deg, grid_lats * u.deg,
                         1.001 * u.R_sun, frame=solution.coordinate_frame)
        lines, valid_lines = _trace_patch(seeds, solution, tracer, rss, tolerance)
        repair_count = int(source_fallback)
        valid_mask = valid_lines.astype(np.uint8).reshape(grid_x.shape)
        open_mask = np.asarray([
            bool(line.is_open) if valid else False for line, valid in zip(lines, valid_lines)
        ], dtype=np.uint8).reshape(grid_x.shape)
        expansion = np.asarray([
            float(line.expansion_factor) if line.is_open and valid else np.nan
            for line, valid in zip(lines, valid_lines)
        ], dtype=np.float64).reshape(grid_x.shape)

        map_coords = SkyCoord(grid_lons * u.deg, grid_lats * u.deg,
                              1.0 * u.R_sun, frame=solution.coordinate_frame)
        px, py = magnetogram.world_to_pixel(map_coords)
        width = balanced_br.shape[1]
        px = np.mod(np.asarray(px, dtype=float), width) + width
        py = np.asarray(py, dtype=float)
        periodic = np.concatenate([balanced_br, balanced_br, balanced_br], axis=1)
        abs_br = np.abs(map_coordinates(periodic, [py, px], order=1, mode="nearest")).reshape(grid_x.shape)
        cached = {
            "open_mask": open_mask,
            "valid_mask": valid_mask,
            "expansion": expansion,
            "abs_br": abs_br,
            "grid_x": grid_x,
            "grid_y": grid_y,
        }
        temporary = cache_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, cache_key=key, source_sha256=source_sha,
                code_sha256=code_sha256, config_sha256=config_sha256,
                reference_origin_utc=reference_origin,
                reference_footpoint_lat=ref_lat, reference_footpoint_lon=ref_lon,
                reference_expansion=reference_expansion, grid_repair_count=repair_count,
                **cached)
        temporary.replace(cache_path)

    return {
        "row": row,
        "solution": solution,
        "tracer": tracer,
        "cache": cached,
        "reference_missing": False,
        "reference_lat": ref_lat,
        "reference_lon": ref_lon,
        "reference_expansion": reference_expansion,
        "repair_count": repair_count if cached is not None else 0,
        "cache_path": str(cache_path),
    }


def _line_valid(line, rss, tolerance):
    if len(line) < 2:
        return False
    endpoints = (float(line._r[0]), float(line._r[-1]))
    return all(min(abs(radius - 1.0), abs(radius - rss)) <= tolerance
               for radius in endpoints)


def _trace_checked(seeds, solution, tracer, rss, tolerance, PythonTracer):
    lines = tracer.trace(seeds, solution)
    # StreamTracer's ROT=1 only means it hit its step limit; the sampled
    # endpoints can still both be on valid PFSS boundaries. Retry only lines
    # whose endpoints fail the physical boundary check below.
    bad = [i for i, line in enumerate(lines) if not _line_valid(line, rss, tolerance)]
    repairs = 0
    if bad:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        fallback = PythonTracer(atol=OCB_CONFIG["fallback_atol"],
                                rtol=OCB_CONFIG["fallback_rtol"])
        retry_seeds = seeds[bad] if getattr(seeds, "isscalar", False) is False else seeds
        retry = fallback.trace(retry_seeds, solution)
        updated = list(lines.field_lines)
        for index, line in zip(bad, retry):
            updated[index] = line
            repairs += 1
        lines = type(lines)(updated)
        bad = [i for i, line in enumerate(lines) if not _line_valid(line, rss, tolerance)]
    if bad:
        raise RuntimeError(f"PFSS endpoint check failed for {len(bad)} field lines")
    return lines, repairs


def _trace_patch(seeds, solution, tracer, rss, tolerance):
    """Trace the local topology grid once; incomplete cells stay explicitly missing."""
    lines = tracer.trace(seeds, solution)
    valid = np.asarray([_line_valid(line, rss, tolerance) for line in lines], dtype=bool)
    return lines, valid


def _trace_source_line(solution, tracer, origin, config, rss, tolerance,
                       PythonTracer, SkyCoord, u):
    longitude = earth_source_surface_longitude(origin, config)
    seed = SkyCoord(longitude * u.deg, 0.0 * u.deg, (rss - 0.001) * u.R_sun,
                    frame=solution.coordinate_frame)
    lines, repairs = _trace_checked(seed, solution, tracer, rss, tolerance, PythonTracer)
    line = lines[0]
    if not line.is_open:
        raise RuntimeError(f"Earth source-surface line did not reach the photosphere: {origin}")
    return line, repairs


def _line_footpoint(line, u):
    point = line.solar_footpoint
    lat = float(point.lat.to_value(u.deg))
    lon = float(point.lon.to_value(u.deg)) % 360.0
    if not np.isfinite([lat, lon]).all():
        raise ValueError("nonfinite PFSS photospheric footpoint")
    return lat, lon


def _wrap_delta_deg(value):
    return (float(value) + 180.0) % 360.0 - 180.0


def _sample_grid(array, x_rel, y_rel, config, *, order):
    from scipy.ndimage import map_coordinates
    spacing = float(config["footpoint_patch_spacing_deg"])
    half = float(config["footpoint_patch_half_width_deg"])
    x = (np.asarray(x_rel, dtype=float) + half) / spacing
    y = (np.asarray(y_rel, dtype=float) + half) / spacing
    center = half / spacing
    x = np.clip(x, 0.0, 2.0 * center)
    y = np.clip(y, 0.0, 2.0 * center)
    return map_coordinates(np.asarray(array, dtype=float), [y, x], order=order, mode="nearest")


def _angular_distance_tangent(dx, dy):
    return np.sqrt(np.asarray(dx, dtype=float) ** 2 + np.asarray(dy, dtype=float) ** 2)


def _ocb_local_geometry(origin, context, config, source_line, source_fallback):
    from scipy.ndimage import distance_transform_edt, label

    rss = float(config["source_surface_radius_solar_radii"])
    foot_lat, foot_lon = _line_footpoint(source_line, __import__("astropy.units", fromlist=["deg"]))
    # The map patch is stored in a tangent plane centred on its first admissible origin.
    ref_lat = float(context["reference_lat"])
    ref_lon = float(context["reference_lon"])
    ref_cos = max(0.2, math.cos(math.radians(ref_lat)))
    fp_dx = _wrap_delta_deg(foot_lon - ref_lon) * ref_cos
    fp_dy = foot_lat - ref_lat
    expansion = float(source_line.expansion_factor)
    age_h = (pd.Timestamp(origin).tz_convert("UTC").value
             - _timestamp_ns(context["row"]["observation_time_utc"])) / HOUR_NS
    if (abs(fp_dx) + float(config["local_geometry_radius_deg"]) >
            float(config["footpoint_patch_half_width_deg"]) + 1e-6
            or abs(fp_dy) + float(config["local_geometry_radius_deg"]) >
            float(config["footpoint_patch_half_width_deg"]) + 1e-6):
        values = np.asarray([
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            expansion, foot_lat, foot_lon, float(age_h), 1.0,
        ], dtype=np.float64)
        return values, int(source_fallback)

    cache = context["cache"]
    spacing = float(config["footpoint_patch_spacing_deg"])
    offsets = np.arange(-float(config["local_geometry_radius_deg"]),
                        float(config["local_geometry_radius_deg"]) + spacing * 0.5,
                        spacing, dtype=float)
    ox, oy = np.meshgrid(offsets, offsets)
    shifted_x = fp_dx + ox
    shifted_y = fp_dy + oy
    local_open = (_sample_grid(cache["open_mask"], shifted_x, shifted_y, config, order=0) >= 0.5)
    local_valid = (_sample_grid(cache["valid_mask"], shifted_x, shifted_y, config, order=0) >= 0.5)
    if not local_valid.all():
        values = np.asarray([
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            expansion, foot_lat, foot_lon, float(age_h), 1.0,
        ], dtype=np.float64)
        return values, int(source_fallback)
    local_abs_br = _sample_grid(cache["abs_br"], shifted_x, shifted_y, config, order=1)
    local_expansion = _sample_grid(cache["expansion"], shifted_x, shifted_y, config, order=1)

    flux_fracs = []
    for radius in (5.0, 10.0):
        distance = _angular_distance_tangent(ox, oy)
        selected = distance <= radius + 1e-6
        area_weight = np.cos(np.deg2rad(np.clip(foot_lat + oy, -89.0, 89.0)))
        weights = np.nan_to_num(local_abs_br, nan=0.0) * area_weight
        denom = float(weights[selected].sum())
        numer = float(weights[selected & local_open].sum())
        flux_fracs.append(numer / denom if denom > 0 else float("nan"))

    # Open/closed transitions in the stored, wider patch approximate the OCB contour.
    open_mask = np.asarray(cache["open_mask"], dtype=bool)
    grid_x = np.asarray(cache["grid_x"], dtype=float)
    grid_y = np.asarray(cache["grid_y"], dtype=float)
    boundaries = []
    for iy in range(open_mask.shape[0]):
        for ix in range(open_mask.shape[1] - 1):
            if open_mask[iy, ix] != open_mask[iy, ix + 1]:
                boundaries.append((0.5 * (grid_x[iy, ix] + grid_x[iy, ix + 1]),
                                   grid_y[iy, ix], spacing,
                                   abs(foot_lat + grid_y[iy, ix])))
    for iy in range(open_mask.shape[0] - 1):
        for ix in range(open_mask.shape[1]):
            if open_mask[iy, ix] != open_mask[iy + 1, ix]:
                lat_here = foot_lat + grid_y[iy, ix]
                length = spacing * max(0.05, math.cos(math.radians(lat_here)))
                boundaries.append((grid_x[iy, ix],
                                   0.5 * (grid_y[iy, ix] + grid_y[iy + 1, ix]),
                                   length, abs(lat_here)))
    if boundaries:
        b = np.asarray(boundaries, dtype=float)
        bdx, bdy = b[:, 0] - fp_dx, b[:, 1] - fp_dy
        bdist = _angular_distance_tangent(bdx, bdy)
        footpoint_distance = float(np.min(bdist))
        local_boundary = bdist <= float(config["local_geometry_radius_deg"]) + spacing * 0.5
        local_length = float(np.sum(b[local_boundary, 2]))
    else:
        footpoint_distance = float(config["footpoint_patch_half_width_deg"])
        local_boundary = np.zeros(0, dtype=bool)
        local_length = 0.0

    # Curvature is mean |div(n)| on the signed-distance contour band (deg^-1).
    inside = distance_transform_edt(open_mask, sampling=(spacing, spacing))
    outside = distance_transform_edt(~open_mask, sampling=(spacing, spacing))
    signed_distance = inside - outside
    gy, gx = np.gradient(signed_distance, spacing, spacing)
    norm = np.maximum(np.sqrt(gx * gx + gy * gy), 1e-8)
    nx, ny = gx / norm, gy / norm
    curvature = np.gradient(nx, spacing, axis=1) + np.gradient(ny, spacing, axis=0)
    cache_distance = _angular_distance_tangent(grid_x - fp_dx, grid_y - fp_dy)
    contour_band = (np.abs(signed_distance) <= spacing) & (
        cache_distance <= float(config["local_geometry_radius_deg"]) + spacing)
    curvature_values = np.abs(curvature[contour_band])
    local_curvature = float(np.mean(curvature_values)) if boundaries and curvature_values.size else 0.0

    # Area/width of the closest open component in the +/-10 degree neighbourhood.
    labels, count = label(local_open, structure=np.ones((3, 3), dtype=np.uint8))
    center = np.unravel_index(np.argmin(ox * ox + oy * oy), ox.shape)
    component = int(labels[center])
    if component == 0 and count:
        candidates = np.argwhere(labels > 0)
        nearest = candidates[np.argmin(ox[labels > 0] ** 2 + oy[labels > 0] ** 2)]
        component = int(labels[tuple(nearest)])
    patch = labels == component if component else np.zeros_like(local_open, dtype=bool)
    patch_indices = np.argwhere(patch)
    if len(patch_indices):
        patch_lats = np.clip(foot_lat + oy[patch], -89.0, 89.0)
        cell_area = spacing * spacing * np.cos(np.deg2rad(patch_lats))
        patch_area = float(np.sum(cell_area))
        coords = np.column_stack((ox[patch] * math.cos(math.radians(foot_lat)), oy[patch]))
        pairwise = coords[:, None, :] - coords[None, :, :]
        patch_width = float(np.sqrt(np.max(np.sum(pairwise * pairwise, axis=2)))) if len(coords) > 1 else spacing
    else:
        patch_area, patch_width = 0.0, 0.0

    values = np.asarray([
        footpoint_distance, flux_fracs[0], flux_fracs[1], local_length,
        local_curvature, patch_area, patch_width, expansion, foot_lat, foot_lon,
        float(age_h), 0.0,
    ], dtype=np.float64)
    if len(values) != len(OCB_COLUMNS) or not np.isfinite(values[[0, 3, 4, 5, 6, 8, 9, 10, 11]]).all():
        raise ValueError(f"nonfinite mandatory OCB feature at {origin}")
    return values, int(source_fallback)


def build_ocb_features(origin, gong_map, config) -> np.ndarray:
    """Return the fixed 12 causal footpoint/OCB floats for one origin and map context."""
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from pfsspy.tracing import PythonTracer

    rss = float(config["source_surface_radius_solar_radii"])
    try:
        line, fallback_count = _trace_source_line(
            gong_map["solution"], gong_map["tracer"], origin, config, rss,
            float(config["endpoint_tolerance_solar_radii"]), PythonTracer, SkyCoord, u)
    except RuntimeError as exc:
        if "Earth source-surface line did not reach the photosphere" not in str(exc):
            raise
        age_h = (pd.Timestamp(origin).tz_convert("UTC").value
                 - _timestamp_ns(gong_map["row"]["observation_time_utc"])) / HOUR_NS
        return np.asarray([np.nan] * 10 + [float(age_h), 1.0], dtype=np.float64)
    features, repair_count = _ocb_local_geometry(origin, gong_map, config, line, fallback_count)
    gong_map["feature_fallbacks"] = int(gong_map.get("feature_fallbacks", 0)) + int(repair_count)
    if len(features) != 12:
        raise ValueError(f"expected 12 OCB features, got {len(features)}")
    return features


def _worker_map_task(task):
    """Compute origin-level features for all origins selecting one GONG map."""
    row = task["row"]
    origins = task["origins"]
    context = _load_pfss_context(
        row, task["config"], code_sha256=task["code_sha256"],
        config_sha256=task["config_sha256"], cache_dir=task["cache_dir"],
        reference_origins=origins)
    if context["reference_missing"]:
        observation_ns = _timestamp_ns(row["observation_time_utc"])
        features = []
        for origin in origins:
            age_h = (_timestamp_ns(origin) - observation_ns) / HOUR_NS
            features.append(np.asarray([np.nan] * 10 + [float(age_h), 1.0], dtype=np.float64))
        return {
            "map_id": int(task["map_id"]),
            "origins_ns": [int(_timestamp_ns(value)) for value in origins],
            "features": np.asarray(features, dtype=np.float64),
            "fallback_count": 0,
            "cache_path": None,
        }
    features = [build_ocb_features(origin, context, task["config"]) for origin in origins]
    fallback_count = int(context.get("feature_fallbacks", 0)) + int(context["repair_count"])
    return {
        "map_id": int(task["map_id"]),
        "origins_ns": [int(_timestamp_ns(value)) for value in origins],
        "features": np.asarray(features, dtype=np.float64),
        "fallback_count": int(fallback_count),
        "cache_path": context["cache_path"],
    }


class GongOCBCatalog:
    """Read the hash-bound hourly OCB table and expose the E1022-compatible API."""

    def __init__(self, manifest_path=MANIFEST, config=None):
        self.manifest_path = Path(manifest_path).resolve()
        self.config = dict(OCB_CONFIG if config is None else config)
        self.manifest = json.loads(self.manifest_path.read_text())
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self.code_sha256 = _bundle_hash()
        self.config_sha256 = _config_hash(self.config)
        self.cache_dir = Path(self.config["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.origin_start = pd.Timestamp(self.config["origin_feature_start_utc"])
        self.origin_end = pd.Timestamp(self.config["origin_feature_end_utc"])
        self.feature_key = _file_digest(json.dumps({
            "manifest_sha256": self.manifest_sha256,
            "code_sha256": self.code_sha256,
            "config_sha256": self.config_sha256,
            "origin_start": str(self.origin_start),
            "origin_end": str(self.origin_end),
            "cadence": "1h",
        }, sort_keys=True).encode())
        self.feature_path = self.cache_dir / f"origin-features-{self.feature_key}.parquet"
        self.feature_receipt_path = self.cache_dir / f"origin-features-{self.feature_key}.json"
        self.rows = []
        for raw in self.manifest["maps"]:
            row = dict(raw)
            row["observation_ns"] = _timestamp_ns(row["observation_time_utc"])
            row["available_ns"] = _availability_ns(row)
            self.rows.append(row)
        self.rows.sort(key=lambda item: (item["observation_ns"], item["file"]))
        self.observation_ns = np.asarray([row["observation_ns"] for row in self.rows], dtype=np.int64)
        self.available_ns = np.asarray([row["available_ns"] for row in self.rows], dtype=np.int64)
        if not self.rows:
            raise ValueError("empty GONG OCB source manifest")
        if np.any(np.diff(self.observation_ns) <= 0):
            raise ValueError("GONG observations must be unique and increasing")
        if not self.feature_path.is_file() or not self.feature_receipt_path.is_file():
            raise FileNotFoundError("GONG OCB origin cache is missing; run the registered precompute stage")
        receipt = json.loads(self.feature_receipt_path.read_text())
        if (receipt.get("feature_key") != self.feature_key
                or receipt.get("manifest_sha256") != self.manifest_sha256
                or receipt.get("code_sha256") != self.code_sha256
                or receipt.get("config_sha256") != self.config_sha256
                or receipt.get("feature_table_sha256") != sha256_file(self.feature_path)):
            raise ValueError("GONG OCB feature table receipt/hash mismatch")
        frame = pd.read_parquet(self.feature_path)
        required = {"origin_ns", "source_file", "source_sha256", "available_at_utc", *OCB_COLUMNS}
        if not required.issubset(frame.columns):
            raise ValueError("GONG OCB feature cache is missing required columns")
        self.feature_frame = frame.set_index("origin_ns").sort_index()
        self.feature_receipt = receipt
        self._row_by_file = {row["file"]: row for row in self.rows}
        self.selected_files = set()

    def _index_for(self, origin):
        origin = pd.Timestamp(origin)
        if origin.tz is None:
            origin = origin.tz_localize("UTC")
        else:
            origin = origin.tz_convert("UTC")
        idx = int(np.searchsorted(self.observation_ns, origin.value, side="right") - 1)
        while idx >= 0 and self.available_ns[idx] > origin.value:
            idx -= 1
        return idx, origin

    def features_for(self, origins, overrides=None, with_metadata=False):
        overrides = {} if overrides is None else overrides
        origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
        origin_ns_values = _datetime_ns(origins)
        missing_ns = [int(value) for value in origin_ns_values
                      if int(value) not in self.feature_frame.index]
        if missing_ns:
            raise KeyError(f"origin not in precomputed OCB table: {pd.Timestamp(missing_ns[0], tz='UTC')}")
        values = self.feature_frame.reindex(origin_ns_values)[list(OCB_COLUMNS)].to_numpy(dtype=float)
        metadata = []
        indices = np.full(len(origins), -1, dtype=int)
        for position, origin in enumerate(origins):
            row_index, normalized = self._index_for(origin)
            record = self.feature_frame.loc[int(origin.value)]
            source_file = record["source_file"]
            if pd.isna(source_file):
                metadata.append(None)
                continue
            source_file = str(source_file)
            selected = self._row_by_file[source_file]
            if selected["file"] != source_file:
                raise ValueError(f"GONG source-map key mismatch at {origin}")
            indices[position] = row_index
            self.selected_files.add(source_file)
            if source_file in overrides:
                values[position] = np.asarray(overrides[source_file], dtype=float)
            metadata.append({
                "file": source_file,
                "observation_time_utc": selected["observation_time_utc"],
                "available_at_utc": selected["available_at_utc"],
                "age_hours": (normalized.value - selected["observation_ns"]) / HOUR_NS,
                "sha256": selected["sha256"],
                "url": selected["url"],
                "retrieval_time_utc": selected.get("retrieval_time_utc"),
            })
        if with_metadata:
            return values, indices, metadata
        return values

    def poison_overrides(self, origin):
        index, normalized = self._index_for(origin)
        if index < 0:
            raise ValueError("no admissible GONG map for OCB poison probe")
        selected = self.rows[index]
        future = next((row for row in self.rows
                       if row["observation_ns"] > normalized.value
                       and row["available_ns"] > normalized.value), None)
        if future is None:
            raise ValueError("no post-origin GONG map available for poison probe")
        base = self.features_for(pd.DatetimeIndex([normalized]))[0]
        future_values = np.full(len(OCB_COLUMNS), 999.0, dtype=float)
        future_values[-1] = 0.0
        positive_values = np.where(np.isfinite(base), base + 13.0, 13.0)
        positive_values[-1] = 0.0
        return ({future["file"]: future_values},
                {selected["file"]: positive_values}, selected, future)

    def official_report(self, origins):
        values, indices, metadata = self.features_for(origins, with_metadata=True)
        covered = indices >= 0
        ages = np.asarray([item["age_hours"] for item in metadata if item is not None], dtype=float)
        variance = {}
        for position, name in enumerate(OCB_COLUMNS):
            finite = values[:, position][np.isfinite(values[:, position])]
            variance[name] = float(np.var(finite)) if finite.size else None
        return {
            "source_coverage_origins": int(covered.sum()),
            "source_coverage_total": int(len(origins)),
            "source_coverage_fraction": float(covered.mean()) if len(origins) else 0.0,
        "topology_missing_origins": int(np.sum(values[:, -1] >= 1.0)),
        "topology_missing_fraction": float(np.mean(values[:, -1] >= 1.0)) if len(origins) else 0.0,
        "topology_finite_fraction": float(np.mean(values[:, -1] < 1.0)) if len(origins) else 0.0,
            "unique_selected_maps": int(len({item["file"] for item in metadata if item is not None})),
            "feature_variance": variance,
            "age_hours_min_median_p95_max": {
                "min": float(np.min(ages)) if ages.size else None,
                "median": float(np.median(ages)) if ages.size else None,
                "p95": float(np.quantile(ages, 0.95)) if ages.size else None,
                "max": float(np.max(ages)) if ages.size else None,
            },
            "latest_selected_map": max(
                (item["observation_time_utc"] for item in metadata if item is not None), default=None),
            "all_selected_maps_have_hash_url_and_retrieval": bool(all(
                item is not None and item["sha256"] and item["url"] and item["retrieval_time_utc"]
                for item in metadata if item is not None)),
            "feature_table": str(self.feature_path),
            "feature_table_sha256": self.feature_receipt["feature_table_sha256"],
            "manifest_sha256": self.manifest_sha256,
            "code_sha256": self.code_sha256,
            "configuration": self.config,
            "maps_precomputed": self.feature_receipt["unique_selected_maps"],
            "trace_fallback_count": self.feature_receipt["trace_fallback_count"],
            "origin_source_sha256_count": self.feature_receipt["source_hash_count"],
        }


def _origin_map_indices(origins, rows):
    obs = np.asarray([row["observation_ns"] for row in rows], dtype=np.int64)
    available = np.asarray([row["available_ns"] for row in rows], dtype=np.int64)
    origin_ns_values = _datetime_ns(origins)
    chosen = np.searchsorted(obs, origin_ns_values, side="right") - 1
    for i, origin_ns in enumerate(origin_ns_values):
        while chosen[i] >= 0 and available[chosen[i]] > origin_ns:
            chosen[i] -= 1
    return chosen.astype(np.int32)


def precompute_origin_table(manifest_path=MANIFEST, config=None, n_jobs=3):
    config = dict(OCB_CONFIG if config is None else config)
    if n_jobs < 1 or n_jobs > 3:
        raise ValueError("GONG OCB precompute is limited to the dispatch-w4 three CPU threads")
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    catalog = GongOCBCatalog.__new__(GongOCBCatalog)
    catalog.manifest_path = manifest_path
    catalog.config = config
    catalog.manifest = manifest
    catalog.manifest_sha256 = sha256_file(manifest_path)
    catalog.code_sha256 = _bundle_hash()
    catalog.config_sha256 = _config_hash(config)
    catalog.cache_dir = Path(config["cache_dir"])
    catalog.cache_dir.mkdir(parents=True, exist_ok=True)
    catalog.origin_start = pd.Timestamp(config["origin_feature_start_utc"])
    catalog.origin_end = pd.Timestamp(config["origin_feature_end_utc"])
    catalog.feature_key = _file_digest(json.dumps({
        "manifest_sha256": catalog.manifest_sha256,
        "code_sha256": catalog.code_sha256,
        "config_sha256": catalog.config_sha256,
        "origin_start": str(catalog.origin_start),
        "origin_end": str(catalog.origin_end),
        "cadence": "1h",
    }, sort_keys=True).encode())
    catalog.feature_path = catalog.cache_dir / f"origin-features-{catalog.feature_key}.parquet"
    catalog.feature_receipt_path = catalog.cache_dir / f"origin-features-{catalog.feature_key}.json"
    if catalog.feature_path.is_file() and catalog.feature_receipt_path.is_file():
        old = json.loads(catalog.feature_receipt_path.read_text())
        if (old.get("feature_key") == catalog.feature_key
                and old.get("feature_table_sha256") == sha256_file(catalog.feature_path)):
            print(json.dumps({"event": "origin_feature_cache_reuse", "receipt": str(catalog.feature_receipt_path),
                              "origins": old.get("origin_count"), "maps": old.get("unique_selected_maps")}), flush=True)
            return old

    catalog.rows = []
    for raw in manifest["maps"]:
        row = dict(raw)
        row["observation_ns"] = _timestamp_ns(row["observation_time_utc"])
        row["available_ns"] = _availability_ns(row)
        catalog.rows.append(row)
    catalog.rows.sort(key=lambda item: (item["observation_ns"], item["file"]))
    if np.any(np.diff([row["observation_ns"] for row in catalog.rows]) <= 0):
        raise ValueError("GONG observations must be unique and increasing")

    origins = pd.date_range(catalog.origin_start, catalog.origin_end, freq="h").as_unit("ns")
    choices = _origin_map_indices(origins, catalog.rows)
    feature_values = np.full((len(origins), len(OCB_COLUMNS)), np.nan, dtype=np.float64)
    feature_values[:, -1] = 1.0
    source_files = np.full(len(origins), None, dtype=object)
    source_hashes = np.full(len(origins), None, dtype=object)
    available_at = np.full(len(origins), None, dtype=object)
    groups = {}
    for index, map_id in enumerate(choices):
        if map_id >= 0:
            groups.setdefault(int(map_id), []).append(index)
    tasks = []
    for map_id, positions in groups.items():
        row = catalog.rows[map_id]
        tasks.append({
            "map_id": map_id,
            "row": row,
            "origins": [origins[position].isoformat().replace("+00:00", "Z") for position in positions],
            "config": config,
            "code_sha256": catalog.code_sha256,
            "config_sha256": catalog.config_sha256,
            "cache_dir": str(catalog.cache_dir),
        })

    fallback_count = 0
    complete_maps = 0
    with ProcessPoolExecutor(max_workers=min(int(n_jobs), 3)) as pool:
        futures = [pool.submit(_worker_map_task, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            complete_maps += 1
            fallback_count += int(result["fallback_count"])
            row = catalog.rows[int(result["map_id"])]
            source = row["file"]
            for position, origin_ns, values in zip(
                    groups[int(result["map_id"])], result["origins_ns"], result["features"]):
                if int(origins[position].value) != int(origin_ns):
                    raise ValueError("GONG OCB map-worker origin ordering mismatch")
                feature_values[position] = values
                source_files[position] = source
                source_hashes[position] = row["sha256"]
                available_at[position] = row["available_at_utc"]
            if complete_maps % 50 == 0 or complete_maps == len(tasks):
                print(json.dumps({"event": "ocb_map_progress", "maps": complete_maps,
                                  "total": len(tasks), "fallbacks": fallback_count}), flush=True)

    frame = pd.DataFrame(feature_values, columns=list(OCB_COLUMNS))
    frame.insert(0, "available_at_utc", available_at)
    frame.insert(0, "source_sha256", source_hashes)
    frame.insert(0, "source_file", source_files)
    frame.insert(0, "origin_ns", _datetime_ns(origins))
    temporary = catalog.feature_path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(catalog.feature_path)
    covered = choices >= 0
    ages = np.asarray([
        (origins[position].value - catalog.rows[int(map_id)]["observation_ns"]) / HOUR_NS
        for position, map_id in enumerate(choices) if map_id >= 0
    ], dtype=float)
    receipt = {
        "feature_key": catalog.feature_key,
        "manifest_sha256": catalog.manifest_sha256,
        "code_sha256": catalog.code_sha256,
        "config_sha256": catalog.config_sha256,
        "feature_table": str(catalog.feature_path),
        "feature_table_sha256": sha256_file(catalog.feature_path),
        "source_manifest": str(manifest_path),
        "source_map_count": len(catalog.rows),
        "unique_selected_maps": len(groups),
        "source_hash_count": len({catalog.rows[index]["sha256"] for index in groups}),
        "origin_count": len(origins),
        "source_covered_origins": int(covered.sum()),
        "source_coverage_fraction": float(covered.mean()),
        "topology_missing_origins": int(np.sum(feature_values[:, -1] >= 1.0)),
        "topology_missing_fraction": float(np.mean(feature_values[:, -1] >= 1.0)),
        "topology_finite_fraction": float(np.mean(feature_values[:, -1] < 1.0)),
        "age_hours_min_median_p95_max": {
            "min": float(np.min(ages)) if ages.size else None,
            "median": float(np.median(ages)) if ages.size else None,
            "p95": float(np.quantile(ages, 0.95)) if ages.size else None,
            "max": float(np.max(ages)) if ages.size else None,
        },
        "trace_fallback_count": int(fallback_count),
        "pfss": {
            "source_surface_radius_solar_radii": config["source_surface_radius_solar_radii"],
            "pfss_n_r": config["pfss_n_r"],
            "input_grid": [config["input_grid_latitude"], config["input_grid_longitude"]],
        },
        "availability_rule": config["availability_rule"],
        "n_jobs": int(n_jobs),
    }
    temporary_receipt = catalog.feature_receipt_path.with_suffix(".json.tmp")
    temporary_receipt.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    temporary_receipt.replace(catalog.feature_receipt_path)
    print(json.dumps({"event": "ocb_feature_table_complete", "receipt": receipt}, sort_keys=True), flush=True)
    return receipt
