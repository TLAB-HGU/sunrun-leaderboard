"""Build fixed-Rss PFSS patch statistics, cached by producer/config/source hashes."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import warnings

ROOT = Path('/home/t-lab01/sunrun')
LB = ROOT / "leaderboard"
STORE_ROOT = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store')
HERE = Path(__file__).resolve().parent
VENDOR = LB / "experiments/q78_wsa_smoke/vendor"
CONFIG_PATH = HERE / "pfss_config.json"
MANIFEST_PATH = HERE / "gong_manifest.json"
CACHE_ROOT = STORE_ROOT / "pfssconn-cache"
STATS = (
    "br_signed_mean",
    "br_abs_mean",
    "br_abs_p90",
    "open_field_fraction",
    "expansion_factor_median",
    "expansion_factor_p90",
    "polarity_boundary_distance_deg",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def namespace() -> tuple[str, str, str]:
    config_sha = sha256(CONFIG_PATH.read_bytes())
    code_sha = sha256(Path(__file__).read_bytes())
    cache_namespace = sha256((code_sha + config_sha).encode())
    return code_sha, config_sha, cache_namespace


def finite_json(value):
    number = float(value)
    return number if math.isfinite(number) else None


def angular_distance_deg(lat_deg, lon_offset_deg):
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_offset_deg))
    cosine = math.cos(lat) * math.cos(lon)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def features_for_map(record, config, code_sha, config_sha, cache_dir):
    input_path = Path(record["path"])
    body = input_path.read_bytes()
    actual_sha = sha256(body)
    if actual_sha != record["body_sha256"]:
        raise ValueError(f"GONG source hash mismatch: {record['file']}")

    sys.path.insert(0, str(VENDOR))
    tracer_extensions = list((VENDOR / "streamtracer").glob("_streamtracer_rust*.so"))
    if len(tracer_extensions) != 1:
        raise RuntimeError(f"expected one vendored streamtracer extension, found {len(tracer_extensions)}")
    if sha256(tracer_extensions[0].read_bytes()) != config["streamtracer_extension_sha256"]:
        raise RuntimeError("vendored streamtracer extension SHA256 mismatch")
    import astropy.units as u
    import numpy as np
    import sunpy.map
    from astropy.coordinates import SkyCoord
    from scipy.ndimage import map_coordinates
    from sunpy.coordinates.sun import L0
    import pfsspy
    from pfsspy.tracing import FortranTracer

    date_obs = pd_timestamp(record["date_obs_utc"])
    raw_map = sunpy.map.Map(str(input_path))
    raw = np.asarray(raw_map.data, dtype=np.float64)
    nlat, nlon = config["input_grid_latitude"], config["input_grid_longitude"]
    if raw.shape != (nlat * 10, nlon * 10):
        raise ValueError(f"unexpected GONG map shape {raw.shape}: {record['file']}")
    br = raw.reshape(nlat, 10, nlon, 10).mean((1, 3))
    header = pfsspy.utils.carr_cea_wcs_header(
        date_obs, br.T.shape,
        map_center_longitude=float(raw_map.meta["crval1"]) * u.deg,
    )
    magnetogram = sunpy.map.Map(br - np.nanmean(br), header)
    solution = pfsspy.pfss(pfsspy.Input(
        magnetogram,
        int(config["pfss_n_r"]),
        float(config["source_surface_radius_solar_radii"]),
    ))

    center_lon = float(L0(date_obs).value) % 360.0
    patch_lon = np.asarray(config["subearth_patch_longitude_offsets_deg"], dtype=float)
    patch_lat = np.asarray(config["subearth_patch_latitude_offsets_deg"], dtype=float)
    patch_llon, patch_llat = np.meshgrid(patch_lon, patch_lat)
    ssmap = solution.source_surface_br

    def sample_source_surface(lon_offsets, lat_offsets):
        llon, llat = np.meshgrid(lon_offsets, lat_offsets)
        coords = SkyCoord(
            ((center_lon + llon.ravel()) % 360.0) * u.deg,
            llat.ravel() * u.deg,
            float(config["source_surface_radius_solar_radii"]) * u.R_sun,
            frame=solution.coordinate_frame,
        )
        x, y = ssmap.world_to_pixel(coords)
        width = ssmap.data.shape[1]
        x = np.mod(np.asarray(x, dtype=float), width) + width
        y = np.asarray(y, dtype=float)
        periodic = np.concatenate([ssmap.data, ssmap.data, ssmap.data], axis=1)
        sampled = map_coordinates(np.asarray(periodic, dtype=float), [y, x], order=1, mode="nearest")
        return sampled.reshape(llon.shape)

    source_patch = sample_source_surface(patch_lon, patch_lat)
    finite_br = source_patch[np.isfinite(source_patch)]
    if not len(finite_br):
        raise ValueError(f"no finite source-surface Br values in patch: {record['file']}")

    seed_lon = np.asarray(config["fieldline_seed_longitude_offsets_deg"], dtype=float)
    seed_lat = np.asarray(config["fieldline_seed_latitude_offsets_deg"], dtype=float)
    seed_llon, seed_llat = np.meshgrid(seed_lon, seed_lat)
    seeds = SkyCoord(
        ((center_lon + seed_llon.ravel()) % 360.0) * u.deg,
        seed_llat.ravel() * u.deg,
        1.001 * u.R_sun,
        frame=solution.coordinate_frame,
    )
    tracer = FortranTracer(max_steps=int(config["fieldline_max_steps"]),
                           step_size=float(config["fieldline_step_size"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        lines = tracer.trace(seeds, solution)
    rotations = np.asarray(tracer.tracer.ROT)
    trace_exhausted = int(np.count_nonzero(rotations == 1))
    trace_invalid_directions = int(np.count_nonzero(rotations != 2))
    rss = float(config["source_surface_radius_solar_radii"])
    endpoint_failed_paths = 0
    for line in lines:
        radii = np.asarray(line._r, dtype=float)
        endpoint_failed = (len(radii) < 2 or not np.isfinite(radii).all()
                           or any(min(abs(float(r) - 1.0), abs(float(r) - rss)) >= 0.05
                                  for r in (radii[0], radii[-1])))
        endpoint_failed_paths += int(endpoint_failed)
    trace_complete = (len(lines) == len(seeds) and trace_invalid_directions == 0
                      and endpoint_failed_paths == 0)
    opened = np.asarray([line.is_open for line in lines], dtype=bool)
    factors = (np.asarray([line.expansion_factor for line in lines if line.is_open], dtype=float)
               if trace_complete else np.asarray([], dtype=float))
    if trace_complete:
        factors = factors[np.isfinite(factors) & (factors > 0)]

    # Polarity-boundary distance: nearest zero crossing among adjacent samples
    # on the same fixed +/-30-degree Carrington connection patch.
    boundary_lon = patch_lon
    boundary_lat = patch_lat
    boundary_br = sample_source_surface(boundary_lon, boundary_lat)
    distances = []
    for i in range(boundary_br.shape[0]):
        for j in range(boundary_br.shape[1] - 1):
            a, b = float(boundary_br[i, j]), float(boundary_br[i, j + 1])
            if math.isfinite(a) and math.isfinite(b) and a * b <= 0:
                frac = abs(a) / (abs(a) + abs(b)) if abs(a) + abs(b) else 0.5
                lon = boundary_lon[j] + frac * (boundary_lon[j + 1] - boundary_lon[j])
                distances.append(angular_distance_deg(boundary_lat[i], lon))
    for i in range(boundary_br.shape[0] - 1):
        for j in range(boundary_br.shape[1]):
            a, b = float(boundary_br[i, j]), float(boundary_br[i + 1, j])
            if math.isfinite(a) and math.isfinite(b) and a * b <= 0:
                frac = abs(a) / (abs(a) + abs(b)) if abs(a) + abs(b) else 0.5
                lat = boundary_lat[i] + frac * (boundary_lat[i + 1] - boundary_lat[i])
                distances.append(angular_distance_deg(lat, boundary_lon[j]))
    boundary_cap = angular_distance_deg(float(np.max(np.abs(boundary_lat))),
                                        float(np.max(np.abs(boundary_lon))))

    values = {
        "br_signed_mean": float(np.mean(finite_br)),
        "br_abs_mean": float(np.mean(np.abs(finite_br))),
        "br_abs_p90": float(np.percentile(np.abs(finite_br), 90)),
        "open_field_fraction": float(opened.mean()) if trace_complete else None,
        "expansion_factor_median": finite_json(np.median(factors)) if len(factors) else None,
        "expansion_factor_p90": finite_json(np.percentile(factors, 90)) if len(factors) else None,
        "polarity_boundary_distance_deg": float(min(distances) if distances else boundary_cap),
    }
    if values["open_field_fraction"] is not None and not math.isfinite(values["open_field_fraction"]):
        raise ValueError(f"nonfinite PFSS open fraction: {record['file']}")
    out = {
        "source_file": record["file"],
        "source_sha256": actual_sha,
        "compute_sha256": code_sha,
        "config_sha256": config_sha,
        "source_version": record.get("source_version", "GONG mrzqs ZQS"),
        "date_obs_utc": record["date_obs_utc"],
        "last_modified": record["last_modified"],
        "available_at_utc": record["available_at_utc"],
        "pfsspy_version": pfsspy.__version__,
        "fieldline_tracer": config["fieldline_tracer"],
        "fieldline_step_size": config["fieldline_step_size"],
        "fieldline_max_steps": config["fieldline_max_steps"],
        "fieldline_trace_complete": trace_complete,
        "fieldline_trace_exhausted_paths": trace_exhausted,
        "fieldline_trace_invalid_directions": trace_invalid_directions,
        "fieldline_endpoint_failed_paths": endpoint_failed_paths,
        "fieldline_returned_paths": len(lines),
        "features": values,
    }
    cache_path = cache_dir / f"{actual_sha}.json"
    temp = cache_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(out, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(cache_path)
    return cache_path


def pd_timestamp(value):
    from datetime import datetime
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def cache_valid(path, record, code_sha, config_sha):
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text())
        return (row.get("source_sha256") == record["body_sha256"]
                and row.get("compute_sha256") == code_sha
                and row.get("config_sha256") == config_sha
                and row.get("source_file") == record["file"])
    except (OSError, ValueError, KeyError):
        return False


def main():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    config = json.loads(CONFIG_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())
    code_sha, config_sha, cache_namespace = namespace()
    cache_dir = CACHE_ROOT / cache_namespace
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for row in manifest["maps"]:
        cache_path = cache_dir / f"{row['body_sha256']}.json"
        if not cache_valid(cache_path, row, code_sha, config_sha):
            missing.append(row)
    total = len(manifest["maps"])
    print(json.dumps({"event": "pfss_cache_start", "maps": total, "cached": total - len(missing),
                      "pending": len(missing), "workers": 4, "cache": str(cache_dir)}, sort_keys=True), flush=True)
    complete = total - len(missing)
    if missing:
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(features_for_map, row, config, code_sha, config_sha, cache_dir)
                       for row in missing]
            for future in as_completed(futures):
                future.result()
                complete += 1
                if complete % 50 == 0 or complete == total:
                    print(json.dumps({"event": "pfss_cache_progress", "complete": complete, "total": total}), flush=True)

    trace_summary = {"fieldline_trace_complete_maps": 0, "fieldline_trace_incomplete_maps": 0,
                     "fieldline_trace_exhausted_paths": 0, "fieldline_trace_invalid_directions": 0,
                     "fieldline_endpoint_failed_paths": 0}
    for row in manifest["maps"]:
        cached_row = json.loads((cache_dir / f"{row['body_sha256']}.json").read_text())
        trace_summary["fieldline_trace_complete_maps" if cached_row["fieldline_trace_complete"]
                      else "fieldline_trace_incomplete_maps"] += 1
        for key in ("fieldline_trace_exhausted_paths", "fieldline_trace_invalid_directions",
                    "fieldline_endpoint_failed_paths"):
            trace_summary[key] += int(cached_row.get(key, 0))
    summary = {
        **trace_summary,
        "source": manifest["source"],
        "manifest": str(MANIFEST_PATH),
        "manifest_sha256": sha256(MANIFEST_PATH.read_bytes()),
        "config_sha256": config_sha,
        "compute_sha256": code_sha,
        "cache_namespace": cache_namespace,
        "total_maps": total,
        "cached_before_run": total - len(missing),
        "computed_this_run": len(missing),
        "workers": 4,
        "pfsspy_version": "1.2.0 (q78 vendored source)",
    }
    out_dir = Path(os.environ.get("EXPOS_OUT", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gong_source_manifest.json").write_bytes(MANIFEST_PATH.read_bytes())
    out = out_dir / "pfss_precompute_receipt.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
