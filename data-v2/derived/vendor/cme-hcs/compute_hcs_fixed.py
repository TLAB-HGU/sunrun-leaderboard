"""Hash-keyed causal PFSS + Schatten current-sheet geometry cache."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
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
CONFIG_PATH = HERE / "hcs_config.json"
MANIFEST_PATH = HERE / "gong_manifest.json"
CACHE_ROOT = STORE_ROOT / "cmehcs72-hcs-cache"
VENDOR = LB / "experiments/q78_wsa_smoke/vendor"
UTC = timezone.utc


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def timestamp(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    result = datetime.fromisoformat(value)
    return result.replace(tzinfo=UTC) if result.tzinfo is None else result.astimezone(UTC)


def cache_names(code_hash: str, config_hash: str, source_manifest_hash: str):
    namespace = sha256((code_hash + config_hash + source_manifest_hash).encode())
    return namespace


def _crossings(signed_field, lat_axis):
    rows, columns = signed_field.shape
    all_crossings = []
    for x in range(columns):
        column = signed_field[:, x]
        values = []
        for y in range(rows - 1):
            a, b = float(column[y]), float(column[y + 1])
            if not (math.isfinite(a) and math.isfinite(b)):
                continue
            if a == 0.0 and b == 0.0:
                continue
            if a * b > 0.0:
                continue
            fraction = abs(a) / (abs(a) + abs(b)) if abs(a) + abs(b) else 0.5
            values.append(float(lat_axis[y] + fraction * (lat_axis[y + 1] - lat_axis[y])))
        values.sort()
        deduped = []
        for value in values:
            if not deduped or abs(value - deduped[-1]) > 1e-5:
                deduped.append(value)
        all_crossings.append(deduped)
    max_crossings = max((len(values) for values in all_crossings), default=0)
    padded = [[float("nan")] * max_crossings for _ in all_crossings]
    counts = []
    for x, values in enumerate(all_crossings):
        counts.append(len(values))
        padded[x][:len(values)] = values
    return padded, counts


def _compute_map(record, config, code_hash, config_hash, cache_dir):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    source_path = Path(record["path"])
    body_hash = file_sha256(source_path)
    if body_hash != record["body_sha256"]:
        raise ValueError(f"GONG source SHA256 mismatch: {record['file']}")

    sys.path.insert(0, str(VENDOR))
    import astropy.units as u
    import numpy as np
    import sunpy
    import sunpy.map
    import pfsspy
    from astropy.time import Time

    date_obs = timestamp(record["date_obs_utc"])
    raw_map = sunpy.map.Map(str(source_path))
    raw = np.asarray(raw_map.data, dtype=np.float64)
    nlat = int(config["input_grid_latitude"])
    nlon = int(config["input_grid_longitude"])
    if raw.shape != (nlat * 10, nlon * 10):
        raise ValueError(f"unexpected GONG shape {raw.shape} in {record['file']}")
    br = raw.reshape(nlat, 10, nlon, 10).mean((1, 3))
    if not np.isfinite(br).all():
        raise ValueError(f"nonfinite regridded GONG field in {record['file']}")
    header = pfsspy.utils.carr_cea_wcs_header(
        date_obs, br.T.shape, map_center_longitude=float(raw_map.meta["crval1"]) * u.deg
    )
    photo_map = sunpy.map.Map(br - np.nanmean(br), header)
    pfss = pfsspy.pfss(pfsspy.Input(
        photo_map, int(config["pfss_radial_shells"]),
        float(config["pfss_source_surface_radius_solar_radii"]),
    ))
    inner_surface = np.asarray(pfss.source_surface_br.data, dtype=np.float64)
    if not np.isfinite(inner_surface).all():
        raise ValueError(f"PFSS source surface contains nonfinite values: {record['file']}")

    # WSA/SCS boundary prescription: make the 2.5 R_sun radial input positive,
    # solve the pseudo-potential shell to 5 R_sun, then restore the PFSS sign.
    scs_header = pfsspy.utils.carr_cea_wcs_header(
        date_obs, inner_surface.T.shape,
        map_center_longitude=float(raw_map.meta["crval1"]) * u.deg,
    )
    scs_inner = sunpy.map.Map(np.abs(inner_surface), scs_header)
    scs = pfsspy.pfss(pfsspy.Input(
        scs_inner, int(config["scs_radial_shells"]),
        float(config["scs_outer_radius_solar_radii"])
        / float(config["scs_inner_radius_solar_radii"]),
    ))
    outer_unsigned = np.asarray(scs.source_surface_br.data, dtype=np.float64)
    if outer_unsigned.shape != inner_surface.shape or not np.isfinite(outer_unsigned).all():
        raise ValueError(f"SCS outer surface invalid for {record['file']}")
    outer_signed = np.sign(inner_surface) * np.abs(outer_unsigned)

    outer_map = scs.source_surface_br
    ny, nx = outer_signed.shape
    xpix = np.arange(nx) * u.pix
    ypix = np.arange(ny) * u.pix
    lon_coord = outer_map.pixel_to_world(xpix, np.full(nx, (ny - 1) / 2) * u.pix)
    lat_coord = outer_map.pixel_to_world(np.full(ny, (nx - 1) / 2) * u.pix, ypix)
    lon = np.asarray(lon_coord.lon.to_value(u.deg), dtype=np.float64)
    lat = np.asarray(lat_coord.lat.to_value(u.deg), dtype=np.float64)
    lon_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(lon)))
    if np.nanmedian(np.diff(lon_unwrapped)) < 0:
        lon_unwrapped = lon_unwrapped[::-1]
        outer_signed = outer_signed[:, ::-1]
    if np.nanmedian(np.diff(lat)) < 0:
        lat = lat[::-1]
        outer_signed = outer_signed[::-1, :]
    crossings, counts = _crossings(outer_signed, lat)

    path = cache_dir / f"{body_hash}.npz"
    tmp_path = cache_dir / f"{body_hash}.tmp.npz"
    np.savez_compressed(
        tmp_path,
        longitude_axis_unwrapped_deg=lon_unwrapped.astype(np.float32),
        latitude_axis_deg=lat.astype(np.float32),
        latitude_crossings_deg=np.asarray(crossings, dtype=np.float32),
        crossing_count_by_longitude=np.asarray(counts, dtype=np.int16),
        outer_signed_field=outer_signed.astype(np.float32),
    )
    os.replace(tmp_path, path)
    metadata = {
        "source_file": record["file"], "source_url": record.get("url"),
        # Older hash-pinned GONG records predate the optional byte-count field.
        # The body hash has already been verified, so use the local stat size
        # as the exact byte count for those records.
        "source_sha256": body_hash,
        "source_bytes": record.get("bytes", source_path.stat().st_size),
        "date_obs_utc": record["date_obs_utc"],
        "last_modified": record.get("last_modified"),
        "available_at_utc": record["available_at_utc"],
        "compute_sha256": code_hash, "config_sha256": config_hash,
        "npz_sha256": file_sha256(path), "sunpy_version": sunpy.__version__,
        "pfsspy_version": pfsspy.__version__, "pfss_rss_rsun": 2.5,
        "scs_rin_rsun": 2.5, "scs_rout_rsun": 5.0,
        "scs_boundary": "abs(PFSS Br at 2.5 R_sun), pseudo-potential shell, then restore PFSS polarity",
        "scs_polarity_mapping": "radial angular mapping; no outer-shell field-line trace",
        "outer_grid_shape": list(outer_signed.shape),
        "hcs_contour_sample_count": int(sum(counts)),
        "valid": bool(sum(counts) > 0),
    }
    metadata_path = cache_dir / f"{body_hash}.json"
    metadata_tmp = cache_dir / f"{body_hash}.tmp.json"
    metadata_tmp.write_text(json.dumps(metadata, sort_keys=True, allow_nan=False) + "\n")
    os.replace(metadata_tmp, metadata_path)
    return metadata


def _cache_valid(record, cache_dir, code_hash, config_hash):
    path = cache_dir / f"{record['body_sha256']}.json"
    npz = cache_dir / f"{record['body_sha256']}.npz"
    if not path.is_file() or not npz.is_file():
        return False
    try:
        row = json.loads(path.read_text())
        return (row.get("source_sha256") == record["body_sha256"]
                and row.get("compute_sha256") == code_hash
                and row.get("config_sha256") == config_hash
                and row.get("npz_sha256") == file_sha256(npz))
    except (OSError, ValueError, KeyError):
        return False


def _write_ephemeris(cache_dir, namespace, start, stop):
    import astropy.units as u
    import numpy as np
    from astropy.time import Time
    from sunpy.coordinates.sun import B0, L0
    from sunpy import __version__ as sunpy_version

    start_hour = int(start.timestamp() // 3600) * 3600
    stop_hour = (int(stop.timestamp() // 3600) + 2) * 3600
    unix_seconds = np.arange(start_hour, stop_hour + 1, 3600, dtype=np.int64)
    times = Time(unix_seconds, format="unix", scale="utc")
    l0 = np.asarray(L0(times).to_value(u.deg), dtype=np.float64)
    b0 = np.asarray(B0(times).to_value(u.deg), dtype=np.float64)
    path = cache_dir / "ephemeris.npz"
    tmp = cache_dir / "ephemeris.tmp.npz"
    np.savez_compressed(tmp, unix_seconds=unix_seconds, l0_deg=l0, b0_deg=b0)
    os.replace(tmp, path)
    return {
        "path": str(path), "sha256": file_sha256(path), "sunpy_version": sunpy_version,
        "hours": len(unix_seconds), "start_utc": datetime.fromtimestamp(start_hour, UTC).isoformat(),
        "stop_utc": datetime.fromtimestamp(stop_hour, UTC).isoformat(),
        "namespace": namespace,
    }


def main():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    if not os.environ.get("EXPOS_OUT"):
        raise RuntimeError("compute_hcs.py must run inside expos run")
    out = Path(os.environ["EXPOS_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    manifest_bytes = MANIFEST_PATH.read_bytes()
    manifest = json.loads(manifest_bytes)
    config_bytes = CONFIG_PATH.read_bytes()
    config = json.loads(config_bytes)
    code_hash = file_sha256(Path(__file__))
    config_hash = sha256(config_bytes)
    manifest_hash = sha256(manifest_bytes)
    namespace = cache_names(code_hash, config_hash, manifest_hash)
    cache_dir = CACHE_ROOT / namespace
    cache_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(VENDOR))
    # Fail before dispatching workers if the chosen PFSS environment is broken;
    # an import/configuration failure must never be mistaken for map missingness.
    import astropy.units as u  # noqa: F401
    import pfsspy  # noqa: F401
    import sunpy  # noqa: F401
    import sunpy.map  # noqa: F401
    maps = manifest["maps"]
    if not maps:
        raise ValueError("GONG manifest has no source maps")
    bad_paths = [row["file"] for row in maps if not Path(row["path"]).is_file()]
    if bad_paths:
        raise FileNotFoundError(f"GONG source bodies missing: {bad_paths[:5]}")
    pending = [row for row in maps if not _cache_valid(row, cache_dir, code_hash, config_hash)]
    print(json.dumps({"event": "hcs_cache_start", "total_maps": len(maps),
                      "cached": len(maps) - len(pending), "pending": len(pending),
                      "workers": 4, "cache_dir": str(cache_dir)}, sort_keys=True), flush=True)
    completed = len(maps) - len(pending)
    invalid = []
    if pending:
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_compute_map, row, config, code_hash, config_hash, cache_dir): row
                       for row in pending}
            for future in as_completed(futures):
                row = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    # A source-specific failure remains an explicit missing map;
                    # import/config/runtime infrastructure errors surface globally.
                    result = {
                        "source_file": row["file"], "source_url": row.get("url"),
                        "source_sha256": row["body_sha256"],
                        "compute_sha256": code_hash, "config_sha256": config_hash,
                        "available_at_utc": row["available_at_utc"], "valid": False,
                        "error_type": type(error).__name__, "error": str(error)[:500],
                    }
                    (cache_dir / f"{row['body_sha256']}.json").write_text(
                        json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
                    )
                    invalid.append(result)
                completed += 1
                if completed % 100 == 0 or completed == len(maps):
                    print(json.dumps({"event": "hcs_cache_progress", "complete": completed,
                                      "total": len(maps), "source_failures": len(invalid)}), flush=True)

    # SunPy supplies the time-dependent Stonyhurst/Carrington transform and B0.
    start = timestamp(maps[0]["date_obs_utc"]) - timedelta(days=7)
    stop = timestamp(manifest.get("reach_probe", {}).get("date_obs_utc", maps[-1]["date_obs_utc"])) + timedelta(days=7)
    ephemeris = _write_ephemeris(cache_dir, namespace, start, stop)
    cache_manifest = {
        "namespace": namespace, "compute_sha256": code_hash,
        "config_sha256": config_hash, "source_manifest_sha256": manifest_hash,
        "map_count": len(maps), "valid_count": 0, "invalid_count": 0,
        "ephemeris": ephemeris, "config": config,
        "source_availability_rule": manifest["availability_rule"],
    }
    for row in maps:
        metadata_path = cache_dir / f"{row['body_sha256']}.json"
        if metadata_path.is_file():
            meta = json.loads(metadata_path.read_text())
            cache_manifest["valid_count"] += int(meta.get("valid", False))
            cache_manifest["invalid_count"] += int(not meta.get("valid", False))
    cache_manifest_path = cache_dir / "cache_manifest.json"
    cache_manifest_path.write_text(json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n")
    receipt = {
        "cache_dir": str(cache_dir), "cache_namespace": namespace,
        "compute_sha256": code_hash, "config_sha256": config_hash,
        "source_manifest_sha256": manifest_hash,
        "total_maps": len(maps), "cached_before_run": len(maps) - len(pending),
        "computed_this_run": len(pending), "valid_maps": cache_manifest["valid_count"],
        "invalid_maps": cache_manifest["invalid_count"],
        "invalid_source_examples": invalid[:20], "ephemeris": ephemeris,
        "source": manifest["source"], "source_reach_probe": manifest["reach_probe"],
        "availability_rule": manifest["availability_rule"], "workers": 4,
        "pfss_rss_rsun": 2.5, "scs_rin_rsun": 2.5, "scs_rout_rsun": 5.0,
        "scs_polarity_mapping": "radial angular mapping; no field-line tracing",
    }
    receipt_path = out / "hcs_precompute_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"event": "hcs_cache_ready", "maps": len(maps),
                      "valid": cache_manifest["valid_count"], "invalid": cache_manifest["invalid_count"],
                      "namespace": namespace, "receipt": str(receipt_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
