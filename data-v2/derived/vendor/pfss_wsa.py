"""Build hash-keyed, single-Rss PFSS/WSA longitude profiles from GONG receipts.

The mesh, tracer, field-line expansion factor, and closed-footpoint distance
follow the q78 PFSS implementation. No target labels or forecast outputs enter
this extraction.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

import numpy as np

ROOT = Path('/home/t-lab01/sunrun')
LB = ROOT / "leaderboard"
STORE_ROOT = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store')
VENDOR = LB / "experiments/q78_wsa_smoke/vendor"
SOURCE_RECEIPT = STORE_ROOT / "scratch/worker-t2gr/collect_receipt.json"
CACHE_SCHEMA = "solar-inc-pfss-wsa-v1"
RSS = 2.5
NR = 15
LON_STEP_DEG = 10
PHOTO_LAT_DEG = (-10.0, 0.0, 10.0)
FS_MIN, FS_MAX = 1.0, 300.0
THETA_MAX = 30.0
WSA_V0, WSA_DV, WSA_FS_EXP, WSA_TB_SCALE = 267.5, 410.0, -0.4, 2.0
V_MIN, V_MAX = 240.0, 800.0
# This pins the unchanged map-to-profile calculation used by E1144. Scheduling
# and timeout handling may change without invalidating successful profile data.
PROFILE_CODE_SHA256 = "5cb9bf9ecdee1120b6ea8fd63b52001d296cd5ec4d4bfb94a323e3bb80f591d6"
TRACE_TIMEOUT_SECONDS = 15
_TRACING_READY = False


def _initialize_tracing() -> None:
    global _TRACING_READY
    if _TRACING_READY:
        return
    sys.path.insert(0, str(VENDOR))
    import astropy.units  # noqa: F401
    import sunpy.map  # noqa: F401
    import pfsspy  # noqa: F401
    _TRACING_READY = True


def _profile_child(record: dict, output_path: str, error_path: str) -> None:
    try:
        profile = map_profiles(record)
        with Path(output_path).open("wb") as stream:
            np.savez_compressed(stream, **profile)
    except BaseException:
        Path(error_path).write_text(traceback.format_exc())
        raise


def _isolated_profile(record: dict, cache_dir: Path, key: str) -> dict | None:
    """Run one native PFSS trace in a killable child; return None on timeout."""
    with tempfile.TemporaryDirectory(prefix=f"pfss-{key[:12]}-", dir=cache_dir) as work:
        output_path, error_path = Path(work) / "profile.npz", Path(work) / "error.txt"
        process = mp.get_context("fork").Process(
            target=_profile_child, args=(record, str(output_path), str(error_path)))
        process.start()
        process.join(TRACE_TIMEOUT_SECONDS)
        if process.is_alive():
            process.terminate()
            process.join(2)
            if process.is_alive():
                process.kill()
                process.join(2)
            if process.is_alive():
                raise RuntimeError(f"{record['file']}: timed-out PFSS child could not be stopped")
            process.close()
            return None
        exitcode = process.exitcode
        process.close()
        if exitcode != 0:
            detail = error_path.read_text() if error_path.exists() else "no child traceback"
            raise RuntimeError(f"{record['file']}: PFSS child exit={exitcode}: {detail}")
        with np.load(output_path, allow_pickle=False) as data:
            profile = {name: np.asarray(data[name]) for name in
                       ("longitude_deg", "fs", "theta_b_deg", "v_wsa_kms")}
            profile["open_source_count"] = int(data["open_source_count"].item())
            profile["open_photosphere_count"] = int(data["open_photosphere_count"].item())
            return profile


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def vendor_hash() -> str:
    rows = []
    for path in sorted(VENDOR.rglob("*.py")):
        rows.append((str(path.relative_to(VENDOR)), sha256(path)))
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def cache_identity(record: dict, extractor_sha: str, pfss_vendor_sha: str) -> tuple[str, dict]:
    body = {
        "schema": CACHE_SCHEMA,
        "body_sha256": record["body_sha256"],
        "extractor_sha256": extractor_sha,
        "pfss_vendor_sha256": pfss_vendor_sha,
        "rss": RSS,
        "nr": NR,
        "longitude_grid_deg": list(range(0, 360, LON_STEP_DEG)),
        "photosphere_latitude_seeds_deg": list(PHOTO_LAT_DEG),
        "boundary_distance": "nearest traced closed photospheric seed, great-circle degrees",
        "wsa": [WSA_V0, WSA_DV, WSA_FS_EXP, WSA_TB_SCALE, V_MIN, V_MAX],
    }
    key = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return key, body


def _endpoint_ok(line) -> bool:
    if len(line) <= 1:
        return False
    r = np.asarray(line._r, dtype=float)
    return (min(abs(r[0] - 1.0), abs(r[0] - RSS)) < 0.05
            and min(abs(r[-1] - 1.0), abs(r[-1] - RSS)) < 0.05)


def _great_circle_deg(a_lon, a_lat, b_lon, b_lat) -> np.ndarray:
    a_lon = np.deg2rad(float(a_lon))
    a_lat = np.deg2rad(float(a_lat))
    b_lon = np.deg2rad(np.asarray(b_lon, dtype=float))
    b_lat = np.deg2rad(np.asarray(b_lat, dtype=float))
    cos_d = np.sin(a_lat) * np.sin(b_lat) + np.cos(a_lat) * np.cos(b_lat) * np.cos(a_lon - b_lon)
    return np.rad2deg(np.arccos(np.clip(cos_d, -1.0, 1.0)))


def wsa_speed(fs: np.ndarray, theta_b: np.ndarray) -> np.ndarray:
    fs = np.clip(np.asarray(fs, dtype=float), FS_MIN, FS_MAX)
    theta_b = np.clip(np.asarray(theta_b, dtype=float), 0.0, THETA_MAX)
    speed = WSA_V0 + WSA_DV * np.power(fs, WSA_FS_EXP) * np.power(
        1.0 - 0.8 * np.exp(-np.square(theta_b / WSA_TB_SCALE)), 3.0)
    return np.clip(speed, V_MIN, V_MAX)


def map_profiles(record: dict) -> dict:
    sys.path.insert(0, str(VENDOR))
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    import sunpy.map
    import pfsspy
    from pfsspy.tracing import PythonTracer

    path = Path(record["path"])
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"{record['file']}: byte length differs from receipt")
    if sha256(path) != record["body_sha256"]:
        raise ValueError(f"{record['file']}: body SHA-256 differs from receipt")

    source_map = sunpy.map.Map(str(path))
    raw = np.asarray(source_map.data, dtype=float)
    if raw.size != 180 * 360:
        raise ValueError(f"{record['file']}: expected 180x360 GONG map, got {raw.shape}")
    coarse = raw.reshape(18, 10, 36, 10).mean((1, 3))
    header = pfsspy.utils.carr_cea_wcs_header(
        record["date_obs_utc"], coarse.T.shape,
        map_center_longitude=float(record["crval1"]) * u.deg)
    gong_map = sunpy.map.Map(coarse - coarse.mean(), header)
    solution = pfsspy.pfss(pfsspy.Input(gong_map, NR, RSS))
    tracer = PythonTracer(atol=1e-4, rtol=1e-4)

    longitude = np.arange(0.0, 360.0, LON_STEP_DEG)
    photo_lon = np.tile(longitude, len(PHOTO_LAT_DEG))
    photo_lat = np.repeat(np.asarray(PHOTO_LAT_DEG), len(longitude))
    photo_seeds = SkyCoord(photo_lon * u.deg, photo_lat * u.deg,
                           1.001 * u.R_sun, frame=solution.coordinate_frame)
    photo_lines = tracer.trace(photo_seeds, solution)
    if not all(_endpoint_ok(line) for line in photo_lines):
        raise ValueError(f"{record['file']}: photospheric trace endpoint failed")
    photo_open = np.asarray([line.is_open for line in photo_lines], dtype=bool)
    closed = photo_seeds[~photo_open]

    source_seeds = SkyCoord(longitude * u.deg, np.zeros(len(longitude)) * u.deg,
                            (RSS - 0.001) * u.R_sun, frame=solution.coordinate_frame)
    source_lines = tracer.trace(source_seeds, solution)
    if not all(_endpoint_ok(line) for line in source_lines):
        raise ValueError(f"{record['file']}: source-surface trace endpoint failed")

    fs = np.full(len(longitude), np.nan, dtype=np.float64)
    theta = np.full(len(longitude), np.nan, dtype=np.float64)
    for i, line in enumerate(source_lines):
        if not line.is_open or not np.isfinite(line.expansion_factor):
            continue
        foot = line.solar_footpoint
        expansion = float(np.clip(line.expansion_factor, FS_MIN, FS_MAX))
        if len(closed):
            closed_lon = closed.lon.to_value(u.deg)
            closed_lat = closed.lat.to_value(u.deg)
            boundary = float(_great_circle_deg(foot.lon.to_value(u.deg),
                                               foot.lat.to_value(u.deg),
                                               closed_lon, closed_lat).min())
        else:
            boundary = THETA_MAX
        fs[i] = expansion
        theta[i] = float(np.clip(boundary, 0.0, THETA_MAX))
    valid = np.isfinite(fs) & np.isfinite(theta)
    if int(valid.sum()) < 6:
        raise ValueError(f"{record['file']}: only {int(valid.sum())}/36 usable open-source profiles")
    speed = np.full(len(longitude), np.nan, dtype=np.float64)
    speed[valid] = wsa_speed(fs[valid], theta[valid])
    return {"longitude_deg": longitude, "fs": fs, "theta_b_deg": theta,
            "v_wsa_kms": speed, "open_source_count": int(valid.sum()),
            "open_photosphere_count": int(photo_open.sum())}


def _task(args: tuple) -> dict:
    record, cache_dir, extractor_sha, pfss_vendor_sha = args
    key, identity = cache_identity(record, extractor_sha, pfss_vendor_sha)
    success_path = Path(cache_dir) / f"map_{key}.npz"
    timeout_identity = {**identity, "profile_status": "timed_out",
                        "timeout_seconds": TRACE_TIMEOUT_SECONDS}
    timeout_key = hashlib.sha256(json.dumps(timeout_identity, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
    timeout_path = Path(cache_dir) / f"map_{timeout_key}.npz"
    for path, expected_key, timed_out in ((success_path, key, False),
                                           (timeout_path, timeout_key, True)):
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                if (str(data["cache_key"].item()) != expected_key
                        or str(data["body_sha256"].item()) != record["body_sha256"]):
                    raise ValueError(f"{record['file']}: incompatible cached PFSS profile")
                return {"file": record["file"], "cache_key": expected_key,
                        "cache_path": str(path), "body_sha256": record["body_sha256"],
                        "cached": True, "timed_out": timed_out,
                        "open_source_count": int(data["open_source_count"]),
                        "open_photosphere_count": int(data["open_photosphere_count"])}
    _initialize_tracing()
    profile = _isolated_profile(record, Path(cache_dir), key)
    timed_out = profile is None
    if timed_out:
        profile = {"longitude_deg": np.arange(0.0, 360.0, LON_STEP_DEG),
                   "fs": np.full(36, np.nan), "theta_b_deg": np.full(36, np.nan),
                   "v_wsa_kms": np.full(36, np.nan), "open_source_count": 0,
                   "open_photosphere_count": 0}
        path, expected_key = timeout_path, timeout_key
    else:
        path, expected_key = success_path, key
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, cache_key=expected_key, body_sha256=record["body_sha256"],
                            identity=json.dumps(timeout_identity if timed_out else identity,
                                                sort_keys=True), **profile)
    os.replace(tmp, path)
    return {"file": record["file"], "cache_key": expected_key, "cache_path": str(path),
            "body_sha256": record["body_sha256"], "cached": False, "timed_out": timed_out,
            "open_source_count": profile["open_source_count"],
            "open_photosphere_count": profile["open_photosphere_count"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, default=SOURCE_RECEIPT)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=range(1, 4), required=True)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    receipt = json.loads(args.receipt.read_text())
    usable = {row["file"] for row in receipt["proxy"] if row.get("usable_for_features")}
    files = [row for row in receipt["files"] if row["file"] in usable
             and row.get("header_ok") and row.get("content_length_ok")
             and row.get("gunzip_simple_ok")]
    if not files:
        raise RuntimeError("GONG receipt contains no usable hash-bound maps")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    extractor_sha = PROFILE_CODE_SHA256
    pfss_vendor_sha = vendor_hash()
    tasks = [(row, str(args.cache_dir), extractor_sha, pfss_vendor_sha) for row in files]
    started = time.monotonic()
    rows = []
    if args.jobs == 1:
        for i, task in enumerate(tasks, 1):
            rows.append(_task(task))
            if i % 10 == 0 or i == len(tasks):
                print(f"PFSS profiles {i}/{len(tasks)}; elapsed={time.monotonic()-started:.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for i, result in enumerate(pool.map(_task, tasks, chunksize=4), 1):
                rows.append(result)
                if i % 10 == 0 or i == len(tasks):
                    print(f"PFSS profiles {i}/{len(tasks)}; elapsed={time.monotonic()-started:.1f}s", flush=True)
    manifest = {
        "schema": CACHE_SCHEMA,
        "source_receipt": str(args.receipt.resolve()),
        "source_receipt_sha256": sha256(args.receipt),
        "extractor_sha256": extractor_sha,
        "extractor_runtime_sha256": sha256(Path(__file__)),
        "pfss_vendor_sha256": pfss_vendor_sha,
        "trace_timeout_seconds": TRACE_TIMEOUT_SECONDS,
        "rss_rsun": RSS,
        "radial_shells": NR,
        "longitude_step_deg": LON_STEP_DEG,
        "wsa_constants": {"v0": WSA_V0, "dv": WSA_DV, "fs_exponent": WSA_FS_EXP,
                          "theta_scale_deg": WSA_TB_SCALE, "v_min": V_MIN, "v_max": V_MAX},
        "map_count": len(rows),
        "cache_hits": sum(bool(row["cached"]) for row in rows),
        "cache_created": sum(not bool(row["cached"]) for row in rows),
        "timed_out_maps": sum(bool(row.get("timed_out")) for row in rows),
        "elapsed_seconds": time.monotonic() - started,
        "maps": rows,
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ("map_count", "cache_hits", "cache_created", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
