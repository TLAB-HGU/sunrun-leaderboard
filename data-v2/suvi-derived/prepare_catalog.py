"""Bind staged GOES-18 SUVI Fe195 NetCDF frames to public S3 provenance."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import hashlib
import http.client
import json
import re
import threading
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "sun-img/data/g18"
STORE = ROOT / "data-v2/model-inputs/leaderboard-store/solar-inc-euvwave72-20261014"
HOST = "noaa-goes18.s3.amazonaws.com"
NAME = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)_e"
                  r"(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)_c")
LOCAL = threading.local()


def connection():
    if not hasattr(LOCAL, "conn"):
        LOCAL.conn = http.client.HTTPSConnection(HOST, timeout=40)
    return LOCAL.conn


def sha_file(path: Path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hashes(path: Path):
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def stamp(parts):
    year, doy, hour, minute, second, tenth = map(int, parts)
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=doy - 1, hours=hour, minutes=minute, seconds=second, milliseconds=tenth * 100
    )


def inspect(path: Path):
    match = NAME.search(path.name)
    if not match:
        raise ValueError(f"unrecognized SUVI filename: {path.name}")
    parts = match.groups()
    obs_start = stamp(parts[:6])
    obs_end = stamp(parts[6:])
    key = (f"SUVI-L1b-Fe195/{obs_start:%Y}/{obs_start.timetuple().tm_yday:03d}/"
           f"{obs_start:%H}/{path.name}")
    last_error = None
    for attempt in range(5):
        try:
            conn = connection()
            conn.request("HEAD", "/" + key)
            response = conn.getresponse()
            response.read()
            if response.status == 404:
                return {
                    "filename": path.name, "local_path": str(path), "s3_key": key,
                    "obs_start_utc": obs_start.isoformat(), "obs_end_utc": obs_end.isoformat(),
                    "http_status": 404, "byte_identity_passed": False,
                }
            if response.status != 200:
                raise RuntimeError(f"S3 HEAD returned {response.status} for {key}")
            etag = (response.getheader("ETag") or "").strip().strip('"').lower()
            last_modified = response.getheader("Last-Modified")
            if not last_modified:
                raise RuntimeError(f"S3 object has no Last-Modified: {key}")
            source_time = datetime.strptime(last_modified, "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=timezone.utc)
            source_size = int(response.getheader("Content-Length"))
            local_stat = path.stat()
            local_md5, local_sha256 = file_hashes(path)
            simple_md5 = bool(re.fullmatch(r"[0-9a-f]{32}", etag))
            identical = bool(simple_md5 and local_stat.st_size == source_size and local_md5 == etag)
            return {
                "filename": path.name,
                "local_path": str(path),
                "local_size": local_stat.st_size,
                "local_mtime_ns": local_stat.st_mtime_ns,
                "local_md5": local_md5,
                "local_sha256": local_sha256,
                "s3_key": key,
                "s3_url": f"https://{HOST}/{key}",
                "s3_etag": etag,
                "s3_etag_is_singlepart_md5": simple_md5,
                "s3_size": source_size,
                "s3_last_modified_utc": source_time.isoformat(),
                "obs_start_utc": obs_start.isoformat(),
                "obs_end_utc": obs_end.isoformat(),
                "byte_identity_passed": identical,
                "http_status": response.status,
            }
        except Exception as exc:
            last_error = exc
            conn = getattr(LOCAL, "conn", None)
            if conn is not None:
                conn.close()
                del LOCAL.conn
            if attempt < 4:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"failed to inspect {key}: {last_error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be in [1, 4]")
    paths = sorted(DATA.glob("**/*Fe195*.nc"))
    if not paths:
        raise RuntimeError(f"no staged Fe195 NetCDF files in {DATA}")
    STORE.mkdir(parents=True, exist_ok=True)
    rows = [None] * len(paths)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(inspect, path): i for i, path in enumerate(paths)}
        for done, future in enumerate(as_completed(futures), 1):
            rows[futures[future]] = future.result()
            if done % 1000 == 0 or done == len(paths):
                print(f"provenance {done}/{len(paths)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    catalog_path = STORE / "source_catalog.jsonl"
    temp = catalog_path.with_suffix(".jsonl.tmp")
    with temp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temp.replace(catalog_path)
    catalog_sha = sha_file(catalog_path)
    valid = [row for row in rows if row.get("byte_identity_passed")]
    report = {
        "study_id": "solar-inc-euvwave72-20261014",
        "source": "NOAA GOES-18 public S3 SUVI-L1b-Fe195 NetCDF objects",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_root": str(DATA),
        "catalog_path": str(catalog_path),
        "catalog_sha256": catalog_sha,
        "local_file_count": len(rows),
        "head_ok_count": sum(row.get("http_status") == 200 for row in rows),
        "byte_identity_count": len(valid),
        "byte_identity_mismatch_count": len(rows) - len(valid),
        "first_obs_end_utc": min(row["obs_end_utc"] for row in rows if row.get("obs_end_utc")),
        "last_obs_end_utc": max(row["obs_end_utc"] for row in rows if row.get("obs_end_utc")),
        "last_s3_modified_utc": max((row["s3_last_modified_utc"] for row in valid), default=None),
        "etag_rule": "Only a single-part 32-hex S3 ETag equal to the local MD5 and equal Content-Length is admitted.",
        "cadence_note": "Existing staged path contains selected hourly representative frames, not the full 10-minute series.",
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    report_path = STORE / "source_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
