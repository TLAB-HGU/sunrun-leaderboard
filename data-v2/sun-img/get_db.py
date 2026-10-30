#!/usr/bin/env python3
"""Collect hourly GOES-18 SUVI Fe195 records and their selected NetCDF images.

Run ``python get_db.py`` to collect every UTC slot from 2022-08-22 21:00 through
2026-09-30 00:00, inclusive. Results go to get_db_output/ by default. Re-running
fills missing database rows and images. Only ``duckdb`` is required externally.
"""

import http.client
import re
import sys
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta
from pathlib import Path



HERE = Path(__file__).resolve().parent
START = datetime(2022, 8, 22, 21)
END = datetime(2026, 9, 30, 0)  # inclusive UTC slot
HOUR = timedelta(hours=1)
SAT = 18
CHANNEL = "Fe195"
MARGIN = timedelta(seconds=2)
DUR_TOLERANCE = 0.15
S3_HOST = "noaa-goes18.s3.amazonaws.com"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
FITS_BLOCK = 2880
NAME_RE = re.compile(
    r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)_e"
    r"(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)_c"
)
COLS = (
    "slot", "sat", "channel", "status", "filename", "s3_key", "obs_start",
    "obs_end", "age_min", "s3_modified", "exptime", "cmd_exp", "crpix1",
    "crpix2", "crota", "rsun", "solar_b0", "yaw_flip", "eclipse", "filter1",
    "filter2", "n_frames", "n_headers", "cutoff_applied", "nc_path", "nc_size",
)
SCHEMA = """
CREATE TABLE IF NOT EXISTS suvi_hourly (
    slot TIMESTAMP PRIMARY KEY,
    sat TEXT, channel TEXT, status TEXT,
    filename TEXT, s3_key TEXT,
    obs_start TIMESTAMP, obs_end TIMESTAMP, age_min REAL,
    s3_modified TIMESTAMP,
    exptime REAL, cmd_exp REAL, crpix1 REAL, crpix2 REAL, crota REAL,
    rsun REAL, solar_b0 REAL, yaw_flip SMALLINT, eclipse SMALLINT,
    filter1 TEXT, filter2 TEXT,
    n_frames INTEGER, n_headers INTEGER, cutoff_applied BOOLEAN,
    nc_path TEXT, nc_size BIGINT
)
"""
_local = threading.local()


def connection():
    """Keep one HTTPS connection per worker thread."""
    if not hasattr(_local, "connection"):
        _local.connection = http.client.HTTPSConnection(S3_HOST, timeout=30)
    return _local.connection


def reset_connection():
    conn = getattr(_local, "connection", None)
    if conn is not None:
        conn.close()
        del _local.connection


def request(path, byte_range=None, tries=5):
    """Read a small S3 response; a missing object returns None."""
    headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {}
    for attempt in range(tries):
        try:
            conn = connection()
            conn.request("GET", path, headers=headers)
            response = conn.getresponse()
            body = response.read()
            if response.status == 404:
                return None
            if response.status in (200, 206):
                return body
            error = RuntimeError(f"HTTP {response.status}: {path}")
        except Exception as exc:
            error = exc
        reset_connection()
        if attempt == tries - 1:
            raise error
        time.sleep(1.5**attempt)


def list_hour(slot):
    """List FITS frames in the hour preceding a UTC slot."""
    hour = slot - HOUR
    prefix = f"SUVI-L1b-{CHANNEL}/{hour:%Y}/{hour.timetuple().tm_yday:03d}/{hour:%H}/"
    frames = []
    token = None
    while True:
        query = {"list-type": "2", "prefix": prefix}
        if token:
            query["continuation-token"] = token
        raw = request("/?" + urllib.parse.urlencode(query))
        if raw is None:
            raise RuntimeError(f"S3 listing missing: {prefix}")
        root = ET.fromstring(raw)
        for item in root.findall(f"{S3_NS}Contents"):
            key = item.findtext(f"{S3_NS}Key")
            if not key or not key.endswith(".fits"):
                continue
            match = NAME_RE.search(key)
            if not match:
                continue
            start, end = stamp(match.groups()[:6]), stamp(match.groups()[6:])
            modified = datetime.strptime(
                item.findtext(f"{S3_NS}LastModified"), "%Y-%m-%dT%H:%M:%S.%fZ"
            )
            frames.append({
                "key": key, "filename": key.rsplit("/", 1)[1],
                "obs_start": start, "obs_end": end,
                "dur": (end - start).total_seconds(), "s3_modified": modified,
            })
        if root.findtext(f"{S3_NS}IsTruncated") != "true":
            return frames
        token = root.findtext(f"{S3_NS}NextContinuationToken")
        if not token:
            raise ValueError(f"S3 listing has no continuation token: {prefix}")


def stamp(parts):
    year, day, hour, minute, second, tenth = map(int, parts)
    return datetime(year, 1, 1) + timedelta(
        days=day - 1, hours=hour, minutes=minute, seconds=second,
        milliseconds=tenth * 100,
    )


def read_header(key):
    """Read only enough FITS blocks to reach the END card."""
    for blocks in (3, 12, 60):
        raw = request("/" + key, (0, blocks * FITS_BLOCK - 1))
        if raw is None:
            raise FileNotFoundError(key)
        cards = {}
        complete = False
        for pos in range(0, len(raw) - 79, 80):
            card = raw[pos:pos + 80].decode("ascii", "replace")
            if card.startswith("END") and not card[3:].strip():
                complete = True
                break
            if card[8:9] == "=":
                cards[card[:8].strip()] = card[10:].split(" /")[0].strip().strip("'").strip()
        if complete:
            break
    else:
        raise ValueError(f"FITS END card missing: {key}")

    def number(name):
        try:
            return float(cards[name])
        except (KeyError, ValueError):
            return None

    values = {
        name.lower(): number(name.upper())
        for name in ("exptime", "cmd_exp", "crpix1", "crpix2", "crota", "rsun", "solar_b0")
    }
    for name in ("yaw_flip", "eclipse"):
        value = number(name.upper())
        values[name] = int(value) if value is not None else None
    values["filter1"] = cards.get("FILTER1")
    values["filter2"] = cards.get("FILTER2")
    return values


def select_for_hour(slot):
    """Apply the same backfill selection rule as collect_hourly.py."""
    row = {
        "slot": slot, "sat": "G18", "channel": CHANNEL, "status": "missing",
        "n_frames": 0, "n_headers": 0, "cutoff_applied": False,
    }
    frames = [frame for frame in list_hour(slot) if frame["obs_end"] + MARGIN <= slot]
    row["n_frames"] = len(frames)
    if not frames:
        return row
    longest = max(frame["dur"] for frame in frames)
    candidates = [frame for frame in frames if frame["dur"] >= longest - DUR_TOLERANCE]
    scored = []
    for frame in candidates:
        header = read_header(frame["key"])
        exposure = round(header["exptime"], 1) if header["exptime"] is not None else -1
        scored.append(((header["eclipse"] == 0, exposure, frame["obs_end"]), frame, header))
    row["n_headers"] = len(scored)
    _, frame, header = max(scored, key=lambda item: item[0])
    row.update(header)
    row.update({
        "status": "ok", "filename": frame["filename"], "s3_key": frame["key"],
        "obs_start": frame["obs_start"], "obs_end": frame["obs_end"],
        "s3_modified": frame["s3_modified"],
        "age_min": (slot - frame["obs_end"]).total_seconds() / 60,
    })
    return row


def download_image(row, output_dir, tries=5):
    """Stream a selected .nc to an atomic local file; return (path, bytes)."""
    observed = row["obs_end"]
    dest = (output_dir / "data" / "g18" / f"{observed:%Y}"
            / f"{observed.timetuple().tm_yday:03d}" / row["filename"].replace(".fits", ".nc"))
    if dest.is_file():
        size = dest.stat().st_size
        expected_size = row.get("nc_size")
        if size > 0 and (expected_size is None or size == expected_size):
            return str(dest), size
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + f".part.{threading.get_ident()}")
    path = "/" + row["s3_key"].rsplit(".", 1)[0] + ".nc"
    for attempt in range(tries):
        try:
            conn = connection()
            conn.request("GET", path)
            response = conn.getresponse()
            if response.status == 404:
                response.read()
                return None
            if response.status != 200:
                response.read()
                raise RuntimeError(f"HTTP {response.status}: {path}")
            size = 0
            with part.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
                    size += len(chunk)
            expected = response.getheader("Content-Length")
            if size == 0 or (expected is not None and size != int(expected)):
                raise IOError(f"incomplete download: {path} ({size} bytes)")
            part.replace(dest)
            return str(dest), size
        except Exception as exc:
            error = exc
            part.unlink(missing_ok=True)
            reset_connection()
            if attempt == tries - 1:
                raise error
            time.sleep(1.5**attempt)


def parallel(items, job, workers):
    """Yield results as they finish, with at most four jobs queued per worker."""
    source = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}

        def fill():
            while len(pending) < workers * 4:
                try:
                    item = next(source)
                except StopIteration:
                    break
                pending[pool.submit(job, item)] = item

        fill()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                item = pending.pop(future)
                try:
                    yield item, future.result(), None
                except Exception as exc:
                    yield item, None, exc
            fill()


def slots():
    slot = START
    while slot <= END:
        yield slot
        slot += HOUR


def save_rows(db, rows):
    if rows:
        columns = ", ".join(COLS)
        placeholders = ", ".join("?" for _ in COLS)
        db.executemany(
            f"INSERT OR REPLACE INTO suvi_hourly ({columns}) VALUES ({placeholders})",
            [[row.get(column) for column in COLS] for row in rows],
        )


def collect_metadata(db, workers):
    existing = {record[0] for record in db.execute("SELECT slot FROM suvi_hourly").fetchall()}
    remaining = [slot for slot in slots() if slot not in existing]
    print(f"metadata: {len(remaining)} of {(END - START) // HOUR + 1} slots, workers={workers}", flush=True)
    rows, errors = [], 0
    for index, (slot, row, error) in enumerate(parallel(remaining, select_for_hour, workers), 1):
        if error:
            errors += 1
            print(f"[metadata error] {slot:%Y-%m-%d %H}Z: {error!r}", file=sys.stderr)
        else:
            rows.append(row)
        if len(rows) >= 200 or index == len(remaining):
            save_rows(db, rows)
            rows.clear()
        if index % 200 == 0 or index == len(remaining):
            print(f"metadata: {index}/{len(remaining)}, errors={errors}", flush=True)
    return errors


def collect_images(db, output_dir, workers):
    records = db.execute(
        "SELECT slot, filename, s3_key, obs_end, nc_path, nc_size "
        "FROM suvi_hourly WHERE status = 'ok' AND slot >= ? AND slot <= ? ORDER BY slot",
        [START, END],
    ).fetchall()
    pending = []
    for slot, filename, key, observed, path, size in records:
        if path and size and Path(path).is_file() and Path(path).stat().st_size == size:
            continue
        pending.append({
            "slot": slot, "filename": filename, "s3_key": key,
            "obs_end": observed, "nc_size": size,
        })
    print(f"images: {len(pending)} of {len(records)} selected frames, workers={workers}", flush=True)
    updates, errors, absent = [], 0, 0
    job = lambda row: download_image(row, output_dir)
    for index, (row, result, error) in enumerate(parallel(pending, job, workers), 1):
        if error:
            errors += 1
            print(f"[image error] {row['slot']:%Y-%m-%d %H}Z: {error!r}", file=sys.stderr)
        elif result is None:
            absent += 1
            print(f"[image absent] {row['filename']}", file=sys.stderr)
        else:
            path, size = result
            updates.append((path, size, row["slot"]))
        if len(updates) >= 200 or index == len(pending):
            if updates:
                db.executemany(
                    "UPDATE suvi_hourly SET nc_path = ?, nc_size = ? WHERE slot = ?", updates
                )
                updates.clear()
        if index % 200 == 0 or index == len(pending):
            print(f"images: {index}/{len(pending)}, errors={errors}, absent={absent}", flush=True)
    return errors, absent


def main():
    raise SystemExit("get_db.py standalone collection is retired; use collect_hourly.py with the central RawDB service")


if __name__ == "__main__":
    sys.exit(main())
