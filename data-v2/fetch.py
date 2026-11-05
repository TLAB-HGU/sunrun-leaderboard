#!/usr/bin/env python3
"""Insert-only refresh of the data-v2 raw sources. Run by hand; there is no timer.

    /home/t-lab01/sunrun/.venv/bin/python data-v2/fetch.py [ace enlil suvi] [--full] [--dry-run] [--workers N]

Only records that are missing locally are added. A record already stored is never changed, even when the source now
shows a different value, and no history or version is kept.

Layout (2026-11-02): one directory per source under raw/, then input/ (built from raw by notebooks/raw_to_input_df.ipynb).

* ace:   NASA monthly hourly files -> raw/ACE/monthly-txt/. A new file is stored as received; an existing file gains only
         the hours it does not have. Unit: one hourly row.
* enlil: DONKI WSAEnlilSimulations -> raw/ENLIL/donki/. A new month is stored as received; an existing month gains only
         unseen simulations, appended so that its bytes stay as they are. Unit: simulationID.
* suvi:  one GOES-18 Fe195 frame per hour -> its NetCDF in raw/SUVI/nc/. Unit: file. The file is the only record of the
         choice: an hour with a local NetCDF is done, and an hour without one is looked at again on every run.

Selection of the SUVI frame for hour H: frames of [H-1h, H) with obs_end + 2 s <= H, eclipse-free first, then the longest
exposure in 0.1 s bins, then the latest. Without a candidate, or when the frame has no NetCDF on S3, nothing is stored;
there is no fallback to another satellite.

ace and enlil look at the months from the latest local month - 1 up to the current month (--full: every month from the
start). suvi always looks at every closed hour that has no local NetCDF. input/ is not touched.
"""
from __future__ import annotations

import argparse
import calendar
import http.client
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA = Path(__file__).resolve().parent
ACE_DIR = DATA / "raw/ACE/monthly-txt"
ENLIL_DIR = DATA / "raw/ENLIL/donki"
NC_DIR = DATA / "raw/SUVI/nc"

UA = "sunrun-data-v2/1.0"
ACE_URL = "https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/"
ACE_START = "2022-08"
ACE_COLUMNS = {"swepam": 10, "mag": 13, "epam": 16, "sis": 10, "loc": 9}  # whitespace-separated fields of a data row
ACE_NAME = re.compile(r"^(\d{4})(\d{2})_ace_([a-z]+)_1h\.txt$")
DATA_ROW = re.compile(rb"\d{4}\s")

ENLIL_URL = "https://ccmc.gsfc.nasa.gov/DONKI/WS/get/WSAEnlilSimulations"
ENLIL_START = "2022-07"
ENLIL_SLEEP = 1.5  # seconds between DONKI requests
ENLIL_NAME = re.compile(r"^WSAEnlilSimulations_(\d{4}-\d{2})\.json$")

SAT = 18
CHANNEL = "Fe195"
S3_BASE = f"https://noaa-goes{SAT}.s3.amazonaws.com"
FIRST_SLOT = datetime(2022, 8, 22, 21)
HOUR = timedelta(hours=1)
MARGIN = 2             # seconds: obs_end + margin <= H
DUR_TOLERANCE = 0.15   # seconds: file-name duration (0.1 s resolution) against the header EXPTIME
FITS_BLOCK = 2880
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
SUVI_NAME = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)_e(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)_c")


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


def write_atomic(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with tmp.open("wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def http_get(url: str, tries: int = 5) -> bytes | None:
    """GET with retries. 404 returns None."""
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            failure = error
        except OSError as error:
            failure = error
        if attempt < tries - 1:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"GET failed {url}: {failure}")


def month_list(start: str, end: str) -> list[str]:
    """Inclusive YYYY-MM range."""
    year, month = int(start[:4]), int(start[5:7])
    out = []
    while f"{year:04d}-{month:02d}" <= end:
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def previous_month(month: str) -> str:
    year, number = int(month[:4]), int(month[5:7])
    return f"{year - 1:04d}-12" if number == 1 else f"{year:04d}-{number - 1:02d}"


def months_to_check(local: list[str], first: str, today: datetime, full: bool) -> list[str]:
    start = first if full or not local else max(first, previous_month(max(local)))
    return month_list(start, f"{today:%Y-%m}")


# ---------------------------------------------------------------- ACE
def split_rows(payload: bytes, columns: int, name: str) -> tuple[list[bytes], list[tuple], list[bytes]]:
    """Header lines, (key, line) data rows and trailing lines of one monthly file."""
    head, rows, tail = [], [], []
    for line in payload.splitlines(keepends=True):
        if DATA_ROW.match(line):
            fields = line.split()
            if len(fields) != columns:
                raise ValueError(f"{name}: expected {columns} fields, got {len(fields)}: {line!r}")
            rows.append((tuple(fields[:4]), line))
        else:
            (tail if rows else head).append(line)
    if not rows:
        raise ValueError(f"{name}: no data rows")
    return head, rows, tail


def insert_ace_month(path: Path, payload: bytes, columns: int, dry_run: bool = False) -> int:
    """Add the hours `path` lacks; returns the number of rows added."""
    _, web, _ = split_rows(payload, columns, path.name)
    if not path.exists():
        if not dry_run:
            write_atomic(path, payload)
        return len(web)
    head, local, tail = split_rows(path.read_bytes(), columns, path.name)
    known = {key for key, _ in local}
    new, seen = [], set()
    for key, line in web:
        if key not in known and key not in seen:
            seen.add(key)
            new.append((key, line))
    if not new:
        return 0
    if not dry_run:
        rows = [line if line.endswith(b"\n") else line + b"\n" for _, line in
                sorted(local + new, key=lambda row: row[0])]
        write_atomic(path, b"".join(head + rows + tail))
    return len(new)


def ace_months(product: str, today: datetime, full: bool) -> list[str]:
    local = []
    if ACE_DIR.exists():
        for path in ACE_DIR.iterdir():
            match = ACE_NAME.match(path.name)
            if match and match[3] == product:
                local.append(f"{match[1]}-{match[2]}")
    return months_to_check(local, ACE_START, today, full)


def refresh_ace(full: bool = False, dry_run: bool = False, workers: int = 4) -> tuple[int, int, int]:
    """Returns (files changed, rows added, errors)."""
    today = utcnow()
    current = f"{today:%Y-%m}"
    jobs = [(product, month) for product in ACE_COLUMNS for month in ace_months(product, today, full)]

    def job(item):
        product, month = item
        name = f"{month.replace('-', '')}_ace_{product}_1h.txt"
        try:
            payload = http_get(ACE_URL + name)
            if payload is None:
                if month == current:  # the new month may not be published yet
                    return 0, 0
                raise RuntimeError("not found")
            return insert_ace_month(ACE_DIR / name, payload, ACE_COLUMNS[product], dry_run), 0
        except (RuntimeError, ValueError, OSError) as error:
            print(f"[ace] {name}: {error}", file=sys.stderr)
            return 0, 1

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(job, jobs))
    added = [n for n, _ in results if n]
    errors = sum(failed for _, failed in results)
    print(f"ace: {len(jobs)} files checked, {len(added)} changed, {sum(added)} rows added, {errors} errors", flush=True)
    return len(added), sum(added), errors


# ---------------------------------------------------------------- ENLIL
def insert_enlil_month(path: Path, body: bytes, dry_run: bool = False) -> int:
    """Add the simulations `path` lacks; returns the number added. Stored bytes are never rewritten."""
    web = json.loads(body) if body.strip() else []
    if not isinstance(web, list) or any("simulationID" not in row for row in web):
        raise ValueError(f"{path.name}: unexpected response")
    if not path.exists():
        if web and not dry_run:
            write_atomic(path, body)
        return len(web)
    stored = path.read_bytes().rstrip()
    local = json.loads(stored)
    known = {row["simulationID"] for row in local}
    new = []
    for row in web:
        if row["simulationID"] not in known:
            known.add(row["simulationID"])
            new.append(row)
    if not new:
        return 0
    if not dry_run:
        rows = json.dumps(new, ensure_ascii=False, separators=(",", ":")).encode("utf-8")[1:-1]
        merged = stored[:-1] + (b"," if local else b"") + rows + b"]"
        json.loads(merged)
        write_atomic(path, merged)
    return len(new)


def refresh_enlil(full: bool = False, dry_run: bool = False) -> tuple[int, int, int]:
    """Returns (files changed, simulations added, errors)."""
    today = utcnow()
    local = []
    if ENLIL_DIR.exists():
        local = [match[1] for match in map(ENLIL_NAME.match, (p.name for p in ENLIL_DIR.iterdir())) if match]
    months = months_to_check(local, ENLIL_START, today, full)
    changed, total, errors = 0, 0, 0
    for index, month in enumerate(months):
        if index:
            time.sleep(ENLIL_SLEEP)
        year, number = int(month[:4]), int(month[5:7])
        last = min(datetime(year, number, calendar.monthrange(year, number)[1]), today)
        query = urllib.parse.urlencode({"startDate": f"{month}-01", "endDate": f"{last:%Y-%m-%d}"})
        path = ENLIL_DIR / f"WSAEnlilSimulations_{month}.json"
        try:
            body = http_get(f"{ENLIL_URL}?{query}")
            if body is None:
                raise RuntimeError("not found")
            added = insert_enlil_month(path, body, dry_run)
        except (RuntimeError, ValueError, OSError) as error:
            print(f"[enlil] {month}: {error}", file=sys.stderr)
            errors += 1
            continue
        changed += bool(added)
        total += added
    print(f"enlil: {len(months)} months checked, {changed} changed, {total} simulations added, {errors} errors", flush=True)
    return changed, total, errors


# ---------------------------------------------------------------- SUVI: S3 access
_local = threading.local()


def s3_get(url: str, byte_range: tuple[int, int] | None = None, tries: int = 5) -> bytes | None:
    """GET on one keep-alive connection per thread. 404 returns None; other failures are retried."""
    parts = urllib.parse.urlsplit(url)
    headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {}
    connections = _local.__dict__.setdefault("connections", {})
    for attempt in range(tries):
        try:
            if parts.netloc not in connections:
                connections[parts.netloc] = http.client.HTTPSConnection(parts.netloc, timeout=30)
            connection = connections[parts.netloc]
            connection.request("GET", parts.path + (f"?{parts.query}" if parts.query else ""), headers=headers)
            response = connection.getresponse()
            body = response.read()
            if response.status == 404:
                return None
            if response.status in (200, 206):
                return body
            failure = RuntimeError(f"HTTP {response.status} {url}")
        except (OSError, http.client.HTTPException) as error:
            connections.pop(parts.netloc, None)
            failure = error
        if attempt == tries - 1:
            raise failure
        time.sleep(1.5 ** attempt)


def hour_prefix(hour: datetime) -> str:
    return f"SUVI-L1b-{CHANNEL}/{hour:%Y}/{hour.timetuple().tm_yday:03d}/{hour:%H}/"


def stamp(groups: tuple) -> datetime:
    year, doy, hour, minute, second, tenth = map(int, groups)
    return datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hour, minutes=minute, seconds=second,
                                            milliseconds=tenth * 100)


def list_hour(hour: datetime) -> list[dict]:
    """FITS frames in the S3 folder of `hour`, oldest first. dur comes from the file name and hints at the exposure."""
    out, token = [], None
    while True:
        query = {"list-type": "2", "prefix": hour_prefix(hour)}
        if token:
            query["continuation-token"] = token
        root = ET.fromstring(s3_get(f"{S3_BASE}/?{urllib.parse.urlencode(query)}"))
        for item in root.findall(f"{S3_NS}Contents"):
            key = item.findtext(f"{S3_NS}Key")
            match = SUVI_NAME.search(key)
            if not key.endswith(".fits") or not match:
                continue
            start, end = stamp(match.groups()[:6]), stamp(match.groups()[6:])
            out.append(dict(key=key, filename=key.rsplit("/", 1)[1], obs_start=start, obs_end=end,
                            dur=(end - start).total_seconds()))
        if root.findtext(f"{S3_NS}IsTruncated") != "true":
            return sorted(out, key=lambda frame: frame["obs_end"])
        token = root.findtext(f"{S3_NS}NextContinuationToken")


def parse_header(raw: bytes) -> tuple[dict, bool]:
    cards = {}
    for offset in range(0, len(raw) - 79, 80):
        card = raw[offset:offset + 80].decode("ascii", "replace")
        if card.startswith("END") and card[3:].strip() == "":
            return cards, True
        if len(card) > 9 and card[8] == "=":
            cards[card[:8].strip()] = card[10:].split(" /")[0].strip().strip("'").strip()
    return cards, False


def read_header(key: str) -> dict:
    """Header cards of one FITS file through Range requests, growing until END is found."""
    for blocks in (5, 12, 60):  # SUVI headers end at about 12.3 kB, and an hour reads about 59 of them
        raw = s3_get(f"{S3_BASE}/{key}", (0, blocks * FITS_BLOCK - 1))
        if raw is None:
            raise RuntimeError(f"FITS object vanished: {key}")
        cards, done = parse_header(raw)
        if done:
            break
    else:
        raise ValueError(f"no END card: {key}")

    def number(name):
        try:
            return float(cards[name])
        except (KeyError, ValueError):
            return None

    return dict(exptime=number("EXPTIME"), eclipse=number("ECLIPSE"))


def select_for_hour(slot: datetime) -> dict | None:
    """The frame chosen for hour H, None without a candidate. Network errors raise, so the next run looks at H again."""
    frames = [frame for frame in list_hour(slot - HOUR)
              if frame["obs_end"] + timedelta(seconds=MARGIN) <= slot]
    if not frames:
        return None
    longest = max(frame["dur"] for frame in frames)
    scored = []
    for frame in frames:
        if frame["dur"] >= longest - DUR_TOLERANCE:
            header = read_header(frame["key"])
            exposure = round(header["exptime"], 1) if header["exptime"] is not None else -1
            scored.append(((header["eclipse"] == 0, exposure, frame["obs_end"]), frame))
    return max(scored, key=lambda item: item[0])[1]


def nc_file(frame: dict) -> Path:
    filename = frame["filename"].replace(".fits", ".nc")
    return NC_DIR / f"g{SAT}" / f"{frame['obs_end']:%Y}" / f"{frame['obs_end'].timetuple().tm_yday:03d}" / filename


def download_nc(frame: dict) -> Path | None:
    """Store the NetCDF twin of the selected frame. An absent twin is an upstream gap and leaves no file."""
    path = nc_file(frame)
    if not path.exists():
        body = s3_get(f"{S3_BASE}/{frame['key'].rsplit('.', 1)[0]}.nc")
        if body is None:
            return None
        write_atomic(path, body)
    return path


def stored_slots() -> set[datetime]:
    """Hours H that have a local NetCDF. Every stored frame ends inside [H-1h, H), so H is read from the file name."""
    slots = set()
    for path in NC_DIR.glob(f"g{SAT}/*/*/*.nc"):
        match = SUVI_NAME.search(path.name)
        if match:
            end = stamp(match.groups()[6:])
            slots.add(end.replace(minute=0, second=0, microsecond=0) + HOUR)
    return slots


def missing_slots(stored: set[datetime], now: datetime) -> list[datetime]:
    """Closed hours without a local NetCDF, oldest first."""
    last = (now - timedelta(seconds=MARGIN)).replace(minute=0, second=0, microsecond=0)
    hours = (last - FIRST_SLOT) // HOUR + 1
    return [slot for slot in (FIRST_SLOT + index * HOUR for index in range(hours)) if slot not in stored]


def fetch_slot(slot: datetime) -> str:
    frame = select_for_hour(slot)
    if frame is None:
        return "missing"
    return "added" if download_nc(frame) else "gap"


def refresh_suvi(dry_run: bool = False, workers: int = 8) -> tuple[int, int, int]:
    """Returns (files added, hours left without a file, errors)."""
    todo = missing_slots(stored_slots(), utcnow())
    if dry_run:
        span = f" ({todo[0]:%Y-%m-%d %H}Z .. {todo[-1]:%Y-%m-%d %H}Z)" if todo else ""
        print(f"suvi: {len(todo)} hours to fetch{span}", flush=True)
        return 0, 0, 0

    def job(slot):
        try:
            return slot, fetch_slot(slot), None
        except (RuntimeError, ValueError, OSError, ET.ParseError, http.client.HTTPException) as error:
            return slot, "error", error

    counts = {"added": 0, "missing": 0, "gap": 0, "error": 0}
    with ThreadPoolExecutor(workers) as pool:
        for index, (slot, outcome, error) in enumerate(pool.map(job, todo), 1):
            counts[outcome] += 1
            if error is not None:
                print(f"[suvi] {slot:%Y-%m-%d %H}Z: {error}", file=sys.stderr)
            if index % 200 == 0:
                print(f"suvi: {index}/{len(todo)} hours processed", flush=True)
    print(f"suvi: {len(todo)} hours checked, {counts['added']} added, {counts['missing']} without a frame, "
          f"{counts['gap']} without a NetCDF, {counts['error']} errors", flush=True)
    return counts["added"], counts["missing"] + counts["gap"], counts["error"]


# ---------------------------------------------------------------- command line
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Insert missing raw records into data-v2.")
    parser.add_argument("sources", nargs="*", help="ace, enlil, suvi (default: all three)")
    parser.add_argument("--full", action="store_true", help="ace, enlil: check every month from the start")
    parser.add_argument("--dry-run", action="store_true", help="report what would be added and write nothing")
    parser.add_argument("--workers", type=int, default=8, help="suvi: parallel downloads")
    args = parser.parse_args(argv)
    sources = args.sources or ["ace", "enlil", "suvi"]
    unknown = sorted(set(sources) - {"ace", "enlil", "suvi"})
    if unknown:
        parser.error(f"unknown source: {', '.join(unknown)}")
    errors = 0
    if "ace" in sources:
        errors += refresh_ace(args.full, args.dry_run)[2]
    if "enlil" in sources:
        errors += refresh_enlil(args.full, args.dry_run)[2]
    if "suvi" in sources:
        errors += refresh_suvi(args.dry_run, args.workers)[2]
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
