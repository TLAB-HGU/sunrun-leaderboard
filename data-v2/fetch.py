#!/usr/bin/env python3
"""Insert-only refresh of the data-v2 raw sources. Run by hand; there is no timer.

    /home/t-lab01/sunrun/.venv/bin/python data-v2/fetch.py [SOURCE ...] [--full] [--dry-run] [--workers N]
    SOURCE: ace enlil suvi donki srs discussion ace-daily wsa noaa-enlil (default: all; run wsa before noaa-enlil)

Only records that are missing locally are added. A record already stored is never changed, even when the source now
shows a different value, and no history or version is kept.

Layout (2026-11-02): one directory per source under raw/, then input/ (built from raw by notebooks/raw_to_input_df.ipynb).

* ace:   NASA monthly hourly files -> raw/ACE/monthly-txt/. A new file is stored as received; an existing file gains only
         the hours it does not have. Unit: one hourly row.
* enlil: DONKI WSAEnlilSimulations -> raw/ENLIL/donki/. A new month is stored as received; an existing month gains only
         unseen simulations, appended so that its bytes stay as they are. Unit: simulationID.
* suvi:  one GOES-18 Fe195 frame per hour -> its NetCDF in raw/SUVI/nc/. Unit: file. The file is the only record of the
         choice: an hour with a local NetCDF is done, and an hour without one is looked at again on every run.

Sources added for the df 6b section (2026-11-25, plus7 row inputs). Each stores the original file once and never updates it:
* donki:      DONKI CME notifications by issue month -> raw/DONKI/notifications/notifications_{YYYY-MM}.json (unseen
              messageIDs are appended). Unit: messageID.
* srs:        NOAA daily Solar Region Summary -> raw/NOAA/srs/{YYYYMMDD}SRS.txt, from 2022-09-25. Unit: file.
* discussion: SWPC forecast discussion -> raw/NOAA/discussion/txt/ plus the HTTP response headers in hdr/ (Last-Modified
              is the availability time), from 2024-03. Unit: file.
* ace-daily:  ACE SWEPAM 1-minute daily list -> raw/ACE/daily-txt/swepam/{YYYYMMDD}.txt, from 2022-08-01; day D is fetched
              only after D + 1 closed, so `:Created:` is the final publication time. Unit: file.
* wsa:        NOAA WSA-Enlil open-data bucket: listing.json (key -> LastModified first seen, bytes) and every
              evo.earth.nc -> raw/ENLIL/wsa-s3/. Unit: key.
* noaa-enlil: NCEI WSA-Enlil archives of 2025-12..2026-09: listing.json, wayback_evidence.json (WAYBACK below) and, for
              the archives the df rule can use (noaa_enlil_selection, needs wsa), the hourly Earth trajectory read from
              the single *.suball.nc member -> raw/ENLIL/noaa/compact/{archive}.json (the 137 MB member is not kept).
              About 24 GB is streamed at roughly 0.8 MB/s (about 8 h). Unit: archive.

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
import tarfile
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


# ---------------------------------------------------------------- plus7 sources: DONKI notifications
DONKI_DIR = DATA / "raw/DONKI/notifications"
DONKI_URL = "https://ccmc.gsfc.nasa.gov/DONKI-API/get/notifications"
DONKI_START = "2022-01"
DONKI_NAME = re.compile(r"^notifications_(\d{4}-\d{2})\.json$")


def refresh_donki(full: bool = False, dry_run: bool = False) -> tuple[int, int, int]:
    """CME notifications by issue month. Returns (files changed, messages added, errors)."""
    local = []
    if DONKI_DIR.exists():
        local = [match[1] for match in map(DONKI_NAME.match, (p.name for p in DONKI_DIR.iterdir())) if match]
    months = months_to_check(local, DONKI_START, utcnow(), full)
    changed = total = errors = 0
    for index, month in enumerate(months):
        if index:
            time.sleep(ENLIL_SLEEP)
        year, number = int(month[:4]), int(month[5:7])
        query = urllib.parse.urlencode({"startDate": f"{month}-01",
                                        "endDate": f"{month}-{calendar.monthrange(year, number)[1]:02d}", "type": "CME"})
        path = DONKI_DIR / f"notifications_{month}.json"
        try:
            rows = json.loads(http_get(f"{DONKI_URL}?{query}") or b"[]")
            if not isinstance(rows, list):
                raise ValueError("unexpected DONKI response")
            stored = json.loads(path.read_text()) if path.exists() else []
            seen = {row.get("messageID") for row in stored}
            new = [row for row in rows if row.get("messageID") not in seen]
            if new and not dry_run:
                write_atomic(path, (json.dumps(stored + new, ensure_ascii=False, indent=1) + "\n").encode())
            changed, total = changed + bool(new), total + len(new)
        except (RuntimeError, ValueError, OSError) as error:
            print(f"[donki] {month}: {error}", file=sys.stderr)
            errors += 1
    print(f"donki: {len(months)} months checked, {changed} changed, {total} messages added, {errors} errors", flush=True)
    return changed, total, errors


# ---------------------------------------------------------------- plus7 sources: NOAA SRS and SWPC discussion
REPORTS = "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports"
SRS_DIR = DATA / "raw/NOAA/srs"
SRS_START = datetime(2022, 9, 25)
DISCUSSION_DIR = DATA / "raw/NOAA/discussion"
DISCUSSION_START = "2024-03"
DISCUSSION_NAME = re.compile(r"\d{12}forecast_discussion\.txt")


def http_get_headers(url: str, tries: int = 5) -> tuple[bytes, list[tuple[str, str]]] | None:
    """GET with retries, keeping the response headers. 404 returns None."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=120) as response:
                return response.read(), list(response.headers.items())
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            failure = error
        except OSError as error:
            failure = error
        if attempt < tries - 1:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"GET failed {url}: {failure}")


def closed_days(start: datetime, lag_days: int) -> list[datetime]:
    """Days start .. today - lag_days (a day's file is final only after it closed)."""
    last = utcnow().replace(hour=0, minute=0, second=0) - timedelta(days=lag_days)
    return [start + timedelta(days=i) for i in range((last - start).days + 1)]


def fetch_files(name: str, jobs: list[tuple[str, Path]], dry_run: bool, workers: int,
                headers: Path | None = None) -> tuple[int, int, int]:
    """Download (url, path) jobs whose path is missing. Returns (added, missing on the server, errors)."""
    todo = [(url, path) for url, path in jobs if not path.exists()]
    if dry_run:
        print(f"{name}: {len(todo)} of {len(jobs)} files to fetch", flush=True)
        return 0, 0, 0

    def job(item):
        url, path = item
        try:
            found = http_get_headers(url)
            if found is None:
                return "missing"
            if headers is not None:
                write_atomic(headers / path.name, "".join(f"{k}: {v}\n" for k, v in found[1]).encode())
            write_atomic(path, found[0])
            return "added"
        except (RuntimeError, OSError) as error:
            print(f"[{name}] {path.name}: {error}", file=sys.stderr)
            return "error"

    with ThreadPoolExecutor(workers) as pool:
        outcomes = list(pool.map(job, todo))
    counts = {k: outcomes.count(k) for k in ("added", "missing", "error")}
    print(f"{name}: {len(todo)} checked, {counts['added']} added, {counts['missing']} not on the server, "
          f"{counts['error']} errors", flush=True)
    return counts["added"], counts["missing"], counts["error"]


def refresh_srs(dry_run: bool = False, workers: int = 6) -> tuple[int, int, int]:
    jobs = [(f"{REPORTS}/solar_region_summaries/{day:%Y/%m}/{day:%Y%m%d}SRS.txt", SRS_DIR / f"{day:%Y%m%d}SRS.txt")
            for day in closed_days(SRS_START, 1)]
    return fetch_files("srs", jobs, dry_run, workers)


def refresh_discussion(dry_run: bool = False, workers: int = 6) -> tuple[int, int, int]:
    """Every forecast discussion text with its HTTP response headers (Last-Modified = availability time)."""
    jobs, errors = [], 0
    for month in month_list(DISCUSSION_START, f"{utcnow():%Y-%m}"):
        base = f"{REPORTS}/discussion/{month.replace('-', '/')}"
        try:
            index = http_get(base + "/")
        except RuntimeError as error:
            print(f"[discussion] {month}: {error}", file=sys.stderr)
            errors += 1
            continue
        names = sorted(set(DISCUSSION_NAME.findall((index or b"").decode(errors="replace"))))
        jobs += [(f"{base}/{name}", DISCUSSION_DIR / "txt" / name) for name in names
                 if not (DISCUSSION_DIR / "hdr" / name).exists() or not (DISCUSSION_DIR / "txt" / name).exists()]
    added, missing, failed = fetch_files("discussion", jobs, dry_run, workers, headers=DISCUSSION_DIR / "hdr")
    return added, missing, failed + errors


# ---------------------------------------------------------------- plus7 sources: ACE daily 1-minute SWEPAM
ACE_DAILY_DIR = DATA / "raw/ACE/daily-txt/swepam"
ACE_DAILY_URL = "https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/"
ACE_DAILY_START = datetime(2022, 8, 1)


def refresh_ace_daily(dry_run: bool = False, workers: int = 6) -> tuple[int, int, int]:
    """Day D is fetched once D + 1 has closed, so the stored file is the final one (`:Created:` D + 1 00:1x UT)."""
    jobs = [(f"{ACE_DAILY_URL}{day:%Y%m%d}_ace_swepam_1m.txt", ACE_DAILY_DIR / f"{day:%Y%m%d}.txt")
            for day in closed_days(ACE_DAILY_START, 2)]
    return fetch_files("ace-daily", jobs, dry_run, workers)


# ---------------------------------------------------------------- plus7 sources: WSA-Enlil Earth trajectories
WSA_URL = "https://noaa-wsa-enlil-pds.s3.amazonaws.com"
WSA_DIR = DATA / "raw/ENLIL/wsa-s3"
WSA_KEY = re.compile(r"wsa_enlil\.\d{8}_\d+/[^/]+/evo\.earth\.nc")
NOAA_ENLIL_URL = "https://archive.data.noaa.gov/satellite-spaceweather/"
NOAA_ENLIL_DIR = DATA / "raw/ENLIL/noaa"
NOAA_ENLIL_MONTHS = ("2025-12", "2026-09")  # archives that can serve the F1-F3 query origins
NOAA_ENLIL_NAME = re.compile(r"swpc_wsaenlil_(cme|bkg)_(\d{8})_(\d{4})\.tar\.gz")
S3_DOC_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
WAYBACK_IMAGE = "https://services.swpc.noaa.gov/images/animations/enlil/latest.jpg"
WAYBACK = (  # (capture UTC, run UTC shown in the footer, mode, matching NCEI archive); 2026-11-23 visual footer reading
    ("2026-01-01T05:31:46Z", "2025-12-31T20:00Z", "cme", "swpc_wsaenlil_cme_20251231_2052.tar.gz"),
    ("2026-01-03T06:11:49Z", "2026-01-02T18:00Z", "cme", "swpc_wsaenlil_cme_20260102_1859.tar.gz"),
    ("2026-01-06T07:05:23Z", "2026-01-06T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260106_0000.tar.gz"),
    ("2026-01-07T07:09:43Z", "2026-01-06T22:00Z", "cme", "swpc_wsaenlil_cme_20260106_2209.tar.gz"),
    ("2026-01-07T18:49:25Z", "2026-01-07T07:00Z", "cme", "swpc_wsaenlil_cme_20260107_0754.tar.gz"),
    ("2026-01-09T10:22:57Z", "2026-01-09T04:00Z", "cme", "swpc_wsaenlil_cme_20260109_0420.tar.gz"),
    ("2026-01-14T09:46:16Z", "2026-01-14T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260114_0000.tar.gz"),
    ("2026-01-15T07:23:07Z", "2026-01-15T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260115_0000.tar.gz"),
    ("2026-01-16T04:43:33Z", "2026-01-16T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260116_0000.tar.gz"),
    ("2026-01-17T08:11:36Z", "2026-01-17T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260117_0000.tar.gz"),
    ("2026-01-18T05:31:08Z", "2026-01-18T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260118_0000.tar.gz"),
    ("2026-01-19T07:10:08Z", "2026-01-19T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260119_0000.tar.gz"),
    ("2026-01-19T15:07:17Z", "2026-01-19T09:00Z", "cme", "swpc_wsaenlil_cme_20260119_0923.tar.gz"),
    ("2026-01-20T05:08:49Z", "2026-01-19T07:00Z", "cme", "swpc_wsaenlil_cme_20260119_0750.tar.gz"),
    ("2026-01-22T10:23:54Z", "2026-01-22T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260122_0000.tar.gz"),
    ("2026-01-23T04:32:54Z", "2026-01-23T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260123_0000.tar.gz"),
    ("2026-01-24T01:31:20Z", "2026-01-24T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260124_0000.tar.gz"),
    ("2026-01-25T06:30:47Z", "2026-01-25T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260125_0000.tar.gz"),
    ("2026-01-27T09:13:32Z", "2026-01-27T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260127_0000.tar.gz"),
    ("2026-01-28T10:06:59Z", "2026-01-28T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260128_0000.tar.gz"),
    ("2026-01-29T12:32:04Z", "2026-01-29T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260129_0000.tar.gz"),
    ("2026-01-30T05:58:55Z", "2026-01-30T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260130_0000.tar.gz"),
    ("2026-01-31T09:59:52Z", "2026-01-31T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260131_0000.tar.gz"),
    ("2026-02-02T05:24:27Z", "2026-02-02T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260202_0000.tar.gz"),
    ("2026-02-06T08:14:04Z", "2026-02-05T21:00Z", "cme", "swpc_wsaenlil_cme_20260205_2106.tar.gz"),
    ("2026-02-10T04:57:40Z", "2026-02-09T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260209_0000.tar.gz"),
    ("2026-02-11T13:31:06Z", "2026-02-11T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260211_0000.tar.gz"),
    ("2026-02-12T04:47:52Z", "2026-02-11T08:00Z", "cme", "swpc_wsaenlil_cme_20260211_0802.tar.gz"),
    ("2026-02-14T07:19:20Z", "2026-02-13T16:00Z", "cme", "swpc_wsaenlil_cme_20260213_1633.tar.gz"),
    ("2026-02-17T08:35:08Z", "2026-02-16T17:00Z", "cme", "swpc_wsaenlil_cme_20260216_1746.tar.gz"),
    ("2026-02-20T04:16:14Z", "2026-02-20T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260220_0000.tar.gz"),
    ("2026-02-22T06:32:04Z", "2026-02-22T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260222_0000.tar.gz"),
    ("2026-02-23T03:38:23Z", "2026-02-23T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260223_0000.tar.gz"),
    ("2026-02-24T06:27:45Z", "2026-02-23T11:00Z", "cme", "swpc_wsaenlil_cme_20260223_1159.tar.gz"),
    ("2026-02-25T07:13:32Z", "2026-02-25T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260225_0000.tar.gz"),
    ("2026-02-26T00:04:24Z", "2026-02-25T15:00Z", "cme", "swpc_wsaenlil_cme_20260225_1504.tar.gz"),
    ("2026-03-02T05:10:27Z", "2026-03-02T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260302_0000.tar.gz"),
    ("2026-03-04T05:22:44Z", "2026-03-04T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260304_0000.tar.gz"),
    ("2026-03-05T03:31:18Z", "2026-03-05T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260305_0000.tar.gz"),
    ("2026-03-06T06:48:40Z", "2026-03-06T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260306_0000.tar.gz"),
    ("2026-03-07T04:05:56Z", "2026-03-06T10:00Z", "cme", "swpc_wsaenlil_cme_20260306_1049.tar.gz"),
    ("2026-03-10T16:19:14Z", "2026-03-10T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260310_0000.tar.gz"),
    ("2026-03-11T09:08:46Z", "2026-03-11T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260311_0000.tar.gz"),
    ("2026-03-12T04:35:16Z", "2026-03-12T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260312_0000.tar.gz"),
    ("2026-03-13T02:06:29Z", "2026-03-13T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260313_0000.tar.gz"),
    ("2026-03-14T02:34:36Z", "2026-03-13T16:00Z", "cme", "swpc_wsaenlil_cme_20260313_1600.tar.gz"),
    ("2026-03-16T08:03:46Z", "2026-03-16T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260316_0000.tar.gz"),
    ("2026-03-17T04:50:03Z", "2026-03-16T18:00Z", "cme", "swpc_wsaenlil_cme_20260316_1834.tar.gz"),
    ("2026-03-19T05:09:25Z", "2026-03-19T01:00Z", "cme", "swpc_wsaenlil_cme_20260319_0158.tar.gz"),
    ("2026-03-23T05:24:18Z", "2026-03-22T19:00Z", "cme", "swpc_wsaenlil_cme_20260322_1939.tar.gz"),
    ("2026-03-27T20:42:56Z", "2026-03-27T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260327_0000.tar.gz"),
    ("2026-03-28T05:48:16Z", "2026-03-28T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260328_0000.tar.gz"),
    ("2026-03-29T06:33:14Z", "2026-03-29T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260329_0000.tar.gz"),
    ("2026-03-30T07:12:47Z", "2026-03-30T00:00Z", "bkg", "swpc_wsaenlil_bkg_20260330_0000.tar.gz"),
    ("2026-03-31T13:37:45Z", "2026-03-30T04:00Z", "cme", "swpc_wsaenlil_cme_20260330_0422.tar.gz"),
)


def s3_listing(base: str, prefix: str = "") -> list[dict]:
    """Every object under prefix: key, LastModified, size (S3 ListObjectsV2, paginated)."""
    out, token = [], None
    while True:
        query = {"list-type": "2", **({"prefix": prefix} if prefix else {})}
        if token:
            query["continuation-token"] = token
        root = ET.fromstring(http_get(f"{base}?{urllib.parse.urlencode(query)}"))
        for item in root.findall(f"{S3_DOC_NS}Contents"):
            out.append(dict(key=item.findtext(f"{S3_DOC_NS}Key"), last_modified=item.findtext(f"{S3_DOC_NS}LastModified"),
                            bytes=int(item.findtext(f"{S3_DOC_NS}Size"))))
        if root.findtext(f"{S3_DOC_NS}IsTruncated") != "true":
            return out
        token = root.findtext(f"{S3_DOC_NS}NextContinuationToken")


def merge_listing(path: Path, items: list[dict], dry_run: bool) -> dict:
    """Insert-only listing file {key: {last_modified, bytes}}: a key keeps the LastModified first seen."""
    stored = json.loads(path.read_text()) if path.exists() else {}
    new = {item["key"]: {"last_modified": item["last_modified"], "bytes": item["bytes"]}
           for item in items if item["key"] not in stored}
    if new and not dry_run:
        write_atomic(path, (json.dumps({**stored, **new}, indent=1, sort_keys=True) + "\n").encode())
    return {**stored, **new}


def refresh_wsa(dry_run: bool = False, workers: int = 8) -> tuple[int, int, int]:
    """Listing of the NOAA WSA-Enlil open-data bucket (LastModified = availability) and every evo.earth.nc."""
    listing = merge_listing(WSA_DIR / "listing.json",
                            [item for item in s3_listing(WSA_URL + "/") if WSA_KEY.fullmatch(item["key"])], dry_run)
    jobs = [(f"{WSA_URL}/{urllib.parse.quote(key)}", WSA_DIR / "files" / key) for key in sorted(listing)]
    return fetch_files("wsa", jobs, dry_run, workers)


def noaa_enlil_candidates(listing: dict) -> list[str]:
    """NCEI archives whose publication is evidenced within 48 h of the run: object LastModified, or an earlier
    Wayback capture of the SWPC latest-run image showing that run (WAYBACK)."""
    proofs = {key: (pd_ts(capture), pd_ts(run), mode) for capture, run, mode, key in WAYBACK}
    keep = []
    for key, item in sorted(listing.items()):
        match = NOAA_ENLIL_NAME.fullmatch(key.rsplit("/", 1)[1])
        run = datetime.strptime(match[2] + match[3][:2], "%Y%m%d%H").replace(tzinfo=timezone.utc)
        available = pd_ts(item["last_modified"])
        proof = proofs.get(match[0])
        if proof is not None:
            if proof[1] != run or proof[2] != match[1]:
                raise ValueError(f"Wayback evidence does not match {key}")
            available = min(available, proof[0])
        if run <= available <= run + timedelta(hours=48):
            keep.append(key)
    return keep


QUERY_SPAN = ("2025-12-31T23:00Z", "2026-09-26T23:00Z")  # df query origins, hourly (notebook QUERY_SPAN)


def wsa_spans() -> list[tuple[datetime, datetime, datetime, datetime]]:
    """(run, available, first hourly bin, last hourly bin) of every stored evo.earth.nc (needs `wsa` first)."""
    from scipy.io import netcdf_file
    listing = json.loads((WSA_DIR / "listing.json").read_text())
    spans = []
    for key, item in sorted(listing.items()):
        with netcdf_file(WSA_DIR / "files" / key, "r", mmap=False) as dataset:
            run = datetime.fromisoformat(dataset._attributes["rundate_cal"].decode()).replace(tzinfo=timezone.utc)
            seconds = dataset.variables["TIME"][:]
        start = run.timestamp() + float(seconds[0])
        stop = run.timestamp() + float(seconds[-1])
        first, last = int(-(-start // 3600)) + 1, int(stop // 3600)
        if last > first:
            spans.append((run, pd_ts(item["last_modified"]), datetime.fromtimestamp(first * 3600, timezone.utc),
                          datetime.fromtimestamp(last * 3600, timezone.utc)))
    return spans


def noaa_enlil_selection(listing: dict) -> list[str]:
    """The NCEI archives the df rule can use (same rule as the notebook's 6b ENLIL cell): for each hourly query origin,
    among runs published 0-48 h before it that cover origin + 1..72 h, the latest one if it is an NCEI run, and the
    second one when all candidates are NCEI runs (NCEI runs are taken to span 120 h)."""
    proofs = {key: (pd_ts(capture), pd_ts(run)) for capture, run, _, key in WAYBACK}
    horizon = [(*span, None) for span in wsa_spans()]
    for key in noaa_enlil_candidates(listing):
        match = NOAA_ENLIL_NAME.fullmatch(key.rsplit("/", 1)[1])
        run = datetime.strptime(match[2] + match[3][:2], "%Y%m%d%H").replace(tzinfo=timezone.utc)
        available = pd_ts(listing[key]["last_modified"])
        if match[0] in proofs:
            available = min(available, proofs[match[0]][0])
        horizon.append((run, available, run, run + timedelta(hours=120), key))
    horizon.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected = {}
    stamp, stop = pd_ts(QUERY_SPAN[0]), pd_ts(QUERY_SPAN[1])
    while stamp <= stop:
        options = [r for r in horizon if r[1] <= stamp <= r[1] + timedelta(hours=48)
                   and r[2] <= stamp + HOUR and r[3] >= stamp + timedelta(hours=72)]
        if options and options[0][-1] is not None:
            selected[options[0][-1]] = True
        if options and all(option[-1] is not None for option in options):
            for option in options[1:2]:
                selected[option[-1]] = True
        stamp += HOUR
    return sorted(selected)


def pd_ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def earth_compact(nc_bytes: bytes, run: datetime) -> dict:
    """Earth trajectory of one *.suball.nc as complete UTC-hour means (same rule as the notebook's 6b ENLIL cell):
    a bin [H-1h, H) gets the mean of its native samples when samples bracket both ends, every gap is
    <= min(600 s, 2 x median step) and every bracketing speed is finite and positive; otherwise None."""
    import io
    import numpy as np
    from scipy.io import netcdf_file
    with netcdf_file(io.BytesIO(nc_bytes), "r", mmap=False) as dataset:
        reference = datetime.fromisoformat(dataset._attributes["REFDATE_CAL"].decode())
        reference = reference.replace(tzinfo=timezone.utc) if reference.tzinfo is None else reference.astimezone(timezone.utc)
        if reference != run:
            raise ValueError(f"REFDATE_CAL {reference} differs from the archive run hour {run}")
        if dataset.variables["Earth_TIME"].units.decode() not in ("seconds", "s") or dataset.variables["Earth_V1"].units != b"m/s":
            raise ValueError("unexpected Earth_TIME or Earth_V1 units")
        seconds = dataset.variables["Earth_TIME"][:].astype(np.float64)
        speeds = dataset.variables["Earth_V1"][:].astype(np.float64) / 1000.
        model_id = dataset._attributes.get("model_run_id", b"").decode()
    increments = np.diff(seconds)
    if len(seconds) < 2 or not np.isfinite(seconds).all() or not (increments > 0).all():
        raise ValueError("native times must be finite and strictly increasing")
    max_gap = min(600., 2. * float(np.median(increments)))
    absolute = run.timestamp() + seconds
    times, values = [], []
    for hour in range(int(np.ceil(absolute[0] / 3600)) + 1, int(np.floor(absolute[-1] / 3600)) + 1):
        start, stop = (hour - 1) * 3600, hour * 3600
        left = np.searchsorted(absolute, start, side="right")
        right = np.searchsorted(absolute, stop, side="right")
        before, after = left - 1, np.searchsorted(absolute, stop, side="left")
        value = None
        if before >= 0 and after < len(absolute) and right > left:
            segment = speeds[before:after + 1]
            if (np.all(np.diff(absolute[before:after + 1]) <= max_gap) and np.isfinite(segment).all()
                    and (segment > 0).all()):
                value = float(np.mean(speeds[left:right]))
        times.append(datetime.fromtimestamp(hour * 3600, timezone.utc).isoformat())
        values.append(value)
    return dict(run_id=f"noaa:{run:%Y%m%d_%H%M}:{model_id}", run_utc=run.isoformat(), model_run_id=model_id,
                timestamps_utc=times, speed_kms=values, aggregation="complete native hourly-bin mean")


def archive_compact(key: str, path: Path, tries: int = 4) -> None:
    """Stream one NCEI archive, read its single regular *.suball.nc member in memory and store only the hourly Earth
    trajectory (the archive and the member are not kept)."""
    match = NOAA_ENLIL_NAME.fullmatch(key.rsplit("/", 1)[1])
    run = datetime.strptime(match[2] + match[3][:2], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    for attempt in range(tries):
        try:
            request = urllib.request.Request(NOAA_ENLIL_URL + key, headers={"User-Agent": UA})
            with urllib.request.urlopen(request, timeout=300) as response, \
                    tarfile.open(fileobj=response, mode="r|gz") as archive:
                found, name = None, None
                for member in archive:
                    if not member.name.endswith(".suball.nc"):
                        continue
                    if found is not None or not member.isfile() or member.name.startswith("/") or ".." in member.name:
                        raise ValueError(f"ambiguous or unsafe suball.nc member {member.name}")
                    found, name = archive.extractfile(member).read(), member.name
            if found is None:
                raise ValueError("no suball.nc member")
            record = earth_compact(found, run)
            record.update(archive_member=name, native_source=NOAA_ENLIL_URL + key)
            write_atomic(path, (json.dumps({"object": {"key": key}, "runs": [record]}, indent=2) + "\n").encode())
            return
        except (OSError, tarfile.TarError, EOFError) as error:
            failure = error
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"archive failed {key}: {failure}")


def refresh_noaa_enlil(dry_run: bool = False, workers: int = 3) -> tuple[int, int, int]:
    items = []
    for kind in ("bkg", "cme"):
        for month in month_list(*NOAA_ENLIL_MONTHS):
            prefix = f"SWPC/Models/ENLIL/swpc_wsaenlil_{kind}/{month.replace('-', '/')}/"
            items += [item for item in s3_listing(NOAA_ENLIL_URL, prefix) if item["key"].endswith(".tar.gz")]
    listing = merge_listing(NOAA_ENLIL_DIR / "listing.json", items, dry_run)
    evidence = [dict(capture_utc=c, displayed_run_utc=r, mode=m, archive=k,
                     snapshot_url=f"https://web.archive.org/web/{pd_ts(c):%Y%m%d%H%M%S}id_/{WAYBACK_IMAGE}")
                for c, r, m, k in WAYBACK]
    if not dry_run:
        write_atomic(NOAA_ENLIL_DIR / "wayback_evidence.json", (json.dumps(evidence, indent=1) + "\n").encode())
    todo = [key for key in noaa_enlil_selection(listing)
            if not (NOAA_ENLIL_DIR / "compact" / key.rsplit("/", 1)[1].replace(".tar.gz", ".json")).exists()]
    if dry_run:
        print(f"noaa-enlil: {len(listing)} archives listed, {len(todo)} to fetch "
              f"({sum(listing[k]['bytes'] for k in todo) / 1e9:.1f} GB streamed)", flush=True)
        return 0, 0, 0

    def job(key):
        try:
            archive_compact(key, NOAA_ENLIL_DIR / "compact" / key.rsplit("/", 1)[1].replace(".tar.gz", ".json"))
            return "added"
        except (RuntimeError, ValueError) as error:
            print(f"[noaa-enlil] {key}: {error}", file=sys.stderr)
            return "error"

    counts = {"added": 0, "error": 0}
    with ThreadPoolExecutor(workers) as pool:
        for index, outcome in enumerate(pool.map(job, todo), 1):
            counts[outcome] += 1
            if index % 10 == 0:
                print(f"noaa-enlil: {index}/{len(todo)} archives", flush=True)
    print(f"noaa-enlil: {len(todo)} archives, {counts['added']} added, {counts['error']} errors", flush=True)
    return counts["added"], 0, counts["error"]


# ---------------------------------------------------------------- command line
SOURCES = {  # name -> refresh(args) returning (changed, other count, errors)
    "ace": lambda a: refresh_ace(a.full, a.dry_run),
    "enlil": lambda a: refresh_enlil(a.full, a.dry_run),
    "suvi": lambda a: refresh_suvi(a.dry_run, a.workers),
    "donki": lambda a: refresh_donki(a.full, a.dry_run),
    "srs": lambda a: refresh_srs(a.dry_run),
    "discussion": lambda a: refresh_discussion(a.dry_run),
    "ace-daily": lambda a: refresh_ace_daily(a.dry_run),
    "wsa": lambda a: refresh_wsa(a.dry_run),
    "noaa-enlil": lambda a: refresh_noaa_enlil(a.dry_run, a.workers),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Insert missing raw records into data-v2.")
    parser.add_argument("sources", nargs="*", help=f"{', '.join(SOURCES)} (default: all)")
    parser.add_argument("--full", action="store_true", help="ace, enlil, donki: check every month from the start")
    parser.add_argument("--dry-run", action="store_true", help="report what would be added and write nothing")
    parser.add_argument("--workers", type=int, default=8, help="suvi, noaa-enlil: parallel downloads")
    args = parser.parse_args(argv)
    sources = args.sources or list(SOURCES)
    unknown = sorted(set(sources) - set(SOURCES))
    if unknown:
        parser.error(f"unknown source: {', '.join(unknown)}")
    errors = sum(SOURCES[name](args)[2] for name in SOURCES if name in sources)
    return 1 if errors else 0

if __name__ == "__main__":
    sys.exit(main())
