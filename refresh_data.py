"""Snapshot the current September archive and rebuild the hourly series into leaderboard/data/.

    python refresh_data.py [--min-last 2026-09-29T23:00:00Z]

analysis/naive_baselines/raw is an immutable cache and is left untouched. Months before the
current one are read from it; the newest month is downloaded fresh into data/raw/ (dated name).
Parsing and the missing-value policy mirror analysis/naive_baselines/run.py: nonfinite / <=0 / -9999.9
are missing, forward-filled causally without limit, positive speeds kept regardless of status.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD_RAW = HERE.parent / "analysis" / "naive_baselines" / "raw"
OUT = HERE / "data"
SOURCE = "https://sohoftp.nascom.nasa.gov/sdb/ace/monthly/"
MONTH = "202609"
FIRST_MONTH = "202508"
HOUR = timedelta(hours=1)
EVAL_FLAG_START = datetime(2025, 9, 26, tzinfo=timezone.utc)


def parse(data: bytes) -> dict:
    rows = {}
    for line in data.decode().splitlines():
        f = line.split()
        if not f or not f[0].isdigit():
            continue
        if len(f) != 10:
            raise ValueError(f"malformed row: {line}")
        t = datetime(int(f[0]), int(f[1]), int(f[2]), int(f[3][:2]), int(f[3][2:]), tzinfo=timezone.utc)
        if t.minute or t in rows:
            raise ValueError(f"duplicate or non-hourly timestamp: {t}")
        rows[t] = (float(f[8]), int(f[6]))
    return rows


def months(first: str, last: str):
    y, m = int(first[:4]), int(first[4:])
    while f"{y}{m:02d}" <= last:
        yield f"{y}{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-last", default="2026-09-29T23:00:00Z", help="fail unless the archive reaches this hour")
    ap.add_argument("--offline", action="store_true", help="reuse the newest dated snapshot in data/raw instead of downloading")
    a = ap.parse_args()
    (OUT / "raw").mkdir(parents=True, exist_ok=True)

    if a.offline:
        path = sorted((OUT / "raw").glob(f"{MONTH}_ace_swepam_1h_*.txt"))[-1]
        data = path.read_bytes()
    else:
        data = urllib.request.urlopen(SOURCE + f"{MONTH}_ace_swepam_1h.txt", timeout=60).read()
        if not data.startswith(b":Product:"):
            raise SystemExit("unexpected archive format")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
        path = OUT / "raw" / f"{MONTH}_ace_swepam_1h_{stamp}.txt"
        path.write_bytes(data)
    rows, sources = {}, []
    for mo in months(FIRST_MONTH, MONTH):
        if mo == MONTH:
            rows.update(parse(data))
            sources.append({"month": mo, "file": str(path.relative_to(HERE)), "sha256": hashlib.sha256(data).hexdigest(),
                            "issued": next((l for l in data.decode().splitlines() if l.startswith(":Issued:")), "")})
        else:
            b = (OLD_RAW / f"{mo}_ace_swepam_1h.txt").read_bytes()
            rows.update(parse(b))
            sources.append({"month": mo, "file": f"analysis/naive_baselines/raw/{mo}_ace_swepam_1h.txt", "sha256": hashlib.sha256(b).hexdigest()})

    first, last = min(rows), max(rows)
    if iso(last) < a.min_last:
        raise SystemExit(f"archive ends at {iso(last)}, before required {a.min_last}")
    times = [first + i * HOUR for i in range(int((last - first) / HOUR) + 1)]
    lastv, out = None, []
    for t in times:
        v, st = rows.get(t, (None, None))
        miss = v is None or not math.isfinite(v) or v <= 0
        if not miss:
            lastv = v
        out.append((iso(t), v, st, lastv, int(miss), int(t >= EVAL_FLAG_START)))
    assert out[0][3] is not None and all(o[3] is not None for o in out)
    with (OUT / "hourly_series.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_utc", "raw_speed_kms", "status", "filled_speed_kms", "was_missing", "in_evaluation_period"])
        w.writerows(out)
    (OUT / "series_sources.json").write_text(json.dumps({"last_hour": iso(last), "rows": len(out), "sources": sources}, indent=1))
    print(f"{len(out)} hours, {iso(times[0])} .. {iso(last)}; snapshot {path.name}")


if __name__ == "__main__":
    main()
