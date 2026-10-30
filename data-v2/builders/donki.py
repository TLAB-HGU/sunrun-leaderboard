"""DONKI v2: WSAEnlilSimulations + CME + CMEAnalysis + IPS + HSS, one verbatim JSON per (endpoint, month).

Replaces, for new experiments, store/headroom-audit/donki (frozen, to 2026-06) and store/donki-live
(pinned receipt, to 2026-10-07). Those folders are never touched. Every analysis is kept verbatim;
selection happens at feature time (submissionTime / modelCompletionTime <= origin, never isMostAccurate),
as in store/donki-cme/VINTAGE_RULE.md.

    python donki.py [--start 2022-07] [--end now] [--refresh-last 2]

Months already on disk are kept unless they are among the last `--refresh-last` months (those can still
gain runs/analyses). manifest.json / SHA256SUMS always describe every file in the folder.
"""
import argparse
import calendar
import hashlib
import json
import sys
import time
import uuid
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

from raw_sources import preserve_previous, rawdb_request, register_body, save

BASE = "https://ccmc.gsfc.nasa.gov/DONKI/WS/get"
ENDPOINTS = ("WSAEnlilSimulations", "CME", "CMEAnalysis", "IPS", "HSS")
OUT = Path("/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store/v2/donki")


def months(start, end):
    y, m = int(start[:4]), int(start[5:7])
    out = []
    while (y, m) <= (end.year, end.month):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fetch(url, tries=5):
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sunrun-data-v2/1.0", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read()
                headers = dict(resp.getheaders())
            data = json.loads(body.decode("utf-8")) if body.strip() else []
            if not isinstance(data, list):
                raise ValueError(f"expected a JSON list, got {type(data).__name__}")
            return data, body, headers
        except Exception as exc:  # noqa: BLE001 - network, decode, rate limit
            last = exc
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"fetch failed: {url}: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-07")
    ap.add_argument("--refresh-last", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=1.5)
    args = ap.parse_args()
    rawdb_request("status")
    run_id = f"data-v2-donki-{uuid.uuid4().hex}"
    OUT.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date()
    span = months(args.start, today)
    refresh = set(span[-args.refresh_last:])
    fetched = 0
    for month in span:
        last_day = calendar.monthrange(int(month[:4]), int(month[5:7]))[1]
        end = min(date(int(month[:4]), int(month[5:7]), last_day), today)
        for ep in ENDPOINTS:
            path = OUT / f"{ep}_{month}.json"
            url = f"{BASE}/{ep}?startDate={month}-01&endDate={end.isoformat()}"
            if path.exists() and month not in refresh:
                register_body(path, 'donki', ep, url)
                continue
            data, body, headers = fetch(url)
            if path.exists() and path.read_bytes() != body:
                preserve_previous(path, 'donki', ep, url)
            save(path, body)
            register_body(path, 'donki', ep, url, headers, datetime.now(timezone.utc).isoformat())
            (OUT / f"hdr_{ep}_{month}.txt").write_text(
                f"url: {url}\nfetched_at_utc: {datetime.now(timezone.utc).isoformat()}\n"
                + "".join(f"{k}: {v}\n" for k, v in headers.items()))
            fetched += 1
            print(f"{ep} {month}: {len(data)} rows", flush=True)
            time.sleep(args.sleep)
    files = sorted(p for p in OUT.glob("*_*.json"))
    entries = []
    for p in files:
        body = p.read_bytes()
        ep, month = p.stem.rsplit("_", 1)
        entries.append({"endpoint": ep, "month": month, "file": p.name, "sha256": hashlib.sha256(body).hexdigest(),
                        "bytes": len(body), "rows": len(json.loads(body) or [])})
    (OUT / "SHA256SUMS").write_text("".join(f"{e['sha256']}  {e['file']}\n" for e in entries))
    as_of = datetime.now(timezone.utc).isoformat()
    (OUT / "MANIFEST.json").write_text(json.dumps({
        "dataset": "donki", "version": "v2", "as_of_utc": as_of, "source": BASE, "endpoints": ENDPOINTS,
        "builder": __file__, "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "vintage_rule": "keep verbatim; select at feature time by submissionTime/modelCompletionTime <= origin; "
                        "never isMostAccurate. fetched_at is not historical availability.",
        "replaces_for_new_experiments": ["store/headroom-audit/donki", "store/donki-live", "store/donki-cme"],
        "files": entries}, indent=1) + "\n")
    rawdb_request('execute', sql='INSERT INTO raw.runs (run_id, source, status, metadata_json) VALUES (?, ?, ?, ?)',
                  params=[run_id, 'donki', 'complete', (OUT / 'MANIFEST.json').read_text()])
    print(f"done: fetched {fetched} requests, {len(entries)} files, as_of {as_of}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
