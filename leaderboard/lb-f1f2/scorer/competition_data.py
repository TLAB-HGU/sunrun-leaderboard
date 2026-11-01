"""Freeze the notebook's authoritative NASA SWEPAM observations for all clients."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import StringIO
from pathlib import Path
import re
import urllib.request

import numpy as np
import pandas as pd

BASE_URL = "https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/"


def parse_month(payload: bytes, name: str) -> pd.DataFrame:
    rows = []
    for line in payload.decode("utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith(("#", ":")):
            continue
        fields = line.split()
        if len(fields) != 10 or not re.fullmatch(r"\d{4}", fields[0]):
            raise ValueError(f"Malformed SWEPAM row in {name}")
        rows.append(" ".join(fields))
    if not rows:
        raise ValueError(f"Empty SWEPAM file: {name}")
    frame = pd.read_csv(StringIO("\n".join(rows)), sep=r"\s+", header=None)
    frame = frame.apply(pd.to_numeric, errors="raise")
    hhmm = frame[3]
    if ((hhmm % 100 != 0) | (hhmm < 0) | (hhmm > 2300)).any():
        raise ValueError(f"Non-hourly timestamp in {name}")
    times = pd.to_datetime(dict(year=frame[0], month=frame[1], day=frame[2]), utc=True)
    times += pd.to_timedelta(hhmm // 100, unit="h")
    # Some NASA files include neighboring months: the named month is authoritative.
    selected = times.dt.strftime("%Y%m").eq(name[:6])
    valid = frame[6].eq(0) & np.isfinite(frame[8]) & frame[8].gt(0)
    result = pd.DataFrame({"timestamp_utc": times, "speed_kms": frame[8].where(valid)})[selected]
    if result.empty or result.timestamp_utc.duplicated().any():
        raise ValueError(f"Empty or duplicated SWEPAM month: {name}")
    return result


def download_observations(cache: Path) -> tuple[pd.DataFrame, dict]:
    """Dec 2025 supplies causal persistence; Jan–Sep 2026 supplies verification."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    names = [f"{month:%Y%m}_ace_swepam_1h.txt" for month in
             pd.date_range("2025-12-01", "2026-09-01", freq="MS")]

    def fetch(name):
        path = cache / name
        if not path.exists():
            request = urllib.request.Request(BASE_URL + name, headers={"User-Agent": "sunrun-leaderboard/2"})
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = response.read()
            parse_month(payload, name)  # validate before caching
            path.write_bytes(payload)
        payload = path.read_bytes()
        frame = parse_month(payload, name)
        month = pd.Timestamp(f"{name[:4]}-{name[4:6]}-01", tz="UTC")
        expected = pd.date_range(month, month + pd.offsets.MonthBegin(1), freq="h", inclusive="left")
        if not pd.DatetimeIndex(frame.timestamp_utc).sort_values().equals(expected):
            raise ValueError(f"Incomplete NASA monthly file: {name}")
        return frame, {"name": name, "url": BASE_URL + name, "sha256": sha256(payload).hexdigest()}

    with ThreadPoolExecutor(max_workers=4) as pool:
        loaded = list(pool.map(fetch, names))
    observations = pd.concat([frame for frame, _ in loaded], ignore_index=True).sort_values("timestamp_utc")
    return observations, {"source": BASE_URL, "files": [record for _, record in loaded],
                          "quality_rule": "status == 0 and finite speed > 0; missing truth excluded"}
