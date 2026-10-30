#!/usr/bin/env python3
"""One GOES SUVI header record per hour -> the central internal raw DuckDB service. Backfill reads no pixel data; `download` fetches the .nc.

Rule: for the top of hour H, look at the frames observed in [H-1h, H) of ONE satellite and take
the one with the largest EXPTIME (read from the FITS header via HTTP Range; ties -> latest).
  * backfill / hour : decided from observation time only (S3 LastModified was rewritten for old files).
  * live / --realtime: additionally requires S3 LastModified <= H (contest rule).
Missing hours stay missing (status='missing'); there is no fallback to the other satellite.
The same select_for_hour() serves backfill and live.

Usage
  python collect_hourly.py init     --sat g18
  python collect_hourly.py backfill --sat g18 [--start 2022-08-04T00:00 --end 2026-09-30T00:00] [--workers 32]
  python collect_hourly.py backfill --sat g18                  # from START (2022-08-22T21:00), resumable
  python collect_hourly.py hour     --sat g18 --at 2026-09-28T12:00 [--realtime] [--all-headers]
  python collect_hourly.py download --sat g18 [--dir data]     # .nc of every ok row (run after backfill)
  python collect_hourly.py live     --sat g18 [--once] [--download-dir data]   # one per satellite
  python collect_hourly.py status   --sat g18
  python collect_hourly.py check    --sat g18                  # leakage assertions

Only GOES-18 is collected. The central service owns the DB file. All times are UTC.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path


import collect as C

HOUR = timedelta(hours=1)
MARGIN = 2                     # seconds: obs_end + margin <= H
DUR_TOLERANCE = 0.15           # s: file-name duration (0.1 s resolution) vs header EXPTIME pre-filter
HERE = Path(__file__).parent
START = datetime(2022, 8, 22, 21)   # first slot kept (the earlier all-missing span was dropped)

COLS = ["slot", "sat", "channel", "status", "filename", "s3_key", "obs_start", "obs_end", "age_min",
        "s3_modified", "exptime", "cmd_exp", "crpix1", "crpix2", "crota", "rsun", "solar_b0",
        "yaw_flip", "eclipse", "filter1", "filter2", "n_frames", "n_headers", "cutoff_applied",
        "nc_path", "nc_size"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw.suvi_hourly (            -- exactly one row per slot (UTC, naive)
  slot TIMESTAMP NOT NULL,
  sat TEXT NOT NULL, channel TEXT NOT NULL,
  status TEXT,                                      -- ok | missing
  filename TEXT, s3_key TEXT,
  obs_start TIMESTAMP, obs_end TIMESTAMP,           -- from the file name (_s.. / _e..)
  age_min REAL,                                     -- slot - obs_end, minutes
  s3_modified TIMESTAMP,
  exptime REAL, cmd_exp REAL, crpix1 REAL, crpix2 REAL, crota REAL, rsun REAL, solar_b0 REAL,
  yaw_flip SMALLINT, eclipse SMALLINT, filter1 TEXT, filter2 TEXT,
  n_frames INTEGER,                                 -- candidate frames in [H-1h, H)
  n_headers INTEGER,                                -- headers actually read
  cutoff_applied BOOLEAN,                           -- LastModified <= slot was enforced
  PRIMARY KEY (sat, channel, slot)
);
ALTER TABLE raw.suvi_hourly ADD COLUMN IF NOT EXISTS nc_path TEXT; 
ALTER TABLE raw.suvi_hourly ADD COLUMN IF NOT EXISTS nc_size BIGINT;
"""


# ---------------------------------------------------------------- helpers
def utcnow():
    return datetime.now(C.UTC).replace(tzinfo=None, microsecond=0)


def parse_hour(s):
    t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if t.tzinfo is not None:
        t = t.astimezone(C.UTC).replace(tzinfo=None)
    if t.minute or t.second or t.microsecond:
        raise argparse.ArgumentTypeError(f"{s}: must be on an hourly boundary")
    return t


def parse_sat(s):
    n = int(s.lower().lstrip("g"))
    if n != 18:
        raise argparse.ArgumentTypeError(f"unsupported satellite: {s}")
    return n


def rawdb_request(action, **kwargs):
    from spaceweather_collector.rawdb import request
    return request(action, **kwargs)


class RawConnection:
    """Small collector cursor; the central service owns every persistent DB connection."""
    def __init__(self, sat, channel):
        self.sat, self.channel = f"G{sat}", channel
        self.rows, self.description = [], []

    def execute(self, sql, params=None):
        params = list(params or [])
        if sql.lstrip().upper().startswith('SELECT'):
            sql = sql.replace('FROM raw.suvi_hourly',
                              'FROM (SELECT * FROM raw.suvi_hourly WHERE sat = ? AND channel = ?)')
            # Slot queries place all existing placeholders after FROM.
            result = {'columns': [], 'rows': []}
            offset = 0
            while True:
                page = rawdb_request('query', sql=f'SELECT * FROM ({sql}) AS slots LIMIT 10000 OFFSET ?',
                                     params=[self.sat, self.channel, *params, offset])
                result['columns'] = page['columns']
                result['rows'].extend(page['rows'])
                if len(page['rows']) < 10000:
                    break
                offset += 10000
        else:
            if sql.lstrip().upper().startswith('UPDATE raw.suvi_hourly'.upper()):
                sql += ' AND sat = ? AND channel = ?'
                params += [self.sat, self.channel]
            result = rawdb_request('execute', sql=sql, params=params)
        self.description = [(name,) for name in result.get('columns', [])]
        timestamps = {'slot', 'obs_start', 'obs_end', 's3_modified'}
        self.rows = [tuple(datetime.fromisoformat(value).replace(tzinfo=None)
                           if name in timestamps and isinstance(value, str) else value
                           for name, value in zip(result.get('columns', []), row))
                     for row in result.get('rows', [])]
        return self

    def executemany(self, sql, rows):
        statements = []
        for row in rows:
            params = list(row)
            query = sql
            if sql.lstrip().upper().startswith('UPDATE RAW.SUVI_HOURLY'):
                query += ' AND sat = ? AND channel = ?'
                params += [self.sat, self.channel]
            statements.append({'sql': query, 'params': params})
        if statements:
            rawdb_request('execute', statements=statements)
        return self

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def close(self):
        pass


def db_path(a):
    return 'central solar_raw.duckdb (Unix socket)'


def connect(a):
    if getattr(a, 'db', None):
        raise ValueError('--db is retired; configure SOLAR_RAWDB_SOCKET for the central service')
    statements = [{'sql': part.strip()} for part in SCHEMA.split(';') if part.strip()]
    rawdb_request('execute', statements=statements)
    return RawConnection(a.sat, getattr(a, 'channel', 'Fe195'))


def last_slot(margin=MARGIN):
    """Newest hour whose data window has fully closed."""
    return (utcnow() - timedelta(seconds=margin)).replace(minute=0, second=0)


# ---------------------------------------------------------------- the one-record rule
def select_for_hour(sat, H, channel="Fe195", margin=MARGIN, cutoff=None, all_headers=False):
    """Return a row dict for slot H (status 'ok' or 'missing'). Raises on network errors."""
    row = dict(slot=H, sat=f"G{sat}", channel=channel, status="missing", n_frames=0, n_headers=0,
               cutoff_applied=cutoff is not None)
    files = [f for f in C.list_hour(sat, channel, H - HOUR)
             if f["obs_end"] + timedelta(seconds=margin) <= H
             and (cutoff is None or f["s3_modified"] <= cutoff)]
    row["n_frames"] = len(files)
    if not files:
        return row
    top = max(f["dur"] for f in files)
    cands = files if all_headers else [f for f in files if f["dur"] >= top - DUR_TOLERANCE]
    scored = []
    for f in cands:
        h = C.read_header(sat, f["key"])
        # eclipse-free first, then longest exposure (0.1 s bins: 1.0013 vs 1.0010 is noise), then latest
        expo = round(h["exptime"], 1) if h["exptime"] is not None else -1
        scored.append(((h["eclipse"] == 0, expo, f["obs_end"]), f, h))
    row["n_headers"] = len(scored)
    _, f, h = max(scored, key=lambda s: s[0])
    row.update(h, status="ok", filename=f["filename"], s3_key=f["key"], obs_start=f["obs_start"],
               obs_end=f["obs_end"], s3_modified=f["s3_modified"],
               age_min=(H - f["obs_end"]).total_seconds() / 60)
    row.pop("obs_time", None)
    return row


def save(con, rows):
    if not rows:
        return
    sql = f"INSERT OR REPLACE INTO raw.suvi_hourly ({', '.join(COLS)}) VALUES ({', '.join('?' * len(COLS))})"
    con.executemany(sql, [[r.get(c) for c in COLS] for r in rows])


def nc_dest(root, row):
    """data/g18/<YYYY>/<DOY>/<filename>.nc"""
    t = row["obs_end"]
    return Path(root) / row["sat"].lower() / f"{t:%Y}" / f"{t.timetuple().tm_yday:03d}" / \
        row["filename"].replace(".fits", ".nc")


def fetch_nc(sat, row, root):
    """Keep the raw body durable before registering it or publishing its slot."""
    from spaceweather_collector.rawdb import sha256_file
    dest = nc_dest(root, row)
    size = C.download(sat, row['s3_key'], dest)
    if size is None:
        return None, None
    # Existing C.download atomically renames; fsync both body and parent before DB commit.
    for target in (dest, dest.parent):
        fd = os.open(target, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    channel = row.get('channel', 'Fe195')
    rawdb_request('register_files', records=[{
        'path': str(dest.resolve()), 'sha256': sha256_file(dest), 'bytes': size,
        'source': 'suvi', 'product': channel,
        'url': f"{C.base(sat)}/{row['s3_key'].rsplit('.', 1)[0]}.nc",
        'obs_start': row.get('obs_start'), 'obs_end': row.get('obs_end'),
        'http_last_modified': row.get('s3_modified'),
        'fetched_at_utc': datetime.now(C.UTC).isoformat(),
        'metadata_json': {'sat': row['sat'], 'channel': channel, 'slot': row.get('slot')},
    }])
    return str(dest.resolve()), size


def run_slots(slots, a, cutoff_for=lambda H: None):
    """Fetch slots in parallel (threads); yields (H, row, None) or (H, None, exc)."""
    def job(H):
        try:
            row = select_for_hour(a.sat, H, a.channel, a.margin, cutoff_for(H), a.all_headers)
            if row["status"] == "ok" and getattr(a, "download_dir", None):
                row["nc_path"], row["nc_size"] = fetch_nc(a.sat, row, a.download_dir)
            return H, row, None
        except Exception as exc:   # noqa: BLE001 - network errors must not kill the batch
            return H, None, exc
    with ThreadPoolExecutor(a.workers) as ex:
        yield from ex.map(job, slots)


# ---------------------------------------------------------------- commands
def cmd_init(a):
    connect(a)
    print(f"schema ready: {db_path(a)}")


def cmd_backfill(a):
    con = connect(a)
    start = a.start or START
    end = a.end or last_slot(a.margin) + HOUR          # exclusive
    slots, t = [], start
    while t < end:
        slots.append(t)
        t += HOUR
    keep = "status = 'ok'" if a.retry_missing else "TRUE"
    have = {r[0] for r in con.execute(f"SELECT slot FROM raw.suvi_hourly WHERE {keep}").fetchall()}
    todo = [s for s in slots if s not in have]
    print(f"G{a.sat}: {start:%Y-%m-%d %H}Z .. {end - HOUR:%Y-%m-%d %H}Z  {len(slots)} hours, "
          f"{len(todo)} to do, workers={a.workers}", flush=True)
    n_ok = n_missing = n_err = 0
    buf, t0 = [], time.time()
    for i, (H, row, exc) in enumerate(run_slots(todo, a), 1):
        if exc:
            n_err += 1
            print(f"[error] {H:%Y-%m-%d %H}Z {exc!r}", file=sys.stderr)   # not saved -> retried next run
        else:
            buf.append(row)
            n_ok += row["status"] == "ok"
            n_missing += row["status"] == "missing"
        if len(buf) >= a.batch or i == len(todo):
            save(con, buf)
            buf = []
        if i % a.every == 0 or i == len(todo):
            eta = (time.time() - t0) / i * (len(todo) - i)
            print(f"[{i}/{len(todo)}] {H:%Y-%m-%d %H}Z ok={n_ok} missing={n_missing} error={n_err} "
                  f"eta={eta / 60:.1f}min", flush=True)
    print(f"done: ok={n_ok} missing={n_missing} error={n_err}")
    sys.exit(1 if n_err else 0)


def cmd_hour(a):
    con = connect(a)
    (H, row, exc), = run_slots([a.at], a, lambda H: H if a.realtime else None)
    if exc:
        sys.exit(f"error: {exc!r}")
    save(con, [row])
    if row["status"] == "ok":
        print(f"{H:%Y-%m-%d %H}Z -> {row['filename']}\n  exptime {row['exptime']} s, age {row['age_min']:.1f} min, "
              f"LastModified {row['s3_modified']}, frames {row['n_frames']}, headers read {row['n_headers']}")
    else:
        print(f"{H:%Y-%m-%d %H}Z -> MISSING (frames in window: {row['n_frames']})")


def cmd_live(a):
    """At each new hour (+ a few seconds) store the slot with the LastModified <= H cutoff.

    Also fills any slot in the last --catchup hours that is not in the DB yet.
    """
    if a.drive_sync:
        raise RuntimeError('Drive publication is retired; raw data stays internal')
    con = connect(a)
    while True:
        H = utcnow().replace(minute=0, second=0)
        slots = [H - i * HOUR for i in range(a.catchup, -1, -1)]
        have = {r[0] for r in con.execute("SELECT slot FROM raw.suvi_hourly").fetchall()}
        for s, row, exc in run_slots([s for s in slots if s not in have], a, lambda s: s):
            if exc:
                print(f"[error] {s:%Y-%m-%d %H}Z {exc!r}", file=sys.stderr)
                continue
            save(con, [row])
            print(f"[hour] {s:%Y-%m-%d %H}Z " + (f"{row['filename']} exptime={row['exptime']} "
                  f"age={row['age_min']:.1f}min" if row["status"] == "ok" else "MISSING"), flush=True)
        if a.once:
            return
        nxt = H + HOUR + timedelta(seconds=a.delay)
        time.sleep(max(1, (nxt - utcnow()).total_seconds()))


def cmd_download(a):
    """Download the .nc of every 'ok' row that has none yet (parallel, resumable).

    Needs the DB write lock, so run it after (or instead of) a backfill on the same DB.
    """
    con = connect(a)
    where, args = "status = 'ok' AND nc_path IS NULL", []
    if a.start:
        where += " AND slot >= ?"
        args.append(a.start)
    if a.end:
        where += " AND slot < ?"
        args.append(a.end)
    rows = con.execute(f"SELECT slot, sat, channel, filename, s3_key, obs_start, obs_end, s3_modified FROM raw.suvi_hourly WHERE {where} "
                       f"ORDER BY slot" + (f" LIMIT {a.limit}" if a.limit else ""), args).fetchall()
    rows = [dict(zip(("slot", "sat", "channel", "filename", "s3_key", "obs_start", "obs_end", "s3_modified"), r)) for r in rows]
    print(f"G{a.sat}: {len(rows)} files to download -> {a.dir}/g{a.sat}/, workers={a.workers}", flush=True)

    def job(r):
        try:
            return r, *fetch_nc(a.sat, r, a.dir), None
        except Exception as exc:   # noqa: BLE001
            return r, None, None, exc
    n_ok = n_err = n_gone = 0
    nbytes, buf, t0 = 0, [], time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (r, path, size, exc) in enumerate(ex.map(job, rows), 1):
            if exc:
                n_err += 1
                print(f"[error] {r['slot']:%Y-%m-%d %H}Z {exc!r}", file=sys.stderr)
            elif path is None:
                n_gone += 1
                print(f"[no .nc on S3] {r['filename']}", file=sys.stderr)
            else:
                n_ok += 1
                nbytes += size
                buf.append([path, size, r["slot"]])
            if len(buf) >= a.batch or i == len(rows):
                con.executemany("UPDATE raw.suvi_hourly SET nc_path = ?, nc_size = ? WHERE slot = ?", buf)
                buf = []
            if i % a.every == 0 or i == len(rows):
                eta = (time.time() - t0) / i * (len(rows) - i)
                print(f"[{i}/{len(rows)}] {r['slot']:%Y-%m-%d %H}Z ok={n_ok} error={n_err} gone={n_gone} "
                      f"{nbytes / 1e9:.1f}GB eta={eta / 60:.1f}min", flush=True)
    print(f"done: ok={n_ok} error={n_err} gone={n_gone}")
    sys.exit(1 if n_err or n_gone else 0)


def cmd_status(a):
    con = connect(a)
    cur = con.execute("""SELECT count(*) AS hours, count(*) FILTER (status='ok') AS ok,
                                count(*) FILTER (status='missing') AS missing, min(slot) AS first, max(slot) AS last,
                                round(median(exptime), 3) AS median_exptime, round(median(age_min), 1) AS median_age_min
                         FROM raw.suvi_hourly""")
    for name, val in zip([d[0] for d in cur.description], cur.fetchone()):
        print(f"{name:>16}: {val}")
    n, size = con.execute("SELECT count(nc_path), coalesce(sum(nc_size), 0) FROM raw.suvi_hourly").fetchone()
    print(f"{'nc downloaded':>16}: {n} files, {size / 1e9:.2f} GB")


def cmd_check(a):
    """Leakage assertions: nothing observed later than slot - margin, nothing published after slot."""
    con = connect(a)
    n, = con.execute("SELECT count(*) FROM raw.suvi_hourly WHERE status = 'ok'").fetchone()
    late, = con.execute("SELECT count(*) FROM raw.suvi_hourly WHERE status = 'ok' "
                        "AND epoch(slot - obs_end) < ?", [a.margin]).fetchone()
    pub, = con.execute("SELECT count(*) FROM raw.suvi_hourly WHERE cutoff_applied AND s3_modified > slot").fetchone()
    old, = con.execute("SELECT count(*) FROM raw.suvi_hourly WHERE status = 'ok' AND age_min > 60").fetchone()
    print(f"rows ok={n}  obs_end+margin>slot: {late}  LastModified>slot (cutoff rows): {pub}  age>60min: {old}")
    bad = [p for p, s in con.execute("SELECT nc_path, nc_size FROM raw.suvi_hourly WHERE nc_path IS NOT NULL").fetchall()
           if not Path(p).exists() or Path(p).stat().st_size != s]
    print(f"nc files missing/size mismatch: {len(bad)}")
    sys.exit(1 if late or pub or old or bad else 0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s, workers=32):
        s.add_argument("--sat", type=parse_sat, required=True, help="g18 only")
        s.add_argument("--channel", default="Fe195")
        s.add_argument("--margin", type=int, default=MARGIN, help="seconds: require obs_end + margin <= hour")
        s.add_argument("--db", type=Path, help="retired: use the central raw DB socket")
        s.add_argument("--workers", type=int, default=workers, help="parallel slot fetches (threads)")
        s.add_argument("--all-headers", action="store_true",
                       help="read every frame's header instead of only the longest-by-filename ones")

    i = sub.add_parser("init")
    i.add_argument("--sat", type=parse_sat, required=True)
    i.add_argument("--db", type=Path)
    i.add_argument("--channel", default="Fe195")
    b = sub.add_parser("backfill")
    common(b)
    b.add_argument("--start", type=parse_hour, help="default: START (2022-08-22T21:00)")
    b.add_argument("--end", type=parse_hour, help="exclusive; default: newest closed hour")
    b.add_argument("--retry-missing", action="store_true", help="also redo hours stored as missing")
    b.add_argument("--batch", type=int, default=200, help="rows per DB write")
    b.add_argument("--download-dir", help="also download each selected frame's .nc here (e.g. data)")
    b.add_argument("--every", type=int, default=200, help="progress interval (hours)")
    h = sub.add_parser("hour")
    common(h, workers=1)
    h.add_argument("--at", type=parse_hour, required=True)
    h.add_argument("--realtime", action="store_true", help="also require LastModified <= hour")
    h.add_argument("--download-dir", help="also download the selected frame's .nc here")
    lv = sub.add_parser("live")
    common(lv, workers=4)
    lv.add_argument("--once", action="store_true")
    lv.add_argument("--download-dir", help="also download each selected frame's .nc here")
    lv.add_argument("--drive-sync", action="store_true",
                    help="retired; external publication is disabled")
    lv.add_argument("--catchup", type=int, default=3, help="also fill missing slots of the last N hours")
    lv.add_argument("--delay", type=int, default=10, help="seconds after the hour before collecting")
    d = sub.add_parser("download")
    d.add_argument("--sat", type=parse_sat, required=True)
    d.add_argument("--db", type=Path)
    d.add_argument("--channel", default="Fe195")
    d.add_argument("--dir", default=str(HERE / "data"), help="root dir; files go to <dir>/g18/YYYY/DOY/")
    d.add_argument("--workers", type=int, default=16)
    d.add_argument("--start", type=parse_hour)
    d.add_argument("--end", type=parse_hour, help="exclusive")
    d.add_argument("--limit", type=int, help="only the first N pending files (testing)")
    d.add_argument("--batch", type=int, default=200)
    d.add_argument("--every", type=int, default=200)
    for name in ("status", "check"):
        s = sub.add_parser(name)
        s.add_argument("--sat", type=parse_sat, required=True)
        s.add_argument("--db", type=Path)
        s.add_argument("--channel", default="Fe195")
        s.add_argument("--margin", type=int, default=MARGIN)

    a = p.parse_args()
    {"init": cmd_init, "backfill": cmd_backfill, "hour": cmd_hour, "live": cmd_live,
     "download": cmd_download, "status": cmd_status, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
