"""Measure acquisition latency (publication time - observation time) per SolarWind DB dataset.

Read-only over local evidence: HTTP Last-Modified / creation / submission records already collected.
Writes reports/latency-<date>.json; the sheet column '입수지연' is built from it.
A recent window is used for "typical NRT latency" because old archive Last-Modified values are often
rewritten by later reprocessing. Run with the nrt-collect venv (pandas); SUVI uses the sun-img venv (duckdb).

    /home/t-lab01/sunrun/nrt-collect/.venv/bin/python latency_report.py
"""
import datetime as dt
import glob
import json
import re
import shutil
import statistics
import subprocess
import tempfile
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd

HOME = Path("/home/t-lab01")
V2 = HOME / "sunrun/data-v2/model-inputs/leaderboard-store/v2"
LB = HOME / "sunrun/leaderboard"
STORE_ROOT = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store')
NOW = dt.datetime.now(dt.timezone.utc)
RECENT = NOW - dt.timedelta(days=60)
UTC = dt.timezone.utc


def lm(value):
    try:
        return parsedate_to_datetime(value).astimezone(UTC) if value else None
    except (TypeError, ValueError):
        return None


def iso(value):
    if not value or value in ("None", "null"):
        return None
    v = value.replace("Z", "+00:00")
    t = dt.datetime.fromisoformat(v)
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def summary(lags_h, basis, period=None):
    lags = sorted(x for x in lags_h if x is not None and x > -1)
    if not lags:
        return {"n": 0, "basis": basis}
    q = statistics.quantiles(lags, n=10) if len(lags) >= 10 else None
    return {"median_h": round(statistics.median(lags), 2), "p90_h": round(q[8], 2) if q else round(max(lags), 2),
            "n": len(lags), "basis": basis, "period": period}


def fetchlog(name):
    path = V2 / name / "fetch-log.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


out = {}

# ACE: daily lists are created at D+1 ~00:11 UT; the real-time 2-hour list lags ~2-3 min (2026-10-12 check).
ace = pd.read_csv(HOME / "spaceweather-collector/data/ace_release/ace_release_days.csv")
ace = ace[ace["product"].eq("swepam_1m")]
out["ace"] = {**summary((ace["created_minus_day_end_min"] / 60).tolist(),
                         "daily list :Created: - day end (SWEPAM 1m)", "2025-06..2026-10"),
              "note": "daily list D+1 ~00:11 UT, real-time 2-hour list ~2-3 min"}

# SUVI Fe195 / Fe171: S3 LastModified - obs_end (copy DB + WAL, read with duckdb in the sun-img venv).
for key, db in (("suvi195", "solar_g18.duckdb"), ("suvi171", "solar_g18_fe171.duckdb")):
    with tempfile.TemporaryDirectory() as tmp:
        src = HOME / "sunrun/sun-img" / db
        shutil.copy(src, Path(tmp) / db)
        if Path(str(src) + ".wal").exists():
            shutil.copy(str(src) + ".wal", Path(tmp) / (db + ".wal"))
        code = ("import duckdb,json,sys; c=duckdb.connect(sys.argv[1],read_only=True);"
                "r=c.execute(\"select epoch(s3_modified-obs_end)/3600.0 from suvi_hourly where status='ok' and "
                "s3_modified is not null and obs_end>=now()-interval 60 day\").fetchall();print(json.dumps([x[0] for x in r]))")
        res = subprocess.run([str(HOME / "sunrun/sun-img/.venv/bin/python"), "-c", code, str(Path(tmp) / db)],
                             capture_output=True, text=True)
        lags = json.loads(res.stdout) if res.returncode == 0 and res.stdout.strip() else []
    out[key] = summary(lags, "S3 LastModified - obs_end", "last 60 days")

# GONG: v2 fetch-log Last-Modified - map observation time (recent maps only).
lags = []
for r in fetchlog("gong-mrzqs"):
    m = re.search(r"mrzqs(\d{6})t(\d{4})", r.get("file", ""))
    t, l = (dt.datetime.strptime(m.group(1) + m.group(2), "%y%m%d%H%M").replace(tzinfo=UTC) if m else None), lm(r.get("last_modified"))
    if t and l and t >= RECENT:
        lags.append((l - t).total_seconds() / 3600)
out["gong"] = summary(lags, "HTTP Last-Modified - map time (NSO archive)", "last 60 days")
if not lags:  # fall back to the v1 manifest
    m = json.load(open(LB / "experiments/cmehcs72_20261014/gong_manifest.json"))
    items = m if isinstance(m, list) else (m.get("maps") or m.get("files") or [])
    out["gong"] = summary([x.get("lag_hours") for x in items if isinstance(x, dict)], "v1 manifest lag_hours")

# LASCO C2: Last-Modified - frame time.
lags = []
for r in fetchlog("lasco-c2-512"):
    m = re.search(r"(\d{8})_(\d{4})_c2_512", r.get("file", ""))
    t, l = (dt.datetime.strptime("".join(m.groups()), "%Y%m%d%H%M").replace(tzinfo=UTC) if m else None), lm(r.get("last_modified"))
    if t and l and t >= RECENT:
        lags.append((l - t).total_seconds() / 3600)
out["lasco"] = summary(lags, "HTTP Last-Modified - frame time (REPROCESSING dir)", "last 60 days")

# STEREO HI1/HI2 and ISEE STEL from the nrt-collect index (live fetches only).
idx = pd.concat([pd.read_parquet(p) for p in glob.glob(str(HOME / "sunrun/nrt-collect/data/*/index/part-*.parquet"))])
for prod in ("hi_1", "hi_2"):
    g = idx[(idx["product"] == prod) & (idx["mode"] == "live")]
    lags = []
    for _, r in g.iterrows():
        t, l = iso(r["obs_end"] or r["obs_start"]), lm(r["http_last_modified"])
        if t and l and t >= RECENT:
            lags.append((l - t).total_seconds() / 3600)
    out[prod] = summary(lags, "HTTP Last-Modified - DATE-OBS (beacon)", "live fetches since 2026-10-14")
lags = []
for _, r in idx[idx["product"] == "isee_stel"].iterrows():
    path = HOME / "sunrun/nrt-collect/data/ips" / r["relpath"]
    l = lm(r["http_last_modified"])
    if path.exists() and l:
        stamps = re.findall(r"^(\d{8} \d\d:\d\d:\d\d)", path.read_text(errors="ignore"), re.M)
        if stamps:
            last = dt.datetime.strptime(stamps[-1], "%Y%m%d %H:%M:%S").replace(tzinfo=UTC)
            lags.append((l - last).total_seconds() / 3600)
out["isee"] = {**summary(lags, "file Last-Modified - last observation in that file version", "versions seen 2026-10-14..15"),
               "note": "STEL file regenerated daily ~17:20 UT (directory listing shows JST); observations 20:00-09:00 UT"}

# DONKI WSA-Enlil: modelCompletionTime - CME start; CME analyses: submissionTime - time21_5.
lags_e, lags_a, lags_c = [], [], []
for f in glob.glob(str(V2 / "donki/WSAEnlilSimulations_2026-0[6-9].json")):
    for r in json.load(open(f)):
        starts = [iso(c.get("cmeStartTime")) for c in (r.get("cmeInputs") or []) if c.get("cmeStartTime")]
        done = iso(r.get("modelCompletionTime"))
        if starts and done:
            lags_e.append((done - max(starts)).total_seconds() / 3600)
for f in glob.glob(str(V2 / "donki/CMEAnalysis_2026-0[6-9].json")):
    for r in json.load(open(f)):
        s, t = iso(r.get("submissionTime")), iso(r.get("time21_5"))
        if s and t:
            lags_a.append((s - t).total_seconds() / 3600)
for f in glob.glob(str(V2 / "donki/CME_2026-0[6-9].json")):
    for r in json.load(open(f)):
        s, t = iso(r.get("submissionTime")), iso(r.get("startTime"))
        if s and t:
            lags_c.append((s - t).total_seconds() / 3600)
out["donki_enlil"] = summary(lags_e, "modelCompletionTime - latest input CME start", "2026-06..09")
out["donki_cmeanalysis"] = summary(lags_a, "submissionTime - time21_5", "2026-06..09")
out["donki_cme"] = summary(lags_c, "submissionTime - CME startTime", "2026-06..09")

# CCMC CME Scoreboard: submissionTime - CME time in cmeID.
lags = []
for f in glob.glob(str(V2 / "ccmc-cme-scoreboard/api/*.json")):
    for r in json.load(open(f)):
        m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)", str(r.get("cmeID", "")))
        if not m:
            continue
        t = dt.datetime.fromisoformat(m.group(1)).replace(tzinfo=UTC)
        if t < dt.datetime(2026, 6, 1, tzinfo=UTC):
            continue
        for pr in r.get("predictions") or []:
            s = iso(pr.get("submissionTime"))
            if s:
                lags.append((s - t).total_seconds() / 3600)
out["ccmc"] = summary(lags, "submissionTime - CME time (cmeID)", "CMEs since 2026-06")

# NOAA WSA-Enlil ambient: archive Last-Modified - init date 00:00.
lags = []
for f in glob.glob(str(V2 / "enlil-ambient/series/*.http.json")):
    r = json.load(open(f))
    m = re.search(r"_(\d{8})_0000\.tar\.gz", r.get("url", ""))
    l = lm(r.get("http_last_modified"))
    if m and l:
        lags.append((l - dt.datetime.strptime(m.group(1), "%Y%m%d").replace(tzinfo=UTC)).total_seconds() / 3600)
out["enlil_ambient"] = summary(lags, "archive Last-Modified - init 00:00", "runs fetched 2026-10-14")

# PUNCH QAM: recent file Last-Modified - observation; early-mission backfill reported separately.
lags = []
for f in glob.glob(str(V2 / "punch-qam/raw/*/*.http.json")):
    r = json.load(open(f))
    m = re.search(r"QAM_(\d{14})_", r.get("url", ""))
    l = lm(r.get("http_last_modified"))
    if m and l:
        t = dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        if t >= dt.datetime(2026, 9, 1, tzinfo=UTC):
            lags.append((l - t).total_seconds() / 3600)
out["punch"] = {**summary(lags, "HTTP Last-Modified - observation", "2026-09..10"),
                "note": "June 2026 frames were published 2026-07-28 (backfill), so early history is weeks late"}

# GOES SEP daily files: Last-Modified - end of that day.
lags = []
for r in json.load(open(STORE_ROOT / "goes-sep72-20261014/source_manifest.json"))["files"]:
    m = re.search(r"_d(\d{8})_", r.get("file", ""))
    l = lm(r.get("http_last_modified"))
    if m and l:
        end = dt.datetime.strptime(m.group(1), "%Y%m%d").replace(tzinfo=UTC) + dt.timedelta(days=1)
        if end >= RECENT:
            lags.append((l - end).total_seconds() / 3600)
out["sep"] = summary(lags, "daily file Last-Modified - day end (NCEI, v1 manifest)", "last 60 days of v1 manifest")

# SWPC event reports (NGDC archive): Last-Modified - day end, from the v1 Type II manifest.
man = json.load(open(STORE_ROOT / "typeii72_20261014/reports_manifest.json"))
items = man.get("reports") or man.get("files") or man
lags = []
for r in items if isinstance(items, list) else []:
    l = lm((r.get("headers") or {}).get("last-modified"))
    if l and r.get("day"):
        end = dt.datetime.fromisoformat(r["day"]).replace(tzinfo=UTC) + dt.timedelta(days=1)
        if end >= dt.datetime(2026, 6, 1, tzinfo=UTC):
            lags.append((l - end).total_seconds() / 3600)
out["swpc_events"] = {**summary(lags, "NGDC archive Last-Modified - report day end", "2026-06..09"),
                      "note": "the live SWPC text is current within minutes"}

# SWPC forecast discussion: archive Last-Modified - issue time.
lags = []
_rec = json.load(open(HOME / "sunrun/research/notes/swpc76/receipts.json"))
for r in (_rec if isinstance(_rec, list) else list(_rec.values())):
    m = re.search(r"(\d{12})forecast_discussion", r.get("url", ""))
    l = lm(r.get("last_modified"))
    if m and l:
        t = dt.datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=UTC)
        if t >= RECENT:
            lags.append((l - t).total_seconds() / 3600)
out["discussion"] = summary(lags, "NGDC archive Last-Modified - issue time (v1 receipts)", "last 60 days")

# Rule / documented values without per-record evidence.
out["rules"] = {
    "hmi": "규칙 T_REC+4h (JSOC NRT 키워드, 레코드별 공개 시각 없음)",
    "xrs": "규칙 측정+10분 (운영 1분 자료). NCEI 연 파일은 일 단위 갱신(실측 약 1일)",
    "k1": "실측 약 242h(아카이브 끝 vs 2026-10-14 조회) / 규칙 244h",
    "l2": "실측 수개월(2026-10-14 기준 MAG 끝 08-04 약 70일, SWEPAM 2024-07) — 인과 사용 안 함",
    "nmdb": "규칙 시간 bin 끝+64분(spec latency 4분 + bin)",
    "cactus": "live 페이지 6시간 주기 / 월간 카탈로그는 다음 달 초 발행",
}
out["generated_utc"] = NOW.isoformat()
target = Path(__file__).resolve().parent / "reports" / f"latency-{NOW:%Y%m%d}.json"
target.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print(json.dumps(out, indent=1, ensure_ascii=False))
