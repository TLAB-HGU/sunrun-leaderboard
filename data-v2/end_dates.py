"""Latest actual data time per SolarWind DB dataset (for the sheet column '끝').

Reads the local copies (live source, v1 store, v2 store) and reports the newest observation/origin
actually present, with the copy it came from. Read-only. Run with the nrt-collect venv:

    /home/t-lab01/sunrun/nrt-collect/.venv/bin/python end_dates.py   -> reports/end-dates-<date>.json
"""
import datetime as dt
import glob
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

H = Path("/home/t-lab01")
S = H / "sunrun/data-v2/model-inputs/leaderboard-store"
V2 = S / "v2"
UTC = dt.timezone.utc
TIME_COLS = ("observation_time_utc", "origin_ns", "timestamp_utc", "slot", "origin", "origin_utc", "origin_last_input_utc", "time", "obs_time",
             "date_obs_utc", "map_time", "t", "timestamp")


def ts(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    t = pd.Timestamp(x)
    return (t.tz_localize(UTC) if t.tzinfo is None else t.tz_convert(UTC)).to_pydatetime()


def pq_max(pattern, require_values=True):
    files = sorted(glob.glob(str(pattern)))
    if not files:
        return None
    best = None
    for f in files:
        df = pd.read_parquet(f)
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
            mask = df.notna().any(axis=1) if (require_values and df.shape[1]) else pd.Series(True, index=idx)
            cand = idx[mask.to_numpy()].max() if mask.any() else None
        else:
            col = next((c for c in TIME_COLS if c in df.columns), None)
            if col is None:
                continue
            if col == "origin_ns":  # stored as integers; pandas 3 writers used microseconds
                unit = "us" if df[col].max() < 1e17 else "ns"
                times = pd.to_datetime(df[col], unit=unit, utc=True, errors="coerce")
            else:
                times = pd.to_datetime(df[col], utc=True, errors="coerce")
            if require_values:
                vals = df.drop(columns=[col]).select_dtypes("number")
                if "was_missing" in vals:
                    keep = vals["was_missing"].eq(0)
                elif vals.shape[1]:
                    keep = vals.notna().any(axis=1)
                else:
                    keep = pd.Series(True, index=df.index)
                times = times[keep.to_numpy()]
            cand = times.max() if len(times) else None
        c = ts(cand)
        if c and (best is None or c > best):
            best = c
    return best


def name_max(root, regex, fmt, pattern="**/*"):
    best = None
    rx = re.compile(regex)
    for p in Path(root).glob(pattern) if Path(root).exists() else []:
        m = rx.search(p.name)
        if m and p.is_file() and p.stat().st_size > 0:
            try:
                t = dt.datetime.strptime("".join(m.groups()), fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
            if best is None or t > best:
                best = t
    return best


def json_max(pattern, *keys):
    best = None
    for f in glob.glob(str(pattern)):
        try:
            data = json.load(open(f))
        except ValueError:
            continue
        for r in data if isinstance(data, list) else []:
            for k in keys:
                t = ts(r.get(k)) if isinstance(r, dict) and r.get(k) else None
                if t and (best is None or t > best):
                    best = t
    return best


def psql(q):
    out = subprocess.run(["psql", "-d", "spaceweather", "-Atc", q], capture_output=True, text=True).stdout.strip()
    return ts(out) if out else None


def duck(db, q):
    with tempfile.TemporaryDirectory() as tmp:
        src = H / "sunrun/sun-img" / db
        shutil.copy(src, Path(tmp) / db)
        if Path(str(src) + ".wal").exists():
            shutil.copy(str(src) + ".wal", Path(tmp) / (db + ".wal"))
        code = "import duckdb,sys;c=duckdb.connect(sys.argv[1],read_only=True);print(c.execute(sys.argv[2]).fetchone()[0])"
        out = subprocess.run([str(H / "sunrun/sun-img/.venv/bin/python"), "-c", code, str(Path(tmp) / db), q],
                             capture_output=True, text=True).stdout.strip()
    return ts(out) if out and out != "None" else None


def isee_last(path):
    best = None
    for f in glob.glob(str(path)):
        stamps = re.findall(r"^(\d{8} \d\d:\d\d:\d\d)", Path(f).read_text(errors="ignore"), re.M)
        if stamps:
            t = dt.datetime.strptime(stamps[-1], "%Y%m%d %H:%M:%S").replace(tzinfo=UTC)
            best = t if best is None or t > best else best
    return best


def hmi_last(root):
    best = None
    for f in sorted(glob.glob(str(Path(root) / "*.json")))[-3:]:
        try:
            d = json.load(open(f))
        except ValueError:
            continue
        for k in d.get("keywords", []):
            if k.get("name") == "T_REC":
                for v in k.get("values", []):
                    m = re.match(r"(\d{4})\.(\d\d)\.(\d\d)_(\d\d):(\d\d)", v)
                    if m:
                        t = dt.datetime(*map(int, m.groups()), tzinfo=UTC)  # TAI ~ UTC+37s, ignored
                        best = t if best is None or t > best else best
    return best


def cactus_last(root):
    best = None
    for f in glob.glob(str(Path(root) / "**/*.html"), recursive=True):
        for m in re.finditer(r"(20\d\d)/(\d\d)/(\d\d) (\d\d):(\d\d)", Path(f).read_text(errors="ignore")):
            t = dt.datetime(*map(int, m.groups()), tzinfo=UTC)
            best = t if best is None or t > best else best
    return best


def xrs_last(nc):
    code = ("import xarray as xr,sys;d=xr.open_dataset(sys.argv[1]);v=d['xrsb_flux'] if 'xrsb_flux' in d else d[list(d.data_vars)[0]];"
            "t=v['time'][v.notnull()] if 'time' in v.dims else d['time'];print(str(t.max().values))")
    out = subprocess.run([str(H / "sunrun/leaderboard/.venv-fusion/bin/python"), "-c", code, str(nc)], capture_output=True, text=True)
    return ts(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else None


def manifest_max(path, keys=("date_obs_utc", "observation_time_utc", "obs_time_utc", "date_obs")):
    m = json.load(open(path))
    items = m if isinstance(m, list) else (m.get("maps") or m.get("files") or [])
    vals = [ts(x.get(k)) for x in items if isinstance(x, dict) for k in keys if x.get(k)]
    return max(vals) if vals else None


ACE_PG = "select max(timestamp_utc) from solar where ace_speed_kms is not null or ace_bt_nt is not null"
L2_PG = "select max(timestamp_utc) from solar_l2 where l2_mag_bt_nt is not null or l2_swepam_speed_kms is not null"
STAGE = H / ".local/state/sundb/stage"
GONG_RX, GONG_FMT = r"mrzqs(\d{6})t(\d{4})", "%y%m%d%H%M"
NRT = H / "sunrun/nrt-collect/data"

SOURCES = {
    "SW-001": [("수집기 DB", lambda: psql(ACE_PG))],
    "SW-002": [("sun-db stage", lambda: pq_max(STAGE / "ace_solar/year=*/part.parquet"))],
    "SW-003": [("v1", lambda: pq_max(S / "ch-breakthrough-v2/upload/raw-ace-snapshot/year=*/part.parquet")),
               ("v2", lambda: pq_max(V2 / "ace-snapshot/year=*/part.parquet"))],
    "SW-004": [("v1", lambda: pq_max(S / "ch-v1/ace.parquet")), ("v2", lambda: pq_max(V2 / "ch/ace.parquet"))],
    "SW-005": [("v1", lambda: pq_max(S / "ch-v1/ch-hourly.parquet")), ("v2", lambda: pq_max(V2 / "ch/ch-hourly.parquet"))],
    "SW-006": [("v1", lambda: pq_max(S / "suvi-fusion/suvi-hourly-v2.parquet")),
               ("v2", lambda: pq_max(V2 / "suvi-fusion/suvi-hourly-v2.parquet"))],
    "SW-007": [("sun-db stage", lambda: pq_max(STAGE / "suvi_hourly/year=*/part.parquet", require_values=False)),
               ("v2", lambda: pq_max(V2 / "suvi-meta/year=*/part.parquet", require_values=False))],
    "SW-008": [("SUVI DB", lambda: duck("solar_g18.duckdb", "select max(slot) from suvi_hourly where status='ok'"))],
    "SW-009": [("v1", lambda: json_max(S / "headroom-audit/donki/WSAEnlilSimulations_*.json", "modelCompletionTime")),
               ("v2", lambda: json_max(V2 / "donki/WSAEnlilSimulations_*.json", "modelCompletionTime"))],
    "SW-010": [("v1", lambda: json_max(S / "donki-live/WSAEnlilSimulations_*.json", "modelCompletionTime")),
               ("v2", lambda: json_max(V2 / "donki/WSAEnlilSimulations_*.json", "modelCompletionTime"))],
    "SW-012": [("v1", lambda: max(filter(None, [name_max(S / "scratch/pfssconn_20261014/gong_fits", GONG_RX, GONG_FMT),
                                                 name_max(S / "scratch/cmehcs72_20261014_gong_fits", GONG_RX, GONG_FMT)]), default=None)),
               ("v2", lambda: name_max(V2 / "gong-mrzqs", GONG_RX, GONG_FMT))],
    "SW-013": [("v1", lambda: name_max(S / "alphahe72-cache", r"_(\d{8})_v\d+\.cdf$", "%Y%m%d")),
               ("v2", lambda: name_max(V2 / "ace-k1", r"_(\d{8})_v\d+\.cdf$", "%Y%m%d"))],
    "SW-014": [("수집기 DB", lambda: psql(L2_PG))],
    "SW-015": [("수집기", lambda: ts(pd.read_csv(H / "spaceweather-collector/data/ace_release/ace_release_days.csv")["last_obs_utc"].max()))],
    "SW-024": [("v1", lambda: name_max(S / "solar-inc-ambsw72-20261014", r"^(\d{8})\.parquet$", "%Y%m%d")),
               ("v2", lambda: name_max(V2 / "enlil-ambient", r"^(\d{8})\.parquet$", "%Y%m%d"))],
    "SW-025": [("v1", lambda: json_max(S / "donki-cme/CME*_*.json", "startTime", "time21_5")),
               ("v2", lambda: json_max(V2 / "donki/CME*_*.json", "startTime", "time21_5"))],
    "SW-027": [("v2", lambda: json_max(V2 / "ccmc-cme-scoreboard/api/*.json", "observedTime"))],
    "SW-028": [("v1", lambda: name_max(S / "solar-inc-lascovision-20261014/frames", r"(\d{8})_(\d{4})_c2_512", "%Y%m%d%H%M")),
               ("v2", lambda: name_max(V2 / "lasco-c2-512/frames", r"(\d{8})_(\d{4})_c2_512", "%Y%m%d%H%M"))],
    "SW-030": [("nrt-collect", lambda: name_max(NRT / "stereo_a/raw/hi_1", r"(\d{8})_(\d{6})_", "%Y%m%d%H%M%S"))],
    "SW-031": [("nrt-collect", lambda: name_max(NRT / "stereo_a/raw/hi_2", r"(\d{8})_(\d{6})_", "%Y%m%d%H%M%S"))],
    "SW-032": [("nrt-collect", lambda: isee_last(NRT / "ips/raw/stel/STEL2026*.dat"))],
    "SW-033": [("v2", lambda: name_max(V2 / "punch-qam/raw", r"QAM_(\d{14})_", "%Y%m%d%H%M%S", "**/*.fits"))],
    "SW-034": [("v1", lambda: name_max(S / "solar-inc-typeiiie72-20261014/swpc-events/raw", r"^(\d{8})events", "%Y%m%d")),
               ("v2", lambda: name_max(V2 / "swpc-events/raw", r"^(\d{8})events", "%Y%m%d"))],
    "SW-036": [("v1", lambda: name_max(S / "goes-sep72-20261014/raw", r"_d(\d{8})_", "%Y%m%d")),
               ("v2", lambda: name_max(V2 / "goes18-sep/raw", r"_d(\d{8})_", "%Y%m%d"))],
    "SW-037": [("v1", lambda: hmi_last(S / "solar-inc-srmag72-20261014/hmi_nrt")), ("v2", lambda: hmi_last(V2 / "hmi-sharp-nrt/days"))],
    "SW-038": [("v1", lambda: pq_max(S / "nmdb-anis72-cache/*.parquet")), ("v2", lambda: pq_max(V2 / "nmdb/**/*.parquet"))],
    "SW-039": [("SUVI Fe171 DB", lambda: duck("solar_g18_fe171.duckdb", "select max(slot) from suvi_hourly where status='ok'")),
               ("v2 cache", lambda: pq_max(V2 / "fe171/fe171-hourly-v2.parquet"))],
    "SW-040": [("v1", lambda: pq_max(S / "solar-inc-euvwave72-20261014/euvwave_origin_features.parquet")),
               ("v2", lambda: pq_max(V2 / "euvwave/euvwave_origin_features.parquet"))],
    "SW-042": [("v1", lambda: name_max(S / "swpc-discussion76/texts", r"(\d{12})forecast", "%Y%m%d%H%M")),
               ("v2", lambda: name_max(V2 / "swpc-discussion/texts", r"(\d{12})forecast", "%Y%m%d%H%M"))],
    "SW-057": [("v2", lambda: name_max(V2 / "gong-mrzqs", GONG_RX, GONG_FMT))],
    "SW-011": [("v1", lambda: manifest_max(H / "sunrun/leaderboard/experiments/solar_inc_gongpolar72_20261014/gong_manifest.json")),
               ("v2", lambda: pq_max(V2 / "gong-derived/map-features.parquet"))],
    "SW-016": [("v1", lambda: pq_max(S / "solar-inc-alfven72-20261014-cache/*.parquet")),
               ("v2", lambda: max(filter(None, [pq_max(V2 / "alfven/features.parquet"), pq_max(V2 / "aceturb/features.parquet")]), default=None))],
    "SW-018": [("v1", lambda: pq_max(S / "solar-inc-gongocb72-20261014-cache/origin-features-*.parquet")),
               ("v2", lambda: pq_max(V2 / "gong-ocb/cache/origin-features-*.parquet"))],
    "SW-029": [("v1", lambda: cactus_last(S / "solar-inc-cactuskin-20261014/raw")), ("v2", lambda: cactus_last(V2 / "cactus/raw"))],
    "SW-035": [("v1", lambda: xrs_last(S / "goes-xrs72-20261014/dn_xrsf-l2-avg1m_g18_y2026_v2-2-1.nc")),
               ("v2", lambda: xrs_last(V2 / "goes18-xrs/dn_xrsf-l2-avg1m_g18_y2026_v2-2-1.nc"))],
    "SW-041": [("v1", lambda: pq_max(S / "vjepa-frozen-fusion/run-*/inventory.parquet", require_values=False))],
}
# Daily/monthly granular sources: the newest file covers its whole day.
DAY_END = {"SW-013", "SW-034", "SW-036"}


def main():
    out = {}
    for sw, cands in SOURCES.items():
        seen = []
        for label, fn in cands:
            try:
                t = fn()
            except Exception as e:  # noqa: BLE001 - report and continue
                seen.append({"copy": label, "error": f"{type(e).__name__}: {e}"[:160]})
                continue
            if t:
                if sw in DAY_END:
                    t = t + dt.timedelta(hours=23, minutes=59)
                seen.append({"copy": label, "latest_utc": t.isoformat()})
        ok = [s for s in seen if "latest_utc" in s]
        best = max(ok, key=lambda s: s["latest_utc"]) if ok else None
        out[sw] = {"latest_utc": best["latest_utc"] if best else None, "copy": best["copy"] if best else None, "all": seen}
        print(sw, out[sw]["latest_utc"], out[sw]["copy"], [s.get("copy") + ":" + (s.get("latest_utc") or s.get("error", ""))[:19] for s in seen])
    target = Path(__file__).resolve().parent / "reports" / f"end-dates-{dt.datetime.now(UTC):%Y%m%d}.json"
    target.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
