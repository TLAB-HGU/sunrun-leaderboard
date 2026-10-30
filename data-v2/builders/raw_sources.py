"""data-v2 raw collectors: bring each raw source up to now under store/v2/<dataset>/.

Every collector only ADDS files (download to *.tmp, then os.replace) and never writes into an existing
file, so v1 bodies may be seeded into the v2 folder with hard links (`--seed`) without any risk to v1.
Mutable upstream files (current month/year) are re-fetched and replaced as a new inode.
Each dataset folder gets MANIFEST.json (as_of, source, builder hash, per-file sha256/bytes/Last-Modified,
fetched_at). fetched_at is NOT historical availability; causal rules live with the feature builders.

    python raw_sources.py <dataset> [--seed]
    datasets: cactus swpc_events discussion srs enlil_ambient enlil_cme sep xrs hmi ccmc lasco_c2 gong all
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
import re
import sys
import threading
import uuid
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

LB = Path("/home/t-lab01/sunrun/leaderboard")
STORE_ROOT = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store')

V2 = STORE_ROOT / "v2"
UA = "sunrun-data-v2/1.0 (research archive refresh)"
TODAY = dt.datetime.now(dt.timezone.utc).date()


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self.hrefs.append(v)


def get(url, tries=5, timeout=120, data=None, headers=None):
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, **(headers or {})})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read(), dict(r.getheaders())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            last = e
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def listing(url):
    body, _ = get(url)
    p = Links()
    p.feed(body.decode("latin-1"))
    return [urllib.parse.urljoin(url, h) for h in p.hrefs]


def rawdb_request(action, **kwargs):
    from spaceweather_collector.rawdb import request
    return request(action, **kwargs)


def fsync_path(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(path: Path, body: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}-{threading.get_ident()}')
    with tmp.open('wb') as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fsync_path(path.parent)


def preserve_previous(path, source, product, url):
    """Keep bytes and their DB location before replacing a mutable compatibility path."""
    if not path.exists():
        return
    from spaceweather_collector.rawdb import sha256_file
    digest = sha256_file(path)
    archive = path.parent / '.raw-versions' / digest / path.name
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        os.link(path, archive)
        fsync_path(archive.parent)
    rawdb_request('execute', sql='UPDATE raw.files SET path = ? WHERE path = ? AND sha256 = ?',
                  params=[str(archive.resolve()), str(path.resolve()), digest])
    rawdb_request('register_files', records=[{
        'path': str(archive.resolve()), 'sha256': digest, 'bytes': archive.stat().st_size,
        'source': source, 'product': product, 'url': url,
        'metadata_json': {'preserved_from': str(path.resolve()), 'historical_availability': 'unknown'},
    }])


def register_body(path, source, product, url, headers=None, fetched_at=None):
    from spaceweather_collector.rawdb import sha256_file
    headers = headers or {}
    record = {'path': str(path.resolve()), 'source': source, 'product': product, 'url': url,
              'sha256': sha256_file(path), 'bytes': path.stat().st_size,
              'last_modified': next((v for k, v in headers.items() if k.lower() == 'last-modified'), None),
              'fetched_at_utc': fetched_at}
    rawdb_request('register_files', records=[record])
    return record


def seed(src: Path, dst: Path, pattern="**/*"):
    """Hard-link v1 files into v2 (same filesystem); collectors never write into existing files."""
    n = 0
    for p in src.glob(pattern):
        if p.is_file():
            q = dst / p.relative_to(src)
            if not q.exists():
                q.parent.mkdir(parents=True, exist_ok=True)
                os.link(p, q)
                n += 1
    return n


class Run:
    def __init__(self, name, source):
        self.name, self.source = name, source
        self.run_id = f"data-v2-{name}-{uuid.uuid4().hex}"
        self.dir = V2 / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "fetch-log.jsonl"
        self.new = 0
        self.lock = threading.Lock()
        rawdb_request("status")

    def fetch_to(self, url, rel, replace=False):
        path = self.dir / rel
        if path.exists() and not replace:
            register_body(path, self.name, self.name, url)
            return False
        body, headers = get(url)
        with self.lock:
            if path.exists() and hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(body).digest():
                preserve_previous(path, self.name, self.name, url)
            save(path, body)
            record = register_body(path, self.name, self.name, url, headers,
                                   dt.datetime.now(dt.timezone.utc).isoformat())
            # Compatibility evidence for existing feature builders; DB remains authoritative.
            with self.log_path.open('a') as handle:
                handle.write(json.dumps({**record, 'file': rel}) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
            self.new += 1
        return True

    def manifest(self, extra=None, data_glob="**/*"):
        files = []
        for p in sorted(self.dir.glob(data_glob)):
            if '.raw-versions' in p.parts:
                continue
            if p.is_file() and p.name not in ("MANIFEST.json", "fetch-log.jsonl") and not p.name.endswith(".tmp"):
                files.append({"file": str(p.relative_to(self.dir)), "bytes": p.stat().st_size})
        doc = {"dataset": self.name, "version": "v2", "as_of_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "source": self.source, "builder": __file__,
               "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "new_files_this_run": self.new, "n_files": len(files), "bytes": sum(f["bytes"] for f in files),
               "per_file_sha256_and_last_modified": "fetch-log.jsonl (seeded v1 files: see v1 manifests)",
               "fetched_at_is_not_availability": True, **(extra or {})}
        save(self.dir / "MANIFEST.json", (json.dumps(doc, indent=1) + "\n").encode())
        rawdb_request('execute', sql="""CREATE TABLE IF NOT EXISTS raw.collector_manifests (
            dataset VARCHAR, manifest_sha256 VARCHAR, recorded_at TIMESTAMPTZ,
            payload_json JSON, PRIMARY KEY(dataset, manifest_sha256))""")
        encoded = json.dumps(doc, sort_keys=True)
        rawdb_request('execute', sql='INSERT INTO raw.collector_manifests VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING',
                      params=[self.name, hashlib.sha256(encoded.encode()).hexdigest(), doc['as_of_utc'], encoded])
        rawdb_request('execute', sql='INSERT INTO raw.runs (run_id, source, status, metadata_json) VALUES (?, ?, ?, ?) ON CONFLICT (run_id) DO UPDATE SET metadata_json = excluded.metadata_json',
                      params=[self.run_id, self.name, 'complete', encoded])
        print(f"{self.name}: +{self.new} files, {len(files)} total", flush=True)


def days(start, end=TODAY):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


# ---------------------------------------------------------------- collectors
def cactus(do_seed):
    r = Run("cactus", "SIDC CACTus LASCO 2.5.0 quicklook catalog")
    if do_seed:
        seed(STORE_ROOT / "solar-inc-cactuskin-20261014/raw", r.dir / "raw", "catalog-*.html")
    month = dt.date(2022, 10, 1)
    while month <= TODAY:
        url = f"https://www.sidc.be/cactus/catalog/LASCO/2_5_0/qkl/{month:%Y}/{month:%m}/latestCMEs.html"
        current = (month.year, month.month) in {(TODAY.year, TODAY.month),
                                                ((TODAY.replace(day=1) - dt.timedelta(days=1)).year,
                                                 (TODAY.replace(day=1) - dt.timedelta(days=1)).month)}
        try:
            r.fetch_to(url, f"raw/catalog-{month:%Y-%m}.html", replace=current)
        except urllib.error.HTTPError:
            pass
        month = (month.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    r.fetch_to("https://www.sidc.be/cactus/out/latestCMEs.html", f"raw/latest/latestCMEs-{TODAY:%Y%m%d}.html", replace=True)
    r.manifest({"note": "current and previous month re-fetched every run; live page snapshot kept per day"})


def swpc_events(do_seed):
    r = Run("swpc-events", "NOAA SWPC Solar and Geophysical Event Reports (NGDC daily archive + live text)")
    if do_seed:
        seed(STORE_ROOT / "solar-inc-typeiiie72-20261014/swpc-events/raw", r.dir / "raw", "**/*.txt")
    base = "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_event_reports"
    for d in days(dt.date(2022, 9, 1)):
        try:
            r.fetch_to(f"{base}/{d:%Y}/{d:%m}/{d:%Y%m%d}events.txt", f"raw/{d:%Y}/{d:%m}/{d:%Y%m%d}events.txt")
        except urllib.error.HTTPError:
            pass  # not yet published (archive lags about 3 days)
    now = dt.datetime.now(dt.timezone.utc)
    r.fetch_to("https://services.swpc.noaa.gov/text/solar-geophysical-event-reports.txt",
               f"live/{now:%Y%m%dT%H%M}-solar-geophysical-event-reports.txt", replace=True)
    r.manifest({"availability_rule_v1": "max(Last-Modified, :Created:) (Type IV: +72h)"})


# 2026-10-28 user: the SWPC forecast discussion collector (SW-042) was removed.
def srs(do_seed):
    # 2026-10-20 user: SWPC Solar Region Summary history (sunrun gjc asked for regions beyond the 30-day JSON).
    r = Run("swpc-srs", "NOAA SWPC Solar Region Summary archive (NGDC daily_reports)")
    base = "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_region_summaries/"
    for y in [u for u in listing(base) if re.search(r"/\d{4}/$", u)]:
        if int(y.rstrip("/").rsplit("/", 1)[1]) < 2022:
            continue
        for mdir in [u for u in listing(y) if re.search(r"/\d{2}/$", u)]:
            if mdir.rstrip("/").rsplit("/", 2)[-2:] < ["2022", "07"]:
                continue
            for u in listing(mdir):
                m = re.search(r"(\d{8})SRS\.txt$", u)
                if m:
                    r.fetch_to(u, f"texts/{m.group(1)[:4]}/{m.group(1)[4:6]}/{m.group(1)}SRS.txt")
    r.manifest({"availability_rule_v1": ":Issued: time (about 00:30 UTC, regions valid at previous day 24:00Z) <= origin"})


ISWA = "https://iswa.gsfc.nasa.gov/iswa_data_tree/model/heliosphere/"


def _iswa(name, source, sub, name_re, start=(2022, 7)):
    """iSWA archive tree <sub>/YYYY/MM/<file>; single thread, 1 s between requests (CCMC is easily overloaded).
    Any fetch error aborts the run (no per-file retry storm on 403); rerun resumes by file."""
    r = Run(name, source)
    for y in [u for u in listing(ISWA + sub) if re.search(r"/\d{4}/$", u)]:
        if int(y.rstrip("/").rsplit("/", 1)[1]) < start[0]:
            continue
        for mdir in [u for u in listing(y) if re.search(r"/\d{2}/$", u)]:
            ym = tuple(int(x) for x in mdir.rstrip("/").rsplit("/", 2)[-2:])
            if ym < start:
                continue
            for u in listing(mdir):
                m = re.search(name_re, u)
                if m and r.fetch_to(u, f"{ym[0]:04d}/{ym[1]:02d}/{m.group(0)}"):
                    time.sleep(1.0)
            time.sleep(1.0)
    return r


def enlil_ambient(do_seed):
    # 2026-10-20 user: archived WSA-ENLIL ambient runs (6-hourly) for the 25-72h v_r axis requested by sunrun gjc.
    r = _iswa("iswa-enlil-ambient", "NASA CCMC iSWA WSA-ENLIL ambient IMF_1hour archive", "enlil/IMF_1hour/",
              r"\d{8}_\d{6}_IMF\.dat$")
    r.manifest({"availability_rule_v1": "run time in file name + measured lag (see SolarWind DB row); Last-Modified "
                                        "of old files can be a later re-upload"})


def enlil_cme(do_seed):
    # 2026-10-20 user: archived WSA-ENLIL cone (CME) runs, Earth velocity/density timeline data.
    r = _iswa("iswa-enlil-cme-timeline", "NASA CCMC iSWA WSA-ENLIL cone velocity-density-timeline-DATA archive",
              "wsa-enlil-cone/velocity-density-timeline-DATA/", r"\d{8}_\d{6}_[\d.]+_ENLIL_time_line\.dat$")
    r.manifest({"availability_rule_v1": "run time in file name + measured lag (see SolarWind DB row); Last-Modified "
                                        "of old files can be a later re-upload (2025-07 files show 2025-09-08)"})


def sep(do_seed):
    r = Run("goes18-sep", "NOAA NCEI GOES-18 MPS-HI L2 avg5m science")
    product = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes18/l2/data/mpsh-l2-avg5m_science"
    if do_seed:
        seed(STORE_ROOT / "goes-sep72-20261014/raw", r.dir / "raw", "**/*.nc")
    name_re = re.compile(r"sci_mpsh-l2-avg5m_g18_d(\d{8})_v([\w.-]+)\.nc$")
    month = dt.date(2022, 9, 1)
    while month <= TODAY:
        for u in listing(f"{product}/{month:%Y/%m}/"):
            m = name_re.search(urllib.parse.unquote(u))
            if m:
                fname = urllib.parse.unquote(u.rsplit("/", 1)[1])
                r.fetch_to(u, f"raw/{month:%Y/%m}/{fname}")
        month = (month.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    r.manifest({"note": "every published version kept; v1 bound each day to its newest version"})


def xrs(do_seed):
    r = Run("goes18-xrs", "NOAA NCEI GOES-18 XRS L2 avg1m v2-2-1 (yearly files, current year grows daily)")
    base = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes18/l2/data/xrsf-l2-avg1m"
    for year in range(2022, TODAY.year + 1):
        name = f"dn_xrsf-l2-avg1m_g18_y{year}_v2-2-1.nc"
        v1 = STORE_ROOT / "goes-xrs72-20261014" / name
        if do_seed and v1.exists() and year < TODAY.year and not (r.dir / name).exists():
            os.link(v1, r.dir / name)
        r.fetch_to(f"{base}/{name}", name, replace=(year == TODAY.year))
    r.manifest()


def hmi(do_seed):
    r = Run("hmi-sharp-nrt", "JSOC hmi.sharp_cea_720s_nrt keywords via jsoc_info rs_list")
    fields = ["T_REC", "HARPNUM", "NOAA_ARS", "CRVAL1", "CRVAL2", "R_VALUE", "TOTPOT", "TOTUSJH", "USFLUX", "MEANSHR", "AREA_ACR"]
    if do_seed:
        for p in (STORE_ROOT / "solar-inc-srmag72-20261014/hmi_nrt").glob("*.json"):
            if p.stem < (TODAY - dt.timedelta(days=2)).isoformat() and not (r.dir / "days" / p.name).exists():
                (r.dir / "days").mkdir(parents=True, exist_ok=True)
                os.link(p, r.dir / "days" / p.name)

    def one(d):
        ds = f"hmi.sharp_cea_720s_nrt[][{d:%Y.%m.%d}_00:00:00_TAI/1d]"
        url = "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info?" + urllib.parse.urlencode(
            {"op": "rs_list", "ds": ds, "key": ",".join(fields)})
        return r.fetch_to(url, f"days/{d.isoformat()}.json", replace=d >= TODAY - dt.timedelta(days=2))

    with cf.ThreadPoolExecutor(4) as pool:
        list(pool.map(one, list(days(dt.date(2024, 12, 20)))))
    r.manifest({"availability_rule_v1": "T_REC + 4h <= origin"})


def ccmc(do_seed):
    r = Run("ccmc-cme-scoreboard", "NASA CCMC Earth CME Scoreboard public API")
    start = dt.date(2022, 10, 1)
    while start <= TODAY:
        end = min(dt.date(start.year + 1, start.month, 1) - dt.timedelta(days=1), TODAY)
        url = ("https://ccmc.gsfc.nasa.gov/CMESB-Earth/WS/get/predictions?" + urllib.parse.urlencode(
            {"CMEtimeStart": start.isoformat(), "CMEtimeEnd": end.isoformat(),
             "skipNoArrivalObservedCMEs": "false", "closeOutCMEsOnly": "false"}))
        r.fetch_to(url, f"api/predictions_{start.isoformat()}_{end.isoformat()}.json", replace=end >= TODAY - dt.timedelta(days=60))
        start = end + dt.timedelta(days=1)
    r.manifest({"forbidden_fields_not_read": ["arrivalTime", "noArrivalObserved", "differenceInHrs", "leadTimeInHrs",
                                              "maxKP", "dstMin"], "availability_rule_v1": "submissionTime <= origin"})


def lasco_c2(do_seed):
    r = Run("lasco-c2-512", "SOHO LASCO C2 512px quicklook, REPROCESSING/Completed daily directories")
    if do_seed:
        seed(STORE_ROOT / "solar-inc-lascovision-20261014/frames", r.dir / "frames", "**/*.jpg")
    name_re = re.compile(r"(\d{8})_(\d{4})_c2_512\.jpg$")

    def day(d):
        url = f"https://soho.nascom.nasa.gov/data/REPROCESSING/Completed/{d:%Y}/c2/{d:%Y%m%d}/"
        try:
            urls = listing(url)
        except urllib.error.HTTPError:
            return
        for u in urls:
            m = name_re.search(u)
            if m:
                r.fetch_to(u, f"frames/{d:%Y}/{d:%m}/{m.group(0)}")

    with cf.ThreadPoolExecutor(3) as pool:
        list(pool.map(day, list(days(TODAY - dt.timedelta(days=30)))))
    r.manifest({"note": "v1 frames 2022-09-30..2026-09-26 hard-linked; last 30 days re-listed each run",
                "availability_rule_v1": "frame Last-Modified about 1h after observation"})


def gong(do_seed):
    r = Run("gong-mrzqs", "NSO GONG QuickReduce zero-point-corrected synoptic maps (mrzqs), six-hour grid")
    base = "https://gong.nso.edu/data/magmap/QR/zqs"
    name_re = re.compile(r"mrzqs(\d{6})t(\d{4})c(\d+)_(\d+)\.fits\.gz$")
    if do_seed:
        for src in (STORE_ROOT / "scratch/pfssconn_20261014/gong_fits", STORE_ROOT / "scratch/cmehcs72_20261014_gong_fits"):
            for p in src.glob("*.fits.gz"):
                m = name_re.search(p.name)
                if m:
                    q = r.dir / "maps" / f"20{m.group(1)[:2]}" / m.group(1)[2:4] / p.name
                    if not q.exists():
                        q.parent.mkdir(parents=True, exist_ok=True)
                        os.link(p, q)

    def obs(name):
        m = name_re.search(name)
        return dt.datetime.strptime(m.group(1) + m.group(2), "%y%m%d%H%M")

    def day(d):
        url = f"{base}/{d:%Y%m}/mrzqs{d:%y%m%d}/"
        try:
            names = [u.rsplit("/", 1)[1] for u in listing(url) if name_re.search(u)]
        except urllib.error.HTTPError:
            return
        if not names:
            return
        chosen = set()
        for h in (0, 6, 12, 18):
            target = dt.datetime(d.year, d.month, d.day, h, 14)
            chosen.add(min(names, key=lambda n: abs((obs(n) - target).total_seconds())))
        for n in sorted(chosen):
            r.fetch_to(url + n, f"maps/{d:%Y}/{d:%m}/{n}")

    with cf.ThreadPoolExecutor(4) as pool:
        list(pool.map(day, list(days(dt.date(2022, 10, 1)))))
    r.manifest({"selection": "per day the maps nearest 00:14/06:14/12:14/18:14 UTC (same rule as q78/t2gr, pfssconn)",
                "availability_rule_v1": "max(observation + 2h, HTTP Last-Modified)",
                "note": "the 2022-10..2026-05 bulk of v1 (worker-t2gr) was deleted; re-downloaded here"})


COLLECTORS = {f.__name__: f for f in (cactus, swpc_events, srs, enlil_ambient, enlil_cme, sep, xrs, hmi, ccmc, lasco_c2, gong)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=[*COLLECTORS, "all"])
    ap.add_argument("--seed", action="store_true", help="hard-link v1 bodies first")
    a = ap.parse_args()
    names = list(COLLECTORS) if a.dataset == "all" else [a.dataset]
    failed = []
    for n in names:
        try:
            COLLECTORS[n](a.seed)
        except Exception as e:  # noqa: BLE001 - keep going, report at the end
            failed.append(n)
            print(f"{n}: FAILED {e!r}", flush=True)
    sys.exit(1 if failed else 0)
