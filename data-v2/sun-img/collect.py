"""GOES SUVI L1b access layer: S3 listing, FITS header-only reads, and .nc downloads.

Layout (both noaa-goes18 / noaa-goes19):
  SUVI-L1b-<channel>/<YYYY>/<DOY 3 digits>/<HH>/OR_SUVI-L1b-<channel>_G<nn>_s<start>_e<end>_c<created>.{nc,fits}
The FITS header sits at the start of the file, so a small HTTP Range request is enough.
"""

import http.client
import re
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

UTC = timezone.utc
HOUR = timedelta(hours=1)
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
FITS_BLOCK = 2880
HEADER_KEYS = ("DATE-OBS", "DATE-END", "EXPTIME", "CMD_EXP", "CRPIX1", "CRPIX2", "CROTA",
               "SOLAR_B0", "RSUN", "YAW_FLIP", "ECLIPSE", "FILTER1", "FILTER2")
NAME_RE = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)_e(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)_c")


def base(sat):
    return f"https://noaa-goes{sat}.s3.amazonaws.com"


_local = threading.local()


def _conn(host):
    """One keep-alive HTTPS connection per (thread, host): avoids a TLS handshake per request."""
    conns = _local.__dict__.setdefault("conns", {})
    if host not in conns:
        conns[host] = http.client.HTTPSConnection(host, timeout=30)
    return conns[host]


def get(url, byte_range=None, tries=5):
    """GET with retry/backoff. 404 is returned as None (permanent), other errors are retried."""
    u = urllib.parse.urlsplit(url)
    headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {}
    for i in range(tries):
        try:
            c = _conn(u.netloc)
            c.request("GET", u.path + (f"?{u.query}" if u.query else ""), headers=headers)
            r = c.getresponse()
            body = r.read()
            if r.status == 404:
                return None
            if r.status in (200, 206):
                return body
            err = RuntimeError(f"HTTP {r.status} {url}")
        except Exception as e:   # noqa: BLE001 - timeouts / resets: reconnect and retry
            _local.conns.pop(u.netloc, None)
            err = e
        if i == tries - 1:
            raise err
        time.sleep(1.5 ** i)


def _list(sat, prefix, delimiter=None):
    """ListObjectsV2 with pagination -> (keys[(key, LastModified, size)], common_prefixes)."""
    keys, prefixes, token = [], [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["continuation-token"] = token
        root = ET.fromstring(get(f"{base(sat)}/?{urllib.parse.urlencode(q)}"))
        for c in root.findall(f"{NS}Contents"):
            keys.append((c.findtext(f"{NS}Key"),
                         datetime.strptime(c.findtext(f"{NS}LastModified"), "%Y-%m-%dT%H:%M:%S.%fZ"),
                         int(c.findtext(f"{NS}Size"))))
        prefixes += [p.findtext(f"{NS}Prefix") for p in root.findall(f"{NS}CommonPrefixes")]
        if root.findtext(f"{NS}IsTruncated") != "true":
            return keys, prefixes
        token = root.findtext(f"{NS}NextContinuationToken")


def hour_prefix(channel, hour):
    return f"SUVI-L1b-{channel}/{hour:%Y}/{hour.timetuple().tm_yday:03d}/{hour:%H}/"


def _stamp(g):
    y, doy, h, m, s, t = map(int, g)
    return (datetime(y, 1, 1) + timedelta(days=doy - 1, hours=h, minutes=m, seconds=s, milliseconds=t * 100))


def list_hour(sat, channel, hour):
    """FITS frames in the folder of `hour` (naive UTC), sorted oldest -> newest.

    dur = e - s from the file name (0.1 s resolution) is a cheap hint of the exposure.
    """
    keys, _ = _list(sat, hour_prefix(channel, hour))
    out = []
    for key, mod, size in keys:
        if not key.endswith(".fits"):
            continue
        m = NAME_RE.search(key)
        if not m:
            continue
        s, e = _stamp(m.groups()[:6]), _stamp(m.groups()[6:])
        out.append(dict(sat=sat, channel=channel, key=key, filename=key.rsplit("/", 1)[1],
                        obs_start=s, obs_end=e, dur=(e - s).total_seconds(), s3_modified=mod, size=size))
    return sorted(out, key=lambda f: f["obs_end"])


def _parse_header(raw):
    cards = {}
    for i in range(0, len(raw) - 79, 80):
        card = raw[i:i + 80].decode("ascii", "replace")
        if card.startswith("END") and card[3:].strip() == "":
            return cards, True
        if len(card) > 9 and card[8] == "=":
            val = card[10:].split(" /")[0].strip().strip("'").strip()
            cards[card[:8].strip()] = val
    return cards, False


def read_header(sat, key):
    """Header cards of one FITS file via Range request (grows until END is found)."""
    for nblocks in (3, 12, 60):
        raw = get(f"{base(sat)}/{key}", (0, nblocks * FITS_BLOCK - 1))
        cards, done = _parse_header(raw)
        if done:
            break
    else:
        raise ValueError(f"no END card within {nblocks} blocks: {key}")

    def num(k):
        try:
            return float(cards[k])
        except (KeyError, ValueError):
            return None
    return dict(
        obs_time=datetime.fromisoformat(cards["DATE-OBS"]) if "DATE-OBS" in cards else None,
        exptime=num("EXPTIME"), cmd_exp=num("CMD_EXP"), crpix1=num("CRPIX1"), crpix2=num("CRPIX2"),
        crota=num("CROTA"), rsun=num("RSUN"), solar_b0=num("SOLAR_B0"),
        yaw_flip=None if num("YAW_FLIP") is None else int(num("YAW_FLIP")),
        eclipse=None if num("ECLIPSE") is None else int(num("ECLIPSE")),
        filter1=cards.get("FILTER1"), filter2=cards.get("FILTER2"))


def first_hour(sat, channel):
    """Oldest hour folder in the archive (naive UTC)."""
    root = f"SUVI-L1b-{channel}/"
    for _ in range(3):   # year -> doy -> hour
        _, subs = _list(sat, root, "/")
        root = sorted(subs)[0]
    y, doy, hh = root.rstrip("/").split("/")[-3:]
    return datetime(int(y), 1, 1) + timedelta(days=int(doy) - 1, hours=int(hh))


def download(sat, key, dest):
    """Download the .nc twin of a .fits key to `dest` (atomic: .part -> rename). Returns size in bytes.

    An existing dest is kept as is (resume). Returns None if the object does not exist.
    """
    dest = Path(dest)
    if dest.exists():
        return dest.stat().st_size
    body = get(f"{base(sat)}/{key.rsplit('.', 1)[0]}.nc")
    if body is None:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + f".part{threading.get_ident()}")
    part.write_bytes(body)
    part.replace(dest)
    return len(body)
