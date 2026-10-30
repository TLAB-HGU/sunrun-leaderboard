"""Origin-causal CME × coronal-hole magnetic influence features.

SUVI CH regions use the existing Fe195 dark-region recipe (0.45 of the valid
disk median, 8-connected opening/closing, 0.0005 minimum solar-area fraction).
The latest source is selected only after both observation and S3 release plus
the conservative two-hour latency. Legacy local NetCDF products are derived
from S3 FITS objects, so byte identity is not claimed; the receipt records that
proxy and verifies the staged local size. GONG maps use issue+2h / LastModified.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
import gzip
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import glob

import h5py
import numpy as np
import pandas as pd
from scipy import ndimage

LB = Path('/home/t-lab01/sunrun/leaderboard')
STORE = LB / "store"
DONKI_DIR = STORE / "v2/donki"
SUVI_META_GLOB = "/home/t-lab01/.local/state/sundb/stage/suvi_hourly/year=*/part.parquet"
SUVI_ROOT = LB.parent / "sun-img"
HERE = Path(__file__).resolve().parent
CACHE_ROOT = STORE / "v2/cmechip/cache"
HOUR_NS = 3_600_000_000_000
DAY_NS = 24 * HOUR_NS
SOLAR_RADIUS_KM = 695_700.0
SYNODIC_RATE_DEG_DAY = 360.0 / 27.2753
SUVI_LATENCY_HOURS = 2.0
GONG_LATENCY_HOURS = 2.0
MIN_CH_AREA_FRACTION = 0.0005
CH_DARK_RATIO = 0.45
CHIP_CACHE_VERSION = "cme-chip72-v1"
# This identifies the unchanged pixel/field extraction algorithms. Availability
# fallback logic can evolve without invalidating deterministic map caches.
MAP_EXTRACTOR_SHA256 = "a33b659159ab264848079180d82d1cc481977e6ebc8a4e0df4295faf89be16d8"

FEATURE_COLUMNS = (
    "chip_max", "chip_sum", "chip_dx", "chip_dy",
    "nearest_ch_dist_deg", "nearest_ch_area", "nearest_ch_absbr",
    "cme_speed", "cme_halfwidth", "predicted_deflect_lon_sign",
    "predicted_deflect_lat_sign", "chip_missing",
)


class UnusableSuviFrameError(ValueError):
    """The staged SUVI product cannot supply a valid solar-disk mask."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _ns(value):
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
        return int(ts.value)
    except (TypeError, ValueError, OverflowError):
        return None


def _utc(value):
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _wrapped_delta_deg(a, b):
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


def _card_value(card):
    raw = card[10:].split("/", 1)[0].strip()
    if raw.startswith("'"):
        return raw[1:].split("'", 1)[0].strip()
    if raw in ("T", "F"):
        return raw == "T"
    try:
        value = float(raw.replace("D", "E"))
    except ValueError:
        return raw
    return int(value) if value.is_integer() else value


def read_gong_fits(path: Path):
    """Read a GONG primary 2-D gzipped FITS image without Astropy."""
    header_bytes = bytearray()
    with gzip.open(path, "rb") as stream:
        while True:
            block = stream.read(2880)
            if len(block) != 2880:
                raise ValueError(f"truncated GONG FITS header: {path}")
            header_bytes.extend(block)
            cards = [block[i:i + 80] for i in range(0, 2880, 80)]
            if any(card[:8].strip() == b"END" for card in cards):
                break
        header = {}
        for offset in range(0, len(header_bytes), 80):
            card = bytes(header_bytes[offset:offset + 80]).decode("ascii", "replace")
            if card[:8].strip() == "END":
                break
            if card[8:10] == "= ":
                header[card[:8].strip()] = _card_value(card)
        if int(header.get("NAXIS", 0)) != 2:
            raise ValueError(f"GONG primary image must be 2-D: {path}")
        nx, ny = int(header["NAXIS1"]), int(header["NAXIS2"])
        bitpix = int(header["BITPIX"])
        dtype = {-64: ">f8", -32: ">f4", 16: ">i2", 32: ">i4", 64: ">i8"}.get(bitpix)
        if dtype is None:
            raise ValueError(f"unsupported GONG BITPIX={bitpix}: {path}")
        payload = stream.read(nx * ny * (abs(bitpix) // 8))
        if len(payload) != nx * ny * (abs(bitpix) // 8):
            raise ValueError(f"truncated GONG FITS image: {path}")
    image = np.frombuffer(payload, dtype=np.dtype(dtype)).reshape(ny, nx).astype(np.float32)
    image *= np.float32(header.get("BSCALE", 1.0))
    image += np.float32(header.get("BZERO", 0.0))
    return header, image


def _read_suvi(path: Path):
    with h5py.File(path, "r") as nc:
        rad_var = nc["RAD"]
        raw = np.asarray(rad_var[...])
        rad = raw.astype(np.float32)
        invalid = ~np.isfinite(rad)
        for name in ("_FillValue", "missing_value"):
            if name in rad_var.attrs:
                invalid |= raw == np.asarray(rad_var.attrs[name]).reshape(-1)[0]
        rad *= np.float32(np.asarray(rad_var.attrs.get("scale_factor", 1)).item())
        rad += np.float32(np.asarray(rad_var.attrs.get("add_offset", 0)).item())
        rad[invalid] = np.nan
        header = dict(nc.attrs)
        for key in ("CRPIX1", "CRPIX2", "RSUN", "CROTA", "SOLAR_B0", "PC1_1", "PC1_2", "PC2_1", "PC2_2"):
            if key in nc:
                header[key] = np.asarray(nc[key][...]).item()
        return rad.squeeze(), np.asarray(nc["DQF"][...]).squeeze(), header


def _solar_geometry(rad_shape, header):
    h = {str(k).upper(): v for k, v in header.items()}

    def num(key, default=None):
        value = h.get(key, default)
        if value is None:
            raise ValueError(f"missing SUVI geometry field {key}")
        return float(np.asarray(value).item())

    rsun = num("RSUN")
    if not math.isfinite(rsun) or rsun <= 0:
        raise ValueError("invalid SUVI RSUN")
    if all(k in h for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2")):
        matrix = np.array([[num("PC1_1"), num("PC1_2")], [num("PC2_1"), num("PC2_2")]], dtype=float)
        scale = np.linalg.svd(matrix, compute_uv=False).min()
        if scale <= 0:
            raise ValueError("singular SUVI PC matrix")
    else:
        scale = 1.0
        theta = np.deg2rad(num("CROTA", 0.0))
        matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    cx, cy = num("CRPIX1") - 1.0, num("CRPIX2") - 1.0
    radius = rsun / scale + 2
    height, width = rad_shape
    x0, x1 = max(0, int(np.floor(cx - radius))), min(width, int(np.ceil(cx + radius)) + 1)
    y0, y1 = max(0, int(np.floor(cy - radius))), min(height, int(np.ceil(cy + radius)) + 1)
    return x0, x1, y0, y1, cx, cy, rsun, matrix, float(np.asarray(h.get("SOLAR_B0", 0.0)).item())


def _component_mask_from_suvi(rad, dqf, header):
    """Make component IDs and areas on a 1-degree Stonyhurst grid."""
    rad, dqf = np.asarray(rad), np.asarray(dqf)
    if rad.ndim != 2 or rad.shape != dqf.shape:
        raise ValueError("SUVI RAD/DQF must be same-size 2-D arrays")
    x0, x1, y0, y1, cx, cy, rsun, matrix, b0_deg = _solar_geometry(rad.shape, header)
    rad = rad[y0:y1, x0:x1].astype(np.float32, copy=False)
    dqf = dqf[y0:y1, x0:x1]
    yy, xx = np.indices(rad.shape, dtype=np.float32)
    xx += x0 - cx
    yy += y0 - cy
    x = (matrix[0, 0] * xx + matrix[0, 1] * yy) / rsun
    y = (matrix[1, 0] * xx + matrix[1, 1] * yy) / rsun
    r2 = x * x + y * y
    disk = r2 <= 0.98 ** 2
    z = np.sqrt(np.maximum(1e-8, 1.0 - r2))
    b0 = np.deg2rad(b0_deg)
    lat = np.rad2deg(np.arcsin(np.clip(y * np.cos(b0) + z * np.sin(b0), -1.0, 1.0)))
    lon = np.rad2deg(np.arctan2(x, z * np.cos(b0) - y * np.sin(b0)))
    valid = disk & (dqf == 0) & np.isfinite(rad) & (rad >= 0)
    if not valid.any():
        raise UnusableSuviFrameError("SUVI frame has no valid solar-disk pixels")
    reference = float(np.median(rad[valid]))
    dark = valid & (rad < CH_DARK_RATIO * reference)
    structure = np.ones((3, 3), dtype=bool)
    dark = ndimage.binary_closing(ndimage.binary_opening(dark, structure), structure) & valid
    z = np.maximum(z, 1e-5)
    pixel_area = np.where(disk, 1.0 / (z * 2.0 * np.pi * rsun * rsun), 0.0).astype(np.float32)
    native_labels, count = ndimage.label(dark, structure=structure)
    totals = np.bincount(native_labels.ravel(), weights=pixel_area.ravel(), minlength=count + 1)
    keep = totals >= MIN_CH_AREA_FRACTION
    keep[0] = False
    old_ids = np.flatnonzero(keep)
    label_map = np.zeros(count + 1, dtype=np.uint16 if len(old_ids) < 65535 else np.uint32)
    label_map[old_ids] = np.arange(1, len(old_ids) + 1, dtype=label_map.dtype)
    native = label_map[native_labels]
    grid_labels = np.zeros((180, 360), dtype=np.uint16 if len(old_ids) < 65535 else np.uint32)
    grid_area = np.zeros((180, 360), dtype=np.float32)
    latitudes, longitudes, areas = [], [], []
    for new_id, old_id in enumerate(old_ids, 1):
        mask = native == new_id
        if not mask.any():
            continue
        weights = pixel_area[mask].astype(np.float64)
        la = lat[mask].astype(np.float64)
        lo = lon[mask].astype(np.float64)
        latitudes.append(float(np.average(la, weights=weights)))
        angle = np.deg2rad(lo)
        longitudes.append(float(np.rad2deg(np.arctan2(np.sum(np.sin(angle) * weights), np.sum(np.cos(angle) * weights)))))
        areas.append(float(weights.sum()))
    iy, ix = np.nonzero(native)
    if len(iy):
        ids = native[iy, ix].astype(np.int64)
        flat = (np.clip(np.floor(lat[iy, ix] + 90).astype(np.int64), 0, 179) * 360
                + (np.floor(lon[iy, ix] + 180).astype(np.int64) % 360))
        pairs = np.column_stack((ids, flat))
        unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
        summed_area = np.bincount(inverse, weights=pixel_area[iy, ix], minlength=len(unique))
        pair_component, pair_flat = unique[:, 0], unique[:, 1]
        order = np.lexsort((-summed_area, pair_flat))
        first = np.r_[True, pair_flat[order][1:] != pair_flat[order][:-1]]
        chosen = order[first]
        grid_labels.reshape(-1)[pair_flat[chosen]] = pair_component[chosen].astype(grid_labels.dtype)
        grid_area.reshape(-1)[pair_flat[chosen]] = summed_area[chosen].astype(np.float32)
    return {
        "labels": grid_labels,
        "grid_area": grid_area,
        "component_area_fraction": np.asarray(areas, dtype=np.float32),
        "component_lat": np.asarray(latitudes, dtype=np.float32),
        "component_lon_stonyhurst": np.asarray(longitudes, dtype=np.float32),
        "threshold": CH_DARK_RATIO,
        "min_area_fraction": MIN_CH_AREA_FRACTION,
    }


class _AvailabilityIndex:
    """Sweep source rows in availability order and choose the newest prior map."""
    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda r: (int(r["available_ns"]), int(r["slot_ns"]), str(r.get("file", ""))))

    def select_many(self, origins, extra_rows=()):
        rows = sorted([*self.rows, *list(extra_rows)], key=lambda r: (int(r["available_ns"]), int(r["slot_ns"]), str(r.get("file", ""))))
        result = [None] * len(origins)
        idxs = sorted(range(len(origins)), key=lambda i: _ns(origins[i]))
        heap, j = [], 0
        for oi in idxs:
            origin_ns = _ns(origins[oi])
            while j < len(rows) and int(rows[j]["available_ns"]) <= origin_ns:
                row = rows[j]
                if int(row["slot_ns"]) <= origin_ns:
                    heapq.heappush(heap, (-int(row["slot_ns"]), j))
                j += 1
            while heap and rows[heap[0][1]]["available_ns"] > origin_ns:
                heapq.heappop(heap)
            result[oi] = rows[heap[0][1]] if heap else None
        return result


class SuviCHCatalog:
    """SUVI Fe195 image source rows and hash-keyed 1-degree CH masks."""
    def __init__(self, cap, cache_root=CACHE_ROOT):
        self.cap_ns = _ns(cap)
        self.cache_root = Path(cache_root) / "suvi"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        paths = sorted(glob.glob(SUVI_META_GLOB))
        if not paths:
            raise FileNotFoundError(f"no staged SUVI metadata under {SUVI_META_GLOB}")
        frames = [pd.read_parquet(p, columns=["slot", "sat", "channel", "status", "s3_key", "obs_end", "s3_modified", "nc_path", "nc_size"]) for p in paths]
        meta = pd.concat(frames, ignore_index=True)
        meta = meta[(meta.sat == "G18") & (meta.channel == "Fe195") & (meta.status == "ok") & meta.nc_path.notna()].copy()
        meta["slot_utc"] = pd.to_datetime(meta["slot"], utc=True)
        meta = meta[meta["slot_utc"].astype("int64") <= self.cap_ns]
        rows, integrity_missing = [], 0
        for item in meta.to_dict("records"):
            slot_ns, obs_ns, mod_ns = _ns(item["slot"]), _ns(item["obs_end"]), _ns(item["s3_modified"])
            if slot_ns is None or obs_ns is None or mod_ns is None:
                continue
            path = Path(str(item["nc_path"]))
            if not path.is_absolute():
                path = SUVI_ROOT / path
            if not path.is_file() or path.stat().st_size != int(item["nc_size"]):
                integrity_missing += 1
                continue
            available_ns = max(obs_ns, mod_ns) + int(SUVI_LATENCY_HOURS * HOUR_NS)
            if slot_ns > available_ns:
                raise ValueError(f"SUVI source slot follows conservative availability time: {item['s3_key']}")
            rows.append({
                "kind": "suvi", "slot_ns": slot_ns, "obs_ns": obs_ns, "modified_ns": mod_ns,
                "available_ns": available_ns, "file": str(item["s3_key"]), "path": str(path),
                "size": int(item["nc_size"]), "stage_slot": str(item["slot"]),
            })
        self.index = _AvailabilityIndex(rows)
        self.rows = self.index.rows
        self.integrity_missing = integrity_missing
        self._mask_cache = {}
        self.invalid_rows = {}
        self.cache_hits = 0
        self.cache_created = 0
        self.extractor_sha256 = MAP_EXTRACTOR_SHA256
        self.metadata_files = paths

    def select_many(self, origins, extra_rows=()):
        origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
        selected = self.index.select_many(origins, extra_rows=extra_rows)
        result = []
        for origin, row in zip(origins, selected):
            fallback = False
            while row is not None:
                if row["file"] in self.invalid_rows:
                    fallback = True
                else:
                    try:
                        self.mask_for(row)
                        break
                    except UnusableSuviFrameError as exc:
                        self.invalid_rows[row["file"]] = str(exc)
                        fallback = True
                available_rows = [r for r in self.rows if r["file"] not in self.invalid_rows]
                available_extra = [r for r in extra_rows if r["file"] not in self.invalid_rows]
                row = _AvailabilityIndex(available_rows).select_many([origin], extra_rows=available_extra)[0]
            if row is not None:
                row = dict(row)
                row["quality_fallback"] = fallback
                row["fallback_age_h"] = max(0.0, (int(origin.value) - int(row["slot_ns"])) / HOUR_NS) if fallback else 0.0
            result.append(row)
        return result

    def mask_for(self, row):
        key = row["file"]
        if key in self._mask_cache:
            return self._mask_cache[key]
        path = Path(row["path"])
        source_hash = sha256(path)
        cache_key = _json_hash({"version": CHIP_CACHE_VERSION, "extractor": self.extractor_sha256,
                                "source_sha256": source_hash, "config": [CH_DARK_RATIO, MIN_CH_AREA_FRACTION]})
        cache_path = self.cache_root / f"mask_{cache_key}.npz"
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as npz:
                data = {k: npz[k] for k in npz.files}
            if str(data["source_sha256"].item()) != source_hash or str(data["extractor_sha256"].item()) != self.extractor_sha256:
                raise ValueError(f"stale SUVI CH cache: {cache_path}")
            self.cache_hits += 1
        else:
            try:
                rad, dqf, header = _read_suvi(path)
                built = _component_mask_from_suvi(rad, dqf, header)
            except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
                raise UnusableSuviFrameError(f"unusable SUVI frame {path}: {exc}") from exc
            data = {**built, "source_sha256": np.asarray(source_hash), "extractor_sha256": np.asarray(self.extractor_sha256)}
            temp = cache_path.with_suffix(".tmp")
            with temp.open("wb") as stream:
                np.savez_compressed(stream, **data)
            temp.replace(cache_path)
            self.cache_created += 1
        self._mask_cache[key] = data
        return data


class GongFieldCatalog:
    """Vintage-filtered GONG map list and per-map Br caches."""
    def __init__(self, manifest_path, cap=None, cache_root=CACHE_ROOT):
        self.manifest_path = Path(manifest_path).resolve()
        self.cap_ns = _ns(cap) if cap is not None else None
        self.manifest_sha256 = sha256(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        self.cache_root = Path(cache_root) / "gong"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        rows = []
        for item in self.manifest["maps"]:
            row = dict(item)
            issue_ns = _ns(row["issue_time_utc"])
            obs_ns = _ns(row.get("observation_time_utc", row["issue_time_utc"]))
            avail_ns = _ns(row.get("available_at_utc"))
            if avail_ns is None:
                avail_ns = max(issue_ns + int(GONG_LATENCY_HOURS * HOUR_NS), _ns(row["last_modified"]))
            row.update({"kind": "gong", "slot_ns": issue_ns, "issue_ns": issue_ns,
                        "obs_ns": obs_ns, "available_ns": avail_ns, "file": row["file"]})
            if self.cap_ns is not None and avail_ns > self.cap_ns:
                continue
            rows.append(row)
        self.index = _AvailabilityIndex(rows)
        self.rows = self.index.rows
        self._field_cache = {}
        self.cache_hits = 0
        self.cache_created = 0
        self.extractor_sha256 = MAP_EXTRACTOR_SHA256

    def select_many(self, origins, extra_rows=()):
        return self.index.select_many(origins, extra_rows=extra_rows)

    def field_for(self, row, scale=1.0):
        key = row["file"]
        if key not in self._field_cache:
            path = Path(row["path"])
            if not path.is_file():
                raise FileNotFoundError(f"hash-pinned GONG map missing: {path}")
            if path.stat().st_size != int(row["bytes"]):
                raise ValueError(f"GONG map byte-size mismatch: {row['file']}")
            actual_sha = sha256(path)
            if actual_sha != row["sha256"]:
                raise ValueError(f"GONG map SHA256 mismatch: {row['file']}")
            cache_key = _json_hash({"version": CHIP_CACHE_VERSION, "extractor": self.extractor_sha256,
                                    "source_sha256": actual_sha})
            cache_path = self.cache_root / f"field_{cache_key}.npz"
            if cache_path.is_file():
                with np.load(cache_path, allow_pickle=False) as npz:
                    data = {k: npz[k] for k in npz.files}
                if str(data["source_sha256"].item()) != actual_sha or str(data["extractor_sha256"].item()) != self.extractor_sha256:
                    raise ValueError(f"stale GONG field cache: {cache_path}")
                self.cache_hits += 1
            else:
                header, field = read_gong_fits(path)
                if "CRLN" not in str(header.get("CTYPE1", "")) or "CRLT-CEA" not in str(header.get("CTYPE2", "")):
                    raise ValueError(f"unexpected GONG map WCS: {row['file']}")
                header_date = _ns(header.get("DATE"))
                if header_date is not None and abs(header_date - row["issue_ns"]) > 60 * 1_000_000_000:
                    raise ValueError(f"GONG issue time disagrees with FITS DATE: {row['file']}")
                data = {"field": field.astype(np.float32), "crval1": np.asarray(float(header["CRVAL1"])),
                        "crval2": np.asarray(float(header["CRVAL2"])), "crpix1": np.asarray(float(header["CRPIX1"])),
                        "crpix2": np.asarray(float(header["CRPIX2"])), "cdelt1": np.asarray(float(header["CDELT1"])),
                        "cdelt2": np.asarray(float(header["CDELT2"])), "issue_ns": np.asarray(row["issue_ns"], dtype=np.int64),
                        "source_sha256": np.asarray(actual_sha), "extractor_sha256": np.asarray(self.extractor_sha256)}
                temp = cache_path.with_suffix(".tmp")
                with temp.open("wb") as stream:
                    np.savez_compressed(stream, **data)
                temp.replace(cache_path)
                self.cache_created += 1
            self._field_cache[key] = data
        data = self._field_cache[key]
        if scale == 1.0:
            return data
        result = dict(data)
        result["field"] = np.asarray(data["field"], dtype=np.float32) * np.float32(scale)
        return result


class CmeCatalog:
    """DONKI CMEAnalysis, selecting the newest submitted version at origin."""
    def __init__(self, cap=None, source_dir=DONKI_DIR):
        self.source_dir = Path(source_dir)
        self.cap_ns = _ns(cap) if cap is not None else None
        paths = sorted(self.source_dir.glob("CMEAnalysis_*.json"))
        if not paths:
            raise FileNotFoundError(f"no CMEAnalysis monthly files in {self.source_dir}")
        self.source_hashes = {}
        self.raw_rows = []
        self.by_event = defaultdict(list)
        for path in paths:
            body = path.read_bytes()
            self.source_hashes[path.name] = hashlib.sha256(body).hexdigest()
            payload = json.loads(body)
            if not isinstance(payload, list):
                raise ValueError(f"DONKI monthly source must be a JSON list: {path.name}")
            for index, row in enumerate(payload):
                level = str(row.get("dataLevel"))
                if level not in ("0", "1"):
                    continue
                event_id = row.get("associatedCMEID")
                submitted_ns = _ns(row.get("submissionTime"))
                launch_time = row.get("associatedCMEstartTime") or row.get("time21_5")
                launch_ns = _ns(launch_time)
                if (not event_id or submitted_ns is None or launch_ns is None
                        or (self.cap_ns is not None and submitted_ns > self.cap_ns)):
                    continue
                record = {
                    "event_id": str(event_id), "submission_ns": submitted_ns, "launch_ns": launch_ns,
                    "speed": _finite(row.get("speed")), "longitude": _finite(row.get("longitude")),
                    "latitude": _finite(row.get("latitude")), "halfwidth": _finite(row.get("halfAngle")),
                    "data_level": level, "source_file": path.name, "source_index": index,
                }
                self.by_event[record["event_id"]].append(record)
                self.raw_rows.append(record)
        for rows in self.by_event.values():
            rows.sort(key=lambda r: (r["submission_ns"], r["source_file"], r["source_index"]))
        self.launch_rows = sorted(self.raw_rows, key=lambda r: (r["launch_ns"], r["event_id"], r["submission_ns"]))
        self.launch_ns = [r["launch_ns"] for r in self.launch_rows]
        self._origin_cache = {}
        self.files = [p.name for p in paths]

    @staticmethod
    def _earth_directed(event):
        lon, lat, width = event.get("longitude"), event.get("latitude"), event.get("halfwidth")
        if lon is None or lat is None or width is None or width < 0:
            return False
        cos_sep = math.cos(math.radians(lat)) * math.cos(math.radians(lon))
        sep = math.degrees(math.acos(max(-1.0, min(1.0, cos_sep))))
        return sep <= width

    def events_at(self, origin, extra_records=()):
        origin_ns = _ns(origin)
        cache_key = origin_ns if not extra_records else None
        if cache_key in self._origin_cache:
            return self._origin_cache[cache_key]
        low_ns = origin_ns - 72 * HOUR_NS
        left, right = bisect_left(self.launch_ns, low_ns), bisect_right(self.launch_ns, origin_ns)
        candidate_ids = {r["event_id"] for r in self.launch_rows[left:right] if r["submission_ns"] <= origin_ns}
        chosen = []
        for event_id in candidate_ids:
            versions = self.by_event[event_id]
            submissions = [r["submission_ns"] for r in versions]
            pos = bisect_right(submissions, origin_ns) - 1
            if pos < 0:
                continue
            current = versions[pos]
            if low_ns <= current["launch_ns"] <= origin_ns and self._earth_directed(current):
                chosen.append(current)
        for raw in extra_records:
            event = dict(raw)
            if "submission_ns" not in event:
                event["submission_ns"] = _ns(event.get("submissionTime"))
            if "launch_ns" not in event:
                event["launch_ns"] = _ns(event.get("associatedCMEstartTime") or event.get("time21_5"))
            if "event_id" not in event:
                event["event_id"] = str(event.get("associatedCMEID", "synthetic"))
            for key, source in (("speed", "speed"), ("longitude", "longitude"), ("latitude", "latitude"), ("halfwidth", "halfAngle")):
                event[key] = _finite(event.get(key, event.get(source)))
            if (event["submission_ns"] is not None and event["submission_ns"] <= origin_ns
                    and event["launch_ns"] is not None and low_ns <= event["launch_ns"] <= origin_ns
                    and self._earth_directed(event)):
                chosen.append(event)
        chosen.sort(key=lambda r: (r["launch_ns"], r["event_id"], r["submission_ns"]))
        if cache_key is not None:
            self._origin_cache[cache_key] = chosen
        return chosen


def _grid_components_with_br(mask_data, gong_data, suvi_obs_ns):
    labels = np.asarray(mask_data["labels"])
    area = np.asarray(mask_data["grid_area"], dtype=np.float64)
    field = np.asarray(gong_data["field"], dtype=np.float64)
    ny, nx = field.shape
    rows, cols = np.nonzero(labels)
    if not len(rows):
        return []
    lat = rows.astype(np.float64) - 89.5
    lon_stony = cols.astype(np.float64) - 179.5
    rotation = SYNODIC_RATE_DEG_DAY * ((int(suvi_obs_ns) - int(gong_data["issue_ns"])) / DAY_NS)
    lon_carr = float(gong_data["crval1"]) + rotation + lon_stony
    x = np.rint(float(gong_data["crpix1"]) - 1.0 + (lon_carr - float(gong_data["crval1"])) / float(gong_data["cdelt1"])).astype(np.int64) % nx
    sin_lat = np.sin(np.deg2rad(lat))
    y = np.rint(float(gong_data["crpix2"]) - 1.0 + (sin_lat - float(gong_data["crval2"])) / float(gong_data["cdelt2"])).astype(np.int64)
    valid = (y >= 0) & (y < ny) & np.isfinite(field[y.clip(0, ny - 1), x]) & np.isfinite(area[rows, cols]) & (area[rows, cols] > 0)
    rows, cols, y, x = rows[valid], cols[valid], y[valid], x[valid]
    weights = area[rows, cols]
    br = np.abs(field[y, x])
    comp_ids = labels[rows, cols].astype(np.int64)
    n = int(np.max(labels))
    wsum = np.bincount(comp_ids, weights=weights, minlength=n + 1)
    bsum = np.bincount(comp_ids, weights=weights * br, minlength=n + 1)
    out = []
    for component_id in range(1, n + 1):
        if wsum[component_id] <= 0:
            continue
        out.append({"id": component_id, "mean_absbr": float(bsum[component_id] / wsum[component_id])})
    return out


def _component_table(suvi_row, gong_row, suvi_catalog, gong_catalog, overrides=None):
    overrides = overrides or {}
    mask_data = suvi_catalog.mask_for(suvi_row)
    gong_data = gong_catalog.field_for(gong_row, scale=float(overrides.get("gong_scale", 1.0)))
    pair_key = _json_hash({"version": CHIP_CACHE_VERSION, "extractor": suvi_catalog.extractor_sha256,
                           "suvi": str(mask_data["source_sha256"].item()),
                           "gong": gong_row["sha256"], "gong_extractor": gong_catalog.extractor_sha256,
                           "gong_scale": float(overrides.get("gong_scale", 1.0))})
    pair_path = CACHE_ROOT / "component-br" / f"pair_{pair_key}.json"
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    if pair_path.is_file():
        payload = json.loads(pair_path.read_text())
        if payload.get("suvi_sha256") != str(mask_data["source_sha256"].item()) or payload.get("gong_sha256") != gong_row["sha256"]:
            raise ValueError(f"stale SUVI/GONG component cache: {pair_path}")
        comps_br = payload["components"]
    else:
        comps_br = _grid_components_with_br(mask_data, gong_data, suvi_row["obs_ns"])
        payload = {"cache_key": pair_key, "suvi_sha256": str(mask_data["source_sha256"].item()),
                   "gong_sha256": gong_row["sha256"], "components": comps_br}
        tmp = pair_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        tmp.replace(pair_path)
    areas = np.asarray(mask_data["component_area_fraction"], dtype=np.float64)
    latitudes = np.asarray(mask_data["component_lat"], dtype=np.float64)
    longitudes = np.asarray(mask_data["component_lon_stonyhurst"], dtype=np.float64)
    rotation = SYNODIC_RATE_DEG_DAY * ((int(suvi_row["obs_ns"]) - int(gong_row["issue_ns"])) / DAY_NS)
    by_id = {int(item["id"]): float(item["mean_absbr"]) for item in comps_br}
    out = []
    for i, area_fraction in enumerate(areas, 1):
        if i not in by_id or not math.isfinite(by_id[i]):
            continue
        out.append({"area_fraction": float(area_fraction), "area_km2": float(area_fraction * 4 * math.pi * SOLAR_RADIUS_KM ** 2),
                    "lat": float(latitudes[i - 1]), "lon_stonyhurst": float(longitudes[i - 1]),
                    "lon_carrington": float((float(gong_data["crval1"]) + rotation + longitudes[i - 1]) % 360.0),
                    "mean_absbr": by_id[i]})
    return out, mask_data, gong_data


def _great_circle_deg(lon1, lat1, lon2, lat2):
    dlon = math.radians(_wrapped_delta_deg(lon1, lon2))
    la1, la2 = math.radians(lat1), math.radians(lat2)
    dot = math.sin(la1) * math.sin(la2) + math.cos(la1) * math.cos(la2) * math.cos(dlon)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def build_cme_chip_features(origin, cme_catalog, ch_masks, gong_manifest, config, overrides=None):
    """Build exactly 12 CHIP summary columns for one origin."""
    overrides = overrides or {}
    origin = _utc(origin)
    suvi_extra = overrides.get("future_suvi_rows", ())
    gong_extra = overrides.get("future_gong_rows", ())
    suvi_row = overrides.get("selected_suvi_row")
    gong_row = overrides.get("selected_gong_row")
    if suvi_row is None:
        suvi_row = ch_masks.select_many([origin], extra_rows=suvi_extra)[0]
    if gong_row is None:
        gong_row = gong_manifest.select_many([origin], extra_rows=gong_extra)[0]
    empty = np.full(len(FEATURE_COLUMNS), np.nan, dtype=np.float64)
    empty[FEATURE_COLUMNS.index("chip_max")] = 0.0
    empty[FEATURE_COLUMNS.index("chip_sum")] = 0.0
    empty[FEATURE_COLUMNS.index("chip_dx")] = 0.0
    empty[FEATURE_COLUMNS.index("chip_dy")] = 0.0
    empty[FEATURE_COLUMNS.index("predicted_deflect_lon_sign")] = 0.0
    empty[FEATURE_COLUMNS.index("predicted_deflect_lat_sign")] = 0.0
    empty[FEATURE_COLUMNS.index("chip_missing")] = 1.0
    if suvi_row is None or gong_row is None:
        return empty, {"suvi": None if suvi_row is None else suvi_row["file"],
                       "gong": None if gong_row is None else gong_row["file"], "events": 0, "active": False}
    components, mask_data, gong_data = _component_table(suvi_row, gong_row, ch_masks, gong_manifest, overrides)
    # Keep the fixed 12-column interface: this field is zero for the newest
    # usable image, one for no source, and the selected prior image's age (h)
    # when an otherwise-newer staged image is quality-invalid.
    fallback_age_h = float(suvi_row.get("fallback_age_h", 0.0))
    empty[FEATURE_COLUMNS.index("chip_missing")] = fallback_age_h
    events = cme_catalog.events_at(origin, extra_records=overrides.get("extra_cme_records", ()))
    event_summaries = []
    gong_ref = float(gong_data["crval1"])
    gong_issue = int(gong_data["issue_ns"])
    for event in events:
        if event.get("longitude") is None or event.get("latitude") is None:
            continue
        event_lon = (gong_ref + SYNODIC_RATE_DEG_DAY * ((int(event["launch_ns"]) - gong_issue) / DAY_NS)
                     + float(event["longitude"])) % 360.0
        contributions = []
        for component in components:
            ddeg = _great_circle_deg(event_lon, event["latitude"], component["lon_carrington"], component["lat"])
            radians = math.radians(ddeg)
            distance_km = max(radians * SOLAR_RADIUS_KM, math.radians(0.25) * SOLAR_RADIUS_KM)
            chip = component["area_km2"] * component["mean_absbr"] / (distance_km * distance_km)
            dlon = math.radians(_wrapped_delta_deg(event_lon, component["lon_carrington"]))
            lat_src = math.radians(float(event["latitude"]))
            lat_ch = math.radians(component["lat"])
            east = math.sin(dlon) * math.cos(lat_src)
            north = math.sin(lat_src) * math.cos(lat_ch) - math.cos(lat_src) * math.sin(lat_ch) * math.cos(dlon)
            norm = math.hypot(east, north)
            dx = chip * east / norm if norm > 0 else 0.0
            dy = chip * north / norm if norm > 0 else 0.0
            contributions.append((float(chip), float(dx), float(dy), float(ddeg), component))
        if contributions:
            event_sum = sum(item[0] for item in contributions)
            event_dx = sum(item[1] for item in contributions)
            event_dy = sum(item[2] for item in contributions)
            nearest = min(contributions, key=lambda item: (item[3], -item[0]))
            event_summaries.append({"event": event, "chip": event_sum, "dx": event_dx, "dy": event_dy,
                                    "nearest": nearest[3:], "max_component_chip": max(item[0] for item in contributions)})
        else:
            event_summaries.append({"event": event, "chip": 0.0, "dx": 0.0, "dy": 0.0, "nearest": None,
                                    "max_component_chip": 0.0})
    if event_summaries:
        chosen = max(event_summaries, key=lambda item: (item["chip"], item["event"]["launch_ns"], item["event"]["event_id"]))
        totals = [float(item["chip"]) for item in event_summaries]
        dx = sum(float(item["dx"]) for item in event_summaries)
        dy = sum(float(item["dy"]) for item in event_summaries)
        empty[FEATURE_COLUMNS.index("chip_max")] = max(totals, default=0.0)
        empty[FEATURE_COLUMNS.index("chip_sum")] = sum(totals)
        empty[FEATURE_COLUMNS.index("chip_dx")] = dx
        empty[FEATURE_COLUMNS.index("chip_dy")] = dy
        if chosen["event"].get("speed") is not None:
            empty[FEATURE_COLUMNS.index("cme_speed")] = chosen["event"]["speed"]
        if chosen["event"].get("halfwidth") is not None:
            empty[FEATURE_COLUMNS.index("cme_halfwidth")] = chosen["event"]["halfwidth"]
        if chosen["nearest"] is not None:
            distance, component = chosen["nearest"]
            empty[FEATURE_COLUMNS.index("nearest_ch_dist_deg")] = distance
            empty[FEATURE_COLUMNS.index("nearest_ch_area")] = component["area_km2"]
            empty[FEATURE_COLUMNS.index("nearest_ch_absbr")] = component["mean_absbr"]
        empty[FEATURE_COLUMNS.index("predicted_deflect_lon_sign")] = float(np.sign(dx))
        empty[FEATURE_COLUMNS.index("predicted_deflect_lat_sign")] = float(np.sign(dy))
    summary = {"suvi": suvi_row["file"], "suvi_sha256": str(mask_data["source_sha256"].item()),
               "suvi_available_at": str(pd.Timestamp(suvi_row["available_ns"], tz="UTC")),
               "suvi_quality_fallback": bool(suvi_row.get("quality_fallback", False)),
               "suvi_fallback_age_h": fallback_age_h,
               "gong": gong_row["file"], "gong_sha256": gong_row["sha256"],
               "gong_available_at": str(pd.Timestamp(gong_row["available_ns"], tz="UTC")),
               "events": len(event_summaries), "components": len(components),
               "active": bool(np.isfinite(empty[FEATURE_COLUMNS.index("chip_sum")]) and empty[FEATURE_COLUMNS.index("chip_sum")] > 0)}
    return empty, summary


class ChipFeatureCatalog:
    def __init__(self, cap, manifest_path):
        self.cme = CmeCatalog(cap=cap)
        self.suvi = SuviCHCatalog(cap)
        self.gong = GongFieldCatalog(manifest_path, cap=cap)
        self._features = {}
        self._summaries = {}
        self.config = {"version": CHIP_CACHE_VERSION, "suvi_latency_hours": SUVI_LATENCY_HOURS,
                       "gong_latency_hours": GONG_LATENCY_HOURS, "ch_dark_ratio": CH_DARK_RATIO,
                       "minimum_ch_area_fraction": MIN_CH_AREA_FRACTION,
                       "longitude_alignment": "GONG CRLN WCS plus synodic rotation from SUVI/CME time to map issue time"}

    def features_for(self, origins, overrides=None):
        origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
        if overrides:
            return np.stack([build_cme_chip_features(o, self.cme, self.suvi, self.gong, self.config, overrides)[0]
                             for o in origins])
        missing = [o for o in origins if o.value not in self._features]
        if missing:
            selected_suvi = self.suvi.select_many(missing)
            selected_gong = self.gong.select_many(missing)
            for origin, suvi_row, gong_row in zip(missing, selected_suvi, selected_gong):
                values, summary = build_cme_chip_features(
                    origin, self.cme, self.suvi, self.gong, self.config,
                    overrides={"selected_suvi_row": suvi_row, "selected_gong_row": gong_row})
                self._features[origin.value] = values
                self._summaries[origin.value] = summary
        return np.stack([self._features[o.value] for o in origins]) if len(origins) else np.empty((0, len(FEATURE_COLUMNS)))

    def poison_probe(self, preferred_origin):
        preferred = _utc(preferred_origin)
        candidates = [preferred + pd.Timedelta(hours=h) for h in (0, -24, 24, -48, 48, -72, 72)]
        for origin in candidates:
            if origin.value > self.suvi.cap_ns:
                continue
            rows = self.features_for([origin])
            summary = self._summaries[origin.value]
            if summary.get("suvi") and summary.get("gong") and summary.get("components", 0) > 0:
                # Synthetic record is admitted before origin; it is a positive control for CME source response.
                past_event = {"associatedCMEID": f"poison-past-{origin.value}",
                              "associatedCMEstartTime": str(origin - pd.Timedelta(hours=6)),
                              "submissionTime": str(origin - pd.Timedelta(hours=3)), "dataLevel": "0",
                              "longitude": 0.0, "latitude": 0.0, "halfAngle": 90.0, "speed": 1200.0}
                positive = self.features_for([origin], overrides={"extra_cme_records": [past_event]})[0]
                base = rows[0]
                if not np.allclose(base, positive, equal_nan=True):
                    return origin, base, positive, summary
        raise ValueError("no origin with causal SUVI/GONG CH support responded to an admissible past CME control")

    def future_rows(self, origin):
        origin = _utc(origin)
        suvi_row = self.suvi.select_many([origin])[0]
        gong_row = self.gong.select_many([origin])[0]
        if suvi_row is None or gong_row is None:
            raise ValueError("poison origin lacks causal SUVI or GONG map")
        future_suvi = dict(suvi_row)
        future_suvi.update({"file": "future-poison-suvi", "slot_ns": origin.value - HOUR_NS,
                            "available_ns": origin.value + HOUR_NS, "path": "/not/read/future-poison.nc"})
        future_gong = dict(gong_row)
        future_gong.update({"file": "future-poison-gong", "slot_ns": origin.value - HOUR_NS,
                            "issue_ns": origin.value - HOUR_NS, "available_ns": origin.value + HOUR_NS,
                            "path": "/not/read/future-poison.fits.gz", "sha256": "future-poison"})
        future_cme = {"associatedCMEID": "future-poison-cme", "associatedCMEstartTime": str(origin - pd.Timedelta(hours=6)),
                      "submissionTime": str(origin + pd.Timedelta(hours=1)), "dataLevel": "0",
                      "longitude": 0.0, "latitude": 0.0, "halfAngle": 90.0, "speed": 9999.0}
        return {"future_suvi_rows": [future_suvi], "future_gong_rows": [future_gong],
                "extra_cme_records": [future_cme]}

    def report(self, origins):
        origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
        rows = self.features_for(origins)
        summaries = [self._summaries[o.value] for o in origins]
        def age(key, avail_key):
            vals = []
            for o, s in zip(origins, summaries):
                if s.get(key) and s.get(avail_key):
                    ns = _ns(s[avail_key])
                    if ns is not None:
                        vals.append((o.value - ns) / HOUR_NS)
            return {"median_h": float(np.median(vals)) if vals else None, "max_h": float(np.max(vals)) if vals else None}
        suvi_ok = np.asarray([bool(s.get("suvi")) for s in summaries])
        gong_ok = np.asarray([bool(s.get("gong")) for s in summaries])
        active = np.asarray([bool(s.get("active")) for s in summaries])
        return {"n_origins": len(origins), "suvi_coverage": int(suvi_ok.sum()),
                "suvi_coverage_fraction": float(suvi_ok.mean()) if len(origins) else 0.0,
                "unique_suvi_frames": len({s.get("suvi") for s in summaries if s.get("suvi")}),
                "gong_coverage": int(gong_ok.sum()), "gong_coverage_fraction": float(gong_ok.mean()) if len(origins) else 0.0,
                "unique_gong_maps": len({s.get("gong") for s in summaries if s.get("gong")}),
                "chip_active_origins": int(active.sum()), "chip_active_fraction": float(active.mean()) if len(origins) else 0.0,
                "eligible_cme_origins": int(sum(int(s.get("events", 0)) > 0 for s in summaries)),
                "suvi_age": age("suvi", "suvi_available_at"), "gong_age": age("gong", "gong_available_at"),
                "suvi_quality_fallback_origins": int(sum(bool(s.get("suvi_quality_fallback")) for s in summaries)),
                "suvi_fallback_age_h_max": float(max((float(s.get("suvi_fallback_age_h", 0.0)) for s in summaries), default=0.0)),
                "suvi_unusable_frames": dict(self.suvi.invalid_rows),
                "suvi_integrity_missing_rows": self.suvi.integrity_missing,
                "cache": {"suvi_mask_hits": self.suvi.cache_hits, "suvi_mask_created": self.suvi.cache_created,
                          "gong_map_hits": self.gong.cache_hits, "gong_map_created": self.gong.cache_created},
                "availability": self.config,
                "byte_identity": "not asserted for legacy S3 FITS to local NetCDF conversions; local nc_size integrity plus max(obs_end,S3 LastModified)+2h latency used",
                "cme_catalog_files": self.cme.files, "cme_catalog_sha256": self.cme.source_hashes,
                "gong_manifest_sha256": self.gong.manifest_sha256,
                "suvi_metadata_files": [str(p) for p in self.suvi.metadata_files]}

