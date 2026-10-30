"""Causal polar-cap summaries from hash-pinned GONG mrzqs maps."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


POLAR_COLUMNS = (
    "north_mean_br",
    "south_mean_br",
    "north_abs_br",
    "south_abs_br",
    "polar_signed_diff",
    "polar_abs_sum",
    "polar_asymmetry",
    "polar_missing",
)
POLAR_CONFIG = {
    "latitude_cutoff_deg": 60.0,
    "availability_lag_hours": 2.0,
    "availability_rule": "max(issue_time + 2 h, source HTTP Last-Modified)",
    "cache_dir": "/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store/solar-inc-gongpolar72-20261014-cache",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _card_value(card: str):
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


def read_fits_image(path: Path):
    """Read the primary 2-D FITS image without a runtime astronomy dependency."""
    header_bytes = bytearray()
    with gzip.open(path, "rb") as stream:
        while True:
            block = stream.read(2880)
            if len(block) != 2880:
                raise ValueError(f"truncated FITS header: {path}")
            header_bytes.extend(block)
            cards = [block[i:i + 80] for i in range(0, len(block), 80)]
            if any(card[:8].strip() == b"END" for card in cards):
                break
        header = {}
        for offset in range(0, len(header_bytes), 80):
            card = bytes(header_bytes[offset:offset + 80]).decode("ascii", "replace")
            if card[:8].strip() == "END":
                break
            if card[8:10] == "= ":
                header[card[:8].strip()] = _card_value(card)

        naxis = int(header.get("NAXIS", 0))
        if naxis != 2:
            raise ValueError(f"expected a 2-D GONG map, got NAXIS={naxis}: {path}")
        nx, ny = int(header["NAXIS1"]), int(header["NAXIS2"])
        bitpix = int(header["BITPIX"])
        dtype = {-64: ">f8", -32: ">f4", 16: ">i2", 32: ">i4", 64: ">i8"}.get(bitpix)
        if dtype is None:
            raise ValueError(f"unsupported FITS BITPIX={bitpix}: {path}")
        nbytes = nx * ny * (abs(bitpix) // 8)
        payload = stream.read(nbytes)
        if len(payload) != nbytes:
            raise ValueError(f"truncated FITS image: {path}")

    image = np.frombuffer(payload, dtype=np.dtype(dtype)).reshape(ny, nx).astype(np.float64)
    image *= float(header.get("BSCALE", 1.0))
    image += float(header.get("BZERO", 0.0))
    return header, image


def polar_cap_summaries(header, image, latitude_cutoff_deg=60.0):
    if "CEA" not in str(header.get("CTYPE2", "")) or "CRLT" not in str(header.get("CTYPE2", "")):
        raise ValueError("GONG map axis 2 must be CRLT-CEA")
    ny = image.shape[0]
    rows = np.arange(ny, dtype=float) + 1.0
    sin_latitude = float(header["CRVAL2"]) + (rows - float(header["CRPIX2"])) * float(header["CDELT2"])
    threshold = float(np.sin(np.deg2rad(latitude_cutoff_deg)))
    north = np.broadcast_to((sin_latitude >= threshold)[:, None], image.shape)
    south = np.broadcast_to((sin_latitude <= -threshold)[:, None], image.shape)

    def stats(mask):
        values = image[mask & np.isfinite(image)]
        if values.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(values)), float(np.mean(np.abs(values)))

    north_mean, north_abs = stats(north)
    south_mean, south_abs = stats(south)
    if not np.isfinite([north_mean, north_abs, south_mean, south_abs]).all():
        return [float("nan")] * 7 + [1.0]
    abs_sum = north_abs + south_abs
    return [north_mean, south_mean, north_abs, south_abs,
            north_mean - south_mean, abs_sum,
            (north_abs - south_abs) / max(abs_sum, 1e-6), 0.0]


class GongPolarCatalog:
    def __init__(self, manifest_path, config=None):
        self.manifest_path = Path(manifest_path).resolve()
        self.config = dict(POLAR_CONFIG if config is None else config)
        self.manifest = json.loads(self.manifest_path.read_text())
        self.cache_dir = Path(self.config["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.extractor_sha256 = sha256(Path(__file__).resolve())
        self.rows = []
        for row in self.manifest["maps"]:
            item = dict(row)
            item["observation_ns"] = pd.Timestamp(item["observation_time_utc"]).value
            item["available_ns"] = pd.Timestamp(item["available_at_utc"]).value
            self.rows.append(item)
        self.rows.sort(key=lambda x: (x["observation_ns"], x["file"]))
        self.observation_ns = np.asarray([row["observation_ns"] for row in self.rows], dtype=np.int64)
        self._memory_cache = {}
        self.cache_hits = 0
        self.cache_created = 0
        self.selected_files = set()

    def _index_for(self, origin):
        origin = pd.Timestamp(origin)
        if origin.tz is None:
            origin = origin.tz_localize("UTC")
        else:
            origin = origin.tz_convert("UTC")
        allowed_obs = origin - pd.Timedelta(hours=float(self.config["availability_lag_hours"]))
        index = int(np.searchsorted(self.observation_ns, allowed_obs.value, side="right") - 1)
        while index >= 0 and self.rows[index]["available_ns"] > origin.value:
            index -= 1
        return index, origin

    def _map_features(self, row):
        name = row["file"]
        if name in self._memory_cache:
            return self._memory_cache[name]
        cache_key = hashlib.sha256(
            (self.extractor_sha256 + ":" + row["sha256"]).encode("ascii")
        ).hexdigest()
        cache_path = self.cache_dir / f"polar_{cache_key}.json"
        if cache_path.is_file():
            payload = json.loads(cache_path.read_text())
            if (payload.get("cache_key") != cache_key
                    or payload.get("source_sha256") != row["sha256"]
                    or payload.get("extractor_sha256") != self.extractor_sha256):
                raise ValueError(f"stale GONG feature cache: {cache_path}")
            values = [float("nan") if value is None else float(value) for value in payload["features"]]
            self.cache_hits += 1
        else:
            path = Path(row["path"])
            if not path.is_file():
                raise FileNotFoundError(f"missing hash-pinned GONG map: {path}")
            actual_sha = sha256(path)
            if actual_sha != row["sha256"]:
                raise ValueError(f"GONG map SHA256 mismatch for {row['file']}")
            header, image = read_fits_image(path)
            issue = pd.Timestamp(header.get("DATE", row["issue_time_utc"]))
            if issue.tz is None:
                issue = issue.tz_localize("UTC")
            else:
                issue = issue.tz_convert("UTC")
            recorded = pd.Timestamp(row["issue_time_utc"])
            if recorded.tz is None:
                recorded = recorded.tz_localize("UTC")
            else:
                recorded = recorded.tz_convert("UTC")
            if abs((issue - recorded).total_seconds()) > 60:
                raise ValueError(f"GONG issue timestamp differs from FITS header: {row['file']}")
            values = polar_cap_summaries(header, image, self.config["latitude_cutoff_deg"])
            payload = {
                "cache_key": cache_key,
                "extractor_sha256": self.extractor_sha256,
                "source_sha256": row["sha256"],
                "features": [float(value) if np.isfinite(value) else None for value in values],
            }
            cache_path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
            self.cache_created += 1
        if len(values) != len(POLAR_COLUMNS):
            raise ValueError(f"expected eight GONG polar values for {name}")
        result = np.asarray(values, dtype=float)
        if result[7] not in (0.0, 1.0):
            raise ValueError(f"invalid polar_missing value for {name}")
        self._memory_cache[name] = result
        return result

    def features_for(self, origins, overrides=None, with_metadata=False):
        overrides = {} if overrides is None else overrides
        values = np.full((len(origins), len(POLAR_COLUMNS)), np.nan, dtype=float)
        values[:, -1] = 1.0
        indices = np.full(len(origins), -1, dtype=int)
        metadata = []
        for position, origin in enumerate(origins):
            index, normalized_origin = self._index_for(origin)
            if index < 0:
                metadata.append(None)
                continue
            row = self.rows[index]
            feature_values = overrides.get(row["file"])
            if feature_values is None:
                feature_values = self._map_features(row)
            values[position] = np.asarray(feature_values, dtype=float)
            indices[position] = index
            self.selected_files.add(row["file"])
            metadata.append({
                "file": row["file"],
                "observation_time_utc": row["observation_time_utc"],
                "available_at_utc": row["available_at_utc"],
                "age_hours": (normalized_origin.value - row["observation_ns"]) / 3.6e12,
                "sha256": row["sha256"],
                "url": row["url"],
                "retrieval_time_utc": row.get("retrieval_time_utc"),
            })
        if with_metadata:
            return values, indices, metadata
        return values

    def poison_overrides(self, origin):
        index, normalized_origin = self._index_for(origin)
        if index < 0:
            raise ValueError("no admissible GONG map for poison probe")
        selected = self.rows[index]
        future = next((row for row in self.rows if row["observation_ns"] > normalized_origin.value), None)
        if future is None:
            raise ValueError("no future GONG map available for poison probe")
        base = self._map_features(selected)
        future_poison = {future["file"]: np.asarray([999.0] * 7 + [0.0])}
        positive = {selected["file"]: np.asarray(
            [value + 13.0 if np.isfinite(value) else 13.0 for value in base[:7]] + [0.0]
        )}
        return future_poison, positive, selected, future

    def official_report(self, origins):
        values, indices, metadata = self.features_for(origins, with_metadata=True)
        covered = indices >= 0
        ages = np.asarray([item["age_hours"] for item in metadata if item is not None], dtype=float)
        variance = {}
        for i, name in enumerate(POLAR_COLUMNS):
            finite = values[:, i][np.isfinite(values[:, i])]
            variance[name] = float(np.var(finite)) if finite.size else None
        return {
            "source_coverage_origins": int(covered.sum()),
            "source_coverage_total": int(len(origins)),
            "source_coverage_fraction": float(covered.mean()) if len(origins) else 0.0,
            "unique_selected_maps": int(len({item["file"] for item in metadata if item is not None})),
            "feature_variance": variance,
            "age_hours_min_median_p95_max": {
                "min": float(np.min(ages)) if ages.size else None,
                "median": float(np.median(ages)) if ages.size else None,
                "p95": float(np.quantile(ages, 0.95)) if ages.size else None,
                "max": float(np.max(ages)) if ages.size else None,
            },
            "latest_selected_map": max(
                (item["observation_time_utc"] for item in metadata if item is not None), default=None
            ),
            "all_selected_maps_have_hash_and_url": bool(all(
                item is not None and item["sha256"] and item["url"] and item["retrieval_time_utc"]
                for item in metadata if item is not None
            )),
            "cache_created_this_process": int(self.cache_created),
            "cache_hits_this_process": int(self.cache_hits),
            "cache_entries_used_this_process": int(len(self._memory_cache)),
        }


_CATALOGS = {}


def build_polar_field_features(origin, gong_manifest, config):
    """Return the eight causal polar-cap values for one origin."""
    key = (str(Path(gong_manifest).resolve()), json.dumps(config, sort_keys=True))
    catalog = _CATALOGS.get(key)
    if catalog is None:
        catalog = GongPolarCatalog(gong_manifest, config)
        _CATALOGS[key] = catalog
    return catalog.features_for([origin])[0]
