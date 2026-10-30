"""Causal low-order spherical-harmonic summaries from pinned GONG maps."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import HTTPException
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from gong_polar import read_fits_image, sha256
EXPECTED_MANIFEST_SHA256 = None  # v2 source manifest is hashed by the isolated builder


MULTIPOLE_COLUMNS = (
    "dipole_axial_l1m0",
    "dipole_equatorial_amp_l1",
    "dipole_equatorial_phase_sin",
    "dipole_equatorial_phase_cos",
    "quadrupole_amp_l2",
    "octupole_amp_l3",
    "dipole_tilt_deg",
    "dipole_flux_norm",
    "loworder_power_frac_l1_3",
    "axial_to_equatorial_ratio",
    "map_age_h",
    "multipole_missing",
)

MULTIPOLE_CONFIG = {
    "availability_lag_hours": 2.0,
    "availability_rule": "manifest available_at_utc=max(issue_time+2h, HTTP Last-Modified); choose latest observation with available_at<=origin",
    "map_cache_dir": "/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store/solar-inc-gongmulti72-20261014/gong_fits",
    "feature_cache_dir": "/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store/solar-inc-gongmulti72-20261014/multipole-cache",
    "fetch_workers": 3,
    "download_attempts": 5,
    "download_retry_backoff_seconds": [2, 4, 8, 16],
    "coordinates": "Carrington longitude phi from CRLN-CEA; mu=sin(CRLT-CEA latitude)",
    "projection": "area-weighted real orthonormal spherical harmonics through l=3; Condon-Shortley phase; coefficient=int(Br*Y_lm dOmega)",
    "real_harmonics": "m>0 cosine and sine bases each sqrt(2)*Re/Im(complex Y_lm); unit norm on S2",
    "dipole_phase": "atan2(a_11_sin,a_11_cos); feature values are its sine and cosine",
    "dipole_tilt": "degrees(atan2(equatorial amplitude, abs(axial coefficient)))",
    "dipole_flux_norm": "dipole coefficient-vector norm divided by area-weighted map RMS field",
    "loworder_power_fraction": "sum(a_lm^2 for l=1..3) / area-weighted integral(Br^2 dOmega), clipped to [0,1] for quadrature roundoff",
    "axial_equatorial_ratio": "abs(a_10)/equatorial amplitude; missing if denominator is numerically zero",
    "minimum_finite_fraction": 0.95,
}


def _utc(value):
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tz is None else timestamp.tz_convert("UTC")


def _normalization(l, m):
    return math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - m) / math.factorial(l + m))


def real_harmonic_basis(header, shape):
    """Return pixel-centered real Y_lm through l=3 on the GONG CEA grid."""
    if "CEA" not in str(header.get("CTYPE1", "")) or "CRLN" not in str(header.get("CTYPE1", "")):
        raise ValueError("GONG map axis 1 must be CRLN-CEA")
    if "CEA" not in str(header.get("CTYPE2", "")) or "CRLT" not in str(header.get("CTYPE2", "")):
        raise ValueError("GONG map axis 2 must be CRLT-CEA")
    ny, nx = shape
    rows = np.arange(ny, dtype=float) + 1.0
    columns = np.arange(nx, dtype=float) + 1.0
    mu = float(header["CRVAL2"]) + (rows - float(header["CRPIX2"])) * float(header["CDELT2"])
    if np.any(np.abs(mu) > 1.00001):
        raise ValueError("GONG CEA latitude coordinate lies outside [-1,1]")
    mu = np.clip(mu, -1.0, 1.0)
    phi = np.deg2rad(float(header["CRVAL1"]) +
                     (columns - float(header["CRPIX1"])) * float(header["CDELT1"]))
    root = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
    legendre = {
        (0, 0): np.ones_like(mu),
        (1, 0): mu,
        (1, 1): -root,
        (2, 0): 0.5 * (3.0 * mu * mu - 1.0),
        (2, 1): -3.0 * mu * root,
        (2, 2): 3.0 * root * root,
        (3, 0): 0.5 * (5.0 * mu ** 3 - 3.0 * mu),
        (3, 1): -1.5 * (5.0 * mu * mu - 1.0) * root,
        (3, 2): 15.0 * mu * root * root,
        (3, 3): -15.0 * root ** 3,
    }
    basis = {}
    for l in range(4):
        for m in range(l + 1):
            amplitude = _normalization(l, m) * legendre[(l, m)][:, None]
            if m == 0:
                basis[(l, m, "cos")] = np.broadcast_to(amplitude, (ny, nx))
            else:
                amplitude = math.sqrt(2.0) * amplitude
                basis[(l, m, "cos")] = amplitude * np.cos(m * phi)[None, :]
                basis[(l, m, "sin")] = amplitude * np.sin(m * phi)[None, :]
    return basis


def project_loworder(header, image, config=None):
    """Area-weighted projection of Br onto the real l<=3 harmonic basis."""
    config = MULTIPOLE_CONFIG if config is None else config
    image = np.asarray(image, dtype=np.float64)
    basis = real_harmonic_basis(header, image.shape)
    area = abs(math.radians(float(header["CDELT1"])) * float(header["CDELT2"]))
    if not math.isfinite(area) or area <= 0:
        raise ValueError("invalid CRLN-CEA / CRLT-CEA pixel solid angle")
    sphere_area = area * image.size
    if abs(sphere_area - 4.0 * math.pi) / (4.0 * math.pi) > 0.01:
        raise ValueError(f"CEA grid area differs from 4pi by {sphere_area / (4*math.pi) - 1:.4%}")
    finite = np.isfinite(image)
    if float(finite.mean()) < float(config["minimum_finite_fraction"]):
        raise ValueError("GONG map has insufficient finite area for a global multipole projection")
    coefficients = {}
    if finite.all():
        for key, values in basis.items():
            coefficients[key] = float(np.sum(image * values, dtype=np.float64) * area)
    else:
        # Missing pixels break harmonic orthogonality; solve the area-weighted
        # least-squares projection over the observed pixels instead.
        keys = list(basis)
        design = np.column_stack([basis[key][finite] for key in keys])
        root_weight = math.sqrt(area)
        fit, _, rank, _ = np.linalg.lstsq(design * root_weight,
                                          image[finite] * root_weight, rcond=None)
        if rank != len(keys):
            raise ValueError("incomplete GONG map does not identify all l<=3 coefficients")
        coefficients.update({key: float(value) for key, value in zip(keys, fit)})
    return coefficients, area


def build_multipole_features(origin, gong_map, config=None):
    """Return the requested 12 causal global-field features for one selected map."""
    config = MULTIPOLE_CONFIG if config is None else config
    if gong_map is None:
        return np.asarray([np.nan] * 11 + [1.0], dtype=float)
    header, image, observation_time = gong_map["header"], gong_map["image"], gong_map["observation_time_utc"]
    coefficients, area = project_loworder(header, image, config)
    axial = coefficients[(1, 0, "cos")]
    equatorial_cos = coefficients[(1, 1, "cos")]
    equatorial_sin = coefficients[(1, 1, "sin")]
    equatorial = math.hypot(equatorial_cos, equatorial_sin)
    dipole_norm = math.hypot(axial, equatorial)
    rms = math.sqrt(float(np.nansum(np.asarray(image, dtype=float) ** 2) * area / (4.0 * math.pi)))
    if not math.isfinite(rms) or rms <= np.finfo(float).tiny:
        return np.asarray([np.nan] * 11 + [1.0], dtype=float)
    power_l = {l: sum(value * value for (degree, _, _), value in coefficients.items() if degree == l)
               for l in range(4)}
    loworder_fraction = float(np.clip(sum(power_l[l] for l in (1, 2, 3)) /
                                      max(float(np.nansum(np.asarray(image, dtype=float) ** 2) * area),
                                          np.finfo(float).tiny), 0.0, 1.0))
    phase_norm = equatorial
    if phase_norm > np.finfo(float).eps * max(rms, 1.0):
        phase_sin, phase_cos = equatorial_sin / phase_norm, equatorial_cos / phase_norm
        axial_ratio = abs(axial) / phase_norm
    else:
        phase_sin = phase_cos = axial_ratio = np.nan
    age_hours = (_utc(origin) - _utc(observation_time)).total_seconds() / 3600.0
    values = np.asarray([
        axial,
        equatorial,
        phase_sin,
        phase_cos,
        math.sqrt(max(0.0, power_l[2])),
        math.sqrt(max(0.0, power_l[3])),
        math.degrees(math.atan2(equatorial, abs(axial))),
        dipole_norm / rms,
        loworder_fraction,
        axial_ratio,
        age_hours,
        0.0,
    ], dtype=float)
    if values.shape != (len(MULTIPOLE_COLUMNS),):
        raise AssertionError("multipole feature contract changed")
    return values


class GongMultipoleCatalog:
    def __init__(self, manifest_path, config=None):
        self.manifest_path = Path(manifest_path).resolve()
        self.config = dict(MULTIPOLE_CONFIG if config is None else config)
        self.manifest_sha256 = sha256(self.manifest_path)
        if self.manifest_sha256 != EXPECTED_MANIFEST_SHA256:
            raise ValueError(f"GONG source manifest hash changed: {self.manifest_sha256}")
        self.manifest = json.loads(self.manifest_path.read_text())
        self.map_cache_dir = Path(self.config["map_cache_dir"])
        self.feature_cache_dir = Path(self.config["feature_cache_dir"])
        self.map_cache_dir.mkdir(parents=True, exist_ok=True)
        self.feature_cache_dir.mkdir(parents=True, exist_ok=True)
        self.extractor_sha256 = sha256(Path(__file__).resolve())
        self.config_sha256 = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        self.rows = []
        for row in self.manifest["maps"]:
            item = dict(row)
            item["observation_ns"] = pd.Timestamp(item["observation_time_utc"]).value
            item["available_ns"] = pd.Timestamp(item["available_at_utc"]).value
            self.rows.append(item)
        self.rows.sort(key=lambda row: (row["observation_ns"], row["file"]))
        self.observation_ns = np.asarray([row["observation_ns"] for row in self.rows], dtype=np.int64)
        self._memory_cache = {}
        self._decoded_maps = {}
        self.selected_files = set()
        self.cache_hits = 0
        self.cache_created = 0
        self.downloaded = 0
        self.restored = 0
        self._receipt_lock = threading.Lock()
        self.receipt_path = self.map_cache_dir.parent / "source-materialization.jsonl"

    def _index_for(self, origin):
        origin = _utc(origin)
        eligible_observation = origin - pd.Timedelta(hours=float(self.config["availability_lag_hours"]))
        index = int(np.searchsorted(self.observation_ns, eligible_observation.value, side="right") - 1)
        while index >= 0 and self.rows[index]["available_ns"] > origin.value:
            index -= 1
        return index, origin

    def _target_path(self, row):
        return self.map_cache_dir / row["file"]

    def _append_receipt(self, entry):
        with self._receipt_lock:
            with self.receipt_path.open("a") as stream:
                stream.write(json.dumps(entry, sort_keys=True) + "\n")

    def _materialize(self, row):
        target = self._target_path(row)
        if target.is_file():
            if target.stat().st_size != int(row["bytes"]) or sha256(target) != row["sha256"]:
                raise ValueError(f"materialized GONG map failed manifest verification: {target}")
            return target
        original = Path(row["path"])
        method, last_modified = "copy-existing-store", None
        temporary = target.with_name(target.name + f".{os.getpid()}.part")
        if original.is_file():
            shutil.copyfile(original, temporary)
        else:
            method = "recover-exact-manifest-object"
            request = Request(row["url"], headers={"User-Agent": "sunrun-research/1.0"})
            attempts = int(self.config["download_attempts"])
            backoff = self.config["download_retry_backoff_seconds"]
            for attempt in range(attempts):
                try:
                    with urlopen(request, timeout=90) as response:
                        body = response.read()
                        last_modified = response.headers.get("Last-Modified")
                    break
                # HTTPResponse.read() can raise IncompleteRead (HTTPException)
                # when the server closes a body early; treat it like the other
                # transient transport failures and retry the pinned object.
                except (URLError, TimeoutError, OSError, HTTPException) as exc:
                    retryable_http = not isinstance(exc, HTTPError) or exc.code in (408, 429) or exc.code >= 500
                    if not retryable_http or attempt + 1 >= attempts:
                        raise RuntimeError(
                            f"could not retrieve pinned GONG object {row['file']} from {row['url']} "
                            f"after {attempt + 1} attempt(s): {exc}"
                        ) from exc
                    delay = float(backoff[min(attempt, len(backoff) - 1)])
                    print(f"retry GONG fetch {row['file']} after {exc}; attempt {attempt + 2}/{attempts} "
                          f"in {delay:g}s", flush=True)
                    time.sleep(delay)
            if len(body) != int(row["bytes"]):
                raise ValueError(f"GONG content length changed for {row['file']}: {len(body)} != {row['bytes']}")
            temporary.write_bytes(body)
        if temporary.stat().st_size != int(row["bytes"]) or sha256(temporary) != row["sha256"]:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"recovered GONG object does not match pinned bytes: {row['file']}")
        if last_modified:
            parsed = parsedate_to_datetime(last_modified)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            observed_lm = pd.Timestamp(parsed.astimezone(timezone.utc))
            recorded_available = _utc(row["available_at_utc"])
            if observed_lm > recorded_available + pd.Timedelta(minutes=1):
                temporary.unlink(missing_ok=True)
                raise ValueError(f"HTTP Last-Modified exceeds manifest availability for {row['file']}: {last_modified}")
        os.replace(temporary, target)
        if method == "copy-existing-store":
            self.restored += 1
        else:
            self.downloaded += 1
        self._append_receipt({
            "file": row["file"], "url": row["url"], "source_sha256": row["sha256"],
            "bytes": int(row["bytes"]), "method": method,
            "source_last_modified": last_modified,
            "manifest_available_at_utc": row["available_at_utc"],
            "materialized_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        return target

    def _decoded(self, row):
        name = row["file"]
        if name in self._decoded_maps:
            return self._decoded_maps[name]
        path = self._materialize(row)
        header, image = read_fits_image(path)
        issue = _utc(header.get("DATE", row["issue_time_utc"]))
        recorded_issue = _utc(row["issue_time_utc"])
        if abs((issue - recorded_issue).total_seconds()) > 60:
            raise ValueError(f"GONG issue timestamp differs from FITS header: {name}")
        result = (header, image)
        self._decoded_maps[name] = result
        return result

    def _feature_values(self, row, origin):
        name = row["file"]
        age_hours = (_utc(origin) - _utc(row["observation_time_utc"])).total_seconds() / 3600.0
        if name in self._memory_cache:
            result = self._memory_cache[name].copy()
            result[-2] = age_hours
            return result
        key_payload = {"extractor_sha256": self.extractor_sha256,
                       "source_sha256": row["sha256"], "config_sha256": self.config_sha256}
        cache_key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()
        cache_path = self.feature_cache_dir / f"multi_{cache_key}.json"
        if cache_path.is_file():
            payload = json.loads(cache_path.read_text())
            if (payload.get("cache_key") != cache_key or payload.get("source_sha256") != row["sha256"]
                    or payload.get("extractor_sha256") != self.extractor_sha256
                    or payload.get("config_sha256") != self.config_sha256):
                raise ValueError(f"stale GONG multipole cache: {cache_path}")
            values = np.asarray([np.nan if value is None else float(value) for value in payload["features"]])
            self.cache_hits += 1
        else:
            header, image = self._decoded(row)
            values = build_multipole_features(row["observation_time_utc"], {
                "header": header, "image": image, "observation_time_utc": row["observation_time_utc"]
            }, self.config)
            values[-2] = np.nan  # map-age is origin-dependent and is added after cache lookup.
            payload = {**key_payload, "cache_key": cache_key,
                       "features": [float(value) if np.isfinite(value) else None for value in values]}
            cache_path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
            self.cache_created += 1
        if values.shape != (len(MULTIPOLE_COLUMNS),) or values[-1] not in (0.0, 1.0):
            raise ValueError(f"invalid GONG multipole feature vector for {name}")
        self._memory_cache[name] = values
        result = values.copy()
        result[-2] = age_hours
        return result

    def features_for(self, origins, overrides=None, with_metadata=False):
        overrides = {} if overrides is None else overrides
        origins = pd.DatetimeIndex(origins)
        values = np.full((len(origins), len(MULTIPOLE_COLUMNS)), np.nan, dtype=float)
        values[:, -1] = 1.0
        indices = np.full(len(origins), -1, dtype=int)
        metadata = [None] * len(origins)
        selections = []
        for position, origin in enumerate(origins):
            index, normalized = self._index_for(origin)
            if index < 0:
                continue
            row = self.rows[index]
            selections.append((position, normalized, index, row))
        unique = {row["file"]: row for _, _, _, row in selections}
        missing = [row for row in unique.values()
                   if not self._target_path(row).is_file() and not Path(row["path"]).is_file()]
        if missing:
            with ThreadPoolExecutor(max_workers=int(self.config["fetch_workers"])) as pool:
                futures = {pool.submit(self._materialize, row): row for row in missing}
                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        raise RuntimeError(f"failed to materialize selected GONG map {row['file']} "
                                           f"({row['url']}): {exc}") from exc
        for position, normalized, index, row in selections:
            feature_values = overrides.get(row["file"])
            if feature_values is None:
                feature_values = self._feature_values(row, normalized)
            values[position] = np.asarray(feature_values, dtype=float)
            indices[position] = index
            self.selected_files.add(row["file"])
            metadata[position] = {
                "file": row["file"], "observation_time_utc": row["observation_time_utc"],
                "available_at_utc": row["available_at_utc"],
                "age_hours": (normalized.value - row["observation_ns"]) / 3.6e12,
                "sha256": row["sha256"], "url": row["url"],
                "retrieval_time_utc": row.get("retrieval_time_utc"),
            }
        return (values, indices, metadata) if with_metadata else values

    def poison_overrides(self, origin):
        index, normalized = self._index_for(origin)
        if index < 0:
            raise ValueError("no admissible GONG map for multipole poison probe")
        selected = self.rows[index]
        future = next((row for row in self.rows if row["observation_ns"] > normalized.value), None)
        if future is None:
            raise ValueError("no future GONG map for multipole poison probe")
        header, image = self._decoded(selected)
        future_header, future_image = self._decoded(future)
        selected_basis = np.asarray([float(header["CRVAL2"]) +
                                     (r + 1.0 - float(header["CRPIX2"])) * float(header["CDELT2"])
                                     for r in range(image.shape[0])])[:, None]
        future_basis = np.asarray([float(future_header["CRVAL2"]) +
                                   (r + 1.0 - float(future_header["CRPIX2"])) * float(future_header["CDELT2"])
                                   for r in range(future_image.shape[0])])[:, None]
        changed_selected = build_multipole_features(normalized, {
            "header": header, "image": image + 13.0 * selected_basis,
            "observation_time_utc": selected["observation_time_utc"]}, self.config)
        changed_future = build_multipole_features(normalized, {
            "header": future_header, "image": future_image + 100000.0 * future_basis,
            "observation_time_utc": future["observation_time_utc"]}, self.config)
        return ({future["file"]: changed_future}, {selected["file"]: changed_selected}, selected, future)

    def official_report(self, origins):
        values, indices, metadata = self.features_for(origins, with_metadata=True)
        covered = indices >= 0
        ages = np.asarray([item["age_hours"] for item in metadata if item is not None], dtype=float)
        variance = {}
        finite_fraction = {}
        for i, name in enumerate(MULTIPOLE_COLUMNS):
            finite = values[:, i][np.isfinite(values[:, i])]
            variance[name] = float(np.var(finite)) if finite.size else None
            finite_fraction[name] = float(finite.size / len(values)) if len(values) else 0.0
        return {
            "source_coverage_origins": int(covered.sum()), "source_coverage_total": int(len(origins)),
            "source_coverage_fraction": float(covered.mean()) if len(origins) else 0.0,
            "feature_finite_fraction": finite_fraction, "feature_variance": variance,
            "unique_selected_maps": int(len({item["file"] for item in metadata if item is not None})),
            "age_hours_min_median_p95_max": {
                "min": float(np.min(ages)) if ages.size else None,
                "median": float(np.median(ages)) if ages.size else None,
                "p95": float(np.quantile(ages, 0.95)) if ages.size else None,
                "max": float(np.max(ages)) if ages.size else None,
            },
            "latest_selected_map": max((item["observation_time_utc"] for item in metadata if item is not None),
                                       default=None),
            "all_selected_maps_have_hash_and_url": bool(all(
                item is not None and item["sha256"] and item["url"] and item["retrieval_time_utc"]
                for item in metadata if item is not None)),
            "cache_created_this_process": int(self.cache_created),
            "cache_hits_this_process": int(self.cache_hits),
            "cache_entries_used_this_process": int(len(self._memory_cache)),
            "raw_maps_copied_from_store": int(self.restored),
            "raw_maps_recovered_from_exact_manifest_url": int(self.downloaded),
            "source_materialization_receipt": str(self.receipt_path),
            "feature_cache_key_includes_code_source_and_config_sha256": True,
        }
