"""Causal CH-local magnetic flux imbalance on the frozen E1198 source pair."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.solar_inc_chmorph72_20261014 import morph_features, spec as morph_spec
import cmechip as chip

LB = Path('/home/t-lab01/sunrun/leaderboard')
STORE_ROOT = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store')

FEATURE_CACHE = STORE_ROOT / "v2/chunip/cache"
PAIR_CACHE = FEATURE_CACHE / "pairs"
FEATURE_COLUMNS = (
    "ch_phi_imb_abs", "ch_phi_imb_signed", "ch_br_mean", "ch_br_abs_mean",
    "ch_br_median", "ch_br_std", "ch_skew_pearson", "ch_dompol_area_frac",
    "ch_signed_flux_norm", "ch_unsigned_flux_norm", "ch_eq20_phi_imb_abs",
    "ch_cm30_phi_imb_abs", "ch_valid_overlap_frac", "ch_unip_missing",
)
MISSING_INDEX = len(FEATURE_COLUMNS) - 1
HOUR_NS = 3_600_000_000_000
DAY_NS = 24 * HOUR_NS
SYNODIC_RATE_DEG_DAY = 360.0 / 27.2753
CONFIG = {
    "mask_selector": "E1198/E1022 causal MorphologyContext: slot<=origin, obs_end+1h<=origin, S3 LastModified<=origin",
    "mask_representation": "existing t035 CH area grid (12 longitude x 3 latitude cells); no new segmentation",
    "gong_selector": "E1198 hash-pinned manifest; available_at<=origin, existing selector",
    "minimum_valid_cells": 1,
    "field_statistics": "surface-area weighted; population std; weighted median",
    "flux_norm": "signed/unsigned CH-integrated Br*dA divided by valid CH area",
    "eq20_latitude_deg": 20.0,
    "cm30_stonyhurst_deg": 30.0,
    "alignment": "existing 12x3 t035 area-grid cell centers -> GONG CRLN/CRLT-CEA nearest pixel; synodic rotation from mask obs_end to GONG issue",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ns(value) -> int:
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tz is None else stamp.tz_convert("UTC")
    return int(stamp.value)


def _weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    index = min(int(np.searchsorted(np.cumsum(weights), weights.sum() / 2.0, side="left")), len(values) - 1)
    return float(values[index])


def _imbalance(field, weights):
    if len(field) < CONFIG["minimum_valid_cells"] or float(weights.sum()) <= 0:
        return float("nan")
    signed = float(np.dot(field, weights))
    unsigned = float(np.dot(np.abs(field), weights))
    return abs(signed / unsigned) if unsigned > 0 else float("nan")


def _aligned_cell_data(ch_mask, gong_map):
    area_grid = np.asarray(ch_mask["area_grid"], dtype=np.float64).reshape(3, 12)
    field = np.asarray(gong_map["field"], dtype=np.float64)
    lat_centers = np.asarray(morph_spec.LAT_CENTERS, dtype=np.float64)
    lon_centers = np.asarray(morph_spec.LON_CENTERS, dtype=np.float64)
    rows, cols = np.nonzero(np.isfinite(area_grid) & (area_grid > 0))
    if not len(rows):
        return {"area_grid": area_grid, "rows": rows, "cols": cols, "latitude": np.array([]),
                "longitude": np.array([]), "br": np.array([]), "cell_area": np.array([]),
                "valid": np.array([], dtype=bool)}

    latitude = lat_centers[rows]
    longitude = lon_centers[cols]
    rotation = SYNODIC_RATE_DEG_DAY * (
        (_ns(ch_mask["obs_end_utc"]) - int(gong_map["issue_ns"])) / DAY_NS
    )
    crval1, crval2 = float(gong_map["crval1"]), float(gong_map["crval2"])
    crpix1, crpix2 = float(gong_map["crpix1"]), float(gong_map["crpix2"])
    cdelt1, cdelt2 = float(gong_map["cdelt1"]), float(gong_map["cdelt2"])
    ny, nx = field.shape
    carrington_lon = crval1 + rotation + longitude
    x = np.rint(crpix1 - 1.0 + (carrington_lon - crval1) / cdelt1).astype(np.int64) % nx
    sin_latitude = np.sin(np.deg2rad(latitude))
    y = np.rint(crpix2 - 1.0 + (sin_latitude - crval2) / cdelt2).astype(np.int64)
    cell_area = area_grid[rows, cols]
    in_bounds = (y >= 0) & (y < ny) & np.isfinite(cell_area) & (cell_area > 0)
    br = field[y.clip(0, ny - 1), x]
    valid = in_bounds & np.isfinite(br)
    return {"area_grid": area_grid, "rows": rows, "cols": cols, "latitude": latitude,
            "longitude": longitude, "br": br, "cell_area": cell_area, "valid": valid}


def build_ch_unip_features(origin, ch_mask, gong_map, config=CONFIG):
    """Summarize GONG field with the existing coarse, causal CH area mask.

    The stored E1198/E1022 CH representation is a 12x3 t035 area grid rather
    than a pixel mask. Each occupied bin contributes its recorded CH area and
    uses the nearest GONG CEA pixel at that bin's center. This preserves the
    existing mask and avoids making a second segmentation.
    """
    out = np.full(len(FEATURE_COLUMNS), np.nan, dtype=np.float64)
    out[12] = 0.0
    out[MISSING_INDEX] = 1.0
    origin_ns = _ns(origin)
    mask_available = max(_ns(ch_mask["obs_end_utc"]) + HOUR_NS, _ns(ch_mask["s3_modified_utc"]))
    if (_ns(ch_mask["slot_utc"]) > origin_ns or mask_available > origin_ns
            or int(gong_map["available_ns"]) > origin_ns or int(gong_map["issue_ns"]) > origin_ns):
        raise ValueError("CH mask/GONG map is not origin-admissible")

    aligned = _aligned_cell_data(ch_mask, gong_map)
    if not len(aligned["rows"]):
        return out
    area_grid = aligned["area_grid"]
    latitude = aligned["latitude"]
    longitude_stonyhurst = aligned["longitude"]
    br = aligned["br"]
    cell_area = aligned["cell_area"]
    valid = aligned["valid"]
    total_ch_area = float(cell_area.sum())
    valid_area = float(cell_area[valid].sum())
    out[12] = valid_area / total_ch_area if total_ch_area > 0 else 0.0
    if valid.sum() < int(config["minimum_valid_cells"]) or valid_area <= 0:
        return out

    br = br[valid]
    weights = cell_area[valid]
    total_area = float(weights.sum())
    signed_flux = float(np.dot(br, weights))
    unsigned_flux = float(np.dot(np.abs(br), weights))
    if unsigned_flux <= 0 or not np.isfinite([signed_flux, unsigned_flux]).all():
        return out
    mean = signed_flux / total_area
    abs_mean = unsigned_flux / total_area
    median = _weighted_median(br, weights)
    variance = float(np.dot((br - mean) ** 2, weights) / total_area)
    std = float(np.sqrt(max(variance, 0.0)))
    out[:6] = [abs(signed_flux / unsigned_flux), signed_flux / unsigned_flux,
               mean, abs_mean, median, std]
    out[7] = max(float(weights[br > 0].sum()), float(weights[br < 0].sum())) / total_area
    out[8] = signed_flux / total_area
    out[9] = unsigned_flux / total_area
    if std > 0:
        out[6] = 3.0 * (mean - median) / std

    equatorial = np.abs(latitude[valid]) <= float(config["eq20_latitude_deg"])
    central = np.abs(longitude_stonyhurst[valid]) <= float(config["cm30_stonyhurst_deg"])
    out[10] = _imbalance(br[equatorial], weights[equatorial])
    out[11] = _imbalance(br[central], weights[central])
    out[MISSING_INDEX] = 0.0
    return out


class CHUnipCatalog:
    """Causal latest-eligible SUVI masks joined with E1198 GONG vintages."""

    def __init__(self, cap, gong_manifest_path, expected_gong_manifest_sha256):
        self.cap = pd.Timestamp(cap)
        self.cache_dir = FEATURE_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        PAIR_CACHE.mkdir(parents=True, exist_ok=True)
        morph_spec.STORE = FEATURE_CACHE / "ch-selector"
        self.mask_context = morph_features.MorphologyContext(self.cap)
        self.gong = chip.GongFieldCatalog(
            gong_manifest_path, cap=self.cap, cache_root=FEATURE_CACHE / "source-cache"
        )
        if self.gong.manifest_sha256 != expected_gong_manifest_sha256:
            raise ValueError(f"E1198 GONG manifest changed: {self.gong.manifest_sha256}")
        self.manifest_sha256 = self.gong.manifest_sha256
        self.helper_hashes = {
            str(Path(__file__).resolve()): _sha(Path(__file__).resolve()),
            str(Path(morph_features.__file__).resolve()): _sha(Path(morph_features.__file__).resolve()),
            str(Path(morph_spec.__file__).resolve()): _sha(Path(morph_spec.__file__).resolve()),
            str(Path(chip.__file__).resolve()): _sha(Path(chip.__file__).resolve()),
        }
        self.cache_hits = 0
        self.cache_created = 0
        self._memory = {}
        self._last_summaries = []

    def _selected_mask_rows(self, origins, extra_rows=()):
        origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
        selected = self.mask_context.select_indices(origins)
        result = []
        for position, candidate_index in enumerate(selected):
            origin_ns = int(origins[position].value)
            chosen = self.mask_context.candidates.iloc[int(candidate_index)] if candidate_index >= 0 else None
            chosen_slot = _ns(chosen["slot"]) if chosen is not None else -1
            for extra in extra_rows:
                slot_ns = _ns(extra["slot"])
                if (slot_ns <= origin_ns and _ns(extra["obs_end"]) + HOUR_NS <= origin_ns
                        and _ns(extra["s3_modified"]) <= origin_ns and slot_ns > chosen_slot):
                    chosen, chosen_slot = extra, slot_ns
            result.append(chosen)
        return result

    def _pair_features(self, origin, mask_candidate, gong_row, *, gong_scale=1.0, mask_roll=0):
        values = np.full(len(FEATURE_COLUMNS), np.nan, dtype=np.float64)
        values[12] = 0.0
        values[MISSING_INDEX] = 1.0
        if mask_candidate is None or gong_row is None:
            return values, {"missing": True,
                            "mask": None if mask_candidate is None else str(mask_candidate["filename"]),
                            "gong": None if gong_row is None else str(gong_row["file"])}
        ch_row = int(mask_candidate["ch_row"])
        area_grid = np.asarray(self.mask_context.area_grid[ch_row], dtype=np.float64).reshape(3, 12)
        mask_sha = str(mask_candidate["local_sha256"])
        gong_data = dict(self.gong.field_for(gong_row, scale=float(gong_scale)))
        gong_data["issue_ns"] = int(gong_row["issue_ns"])
        gong_data["available_ns"] = int(gong_row["available_ns"])
        ch_mask = {
            "area_grid": area_grid,
            "obs_end_utc": mask_candidate["obs_end"],
            "slot_utc": mask_candidate["slot"],
            "s3_modified_utc": mask_candidate["s3_modified"],
        }
        if mask_roll:
            ch_mask["area_grid"] = np.roll(area_grid, int(mask_roll), axis=1)
        gong_sha = str(gong_row["sha256"])
        mask_obs_ns = _ns(mask_candidate["obs_end"])
        mask_available_ns = max(mask_obs_ns + HOUR_NS, _ns(mask_candidate["s3_modified"]))
        key_payload = {
            "version": 2,
            "extractor_sha256": self.helper_hashes[str(Path(__file__).resolve())],
            "helper_hashes": self.helper_hashes,
            "mask_source_sha256": mask_sha,
            "mask_area_grid_sha256": hashlib.sha256(np.ascontiguousarray(area_grid).view(np.uint8)).hexdigest(),
            "mask_grid_input_sha256": self.mask_context.data_hashes[str(morph_spec.CH_PATH)],
            "gong_sha256": gong_sha,
            "mask_obs_end_ns": mask_obs_ns,
            "mask_available_ns": mask_available_ns,
            "gong_issue_ns": int(gong_row["issue_ns"]),
            "gong_available_ns": int(gong_row["available_ns"]),
            "config": CONFIG,
        }
        key = hashlib.sha256(json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if gong_scale == 1.0 and mask_roll == 0 and key in self._memory:
            values = self._memory[key].copy()
            self.cache_hits += 1
        else:
            cache_path = PAIR_CACHE / f"{key}.json"
            cached = None
            if gong_scale == 1.0 and mask_roll == 0 and cache_path.is_file():
                cached = json.loads(cache_path.read_text())
                if cached.get("key") != key or cached.get("key_payload") != key_payload:
                    raise ValueError(f"stale CH-UNIP pair cache: {cache_path}")
            if cached is not None:
                values = np.asarray([np.nan if value is None else float(value) for value in cached["features"]])
                self._memory[key] = values.copy()
                self.cache_hits += 1
            else:
                values = build_ch_unip_features(origin, ch_mask, gong_data, CONFIG)
                if gong_scale == 1.0 and mask_roll == 0:
                    payload = {"key": key, "key_payload": key_payload,
                               "features": [float(value) if np.isfinite(value) else None for value in values]}
                    temp = cache_path.with_suffix(".tmp")
                    temp.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
                    temp.replace(cache_path)
                    self._memory[key] = values.copy()
                    self.cache_created += 1
        mask_available = pd.Timestamp(mask_available_ns, tz="UTC")
        gong_available = pd.Timestamp(int(gong_row["available_ns"]), tz="UTC")
        summary = {
            "missing": bool(values[MISSING_INDEX] == 1.0),
            "mask": str(mask_candidate["filename"]), "mask_sha256": mask_sha,
            "mask_slot_utc": str(mask_candidate["slot"]), "mask_obs_end_utc": str(mask_candidate["obs_end"]),
            "mask_last_modified_utc": str(mask_candidate["s3_modified"]),
            "mask_available_at_utc": str(mask_available),
            "gong": str(gong_row["file"]), "gong_sha256": gong_sha,
            "gong_issue_time_utc": str(gong_row["issue_time_utc"]),
            "gong_available_at_utc": str(gong_available),
            "pair_cache_key": key,
            "valid_overlap_fraction": float(values[12]) if np.isfinite(values[12]) else 0.0,
            "finite_gong_pixels": int(np.count_nonzero(np.isfinite(gong_data["field"]))),
        }
        aligned = _aligned_cell_data(ch_mask, gong_data)
        summary["valid_overlap_cells"] = int(aligned["valid"].sum())
        summary["valid_overlap_area"] = float(aligned["cell_area"][aligned["valid"]].sum())
        return values, summary

    def features_for(self, origins, overrides=None):
        origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
        overrides = {} if overrides is None else overrides
        mask_rows = self._selected_mask_rows(origins, overrides.get("future_suvi_rows", ()))
        gong_rows = self.gong.select_many(origins, extra_rows=overrides.get("future_gong_rows", ()))
        result, summaries = [], []
        for origin, mask_row, gong_row in zip(origins, mask_rows, gong_rows):
            values, summary = self._pair_features(
                origin, mask_row, gong_row,
                gong_scale=float(overrides.get("gong_scale", 1.0)),
                mask_roll=int(overrides.get("mask_roll", 0)),
            )
            result.append(values)
            summaries.append(summary)
        self._last_summaries = summaries
        return np.vstack(result) if result else np.empty((0, len(FEATURE_COLUMNS)))

    def poison_overrides(self, origin):
        origin = pd.Timestamp(origin)
        origin = origin.tz_localize("UTC") if origin.tz is None else origin.tz_convert("UTC")
        mask_rows = self._selected_mask_rows(pd.DatetimeIndex([origin]))
        gong_rows = self.gong.select_many([origin])
        if mask_rows[0] is None or gong_rows[0] is None:
            raise ValueError("no admissible CH mask/GONG pair at poison probe")
        future_suvi = {"slot": origin - pd.Timedelta(hours=1), "obs_end": origin + pd.Timedelta(hours=1),
                       "s3_modified": origin + pd.Timedelta(hours=1), "filename": "future-poison-suvi",
                       "local_path": "/not/read/future-poison.nc"}
        gong = dict(gong_rows[0])
        future_gong = dict(gong)
        future_gong.update({"file": "future-poison-gong", "slot_ns": origin.value - HOUR_NS,
                            "issue_ns": origin.value - HOUR_NS, "available_ns": origin.value + HOUR_NS,
                            "path": "/not/read/future-poison.fits.gz", "sha256": "future-poison"})
        return ({"future_suvi_rows": [future_suvi], "future_gong_rows": [future_gong]},
                {"mask_roll": 1}, {"gong_scale": 1.25}, mask_rows[0], gong_rows[0])

    def official_report(self, origins):
        values = self.features_for(origins)
        summaries = list(self._last_summaries)
        stats = {}
        for index, column in enumerate(FEATURE_COLUMNS):
            finite = np.isfinite(values[:, index])
            stats[column] = {
                "finite_fraction": float(finite.mean()) if len(finite) else 0.0,
                "variance": float(np.var(values[finite, index])) if finite.any() else None,
            }
        overlap = np.asarray([row.get("valid_overlap_fraction", 0.0) for row in summaries], dtype=float)
        mask_rows = {row["mask_sha256"]: {k: row[k] for k in ("mask_sha256", "mask_slot_utc", "mask_obs_end_utc", "mask_last_modified_utc", "mask_available_at_utc")}
                     for row in summaries if row.get("mask_sha256")}
        gong_rows = {row["gong_sha256"]: {k: row[k] for k in ("gong_sha256", "gong", "gong_issue_time_utc", "gong_available_at_utc")}
                     for row in summaries if row.get("gong_sha256")}
        return {
            "origins": len(origins),
            "mask_source_coverage": sum(bool(row.get("mask_sha256")) for row in summaries),
            "gong_source_coverage": sum(bool(row.get("gong_sha256")) for row in summaries),
            "joint_pair_coverage": sum(bool(row.get("mask_sha256") and row.get("gong_sha256")) for row in summaries),
            "feature_complete_coverage": int(np.count_nonzero(values[:, MISSING_INDEX] == 0.0)),
            "missing_fraction": float(np.mean(values[:, MISSING_INDEX] == 1.0)) if len(values) else 1.0,
            "valid_overlap_fraction": {
                "mean": float(np.mean(overlap)) if len(overlap) else None,
                "median": float(np.median(overlap)) if len(overlap) else None,
                "min": float(np.min(overlap)) if len(overlap) else None,
                "max": float(np.max(overlap)) if len(overlap) else None,
                "cells_total": int(sum(row.get("valid_overlap_cells", 0) for row in summaries)),
                "area_total": float(sum(row.get("valid_overlap_area", 0.0) for row in summaries)),
            },
            "spatial_resolution": "existing t035 12x3 area grid; nearest GONG CEA field sampled at occupied bin centers",
            "features": stats,
            "unique_mask_sources": list(mask_rows.values()),
            "unique_gong_sources": list(gong_rows.values()),
            "cache": {"path": str(PAIR_CACHE), "hits": self.cache_hits, "created": self.cache_created,
                      "key_fields": ["feature/helper code hashes", "mask hash+observation/availability", "GONG hash+issue/availability", "config"]},
            "mask_selector_receipt": self.mask_context.source_receipt,
            "mask_input_hashes": self.mask_context.data_hashes,
            "gong_manifest_sha256": self.manifest_sha256,
            "mask_availability_rule": CONFIG["mask_selector"],
            "gong_availability_rule": CONFIG["gong_selector"],
            "source_pairs": summaries,
        }
