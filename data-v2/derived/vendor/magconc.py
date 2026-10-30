"""Causal CH-local magnetic concentration features on the existing SUVI mask."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.solar_inc_chmorph72_20261014 import morph_features, spec as morph_spec
import cmechip as chip

LB = Path('/home/t-lab01/sunrun/leaderboard')
STORE_ROOT = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store')

FEATURE_CACHE = STORE_ROOT / "v2/magconc/cache/pairs"
FEATURE_COLUMNS = (
    "ch_magconc_top1_flux_frac",
    "ch_magconc_top5_flux_frac",
    "ch_magconc_top10_flux_frac",
    "ch_magconc_br_abs_p90",
    "ch_magconc_br_abs_p95",
    "ch_magconc_br_abs_p99",
    "ch_magconc_strong_majority_area_frac",
    "ch_magconc_strong_minority_area_frac",
    "ch_magconc_flux_gini",
    "ch_magconc_strongfield_signed_ratio",
    "ch_magconc_valid_overlap_frac",
    "ch_magconc_missing",
)
CONFIG = {
    "strong_threshold_gauss": 20.0,
    "ch_mask": "existing source-verified SUVI Fe195 1-degree Stonyhurst mask",
    "ch_mask_latency": "obs_end + 1 h <= origin and S3 LastModified <= origin",
    "gong_alignment": "existing CRLN/CRLT-CEA WCS; synodic rotation at SUVI obs_end",
    "cell_sampling": "nearest GONG CEA grid point at each occupied 1-degree CH cell center",
    "area_weight": "existing SUVI grid_area fraction per heliographic cell",
    "minimum_valid_overlap_fraction": 0.0,
}
MISSING_INDEX = len(FEATURE_COLUMNS) - 1
HOUR_NS = 3_600_000_000_000
DAY_NS = 24 * HOUR_NS
SYNODIC_RATE_DEG_DAY = 360.0 / 27.2753


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ns(value) -> int:
    timestamp = pd.Timestamp(value)
    timestamp = timestamp.tz_localize("UTC") if timestamp.tz is None else timestamp.tz_convert("UTC")
    return int(timestamp.value)


def _weighted_quantile(values, weights, quantile):
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    threshold = float(quantile) * float(weights.sum())
    index = min(int(np.searchsorted(np.cumsum(weights), threshold, side="left")), len(values) - 1)
    return float(values[index])


def _gini(values):
    values = np.sort(np.asarray(values, dtype=np.float64))
    total = float(values.sum())
    if total <= 0 or len(values) <= 1:
        return 0.0
    rank_sum = float(np.dot(np.arange(1, len(values) + 1, dtype=np.float64), values))
    return float(2.0 * rank_sum / (len(values) * total) - (len(values) + 1.0) / len(values))


def build_ch_magconc_features(origin, ch_mask, gong_map, config):
    """Return 12 CH-local strong-field concentration values for one origin."""
    out = np.full(len(FEATURE_COLUMNS), np.nan, dtype=np.float64)
    out[MISSING_INDEX] = 1.0
    labels = np.asarray(ch_mask["labels"])
    area_grid = np.asarray(ch_mask["grid_area"], dtype=np.float64)
    field = np.asarray(gong_map["field"], dtype=np.float64)
    rows, cols = np.nonzero(labels)
    if not len(rows):
        return out

    # The existing binary mask is already on a 1-degree Stonyhurst grid.
    # Reproject only its cell centers through the GONG CEA WCS; no new CH
    # segmentation or connected-component calculation is performed here.
    ny, nx = field.shape
    lat = rows.astype(np.float64) - 89.5
    lon_stony = cols.astype(np.float64) - 179.5
    rotation = SYNODIC_RATE_DEG_DAY * (
        (_ns(ch_mask["obs_end_utc"]) - int(gong_map["issue_ns"])) / DAY_NS
    )
    lon_carrington = float(gong_map["crval1"]) + rotation + lon_stony
    x = np.rint(float(gong_map["crpix1"]) - 1.0
                + (lon_carrington - float(gong_map["crval1"])) / float(gong_map["cdelt1"])).astype(np.int64) % nx
    sin_lat = np.sin(np.deg2rad(lat))
    y = np.rint(float(gong_map["crpix2"]) - 1.0
                + (sin_lat - float(gong_map["crval2"])) / float(gong_map["cdelt2"])).astype(np.int64)
    cell_area = area_grid[rows, cols]
    in_bounds = (y >= 0) & (y < ny) & np.isfinite(cell_area) & (cell_area > 0)
    safe_y = y.clip(0, ny - 1)
    br = field[safe_y, x]
    valid = in_bounds & np.isfinite(br)
    total_ch_area = float(cell_area[np.isfinite(cell_area) & (cell_area > 0)].sum())
    valid_area = float(cell_area[valid].sum())
    overlap = valid_area / total_ch_area if total_ch_area > 0 else 0.0
    if (valid_area <= 0 or overlap < float(config["minimum_valid_overlap_fraction"])):
        return out

    br = br[valid]
    weights = cell_area[valid]
    absolute = np.abs(br)
    flux = absolute * weights
    total_flux = float(flux.sum())
    if not math.isfinite(total_flux) or total_flux <= 0:
        return out

    ordered_flux = np.sort(flux)[::-1]
    out[:3] = [float(ordered_flux[:k].sum() / total_flux) for k in (1, 5, 10)]
    out[3:6] = [_weighted_quantile(absolute, weights, q) for q in (0.90, 0.95, 0.99)]
    out[8] = _gini(flux)
    threshold = float(config["strong_threshold_gauss"])
    strong = absolute >= threshold
    strong_signed = br[strong] * weights[strong]
    strong_flux = float(np.abs(strong_signed).sum())
    if strong_flux > 0:
        net = float(strong_signed.sum())
        majority_sign = 1.0 if net > 0 else -1.0 if net < 0 else float(np.sign(br[strong][np.argmax(np.abs(strong_signed))]))
        strong_area = weights[strong]
        majority_area = float(strong_area[br[strong] * majority_sign > 0].sum())
        minority_area = float(strong_area[br[strong] * majority_sign < 0].sum())
        out[6] = majority_area / valid_area
        out[7] = minority_area / valid_area
        out[9] = net / strong_flux
    else:
        out[6:8] = 0.0
        out[9] = 0.0
    out[10] = overlap
    out[11] = 0.0
    return out


class CHMagConcCatalog:
    """Join the verified existing SUVI selector to E1198's GONG map vintage."""

    def __init__(self, cap, gong_manifest_path, expected_gong_manifest_sha256):
        self.cap = pd.Timestamp(cap)
        # Keep selector-derived geometry and provenance receipts inside this
        # queue item's cache area; the other experiment's files stay read-only.
        morph_spec.STORE = STORE_ROOT / "v2/magconc/cache/ch-selector"
        self.mask_context = morph_features.MorphologyContext(self.cap)
        self.suvi_masks = chip.SuviCHCatalog(self.cap)
        self.gong = chip.GongFieldCatalog(
            gong_manifest_path, cap=self.cap, cache_root=STORE_ROOT / "v2/cmechip/cache"
        )
        self.gong_manifest_sha256 = self.gong.manifest_sha256
        if self.gong_manifest_sha256 != expected_gong_manifest_sha256:
            raise ValueError(f"E1198 GONG manifest hash changed: {self.gong_manifest_sha256}")
        self.extractor_sha256 = _sha(Path(__file__).resolve())
        # Record the exact morphology selector and spec implementations too.
        self.helper_hashes = {
            str(Path(__file__).resolve()): self.extractor_sha256,
            str(Path(chip.__file__).resolve()): _sha(Path(chip.__file__).resolve()),
            str(Path(morph_features.__file__).resolve()): _sha(Path(morph_features.__file__).resolve()),
            str(Path(morph_spec.__file__).resolve()): _sha(Path(morph_spec.__file__).resolve()),
        }
        self.cache_dir = FEATURE_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_hits = 0
        self.cache_created = 0
        self._memory = {}

    def _mask_row(self, candidate):
        return {"file": str(candidate["filename"]), "path": str(candidate["local_path"])}

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
                if (slot_ns <= origin_ns and _ns(extra["obs_end"]) <= origin_ns - HOUR_NS
                        and _ns(extra["s3_modified"]) <= origin_ns and slot_ns > chosen_slot):
                    chosen, chosen_slot = extra, slot_ns
            result.append(chosen)
        return result

    def _features_one(self, origin, mask_candidate, gong_row, *, gong_scale=1.0):
        if mask_candidate is None or gong_row is None:
            missing = np.full(len(FEATURE_COLUMNS), np.nan, dtype=np.float64)
            missing[MISSING_INDEX] = 1.0
            return missing, {"suvi": None if mask_candidate is None else mask_candidate["filename"],
                             "gong": None if gong_row is None else gong_row["file"], "missing": True}
        mask_row = self._mask_row(mask_candidate)
        mask_data = self.suvi_masks.mask_for(mask_row)
        gong_data = self.gong.field_for(gong_row, scale=float(gong_scale))
        source_sha = str(mask_data["source_sha256"].item())
        gong_sha = str(gong_row["sha256"])
        key_payload = {
            "version": 1,
            "extractor_sha256": self.extractor_sha256,
            "helpers": self.helper_hashes,
            "suvi_sha256": source_sha,
            "gong_sha256": gong_sha,
            "suvi_obs_end_ns": _ns(mask_candidate["obs_end"]),
            "gong_issue_ns": int(gong_row["issue_ns"]),
            "config": CONFIG,
            "gong_scale": float(gong_scale),
        }
        key = hashlib.sha256(json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        cache_path = self.cache_dir / f"{key}.json"
        if float(gong_scale) == 1.0 and key in self._memory:
            self.cache_hits += 1
            values = self._memory[key].copy()
        elif float(gong_scale) == 1.0 and cache_path.is_file():
            cached = json.loads(cache_path.read_text())
            if cached.get("key_payload") != key_payload or cached.get("key") != key:
                raise ValueError(f"stale CH/GONG concentration cache: {cache_path}")
            values = np.asarray([np.nan if value is None else float(value) for value in cached["features"]])
            self._memory[key] = values.copy()
            self.cache_hits += 1
        else:
            ch_mask = {**mask_data, "obs_end_utc": mask_candidate["obs_end"]}
            values = build_ch_magconc_features(origin, ch_mask, gong_data, CONFIG)
            if float(gong_scale) == 1.0:
                cached = {"key": key, "key_payload": key_payload,
                          "features": [float(value) if np.isfinite(value) else None for value in values]}
                tmp = cache_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(cached, sort_keys=True, allow_nan=False) + "\n")
                tmp.replace(cache_path)
                self._memory[key] = values.copy()
                self.cache_created += 1
        summary = {"suvi": str(mask_candidate["filename"]), "suvi_sha256": source_sha,
                   "suvi_slot": str(mask_candidate["slot"]), "suvi_obs_end": str(mask_candidate["obs_end"]),
                   "suvi_last_modified": str(mask_candidate["s3_modified"]), "gong": gong_row["file"],
                   "gong_sha256": gong_sha, "gong_issue": str(gong_row["issue_time_utc"]),
                   "cache_key": key, "magconc_missing": float(values[MISSING_INDEX])}
        return values, summary

    def features_for(self, origins, overrides=None):
        origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
        overrides = overrides or {}
        suvi_rows = self._selected_mask_rows(origins, overrides.get("future_suvi_rows", ()))
        gong_rows = self.gong.select_many(origins, extra_rows=overrides.get("future_gong_rows", ()))
        output = []
        summaries = []
        scale = float(overrides.get("gong_scale", 1.0))
        for origin, suvi_row, gong_row in zip(origins, suvi_rows, gong_rows):
            values, summary = self._features_one(origin, suvi_row, gong_row, gong_scale=scale)
            output.append(values)
            summaries.append(summary)
        if overrides:
            self.last_override_summaries = summaries
        else:
            for origin, summary in zip(origins, summaries):
                self._memory_summary = getattr(self, "_memory_summary", {})
                self._memory_summary[int(origin.value)] = summary
        return np.vstack(output) if output else np.empty((0, len(FEATURE_COLUMNS)))

    def poison_overrides(self, origin):
        origin = pd.Timestamp(origin)
        origin = origin.tz_localize("UTC") if origin.tz is None else origin.tz_convert("UTC")
        suvi_rows = self._selected_mask_rows(pd.DatetimeIndex([origin]))
        gong_rows = self.gong.select_many([origin])
        if suvi_rows[0] is None or gong_rows[0] is None:
            raise ValueError("no admissible CH mask and E1198 GONG map for magconc poison probe")
        future_suvi = {"slot": origin - pd.Timedelta(hours=1), "obs_end": origin + pd.Timedelta(hours=1),
                       "s3_modified": origin + pd.Timedelta(hours=1), "filename": "future-poison-suvi",
                       "local_path": "/not/read/future-poison.nc"}
        gong = gong_rows[0]
        future_gong = dict(gong)
        future_gong.update({"file": "future-poison-gong", "slot_ns": origin.value - HOUR_NS,
                            "issue_ns": origin.value - HOUR_NS, "available_ns": origin.value + HOUR_NS,
                            "path": "/not/read/future-poison.fits.gz", "sha256": "future-poison"})
        return ({"future_suvi_rows": [future_suvi], "future_gong_rows": [future_gong]},
                {"gong_scale": 1.25}, suvi_rows[0], gong_rows[0])

    def official_report(self, origins):
        origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
        selected_suvi = self._selected_mask_rows(origins)
        selected_gong = self.gong.select_many(origins)
        values = self.features_for(origins)
        both = np.asarray([s is not None and g is not None for s, g in zip(selected_suvi, selected_gong)])
        ages_suvi = np.asarray([_ns(origin) - _ns(row["obs_end"]) for origin, row in zip(origins, selected_suvi)
                                if row is not None], dtype=np.float64) / HOUR_NS
        ages_gong = np.asarray([_ns(origin) - int(row["issue_ns"]) for origin, row in zip(origins, selected_gong)
                                if row is not None], dtype=np.float64) / HOUR_NS
        distributions = {}
        for i, name in enumerate(FEATURE_COLUMNS):
            finite = values[:, i][np.isfinite(values[:, i])]
            distributions[name] = {"finite": int(len(finite)),
                                   "min": float(np.min(finite)) if len(finite) else None,
                                   "median": float(np.median(finite)) if len(finite) else None,
                                   "p95": float(np.percentile(finite, 95)) if len(finite) else None,
                                   "max": float(np.max(finite)) if len(finite) else None}
        return {
            "columns": list(FEATURE_COLUMNS), "config": CONFIG,
            "threshold_fixed_before_official_truth": True,
            "source_policy": ("source-catalog byte identity (S3 ETag == local MD5 and S3/local sizes equal); "
                              "select latest SUVI slot with obs_end+1h<=origin and S3 LastModified<=origin; "
                              "GONG selected by E1198 manifest availability_at<=origin"),
            "suvi_source_receipt": self.mask_context.source_receipt,
            "suvi_mask_manifest_sha256": self.mask_context.source_manifest["catalog_sha256"],
            "gong_manifest_sha256": self.gong_manifest_sha256,
            "input_hashes": self.mask_context.data_hashes,
            "origins": len(origins), "suvi_mask_coverage": int(sum(row is not None for row in selected_suvi)),
            "gong_map_coverage": int(sum(row is not None for row in selected_gong)),
            "joint_coverage": int(both.sum()), "unique_suvi_masks": len({row["filename"] for row in selected_suvi if row is not None}),
            "unique_gong_maps": len({row["file"] for row in selected_gong if row is not None}),
            "suvi_obs_end_age_hours_median_max": [float(np.median(ages_suvi)), float(np.max(ages_suvi))] if len(ages_suvi) else [],
            "gong_issue_age_hours_median_max": [float(np.median(ages_gong)), float(np.max(ages_gong))] if len(ages_gong) else [],
            "distributions": distributions,
            "cache": {"hits": self.cache_hits, "created": self.cache_created,
                      "path": str(self.cache_dir), "key_includes": ["code", "SUVI bytes", "GONG bytes", "mask/map timestamps", "config"]},
        }
