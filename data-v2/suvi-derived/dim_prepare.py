"""Build a provenance-checked, hash-keyed SUVI dimming feature cache."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from dim_features import (CONFIG, FEATURES, FrameCatalog, FrameProductCache, _median_with_min_count, _origin_arrays,
                          current_reference_groups, extract_carrington_map)

ROOT = Path("/home/t-lab01/sunrun")
LB = ROOT / "leaderboard"
STORE_ROOT = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store')
HERE = Path(__file__).resolve().parent
CUTOFF = pd.Timestamp("2026-05-29", tz="UTC")
S3_HOST = "noaa-goes18.s3.amazonaws.com"
S3_ROOT = f"https://{S3_HOST}"
STAGE_ROOT = Path("/home/t-lab01/.local/state/sundb/stage/suvi_hourly")
CACHE_ROOT = STORE_ROOT / "v2/suvi-dim"
POINTER = CACHE_ROOT / "solar-inc-suvidim72-20261014-pointer.json"
MAP_CACHE_ROOT = Path("/home/t-lab01/.local/state/research-dispatch/runs/solar/solar-inc-suvidim72-20261014-20261014-031438/cache/frame-maps")
PRODUCT_CACHE_ROOT = MAP_CACHE_ROOT.parent / "frame-products"
N_THREADS = 4
NS_H = int(3600 * 1e9)
LEGACY_FRAME_MAP_CODE_HASHES = (
    "bf8990538ec52ae463f03da712046742a349bd6f81fbd35556e51ec151a53b25",
    "1b9875d459acab7a0dd38fd3735f02e5d484c1b7a35819cb2b59e9eaf4f0a786",
)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_hash():
    h = hashlib.sha256()
    for path in (HERE / "config.json", HERE / "dim_features.py", HERE / "dim_prepare.py"):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def load_origins():
    raise RuntimeError("The v2 wrapper must supply an explicit hourly origin grid")


def local_path(value):
    if value is None or pd.isna(value):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return ROOT / "sun-img" / path


def stage_catalog(min_origin, max_origin):
    paths = sorted(STAGE_ROOT.glob("year=*/part.parquet"))
    if not paths:
        raise FileNotFoundError(f"SUVI hourly source metadata missing: {STAGE_ROOT}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame = frame.loc[(frame["sat"].astype(str).str.upper() == "G18") &
                      (frame["channel"].astype(str).str.lower() == "fe195")].copy()
    frame["obs_start"] = pd.to_datetime(frame["obs_start"], utc=True, errors="coerce")
    frame["obs_end"] = pd.to_datetime(frame["obs_end"], utc=True, errors="coerce")
    low = min_origin - pd.Timedelta(hours=45)
    high = max_origin + pd.Timedelta(hours=6)
    frame = frame.loc[frame["obs_end"].between(low, high, inclusive="both")].copy()
    frame = frame.loc[frame["obs_end"].notna() & frame["s3_key"].notna()].copy()
    frame["nc_key"] = frame["s3_key"].astype(str).str.replace(r"\.fits$", ".nc", regex=True)
    frame["frame_id"] = frame["nc_key"].map(lambda key: hashlib.sha256(key.encode()).hexdigest()[:24])
    frame["local_file"] = frame["nc_path"].map(local_path)
    frame = frame.drop_duplicates("nc_key").sort_values(["obs_end", "nc_key"]).reset_index(drop=True)
    return frame


def list_day(day):
    prefix = f"SUVI-L1b-Fe195/{day:%Y}/{day.timetuple().tm_yday:03d}/"
    token = None
    objects = {}
    for attempt in range(5):
        try:
            while True:
                query = {"list-type": "2", "prefix": prefix}
                if token:
                    query["continuation-token"] = token
                url = f"{S3_ROOT}/?{urllib.parse.urlencode(query)}"
                request = urllib.request.Request(url, headers={"User-Agent": "solar-inc-suvidim72/1.0"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    body = response.read()
                root = ET.fromstring(body)
                namespace = "{http://s3.amazonaws.com/doc/2006-03-01/}"
                for item in root.findall(f"{namespace}Contents"):
                    key = item.findtext(f"{namespace}Key")
                    if not key or not key.endswith(".nc"):
                        continue
                    modified = item.findtext(f"{namespace}LastModified")
                    etag = (item.findtext(f"{namespace}ETag") or "").strip().strip('"')
                    size = int(item.findtext(f"{namespace}Size") or 0)
                    objects[key] = {"etag": etag, "size": size,
                                    "last_modified": datetime.fromisoformat(modified.replace("Z", "+00:00"))}
                if root.findtext(f"{namespace}IsTruncated") != "true":
                    return objects
                token = root.findtext(f"{namespace}NextContinuationToken")
                if not token:
                    raise ValueError(f"S3 listing has no continuation token for {prefix}")
        except Exception:
            if attempt == 4:
                raise
            time.sleep(1.5 ** attempt)
    return objects


def remote_catalog(frame):
    days = sorted(set(pd.Timestamp(value).date() for value in frame["obs_end"]))
    remote = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        for day_objects in pool.map(list_day, days):
            remote.update(day_objects)
    records = []
    for row in frame.to_dict("records"):
        obj = remote.get(row["nc_key"])
        obs_start = row["obs_start"]
        obs_end = row["obs_end"]
        path = row["local_file"]
        records.append({
            "frame_id": row["frame_id"],
            "source_url": f"{S3_ROOT}/{row['nc_key']}",
            "s3_key": row["nc_key"],
            "obs_start": obs_start.isoformat() if pd.notna(obs_start) else None,
            "obs_end": obs_end.isoformat(),
            "obs_end_ns": int(obs_end.value),
            "stage_fits_last_modified": (pd.Timestamp(row["s3_modified"]).tz_localize("UTC").isoformat()
                if pd.Timestamp(row["s3_modified"]).tzinfo is None else pd.Timestamp(row["s3_modified"]).tz_convert("UTC").isoformat()),
            "last_modified": obj["last_modified"].isoformat() if obj else None,
            "last_modified_ns": int(obj["last_modified"].timestamp() * 1e9) if obj else 2**63 - 1,
            "remote_etag": obj["etag"] if obj else None,
            "remote_size": obj["size"] if obj else None,
            "local_path": str(path) if path else None,
            "stage_local_size": int(row["nc_size"]) if pd.notna(row["nc_size"]) else None,
            "byte_equal": False,
            "source_usable": False,
            "source_unusable_reason": None,
            "map_reused_from_previous_code": False,
            "local_bytes": None,
            "local_md5": None,
            "local_sha256": None,
            "map_path": None,
            "map_sha256": None,
            "map_error": None,
        })
    return records


def process_frame(record, image_code_hash):
    result = dict(record)
    path = Path(record["local_path"]) if record.get("local_path") else None
    if not path or not path.is_file() or not record.get("remote_etag"):
        result["map_error"] = "local file or remote object metadata missing"
        return result
    blob = path.read_bytes()
    md5 = hashlib.md5(blob).hexdigest()
    sha256 = sha_bytes(blob)
    etag = str(record["remote_etag"])
    byte_equal = ("-" not in etag and len(etag) == 32 and md5 == etag and
                  len(blob) == int(record["remote_size"]))
    result.update({"local_bytes": len(blob), "local_md5": md5, "local_sha256": sha256,
                   "byte_equal": bool(byte_equal)})
    if not byte_equal:
        result["map_error"] = "local bytes do not match single-part S3 ETag and size"
        return result
    map_key = hashlib.sha256((image_code_hash + record["frame_id"] + sha256).encode()).hexdigest()
    map_path = MAP_CACHE_ROOT / f"{map_key}.npz"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    if not map_path.is_file():
        legacy_paths = [MAP_CACHE_ROOT / (
            hashlib.sha256((legacy_hash + record["frame_id"] + sha256).encode()).hexdigest() + ".npz")
            for legacy_hash in LEGACY_FRAME_MAP_CODE_HASHES]
        legacy_path = next((path for path in legacy_paths if path.is_file()), None)
        if legacy_path is not None:
            try:
                os.link(legacy_path, map_path)
            except OSError:
                shutil.copy2(legacy_path, map_path)
            result["map_reused_from_previous_code"] = True
        else:
            try:
                image = extract_carrington_map(blob, record["obs_end"])
            except ValueError as error:
                if "RSUN sentinel -999" not in str(error):
                    raise
                result["source_unusable_reason"] = "RSUN sentinel -999; source geometry unavailable"
                result["map_error"] = result["source_unusable_reason"]
                return result
            temp = map_path.with_suffix(f".tmp.{os.getpid()}.npz")
            np.savez_compressed(temp, radiance=image)
            temp.replace(map_path)
    result["map_path"] = str(map_path)
    result["map_sha256"] = sha_file(map_path)
    result["source_usable"] = True
    result["map_error"] = None
    return result


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _digest_ids(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def _get_probe_maps(records):
    paths = {r["frame_id"]: r["map_path"] for r in records if r.get("map_path")}

    @lru_cache(maxsize=128)
    def get_map(frame_id):
        path = paths.get(frame_id)
        if not path:
            return np.full((180, 360), np.nan, dtype=np.float32)
        with np.load(path) as saved:
            return saved["radiance"].copy()

    return get_map


def make_probes(origin, catalog, get_map, original_features, original_ids):
    origin_ns = np.datetime64(origin.tz_localize(None), "ns").astype(np.int64)
    future_records = [r for r in catalog if (r["obs_end_ns"] + NS_H > origin_ns) or
                      (r["last_modified_ns"] > origin_ns)]
    future_ids = {r["frame_id"] for r in future_records}
    hostile_catalog = []
    for row in catalog:
        changed = dict(row)
        if changed["obs_end_ns"] + NS_H > origin_ns:
            changed["obs_end_ns_ns_original"] = changed["obs_end_ns"]
            changed["obs_end_ns"] += int(120 * 86400 * 1e9)
        if changed["last_modified_ns"] > origin_ns:
            changed["last_modified_ns"] = origin_ns + int(120 * 86400 * 1e9)
        hostile_catalog.append(changed)

    def poisoned_map(frame_id):
        if frame_id in future_ids:
            return np.full((180, 360), 1e12, dtype=np.float32)
        return get_map(frame_id)

    hostile, hostile_ids, hostile_stats = _origin_arrays(origin, FrameCatalog(hostile_catalog), poisoned_map)
    assert hostile_ids == original_ids, "future pixel/metadata poison changed selected frame IDs"
    assert all(hostile[k] == original_features[k] for k in FEATURES), "future poison changed causal dimming inputs"

    groups = current_reference_groups(origin, catalog)
    chosen = None
    for (i0, refs0), (i1, refs1) in reversed(list(zip(groups, groups[1:]))):
        if (catalog[i1]["obs_end_ns"] - catalog[i0]["obs_end_ns"]) > int(1.5 * 3600 * 1e9):
            continue
        current0, current1 = get_map(catalog[i0]["frame_id"]), get_map(catalog[i1]["frame_id"])
        ref0, count0 = _median_with_min_count(np.stack([get_map(catalog[j]["frame_id"]) for j in refs0]), CONFIG["minimum_reference_frames"])
        ref1, count1 = _median_with_min_count(np.stack([get_map(catalog[j]["frame_id"]) for j in refs1]), CONFIG["minimum_reference_frames"])
        ref0[count0 < CONFIG["minimum_reference_frames"]] = np.nan
        ref1[count1 < CONFIG["minimum_reference_frames"]] = np.nan
        common = np.isfinite(current0) & np.isfinite(current1) & np.isfinite(ref0) & np.isfinite(ref1) & (ref0 > 0) & (ref1 > 0)
        common[:50, :] = False
        common[130:, :] = False
        patch_centers = np.argwhere(ndimage.binary_erosion(common, structure=np.ones((5, 5), dtype=bool)))
        patch_centers = [xy for xy in patch_centers if 3 <= xy[1] < 357]
        if len(patch_centers):
            chosen = (i0, refs0, i1, refs1, current0, current1, ref0, ref1, patch_centers[0])
            break
    if chosen is None:
        raise RuntimeError("no co-visible current/reference image pair for positive-control poison")
    i0, refs0, i1, refs1, current0, current1, ref0, ref1, (row, col) = chosen
    rs, cs = slice(int(row) - 2, int(row) + 3), slice(int(col) - 2, int(col) + 3)
    overrides = {}
    for index, reference in ((i0, ref0), (i1, ref1)):
        frame_id = catalog[index]["frame_id"]
        modified = get_map(frame_id).copy()
        modified[rs, cs] = np.maximum(reference[rs, cs] * 0.05, 0.0)
        overrides[frame_id] = modified

    def positive_map(frame_id):
        return overrides.get(frame_id, get_map(frame_id))

    positive, positive_ids, positive_stats = _origin_arrays(origin, catalog, positive_map)
    assert positive_ids == original_ids, "positive-control pixels changed selected source frame IDs"
    assert any(abs(positive[k] - original_features[k]) > 1e-12 for k in FEATURES if k != "dim_missing"), \
        "eligible source-pixel positive control did not change the dimming inputs"
    assert positive["dim_area_frac"] > 0.0, "positive-control dimming area did not become active"

    return {
        "origin": origin.isoformat(),
        "selected_ids_sha256": _digest_ids(original_ids),
        "selected_ids_count": len(original_ids),
        "future_poison_ids_sha256": _digest_ids(hostile_ids),
        "future_poison_ids_count": len(hostile_ids),
        "future_poison_pixels_changed": len(future_ids),
        "future_poison_metadata_changed": len(future_records),
        "future_poison_inputs_identical": True,
        "future_poison_stats": hostile_stats,
        "positive_control_frame_ids": [catalog[i0]["frame_id"], catalog[i1]["frame_id"]],
        "positive_control_patch_rows_cols": [int(row), int(col), 5],
        "positive_control_source_pixels_changed": True,
        "positive_control_selected_ids_identical": True,
        "positive_control_inputs_changed": True,
        "positive_control_stats": positive_stats,
        "base_features": [float(original_features[name]) for name in FEATURES],
        "future_features": [float(hostile[name]) for name in FEATURES],
        "positive_features": [float(positive[name]) for name in FEATURES],
    }


def local_store_latest_obs():
    pattern = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)_e(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)_c")
    latest = None
    for path in (ROOT / "sun-img/data/g18").glob("20*/*/*Fe195*.nc"):
        match = pattern.search(path.name)
        if not match:
            continue
        parts = list(map(int, match.groups()[6:]))
        end = datetime(parts[0], 1, 1, tzinfo=timezone.utc) + timedelta(days=parts[1] - 1,
              hours=parts[2], minutes=parts[3], seconds=parts[4], milliseconds=parts[5] * 100)
        latest = max(latest, end) if latest else end
    return latest.isoformat() if latest else None


def prepare():
    started = time.monotonic()
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    train_origins, official_origins, origins = load_origins()
    frame = stage_catalog(origins.min(), origins.max())
    candidate_records = remote_catalog(frame)
    fingerprint = code_hash()
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        records = list(pool.map(lambda r: process_frame(r, fingerprint), candidate_records, chunksize=8))
    map_errors = [r for r in records if r.get("byte_equal") and r.get("map_error") and not r.get("source_unusable_reason")]
    if map_errors:
        raise RuntimeError(f"{len(map_errors)} byte-verified frames failed image-map extraction; sample={map_errors[0]['s3_key']}: {map_errors[0]['map_error']}")
    catalog = FrameCatalog(records)
    get_map = _get_probe_maps(records)
    product_cache = FrameProductCache(catalog, get_map, PRODUCT_CACHE_ROOT, fingerprint)
    feature_rows, selection_rows, stats_rows = [], [], []
    selected_ids_all = set()
    for n, origin in enumerate(origins, 1):
        values, selected_ids, stats = _origin_arrays(origin, catalog, get_map, product_cache)
        selected_ids_all.update(selected_ids)
        feature_rows.append({"origin_last_input_utc": origin, **values})
        selection_rows.append({"origin": origin.isoformat(), "selected_ids_sha256": _digest_ids(selected_ids),
                               "selected_count": len(selected_ids)})
        stats_rows.append(stats)
        if n % 1000 == 0 or n == len(origins):
            print(f"SUVI-DIM72 origins {n}/{len(origins)}", flush=True)
    features = pd.DataFrame(feature_rows).sort_values("origin_last_input_utc").reset_index(drop=True)
    official_set = set(official_origins)
    official_mask = features["origin_last_input_utc"].isin(official_set)
    official_features = features.loc[official_mask]
    coverage = int((official_features["dim_missing"] == 0).sum())
    active = float((official_features["dim_area_frac"] > 0).mean())
    select_payload = "\n".join(_json_bytes(row).decode().rstrip("\n") for row in selection_rows).encode()
    selection_hash = sha_bytes(select_payload)
    by_id = {r["frame_id"]: r for r in records}
    selected_records = [by_id[frame_id] for frame_id in sorted(selected_ids_all)]
    selected_manifest_lines = [
        _json_bytes({key: row.get(key) for key in (
            "frame_id", "source_url", "s3_key", "obs_start", "obs_end", "last_modified",
            "local_path", "local_bytes", "local_sha256", "remote_etag", "remote_size", "byte_equal",
            "source_usable", "map_sha256"
        )}).decode().rstrip("\n") for row in selected_records
    ]
    selected_manifest_payload = ("\n".join(selected_manifest_lines) + "\n").encode()
    source_manifest_hash = sha_bytes(selected_manifest_payload)
    origins_hash = sha_bytes("\n".join(origin.isoformat() for origin in origins).encode())
    cache_key = sha_bytes((fingerprint + source_manifest_hash + origins_hash).encode())
    feature_dir = CACHE_ROOT / "feature-sets" / cache_key
    feature_dir.mkdir(parents=True, exist_ok=True)
    features_path = feature_dir / "suvi_dim72_features.parquet"
    manifest_path = feature_dir / "suvi_dim72_source_manifest.jsonl"
    probe_path = feature_dir / "suvi_dim72_poison_probes.json"
    if not features_path.is_file() or not manifest_path.is_file() or not probe_path.is_file():
        features.to_parquet(features_path, index=False)
        manifest_path.write_bytes(selected_manifest_payload)
        probe_origin = next((o for o in official_origins if (features.loc[features["origin_last_input_utc"] == o, "dim_missing"] == 0).any()), None)
        if probe_origin is None:
            raise RuntimeError("no source-covered official origin available for future/positive poison checks")
        probe_values, probe_ids, _ = _origin_arrays(probe_origin, catalog, get_map)
        probes = make_probes(probe_origin, catalog, get_map, probe_values, probe_ids)
        probe_path.write_bytes(_json_bytes(probes))
    else:
        probes = json.loads(probe_path.read_text())
    summary = {
        "item_id": "solar-inc-suvidim72-20261014",
        "cache_key": cache_key,
        "code_config_sha256": fingerprint,
        "config": CONFIG,
        "features": list(FEATURES),
        "origin_count_total": len(features),
        "oof_origin_count": len(train_origins),
        "official_origin_count": len(official_origins),
        "official_coverage_count": coverage,
        "official_coverage_fraction": coverage / len(official_origins),
        "coverage_feasibility_70pct": coverage / len(official_origins) >= 0.70,
        "active_dimming_origin_fraction": active,
        "selected_source_frame_count": len(selected_records),
        "byte_equal_geometry_unusable_frame_count": sum(r["byte_equal"] and not r["source_usable"] for r in records),
        "legacy_map_cache_reused_count": sum(r["map_reused_from_previous_code"] for r in records),
        "frame_product_cache": product_cache.receipt(),
        "all_selected_frames_byte_equal_s3": all(r["byte_equal"] for r in selected_records),
        "all_selected_frames_have_last_modified": all(r["last_modified"] is not None for r in selected_records),
        "selected_source_manifest_sha256": source_manifest_hash,
        "origin_selection_sha256": selection_hash,
        "feature_parquet_sha256": sha_file(features_path),
        "source_manifest_jsonl_sha256": sha_file(manifest_path),
        "poison_probe_json_sha256": sha_file(probe_path),
        "local_source_store_latest_observation": local_store_latest_obs(),
        "operational_proxy_evidence": "S3 LastModified<=origin, obs_end+1h<=origin, and single-part S3 ETag MD5 plus size equal local bytes; SHA256 recorded per selected frame",
        "poison": probes,
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path = feature_dir / "suvi_dim72_preparation_receipt.json"
    summary_path.write_bytes(_json_bytes(summary))
    pointer = {"cache_key": cache_key, "feature_dir": str(feature_dir),
               "code_config_sha256": fingerprint, "source_manifest_sha256": source_manifest_hash,
               "origin_selection_sha256": selection_hash}
    POINTER.write_bytes(_json_bytes(pointer))
    print(json.dumps({k: summary[k] for k in (
        "cache_key", "origin_count_total", "official_coverage_count", "official_coverage_fraction",
        "active_dimming_origin_fraction", "selected_source_frame_count", "all_selected_frames_byte_equal_s3",
        "local_source_store_latest_observation", "runtime_seconds")}, indent=2))
    return {**pointer, "summary_path": str(summary_path), "features_path": str(features_path),
            "manifest_path": str(manifest_path), "probe_path": str(probe_path)}


def export_to_expos(out_dir):
    pointer = json.loads(POINTER.read_text())
    feature_dir = Path(pointer["feature_dir"])
    receipt = feature_dir / "suvi_dim72_preparation_receipt.json"
    if not receipt.is_file():
        raise FileNotFoundError("SUVI-DIM72 cache pointer has no preparation receipt; run --prepare-only first")
    summary = json.loads(receipt.read_text())
    assert pointer["code_config_sha256"] == code_hash(), "SUVI-DIM72 cache code/config hash differs from registered source"
    assert summary["feature_parquet_sha256"] == sha_file(feature_dir / "suvi_dim72_features.parquet")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("suvi_dim72_features.parquet", "suvi_dim72_source_manifest.jsonl",
                 "suvi_dim72_poison_probes.json"):
        shutil.copy2(feature_dir / name, out_dir / name)
    summary["exported_to_expos_out"] = True
    (out_dir / "suvi_dim72_preparation_receipt.json").write_bytes(_json_bytes(summary))
    print(f"SUVI-DIM72 exported cache={pointer['cache_key']} origins={summary['origin_count_total']} coverage={summary['official_coverage_fraction']:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare()
    elif args.export:
        if not os.environ.get("EXPOS_OUT"):
            raise RuntimeError("EXPOS_OUT is required for export")
        export_to_expos(os.environ["EXPOS_OUT"])
    else:
        parser.error("choose --prepare-only or --export")

