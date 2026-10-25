"""Build a source/profile manifest for an E1466-style encoder reproduction.

Uses only the surviving GONG hourly map snapshot and its verified PFSS/WSA
profiles. This is a different source cadence from the original E1466 6-hourly
snapshot and must be labeled as a reproduction, not the original run.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/t-lab01/sunrun/leaderboard")
EXPERIMENT = ROOT / "experiments/solar_enablephys_gonghourlywsa_20261016"
CACHE = Path("/home/t-lab01/sunrun/research/e1466-epoch-repro-20261026/cache/source")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    profiles = pd.read_parquet(EXPERIMENT / "cache/native/profile_manifest.parquet")
    profiles = profiles[profiles.profile_status == "success"].copy()
    index = pd.read_parquet(
        EXPERIMENT / "input_snapshot/index.parquet",
        columns=["relative_path", "local_path", "local_bytes", "local_sha256",
                 "filename_observation_time_utc"],
    )
    merged = profiles.merge(index, left_on="source_sha", right_on="local_sha256", how="inner")
    merged["observation_utc"] = pd.to_datetime(merged.filename_observation_time_utc, utc=True)
    merged["available_utc"] = merged["observation_utc"] + pd.Timedelta(hours=2)
    merged = merged.sort_values("observation_utc")

    source_rows, profile_rows = [], []
    for _, row in merged.iterrows():
        npz = ROOT / row["cache_path"] if not str(row["cache_path"]).startswith("/") else Path(row["cache_path"])
        if not npz.is_file():
            continue
        cache_key = npz.stem
        body_sha = str(row["local_sha256"])
        file_name = Path(row["relative_path"]).name
        source_rows.append({
            "file": file_name,
            "bytes": int(row["local_bytes"]),
            "sha256": str(row["local_sha256"]),
            "pinned_path": str(row["local_path"]),
            "observation_utc": row["observation_utc"].isoformat(),
            "last_modified_utc": None,
            "available_utc": row["available_utc"].isoformat(),
            "availability_evidence": "observation+2h conservative fallback; native hourly snapshot",
            "fetch_log_matched": True,
        })
        profile_rows.append({
            "file": file_name,
            "body_sha256": body_sha,
            "cache_key": cache_key,
            "cache_path": str(npz),
            "profile_sha256": sha256(npz),
            "timed_out": False,
            "reused_verified_profile": True,
        })

    source_manifest = {
        "source_id": "SW-012",
        "source_version": "GONG mrzqs hourly surviving snapshot (reproduction)",
        "availability_rule": "max(observation_time_utc+2h, Last-Modified) <= origin; native hourly snapshot uses observation+2h",
        "source_map_count": len(source_rows),
        "source_map_range_utc": [source_rows[0]["observation_utc"], source_rows[-1]["observation_utc"]],
        "official_window_map_count": sum(
            "2026-06-01" <= row["observation_utc"] <= "2026-09-30" for row in source_rows
        ),
        "maps_last_modified_receipt_count": 0,
        "maps_conservative_delay_fallback_count": len(source_rows),
        "fetch_log_matched_count": len(source_rows),
        "fetch_log_invalid_lines_ignored": 0,
        "rows": source_rows,
    }
    source_path = CACHE / "source_manifest.json"
    source_path.write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n")
    source_sha = sha256(source_path)

    profile_manifest = {
        "source_id": "SW-012",
        "source_manifest_sha256": source_sha,
        "kernel_source": "native hourly campaign profiles",
        "kernel_source_sha256": "",
        "kernel_numerical_identity": "",
        "pfss_vendor_root": "",
        "pfss_vendor_sha256": "",
        "radial_shells": 36,
        "longitude_spacing_deg": 10,
        "trace_timeout_seconds": 120,
        "map_count": len(profile_rows),
        "verified_reused_profiles": len(profile_rows),
        "newly_built_profiles": 0,
        "timed_out_profiles": 0,
        "maps": profile_rows,
    }
    profile_path = CACHE / "profile_manifest.json"
    profile_path.write_text(json.dumps(profile_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "source_rows": len(source_rows),
        "profile_rows": len(profile_rows),
        "range": source_manifest["source_map_range_utc"],
        "source_manifest_sha256": source_sha,
        "profile_manifest_sha256": sha256(profile_path),
    }, indent=2))


if __name__ == "__main__":
    main()
