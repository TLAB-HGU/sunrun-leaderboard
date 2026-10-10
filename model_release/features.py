"""Causal feature construction from explicit, portable numeric inputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import numeric, strips

REQUIRED_INPUTS = ("raw_ace_dir", "speed_parquet", "ch_parquet", "suvi_parquet", "suvi_manifest")
RAW_COLUMNS = ("ace_speed_kms", "ace_density_cm3", "ace_temperature_k", "ace_bt_nt", "ace_bz_gsm_nt",
               "ace_by_gsm_nt", "ace_epam_e_channel1", "ace_epam_e_channel2", "ace_epam_p_channel1",
               "ace_epam_p_channel3", "ace_epam_p_channel5", "ace_sis_gt10_flux")


def normalize_origins(origins):
    result = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
    if not len(result) or result.hasnans or result.has_duplicates:
        raise ValueError("Origins must be nonempty, unique UTC timestamps")
    if not result.equals(result.floor("h")):
        raise ValueError("Origins must lie on the hourly grid")
    return result


def _require_columns(frame, columns, label):
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


class FeatureInputs:
    """Read input tables once; predictions may request several causal grids."""

    def __init__(self, inputs, *, include_enlil=False):
        missing = set(REQUIRED_INPUTS).difference(inputs)
        if missing:
            raise ValueError(f"Missing inputs: {sorted(missing)}")
        for name in REQUIRED_INPUTS:
            if not Path(inputs[name]).exists():
                raise FileNotFoundError(f"{name}: {inputs[name]}")
        raw_paths = sorted(Path(inputs["raw_ace_dir"]).glob("year=*/part.parquet"))
        if not raw_paths:
            raise ValueError("raw_ace_dir must contain year=*/part.parquet")
        self.raw = pd.concat([pd.read_parquet(p) for p in raw_paths], ignore_index=True)
        _require_columns(self.raw, ("timestamp_utc", *RAW_COLUMNS), "raw ACE")
        speed = pd.read_parquet(inputs["speed_parquet"])
        _require_columns(speed, ("timestamp_utc", "filled_speed_kms"), "speed")
        self.speed = numeric.hourly(speed)["filled_speed_kms"]
        self.ch = pd.read_parquet(inputs["ch_parquet"])
        self.meta = pd.read_parquet(inputs["suvi_manifest"])
        _require_columns(self.meta, ("slot", "obs_end", "s3_modified", "status", "available"), "SUVI manifest")
        if self.meta.empty or self.meta["available"].isna().any() or self.meta["available"].dtype != bool:
            raise ValueError("SUVI manifest must contain rows and non-null boolean available flags")
        for name in ("slot", "obs_end", "s3_modified"):
            self.meta[name] = pd.to_datetime(self.meta[name], utc=True)
        self.meta = self.meta.sort_values("slot").reset_index(drop=True)
        if (self.meta["obs_end"] > self.meta["slot"]).any():
            raise ValueError("SUVI obs_end must not exceed its hourly slot")
        cache = pd.read_parquet(inputs["suvi_parquet"])
        cache.index = pd.to_datetime(cache.index, utc=True)
        lat1 = sorted(c for c in cache.columns if "_lat1_" in c)
        expected = {f"lon{i:02d}_lat1_{suffix}" for i in range(12)
                    for suffix in ("log_median", "log_q10", "dark_fraction", "valid_fraction")}
        if set(lat1) != expected or len(lat1) != len(expected):
            raise ValueError("SUVI numeric input requires exactly 48 equatorial statistics: "
                             "lon00..lon11 with log_median, log_q10, dark_fraction, valid_fraction")
        self.block = cache[lat1]
        # The frozen source indexes strip values by metadata row position.
        # Reject misaligned inputs instead of silently changing that arithmetic.
        if not self.block.index.equals(pd.DatetimeIndex(self.meta["slot"])):
            raise ValueError("SUVI numeric rows must match the sorted manifest slots exactly")
        self.events = None
        if include_enlil:
            directory = inputs.get("donki_dir")
            paths = sorted(Path(directory).glob("WSAEnlilSimulations_*.json")) if directory else []
            if not paths:
                raise ValueError("This model requires donki_dir containing WSAEnlilSimulations_*.json")
            rows = []
            for path in paths:
                for event in json.loads(path.read_text()):
                    if event.get("estimatedShockArrivalTime"):
                        rows.append((pd.Timestamp(event["modelCompletionTime"]),
                                     pd.Timestamp(event["estimatedShockArrivalTime"]),
                                     max(event.get("kp_90") or 0, event.get("kp_135") or 0, event.get("kp_180") or 0),
                                     bool(event.get("isEarthGB"))))
            self.events = pd.DataFrame(rows, columns=["done", "arrival", "kp", "glancing"])
            for name in ("done", "arrival"):
                self.events[name] = pd.to_datetime(self.events[name], utc=True)
            self.events = self.events.drop_duplicates().sort_values("done")

    def build(self, origins):
        origins = normalize_origins(origins)
        feats = (numeric.speed_features(origins, self.speed)
                 .join(numeric.ace_features(origins, self.raw), rsuffix="_raw")
                 .join(numeric.ch_features(origins, self.ch)))
        if self.events is not None:
            feats = feats.join(numeric.enlil_features(origins, self.events))
        grid = pd.DataFrame({"origin": np.repeat(origins, 72), "h": np.tile(np.arange(1, 73), len(origins))})
        grid["target_t"] = grid["origin"] + pd.to_timedelta(grid["h"], unit="h")
        for days in (26, 27, 28):
            grid[f"rec{days}"] = self.speed.reindex(pd.DatetimeIndex(grid["target_t"] - pd.Timedelta(days=days))).to_numpy()
        grid["rec_mean"] = grid[["rec26", "rec27", "rec28"]].mean(axis=1)
        grid = grid.join(feats, on="origin")
        if self.events is not None:
            grid["enlil_dt_target"] = grid["h"] - grid["enlil_next_arr_h"]
        strip_frame, _ = strips.strip_features(origins, self.meta, self.block)
        for column in strips.strip_columns():
            grid[column] = strip_frame[column].to_numpy(dtype=np.float32)
        return grid


def build_features(origins_utc, inputs, *, include_enlil=False):
    return FeatureInputs(inputs, include_enlil=include_enlil).build(origins_utc)
