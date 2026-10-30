"""Feature-only copy of ACE turbulence recipe; no fit imports."""
import glob

from pathlib import Path

import numpy as np

import pandas as pd

RAW_COLUMNS = ("timestamp_utc", "ace_bt_nt", "ace_speed_kms", "ace_density_cm3")

TURB_COLUMNS = tuple(
    [f"ace_turb_power_{v}_{w}h" for v in ("b", "v", "n") for w in (1, 3, 6)]
    + [f"ace_turb_cv_{v}_6h" for v in ("b", "v", "n")]
    + ["ace_turb_pdyn_power_6h", "ace_turb_valid_fraction_6h", "ace_turb_missing_6h"]
)

def _rolling_power(series, hours):
    # Integrated non-DC periodogram power equals centered mean-square power
    # (Parseval); the store cadence is hourly, so windows contain hours+1 rows.
    return series.rolling(hours + 1, min_periods=2).var(ddof=0)

def build_turb_features(raw):
    raw = raw.sort_index()
    if not isinstance(raw.index, pd.DatetimeIndex) or raw.index.tz is None:
        raise ValueError("ACE raw index must be timezone-aware UTC")
    raw = raw.loc[~raw.index.duplicated(keep="last"), list(RAW_COLUMNS[1:])]
    raw.index = pd.to_datetime(raw.index, utc=True)
    names = {"b": "ace_bt_nt", "v": "ace_speed_kms", "n": "ace_density_cm3"}
    out = {}
    for short, column in names.items():
        values = pd.to_numeric(raw[column], errors="coerce")
        for hours in (1, 3, 6):
            out[f"ace_turb_power_{short}_{hours}h"] = _rolling_power(values, hours)
        win = values.rolling(7, min_periods=2)
        mean = win.mean().abs()
        out[f"ace_turb_cv_{short}_6h"] = win.std(ddof=0).div(mean.where(mean > 1e-12))

    pressure = 1.6726219e-6 * raw.ace_density_cm3 * raw.ace_speed_kms.pow(2)
    out["ace_turb_pdyn_power_6h"] = _rolling_power(pressure, 6)
    valid = raw[list(names.values())].notna().all(axis=1).astype(float)
    valid_fraction = valid.rolling(7, min_periods=1).mean()
    out["ace_turb_valid_fraction_6h"] = valid_fraction
    out["ace_turb_missing_6h"] = (valid_fraction < 1.0 - 1e-12).astype(float)
    features = pd.DataFrame(out, index=raw.index).reindex(columns=TURB_COLUMNS)
    if tuple(features.columns) != TURB_COLUMNS or features.shape[1] != 15:
        raise ValueError("ACE-TURB72 must produce the registered 15 columns")
    # The index is availability time, not observation time.
    features.index = features.index + pd.Timedelta(hours=2)
    return features
