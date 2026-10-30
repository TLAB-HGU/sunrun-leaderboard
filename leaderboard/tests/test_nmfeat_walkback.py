"""D155 NM-precursor origin-frame unit tests (no network, synthetic inputs)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments/d155_nmfeat"))
sys.path.insert(0, str(LB / "experiments/d155_nm1min"))
import nm_markers as NMM  # noqa: E402
import nm_cache as NC  # noqa: E402
import spec_nmfeat as SP  # noqa: E402

ST = ["OULU", "APTY"]
COLS = list(NMM.MARKERS)


def _inputs(n_hours=120, fire_at=None, minutes_end_pad_h=24.0, all_nan=False):
    base = pd.Timestamp("2026-03-01T00:00Z")
    hours = pd.date_range(base, periods=n_hours, freq="h", tz="UTC")
    z = {}
    for c in ST:
        fr = pd.DataFrame(0.0, index=hours, columns=COLS)
        if fire_at is not None:
            fr.loc[hours[fire_at[0]:fire_at[1]], COLS[0]] = 5.0
        z[c] = fr
    glo = (base - pd.Timedelta(hours=2)).ceil("min")
    ghi = (hours[-1] + pd.Timedelta(hours=minutes_end_pad_h)).floor("min")
    full = pd.date_range(glo, ghi, freq="min", tz="UTC")
    minutes = {}
    for c in ST:
        arr = (np.full(len(full), np.nan, np.float32) if all_nan
               else np.full(len(full), 6000.0, np.float32))
        minutes[c] = (full, arr)
    return hours, z, minutes


def test_known_crossing_lead():
    hours, z, minutes = _inputs(fire_at=(50, 53))  # steps 50,51,52 fire
    o = hours[60]
    f = NC.origin_frame(pd.DatetimeIndex([o]), z, minutes, ST)
    assert f["nm_det"].iloc[0] == 1.0
    assert f["nm_nfire"].iloc[0] == 2.0  # both stations fire
    assert abs(f["nm_lead_h"].iloc[0] - 10.0) < 1e-9  # o - step50
    assert f["nm_stale"].iloc[0] == 0.0
    # zmax window [o-6h, o-1h] = rows 54..59 -> all zero
    assert f["nm_zmax"].iloc[0] == 0.0


def test_vintage_walkback_ignores_post_origin_minutes():
    hours, z, minutes = _inputs(fire_at=(50, 53))
    o = hours[60]
    f1 = NC.origin_frame(pd.DatetimeIndex([o]), z, minutes, ST)
    # truncate every minute after o-1h (simulate no future data at all)
    cut = (o - pd.Timedelta(hours=SP.VINTAGE_LAG_H)).value
    minutes2 = {}
    for c in ST:
        full, arr = minutes[c]
        fns = full.values.astype("datetime64[ns]").astype(np.int64)
        keep = fns <= cut
        minutes2[c] = (full[keep], arr[keep])
    f2 = NC.origin_frame(pd.DatetimeIndex([o]), z, minutes2, ST)
    a = f1[SP.NM_COLS].fillna(-999).to_numpy()
    b = f2[SP.NM_COLS].fillna(-999).to_numpy()
    assert np.allclose(a, b, equal_nan=True)


def test_all_stale_flags():
    hours, z, _ = _inputs(fire_at=(50, 53))
    _, _, minutes = _inputs(all_nan=True)
    o = hours[60]
    f = NC.origin_frame(pd.DatetimeIndex([o]), z, minutes, ST)
    assert f["nm_stale"].iloc[0] == 1.0
    assert f["nm_det"].iloc[0] == 0.0
    assert f["nm_nfire"].iloc[0] == 0.0
    assert np.isnan(f["nm_lead_h"].iloc[0])
    assert np.isnan(f["nm_zmax"].iloc[0])


def test_no_fire_means_no_detection():
    hours, z, minutes = _inputs()
    o = hours[100]
    f = NC.origin_frame(pd.DatetimeIndex([o]), z, minutes, ST)
    assert f["nm_det"].iloc[0] == 0.0
    assert np.isnan(f["nm_lead_h"].iloc[0])
    assert f["nm_stale"].iloc[0] == 0.0
