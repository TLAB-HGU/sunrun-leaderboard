"""Unit tests for the D156 NM-precursor panel machinery (directive #57 feature arm).

No network, no model code, no speed truth: synthetic minute/hourly records only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments/d156_nm57_feat"))
sys.path.insert(0, str(LB / "experiments/d155_nm1min"))
import nm_markers as NM  # noqa: E402
import nm_panel as NP  # noqa: E402
import spec_nm57 as SP  # noqa: E402


def _z_frame(values, index, cols):
    panel = pd.DataFrame(values, index=index, columns=cols)
    scales = {str(c): [0.0, 1.0] for c in cols}
    return NP.z_from_scales(panel, scales)


def test_onset_flag_fires_only_inside_long_runs():
    z = np.array([0.0, 5.0, 5.0, 5.0, 5.0, 0.0, 5.0])
    flags = NP._onset_flag(z)
    assert not flags[0] and not flags[5]
    assert flags[1:5].any() and not flags[6]
    short = np.array([0.0, 5.0, 5.0, 0.0])
    assert not NP._onset_flag(short).any()
    with_nan = np.array([np.nan, 5.0, 5.0, 5.0, 5.0])
    f = NP._onset_flag(with_nan)
    assert not f[0] and f.any()


def test_window_stats_ignore_rows_after_origin_minus_vintage():
    cols = pd.MultiIndex.from_tuples([("s", "shannon")], names=["station", "marker"])
    idx = pd.date_range("2026-01-01", periods=120, freq="h", tz="UTC")
    z = np.abs(_z_frame(np.zeros((120, 1)), idx, cols)).to_numpy()
    origin = idx[100]
    z[101:, 0] = 99.0  # rows strictly after origin - 1h must not be visible
    zdf = pd.DataFrame(z, index=idx, columns=cols)
    row = NP.window_stats(zdf, pd.DatetimeIndex([origin]), "nm").iloc[0]
    assert row["nm_zmax"] == pytest.approx(0.0)


def test_window_stats_use_the_full_search_window():
    cols = pd.MultiIndex.from_tuples([("s", "shannon")], names=["station", "marker"])
    idx = pd.date_range("2026-01-01", periods=120, freq="h", tz="UTC")
    z = np.zeros((120, 1))
    z[60, 0] = 7.0  # exactly origin - 40h, inside [origin-48h, origin-1h]
    zdf = pd.DataFrame(z, index=idx, columns=cols)
    origin = idx[100]
    row = NP.window_stats(zdf, pd.DatetimeIndex([origin]), "nm").iloc[0]
    assert row["nm_zmax"] == pytest.approx(7.0)
    z[52, 0] = 7.0  # origin - 48h, the oldest still-visible row
    zdf = pd.DataFrame(z, index=idx, columns=cols)
    assert NP.window_stats(zdf, pd.DatetimeIndex([origin]), "nm").iloc[0]["nm_zmax"] == pytest.approx(7.0)
    z[52, 0] = 0.0
    z[51, 0] = 7.0  # one row earlier: outside the window
    zdf = pd.DataFrame(z, index=idx, columns=cols)
    assert NP.window_stats(zdf, pd.DatetimeIndex([origin]), "nm").iloc[0]["nm_zmax"] == pytest.approx(7.0)
    z[51, 0], z[52, 0] = 0.0, 7.0
    z2 = np.zeros((120, 1))
    z2[52, 0] = 7.0
    z2[51, 0] = 0.0
    zdf = pd.DataFrame(z2, index=idx, columns=cols)
    assert NP.window_stats(zdf, pd.DatetimeIndex([origin]), "nm").iloc[0]["nm_zmax"] == pytest.approx(7.0)
    z3 = np.zeros((120, 1))
    z3[51, 0] = 7.0
    z3[52, 0] = 0.0
    zdf = pd.DataFrame(z3, index=idx, columns=cols)
    assert NP.window_stats(zdf, pd.DatetimeIndex([origin]), "nm").iloc[0]["nm_zmax"] == pytest.approx(0.0)


def test_free_mask_keeps_only_train_era_shock_free_hours():
    shocks = pd.DatetimeIndex([pd.Timestamp("2024-01-03", tz="UTC")])
    idx = pd.date_range("2024-01-01", periods=24 * 30, freq="h", tz="UTC")
    free = NP.free_mask(idx, shocks, pd.Timestamp("2024-07-01", tz="UTC"))
    near = np.abs((idx - shocks[0]).total_seconds() / 3600.0) >= SP.FREE_GAP_H
    assert bool((free & near).sum() == int(near.sum()))
    assert bool(free[idx.get_loc(pd.Timestamp("2024-01-10", tz="UTC"))])
    assert not bool(free[idx >= pd.Timestamp("2024-07-01", tz="UTC")].any())


def test_control_dates_are_shock_free_and_pre_cutoff():
    shocks = pd.DatetimeIndex([pd.Timestamp("2025-01-20", tz="UTC"),
                               pd.Timestamp("2025-02-20", tz="UTC"),
                               pd.Timestamp("2025-04-01", tz="UTC")])
    test = shocks[1:]
    ctls = NP.control_dates(test, shocks)
    for c in ctls:
        d = np.abs((shocks - c).total_seconds() / 3600.0)
        assert bool(np.all(d >= SP.FREE_GAP_H))
        assert c + pd.Timedelta(hours=6) < NP._utc(SP.CUTOFF_ISO)


def test_z_from_scales_uses_supplied_train_only_scales():
    cols = pd.MultiIndex.from_tuples([("s", "shannon")], names=["station", "marker"])
    idx = pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")
    panel = pd.DataFrame(np.arange(6, dtype=float)[:, None], index=idx, columns=cols)
    z = NP.z_from_scales(panel, {"('s', 'shannon')": [0.0, 1.0]})
    assert float(z.iloc[-1, 0]) == pytest.approx(1.0 / 1.4826)
