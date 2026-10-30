"""D69 CH-geometry contract: preregistered constants, causality, single-axis."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1] / "experiments" / "d69_chgeo"
sys.path.insert(0, str(PKG))
import geofeatures as GF  # noqa: E402
import geospec as GS  # noqa: E402


def test_double_tag_front():
    assert GS.HYPOTHESIS.startswith("[family=xgb] [axes:")
    assert GS.FAMILY_TAG == "[family=xgb]" and "feat=ch-geometry" in GS.AXES_TAG
    assert GS.check_spec()["n_geo"] == 13


def test_constants_frozen():
    assert GS.THR == "t035" and GS.LAG_H == (96, 120, 144, 168)
    assert GS.LAT_CENTERS == (-60.0, 0.0, 60.0) and GS.TOL_H == 6.0
    assert GS.LAG_NAMES == ("ch_targ_m4d_area", "ch_targ_m5d_area",
                            "ch_targ_m6d_area", "ch_targ_m7d_area")
    assert len(GS.grid_cols()) == 36 and len(set(GS.grid_cols())) == 36
    assert "E0001" in GS.HYPOTHESIS and "E0003" in GS.HYPOTHESIS
    assert "AXES.md" in GS.HYPOTHESIS and GS.FULL_RULE and GS.SELECT_RULE


def test_threads_budget_and_gpu():
    assert GS.FIT_NJOBS == 3 and GS.PERIOD_JOBS == 3
    assert GS.PERIOD_JOBS * GS.FIT_NJOBS <= 10
    assert "gpu1" in GS.GPU_REASON and "cuda" in GS.GPU_REASON


def test_no_forbidden_strings_in_code():
    from expos import core
    blob = (PKG / "runner.py").read_text() + (PKG / "geospec.py").read_text() + (PKG / "geofeatures.py").read_text()
    assert core.scan(blob) == [], f"forbidden path strings in D69 code: {core.scan(blob)}"
    assert core.scan(GS.HYPOTHESIS) == []


def _synth_slots_grid():
    slots = pd.DatetimeIndex(["2025-01-01T00:00Z", "2025-01-01T01:00Z"], tz="UTC")
    grid = np.zeros((2, 36))
    grid[0, :] = 0.001  # uniform
    grid[1, 0] = 0.01  # single-cell spike in lon00_lat0
    return slots, grid


def test_boundary_curvature_latweight_synthetic():
    slots, grid = _synth_slots_grid()
    per = GF.origin_geometry(pd.DatetimeIndex(["2025-01-01T00:00Z"]), slots, grid)
    # uniform 0.001: total=0.036, boundary 0, curvature 0, latweight=60*0.024/0.036=40
    assert abs(per["ch_geo_total"].iloc[0] - 0.036) < 1e-9
    assert per["ch_geo_boundary_len"].iloc[0] == 0.0
    assert per["ch_geo_curvature"].iloc[0] == 0.0
    assert abs(per["ch_geo_latweight"].iloc[0] - 40.0) < 1e-3
    assert abs(per["ch_geo_polar_share"].iloc[0] - 24 / 36) < 1e-6
    per2 = GF.origin_geometry(pd.DatetimeIndex(["2025-01-01T01:00Z"]), slots, grid)
    assert per2["ch_geo_boundary_len"].iloc[0] > per["ch_geo_boundary_len"].iloc[0]
    assert per2["ch_geo_curvature"].iloc[0] > 0


def test_tolerance_and_future_never_used():
    slots, grid = _synth_slots_grid()
    # query 7h after last slot -> beyond 6h tolerance -> NaN
    per = GF.origin_geometry(pd.DatetimeIndex(["2025-01-01T08:00Z"]), slots, grid)
    assert bool(np.isnan(per["ch_geo_total"].iloc[0]))
    # query exactly at slot uses that slot, not a later one
    df = pd.DataFrame({"origin": pd.to_datetime(["2025-01-01T00:00Z"], utc=True), "h": [1]})
    out = GF.add_geo(df, slots, grid)
    for c in GS.GEO_COLS:
        assert c in out.columns
    # lagged lookups before any slot -> NaN (no future fill)
    assert bool(np.isnan(out["ch_targ_m4d_area"].iloc[0]))


def test_geo_poison_pre_invariant_and_positive_control():
    slots = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    grid = np.abs(rng.normal(0.005, 0.002, size=(48, 36)))
    origins = pd.date_range("2025-01-01", periods=36, freq="h", tz="UTC")
    res = GF.geo_poison_test(origins, slots, grid, "2025-01-01T18:00:00Z")
    assert res["passed"] is True and res["pre_invariant"] is True and len(res["moved_columns"]) > 0
