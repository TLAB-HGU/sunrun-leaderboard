"""D22 hires-strips contract: geometry/proxy/poison invariants on synthetic numerics.

Pure/CPU-only: no fits, no data loading, no training, no scoring, no network,
no official reads. Every case runs synthetic arrays through the actual
hires helpers.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PKG = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(PKG / "d22_hires_strips"))
sys.path.insert(0, str(PKG / "d1_longitude_strips"))

import hires as HZ  # noqa: E402
import strips as ST  # noqa: E402


# --- frozen E0003 alignment ----------------------------------------------------

def test_frozen_alignment_matches_e0003_d1():
    assert HZ.OMEGA == ST.OMEGA == 0.55
    assert HZ.SPEEDS == ST.SPEEDS == (350, 450, 550, 650, 750)
    assert HZ.TAU == ST.TAU
    assert HZ.STRIP_HALF == ST.STRIP_HALF == 5.0
    assert HZ.STALE_H == ST.STALE_H == 6.0
    assert HZ.MARGIN_H == ST.MARGIN_H == 1.0
    for h in (1, 10, 36, 72):
        for v in HZ.SPEEDS:
            assert HZ.lam(h, v, 2.0) == pytest.approx(ST.lam(h, v, 2.0))
    assert HZ.lam(1, 350, 1.0) > 0  # west-positive prograde sign


def test_resolution_is_1deg_not_15deg():
    assert HZ.NBIN == 180 and HZ.LON_BIN_W == 1.0 and HZ.LON_LO == -90.0
    assert HZ.bin_weights(0.0).sum() == pytest.approx(1.0)
    assert (HZ.bin_weights(0.0) > 0).sum() == 10  # 10deg strip = ten 1deg bins
    assert HZ.bin_weights(88.0).sum() < 1.0  # limb truncation is partial, not padded


def test_column_contract_47_no_learned_weights():
    cols = HZ.hstrip_columns()
    assert len(cols) == 47
    assert cols[-2:] == ["himg_age_h", "himg_stale"]
    for v in HZ.SPEEDS:
        for s in HZ.STATS:
            assert f"hstrip_v{v}_{s}" in cols
    d = PKG / "d22_hires_strips"
    files = [p for p in d.iterdir() if p.is_file() and p.suffix != ".pyc"
             and "__pycache__" not in str(p)]
    assert len(files) <= 5, sorted(p.name for p in files)
    assert not list(d.glob("*.pkl")) + list(d.glob("*.pt")) + list(d.glob("*.bin"))


def test_no_forbidden_sources_in_code():
    blob = "\n".join((PKG / "d22_hires_strips" / f).read_text()
                     for f in ("hires.py", "runner.py"))
    for s in ("truth.parquet", "folds.parquet", "naive.parquet",
              "official-predictions", "baseline-replay", "official_check",
              "official-check", "predictions_official_truth",
              "datasets--tlabtlab--sunrun-lb-store", "huggingface"):
        assert s not in blob, s
    low = blob.lower()
    assert "gong" not in low and "donki" not in low


# --- histogram statistics -------------------------------------------------------

def _one_bin_hist(b, k, n=1000):
    H = np.zeros((HZ.NBIN, HZ.LOGB))
    H[b, k] = n
    return H.astype(float)


def test_quantile_of_single_bin_is_exact():
    w = HZ.LOG_EDGES[1] - HZ.LOG_EDGES[0]
    Hw = _one_bin_hist(90, 10)
    q = HZ._hist_quantiles(Hw[90:91], (0.10, 0.50, 0.90))[0]
    e = HZ.LOG_EDGES[10]
    assert q == pytest.approx((e + 0.10 * w, e + 0.50 * w, e + 0.90 * w))
    assert HZ._hist_quantiles(np.zeros((2, HZ.LOGB)), (0.5,))[0, 0] != \
        HZ._hist_quantiles(np.zeros((2, HZ.LOGB)), (0.5,))[0, 0]  # NaN stays NaN


def test_dark_fraction_thresholds_bracket():
    Hw = _one_bin_hist(90, 10)[90:91]
    below = HZ.LOG_EDGES[10] - 0.5 * (HZ.LOG_EDGES[1] - HZ.LOG_EDGES[0])
    above = HZ.LOG_EDGES[11] + 0.5 * (HZ.LOG_EDGES[1] - HZ.LOG_EDGES[0])
    assert HZ._hist_below(Hw, np.array([above]))[0] == pytest.approx(1.0)
    assert HZ._hist_below(Hw, np.array([below]))[0] == pytest.approx(0.0)
    assert np.isnan(HZ._hist_below(np.zeros((1, HZ.LOGB)), np.array([1.0]))[0])


# --- synthetic end-to-end assembly --------------------------------------------------

def _synth_slot(const_bin=10, ref=1.0):
    H = np.zeros((HZ.NBIN, HZ.LOGB), dtype=np.int32)
    H[:, const_bin] = 500
    V = np.full(HZ.NBIN, 500, dtype=np.int32)
    T = np.full(HZ.NBIN, 500, dtype=np.int32)
    E = np.zeros(HZ.NBIN, dtype=np.int32)
    return {"ok": True, "ref": ref, "n_valid": 500 * HZ.NBIN,
            "H": H, "V": V, "T": T, "E": E}


def _synth_meta(slots=("2025-08-04T00:00:00Z", "2025-08-04T06:00:00Z")):
    rows = []
    for s in slots:
        t = pd.Timestamp(s)
        rows.append({"slot": t, "status": "ok", "s3_key": "k.fits",
                     "obs_end": t - pd.Timedelta("30min"),
                     "s3_modified": t + pd.Timedelta("10min"),
                     "nc_path": "/tmp/does-not-matter.nc"})
    return pd.DataFrame(rows)


def test_assemble_constant_image_stats():
    meta = _synth_meta()
    origins = pd.DatetimeIndex(["2025-08-04T08:00:00Z"], tz="UTC")
    sel = HZ.select_slots(origins, meta,
                           present=np.array([True, True]))
    assert (sel["slot_idx"].to_numpy() == [1]).all()
    rec = {1: _synth_slot()}
    df = HZ.assemble(origins, sel, rec)
    assert len(df) == 72
    w = HZ.LOG_EDGES[1] - HZ.LOG_EDGES[0]
    e = HZ.LOG_EDGES[10]
    assert df["hstrip_v550_log_q50"].to_numpy() == pytest.approx(e + 0.5 * w)
    assert df["hstrip_v550_log_q10"].to_numpy() == pytest.approx(e + 0.1 * w)
    assert df["hstrip_v550_cover"].to_numpy() == pytest.approx(1.0)
    assert df["hstrip_v550_edge_frac"].to_numpy() == pytest.approx(0.0)
    assert df["hstrip_v550_grad"].to_numpy() == pytest.approx(0.0)
    assert (df["himg_stale"].to_numpy() == 0.0).all()
    assert df["himg_age_h"].to_numpy() == pytest.approx(2.5)  # O - obs_end
    assert df["hstrip_v550_dark_035"].notna().all()


def test_proxy_walkback_and_stale():
    meta = _synth_meta()
    origins = pd.DatetimeIndex(["2025-08-04T08:00:00Z"], tz="UTC")
    # newest slot file missing -> walk back to the earlier qualifying slot
    sel = HZ.select_slots(origins, meta, present=np.array([True, False]))
    assert (sel["slot_idx"].to_numpy() == [0]).all()
    # s3_modified in the future disqualifies even a present file
    meta2 = meta.copy()
    meta2.loc[1, "s3_modified"] = pd.Timestamp("2025-08-04T09:00:00Z")
    sel2 = HZ.select_slots(origins, meta2, present=np.array([True, True]))
    assert (sel2["slot_idx"].to_numpy() == [0]).all()
    # nothing eligible -> missing + stale
    sel3 = HZ.select_slots(origins, meta, present=np.array([False, False]))
    assert (sel3["slot_idx"].to_numpy() == [-1]).all()
    assert (sel3["img_stale"].to_numpy() == 1.0).all()
    df = HZ.assemble(origins, sel3, {})
    assert df[HZ.hstrip_columns()].isna().all().all() or \
        (df["himg_stale"] == 1.0).all()
    assert df[HZ.hstrip_columns()[:-2]].isna().all().all()


def test_bad_slot_exclusion_is_causal():
    meta = _synth_meta()
    origins = pd.DatetimeIndex(["2025-08-04T08:00:00Z"], tz="UTC")
    sel = HZ.select_slots(origins, meta, present=np.array([True, True]),
                          bad=(1,))
    assert (sel["slot_idx"].to_numpy() == [0]).all()


def test_poison_synthetic_passes_with_positive_control():
    meta = _synth_meta(("2025-08-04T06:00:00Z", "2025-08-05T06:00:00Z"))
    origins = pd.DatetimeIndex(["2025-08-04T08:00:00Z", "2025-08-05T08:00:00Z"],
                               tz="UTC")
    sel = HZ.select_slots(origins, meta, present=np.array([True, True]))
    assert sel["slot_idx"].tolist() == [0, 1]
    records = {0: _synth_slot(const_bin=10), 1: _synth_slot(const_bin=10)}
    out = HZ.poison_test(origins, "2025-08-04T12:00:00Z", sel, records, meta)
    assert out["passed"] is True and out["pre_invariant"] is True
    assert out["n_pre_rows"] == 72 and out["n_post_rows"] == 72
    assert len(out["moved_columns"]) > 0
    assert any("log_q" in c or "dark" in c for c in out["moved_columns"])
