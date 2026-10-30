"""Contract for D1 longitude strips: geometry, schema, proxy rule, poison.

No official-window truth is read. All fixtures are synthetic except small
read-only samples of the approved preofficial inputs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments/d1_longitude_strips"))
import strips as ST  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay.parquet", "official_check", "official-check",
             "predictions_official_truth"]


def test_geometry_constants():
    assert ST.OMEGA == 0.55 and ST.D_KM == 147858620.7
    assert list(ST.SPEEDS) == [350, 450, 550, 650, 750]
    taus = [ST.TAU[v] for v in ST.SPEEDS]
    assert taus == sorted(taus, reverse=True)  # slower wind -> longer travel
    assert abs(ST.TAU[350] - 117.35) < 0.01 and abs(ST.TAU[750] - 54.76) < 0.01
    assert abs(ST.PARKER_WEST[350] - 64.54) < 0.01
    assert abs(ST.PARKER_WEST[750] - 30.12) < 0.01


def test_h_grid_and_single_formula():
    for v in ST.SPEEDS:  # h=0 control recovers Parker WEST exactly
        assert abs(ST.lam(0, v, 0.0) - ST.PARKER_WEST[v]) < 1e-9
    for v in ST.SPEEDS:  # h=72 shifts -39.6 vs h=0, -39.05 vs h=1
        assert abs((ST.lam(72, v, 0.0) - ST.lam(0, v, 0.0)) + 39.6) < 1e-9
        assert abs((ST.lam(72, v, 0.0) - ST.lam(1, v, 0.0)) + 39.05) < 1e-9
        older = ST.lam(24, v, 10.0)  # older image shifts east by Omega*age
        assert abs((older - ST.lam(24, v, 0.0)) + ST.OMEGA * 10.0) < 1e-9


def test_strip_schema_exact():
    cols = ST.strip_columns()
    assert len(cols) == 17
    assert cols == [f"strip_v{v}_{s}" for v in ST.SPEEDS for s in ("logmed", "dark", "cover")] \
        + ["img_age_h", "img_stale"]
    w = ST.strip_weights(7.5)  # strip centred on lon06 centre
    assert sum(w) > 0.99 and w[6] > 0.99  # contained in one 15deg cell


def _synth_meta():
    slots = pd.date_range("2025-08-04", periods=12, freq="h", tz="UTC")
    return pd.DataFrame({
        "slot": slots,
        "status": ["ok"] * 12,
        "obs_end": slots - pd.Timedelta("30min"),
        "s3_modified": slots - pd.Timedelta("10min"),
        "s3_key": ["k/x.fits"] * 12,
        "nc_path": [f"/tmp/synth_{i}.nc" for i in range(12)],
    })


def _synth_block(slots):
    cols = [f"lon{i:02d}_lat1_{s}" for i in range(12) for s in
            ("log_median", "dark_fraction", "valid_fraction")]
    v = np.ones((len(slots), len(cols)))
    return pd.DataFrame(v, index=pd.DatetimeIndex(slots, tz="UTC"), columns=cols)


def test_proxy_selection_newest_eligible_and_fallback(monkeypatch):
    meta = _synth_meta()
    block = _synth_block(meta["slot"])
    monkeypatch.setattr(ST.os.path, "isfile", lambda p: True)
    monkeypatch.setattr(ST.os.path, "getsize", lambda p: 1)
    O = meta["slot"].iloc[-1] + pd.Timedelta("2h")
    sel = ST.select_slots([O], meta, block)
    assert sel["slot"].iloc[0] == meta["slot"].iloc[-1]  # newest <= O-1h
    assert sel["img_stale"].iloc[0] == 0.0
    # Disqualify the newest two via s3_modified > O -> falls back to third.
    meta2 = meta.copy()
    meta2.loc[10:, "s3_modified"] = O + pd.Timedelta("1h")
    sel2 = ST.select_slots([O], meta2, block)
    assert sel2["slot"].iloc[0] == meta["slot"].iloc[9]
    assert sel2["img_age_h"].iloc[0] > 2.0  # age feature records the fallback
    # Nothing qualifies -> missing + stale, NaN age.
    meta3 = meta.copy()
    meta3["s3_modified"] = O + pd.Timedelta("1h")
    sel3 = ST.select_slots([O], meta3, block)
    assert bool(sel3["missing"].iloc[0]) and sel3["img_stale"].iloc[0] == 1.0
    assert np.isnan(sel3["img_age_h"].iloc[0])


def test_stale_rule_and_nan_propagation(monkeypatch):
    meta = _synth_meta()
    block = _synth_block(meta["slot"])
    monkeypatch.setattr(ST.os.path, "isfile", lambda p: True)
    monkeypatch.setattr(ST.os.path, "getsize", lambda p: 1)
    O = meta["slot"].iloc[-1] + pd.Timedelta("30h")  # image age ~29h -> stale
    df, sel = ST.strip_features([O], meta, block)
    assert (sel["img_stale"] == 1.0).all()
    assert df[ST.strip_columns()[:-2]].isna().all(axis=None)
    assert df["img_age_h"].notna().all() and (df["img_stale"] == 1.0).all()
    assert (df["h"].to_numpy() == np.arange(1, 73)).all()
    O2 = meta["slot"].iloc[-1] + pd.Timedelta("2h")  # fresh
    df2, _ = ST.strip_features([O2], meta, block)
    assert df2["strip_v550_logmed"].notna().all()
    assert (df2["strip_v550_cover"] == 1.0).all()


def test_poison_exact_keys_invariance_and_positive_control(monkeypatch):
    meta = _synth_meta()
    slots = meta["slot"]
    block = _synth_block(slots)
    monkeypatch.setattr(ST.os.path, "isfile", lambda p: True)
    monkeypatch.setattr(ST.os.path, "getsize", lambda p: 1)
    origins = [slots.iloc[3] + pd.Timedelta("2h"), slots.iloc[9] + pd.Timedelta("2h")]
    cut = slots.iloc[6]
    out = ST.poison_test(origins, cut, perturb=10.0, meta=meta, block=block)
    assert set(("passed",)) <= set(out) and out["passed"] is True
    assert out["pre_invariant"] is True and out["n_pre_rows"] == 72 and out["n_post_rows"] == 72
    assert len(out["moved_columns"]) > 0
    # No post-cutoff origins -> positive control absent -> must NOT pass.
    out2 = ST.poison_test([origins[0]], cut, perturb=10.0, meta=meta, block=block)
    assert out2["passed"] is False


def test_no_forbidden_references_in_new_code():
    for name in ("strips.py", "runner.py"):
        text = (LB / "experiments/d1_longitude_strips" / name).read_text()
        assert not [p for p in FORBIDDEN if p in text], name


def test_predictions_writer_contract(tmp_path):
    sys.path.insert(0, str(LB / "experiments/expos_runners"))
    import existing_all as ea
    import os
    origins = pd.date_range("2025-08-04", periods=4, freq="6h", tz="UTC")
    df = pd.DataFrame({"origin": np.repeat(origins, 72), "h": np.tile(np.arange(1, 73), 4)})
    monkeypatch_cwd = os.environ.get("EXPOS_OUT")
    os.environ["EXPOS_OUT"] = str(tmp_path)
    try:
        import importlib
        importlib.reload(ea)
        ea.write(df, np.full(len(df), 400.0), "dev")
        got = pd.read_parquet(tmp_path / "predictions_dev.parquet")
    finally:
        if monkeypatch_cwd is None:
            del os.environ["EXPOS_OUT"]
        else:
            os.environ["EXPOS_OUT"] = monkeypatch_cwd
    assert list(got.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert len(got) == 4 * 72 and got["horizon_hours"].between(1, 72).all()
    assert got.duplicated(["origin_last_input_utc", "horizon_hours"]).sum() == 0
