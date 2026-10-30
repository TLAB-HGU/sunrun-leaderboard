"""D146 Fe171-channel cache contract: static + synthetic checks (fast, no fits).

No real data, no arm registration, no official reads. Covers the owned file
budget, spec identity (D146/worker-t1d0/cache), geometry/proxy/walk-back
parity with the Fe195 strip pipeline (d1_longitude_strips/strips.py), the 17
fe171_ column contract, proxy exclusions, stale/missing handling, gap rows
for corrupt frames, poison invariance, and forbidden-source absence.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d146_fe171_channel"
D1 = LB / "experiments/d1_longitude_strips"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_IDS = ["e0147-abl", "d141_e0147", "ablate", "worker-d76", "worker-t1d7",
             "worker-t1a1", "worker-t1a2", "worker-t1a3", "worker-t1a5",
             "worker-d135", "worker-d133", "worker-t2no", "nn-"]
BANNED_SRC = ["GONG", "WSA", "ADAPT", "CME", "DONKI"]

sys.path.insert(0, str(EXP))


def _load(exp_dir, mod_name, alias):
    loc = importlib.util.spec_from_file_location(alias, exp_dir / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(EXP.glob("*.py")))


def _synth(tmp_path, n_slots=48, start="2026-05-01", seed=7):
    slots = pd.date_range(start, periods=n_slots, freq="h", tz="UTC")
    paths = []
    for i in range(n_slots):
        p = tmp_path / f"fe171_{i}.nc"
        p.write_bytes(b"x" * (100 + i))
        paths.append(str(p))
    meta = pd.DataFrame({
        "slot": slots, "status": "ok",
        "obs_end": slots - pd.Timedelta("40min"),
        "s3_modified": slots - pd.Timedelta("10min"),
        "nc_path": paths,
        "s3_key": [f"SUVI-L1b-Fe171/k{i}.fits" for i in range(n_slots)],
        "rsun": np.full(n_slots, 380.0),
        "nc_size": np.array([100 + i for i in range(n_slots)], dtype=float),
    })
    rng = np.random.default_rng(seed)
    block = pd.DataFrame(index=slots)
    for i in range(12):
        block[f"lon{i:02d}_lat1_log_median"] = rng.uniform(-1, 1, n_slots)
        block[f"lon{i:02d}_lat1_dark_fraction"] = rng.uniform(0, 0.4, n_slots)
        block[f"lon{i:02d}_lat1_valid_fraction"] = 1.0
    return meta, block


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 6, files
    assert {"spec171.py", "fe171_hourly.py", "strips171.py", "build_cache.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load(EXP, "spec171", "d146_spec")
    assert SP.DIRECTION == "D146" and SP.OWNER == "worker-t1d0"
    assert SP.KIND == "cache" and SP.REFERENCE == "E0421"
    assert SP.CHANNEL == "Fe171" and SP.AXES.get("feat") == "fe171-channel"
    assert SP.SCRATCH == "store/scratch/worker-t1d0"
    assert SP.CACHE_DIR == "store/fe171-cache"
    assert SP.HOURLY_NAME == "fe171-hourly-v1.parquet"
    assert SP.BUILD_WORKERS <= 6 and SP.DEVICE == "cpu"
    assert tuple(SP.SPEEDS) == (350, 450, 550, 650, 750)
    assert SP.N_STRIP == 17 and SP.COL_PREFIX == "fe171_"
    assert SP.EXTRACTOR_VERSION == 2


def test_geometry_constants_match_fe195_pipeline():
    SP = _load(EXP, "spec171", "d146_spec_b")
    D1S = _load(D1, "strips", "d1_strips_ref")
    assert SP.OMEGA == D1S.OMEGA and SP.D_KM == D1S.D_KM
    assert tuple(SP.SPEEDS) == tuple(D1S.SPEEDS)
    assert SP.CELL_W == D1S.CELL_W and SP.STRIP_HALF == D1S.STRIP_HALF
    assert SP.COVER_MIN == D1S.COVER_MIN and SP.STALE_H == D1S.STALE_H
    assert SP.MARGIN_H == D1S.MARGIN_H
    ST = _load(EXP, "strips171", "d146_strips_b")
    assert ST.TAU == D1S.TAU and ST.PARKER_WEST == D1S.PARKER_WEST
    assert ST.STAT_SUFFIX == D1S.STAT_SUFFIX
    assert list(ST.H_ALL) == list(D1S.H_ALL)
    assert ST.strip_weights(10.0) == D1S.strip_weights(10.0)
    assert float(ST.lam(24, 550, 2.0)) == float(D1S.lam(24, 550, 2.0))


def test_extractor_is_shared_function_object():
    FH = _load(EXP, "fe171_hourly", "d146_hourly_b")
    from experiments.xgboost_suvi_fusion.features import extract_frame_features
    assert FH.extract_frame_features is extract_frame_features


def test_strip_column_contract():
    SP = _load(EXP, "spec171", "d146_spec_c")
    ST = _load(EXP, "strips171", "d146_strips_c")
    cols = ST.strip_columns()
    assert len(cols) == 17 == SP.N_STRIP
    assert all(c.startswith("fe171_") for c in cols)
    assert cols[-2:] == ["fe171_img_age_h", "fe171_img_stale"]
    assert len(set(cols)) == 17


def test_numeric_parity_with_fe195_strips(tmp_path):
    ST = _load(EXP, "strips171", "d146_strips_d")
    D1S = _load(D1, "strips", "d1_strips_num")
    meta, block = _synth(tmp_path)
    origins = pd.DatetimeIndex(["2026-05-03T00:00:00Z", "2026-05-03T06:00:00Z"], tz="UTC")
    got, _ = ST.strip_features(origins, meta, block)
    ref, _ = D1S.strip_features(origins, meta, block)
    assert len(got) == 2 * 72 and list(got["h"].iloc[:72]) == list(range(1, 73))
    for c in ref.columns:
        if c in ("origin", "h"):
            continue
        g = f"fe171_{c}" if not c.startswith("img_") else f"fe171_{c}"
        assert g in got.columns, g
        assert np.array_equal(np.asarray(got[g]), np.asarray(ref[c]), equal_nan=True), g


def test_proxy_excludes_too_new_slot_and_future_s3(tmp_path):
    ST = _load(EXP, "strips171", "d146_strips_e")
    meta, block = _synth(tmp_path, n_slots=6, start="2026-05-01")
    origin = pd.DatetimeIndex(["2026-05-01T06:00:00Z"], tz="UTC")
    sel = ST.select_slots(origin, meta, block)
    assert str(sel["slot"].iloc[0]) == "2026-05-01 05:00:00+00:00"
    late = meta.copy()
    late.loc[5, "s3_modified"] = pd.Timestamp("2026-05-01T07:00:00Z")
    sel2 = ST.select_slots(origin, late, block)
    assert str(sel2["slot"].iloc[0]) == "2026-05-01 04:00:00+00:00"


def test_missing_slot_gives_nan_and_stale(tmp_path):
    ST = _load(EXP, "strips171", "d146_strips_f")
    meta, block = _synth(tmp_path, n_slots=6, start="2026-05-01")
    origin = pd.DatetimeIndex(["2026-05-01T00:30:00Z"], tz="UTC")
    df, sel = ST.strip_features(origin, meta, block)
    assert bool(sel["missing"].iloc[0]) and sel["slot_idx"].iloc[0] == -1
    assert bool((df["fe171_img_stale"] == 1.0).all())
    assert bool(df[ST.strip_columns()].iloc[:, :15].isna().all().all())


def test_stale_age_gives_nan_and_flag(tmp_path):
    ST = _load(EXP, "strips171", "d146_strips_g")
    meta, block = _synth(tmp_path, n_slots=4, start="2026-05-01")
    origin = pd.DatetimeIndex(["2026-05-01T12:00:00Z"], tz="UTC")
    df, sel = ST.strip_features(origin, meta, block)
    assert float(sel["img_age_h"].iloc[0]) > 6.0
    assert float(sel["img_stale"].iloc[0]) == 1.0
    assert bool(df[ST.strip_columns()].iloc[:, :15].isna().all().all())


def test_gap_rows_for_bad_inputs(tmp_path):
    FH = _load(EXP, "fe171_hourly", "d146_hourly_c")
    base = {"status": "ok", "slot": pd.Timestamp("2026-05-01T01:00:00Z"),
            "obs_end": pd.Timestamp("2026-05-01T00:20:00Z"), "rsun": 380.0,
            "nc_path": None, "nc_size": np.nan}
    assert FH._row_path(base) is None
    assert FH._row_path(base | {"status": "missing"}) is None
    assert FH._row_path(base | {"rsun": -1.0}) is None
    assert FH._row_path(base | {"obs_end": pd.Timestamp("2026-05-01T02:00:00Z")}) is None
    p = tmp_path / "f.nc"
    p.write_bytes(b"abc")
    assert FH._row_path(base | {"nc_path": str(p), "nc_size": 3}) == str(p)
    assert FH._row_path(base | {"nc_path": str(p), "nc_size": 4}) is None


def test_source_identity_deterministic(tmp_path):
    FH = _load(EXP, "fe171_hourly", "d146_hourly_d")
    meta, _ = _synth(tmp_path, n_slots=8)
    a = FH.source_identity(meta)["source_sha256"]
    b = FH.source_identity(meta.iloc[::-1].reset_index(drop=True))["source_sha256"]
    assert a != b  # order matters: identity is slot-ordered
    assert FH.source_identity(meta)["source_sha256"] == a


def test_poison_invariance_and_positive_control(tmp_path):
    ST = _load(EXP, "strips171", "d146_strips_h")
    meta, block = _synth(tmp_path, n_slots=72, start="2026-05-01")
    origins = pd.date_range("2026-05-02", periods=6, freq="12h", tz="UTC")
    rec = ST.poison_test(origins, "2026-05-03T00:00:00Z", meta=meta, block=block)
    assert rec["passed"] is True and rec["pre_invariant"] is True
    assert rec["n_pre_rows"] > 0 and rec["n_post_rows"] > 0


def test_no_forbidden_leak_or_banned_source():
    code = _code()
    for w in FORBIDDEN:
        assert w not in code, w
    for w in OTHER_IDS:
        assert w not in code, w
    low = code.lower()
    for w in BANNED_SRC:
        assert w.lower() not in low, w
    assert "cuda" not in low and "--gpu" not in code
    assert "truth" not in low or "official-window truth" in low
