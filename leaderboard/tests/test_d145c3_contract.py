"""D145-C3 contract: pure-helper + static gates for the temporal-evolution run.

No real data, no fits, no registration, no official reads. Covers the own
equatorial mask (threshold values, NaN handling), the 5-speed lambda
geometry, the t0/-6/-12/-24h temporal stats (occ/bnd/chg/per), the
E0408-recipe constants (CPU, jobs budget, cutoff, grid, depth, seeds,
single small group, scratch isolation, axes tag), the stitch/blend
helpers, and the runner's static contract (output names, poison gate,
forbidden-source absence).
"""

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d145_axisC_chmask"
C3 = EXP / "c3"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_IDS = ["e0147-abl", "d141_e0147", "ablate", "worker-d76", "worker-t1d7",
             "worker-d133", "worker-t2no", "worker-t1c1", "worker-t1c2", "worker-t1d4",
             "worker-d135", "e0003_line_alt", "nn-"]
BANNED_SRC = ["GONG", "WSA", "ADAPT", "CME", "DONKI"]
BANNED_DIR = ["D141", "sun-nn"]


def _load(name):
    loc = importlib.util.spec_from_file_location(f"d145c3_{name}", C3 / f"{name}.py")
    mod = importlib.util.module_from_spec(loc)
    sys.path.insert(0, str(C3))
    try:
        loc.loader.exec_module(mod)
    finally:
        sys.path.remove(str(C3))
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(C3.glob("*.py")))


def _synth(n_slots=48):
    slots = pd.date_range("2026-05-01", periods=n_slots, freq="h", tz="UTC")
    meta = pd.DataFrame({
        "slot": slots, "status": "ok",
        "obs_end": slots - pd.Timedelta("40min"),
        "s3_modified": slots - pd.Timedelta("10min"),
        "nc_path": [f"/tmp/d145c3_{i}.nc" for i in range(n_slots)],
        "s3_key": [f"k{i}" for i in range(n_slots)]})
    rng = np.random.default_rng(3)
    dark = pd.DataFrame(rng.uniform(0, 0.4, (n_slots, 12)), index=slots,
                        columns=[f"lon{i:02d}_lat1_dark_fraction" for i in range(12)])
    valid = pd.DataFrame(np.ones((n_slots, 12)), index=slots,
                         columns=[f"lon{i:02d}_lat1_valid_fraction" for i in range(12)])
    return meta, dark, valid


def test_owned_file_budget():
    files = sorted(p.name for p in C3.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) == 5, files
    assert {"spec_c3.py", "chmask_c3.py", "features_c3.py", "lib_c3.py",
            "run_official_c3.py"} == set(pys), files
    parent_pys = sorted(p.name for p in EXP.iterdir() if p.is_file() and p.suffix == ".py")
    assert set(parent_pys) == {"spec_c1.py", "chmask.py", "features_c1.py", "lib_c1.py",
                               "run_official_c1.py"}, parent_pys


def test_spec_identity_and_scope():
    SP = _load("spec_c3")
    assert SP.DIRECTION == "D145" and SP.OWNER == "worker-t1c3"
    assert SP.KIND == "full" and SP.REFERENCE == "E0408"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24
    assert SP.SCRATCH == "store/scratch/worker-t1c3"
    assert SP.MASK_CACHE == "store/chmask-cache"
    assert SP.FIT_NJOBS <= 6
    assert SP.DEVICE == "cpu" and SP.BASE_PARAMS["device"] == "cpu"
    assert SP.DEPTH == 3 and SP.BASE_PARAMS["max_depth"] == 3
    assert SP.BASE_PARAMS["n_estimators"] == 900
    assert abs(SP.BASE_PARAMS["learning_rate"] - 0.03) < 1e-12
    assert SP.MASK_DARK_T == 0.12 and SP.MASK_VALID_MIN == 0.5
    assert tuple(SP.SPEEDS) == (350, 450, 550, 650, 750)
    assert tuple(SP.LAGS) == (0, 6, 12, 24)
    assert SP.N_C3_ADDED == 20
    assert "[axes:" in SP.AXES_TAG and "chmask-temporal" in SP.AXES_TAG
    assert "25-72h" in SP.AXES_TAG or "branch" in SP.AXES_TAG


def test_hypothesis_has_axes_tag_and_single_group_claim():
    SP = _load("spec_c3")
    assert re.search(r"\[axes:\s*[^\]]+\]", SP.HYPOTHESIS, re.I)
    low = SP.HYPOTHESIS.lower()
    assert "exactly one added feature group" in low
    assert "25-72h" in SP.HYPOTHESIS and "1-24h" in SP.HYPOTHESIS
    assert "no" in low and "corrector" in low
    assert "e0408" in low and "3657" in SP.HYPOTHESIS
    assert "5188" in SP.HYPOTHESIS and "0.0833" in SP.HYPOTHESIS
    assert "temporal" in low and "persistence" in low
    assert "t0" in SP.HYPOTHESIS and "-24h" in SP.HYPOTHESIS
    assert "e0421" in low and "3637" in SP.HYPOTHESIS


def test_c3_columns_are_single_small_group():
    CM = _load("chmask_c3")
    cols = CM.c3_columns()
    assert len(cols) == 20 and len(set(cols)) == 20
    assert cols == [f"chm3_v{v}_occ" for v in (350, 450, 550, 650, 750)] + \
        [f"chm3_v{v}_bnd" for v in (350, 450, 550, 650, 750)] + \
        [f"chm3_v{v}_chg" for v in (350, 450, 550, 650, 750)] + \
        [f"chm3_v{v}_per" for v in (350, 450, 550, 650, 750)]


def test_mask_threshold_known_values():
    CM = _load("chmask_c3")
    dark = pd.DataFrame([[0.11, 0.12, 0.50, np.nan]],
                        columns=[f"c{i}" for i in range(4)])
    valid = pd.DataFrame([[1.0, 1.0, 1.0, 1.0]],
                         columns=[f"c{i}" for i in range(4)])
    m = CM.build_mask(dark, valid).to_numpy()[0]
    assert m[0] == 0.0 and m[1] == 1.0 and m[2] == 1.0 and np.isnan(m[3])
    valid2 = pd.DataFrame([[1.0, 0.1, 1.0, 1.0]],
                          columns=[f"c{i}" for i in range(4)])
    m2 = CM.build_mask(dark, valid2).to_numpy()[0]
    assert np.isnan(m2[1]) and m2[2] == 1.0


def test_geometry_lam_matches_frozen_arithmetic():
    CM = _load("chmask_c3")
    SP = _load("spec_c3")
    assert abs(CM.OMEGA - SP.OMEGA) < 1e-12
    assert abs(CM.TAU[550] - SP.D_KM / 550 / 3600.0) < 1e-9
    assert tuple(CM.LAGS) == tuple(SP.LAGS) == (0, 6, 12, 24)
    assert float(CM.lam(0, 550, 0.0)) == pytest.approx(CM.OMEGA * CM.TAU[550])
    assert float(CM.lam(72, 350, 6.0)) < float(CM.lam(1, 350, 1.0))


def test_strip_weights_cover_and_partition():
    CM = _load("chmask_c3")
    w = CM.strip_weights(np.array([0.0]))
    assert w.shape == (1, 12) and abs(w.sum() - 1.0) < 1e-9
    wfar = CM.strip_weights(np.array([500.0]))
    assert wfar.sum() == 0.0


def test_temporal_stats_known_values(monkeypatch):
    CM = _load("chmask_c3")
    # Uniform mask per slot: CH=1 for slots >= 30 else 0 -> occupancy equals
    # the slot value whatever the strip weights are (coverage is full).
    n_slots = 48
    slots = pd.date_range("2026-05-01", periods=n_slots, freq="h", tz="UTC")
    meta = pd.DataFrame({
        "slot": slots, "status": "ok",
        "obs_end": slots - pd.Timedelta("40min"),
        "s3_modified": slots - pd.Timedelta("10min"),
        "nc_path": [f"/tmp/d145c3t_{i}.nc" for i in range(n_slots)],
        "s3_key": [f"k{i}" for i in range(n_slots)]})
    for p in meta["nc_path"]:
        Path(p).write_bytes(b"x")
    monkeypatch.setattr(CM, "resolve_nc", lambda p: p)
    try:
        dark = pd.DataFrame(np.repeat(
            np.where(np.arange(n_slots)[:, None] >= 30, 0.5, 0.0), 12, axis=1),
                            index=slots,
                            columns=[f"lon{i:02d}_lat1_dark_fraction" for i in range(12)])
        valid = pd.DataFrame(np.ones((n_slots, 12)), index=slots,
                             columns=[f"lon{i:02d}_lat1_valid_fraction" for i in range(12)])
        mask = CM.build_mask(dark, valid)
        # Origin at slot 40 (+1h so S=slot40): lags 40/34/28/22 -> 1,1,0,0.
        origins = pd.DatetimeIndex([slots[40] + pd.Timedelta("1h")])
        df, _ = CM.chm3_features(origins, meta, mask)
        assert len(df) == 72
        occ = df["chm3_v550_occ"].to_numpy()
        bnd = df["chm3_v550_bnd"].to_numpy()
        chg = df["chm3_v550_chg"].to_numpy()
        per = df["chm3_v550_per"].to_numpy()
        assert np.all(occ == pytest.approx(1.0))
        assert np.all(bnd == pytest.approx(0.0))
        assert np.all(chg == pytest.approx(1.0 - 1.0 / 3.0))
        assert np.all(per == pytest.approx(0.5))
        # Origin at slot 20: t0 CH=0 (24h lag missing) -> t0 stats written,
        # chg/per NaN.
        origins0 = pd.DatetimeIndex([slots[20] + pd.Timedelta("1h")])
        df0, _ = CM.chm3_features(origins0, meta, mask)
        assert np.all(df0["chm3_v550_occ"].to_numpy() == pytest.approx(0.0))
        assert np.all(df0["chm3_v550_bnd"].to_numpy() == pytest.approx(0.0))
        assert np.isnan(df0["chm3_v550_chg"].to_numpy()).all()
        assert np.isnan(df0["chm3_v550_per"].to_numpy()).all()
        # Origin at slot 10: t0 CH=0 written, chg/per NaN (lag missing).
        origins_early = pd.DatetimeIndex([slots[10] + pd.Timedelta("1h")])
        dfe, _ = CM.chm3_features(origins_early, meta, mask)
        assert np.all(dfe["chm3_v550_occ"].to_numpy() == pytest.approx(0.0))
        assert np.isnan(dfe["chm3_v550_chg"].to_numpy()).all()
        assert np.isnan(dfe["chm3_v550_per"].to_numpy()).all()
    finally:
        for p in meta["nc_path"]:
            Path(p).unlink(missing_ok=True)


def test_stitch_branch_split_only_past_h24():
    DL = _load("lib_c3")
    h = np.arange(1, 73)
    m1 = h <= 6
    m2 = (h >= 7) & (h <= 24)
    m3 = h > 24
    rng = np.random.default_rng(1)
    s1 = {s: rng.normal(400, 10, 6) for s in (0, 1, 2)}
    s2 = {s: rng.normal(400, 10, 18) for s in (0, 1, 2)}
    lo = rng.normal(400, 10, 72)
    hi = lo + 5.0
    mixed = DL.stitch_members(s1, s2, lo, hi, hi[m3], m1, m2, m3, (0, 1, 2), 1)
    base = DL.stitch_members(s1, s2, lo, lo, lo[m3], m1, m2, m3, (0, 1, 2), 1)
    for k in DL.MEMBERS:
        assert np.array_equal(mixed[k][m1 | m2], base[k][m1 | m2])
        assert not np.array_equal(mixed[k][m3], base[k][m3])


def test_select_slots_proxy_rule_synthetic(monkeypatch, tmp_path):
    CM = _load("chmask_c3")
    meta, dark, valid = _synth()
    for i, p in enumerate(meta["nc_path"]):
        Path(p).write_bytes(b"x")
    monkeypatch.setattr(CM, "resolve_nc", lambda p: p)
    try:
        mask = CM.build_mask(dark, valid)
        origins = pd.DatetimeIndex([meta["slot"].iloc[10], meta["slot"].iloc[20]])
        sel = CM.select_slots(origins, meta, mask)
        assert (sel["slot_idx"].to_numpy() >= 0).all()
        assert bool((pd.to_datetime(sel["slot"]) <= origins).all())
        assert bool((pd.to_datetime(sel["s3_modified"]) <= origins).all())
        ages = pd.to_timedelta(origins.to_numpy() - pd.to_datetime(sel["obs_end"]).to_numpy()).total_seconds() / 3600.0
        assert bool((ages >= 1.0).all())
        future = pd.DatetimeIndex([meta["slot"].iloc[0] - pd.Timedelta("2h")])
        sel2 = CM.select_slots(future, meta, mask)
        assert sel2["slot_idx"].iloc[0] == -1 and sel2["stale"].iloc[0] == 1.0
    finally:
        for p in meta["nc_path"]:
            Path(p).unlink(missing_ok=True)


def test_chm3_features_shape_and_causality(monkeypatch):
    CM = _load("chmask_c3")
    meta, dark, valid = _synth()
    for p in meta["nc_path"]:
        Path(p).write_bytes(b"x")
    monkeypatch.setattr(CM, "resolve_nc", lambda p: p)
    try:
        mask = CM.build_mask(dark, valid)
        origins = pd.DatetimeIndex([meta["slot"].iloc[30], meta["slot"].iloc[40]])
        df, sel = CM.chm3_features(origins, meta, mask)
        assert list(df.columns) == ["origin", "h"] + CM.c3_columns()
        assert len(df) == 2 * 72
        assert ((df["h"].to_numpy() == np.tile(np.arange(1, 73), 2))).all()
        # Every lag slot is at or before its origin minus 1h.
        slot_time = pd.to_datetime(sel["slot"])
        assert bool((slot_time <= origins).all())
    finally:
        for p in meta["nc_path"]:
            Path(p).unlink(missing_ok=True)


def test_poison_keys_and_invariance_synthetic(monkeypatch):
    CM = _load("chmask_c3")
    meta, dark, valid = _synth(n_slots=72)
    for p in meta["nc_path"]:
        Path(p).write_bytes(b"x")
    monkeypatch.setattr(CM, "resolve_nc", lambda p: p)
    try:
        cut = meta["slot"].iloc[36]
        probe = pd.DatetimeIndex([meta["slot"].iloc[30], meta["slot"].iloc[60]])
        rec = CM.poison_test(probe, cut, meta=meta, dark=dark, valid=valid)
        assert rec["passed"] is True
        assert rec["pre_invariant"] is True and len(rec["moved_columns"]) > 0
        assert rec["n_pre_rows"] == 72 and rec["n_post_rows"] == 72
    finally:
        for p in meta["nc_path"]:
            Path(p).unlink(missing_ok=True)


def test_seeds_really_vary_the_fit():
    SP = _load("spec_c3")
    DL = _load("lib_c3")
    blocks = [DL.level_seeds(s) for s in SP.SEEDS]
    assert len({seed for b in blocks for seed in b}) == 9
    events = [DL.event_seed(s) for s in SP.SEEDS]
    assert len(set(events)) == 3
    params = [DL.fit_params({**SP.BASE_PARAMS, "max_depth": SP.DEPTH}, s)
              for s in SP.SEEDS for s in DL.level_seeds(s)]
    assert len({p["random_state"] for p in params}) == 9


def test_blend_weights_convex_and_pinned():
    SP = _load("spec_c3")
    DL = _load("lib_c3")
    DL.check_weights(SP.WEIGHTS)
    h = np.arange(1, 73)
    rng = np.random.default_rng(0)
    mem = {m: rng.normal(400, 100, 72) for m in DL.MEMBERS}
    out = DL.blend_per_lead(mem, SP.WEIGHTS, h)
    assert out.shape == (72,) and np.all(np.isfinite(out))


def test_frame_schema():
    DL = _load("lib_c3")
    te = pd.DataFrame({"origin": pd.date_range("2026-06-01", periods=2, freq="h", tz="UTC").repeat(72),
                       "h": np.tile(np.arange(1, 73), 2)})
    fr = DL.frame_predictions(te, np.full(len(te), 400.0))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert len(fr) == 144


def test_runner_writes_seed_files_and_poison_gate():
    src = (C3 / "run_official_c3.py").read_text()
    assert "predictions_official_s{s}.parquet" in src
    assert '"poison.json"' in src and '"mask_receipt.json"' in src
    assert "refusing to present predictions as causal" in src
    assert "CM.poison_test" in src and "ST.poison_test" in src
    assert "fit_short_and_long" in src and "predict_branch" in src
    assert "bit-identical" in src
    assert "--gpu" not in src
    assert "store/chmask-cache" in src or "MASK_CACHE" in src
    assert "worker-t1c3" in src
    assert "chmask_c3" in src and "features_c3" in src
    assert "lib_c3" in src and "spec_c3" in src


def test_runner_touches_no_forbidden_source():
    code = _code()
    for s in FORBIDDEN:
        assert s not in code, s
    for s in BANNED_SRC:
        assert s not in code, s
    for s in BANNED_DIR:
        assert s not in code, s
    for s in OTHER_IDS:
        assert s not in code, s
    for f in sorted(C3.glob("*.py")):
        t = f.read_text()
        assert "core.freeze" not in t and "official-score" not in t
        for w in ("freeze", "submit", "upload"):
            assert w not in t.lower(), (f.name, w)
