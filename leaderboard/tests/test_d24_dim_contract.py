"""D24 dimming contract: causality, geometry sharing, poison/output schema."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments/d1_longitude_strips"))
sys.path.insert(0, str(LB / "experiments/d24_dimming"))
sys.path.insert(0, str(LB / "experiments/headroom_audit"))
import strips as ST  # noqa: E402
import dimming as DM  # noqa: E402


def _synth(tmp_path, n=6):
    slots = pd.date_range("2025-08-04", periods=n, freq="h", tz="UTC")
    files = []
    for i, s in enumerate(slots):
        p = tmp_path / f"f{i}.nc"
        p.write_bytes(b"x")
        files.append(str(p))
    meta = pd.DataFrame({
        "slot": slots,
        "status": ["ok"] * n,
        "obs_end": slots - pd.Timedelta(minutes=30),
        "s3_modified": slots - pd.Timedelta(minutes=50),
        "nc_path": files,
    })
    cols = []
    for i in range(12):
        cols += [f"lon{i:02d}_lat1_log_median", f"lon{i:02d}_lat1_dark_fraction",
                 f"lon{i:02d}_lat1_log_q10", f"lon{i:02d}_lat1_valid_fraction"]
    vals = np.ones((n, len(cols)))
    # spatially varying ramp: cell i brightens at rate (i+1), so strip weights matter
    for r in range(n):
        for i in range(12):
            vals[r, i * 4 + 0] = 1.0 + 0.1 * r * (i + 1) / 12  # log_median per cell
            vals[r, i * 4 + 1] = 0.2  # dark
            vals[r, i * 4 + 2] = 0.5 + 0.05 * r * (i + 1) / 12  # q10
            vals[r, i * 4 + 3] = 1.0  # valid
    block = pd.DataFrame(vals, index=slots, columns=cols)
    return meta, block


def test_dimming_columns_exact_23_order():
    cols = DM.dimming_columns()
    exp = []
    for v in (350, 450, 550, 650, 750):
        exp += [f"dim_drop_v{v}", f"dim_area_v{v}", f"flare_up_v{v}", f"flare_q10_v{v}"]
    exp += ["dim_dt_h", "img_age_h", "img_stale"]
    assert cols == exp
    assert len(cols) == 23


def test_geometry_shared_not_redefined():
    assert DM.OMEGA == ST.OMEGA == 0.55
    assert DM.TAU == ST.TAU
    assert DM.SPEEDS == ST.SPEEDS == (350, 450, 550, 650, 750)
    assert DM.H_ALL is ST.H_ALL or list(DM.H_ALL) == list(ST.H_ALL)


def test_frame_reader_reuse():
    assert callable(DM.read_suvi_frame)
    assert callable(DM.disk_logmedian_diff)
    import inspect
    src = inspect.getsource(DM)
    assert "read_suvi_frame" in src
    assert "DQF" in inspect.getsource(DM.disk_logmedian_diff) or "qa == 0" in inspect.getsource(DM.disk_logmedian_diff)


def test_pair_selection_causality(tmp_path):
    meta, block = _synth(tmp_path)
    origins = pd.DatetimeIndex([meta["slot"].iloc[-1] + pd.Timedelta(hours=1)])
    sel = DM.select_pair_slots(origins, meta, block)
    assert sel["slot_idx0"].iloc[0] >= 0
    assert sel["slot_idx_prev"].iloc[0] >= 0
    assert sel["slot_prev"].iloc[0] < sel["slot"].iloc[0]
    assert sel["slot"].iloc[0] <= origins[0] - pd.Timedelta(hours=1)
    o = origins[0]
    s0 = meta.iloc[int(sel["slot_idx0"].iloc[0])]
    sp = meta.iloc[int(sel["slot_idx_prev"].iloc[0])]
    assert pd.Timestamp(s0["s3_modified"]) <= o
    assert pd.Timestamp(sp["s3_modified"]) <= o
    assert float(sel["dim_dt_h"].iloc[0]) > 0
    assert float(sel["img_age_h"].iloc[0]) >= 1.0


def test_pair_stale_when_single_slot(tmp_path):
    meta, block = _synth(tmp_path, n=1)
    origins = pd.DatetimeIndex([meta["slot"].iloc[0] + pd.Timedelta(hours=2)])
    sel = DM.select_pair_slots(origins, meta, block)
    assert float(sel["img_stale"].iloc[0]) == 1.0
    feats, _ = DM.build_features(origins, meta, block, selection=sel)
    assert len(feats) == 72
    assert feats["img_stale"].eq(1.0).all()
    assert feats["dim_drop_v350"].isna().all()


def test_weights_match_strips_and_sum_one():
    lam0 = 0.0
    w_vec = DM.weights_for_lams(np.array([lam0]))[0]
    w_ref = np.array(ST.strip_weights(lam0))
    np.testing.assert_allclose(w_vec, w_ref, rtol=1e-9)
    lams = np.array([[ST.OMEGA * (ST.TAU[550] - (h + 2.0)) for h in (1, 24, 72)]])
    w = DM.weights_for_lams(lams)
    assert w.shape == (1, 3, 12)
    np.testing.assert_allclose(w.sum(-1), 1.0, atol=1e-9)


def test_build_features_signs_and_schema(tmp_path):
    meta, block = _synth(tmp_path, n=6)
    origins = pd.DatetimeIndex([meta["slot"].iloc[-1] + pd.Timedelta(hours=1)])
    feats, sel = DM.build_features(origins, meta, block)
    assert list(feats["h"]) == list(range(1, 73))
    # brightness ramps up -> flare_up > 0, dim_drop == 0
    assert (feats["flare_up_v550"] > 0).all()
    assert (feats["dim_drop_v550"] == 0).all()
    assert (feats["flare_q10_v550"] > 0).all()
    # horizon dependence: different h must differ (geometry projection)
    assert feats["flare_up_v350"].nunique() > 1
    assert feats["dim_dt_h"].eq(1.0).all()


def test_poison_keys_and_positive_control(tmp_path):
    meta, block = _synth(tmp_path, n=8)
    origins = pd.DatetimeIndex(list(meta["slot"][2:])) + pd.Timedelta(hours=1)
    cut = meta["slot"].iloc[5].isoformat()
    rec = DM.poison_test(origins, cut, perturb=10.0, meta=meta, block=block)
    for k in ("passed", "cutoff_utc", "perturbation", "n_pre_rows", "n_post_rows",
              "pre_invariant", "moved_columns", "note"):
        assert k in rec
    assert rec["passed"] is True
    assert rec["pre_invariant"] is True
    assert len(rec["moved_columns"]) > 0


def test_runner_contract_cpu_no_forbidden():
    text = (LB / "experiments/d24_dimming/runner.py").read_text()
    low = text.lower()
    assert "tree_method" in text and "hist" in text
    for gpu_pat in ('device="cuda', "device='cuda", "torch.cuda", "cupy", "EXPOS_GPUS",
                    "prefer-gpu", "cuda:0", "cuda:1"):
        assert gpu_pat not in low, gpu_pat
    assert "--gpu" not in text
    for bad in ("truth.parquet", "folds.parquet", "naive.parquet", "baseline-replay",
                "official-predictions", "official_check", "noaa", "gong", "donki"):
        assert bad not in low, bad
    for need in ("predictions_", "poison.json", "availability_manifest.json", "--pilot",
                 "from existing_all import write", "DM.SCRATCH"):
        assert need in text, need
    assert "write(te" in text
    import info_ablation as ia
    assert ia.PARAMS["n_jobs"] <= 8
    assert ia.PARAMS["tree_method"] == "hist"


def test_scratch_isolation():
    assert str(DM.SCRATCH) == str(LB / "store" / "scratch" / "worker-d24")
    text = (LB / "experiments/d24_dimming/runner.py").read_text()
    assert "DM.SCRATCH" in text
    assert "worker-d23" not in text
