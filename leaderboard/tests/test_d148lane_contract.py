"""D148 cmelane contract: CME rain + arrival kernel (fast, no fits).

Covers the owned file budget, spec identity (D148/worker-t1e6/full/E0472),
four pre-registered arms (bg/dbm/xgbcorr/amp) with the true-arrival upper
cited as E0481 headroom only, the E0408 base recipe unchanged with the
kernel as post-processing, drag integration (both branches), the 3x3
ensemble median, hit geometry, vintage enforcement, pre-cutoff matched
pairs (247), P(hit) calibration reporting, arrival-correction and
amplitude interfaces, kernel application, poison coverage, CPU/n_jobs
budget, seed propagation, output schema, scratch isolation, and
forbidden-source absence.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d148_cme_timing/cmelane"
D142 = LB / "experiments/d142_e0147depth"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "lean.py", "D147", "d147",
                 "nn-", "sun-nn", "E0471", "donki-live", "d148_enlil",
                 "ips_oracle", "spec_oracle", "run_oracle",
                 "run_s2", "spec_s2", "dbm_lib", "spec_s3", "run_s3",
                 "worker-t1e1", "worker-t1e2", "worker-t1e3", "worker-t1e4",
                 "worker-t1e5", "worker-t1a", "worker-t1b", "worker-t1c",
                 "worker-t1d", "worker-d141", "worker-d147",
                 "worker-d76", "worker-t2no"]
BANNED_SRC = ["GONG", "CDAW", "ADAPT", "WSA", "ENLIL"]
PINNED_W = {"1-6": {"e0003_line": 0.21536233464375223, "d33_hsplit": 0.3899845818930272,
                    "d45_lvlmean": 0.3946530834632206},
            "7-24": {"e0003_line": 0.3152013131693896, "d33_hsplit": 0.3436571856823209,
                     "d45_lvlmean": 0.3411415011482895},
            "25-48": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
                      "d45_lvlmean": 0.3333333333333333},
            "49-72": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
                      "d45_lvlmean": 0.3333333333333333}}

sys.path.insert(0, str(EXP))


def _load(exp_dir, mod_name, alias):
    loc = importlib.util.spec_from_file_location(alias, exp_dir / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(EXP.glob("*.py"))
                     if p.name in ("spec_lane.py", "kernel_lib.py", "run_lane.py"))


def _synth_A():
    # (submissionTime, time21_5, speed, halfAngle, lat, lon)
    rows = [("2026-05-01T00:00:00Z", "2026-05-02T00:00:00Z", 600.0, 30.0, 0.0, 0.0),
            ("2026-05-01T00:00:00Z", "2026-05-08T00:00:00Z", 1000.0, 20.0, 0.0, 60.0),
            ("2026-05-01T00:00:00Z", "2026-05-20T00:00:00Z", 2000.0, 60.0, 0.0, 0.0)]
    sub = np.array([pd.Timestamp(s, tz="UTC").value for s, _, _, _, _, _ in rows])
    t21 = np.array([pd.Timestamp(t, tz="UTC").value for _, t, _, _, _, _ in rows])
    spd = np.array([r[2] for r in rows])
    ha = np.array([r[3] for r in rows])
    la = np.array([r[4] for r in rows]) * np.pi / 180.0
    lo = np.array([r[5] for r in rows]) * np.pi / 180.0
    off = np.arccos(np.clip(np.cos(la) * np.cos(lo), -1.0, 1.0)) * 180.0 / np.pi
    order = np.argsort(t21, kind="stable")
    return {"sub": sub[order], "t21": t21[order], "speed": spd[order],
            "half": ha[order], "off": off[order]}


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert 3 <= len(pys) <= 6, files
    assert {"spec_lane.py", "kernel_lib.py", "run_lane.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load(EXP, "spec_lane", "d148_lane_spec")
    assert SP.DIRECTION == "D148" and SP.OWNER == "worker-t1e6"
    assert SP.KIND == "full" and SP.REFERENCE == "E0472"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24
    assert SP.SCRATCH == "store/scratch/worker-t1e6"
    assert SP.FIT_NJOBS <= 6
    assert SP.BASE_PARAMS["n_estimators"] == 900
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - 0.03) < 1e-12
    assert SP.BASE_PARAMS.get("device") == "cpu" and int(SP.BASE_PARAMS["n_jobs"]) <= 6
    assert SP.WEIGHTS == PINNED_W
    assert "[axes:" in SP.AXES_TAG
    assert tuple(SP.ARMS) == ("bg", "dbm", "xgbcorr", "amp")
    for a in SP.ARMS:
        assert "[axes:" in SP.AXES_TAGS[a], a
        assert "[axes:" in SP.HYPOTHESES[a], a
    assert float(SP.WINDOW_H) == 144.0
    assert abs(float(SP.L_KM) - 134640320.7) < 1.0
    assert abs(float(SP.GAMMA0) - 5.0e-08) < 1e-15
    assert abs(float(SP.W0) - 450.0) < 1e-9
    assert len(SP.GAMMA_ENS) == 3 and len(SP.W_ENS) == 3
    assert int(SP.N_CALIB_PAIRS) == 247 and int(SP.CALIB_BLOCKS) == 4
    assert SP.ORACLE_ID == "E0481" and abs(float(SP.ORACLE_MSE) - 2594.2) < 1.0
    assert abs(float(SP.ORACLE_GAIN) - -687.1) < 1.0
    assert abs(float(SP.BEST_MSE) - 3281.31) < 0.01
    rule = SP.SELECT_RULE
    assert "lb-eval" in rule and "F1" in rule and "3281" in rule
    assert "low_low_high" in rule and "high_high_high" in rule
    assert "0.0833" in rule and "E0481" in rule
    assert tuple(SP.CORR_FEATURES) == ("v0", "half", "off", "clear")
    assert SP.CORR_PARAMS.get("device") == "cpu" and int(SP.CORR_PARAMS["n_jobs"]) <= 6
    assert "v_final=v_bg" in SP.KERNEL_DEF


def test_single_axis_vs_recipe():
    SP = _load(EXP, "spec_lane", "d148_lane_spec_b")
    run_src = (EXP / "run_lane.py").read_text()
    for mod in ("run_eta", "depth_lib", "info_ablation", "strips", "existing_all"):
        assert mod in run_src, mod
    assert "spec_lane" in run_src and "kernel_lib" in run_src
    assert "900" in run_src and "0.03" in run_src
    assert "post-processing only" in run_src
    assert "105" in run_src
    assert "--arm" in run_src
    assert SP.BASE_PARAMS["subsample"] == 0.8 and SP.BASE_PARAMS["colsample_bytree"] == 0.7
    assert SP.BASE_PARAMS["min_child_weight"] == 20 and SP.BASE_PARAMS["reg_lambda"] == 5
    assert SP.BASE_PARAMS["tree_method"] == "hist"


def test_transit_physics():
    KL = _load(EXP, "kernel_lib", "d148_lane_a")
    th_fast, va_fast = KL.dbm_transit(1000.0, 450.0, 5.0e-08)[:2]
    th_slow, va_slow = KL.dbm_transit(300.0, 450.0, 5.0e-08)[:2]
    assert 6.0 < th_fast < th_slow < 300.0
    assert va_fast > 450.0 > va_slow > 300.0
    th_eq, va_eq = KL.dbm_transit(450.0, 450.0, 5.0e-08)[:2]
    assert abs(va_eq - 450.0) < 1e-9 and th_eq > 6.0
    th_weak, _ = KL.dbm_transit(800.0, 450.0, 2.5e-08)[:2]
    th_strong, va_strong = KL.dbm_transit(800.0, 450.0, 1.0e-07)[:2]
    assert th_weak < th_strong and va_strong < 1000.0


def test_ensemble_median_and_hit_geometry():
    KL = _load(EXP, "kernel_lib", "d148_lane_b")
    SP = _load(EXP, "spec_lane", "d148_lane_spec_c")
    mth, mva, sth = KL.ensemble_median(np.array([600.0, 1000.0]))
    assert mth.shape == (2,) and mva.shape == (2,) and sth.shape == (2,)
    assert bool((mth[1] < mth[0]).all()) and bool((mva > 400.0).all())
    assert len(SP.GAMMA_ENS) == 3 and len(SP.W_ENS) == 3
    A = _synth_A()
    assert bool(((A["off"] - A["half"]) <= 0.0).any())
    assert bool(((A["off"] - A["half"]) > 0.0).any())


def test_vintage_window_and_hourly_dt():
    KL = _load(EXP, "kernel_lib", "d148_lane_c")
    A = _synth_A()
    t_arr = A["t21"] + (np.array([50.0, 60.0, 40.0]) * 3600_000_000_000).astype(np.int64)
    mth, mva, _ = KL.ensemble_median(A["speed"])
    o = int(pd.Timestamp("2026-05-09T00:00:00Z", tz="UTC").value)
    assert KL._rep_for_origin(o, A, t_arr, mva, -0.38, -0.094) is not None
    old = int(pd.Timestamp("2026-06-20T00:00:00Z", tz="UTC").value)
    assert KL._rep_for_origin(old, A, A["t21"], mva, -0.38, -0.094) is None


def test_pairs_pre_cutoff_only():
    KL = _load(EXP, "kernel_lib", "d148_lane_d")
    SP = _load(EXP, "spec_lane", "d148_lane_spec_d")
    lib_src = (EXP / "kernel_lib.py").read_text()
    assert "cutoff_iso" in lib_src and "build_calib_pairs" in lib_src
    assert "calibrate_phit" in lib_src and "calibrate_arrival" in lib_src
    assert "calibrate_amplitude" in lib_src
    for bad in ("2026-06-", "2026-07-", "2026-08-", "2026-09-"):
        assert bad not in lib_src, bad
    pairs = KL.build_calib_pairs(SP.CUTOFF_ISO)
    assert len(pairs["pos"]["t21"]) == int(SP.N_CALIB_PAIRS) == 247
    cut = pd.Timestamp(SP.CUTOFF_ISO, tz="UTC")
    assert bool((pd.to_datetime(pairs["pos"]["t21"]).tz_localize("UTC") < cut).all())
    assert bool((pd.to_datetime(pairs["pos"]["arr"]).tz_localize("UTC") < cut).all())


def test_loader_reads_allowed_keys_only():
    run_src = (EXP / "run_lane.py").read_text()
    lib_src = (EXP / "kernel_lib.py").read_text()
    assert "submissionTime" in lib_src and "time21_5" in lib_src
    assert "submissionTime" in run_src and "time21_5" in run_src
    assert "isMostAccurate" not in run_src and "versionId" not in run_src
    assert "isMostAccurate" not in lib_src and "versionId" not in lib_src


def test_phit_calibration_reporting():
    KL = _load(EXP, "kernel_lib", "d148_lane_e")
    p = KL.phit_prob(np.array([-10.0, 0.0, 10.0]), 0.0, -0.2)
    assert p[0] > p[1] > p[2] and bool(((p > 0.0) & (p < 1.0)).all())
    ece, br = KL.ece_brier(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert 0.0 <= ece <= 0.5 and 0.0 <= br <= 0.25
    SP = _load(EXP, "spec_lane", "d148_lane_spec_e")
    (b0, b1), rep = KL.calibrate_phit(SP.CUTOFF_ISO)
    assert float(b1) < 0.0, "larger clearance must lower hit probability"
    for k in ("b0", "b1", "n", "hit_rate", "ece_oof", "brier_oof"):
        assert k in rep, k
    assert rep["ece_oof"] < 0.05 and rep["brier_oof"] < 0.2


def test_arrival_and_amplitude_interfaces():
    KL = _load(EXP, "kernel_lib", "d148_lane_f")
    SP = _load(EXP, "spec_lane", "d148_lane_spec_f")
    assert SP.CORR_PARAMS["n_estimators"] == 300 and SP.CORR_PARAMS["max_depth"] == 3
    assert KL.sigma_clip(2.0) == float(SP.SIGMA_MIN_H)
    assert KL.sigma_clip(13.8) == 13.8
    assert KL.sigma_clip(99.0) == float(SP.SIGMA_MAX_H)
    r = KL.amp_raw(np.array([450.0, 650.0]), np.array([30.0, 60.0]))
    assert r[0] == 0.0 and abs(r[1] - 400.0) < 1e-9
    b = KL.kernel_bump(np.array([10.0, 34.0]), 10.0, 12.0, 0.8, 200.0)
    assert b[0] > b[1] >= 0.0 and b[0] <= 160.0 + 1e-9


def test_kernel_apply_identity_and_bump():
    KL = _load(EXP, "kernel_lib", "d148_lane_g")
    SP = _load(EXP, "spec_lane", "d148_lane_spec_g")
    A = _synth_A()
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-05-09T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    v_bg = np.full(72, 400.0)
    out_bg = KL.apply_kernel(v_bg, te, A, (-0.38, -0.094), None, 16.0, 16.0, 1.2, "bg")
    assert bool((out_bg == 400.0).all())
    out = KL.apply_kernel(v_bg, te, A, (-0.38, -0.094), None, 6.0, 6.0, 1.0, "dbm")
    assert len(out) == 72 and bool(np.isfinite(out).all())
    assert bool((out >= 400.0).all())
    assert float(out.max()) > 400.0
    assert bool((out >= float(SP.CLIP_LO)).all()) and bool((out <= float(SP.CLIP_HI)).all())


def test_poison_probe_invariance_and_positive_control():
    KL = _load(EXP, "kernel_lib", "d148_lane_h")
    A = _synth_A()
    origins = pd.date_range("2026-05-05", "2026-05-25", freq="24h", tz="UTC")
    rec = KL.kernel_poison_probe(origins, "2026-05-10T00:00:00Z", A, shift_h=72.0)
    assert rec["passed"] is True and rec["pre_invariant"] is True
    assert rec["moved"] > 0 and rec["n_pre_rows"] > 0


def test_real_store_loads_vintage_clean():
    KL = _load(EXP, "kernel_lib", "d148_lane_i")
    A = KL.load_analyses()
    assert len(A["t21"]) > 4000
    assert bool((np.diff(A["t21"]) >= 0).all())
    assert bool((A["speed"] > 0).all()) and bool((A["half"] > 0).all())


def test_seed_actually_propagates():
    DL = _load(D142, "depth_lib", "d142_dl_ref_lane")
    run_src = (EXP / "run_lane.py").read_text()
    assert "fit_seed_models" in run_src and "event_seed" in run_src
    assert "level_seeds" in run_src or "level" in run_src
    blocks = [set(DL.level_seeds(s)) for s in (0, 1, 2)]
    assert len(set().union(*blocks)) == 9
    for s in (0, 1, 2):
        assert DL.event_seed(s) in DL.level_seeds(s)


def test_output_frame_schema():
    DL = _load(D142, "depth_lib", "d142_dl_schema_lane")
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = DL.frame_predictions(te, np.arange(72, dtype=float))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert len(fr) == 72 and fr["horizon_hours"].tolist() == list(range(1, 73))
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_scratch_isolation_and_cpu():
    import re
    SP = _load(EXP, "spec_lane", "d148_lane_spec_h")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1e6"
    assert len(set(re.findall(r"worker-[a-z0-9]+", code))) == 1
    assert "worker-t1e6" in code
    run_src = (EXP / "run_lane.py").read_text()
    assert 'device") == "cpu"' in run_src and "n_jobs" in run_src
    assert "--gpu" not in code


def test_poison_covers_new_inputs():
    run_src = (EXP / "run_lane.py").read_text()
    assert "poison_test" in run_src and "kernel_poison_probe" in run_src
    assert '"passed"' in run_src and "poison.json" in run_src


def test_no_forbidden_leak_or_banned_source():
    spec_src = (EXP / "spec_lane.py").read_text()
    run_src = (EXP / "run_lane.py").read_text()
    lib_src = (EXP / "kernel_lib.py").read_text()
    for w in FORBIDDEN:
        assert w not in spec_src, w
        assert w not in run_src, w
        assert w not in lib_src, w
    for w in OTHER_CONTACT:
        assert w not in run_src, w
        assert w not in lib_src, w
    assert "E0471" not in spec_src
    low = (spec_src + run_src + lib_src).lower()
    for w in BANNED_SRC:
        assert w.lower() not in low, w
    assert "cuda" not in low and "--gpu" not in (spec_src + run_src + lib_src)
