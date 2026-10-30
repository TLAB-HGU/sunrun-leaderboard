"""D148 Step 3 DBM/DBEM contract: drag-arrival features (fast, no fits).

No real data fits, no arm registration, no official reads. Covers the owned
file budget, spec identity (D148/worker-t1e3/full/E0408), the single-axis
rule (E0408 base identical + four dbm_* columns only), DBM integration
(dv/dt=-gamma(v-w)|v-w| to 1 AU, both v0>w and v0<w branches), the 3x3 DBEM
ensemble median, Earth-hit geometry, the pre-cutoff linked-pair gamma/w
fixed point (official window never read), vintage enforcement, poison
coverage of the new DBM input, CPU/n_jobs budget, seed propagation, output
schema, scratch isolation, Step 2 raw comparison context, and
forbidden-source absence.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d148_cme_timing"
D142 = LB / "experiments/d142_e0147depth"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "lean.py", "D147", "d147",
                 "nn-", "sun-nn", "E0471", "donki-live", "d148_enlil",
                 "ips_oracle", "spec_oracle", "run_oracle",
                 "run_s2", "spec_s2",
                 "worker-t1e1", "worker-t1e2", "worker-t1a", "worker-t1b",
                 "worker-t1c", "worker-t1d", "worker-d141", "worker-d147",
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
DBM_COLS = ("dbm_n6d", "dbm_dt_arr", "dbm_varr", "dbm_phit")

sys.path.insert(0, str(EXP))


def _load(exp_dir, mod_name, alias):
    loc = importlib.util.spec_from_file_location(alias, exp_dir / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(EXP.glob("*.py"))
                     if p.name in ("spec_s3.py", "dbm_lib.py", "run_s3.py"))


def _synth_A():
    # (submissionTime, time21_5, speed, halfAngle, lat, lon)
    rows = [("2026-06-01T00:00:00Z", "2026-06-02T00:00:00Z", 600.0, 30.0, 0.0, 0.0),
            ("2026-06-01T00:00:00Z", "2026-06-03T00:00:00Z", 1000.0, 20.0, 0.0, 90.0),
            ("2026-06-10T00:00:00Z", "2026-06-02T00:00:00Z", 2000.0, 60.0, 0.0, 0.0)]
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
    assert len(pys) == 9, files
    assert {"spec_s3.py", "dbm_lib.py", "run_s3.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load(EXP, "spec_s3", "d148_s3_spec")
    assert SP.DIRECTION == "D148" and SP.OWNER == "worker-t1e3"
    assert SP.KIND == "full" and SP.REFERENCE == "E0408"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24
    assert SP.SCRATCH == "store/scratch/worker-t1e3"
    assert SP.FIT_NJOBS <= 6
    assert SP.BASE_PARAMS["n_estimators"] == 900
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - 0.03) < 1e-12
    assert SP.BASE_PARAMS.get("device") == "cpu" and int(SP.BASE_PARAMS["n_jobs"]) <= 6
    assert SP.WEIGHTS == PINNED_W
    assert tuple(SP.DBM_COLS) == DBM_COLS
    assert "[axes:" in SP.AXES_TAG
    assert float(SP.WINDOW_H) == 144.0
    assert abs(float(SP.L_KM) - 134640320.7) < 1.0
    assert abs(float(SP.GAMMA0) - 5.0e-08) < 1e-15
    assert abs(float(SP.W0) - 450.0) < 1e-9
    assert len(SP.GAMMA_ENS) == 3 and len(SP.W_ENS) == 3
    assert int(SP.N_CALIB_PAIRS) == 247 and int(SP.CALIB_BLOCKS) == 4
    rule = SP.SELECT_RULE
    assert "lb-eval" in rule and "F1" in rule and "3281" in rule
    assert "low_low_high" in rule and "high_high_high" in rule
    assert "3681" in rule or "Step 2" in rule


def test_single_axis_vs_recipe():
    SP = _load(EXP, "spec_s3", "d148_s3_spec_b")
    run_src = (EXP / "run_s3.py").read_text()
    for mod in ("run_eta", "depth_lib", "info_ablation", "strips", "existing_all"):
        assert mod in run_src, mod
    assert "spec_s3" in run_src and "dbm_lib" in run_src
    assert "900" in run_src and "0.03" in run_src
    assert "DBM_COLS" in run_src
    assert "single axis violated" in run_src
    assert SP.BASE_PARAMS["subsample"] == 0.8 and SP.BASE_PARAMS["colsample_bytree"] == 0.7
    assert SP.BASE_PARAMS["min_child_weight"] == 20 and SP.BASE_PARAMS["reg_lambda"] == 5
    assert SP.BASE_PARAMS["tree_method"] == "hist"


def test_dbm_transit_physics():
    DBM = _load(EXP, "dbm_lib", "d148_dbm_a")
    # coasting at w: transit = L / w exactly.
    th, va = DBM.dbm_transit(450.0, 450.0, 5.0e-08)
    assert abs(th - 134640320.7 / 450.0 / 3600.0) < 1e-6
    assert abs(va - 450.0) < 1e-9
    # faster initial speed arrives earlier; arrival speed stays between w and v0.
    th_fast, va_fast = DBM.dbm_transit(1000.0, 450.0, 5.0e-08)
    th_slow, va_slow = DBM.dbm_transit(600.0, 450.0, 5.0e-08)
    assert th_fast < th_slow
    assert 450.0 < va_fast < 1000.0 and 450.0 < va_slow < 600.0
    assert va_fast > va_slow
    # slow CME accelerates toward w: v0 < varr < w.
    th_creep, va_creep = DBM.dbm_transit(268.0, 450.0, 5.0e-08)
    assert 268.0 < va_creep < 450.0
    assert th_creep > th_slow
    # stronger drag slows a fast CME more (later arrival, slower arrival speed).
    th_weak, _ = DBM.dbm_transit(1000.0, 450.0, 2.5e-08)
    th_strong, va_strong = DBM.dbm_transit(1000.0, 450.0, 1.0e-07)
    assert th_weak < th_strong and va_strong < 1000.0


def test_dbm_ensemble_median_and_hit_geometry():
    DBM = _load(EXP, "dbm_lib", "d148_dbm_b")
    SP = _load(EXP, "spec_s3", "d148_s3_spec_c")
    assert len(SP.GAMMA_ENS) * len(SP.W_ENS) == 9
    th, va, spread = DBM.ensemble_median(800.0)
    lo = min(DBM.dbm_transit(800.0, float(w), float(g))[0]
             for g in SP.GAMMA_ENS for w in SP.W_ENS)
    hi = max(DBM.dbm_transit(800.0, float(w), float(g))[0]
             for g in SP.GAMMA_ENS for w in SP.W_ENS)
    assert lo <= th <= hi and spread > 0.0
    assert 400.0 < va < 800.0
    # hit geometry: disk center inside the cone, limb outside it.
    A = _synth_A()
    offs = sorted(float(x) for x in A["off"])
    assert abs(offs[0] - 0.0) < 1e-9 and abs(offs[-1] - 90.0) < 1e-9
    assert bool(((A["off"] - A["half"]) <= 0.0).any())
    assert bool(((A["off"] - A["half"]) > 0.0).any())


def test_vintage_window_and_hourly_dt():
    A = _synth_A()
    DBM = _load(EXP, "dbm_lib", "d148_dbm_c")
    df = pd.DataFrame({"origin": pd.to_datetime(["2026-06-04T00:00:00Z"] * 3, utc=True),
                       "h": [1, 2, 3]})
    out = DBM.add_dbm_features(df, A)
    assert list(out.columns[-4:]) == list(DBM_COLS)
    r = out.iloc[0]
    # usable: analyses 0-1 (submitted pre-origin, time21_5 in trailing 144 h);
    # analysis 2 excluded (submissionTime after origin despite fast speed).
    assert float(r["dbm_n6d"]) == 2.0
    # one of two usable hits (disk center) -> hit fraction one half.
    assert abs(float(r["dbm_phit"]) - 0.5) < 1e-12
    assert np.isfinite(float(r["dbm_varr"])) and float(r["dbm_varr"]) > 400.0
    # hourly: predicted arrival minus target time falls one hour per horizon step.
    dts = out["dbm_dt_arr"].to_numpy()
    assert np.isfinite(dts).all()
    assert abs((dts[0] - dts[1]) - 1.0) < 1e-9
    assert abs((dts[1] - dts[2]) - 1.0) < 1e-9
    # stale time21_5 (older than 144 h) gives an empty set.
    old = pd.DataFrame({"origin": pd.to_datetime(["2026-06-20T00:00:00Z"] * 2, utc=True),
                        "h": [1, 2]})
    out_old = DBM.add_dbm_features(old, A)
    assert float(out_old["dbm_n6d"].iloc[0]) == 0.0
    assert float(out_old["dbm_phit"].iloc[0]) == 0.0
    assert bool(out_old["dbm_varr"].isna().all())
    assert bool(out_old["dbm_dt_arr"].isna().all())


def test_gamma_w_fixed_pre_cutoff_only():
    DBM = _load(EXP, "dbm_lib", "d148_dbm_d")
    SP = _load(EXP, "spec_s3", "d148_s3_spec_d")
    lib_src = (EXP / "dbm_lib.py").read_text()
    assert "cutoff_iso" in lib_src and "build_linked_pairs" in lib_src
    assert "calibrate_gamma_w" in lib_src
    # calibration path never names official-window months.
    for bad in ("2026-06-", "2026-07-", "2026-08-", "2026-09-"):
        assert bad not in lib_src, bad
    t21, v0, arr, dt = DBM.build_linked_pairs(SP.CUTOFF_ISO)
    assert len(t21) == int(SP.N_CALIB_PAIRS) == 247
    cut = pd.Timestamp(SP.CUTOFF_ISO, tz="UTC")
    assert bool((pd.to_datetime(t21).tz_localize("UTC") < cut).all())
    assert bool((pd.to_datetime(arr).tz_localize("UTC") < cut).all())
    g_star, w_star, rows = DBM.calibrate_gamma_w(SP.CUTOFF_ISO)
    assert abs(g_star - SP.GAMMA0) < 1e-15 and abs(w_star - SP.W0) < 1e-9
    assert len(rows) == len(SP.GAMMA_GRID) * len(SP.W_GRID) == 9


def test_loader_reads_allowed_keys_only():
    run_src = (EXP / "run_s3.py").read_text()
    lib_src = (EXP / "dbm_lib.py").read_text()
    assert "submissionTime" in lib_src and "time21_5" in lib_src
    assert "submissionTime" in run_src and "time21_5" in run_src
    assert "isMostAccurate" not in run_src and "versionId" not in run_src
    assert "isMostAccurate" not in lib_src and "versionId" not in lib_src


def test_poison_probe_invariance_and_positive_control():
    DBM = _load(EXP, "dbm_lib", "d148_dbm_e")
    rows = [("2026-05-31T23:00:00Z", "2026-06-01T00:00:00Z", 500.0, 20.0, 0.0, 10.0),
            ("2026-06-04T23:00:00Z", "2026-06-05T00:00:00Z", 800.0, 30.0, 5.0, -5.0),
            ("2026-06-19T23:00:00Z", "2026-06-20T00:00:00Z", 900.0, 25.0, 0.0, 0.0)]
    sub = np.array([pd.Timestamp(s, tz="UTC").value for s, _, _, _, _, _ in rows])
    t21 = np.array([pd.Timestamp(t, tz="UTC").value for _, t, _, _, _, _ in rows])
    spd = np.array([r[2] for r in rows])
    ha = np.array([r[3] for r in rows])
    la = np.array([r[4] for r in rows]) * np.pi / 180.0
    lo = np.array([r[5] for r in rows]) * np.pi / 180.0
    off = np.arccos(np.clip(np.cos(la) * np.cos(lo), -1.0, 1.0)) * 180.0 / np.pi
    order = np.argsort(t21, kind="stable")
    A = {"sub": sub[order], "t21": t21[order], "speed": spd[order],
         "half": ha[order], "off": off[order]}
    origins = pd.date_range("2026-06-01", "2026-06-25", freq="24h", tz="UTC")
    rec = DBM.dbm_poison_probe(origins, "2026-06-08T00:00:00Z", A, shift_h=72.0)
    assert rec["passed"] is True and rec["pre_invariant"] is True
    assert rec["moved"] > 0 and rec["n_pre_rows"] > 0


def test_real_store_loads_vintage_clean():
    DBM = _load(EXP, "dbm_lib", "d148_dbm_f")
    A = DBM.load_analyses()
    assert len(A["t21"]) > 4000
    assert bool((np.diff(A["t21"]) >= 0).all())
    assert bool((A["speed"] > 0).all()) and bool((A["half"] > 0).all())


def test_seed_actually_propagates():
    DL = _load(D142, "depth_lib", "d142_dl_ref_s3")
    run_src = (EXP / "run_s3.py").read_text()
    assert "fit_seed_models" in run_src and "event_seed" in run_src
    assert "level_seeds" in run_src or "level" in run_src
    blocks = [set(DL.level_seeds(s)) for s in (0, 1, 2)]
    assert len(set().union(*blocks)) == 9
    for s in (0, 1, 2):
        assert DL.event_seed(s) in DL.level_seeds(s)


def test_output_frame_schema():
    DL = _load(D142, "depth_lib", "d142_dl_schema_s3")
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = DL.frame_predictions(te, np.arange(72, dtype=float))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert len(fr) == 72 and fr["horizon_hours"].tolist() == list(range(1, 73))
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_scratch_isolation_and_cpu():
    import re
    SP = _load(EXP, "spec_s3", "d148_s3_spec_e")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1e3"
    assert len(set(re.findall(r"worker-[a-z0-9]+", code))) == 1
    assert "worker-t1e3" in code
    run_src = (EXP / "run_s3.py").read_text()
    assert 'device") == "cpu"' in run_src and "n_jobs" in run_src
    assert "--gpu" not in code


def test_poison_covers_new_inputs():
    run_src = (EXP / "run_s3.py").read_text()
    assert "poison_test" in run_src and "dbm_poison_probe" in run_src
    assert '"passed"' in run_src and "poison.json" in run_src


def test_no_forbidden_leak_or_banned_source():
    spec_src = (EXP / "spec_s3.py").read_text()
    run_src = (EXP / "run_s3.py").read_text()
    lib_src = (EXP / "dbm_lib.py").read_text()
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
