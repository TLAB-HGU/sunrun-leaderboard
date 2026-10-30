"""D151 ENLIL extension contract: static plus synthetic checks (fast, no fits)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d151_enlil_ext"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "worker-t1a", "worker-t1b",
                 "worker-t1c", "worker-t1d", "worker-t1e", "worker-t1f",
                 "worker-t1g", "worker-t2no", "nn-", "sun-nn", "d151_e0472_audit",
                 "E0512", "E0471", "lean.py", "d147_e0421_lean"]
BANNED_SRC = ["GONG", "CDAW"]
CELL_LABELS = ["low_low_high", "low_high_high", "high_low_high", "high_high_high"]
PINNED_W = {"1-6": {"e0003_line": 0.21536233464375223, "d33_hsplit": 0.3899845818930272,
                    "d45_lvlmean": 0.3946530834632206},
            "7-24": {"e0003_line": 0.3152013131693896, "d33_hsplit": 0.3436571856823209,
                     "d45_lvlmean": 0.3411415011482895},
            "25-48": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
                      "d45_lvlmean": 0.3333333333333333},
            "49-72": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
                      "d45_lvlmean": 0.3333333333333333}}

sys.path.insert(0, str(EXP))


def _load(mod_name, alias):
    loc = importlib.util.spec_from_file_location(alias, EXP / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code(names=("spec.py", "ext_lib.py", "run.py")):
    return "\n".join((EXP / n).read_text() for n in names)


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 8, files
    assert {"spec.py", "ext_lib.py", "run.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec", "d151_spec")
    assert SP.DIRECTION == "D151" and SP.OWNER == "worker-t1h0"
    assert SP.KIND == "full" and SP.REFERENCE == "E0472"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24
    assert SP.SCRATCH == "store/scratch/worker-t1h0"
    assert SP.FIT_NJOBS <= 6
    assert SP.BASE_PARAMS["n_estimators"] == 900
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - 0.03) < 1e-12
    assert SP.BASE_PARAMS.get("device") == "cpu" and int(SP.BASE_PARAMS.get("n_jobs")) <= 6
    assert SP.BASE_DEPTH == 3 and SP.WEIGHTS == PINNED_W
    assert SP.K == 15 and SP.SCENARIO_DIM == 96
    assert SP.PATH_SOURCE == "oof-fit" and SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert tuple(SP.ARMS) == ("A", "B", "C", "D", "E", "F")
    assert set(SP.ARM_COLS["A"]) == {"enlil_next_arr_latest_h", "enlil_n_runs", "enlil_revision_h"}
    assert set(SP.ARM_COLS["B"]) == {"enlil_dur_h", "enlil_end_h"}
    assert set(SP.ARM_COLS["C"]) == {"enlil_cme_speed", "enlil_cme_halfangle"}
    assert set(SP.ARM_COLS["D"]) == {"enlil_second_arr_h", "enlil_kp18", "enlil_minor"}
    assert set(SP.ARM_COLS["E"]) == {"enlil_corr_arr_h", "enlil_corr_dt", "enlil_drag_fast",
                                     "enlil_dbm_disagree"}
    assert "enlil_corr_arr_h" in SP.ARM_COLS["F"] and "enlil_dur_h" in SP.ARM_COLS["F"]
    assert "[axes:" in SP.AXES_TAG and "enlil-ext" in SP.AXES_TAG
    rule = SP.SELECT_RULE
    assert "lb-eval" in rule and "F1" in rule and "3281" in rule
    for c in CELL_LABELS:
        assert c in rule, c
    assert SP.NOISE_FLOOR == 90.0
    assert "modelCompletionTime" in SP.VINTAGE_RULE and "CME" in SP.VINTAGE_RULE


def test_single_axis_vs_recipe():
    SP = _load("spec", "d151_spec_b")
    run_src = (EXP / "run.py").read_text()
    assert "lean_enlil" in run_src and "s3_std" in run_src
    assert "single axis violated" in run_src
    assert "modelCompletionTime" in run_src
    assert "new_columns" in run_src or "new columns" in run_src.lower()
    assert SP.BASE_PARAMS["subsample"] == 0.8 and SP.BASE_PARAMS["colsample_bytree"] == 0.7
    assert SP.BASE_PARAMS["min_child_weight"] == 20 and SP.BASE_PARAMS["reg_lambda"] == 5
    assert SP.BASE_PARAMS["tree_method"] == "hist"
    for c in CELL_LABELS:
        assert c not in run_src, c
        assert c not in (EXP / "ext_lib.py").read_text(), c


def test_arm_helpers_vintage():
    XL = _load("ext_lib", "d151_lib")
    runs = pd.DataFrame({
        "done": pd.to_datetime(["2026-06-01T00:00:00Z", "2026-06-02T00:00:00Z"], utc=True),
        "arrival": pd.to_datetime(["2026-06-02T00:00:00Z", "2026-06-03T12:00:00Z"], utc=True),
        "kp": [5.0, 7.0], "glancing": [True, False],
        "duration": [20.0, np.nan], "speed": [500.0, 800.0], "halfAngle": [25.0, 35.0],
        "kp18": [np.nan, np.nan], "minor": [False, True],
        "key": ["K1", "K1"], "sim": ["S1", "S2"]})
    # A: same CME group, latest vs earliest
    ta = XL.arm_table(pd.DatetimeIndex(["2026-06-02T00:00:00Z"], tz="UTC"), runs, "A")
    assert float(ta["enlil_n_runs"].iloc[0]) == 2.0
    assert np.isfinite(float(ta["enlil_next_arr_latest_h"].iloc[0]))
    assert np.isfinite(float(ta["enlil_revision_h"].iloc[0]))
    # B keeps NaN (no median imputation) for the nearest run when missing
    runs_b = runs.copy()
    # nearest to 2026-06-01T12Z is first run (dur 20)
    tb = XL.arm_table(pd.DatetimeIndex(["2026-06-01T12:00:00Z"], tz="UTC"), runs_b, "B")
    assert float(tb["enlil_dur_h"].iloc[0]) == 20.0
    assert np.isfinite(float(tb["enlil_end_h"].iloc[0]))
    # C speed / halfAngle of nearest
    tc = XL.arm_table(pd.DatetimeIndex(["2026-06-01T12:00:00Z"], tz="UTC"), runs, "C")
    assert float(tc["enlil_cme_speed"].iloc[0]) == 500.0
    assert float(tc["enlil_cme_halfangle"].iloc[0]) == 25.0
    # D second pending needs 2 distinct CMEs
    runs_d = runs.copy()
    runs_d.loc[1, "key"] = "K2"
    td = XL.arm_table(pd.DatetimeIndex(["2026-06-02T00:00:00Z"], tz="UTC"), runs_d, "D")
    assert np.isfinite(float(td["enlil_second_arr_h"].iloc[0]))
    # vintage: origin before any run sees NaN / zero
    t_old = XL.arm_table(pd.DatetimeIndex(["2026-05-01T00:00:00Z"], tz="UTC"), runs, "A")
    assert bool(pd.isna(t_old["enlil_next_arr_latest_h"].iloc[0]))
    assert float(t_old["enlil_n_runs"].iloc[0]) == 0.0
    # drag proxy is finite for fast CME and larger for faster speed
    assert np.isfinite(float(XL.dbm_disagree_h(800.0)))
    assert float(XL.dbm_disagree_h(800.0)) > float(XL.dbm_disagree_h(500.0))


def test_pairs_and_blocked_oos_synthetic():
    XL = _load("ext_lib", "d151_lib_b")
    # synthetic pairs with a learnable bias: err = 0.01*(speed-500)
    n = 60
    done = pd.date_range("2026-01-01", periods=n, freq="12h", tz="UTC")
    speed = np.linspace(300, 900, n)
    err = 0.01 * (speed - 500.0)
    pairs = pd.DataFrame({"done": done, "est": done + pd.Timedelta("24h"),
                          "obs": done + pd.Timedelta("24h") + pd.to_timedelta(err, unit="h"),
                          "err": err, "speed": speed, "halfAngle": 30.0,
                          "kp": 5.0, "duration": 20.0, "lead": 24.0})
    oos = XL.blocked_oos_mae(pairs, n_blocks=3)
    assert oos["n"] == n and oos["blocks"] == 3
    assert np.isfinite(oos["mae_before"]) and np.isfinite(oos["mae_after"])


def test_bootstrap_and_split_helpers():
    XL = _load("ext_lib", "d151_lib_c")
    rng = np.random.default_rng(0)
    n_org, per = 50, 72
    y = rng.normal(400, 30, n_org * per)
    base = y + rng.normal(0, 10, n_org * per)
    pred = y + rng.normal(0, 9, n_org * per)
    origins = pd.date_range("2026-03-04", periods=n_org, freq="3h", tz="UTC")
    boot = XL.paired_bootstrap_72h(y, base, pred, origins, block_origins=24, B=50, seed=7)
    assert np.isfinite(boot["mean"]) and np.isfinite(boot["lo"]) and np.isfinite(boot["hi"])
    assert boot["lo"] <= boot["mean"] <= boot["hi"] or np.isfinite(boot["mean"])
    rep = XL.oof_split_report(y, base, pred, cap=600.0)
    assert rep["n"] == n_org * per and rep["n_ordinary"] + rep["n_extreme"] == rep["n"]
    assert np.isfinite(rep["mse_pred_ordinary"])


def test_real_store_loads_vintage_clean():
    XL = _load("ext_lib", "d151_lib_d")
    runs = XL.load_runs()
    assert len(runs) > 100
    assert bool((runs["done"].diff().dropna() >= pd.Timedelta(0)).all())
    assert runs["done"].max() >= pd.Timestamp("2026-06-01T00:00:00Z", tz="UTC")
    assert {"done", "arrival", "kp", "glancing", "duration", "speed",
            "halfAngle", "key", "sim"} <= set(runs.columns)


def test_seed_actually_propagates():
    XL = _load("ext_lib", "d151_lib_e")
    run_src = (EXP / "run.py").read_text()
    assert "level_seeds" in run_src or "level_seeds" in (EXP / "ext_lib.py").read_text()
    blocks = [set(XL.level_seeds(s)) for s in (0, 1, 2)]
    assert len(set().union(*blocks)) == 9
    for s in (0, 1, 2):
        assert XL.event_seed(s) in XL.level_seeds(s)
    p0 = XL.fit_params({"n_jobs": 6}, 3, 5)
    assert p0["max_depth"] == 3 and p0["random_state"] == 5


def test_output_frame_schema():
    XL = _load("ext_lib", "d151_lib_f")
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = XL.frame_predictions(te, np.arange(72, dtype=float))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert len(fr) == 72 and fr["horizon_hours"].tolist() == list(range(1, 73))
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_scratch_isolation_and_cpu():
    import re
    SP = _load("spec", "d151_spec_g")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1h0"
    assert len(set(re.findall(r"worker-[a-z0-9]+", code))) == 1
    assert "worker-t1h0" in code
    run_src = (EXP / "run.py").read_text()
    assert 'device") == "cpu"' in run_src and "n_jobs" in run_src


def test_poison_covers_new_inputs():
    run_src = (EXP / "run.py").read_text()
    assert "_new_input_probe" in run_src
    assert "scenario_poison" in run_src and "frame_poison" in run_src
    assert '"passed"' in run_src and "poison.json" in run_src


def test_no_forbidden_leak_or_banned_source():
    code = _code()
    for w in FORBIDDEN:
        assert w not in code, w
    for w in OTHER_CONTACT:
        assert w not in code, w
    low = code.lower()
    for w in BANNED_SRC:
        assert w.lower() not in low, w
    assert "isMostAccurate" not in code
    assert "cuda" not in low
