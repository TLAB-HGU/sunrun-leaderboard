"""D149 ML-DBM contract: static + synthetic checks (fast, no fits).

No real data fits, no arm registration, no official reads. Covers the owned
file budget, spec identity (D149/worker-t1f6/full/E0472), the single-axis
rule (reference recipe identical plus ML-DBM arrival columns only), DBM
physics, forest block validation on synthetic pairs, vintage enforcement
on synthetic cases, poison coverage of the new ML-DBM input, CPU/n_jobs
budget, seed propagation, output schema, scratch isolation, and
forbidden-source absence.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d149_enlil_cells/mldbm"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "lean.py", "D147", "d147",
                 "d148_enlil", "nn-", "sun-nn", "E0471", "e0471",
                 "worker-t1a", "worker-t1b", "worker-t1c", "worker-t1d",
                 "worker-t1e", "worker-t1f0", "worker-t1f1", "worker-t1f2",
                 "worker-t1f3", "worker-t1f4", "worker-t1f5",
                 "worker-d141", "worker-d147", "worker-d76", "worker-t2no"]
BANNED_SRC = ["GONG", "CDAW"]
POSTHOC = ["isMostAccurate", "versionId"]
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


def _code(names=("spec_mldbm.py", "mldbm_lib.py", "run_mldbm.py")):
    return "\n".join((EXP / n).read_text() for n in names)


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 6, files
    assert {"spec_mldbm.py", "mldbm_lib.py", "run_mldbm.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_mldbm", "d149_mldbm_spec")
    assert SP.DIRECTION == "D149" and SP.OWNER == "worker-t1f6"
    assert SP.KIND == "full" and SP.REFERENCE == "E0472"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24
    assert SP.SCRATCH == "store/scratch/worker-t1f6"
    assert SP.FIT_NJOBS <= 6
    assert SP.BASE_PARAMS["n_estimators"] == 900
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - 0.03) < 1e-12
    assert SP.BASE_PARAMS.get("device") == "cpu" and int(SP.BASE_PARAMS.get("n_jobs")) <= 6
    assert SP.BASE_DEPTH == 3 and SP.WEIGHTS == PINNED_W
    assert float(SP.WINDOW_H) == 144.0
    assert float(SP.L_KM) == 134640320.7
    assert float(SP.GAMMA0) == 5.0e-08 and float(SP.W0) == 450.0
    assert int(SP.CALIB_BLOCKS) == 4
    assert int(SP.RF_N_EST) == 200 and int(SP.RF_DEPTH) == 8
    assert int(SP.RF_NJOBS) <= 6
    assert tuple(SP.MLDBM_COLS) == ("mldbm_n6d", "mldbm_dt_arr", "mldbm_varr",
                                    "mldbm_phit", "mldbm_wrf", "mldbm_env")
    assert tuple(SP.ENLIL_COLS) == ("enlil_next_arr_h", "enlil_since_arr_h", "enlil_n_pending",
                                    "enlil_kp", "enlil_glancing", "enlil_dt_target")
    assert SP.PATH_SOURCE == "oof-fit" and SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert SP.K == 15 and SP.SCENARIO_DIM == 96
    assert SP.MAIN_WINDOW == "W2"
    assert [w["name"] for w in SP.WINDOWS] == ["W0", "W1", "W2"]
    assert "[axes:" in SP.AXES_TAG and "mldbm-arrival" in SP.AXES_TAG
    rule = SP.SELECT_RULE
    assert "lb-eval" in rule and "F1" in rule and "3281" in rule and "0.0833" in rule
    for c in CELL_LABELS:
        assert c in rule, c


def test_single_axis_vs_recipe():
    SP = _load("spec_mldbm", "d149_mldbm_spec_b")
    run_src = (EXP / "run_mldbm.py").read_text()
    for mod in ("info_ablation", "strips", "spec_mldbm", "mldbm_lib", "existing_all"):
        assert mod in run_src, mod
    assert "900" in run_src and "0.03" in run_src
    assert "ENLIL_COLS" in run_src and "enlil_dt_target" in run_src
    assert "MLDBM_COLS" in run_src and "mldbm_dt_arr" in run_src
    assert "single axis violated" in run_src
    assert "residual_multi" in run_src and "pool_scales" in run_src
    assert "fit_blend_weight_dv" in run_src and "blend_dv" in run_src
    assert "RandomForest" in run_src and "block_oos_report" in run_src
    assert SP.BASE_PARAMS["subsample"] == 0.8 and SP.BASE_PARAMS["colsample_bytree"] == 0.7
    assert SP.BASE_PARAMS["min_child_weight"] == 20 and SP.BASE_PARAMS["reg_lambda"] == 5
    assert SP.BASE_PARAMS["tree_method"] == "hist"
    for c in CELL_LABELS:
        assert c not in run_src, c
        assert c not in (EXP / "mldbm_lib.py").read_text(), c


def test_dbm_physics_synthetic():
    ML = _load("mldbm_lib", "d149_mldbm_lib")
    SP = _load("spec_mldbm", "d149_mldbm_spec_c")
    th, va, sd = ML.ensemble_median(np.array([500.0, 800.0]), tuple(SP.GAMMA_ENS), tuple(SP.W_ENS), float(SP.L_KM))
    assert th.shape == (2,) and va.shape == (2,) and sd.shape == (2,)
    assert bool(np.isfinite(th).all() and np.isfinite(va).all())
    assert float(th[0]) > float(th[1])
    t0, v0 = ML.dbm_transit(450.0, 450.0, 5.0e-08, float(SP.L_KM))
    assert abs(v0 - 450.0) < 1e-6 and t0 > 60.0
    t1, _ = ML.dbm_transit(800.0, 450.0, 5.0e-08, float(SP.L_KM))
    assert t1 < t0


def test_rf_block_report_synthetic():
    ML = _load("mldbm_lib", "d149_mldbm_lib_b")
    SP = _load("spec_mldbm", "d149_mldbm_spec_d")
    from sklearn.ensemble import RandomForestRegressor
    rng = np.random.default_rng(0)
    n = 60
    t21 = np.sort(rng.integers(1700000000000000000, 1750000000000000000, n).astype(np.int64))
    v0 = rng.uniform(400, 900, n)
    half = rng.uniform(10, 40, n)
    off = rng.uniform(0, 50, n)
    up_m = rng.normal(450, 30, n)
    up_s = np.abs(rng.normal(15, 5, n))
    dbm_pred, _, _ = ML.ensemble_median(v0, tuple(SP.GAMMA_ENS), tuple(SP.W_ENS), float(SP.L_KM))
    true_dt = dbm_pred + rng.normal(0, 5, n)
    rep = ML.block_oos_report(t21, v0, half, off, true_dt, up_m, up_s,
                              float(SP.GAMMA0), float(SP.W0),
                              tuple(SP.GAMMA_ENS), tuple(SP.W_ENS), float(SP.L_KM),
                              RandomForestRegressor,
                              dict(n_estimators=20, max_depth=4, min_samples_leaf=5,
                                   n_jobs=1, random_state=0), 4)
    assert rep["n_pairs"] == n and rep["n_fit"] > 0 and rep["n_oos"] > 0
    assert np.isfinite(rep["dbm_mae"]) and np.isfinite(rep["rf_mae"])
    assert len(rep["block_dbm_mae"]) > 0 and len(rep["block_rf_mae"]) > 0


def test_vintage_mldbm_values():
    ML = _load("mldbm_lib", "d149_mldbm_lib_c")
    SP = _load("spec_mldbm", "d149_mldbm_spec_e")
    assert "submissionTime" in (EXP / "mldbm_lib.py").read_text()
    assert "modelCompletionTime" in (EXP / "run_mldbm.py").read_text()
    H = 3_600_000_000_000
    t0 = pd.Timestamp("2026-06-01T00:00:00Z").value
    A = {"sub": np.array([t0, t0], dtype=np.int64),
         "t21": np.array([t0, t0 + 48 * H], dtype=np.int64),
         "speed": np.array([600.0, 600.0]), "half": np.array([20.0, 20.0]),
         "off": np.array([5.0, 5.0]), "aid": np.array(["a", "b"], dtype=object)}
    ace_idx = pd.date_range("2026-05-01", "2026-06-20", freq="h", tz="UTC")
    ace = pd.Series(450.0, index=ace_idx)
    grid = pd.DataFrame({"origin": np.repeat(pd.DatetimeIndex(["2026-06-02T00:00:00Z"], tz="UTC").to_numpy(), 72),
                         "h": np.arange(1, 73)})
    out = ML.add_mldbm_features(grid, A, None, ace, SP)
    assert list(out.columns[-6:]) == list(SP.MLDBM_COLS)
    assert bool((out["mldbm_n6d"] == 1.0).all())
    assert bool(np.isfinite(out["mldbm_dt_arr"]).all())
    grid_old = pd.DataFrame({"origin": np.repeat(pd.DatetimeIndex(["2026-05-01T00:00:00Z"], tz="UTC").to_numpy(), 72),
                             "h": np.arange(1, 73)})
    out_old = ML.add_mldbm_features(grid_old, A, None, ace, SP)
    assert bool((out_old["mldbm_n6d"] == 0.0).all())
    assert bool(out_old["mldbm_dt_arr"].isna().all())
    rec = ML.mldbm_poison_probe(pd.DatetimeIndex(["2026-05-01T00:00:00Z", "2026-06-02T00:00:00Z"], tz="UTC"),
                                "2026-05-29T00:00:00Z", A, None, ace, SP)
    assert rec["passed"] is True and rec["pre_invariant"] is True
    assert rec["n_pre_rows"] > 0 and rec["moved"] > 0


def test_real_store_loads_vintage_clean():
    ML = _load("mldbm_lib", "d149_mldbm_lib_d")
    RN = _load("run_mldbm", "d149_mldbm_run")
    ev = RN.enlil_runs()
    assert len(ev) > 100
    assert bool((ev["done"].diff().dropna() >= pd.Timedelta(0)).all())
    assert ev["done"].max() >= pd.Timestamp("2026-06-01T00:00:00Z", tz="UTC")
    A = ML.load_analyses(str(LB / "store/donki-cme"))
    assert len(A["t21"]) > 1000
    t21, v0, half, off, arr, dt = ML.build_linked_pairs(str(LB / "store/donki-cme"), "2026-05-29T00:00:00Z")
    assert len(t21) >= 40
    assert bool((np.diff(t21.astype(np.int64)) >= 0).all())
    assert bool(((dt >= 6.0) & (dt <= 200.0)).all())


def test_seed_actually_propagates():
    ML = _load("mldbm_lib", "d149_mldbm_lib_e")
    run_src = (EXP / "run_mldbm.py").read_text()
    assert "fit_set" in run_src and "event_seed" in run_src
    assert "level_seeds" in run_src
    blocks = [set(ML.level_seeds(s)) for s in (0, 1, 2)]
    assert len(set().union(*blocks)) == 9
    for s in (0, 1, 2):
        assert ML.event_seed(s) in ML.level_seeds(s)


def test_output_frame_schema():
    ML = _load("mldbm_lib", "d149_mldbm_lib_f")
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = ML.frame_predictions(te, np.arange(72, dtype=float))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert len(fr) == 72 and fr["horizon_hours"].tolist() == list(range(1, 73))
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_scratch_isolation_and_cpu():
    import re
    SP = _load("spec_mldbm", "d149_mldbm_spec_f")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1f6"
    assert len(set(re.findall(r"worker-[a-z0-9]+", code))) == 1
    assert "worker-t1f6" in code
    run_src = (EXP / "run_mldbm.py").read_text()
    assert 'device") == "cpu"' in run_src and "n_jobs" in run_src
    assert int(SP.FIT_NJOBS) <= 6 and int(SP.RF_NJOBS) <= 6
    assert int(SP.FIT_NJOBS) + int(SP.RF_NJOBS) <= 20


def test_poison_covers_new_inputs():
    run_src = (EXP / "run_mldbm.py").read_text()
    assert "poison_test" in run_src and "mldbm_poison_probe" in run_src
    assert "enlil_poison_probe" in run_src
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
    for w in POSTHOC:
        assert w not in code, w
    assert "cuda" not in low
    assert "load_truth" not in code


def test_recipe_parity_with_reference():
    SP = _load("spec_mldbm", "d149_mldbm_spec_g")
    sys.path.insert(0, str(LB / "experiments/d143_axisA_analog/a2b"))
    import spec_a2b as RECIPE
    assert tuple(SP.SEEDS) == tuple(RECIPE.SEEDS)
    assert SP.SMOKE_SEED == RECIPE.SMOKE_SEED
    assert SP.CUTOFF_ISO == RECIPE.CUTOFF_ISO
    assert SP.N_OFFICIAL_ORIGINS == RECIPE.N_OFFICIAL_ORIGINS
    assert SP.TRAIN_FREQ == RECIPE.TRAIN_FREQ and SP.SPLIT_H == RECIPE.SPLIT_H
    assert SP.BASE_PARAMS == RECIPE.BASE_PARAMS
    assert SP.BASE_DEPTH == RECIPE.BASE_DEPTH
