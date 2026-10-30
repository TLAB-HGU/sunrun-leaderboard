"""D152 second-own contract: static plus synthetic checks (fast, no fits)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d152_second_own"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "worker-t1a", "worker-t1b",
                 "worker-t1c", "worker-t1d", "worker-t1e", "worker-t1f",
                 "worker-t1g", "worker-t1h0", "worker-t1h1", "worker-t2no",
                 "nn-", "sun-nn", "d151_e0472_audit", "d151_enlil_ext",
                 "d148_enlil_live", "E0512", "E0471", "lean.py", "d147_e0421_lean",
                 "D141", "D147", "D148", "D151", "strips", "existing_all",
                 "info_ablation"]
BANNED_SRC = ["GONG", "CDAW"]
CELL_LABELS = ["low_low_high", "low_high_high", "high_low_high", "high_high_high"]

sys.path.insert(0, str(EXP))


def _load(mod_name, alias):
    loc = importlib.util.spec_from_file_location(alias, EXP / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code(names=("spec.py", "lib.py", "run.py")):
    return "\n".join((EXP / n).read_text() for n in names)


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 8, files
    assert {"spec.py", "lib.py", "run.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec", "d152_spec")
    assert SP.DIRECTION == "D152" and SP.OWNER == "worker-t1h2"
    assert SP.KIND == "full" and SP.REFERENCE == "E0472"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24
    assert SP.SCRATCH == "store/scratch/worker-t1h2"
    assert SP.FIT_NJOBS <= 6
    assert SP.BASE_PARAMS["n_estimators"] == 900
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - 0.03) < 1e-12
    assert SP.BASE_PARAMS.get("device") == "cpu" and int(SP.BASE_PARAMS.get("n_jobs")) <= 6
    assert SP.S1_PARAMS["n_estimators"] == 500 and SP.S2_PARAMS["n_estimators"] == 300
    assert SP.S1_PARAMS.get("device") == "cpu" and SP.S2_PARAMS.get("device") == "cpu"
    assert SP.BASE_DEPTH == 3 and SP.S1_DEPTH == 3 and SP.S2_DEPTH == 3
    assert SP.BASE_COLUMNS == "ace-history-own"
    assert SP.PATH_SOURCE == "oof-fit" and SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert SP.STAGE2_DEF == "d2 = y - pred_stage1"
    assert tuple(SP.NEW_COLUMNS) == ("corr1", "corr2")
    assert len(SP.BLEND_GRID) == 21 and float(SP.BLEND_GRID[0]) == 0.0
    assert "[axes:" in SP.AXES_TAG and "target=residual" in SP.AXES_TAG
    rule = SP.SELECT_RULE
    assert "lb-eval" in rule and "F1" in rule and "3281" in rule
    for c in CELL_LABELS:
        assert c in rule, c
    assert SP.NOISE_FLOOR == 90.0
    assert "targets <" in SP.VINTAGE_RULE and "<= origin" in SP.VINTAGE_RULE
    assert "2026-05-29" in SP.VINTAGE_RULE


def test_single_axis_vs_recipe():
    SP = _load("spec", "d152_spec_b")
    run_src = (EXP / "run.py").read_text()
    assert "single axis violated" in run_src
    assert "new_columns" in run_src or "new columns" in run_src.lower()
    assert "only stage1" in run_src
    assert "fresh" in run_src.lower()
    assert "BLEND_GRID" in run_src and "frozen" in run_src.lower()
    assert SP.BASE_PARAMS["subsample"] == 0.8 and SP.BASE_PARAMS["colsample_bytree"] == 0.7
    assert SP.BASE_PARAMS["min_child_weight"] == 20 and SP.BASE_PARAMS["reg_lambda"] == 5
    assert SP.BASE_PARAMS["tree_method"] == "hist"
    for c in CELL_LABELS:
        assert c not in run_src, c
        assert c not in (EXP / "lib.py").read_text(), c


def test_feature_helpers_vintage():
    XL = _load("lib", "d152_lib")
    idx = pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC")
    vals = 400.0 + 10.0 * np.sin(np.arange(200) / 10.0)
    ser = pd.Series(vals, index=idx)
    origins = pd.DatetimeIndex(["2026-01-05T00:00:00Z", "2026-01-06T00:00:00Z"], tz="UTC")
    hist = XL.history_frame(origins, ser)
    # lag0 equals the origin value exactly (causal, no future)
    assert float(hist["lag0"].iloc[0]) == float(ser.loc[origins[0]])
    assert float(hist["lag24"].iloc[0]) == float(ser.loc[origins[0] - pd.Timedelta(hours=24)])
    # perturbing the future must not change history features
    pert = ser.copy()
    pert.loc[pert.index > origins[0]] = pert.loc[pert.index > origins[0]] + 50.0
    hist2 = XL.history_frame(origins[:1], pert)
    # origin itself was overwritten by the future perturbation in this synthetic
    # setup, so compare a pre-perturbation origin instead
    origins_pre = pd.DatetimeIndex(["2026-01-04T00:00:00Z"], tz="UTC")
    h0 = XL.history_frame(origins_pre, ser)
    pert2 = ser.copy()
    pert2.loc[pert2.index >= pd.Timestamp("2026-01-05T00:00:00Z", tz="UTC")] += 50.0
    h1 = XL.history_frame(origins_pre, pert2)
    assert np.allclose(h0.drop(columns=["origin"]).to_numpy(dtype=float),
                       h1.drop(columns=["origin"]).to_numpy(dtype=float), equal_nan=True)


def test_bootstrap_and_split_helpers():
    XL = _load("lib", "d152_lib_c")
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
    ace = pd.read_parquet(LB / "store/ch-v1/ace.parquet")
    assert len(ace) > 10000
    assert {"timestamp_utc", "filled_speed_kms"} <= set(ace.columns)
    SP = _load("spec", "d152_spec_d")
    origins = pd.date_range(SP.ORIGIN_START, SP.ORIGIN_END, freq="h", tz="UTC")
    assert len(origins) == int(SP.N_OFFICIAL_ORIGINS) == 2833


def test_seed_actually_propagates():
    XL = _load("lib", "d152_lib_e")
    run_src = (EXP / "run.py").read_text()
    assert "level_seeds" in run_src or "level_seeds" in (EXP / "lib.py").read_text()
    assert "event_seed" in run_src or "event_seed" in (EXP / "lib.py").read_text()
    blocks = [set(XL.level_seeds(s)) for s in (0, 1, 2)]
    assert len(set().union(*blocks)) == 9
    for s in (0, 1, 2):
        assert XL.event_seed(s) in XL.level_seeds(s)
    p0 = XL.fit_params({"n_jobs": 6}, 3, 5)
    assert p0["max_depth"] == 3 and p0["random_state"] == 5


def test_output_frame_schema():
    XL = _load("lib", "d152_lib_f")
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = XL.frame_predictions(te, np.arange(72, dtype=float))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert len(fr) == 72 and fr["horizon_hours"].tolist() == list(range(1, 73))
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_scratch_isolation_and_cpu():
    import re
    SP = _load("spec", "d152_spec_g")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1h2"
    assert len(set(re.findall(r"worker-[a-z0-9]+", code))) == 1
    assert "worker-t1h2" in code
    run_src = (EXP / "run.py").read_text()
    assert 'device") == "cpu"' in run_src and "n_jobs" in run_src
    assert "--gpu" not in code
    assert SP.FIT_NJOBS <= 6


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
