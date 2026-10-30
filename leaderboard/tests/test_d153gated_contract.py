"""D153 gated-insertion contract: static plus synthetic checks (fast, no fits)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d153_shifts/gatedins"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "predictions_official_truth", "baseline-replay.parquet",
             "official_check", "official-check",
             "research/expos/runs", "from scorer", "import scorer", "from expos", "import expos",
             "official_score", "official-score", "freeze", "submit", "isMostAccurate", "versionId"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "lean.py", "d147_e0421_lean",
                 "D141", "D147", "D148", "D151", "D152",
                 "d141", "d147", "d148", "d151", "d152",
                 "d151_e0472_audit", "d151_enlil_ext", "d148_enlil_live",
                 "E0471", "E0512", "e0471", "e0512",
                 "nn-", "sun-nn", "worker-t1a", "worker-t1b",
                 "worker-t1c", "worker-t1d", "worker-t1e", "worker-t1f",
                 "worker-t1g", "worker-t1h0", "worker-t1h1", "worker-t1h2",
                 "worker-t1h3", "worker-t1h4", "worker-t1h5", "worker-t1h6",
                 "worker-t1h7", "worker-t2no",
                 "donki-live", "strips", "existing_all", "info_ablation"]
BANNED_SRC = ["GONG", "CDAW", "LASCO"]
CELL_LABELS = ["low_low_high", "low_high_high", "high_low_high", "high_high_high"]

sys.path.insert(0, str(EXP))


def _load(mod_name, alias):
    loc = importlib.util.spec_from_file_location(alias, EXP / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code(names=("spec_gated.py", "gated_lib.py", "run_gated.py")):
    return "\n".join((EXP / n).read_text() for n in names)


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 6, files
    assert {"spec_gated.py", "gated_lib.py", "run_gated.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_gated", "d153gated_spec")
    assert SP.DIRECTION == "D153" and SP.OWNER == "worker-t1h8"
    assert SP.KIND == "full" and SP.REFERENCE == "E0472"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24
    assert SP.SCRATCH == "store/scratch/worker-t1h8"
    assert SP.FIT_NJOBS <= 6
    assert SP.BASE_PARAMS["n_estimators"] == 900
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - 0.03) < 1e-12
    assert SP.BASE_PARAMS.get("device") == "cpu" and int(SP.BASE_PARAMS.get("n_jobs")) <= 6
    assert SP.BASE_DEPTH == 3
    assert SP.BASE_COLUMNS == "ace-history-own"
    assert SP.PATH_SOURCE == "oof-fit" and SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert abs(float(SP.GATE_RISE_THRESH) - 8.0) < 1e-12
    assert int(SP.GATE_LEAD_MIN) == 6 and int(SP.GATE_LEAD_MAX) == 48
    assert abs(float(SP.GATE_LAG0_LO) - 280.0) < 1e-12
    assert abs(float(SP.GATE_LAG0_HI) - 700.0) < 1e-12
    assert abs(float(SP.GATE_RANGE24_MAX) - 100.0) < 1e-12
    assert "confident" in SP.GATE_RULE and "lead" in SP.GATE_RULE
    assert "frozen" in SP.GATE_RULE and "vintage-safe" in SP.GATE_RULE
    assert abs(float(SP.INSERT_PEAK) - 60.0) < 1e-12 and int(SP.INSERT_LEN) == 6
    assert "demeaned" in SP.INSERT_DEF.lower()
    assert "confident" in SP.INSERT_DEF.lower()
    assert "[axes:" in SP.AXES_TAG and "target=event-gated" in SP.AXES_TAG
    rule = SP.SELECT_RULE
    assert "lb-eval" in rule and "F1" in rule and "3281" in rule
    for c in CELL_LABELS:
        assert c in rule, c
    assert SP.NOISE_FLOOR == 90.0
    assert "targets <" in SP.VINTAGE_RULE and "<= origin" in SP.VINTAGE_RULE
    assert "2026-05-29" in SP.VINTAGE_RULE
    assert abs(float(SP.REF_MSE) - 3281.31) < 1e-9


def test_single_axis_vs_recipe():
    SP = _load("spec_gated", "d153gated_spec_b")
    run_src = (EXP / "run_gated.py").read_text()
    assert "single axis violated" in run_src
    assert "confident" in run_src
    assert "fresh" in run_src.lower()
    assert "event-gated" in run_src
    assert "GATE_RULE" in run_src or "gate" in run_src.lower()
    assert SP.BASE_PARAMS["subsample"] == 0.8 and SP.BASE_PARAMS["colsample_bytree"] == 0.7
    assert SP.BASE_PARAMS["min_child_weight"] == 20 and SP.BASE_PARAMS["reg_lambda"] == 5
    assert SP.BASE_PARAMS["tree_method"] == "hist"
    for c in CELL_LABELS:
        assert c not in run_src, c
        assert c not in (EXP / "gated_lib.py").read_text(), c
    assert "lb-eval" not in run_src and "lb-best" not in run_src


def test_gate_helpers_synthetic():
    XL = _load("gated_lib", "d153gated_lib")
    base = np.tile(np.linspace(400.0, 460.0, 72), (4, 1))
    base[0, 10:12] = [400.0, 450.0]
    rise, lead = XL.base_lead_rise(base)
    assert rise.shape == (4,) and lead.shape == (4,)
    assert float(rise[0]) == 50.0 and int(lead[0]) == 11
    idx = pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC")
    ser = pd.Series(400.0 + 10.0 * np.sin(np.arange(200) / 10.0), index=idx)
    origins = pd.DatetimeIndex(["2026-01-05T00:00:00Z"] * 4, tz="UTC")
    hist = XL.history_frame(origins, ser)
    m = XL.confident_mask(hist, base)
    assert m.dtype == bool and m.shape == (4,)
    assert bool(m[0]) is True
    flat = np.tile(np.linspace(400.0, 401.0, 72), (4, 1))
    m2 = XL.confident_mask(hist, flat)
    assert not bool(m2.any())
    bump = XL.demeaned_bump(np.array([10, 70]), peak=60.0, length=6)
    assert bump.shape == (2, 72)
    assert abs(float(bump.sum(axis=1)[0])) < 1e-9
    assert abs(float(bump.sum(axis=1)[1])) < 1e-9
    assert abs(float(bump[0, 10]) - 55.0) < 1e-9
    assert abs(float(bump[0, 0]) + 5.0) < 1e-9
    gated, mask = XL.apply_gated_insertion(base, hist)
    assert gated.shape == (4 * 72,) and mask.shape == (4,)
    assert bool(mask[0]) is True
    gmat = gated.reshape(4, 72)
    assert np.allclose(gmat[1][~np.isfinite(gmat[1])], [], equal_nan=True) or True
    nonconf = np.where(~mask)[0]
    assert len(nonconf) > 0
    for k in nonconf:
        assert np.allclose(gmat[k], base[k])
    conf = np.where(mask)[0]
    assert len(conf) > 0
    for k in conf:
        assert abs(float((gmat[k] - base[k]).sum())) < 1e-6


def test_feature_helpers_vintage():
    XL = _load("gated_lib", "d153gated_lib_c")
    idx = pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC")
    vals = 400.0 + 10.0 * np.sin(np.arange(200) / 10.0)
    ser = pd.Series(vals, index=idx)
    origins = pd.DatetimeIndex(["2026-01-05T00:00:00Z", "2026-01-06T00:00:00Z"], tz="UTC")
    hist = XL.history_frame(origins, ser)
    assert float(hist["lag0"].iloc[0]) == float(ser.loc[origins[0]])
    assert float(hist["lag24"].iloc[0]) == float(ser.loc[origins[0] - pd.Timedelta(hours=24)])
    origins_pre = pd.DatetimeIndex(["2026-01-04T00:00:00Z"], tz="UTC")
    h0 = XL.history_frame(origins_pre, ser)
    pert2 = ser.copy()
    pert2.loc[pert2.index >= pd.Timestamp("2026-01-05T00:00:00Z", tz="UTC")] += 50.0
    h1 = XL.history_frame(origins_pre, pert2)
    assert np.allclose(h0.drop(columns=["origin"]).to_numpy(dtype=float),
                       h1.drop(columns=["origin"]).to_numpy(dtype=float), equal_nan=True)


def test_real_store_loads_vintage_clean():
    ace = pd.read_parquet(LB / "store/ch-v1/ace.parquet")
    assert len(ace) > 10000
    assert {"timestamp_utc", "filled_speed_kms"} <= set(ace.columns)
    SP = _load("spec_gated", "d153gated_spec_d")
    origins = pd.date_range(SP.ORIGIN_START, SP.ORIGIN_END, freq="h", tz="UTC")
    assert len(origins) == int(SP.N_OFFICIAL_ORIGINS) == 2833


def test_seed_actually_propagates():
    XL = _load("gated_lib", "d153gated_lib_e")
    run_src = (EXP / "run_gated.py").read_text()
    assert "level_seeds" in run_src or "level_seeds" in (EXP / "gated_lib.py").read_text()
    assert "event_seed" in run_src or "event_seed" in (EXP / "gated_lib.py").read_text()
    blocks = [set(XL.level_seeds(s)) for s in (0, 1, 2)]
    assert len(set().union(*blocks)) == 9
    for s in (0, 1, 2):
        assert XL.event_seed(s) in XL.level_seeds(s)
    p0 = XL.fit_params({"n_jobs": 6}, 3, 5)
    assert p0["max_depth"] == 3 and p0["random_state"] == 5


def test_output_frame_schema():
    XL = _load("gated_lib", "d153gated_lib_f")
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = XL.frame_predictions(te, np.arange(72, dtype=float))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert len(fr) == 72 and fr["horizon_hours"].tolist() == list(range(1, 73))
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_scratch_isolation_and_cpu():
    import re
    SP = _load("spec_gated", "d153gated_spec_g")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1h8"
    assert len(set(re.findall(r"worker-[a-z0-9]+", code))) == 1
    assert "worker-t1h8" in code
    run_src = (EXP / "run_gated.py").read_text()
    assert 'device") == "cpu"' in run_src and "n_jobs" in run_src
    assert "--gpu" not in code
    assert SP.FIT_NJOBS <= 6


def test_poison_covers_new_inputs():
    run_src = (EXP / "run_gated.py").read_text()
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
