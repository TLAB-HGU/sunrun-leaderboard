"""D153 lagshift contract: static plus synthetic checks (fast, no fits)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d153_shifts/lagshift"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs", "from scorer", "import scorer",
             "from expos", "import expos", "isMostAccurate",
             "official_score", "official-score", "--gpu", "cuda",
             "freeze", "submit", "upload", "push", "lb-best"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "lean.py",
                 "D141", "D147", "D148", "D151", "D152",
                 "d141", "d147", "d148", "d151", "d152",
                 "d151_e0472_audit", "d151_enlil_ext", "d148_enlil",
                 "E0471", "E0472", "e0471", "e0472", "E0512",
                 "nn-", "sun-nn", "worker-t1a", "worker-t1b",
                 "worker-t1c", "worker-t1d", "worker-t1e",
                 "worker-t1f", "worker-t1g",
                 "worker-t1h0", "worker-t1h1", "worker-t1h2",
                 "worker-t1h3", "worker-t1h4", "worker-t1h5",
                 "worker-t1h6", "worker-t2no", "donki-live"]
BANNED_SRC = ["GONG", "CDAW", "LASCO"]
CELL_LABELS = ["low_low_high", "low_high_high", "high_low_high", "high_high_high",
               "hlh", "llh", "lhh", "hhh"]
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


def _code(names=("spec_lagshift.py", "lagshift_lib.py", "run_lagshift.py")):
    return "\n".join((EXP / n).read_text() for n in names)


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 6, files
    assert {"spec_lagshift.py", "lagshift_lib.py", "run_lagshift.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_lagshift", "d153lagshift_spec")
    assert SP.DIRECTION == "D153" and SP.OWNER == "worker-t1h9"
    assert SP.KIND == "full" and SP.REFERENCE == "E0408"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.ORIGIN_START == "2026-05-31 23:00" and SP.ORIGIN_END == "2026-09-26 23:00"
    assert int(SP.N_OFFICIAL_ORIGINS) == 2833
    assert SP.TRAIN_FREQ == "3h" and SP.OOF_FREQ == "3h" and int(SP.SPLIT_H) == 24
    assert int(SP.PURGE_H) == 72
    assert SP.OOF_TRAIN_END_ISO == "2026-03-01T00:00:00Z"
    assert SP.OOF_START_ISO == "2026-03-04T00:00:00Z"
    assert SP.OOF_END_ISO == "2026-05-28T23:00:00Z"
    assert SP.MAIN_WINDOW == "W2-main"
    assert SP.SCRATCH == "store/scratch/worker-t1h9"
    assert int(SP.N_JOBS) <= 6 and int(SP.FIT_NJOBS) <= 6
    assert int(SP.BASE_DEPTH) == 3
    assert int(SP.BASE_PARAMS.get("n_estimators")) == 900
    assert abs(float(SP.BASE_PARAMS.get("learning_rate")) - 0.03) < 1e-12
    assert SP.BASE_PARAMS.get("device") == "cpu"
    assert int(SP.BASE_PARAMS.get("n_jobs")) <= 6
    assert SP.BASE_COLUMNS == "existing_all+strips17"
    assert SP.WEIGHTS == PINNED_W
    assert int(SP.SHIFT_H) == -2
    assert "OOF" in SP.SHIFT_RULE and "-2" in SP.SHIFT_RULE
    assert "h+2" in SP.SHIFT_DEF or "h + 2" in SP.SHIFT_DEF
    assert SP.PATH_SOURCE == "oof-fit"
    assert "[axes:" in SP.AXES_TAG and "lagshift" in SP.AXES_TAG
    assert "lb-eval" in SP.SELECT_RULE and "F1" in SP.SELECT_RULE and "3281" in SP.SELECT_RULE
    assert "3 seeds" in SP.SELECT_RULE and "four high cells" in SP.SELECT_RULE
    assert abs(float(SP.NOISE_FLOOR) - 90.0) < 1e-12
    assert abs(float(SP.ORDINARY_CAP) - 600.0) < 1e-12
    assert int(SP.BOOT_BLOCK_ORIGINS) == 24 and int(SP.BOOT_B) == 200
    assert "targets <" in SP.VINTAGE_RULE and "<= origin" in SP.VINTAGE_RULE
    assert "2026-05-29" in SP.VINTAGE_RULE
    assert abs(float(SP.REF_MSE) - 3281.31) < 1e-9
    assert "lagshift" in SP.DIFFERENCE and "E0408" in SP.DIFFERENCE
    assert int(SP.SMOKE_ORIGINS) == 8 and int(SP.SMOKE_OFFICIAL) == 8


def test_single_axis_shift_only():
    SP = _load("spec_lagshift", "d153lagshift_spec_b")
    run_src = (EXP / "run_lagshift.py").read_text()
    assert "single axis violated" in run_src
    assert "apply_shift" in run_src
    assert "SHIFT_H" in run_src
    assert "fresh" in run_src.lower()
    assert "no stored prediction" in run_src.lower()
    assert "full_grid" in run_src or "base_grid" in run_src
    assert "info_ablation" in run_src and "strips" in run_src
    assert "existing_all" in run_src
    assert "fit_blend_weight" not in run_src and "blend_dv" not in run_src
    assert SP.BASE_PARAMS["subsample"] == 0.8 and SP.BASE_PARAMS["colsample_bytree"] == 0.7
    assert SP.BASE_PARAMS["min_child_weight"] == 20 and SP.BASE_PARAMS["reg_lambda"] == 5
    assert SP.BASE_PARAMS["tree_method"] == "hist"


def test_shift_helpers_synthetic():
    XL = _load("lagshift_lib", "d153lagshift_lib")
    base = np.tile(np.arange(72, dtype=float)[None, :], (3, 1))
    out = XL.apply_shift(base, -2)
    assert out.shape == (3, 72)
    assert np.allclose(out[:, :-2], base[:, 2:])
    assert np.allclose(out[:, -2:], base[:, -1:])
    try:
        XL.apply_shift(np.zeros((2, 71)), -2)
    except ValueError:
        pass
    else:
        raise AssertionError("non-72 width must be rejected")
    try:
        XL.apply_shift(base, -3)
    except ValueError:
        pass
    else:
        raise AssertionError("non-minus-2 shift must be rejected")
    try:
        XL.apply_shift(np.full((2, 72), np.nan), -2)
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite base must be rejected")
    # shift is a pure reindex: identical input gives identical output
    assert np.array_equal(XL.apply_shift(base, -2), XL.apply_shift(base.copy(), -2))


def test_mixture_and_seed_helpers():
    XL = _load("lagshift_lib", "d153lagshift_lib_b")
    SP = _load("spec_lagshift", "d153lagshift_spec_c")
    blocks = [set(XL.level_seeds(s)) for s in (0, 1, 2)]
    assert len(set().union(*blocks)) == 9
    for s in (0, 1, 2):
        assert XL.event_seed(s) in XL.level_seeds(s)
    p = XL.fit_params(dict(SP.BASE_PARAMS), 3, 4)
    assert p["max_depth"] == 3 and p["random_state"] == 4 and p["n_estimators"] == 900
    XL.check_weights(SP.WEIGHTS)
    rng = np.random.default_rng(0)
    h = np.tile(np.arange(1, 73), 4)
    mem = {m: rng.normal(400, 20, len(h)) for m in ("e0003_line", "d33_hsplit", "d45_lvlmean")}
    out = XL.blend_per_lead(mem, SP.WEIGHTS, h)
    assert np.isfinite(out).all() and len(out) == len(h)
    try:
        XL.check_weights({"1-6": SP.WEIGHTS["1-6"]})
    except ValueError:
        pass
    else:
        raise AssertionError("partial weights must be rejected")


def test_bootstrap_and_split_helpers():
    XL = _load("lagshift_lib", "d153lagshift_lib_d")
    rng = np.random.default_rng(1)
    n_org, per = 48, 72
    y = rng.normal(400, 30, n_org * per)
    base = y + rng.normal(0, 10, n_org * per)
    pred = y + rng.normal(0, 9, n_org * per)
    origins = pd.date_range("2026-03-04", periods=n_org, freq="3h", tz="UTC")
    boot = XL.paired_bootstrap_72h(y, base, pred, origins, block_origins=24, B=50, seed=7)
    assert np.isfinite(boot["mean"]) and np.isfinite(boot["lo"]) and np.isfinite(boot["hi"])
    assert boot["blocks"] == 2
    rep = XL.oof_split_report(y, base, pred, cap=600.0)
    assert rep["n"] == n_org * per and rep["n_ordinary"] + rep["n_extreme"] == rep["n"]
    assert np.isfinite(rep["mse_pred_ordinary"])


def test_poison_synthetic():
    XL = _load("lagshift_lib", "d153lagshift_lib_e")
    idx = pd.date_range("2026-04-01T00:00:00Z", "2026-07-01T00:00:00Z", freq="h", tz="UTC")
    filled = pd.Series(400.0 + np.sin(np.arange(len(idx)) / 24.0), index=idx)
    cut = "2026-05-29T00:00:00Z"
    origins = pd.DatetimeIndex(["2026-05-01T00:00:00Z", "2026-05-20T00:00:00Z",
                                "2026-06-05T00:00:00Z", "2026-06-10T00:00:00Z"], tz="UTC")
    sc = XL.scenario_poison(origins, filled, cut)
    assert sc["passed"] is True and sc["pre_invariant"] is True and sc["moved"] is True
    assert sc["n_pre_rows"] == 2 and sc["n_post_rows"] == 2

    def _frame_for(o, series):
        return np.tile(series.reindex(o, method="ffill").to_numpy(dtype=float)[:, None], (1, 4))

    fp = XL.frame_poison(origins, filled, cut, _frame_for)
    assert fp["passed"] is True and fp["pre_invariant"] is True


def test_real_store_loads_vintage_clean():
    ace = pd.read_parquet(LB / "store/ch-v1/ace.parquet")
    assert len(ace) > 10000
    assert {"timestamp_utc", "filled_speed_kms"} <= set(ace.columns)
    SP = _load("spec_lagshift", "d153lagshift_spec_d")
    origins = pd.date_range(SP.ORIGIN_START, SP.ORIGIN_END, freq="h", tz="UTC")
    assert len(origins) == int(SP.N_OFFICIAL_ORIGINS) == 2833
    assert (origins > pd.Timestamp(SP.CUTOFF_ISO, tz="UTC")).all()
    oof_start = pd.Timestamp(SP.OOF_START_ISO, tz="UTC")
    oof_end = pd.Timestamp(SP.OOF_END_ISO, tz="UTC")
    assert oof_end < pd.Timestamp(SP.CUTOFF_ISO, tz="UTC")
    assert (oof_start - pd.Timestamp(SP.OOF_TRAIN_END_ISO, tz="UTC")) == pd.Timedelta(hours=72)


def test_seed_actually_propagates():
    run_src = (EXP / "run_lagshift.py").read_text()
    assert "fit_seed_models" in run_src and "event_seed" in run_src
    assert "level_seeds" in run_src and "fit_params" in run_src


def test_output_frame_schema():
    XL = _load("lagshift_lib", "d153lagshift_lib_f")
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = XL.frame_predictions(te, np.arange(72, dtype=float))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert len(fr) == 72 and fr["horizon_hours"].tolist() == list(range(1, 73))
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_full_outputs_present():
    run_src = (EXP / "run_lagshift.py").read_text()
    code = _code()
    for w in ("predictions_official_s",
              "poison.json", "selection.json", "meta.json"):
        assert w in run_src, w
    for w in ("origin_last_input_utc", "pred_kms", "frame_predictions"):
        assert w in code, w
    for w in ("scenario_poison", "frame_poison", "poison_test", '"passed"'):
        assert w in run_src, w


def test_scratch_isolation_and_cpu():
    import re
    SP = _load("spec_lagshift", "d153lagshift_spec_g")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1h9"
    assert len(set(re.findall(r"worker-[a-z0-9]+", code))) == 1
    assert "worker-t1h9" in code
    run_src = (EXP / "run_lagshift.py").read_text()
    assert 'device") == "cpu"' in run_src and "n_jobs" in run_src
    assert "--gpu" not in code
    assert SP.FIT_NJOBS <= 6 and int(SP.N_JOBS) <= 6


def test_vintage_and_recipe():
    SP = _load("spec_lagshift", "d153lagshift_spec_h")
    run_src = (EXP / "run_lagshift.py").read_text()
    assert "targets <" in SP.VINTAGE_RULE and "<= origin" in SP.VINTAGE_RULE
    assert "target_t" in run_src and "cutoff" in run_src.lower()
    assert "PURGE" in run_src or "purge" in run_src.lower()
    assert "paired_bootstrap_ragged" in run_src and "block_origins" in run_src
    assert "shift_flat" in run_src and "apply_shift" in run_src
    assert "select_slots" in run_src and "strip_features" in run_src
    assert "strip_columns" in run_src and "existing_all" in run_src
    assert "full_grid" in run_src or "base_grid" in run_src


def test_no_forbidden_leak_or_banned_source():
    code = _code()
    for w in FORBIDDEN:
        assert w not in code, w
    for w in OTHER_CONTACT:
        assert w not in code, w
    low = code.lower()
    for w in BANNED_SRC:
        assert w.lower() not in low, w
    for w in CELL_LABELS:
        assert w not in low, w
    assert "cuda" not in low and "--gpu" not in code
def test_shift_flat_and_ragged_bootstrap_synthetic():
    XL = _load("lagshift_lib", "d153lagshift_lib_g")
    oo = np.array(["2026-03-04T00:00:00Z"] * 5 + ["2026-03-04T03:00:00Z"] * 4)
    hh = np.array([1, 2, 3, 71, 72, 1, 2, 70, 72])
    bb = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 11.0, 22.0, 33.0, 44.0])
    sf = XL.shift_flat(oo, hh, bb, -2)
    assert list(sf[:3]) == [30.0, 30.0, 30.0]
    assert list(sf[3:5]) == [50.0, 50.0]
    assert list(sf[5:]) == [22.0, 22.0, 44.0, 44.0]
    try:
        XL.shift_flat(oo, hh, bb, -3)
    except ValueError:
        pass
    else:
        raise AssertionError("non-minus-2 flat shift must be rejected")
    rng = np.random.default_rng(3)
    n = 100
    origins = pd.date_range("2026-03-04", periods=10, freq="3h", tz="UTC")
    rows = rng.choice(np.arange(10), size=n)
    y = rng.normal(400, 30, n)
    base = y + rng.normal(0, 10, n)
    pred = y + rng.normal(0, 9, n)
    rb = XL.paired_bootstrap_ragged(y, base, pred, origins[rows].to_numpy(),
                                    block_origins=4, B=50, seed=7)
    assert np.isfinite(rb["mean"]) and np.isfinite(rb["lo"]) and np.isfinite(rb["hi"])
    assert rb["blocks"] == 3
