"""D150 EVEREST-lite contract: static + synthetic checks (fast, no fits).

No real data fits, no official reads. Covers the owned file budget,
spec identity (D150/worker-t1g4/full), the two pre-registered arms with
the hurdle picked on honest OOF only, the frozen P95 plus 5-percent
event rule, isotonic calibration math plus ECE/Brier, the moment tail
mean, hurdle blending, input-only features on synthetic data, seed
propagation across base plus event streams, readonly OOF source
isolation, CPU/n_jobs budget, scratch isolation, poison-key shape, and
forbidden-source absence.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d150_oofset" / "everest"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "lean.py", "D141", "d141",
                 "D147", "d147", "d148_enlil", "nn-", "sun-nn", "E0471", "E0472", "E0474",
                 "worker-t1a", "worker-t1b", "worker-t1c", "worker-t1d",
                 "worker-t1e", "worker-t1f", "worker-t1g1", "worker-t1g2",
                 "worker-t1g3", "worker-d147", "worker-d141", "worker-t2no"]
BANNED_SRC = ["GONG", "CDAW"]
CELL_LABELS = ["low_low_high", "low_high_high", "high_low_high", "high_high_high", "hlh"]
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


def _code(names=("spec_ev.py", "everest_lib.py", "run_official_everest.py")):
    return "\n".join((EXP / n).read_text() for n in names)


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 6, files
    assert {"spec_ev.py", "everest_lib.py", "run_official_everest.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_ev", "d150ev_spec")
    assert SP.DIRECTION == "D150" and SP.OWNER == "worker-t1g4"
    assert SP.KIND == "full"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.ORIGIN_START == "2026-05-31 23:00" and SP.ORIGIN_END == "2026-09-26 23:00"
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24
    assert SP.SCRATCH == "store/scratch/worker-t1g4"
    assert SP.OOF_DIR == "store/scratch/worker-t1g0/oofset-full"
    assert tuple(SP.OOF_FOLDS) == ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7")
    assert SP.SELECT_SEED == 0 and SP.OOF_NFOLD == 3
    assert tuple(SP.ARMS) == ("control", "hurdle")
    assert tuple(float(x) for x in SP.W_GRID) == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert abs(float(SP.EXTREME_Q) - 0.95) < 1e-12
    assert abs(float(SP.EXTREME_FRAC) - 0.05) < 1e-12 and SP.EXTREME_MIN_FINITE == 36
    assert SP.ECE_BINS == 10
    assert SP.Z_DIM == 36 and SP.Z_PAST_H == 12 and SP.Z_PATH_STEP == 3
    assert SP.PART_SEED == 0 and SP.HEAD_SEED_BASE == 2000
    assert SP.FIT_NJOBS <= 6 and SP.HEAD_NJOBS <= 6
    assert SP.FIT_NJOBS + SP.HEAD_NJOBS <= 20
    assert SP.BASE_PARAMS.get("device") == "cpu" and int(SP.BASE_PARAMS.get("n_jobs")) <= 6
    assert SP.EVENT_PARAMS.get("device") == "cpu" and int(SP.EVENT_PARAMS.get("n_jobs")) <= 6
    assert SP.BASE_PARAMS["n_estimators"] == 900
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - 0.03) < 1e-12
    assert SP.EVENT_PARAMS["n_estimators"] == 80 and SP.EVENT_PARAMS["max_depth"] == 2
    assert SP.BASE_DEPTH == 3 and SP.WEIGHTS == PINNED_W
    assert SP.BASE_COLUMNS == "existing_all+strips17"
    assert SP.PATH_SOURCE == "oof-store-for-select-own-fit-for-official"
    assert SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert SP.EVENT_DEF.startswith("event =") and SP.HURDLE_DEF.startswith("final = base")
    assert "ROUTER_DEF" not in dir(SP) or True
    assert "[axes:" in SP.AXES_TAG and "hurdle-calibrated" in SP.AXES_TAG
    assert "SELECT_RULE" in dir(SP) and "OOF" in SP.SELECT_RULE


def test_weights_match_shared_recipe():
    SP = _load("spec_ev", "d150ev_spec_b")
    loc = importlib.util.spec_from_file_location(
        "d150ev_oof", LB / "experiments/d150_oofset/spec_oof.py")
    oof = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(oof)
    assert SP.WEIGHTS == oof.WEIGHTS
    assert SP.BASE_PARAMS["n_estimators"] == oof.BASE_PARAMS["n_estimators"]
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - float(oof.BASE_PARAMS["learning_rate"])) < 1e-12
    assert SP.BASE_DEPTH == oof.BASE_DEPTH
    assert SP.BASE_COLUMNS == oof.BASE_COLUMNS


def test_single_axis_everest():
    run_src = (EXP / "run_official_everest.py").read_text()
    for mod in ("info_ablation", "strips", "spec_ev", "everest_lib", "existing_all"):
        assert mod in run_src, mod
    for w in ("predictions_official_s{s}.parquet", "event_summary_s{s}.parquet",
              "oof_comparison.json", "poison.json", "meta.json", "fit_event",
              "isotonic_fit", "isotonic_apply", "apply_hurdle", "frame_predictions"):
        assert w in run_src, w
    assert "official-score" not in run_src and "official_score" not in run_src


def test_seed_actually_propagates():
    EV = _load("everest_lib", "d150ev_lib_seed")
    SP = _load("spec_ev", "d150ev_spec_seed")
    params = [EV.fit_params(dict(SP.BASE_PARAMS), SP.BASE_DEPTH, s) for s in SP.SEEDS]
    assert len({p["random_state"] for p in params}) == 3
    assert all(p["max_depth"] == 3 for p in params)
    assert EV.level_seeds(1) == (3, 4, 5) and EV.event_seed(1) == 4
    assert EV.head_seed(1) == 2001
    assert len({EV.head_seed(s) for s in SP.SEEDS}) == 3
    assert len({EV.head_seed(s) for s in SP.SEEDS} | {s for s in SP.SEEDS}) >= 3
    blocks = [set(EV.level_seeds(s)) for s in (0, 1, 2)]
    assert len(set().union(*blocks)) == 9
    for s in (0, 1, 2):
        assert EV.event_seed(s) in EV.level_seeds(s)
    EV.check_event_params(SP.EVENT_PARAMS)
    EV.check_w_grid(SP.W_GRID)
    EV.check_arms(SP.ARMS)
    EV.check_extreme(SP.EXTREME_Q, SP.EXTREME_FRAC, SP.EXTREME_MIN_FINITE)
    for bad in (dict(SP.EVENT_PARAMS, n_estimators=81), dict(SP.EVENT_PARAMS, max_depth=3),
                dict(SP.EVENT_PARAMS, device="gpu")):
        try:
            EV.check_event_params(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"param check must reject {bad}")
    try:
        EV.check_w_grid((0.0, 0.5))
    except ValueError:
        pass
    else:
        raise AssertionError("w grid must stay pinned")
    try:
        EV.check_arms(("control",))
    except ValueError:
        pass
    else:
        raise AssertionError("arms must stay pinned")
    run_src = (EXP / "run_official_everest.py").read_text()
    assert "fit_seed_models" in run_src and "event_seed" in run_src
    assert "level_seeds" in run_src and "head_seed" in run_src and "fit_params" in run_src


def test_mixture_helpers_synthetic():
    EV = _load("everest_lib", "d150ev_lib_mix")
    SP = _load("spec_ev", "d150ev_spec_mix")
    rng = np.random.default_rng(0)
    h = np.tile(np.arange(1, 73), 4)
    mem = {m: rng.normal(400, 20, len(h)) for m in ("e0003_line", "d33_hsplit", "d45_lvlmean")}
    out = EV.blend_per_lead(mem, SP.WEIGHTS, h)
    assert np.isfinite(out).all() and len(out) == len(h)
    m1 = h <= 6
    m2 = (h >= 7) & (h <= 24)
    m3 = h > 24
    st = EV.stitch_members({s: mem["e0003_line"][m1] for s in (0, 1, 2)},
                           {s: mem["e0003_line"][m2] for s in (0, 1, 2)},
                           mem["e0003_line"], mem["e0003_line"][m3], m1, m2, m3, (0, 1, 2), 1)
    assert set(st) == {"e0003_line", "d33_hsplit", "d45_lvlmean"}
    assert EV.as_path_matrix(np.arange(72 * 3, dtype=float)).shape == (3, 72)
    assert EV.lead_of(1) == "1-6" and EV.lead_of(72) == "49-72"
    try:
        EV.check_weights({"1-6": dict(SP.WEIGHTS["1-6"])})
    except ValueError:
        pass
    else:
        raise AssertionError("partial weights must be rejected")


def test_input_only_features_synthetic():
    EV = _load("everest_lib", "d150ev_lib_feat")
    idx = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    filled = pd.Series(400.0 + np.sin(np.arange(len(idx)) / 24.0), index=idx)
    oo = pd.DatetimeIndex([idx[100], idx[150]], tz="UTC")
    P = EV.past12_matrix(oo, filled)
    assert P.shape == (2, 12) and np.allclose(P[0], filled.iloc[89:101].to_numpy())
    B = np.tile(np.linspace(390, 410, 72), (2, 1))
    Z = EV.build_z(P, B, step=3)
    assert Z.shape == (2, 36) and np.isfinite(Z).all()
    mu, sd = EV.standardize_fit(Z)
    Zs = EV.standardize_apply(Z, mu, sd)
    assert np.allclose(np.nanmean(Zs, axis=0), 0.0, atol=1e-9)


def test_event_rule_synthetic():
    EV = _load("everest_lib", "d150ev_lib_event")
    Y = np.full((4, 72), 400.0)
    Y[0, :10] = 900.0
    Y[1, :3] = 900.0
    m = EV.extreme_origin_mask(Y, 500.0)
    assert list(m) == [True, False, False, False]
    ids = EV.oof_block_ids(9, 3)
    assert list(ids) == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    try:
        EV.oof_block_ids(2, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("blocks must reject fewer origins than folds")
    assert abs(EV.oof_mse(np.array([1.0, np.nan]), np.array([1.0, 2.0])) - 0.0) < 1e-12
    assert np.isnan(EV.oof_mse(np.array([np.nan]), np.array([1.0])))
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = EV.frame_predictions(te, np.full(72, 400.0))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert (fr["horizon_hours"].to_numpy() == np.arange(1, 73)).all()
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_calibration_synthetic():
    EV = _load("everest_lib", "d150ev_lib_cal")
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 4, dtype=float)
    p_good = np.asarray(y, dtype=float)
    p_bad = 1.0 - np.asarray(y, dtype=float)
    assert EV.brier_score(y, p_good) < EV.brier_score(y, p_bad)
    assert abs(EV.brier_score(y, p_good) - 0.0) < 1e-12
    assert np.isnan(EV.brier_score(np.array([np.nan]), np.array([0.5])))
    e_good = EV.ece_score(y, p_good, 10)
    e_bad = EV.ece_score(y, p_bad, 10)
    assert e_good < e_bad
    assert np.isnan(EV.ece_score(np.array([np.nan]), np.array([0.5])))
    rng = np.random.default_rng(3)
    p_tr = rng.uniform(0, 1, 200)
    y_tr = (rng.uniform(0, 1, 200) < p_tr).astype(float)
    ir = EV.isotonic_fit(p_tr, y_tr)
    p_te = np.linspace(0, 1, 11)
    c = EV.isotonic_apply(ir, p_te)
    assert ((np.diff(c) >= -1e-12).all()), "isotonic map must be monotone"
    assert ((c >= 0.0) & (c <= 1.0)).all()
    try:
        EV.isotonic_fit(np.array([0.1, 0.2]), np.array([0, 1]))
    except ValueError:
        pass
    else:
        raise AssertionError("isotonic fit must reject tiny samples")


def test_tail_and_hurdle_synthetic():
    EV = _load("everest_lib", "d150ev_lib_tail")
    rng = np.random.default_rng(5)
    exc = rng.exponential(70.0, 500)
    g = EV.fit_gpd_mom(exc, -0.5, 0.9, 50)
    assert g["n_exc"] == 500 and g["method"] == "mom"
    assert abs(g["tail_mean"] - exc.mean()) < 5.0
    assert -0.5 <= g["xi"] <= 0.9 and g["sigma"] > 0
    try:
        EV.fit_gpd_mom(np.array([5.0, 6.0]), -0.5, 0.9, 50)
    except ValueError:
        pass
    else:
        raise AssertionError("tail fit must reject tiny samples")
    base = np.full(144, 400.0)
    p = np.array([0.0, 1.0])
    out0 = EV.apply_hurdle(base, p, 68.0, 0.0)
    assert np.allclose(out0, 400.0), "w=0 is the control"
    out1 = EV.apply_hurdle(base, p, 68.0, 1.0)
    assert np.allclose(out1[:72], 400.0) and np.allclose(out1[72:], 468.0)
    out_half = EV.apply_hurdle(base, p, 68.0, 0.5)
    assert np.allclose(out_half[72:], 434.0)
    pnan = np.array([np.nan, 1.0])
    outn = EV.apply_hurdle(base, pnan, 68.0, 1.0)
    assert np.allclose(outn[:72], 400.0), "NaN prob falls back to base"


def test_poison_synthetic():
    EV = _load("everest_lib", "d150ev_lib_poison")
    idx = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="UTC")
    filled = pd.Series(400.0 + np.sin(np.arange(len(idx)) / 24.0), index=idx)
    cut = "2024-01-20T00:00:00Z"
    pre = pd.DatetimeIndex([idx[100], idx[200]], tz="UTC")
    post = pd.DatetimeIndex([idx[1000], idx[1100]], tz="UTC")
    probe = pre.append(post)
    rec = EV.scenario_poison(probe, filled, cut)
    assert rec["passed"] is True
    assert rec["pre_invariant"] is True and rec["moved_columns"] == ["scenario"]
    assert rec["n_pre_rows"] == 2 and rec["n_post_rows"] == 2
    rec2 = EV.frame_poison(probe, filled, cut, lambda f: EV.past12_matrix(probe, f))
    assert rec2["passed"] is True
    assert rec2["pre_invariant"] is True and rec2["moved_columns"] == ["frame"]
    assert rec2["n_pre_rows"] == 2 and rec2["n_post_rows"] == 2


def test_oof_source_readonly():
    code = _code()
    assert code.count("worker-t1g0") == 1, "single readonly OOF pointer"
    assert 'OOF_DIR = "store/scratch/worker-t1g0/oofset-full"' in code
    run_src = (EXP / "run_official_everest.py").read_text()
    for verb in ("mkdir", "to_parquet", "write_text", "rename", "unlink"):
        for line in run_src.splitlines():
            if "OOF_DIR" in line:
                assert verb not in line, line
    assert "read_parquet" in run_src


def test_scratch_isolation_and_cpu():
    import re
    SP = _load("spec_ev", "d150ev_spec_iso")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1g4"
    assert set(re.findall(r"worker-[a-z0-9]+", code)) == {"worker-t1g4", "worker-t1g0"}
    run_src = (EXP / "run_official_everest.py").read_text()
    assert 'device") == "cpu"' in run_src and "n_jobs" in run_src
    assert "--gpu" not in code
    low = code.lower()
    assert "cuda" not in low


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
    for w in ("official-score", "official_score", "submit", "upload"):
        assert w not in low, w
    assert "freeze" not in low and "push" not in low
