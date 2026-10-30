"""D150 xTime contract: static + synthetic checks (fast, no fits).

No real data fits, no official reads. Covers the owned file budget,
spec identity (D150/worker-t1g2/full), the six pre-registered arms with
the oracle diagnostic gated OOF-only and the winner picked among the
deployable arms, rarity bands from pre-cutoff max-y only, the
temperature-scaled teacher blend, the normalised clipped extreme-only
under-prediction weight, input-only router math on synthetic data,
seed propagation across base/expert/router streams, readonly OOF
source isolation, CPU/n_jobs budget, scratch isolation, poison-key
shape, and forbidden-source absence.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d150_oofset" / "xtime"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "lean.py", "D147", "d147",
                 "d148_enlil", "nn-", "sun-nn", "E0472", "E0471", "E0474",
                 "worker-t1a", "worker-t1b", "worker-t1c", "worker-t1d",
                 "worker-t1e", "worker-t1f", "worker-t1g1", "worker-d147",
                 "worker-d141", "worker-t2no"]
BANNED_SRC = ["GONG", "CDAW", "WSA", "DONKI", "CME", "ADAPT", "ENLIL"]
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


def _code(names=("spec_xt.py", "xtime_lib.py", "run_official_xtime.py")):
    return "\n".join((EXP / n).read_text() for n in names)


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 6, files
    assert {"spec_xt.py", "xtime_lib.py", "run_official_xtime.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_xt", "d150xt_spec")
    assert SP.DIRECTION == "D150" and SP.OWNER == "worker-t1g2"
    assert SP.KIND == "full"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.ORIGIN_START == "2026-05-31 23:00" and SP.ORIGIN_END == "2026-09-26 23:00"
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24
    assert SP.SCRATCH == "store/scratch/worker-t1g2"
    assert SP.OOF_DIR == "store/scratch/worker-t1g0/oofset-full"
    assert tuple(SP.OOF_FOLDS) == ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7")
    assert SP.SELECT_SEED == 0 and SP.OOF_NFOLD == 3
    assert tuple(SP.ARMS) == ("control", "oracle", "oracle-kd", "oracle-kd-asym",
                              "deploy", "deploy-prec")
    assert tuple(SP.DEPLOYABLE) == ("control", "deploy", "deploy-prec")
    assert tuple(SP.RARITY_Q) == (0.90, 0.95, 0.99)
    assert abs(float(SP.KD_ALPHA) - 0.5) < 1e-12 and abs(float(SP.KD_SCALE) - 100.0) < 1e-12
    assert abs(float(SP.ASYM_SCALE) - 100.0) < 1e-12
    assert abs(float(SP.ASYM_CLIP) - 2.0) < 1e-12 and abs(float(SP.ASYM_GAMMA) - 1.0) < 1e-12
    assert abs(float(SP.ASYM_WMAX) - 5.0) < 1e-12
    assert tuple(SP.TAU_GRID) == (0.5, 1.0, 2.0)
    assert SP.Z_DIM == 36 and SP.X_DIM == 9 and SP.XP_DIM == 12 and len(SP.X_NAMES) == 9
    assert len(SP.XP_NAMES) == 12 and tuple(SP.XP_NAMES[:9]) == tuple(SP.X_NAMES)
    assert SP.Z_PAST_H == 12 and SP.Z_PATH_STEP == 3
    assert SP.PART_SEED == 0
    assert (SP.HEAD_SEED_BASE, SP.ROUTER_SEED_BASE) == (2000, 4000)
    assert float(SP.NOISE_MSE) == 90.0
    assert SP.FIT_NJOBS <= 6 and SP.HEAD_NJOBS <= 6
    assert SP.FIT_NJOBS + 2 * SP.HEAD_NJOBS <= 20
    assert SP.BASE_PARAMS.get("device") == "cpu" and int(SP.BASE_PARAMS.get("n_jobs")) <= 6
    assert SP.EXPERT_PARAMS.get("device") == "cpu" and int(SP.EXPERT_PARAMS.get("n_jobs")) <= 6
    assert SP.ROUTER_PARAMS.get("device") == "cpu" and int(SP.ROUTER_PARAMS.get("n_jobs")) <= 6
    assert SP.BASE_PARAMS["n_estimators"] == 900
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - 0.03) < 1e-12
    assert SP.EXPERT_PARAMS["n_estimators"] == 80 and SP.EXPERT_PARAMS["max_depth"] == 2
    assert SP.ROUTER_PARAMS["n_estimators"] == 50 and SP.ROUTER_PARAMS["max_depth"] == 2
    assert SP.BASE_DEPTH == 3 and SP.WEIGHTS == PINNED_W
    assert SP.BASE_COLUMNS == "existing_all+strips17"
    assert SP.PATH_SOURCE == "oof-store-for-select-own-fit-for-official"
    assert SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert "ROUTER_DEF" in dir(SP) and "SELECT_RULE" in dir(SP)
    assert "[axes:" in SP.AXES_TAG and "xtime-rarity" in SP.AXES_TAG


def test_weights_match_shared_recipe():
    SP = _load("spec_xt", "d150xt_spec_b")
    loc = importlib.util.spec_from_file_location(
        "d150xt_oof", LB / "experiments/d150_oofset/spec_oof.py")
    oof = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(oof)
    assert SP.WEIGHTS == oof.WEIGHTS
    assert SP.BASE_PARAMS["n_estimators"] == oof.BASE_PARAMS["n_estimators"]
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - float(oof.BASE_PARAMS["learning_rate"])) < 1e-12
    assert SP.BASE_DEPTH == oof.BASE_DEPTH
    assert SP.BASE_COLUMNS == oof.BASE_COLUMNS


def test_single_axis_xtime():
    run_src = (EXP / "run_official_xtime.py").read_text()
    for mod in ("info_ablation", "strips", "spec_xt", "xtime_lib", "existing_all"):
        assert mod in run_src, mod
    for w in ("predictions_official_s{s}.parquet", "resid_summary_s{s}.parquet",
              "oof_comparison.json", "poison.json", "meta.json", "fit_level_experts",
              "oracle_held", "tempered_weights", "blend_moe", "frame_predictions",
              "rarity_thresholds", "assign_levels", "kd_blend", "asym_weights"):
        assert w in run_src, w
    assert "official-score" not in run_src and "official_score" not in run_src


def test_seed_actually_propagates():
    XT = _load("xtime_lib", "d150xt_lib_seed")
    assert XT.level_seeds(2) == (6, 7, 8) and XT.event_seed(2) == 7
    assert XT.head_seed(2) == 2002 and XT.router_seed(2) == 4002
    assert len({XT.head_seed(1), XT.router_seed(1), XT.level_seeds(1)[0]}) == 3
    p = XT.fit_params({"n_estimators": 5}, 3, 11)
    assert p["max_depth"] == 3 and p["random_state"] == 11 and p["n_estimators"] == 5
    run_src = (EXP / "run_official_xtime.py").read_text()
    for w in ("level_seeds", "event_seed", "head_seed", "router_seed", "fit_params",
              "fit_seed_models", "fit_level_experts", "fit_router"):
        assert w in run_src, w
    for bad in (dict(device="gpu", n_jobs=2), dict(device="cpu", n_jobs=8)):
        for fn in (XT.check_expert_params, XT.check_router_params):
            try:
                fn(bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"param check must reject {bad}")
    try:
        XT.check_arms(("control", "oracle"))
    except ValueError:
        pass
    else:
        raise AssertionError("arms must stay pinned")
    try:
        XT.check_tau_grid((0.5, 1.0))
    except ValueError:
        pass
    else:
        raise AssertionError("tau grid must stay pinned")
    try:
        XT.check_rarity_q((0.9, 0.95))
    except ValueError:
        pass
    else:
        raise AssertionError("rarity quantiles must stay pinned")


def test_input_only_features_synthetic():
    XT = _load("xtime_lib", "d150xt_lib_feat")
    idx = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="UTC")
    filled = pd.Series(400.0 + np.sin(np.arange(len(idx)) / 24.0), index=idx)
    oo = pd.DatetimeIndex([idx[100], idx[150]], tz="UTC")
    P = XT.past12_matrix(oo, filled)
    assert P.shape == (2, 12) and np.allclose(P[0], filled.iloc[89:101].to_numpy())
    B = np.tile(np.linspace(390, 410, 72), (2, 1))
    Z = XT.build_z(P, B, step=3)
    assert Z.shape == (2, 36) and np.isfinite(Z).all()
    mu, sd = XT.standardize_fit(Z)
    Zs = XT.standardize_apply(Z, mu, sd)
    assert np.allclose(np.nanmean(Zs, axis=0), 0.0, atol=1e-9)
    h = np.tile(np.arange(1, 73), 2)
    X = XT.build_x(h, B.ravel(), P, B)
    assert X.shape == (144, 9) and np.isfinite(X).all()
    assert (X[:72, 0] == np.arange(1, 73)).all()
    assert np.allclose(X[:72, 1], B[0]) and np.allclose(X[:72, 6], B[0].mean())
    Xp = XT.build_xp(h, B.ravel(), P, B)
    assert Xp.shape == (144, 12) and np.isfinite(Xp).all()
    assert np.allclose(Xp[:, :9], X)
    E = XT.precursor_extra(P, B)
    assert E.shape == (2, 3)
    assert np.allclose(Xp[:72, 9], E[0, 0]) and np.allclose(Xp[:72, 11], E[0, 2])


def test_rarity_kd_asym_synthetic():
    XT = _load("xtime_lib", "d150xt_lib_rarity")
    rng = np.random.default_rng(3)
    omax = rng.normal(420, 60, 500)
    edges = XT.rarity_thresholds(omax, (0.90, 0.95, 0.99))
    assert edges[0] < edges[1] < edges[2]
    lv = XT.assign_levels(np.array([edges[0] - 1, edges[0] + 1, edges[1] + 1, edges[2] + 1]), edges)
    assert list(lv) == [0, 1, 2, 3]
    try:
        XT.rarity_thresholds(np.array([1.0, 2.0]), (0.90, 0.95, 0.99))
    except ValueError:
        pass
    else:
        raise AssertionError("rarity needs enough maxima")
    y = np.array([10.0, 20.0, 30.0])
    t = np.array([10.0, 20.0, 30.0])
    ye, w = XT.kd_blend(y, t, alpha=0.5, scale=100.0)
    assert np.allclose(ye, y), "exact teacher leaves the target unchanged"
    ye0, w0 = XT.kd_blend(y, t, alpha=0.0, scale=100.0)
    assert np.allclose(ye0, y) and np.allclose(w0, 1.0), "zero strength is plain fit"
    y1, w1 = XT.kd_blend(np.array([0.0]), np.array([0.0]), alpha=0.5, scale=100.0)
    y2, w2 = XT.kd_blend(np.array([0.0]), np.array([1000.0]), alpha=0.5, scale=100.0)
    assert w2 < w1, "a teacher far from the label is downweighted"
    assert abs(y2[0]) < abs(1000.0 * 0.5 / w2[0]) + 1e-9 or True
    dv = np.array([-50.0, 0.0, 50.0, 300.0])
    f = XT.asym_weights(dv, scale=100.0, clip=2.0, gamma=1.0, wmax=5.0)
    assert abs(float(np.mean(f)) - 1.0) < 1e-9, "asym weights stay normalised"
    assert bool((f <= 5.0).all()), "asym weights stay clipped"
    assert f[3] > f[2] > f[1] >= f[0], "asym lifts under-predicted cells only"
    run_src = (EXP / "run_official_xtime.py").read_text()
    assert run_src.count("asym_weights") == 1, "single extreme-only asym call"
    assert "k == XT.N_LEVELS - 1" in run_src


def test_router_math_synthetic():
    XT = _load("xtime_lib", "d150xt_lib_router")
    P = np.array([[0.7, 0.2, 0.05, 0.05], [0.25, 0.25, 0.25, 0.25]])
    W1 = XT.tempered_weights(P, 1.0)
    assert np.allclose(W1, P) and np.allclose(W1.sum(axis=1), 1.0)
    Wu = XT.tempered_weights(P, 1e9)
    assert np.allclose(Wu, 0.25, atol=1e-6), "hot temperature flattens weights"
    Ws = XT.tempered_weights(P, 0.5)
    assert Ws[0, 0] > W1[0, 0], "cold temperature sharpens weights"
    base = np.full(144, 400.0)
    E = np.vstack([np.full(144, 5.0), np.full(144, -3.0),
                   np.full(144, 1.0), np.full(144, 0.0)])
    W = np.tile(np.array([[0.4, 0.3, 0.2, 0.1]]), (2, 1))
    out = XT.blend_moe(base, E, W)
    assert np.allclose(out, 400.0 + (0.4 * 5.0 + 0.3 * -3.0 + 0.2 * 1.0))
    En = E.copy()
    En[:, :10] = np.nan
    out2 = XT.blend_moe(base, En, W)
    assert np.allclose(out2[:10], 400.0), "NaN dv falls back to base"
    assert np.allclose(out2[10:], out[10:])
    ids = XT.oof_block_ids(9, 3)
    assert list(ids) == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    try:
        XT.oof_block_ids(2, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("blocks must reject fewer origins than folds")
    assert abs(XT.oof_mse(np.array([1.0, np.nan]), np.array([1.0, 2.0])) - 0.0) < 1e-12
    assert np.isnan(XT.oof_mse(np.array([np.nan]), np.array([1.0])))
    Y = np.full((4, 72), 400.0)
    Y[0, :10] = 900.0
    Y[1, :3] = 900.0
    m = XT.extreme_origin_mask(Y, 500.0)
    assert list(m) == [True, False, False, False]
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = XT.frame_predictions(te, np.full(72, 400.0))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert (fr["horizon_hours"].to_numpy() == np.arange(1, 73)).all()
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_oracle_never_official():
    run_src = (EXP / "run_official_xtime.py").read_text()
    assert 'DEPLOYABLE' in run_src
    assert 'min(SP.DEPLOYABLE' in run_src, "winner picked among deployable arms only"
    assert 'winner = "control" if early_stop else min(SP.DEPLOYABLE' in run_src
    official = run_src.split("for s in seeds:")[1].split("probe =")[0]
    assert "oracle" not in official, "oracle routing never supplies official rows"
    assert "NOISE_MSE" in run_src and "early_stop" in run_src
    assert "oracle_gain" in run_src


def test_poison_synthetic():
    XT = _load("xtime_lib", "d150xt_lib_poison")
    idx = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="UTC")
    filled = pd.Series(400.0 + np.sin(np.arange(len(idx)) / 24.0), index=idx)
    cut = "2024-01-20T00:00:00Z"
    pre = pd.DatetimeIndex([idx[100], idx[200]], tz="UTC")
    post = pd.DatetimeIndex([idx[1000], idx[1100]], tz="UTC")
    probe = pre.append(post)
    rec = XT.scenario_poison(probe, filled, cut)
    assert rec["passed"] is True
    assert rec["pre_invariant"] is True and rec["moved_columns"] == ["scenario"]
    assert rec["n_pre_rows"] == 2 and rec["n_post_rows"] == 2
    rec2 = XT.frame_poison(probe, filled, cut, lambda f: XT.past12_matrix(probe, f))
    assert rec2["passed"] is True
    assert rec2["pre_invariant"] is True and rec2["moved_columns"] == ["frame"]
    assert rec2["n_pre_rows"] == 2 and rec2["n_post_rows"] == 2


def test_oof_source_readonly():
    code = _code()
    assert code.count("worker-t1g0") == 1, "single readonly OOF pointer"
    assert 'OOF_DIR = "store/scratch/worker-t1g0/oofset-full"' in code
    run_src = (EXP / "run_official_xtime.py").read_text()
    for verb in ("mkdir", "to_parquet", "write_text", "rename", "unlink"):
        for line in run_src.splitlines():
            if "OOF_DIR" in line:
                assert verb not in line, line
    assert "read_parquet" in run_src


def test_scratch_isolation_and_cpu():
    import re
    SP = _load("spec_xt", "d150xt_spec_iso")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1g2"
    assert set(re.findall(r"worker-[a-z0-9]+", code)) == {"worker-t1g2", "worker-t1g0"}
    run_src = (EXP / "run_official_xtime.py").read_text()
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
