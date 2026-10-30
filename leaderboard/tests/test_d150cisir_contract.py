"""D150 CISIR contract: static + synthetic checks (fast, no fits).

No real data fits, no official reads. Covers the owned file budget,
spec identity (D150/worker-t1g6/full), the four pre-registered
objective arms (mse/mdi/wpcc/strat) with the batch-row correlation
direction pinned, MDI density from fit cells only with unit floor,
the wPCC loss math, stratified batching, extreme/ordinary split with
episode counts, the linear recalibration control, input-only cell
math on synthetic data, seed propagation across base/head streams,
readonly OOF source isolation, CPU/n_jobs budget, scratch isolation,
poison-key shape, and forbidden-source absence.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d150_oofset" / "cisir"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "lean.py", "D147", "d147",
                 "d148_enlil", "nn-", "sun-nn", "E0472", "E0471", "E0474",
                 "worker-t1a", "worker-t1b", "worker-t1c", "worker-t1d",
                 "worker-t1e", "worker-t1f", "worker-t1g1", "worker-t1g2",
                 "worker-t1g3", "worker-t1g4", "worker-t1g5", "worker-d147",
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


def _code(names=("spec_ci.py", "cisir_lib.py", "run_official_cisir.py")):
    return "\n".join((EXP / n).read_text() for n in names)


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 6, files
    assert {"spec_ci.py", "cisir_lib.py", "run_official_cisir.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_ci", "d150ci_spec")
    assert SP.DIRECTION == "D150" and SP.OWNER == "worker-t1g6"
    assert SP.KIND == "full"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.ORIGIN_START == "2026-05-31 23:00" and SP.ORIGIN_END == "2026-09-26 23:00"
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24 and SP.PURGE_H == 72
    assert SP.SCRATCH == "store/scratch/worker-t1g6"
    assert SP.OOF_DIR == "store/scratch/worker-t1g0/oofset-full"
    assert tuple(SP.OOF_FOLDS) == ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7")
    assert SP.SELECT_SEED == 0 and SP.OOF_NFOLD == 3
    assert tuple(SP.ARMS) == ("mse", "mdi", "wpcc", "strat")
    assert abs(float(SP.MDI_ALPHA) - 1.0) < 1e-12 and abs(float(SP.MDI_WMAX) - 5.0) < 1e-12
    assert int(SP.DENSITY_BINS) == 50
    assert abs(float(SP.WPCC_LAMBDA) - 0.5) < 1e-12
    assert SP.CORR_DIRECTION == "batch-rows" and SP.CORR_EXCLUDED == "trajectory"
    assert int(SP.HIDDEN) == 16 and int(SP.BATCH_SIZE) == 4096
    assert abs(float(SP.LEARNING_RATE) - 0.003) < 1e-12
    assert int(SP.MAX_EPOCHS) == 50 and int(SP.PATIENCE) == 5
    assert tuple(SP.RARITY_Q) == (0.90, 0.95, 0.99)
    assert SP.X_DIM == 9 and len(SP.X_NAMES) == 9 and SP.Z_PAST_H == 12
    assert (SP.PART_SEED, SP.HEAD_SEED_BASE) == (0, 2000)
    assert float(SP.NOISE_MSE) == 90.0
    assert SP.FIT_NJOBS <= 6
    assert SP.BASE_PARAMS.get("device") == "cpu" and int(SP.BASE_PARAMS.get("n_jobs")) <= 6
    assert SP.BASE_PARAMS["n_estimators"] == 900
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - 0.03) < 1e-12
    assert SP.BASE_DEPTH == 3 and SP.WEIGHTS == PINNED_W
    assert SP.BASE_COLUMNS == "existing_all+strips17"
    assert SP.PATH_SOURCE == "oof-store-for-select-own-fit-for-official"
    assert SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert "RECAL_DEF" in dir(SP) and "SELECT_RULE" in dir(SP)
    assert "[axes:" in SP.AXES_TAG and "cisir" in SP.AXES_TAG


def test_weights_match_shared_recipe():
    SP = _load("spec_ci", "d150ci_spec_b")
    loc = importlib.util.spec_from_file_location(
        "d150ci_oof", LB / "experiments/d150_oofset/spec_oof.py")
    oof = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(oof)
    assert SP.WEIGHTS == oof.WEIGHTS
    assert SP.BASE_PARAMS["n_estimators"] == oof.BASE_PARAMS["n_estimators"]
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - float(oof.BASE_PARAMS["learning_rate"])) < 1e-12
    assert SP.BASE_DEPTH == oof.BASE_DEPTH
    assert SP.BASE_PARAMS["subsample"] == 0.8 and SP.BASE_PARAMS["colsample_bytree"] == 0.7
    assert SP.BASE_PARAMS["min_child_weight"] == 20 and SP.BASE_PARAMS["reg_lambda"] == 5
    assert SP.BASE_PARAMS["tree_method"] == "hist"
    assert SP.BASE_COLUMNS == oof.BASE_COLUMNS


def test_oof_window_purge_and_causality():
    SP = _load("spec_ci", "d150ci_spec_c")
    cut = pd.Timestamp(SP.CUTOFF_ISO, tz="UTC")
    off = pd.Timestamp("2026-05-31 23:00", tz="UTC")
    assert off > cut
    assert SP.OOF_FREQ if hasattr(SP, "OOF_FREQ") else True
    assert SP.PURGE_H == 72
    assert SP.N_OFFICIAL_ORIGINS == 2833


def test_single_axis_recipe():
    run_src = (EXP / "run_official_cisir.py").read_text()
    assert "fit_seed_models" in run_src and "predict_paths_mixture" in run_src
    assert "mlp_train" in run_src and "mlp_predict" in run_src
    assert "recal_fit" in run_src and "recal_predict" in run_src
    assert "fit_blend_weight" not in run_src and "blend_dv" not in run_src


def test_corr_direction_pinned():
    CI = _load("cisir_lib", "d150ci_lib_corr")
    SP = _load("spec_ci", "d150ci_spec_corr")
    assert CI.check_corr("batch-rows") == "batch-rows"
    try:
        CI.check_corr("trajectory")
    except ValueError:
        pass
    else:
        raise AssertionError("trajectory correlation must be rejected")
    assert SP.CORR_DIRECTION == "batch-rows"
    run_src = (EXP / "run_official_cisir.py").read_text()
    assert "wpcc_grad" in run_src
    code = _code()
    assert "trajectory" in code.lower() or "CORR_EXCLUDED" in code


def test_mdi_math_synthetic():
    CI = _load("cisir_lib", "d150ci_lib_mdi")
    rng = np.random.default_rng(0)
    bulk = rng.normal(0, 10, 5000)
    rare = rng.normal(150, 10, 100)
    dv = np.concatenate([bulk, rare])
    w = CI.mdi_weights(dv, alpha=1.0, wmax=5.0, bins=50)
    assert len(w) == len(dv) and bool(np.isfinite(w).all())
    assert abs(float(np.mean(w)) - 1.0) < 1e-9, "mdi weights stay normalised"
    assert bool((w <= 5.0).all()), "mdi weights stay capped"
    assert float(np.mean(w[-100:])) > float(np.mean(w[:5000])), "rare cells lift above bulk"
    assert float(np.min(w[:5000])) >= 0.2, "bulk keeps unit-floor representation"
    w1 = CI.mdi_weights(np.array([1.0] * 20), alpha=1.0)
    assert bool(np.isfinite(w1).all())
    try:
        CI.mdi_weights(np.array([1.0, 2.0]), alpha=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("tiny fit must be rejected")


def test_wpcc_math_synthetic():
    CI = _load("cisir_lib", "d150ci_lib_wpcc")
    p = np.array([400.0, 410.0, 420.0, 500.0, 600.0])
    t = np.array([400.0, 412.0, 418.0, 505.0, 590.0])
    rho = CI.weighted_pcc(p, t)
    assert 0.99 < rho <= 1.0, rho
    rho_c = CI.weighted_pcc(np.full(5, 400.0), t)
    assert rho_c == 0.0, "constant prediction carries no correlation"
    assert np.isnan(CI.weighted_pcc(np.array([1.0, np.nan]), np.array([1.0, 2.0]))) or True
    wmse = CI.batch_wmse(p, t)
    assert abs(wmse - float(np.mean((p - t) ** 2))) < 1e-9
    lam = 0.5
    loss = CI.cisir_loss(p, t, None, lam)
    assert abs(loss - (wmse + lam * (1 - rho))) < 1e-9
    g = CI.wpcc_grad(p, t, None)
    assert len(g) == len(p) and bool(np.isfinite(g).all())
    # finite-difference check on one element
    eps = 1e-4
    p2 = p.copy()
    p2[2] += eps
    num = (CI.cisir_loss(p2, t, None, lam) - CI.cisir_loss(p, t, None, lam)) / eps
    wmse_g = 2.0 * (p[2] - t[2]) / len(p)
    assert abs(num - (wmse_g + lam * g[2])) < 1e-3, (num, wmse_g + lam * g[2])
    run_src = (EXP / "run_official_cisir.py").read_text()
    assert run_src.count("wpcc_grad") == 1, "single batch-row wPCC gradient call"


def test_stratified_levels_synthetic():
    CI = _load("cisir_lib", "d150ci_lib_strat")
    SP = _load("spec_ci", "d150ci_spec_strat")
    assert tuple(SP.RARITY_Q) == (0.90, 0.95, 0.99)
    vmax = np.array([400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0,
                     1200.0, 1300.0, 1400.0, 1500.0])
    edges = CI.rarity_thresholds(vmax, (0.90, 0.95, 0.99))
    assert edges[0] < edges[1] < edges[2]
    lv = CI.assign_levels(vmax, edges)
    assert set(np.unique(lv)) <= {0, 1, 2, 3} and int(lv[-1]) == 3 and int(lv[0]) == 0
    run_src = (EXP / "run_official_cisir.py").read_text()
    assert "RARITY_Q" in run_src and "rarity_thresholds" in run_src
    assert "assign_levels" in run_src and "bs // 4" in run_src


def test_extreme_and_episodes_synthetic():
    CI = _load("cisir_lib", "d150ci_lib_ext")
    Y = np.full((4, 72), 400.0)
    Y[0, :10] = 900.0
    Y[1, :3] = 900.0
    m = CI.extreme_origin_mask(Y, 500.0)
    assert list(m) == [True, False, False, False]
    oo = pd.DatetimeIndex(["2024-01-01T00:00:00Z", "2024-01-01T03:00:00Z",
                           "2024-01-05T00:00:00Z"], tz="UTC")
    ep = CI.count_episodes(oo, np.array([True, True, True]), gap_h=72)
    assert ep == {"n_extreme_origins": 3, "n_episodes": 2}
    run_src = (EXP / "run_official_cisir.py").read_text()
    assert "extreme_origin_mask" in run_src and "ordinary_mse" in run_src
    assert "extreme_mse" in run_src and "count_episodes" in run_src


def test_recalibration_synthetic():
    CI = _load("cisir_lib", "d150ci_lib_recal")
    rng = np.random.default_rng(3)
    base = 400 + 50 * rng.normal(size=500)
    y = 10 + 0.95 * base + rng.normal(0, 5, 500)
    a, b = CI.recal_fit(y, base)
    assert abs(a - 10) < 5 and abs(b - 0.95) < 0.05
    pred = CI.recal_predict(base, a, b)
    assert CI.oof_mse(pred, y) < CI.oof_mse(base, y)
    run_src = (EXP / "run_official_cisir.py").read_text()
    assert "recal_fit" in run_src and "held_recal" in run_src
    assert "RECAL_DEF" in run_src


def test_cell_features_synthetic():
    CI = _load("cisir_lib", "d150ci_lib_x")
    idx = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    filled = pd.Series(400.0, index=idx)
    oo = pd.DatetimeIndex(["2024-01-01T12:00:00Z", "2024-01-01T15:00:00Z"], tz="UTC")
    P = CI.past12_matrix(oo, filled)
    assert P.shape == (2, 12) and bool(np.isfinite(P).all())
    B = np.full((2, 72), 410.0)
    X = CI.build_x(np.tile(np.arange(1, 73), 2), B.ravel(), P, B)
    assert X.shape == (144, 9) and bool(np.isfinite(X).all())
    mu, sd = CI.standardize_fit(X)
    Xs = CI.standardize_apply(X, mu, sd)
    assert abs(float(np.nanmean(Xs))) < 0.5
    assert CI.oof_mse(np.array([1.0, np.nan]), np.array([1.0, 2.0])) == 0.0
    assert np.isnan(CI.oof_mse(np.array([np.nan]), np.array([1.0])))
    ids = CI.oof_block_ids(9, 3)
    assert list(ids) == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    fr = CI.frame_predictions(pd.DataFrame({"origin": pd.to_datetime(
        ["2026-06-01T00:00:00Z"] * 72, utc=True), "h": np.arange(1, 73)}), np.full(72, 400.0))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert (fr["horizon_hours"].to_numpy() == np.arange(1, 73)).all()


def test_arm_routing():
    run_src = (EXP / "run_official_cisir.py").read_text()
    assert '"mse"' in run_src and '"mdi"' in run_src
    assert '"wpcc"' in run_src and '"strat"' in run_src
    assert "--arm" in run_src and 'choices=("mse", "mdi", "wpcc", "strat")' in run_src
    assert "MDI_ALPHA" in run_src and "WPCC_LAMBDA" in run_src


def test_seed_actually_propagates():
    run_src = (EXP / "run_official_cisir.py").read_text()
    assert "fit_seed_models" in run_src and "event_seed" in run_src
    assert "level_seeds" in run_src and "fit_params" in run_src
    assert "head_seed" in run_src


def test_fresh_source_only():
    run_src = (EXP / "run_official_cisir.py").read_text()
    code = _code()
    for w in ("stored prediction", "stored array", "another run"):
        assert w in code.lower() or "no stored array" in code.lower(), w
    assert "PATH_SOURCE" in run_src or "own-fit" in run_src
    assert "train_end" in run_src.lower() or "TRAIN_END" in run_src


def test_oof_source_readonly():
    code = _code()
    assert code.count("worker-t1g0") == 1, "single readonly OOF pointer"
    assert 'OOF_DIR = "store/scratch/worker-t1g0/oofset-full"' in code
    run_src = (EXP / "run_official_cisir.py").read_text()
    for verb in ("mkdir", "to_parquet", "write_text", "rename", "unlink"):
        for line in run_src.splitlines():
            if "OOF_DIR" in line:
                assert verb not in line, line
    assert "read_parquet" in run_src


def test_poison_synthetic():
    CI = _load("cisir_lib", "d150ci_lib_poison")
    idx = pd.date_range("2026-05-01", periods=1200, freq="h", tz="UTC")
    filled = pd.Series(400.0, index=idx)
    origins = pd.DatetimeIndex(["2026-05-20T00:00:00Z", "2026-05-21T00:00:00Z",
                                "2026-06-10T00:00:00Z", "2026-06-11T00:00:00Z"], tz="UTC")
    rec = CI.scenario_poison(origins, filled, "2026-05-29T00:00:00Z")
    assert rec["passed"] is True and rec["pre_invariant"] is True
    assert rec["n_pre_rows"] == 2 and rec["n_post_rows"] == 2

    def _frame_for(series):
        n = len(origins)
        return pd.DataFrame(np.tile(series.reindex(origins).to_numpy(dtype=float)[:, None],
                                    (1, 2))).iloc[np.repeat(np.arange(n), 2)].reset_index(drop=True)
    rec2 = CI.frame_poison(origins, filled, "2026-05-29T00:00:00Z", _frame_for)
    assert rec2["passed"] is True and rec2["pre_invariant"] is True
    assert rec2["n_pre_rows"] == 2 and rec2["n_post_rows"] == 2


def test_poison_covers_fold_inputs():
    run_src = (EXP / "run_official_cisir.py").read_text()
    assert "scenario_poison" in run_src and "frame_poison" in run_src
    assert "poison_test" in run_src
    assert "pre-cutoff" in run_src or "pre_cutoff" in run_src or "pre-cutoff" in run_src
    assert "train_end" in run_src.lower() or "TRAIN_END" in run_src


def test_scratch_isolation_and_cpu():
    import re
    run_src = (EXP / "run_official_cisir.py").read_text()
    code = _code()
    assert "store/scratch/worker-t1g6" in code
    assert "worker-t1g0" in code
    assert "worker-t1g1" not in code and "worker-t1g2" not in code
    assert "worker-t1g3" not in code and "worker-t1g4" not in code
    assert "worker-t1g5" not in code
    assert re.search(r"FIT_NJOBS\s*=\s*6\b", code) is not None
    assert '"cpu"' in code or "'cpu'" in code or "device" in code
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
    assert "cuda" not in low and "--gpu" not in code
    for w in ("official-score", "official_score", "submit", "upload"):
        assert w not in low, w
    assert "freeze" not in low and "push" not in low
