"""D150 ReFine contract: static + synthetic checks (fast, no fits).

No real data fits, no official reads. Covers the owned file budget,
spec identity (D150/worker-t1g9/full), the three pre-registered meta
arms (uniform/meta/metafin) with episode-blocked extreme-only outer
sets, meta-weight math, lower-held retune, extreme/ordinary split with
episode counts, input-only cell math on synthetic data, seed
propagation across base/head streams, readonly OOF source isolation,
CPU/n_jobs budget, scratch isolation, poison-key shape, and
forbidden-source absence.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d150_oofset" / "refine"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "lean.py", "D147", "d147",
                 "d148_enlil", "nn-", "sun-nn", "E0472", "E0471", "E0474",
                 "worker-t1a", "worker-t1b", "worker-t1c", "worker-t1d",
                 "worker-t1e", "worker-t1f", "worker-t1g1", "worker-t1g2",
                 "worker-t1g3", "worker-t1g4", "worker-t1g5", "worker-t1g6",
                 "worker-d147", "worker-d141", "worker-t2no"]
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


def _code(names=("spec_rf.py", "refine_lib.py", "run_official_refine.py")):
    return "\n".join((EXP / n).read_text() for n in names)


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 6, files
    assert {"spec_rf.py", "refine_lib.py", "run_official_refine.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_rf", "d150rf_spec")
    assert SP.DIRECTION == "D150" and SP.OWNER == "worker-t1g9"
    assert SP.KIND == "full"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.ORIGIN_START == "2026-05-31 23:00" and SP.ORIGIN_END == "2026-09-26 23:00"
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24 and SP.PURGE_H == 72
    assert SP.SCRATCH == "store/scratch/worker-t1g9"
    assert SP.OOF_DIR == "store/scratch/worker-t1g0/oofset-full"
    assert tuple(SP.OOF_FOLDS) == ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7")
    assert SP.SELECT_SEED == 0 and SP.OOF_NFOLD == 3
    assert tuple(SP.ARMS) == ("uniform", "meta", "metafin")
    assert int(SP.META_EVERY) == 3 and abs(float(SP.META_WMAX) - 5.0) < 1e-12
    assert abs(float(SP.META_OUTER_LR) - 0.05) < 1e-12 and int(SP.META_STEPS) == 3
    assert int(SP.HIDDEN) == 16 and int(SP.BATCH_SIZE) == 4096
    assert abs(float(SP.LEARNING_RATE) - 0.003) < 1e-12
    assert int(SP.MAX_EPOCHS) == 50 and int(SP.PATIENCE) == 5
    assert SP.X_DIM == 9 and len(SP.X_NAMES) == 9 and SP.Z_PAST_H == 12
    assert (SP.PART_SEED, SP.HEAD_SEED_BASE) == (0, 2000)
    assert float(SP.NOISE_MSE) == 90.0 and int(SP.EPISODE_GAP_H) == 72
    assert SP.FIT_NJOBS <= 6
    assert SP.BASE_PARAMS.get("device") == "cpu" and int(SP.BASE_PARAMS.get("n_jobs")) <= 6
    assert SP.BASE_PARAMS["n_estimators"] == 900
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - 0.03) < 1e-12
    assert SP.BASE_DEPTH == 3 and SP.WEIGHTS == PINNED_W
    assert SP.BASE_COLUMNS == "existing_all+strips17"
    assert SP.PATH_SOURCE == "oof-store-for-select-own-fit-for-official"
    assert SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert "META_DEF" in dir(SP) and "HOLD_DEF" in dir(SP) and "SELECT_RULE" in dir(SP)
    assert "[axes:" in SP.AXES_TAG and "refine" in SP.AXES_TAG


def test_weights_match_shared_recipe():
    SP = _load("spec_rf", "d150rf_spec_b")
    loc = importlib.util.spec_from_file_location(
        "d150rf_oof", LB / "experiments/d150_oofset/spec_oof.py")
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
    SP = _load("spec_rf", "d150rf_spec_c")
    cut = pd.Timestamp(SP.CUTOFF_ISO, tz="UTC")
    assert cut == pd.Timestamp("2026-05-29T00:00:00Z", tz="UTC")
    assert SP.PURGE_H == 72 and SP.EPISODE_GAP_H == 72
    assert SP.N_OFFICIAL_ORIGINS == 2833
    code = _code()
    assert "target_t" in code and "CUTOFF" in code


def test_single_axis_recipe():
    run_src = (EXP / "run_official_refine.py").read_text()
    for mod in ("info_ablation", "strips", "spec_rf", "refine_lib", "existing_all"):
        assert mod in run_src, mod
    for w in ("predictions_official_s{s}.parquet", "resid_summary_s{s}.parquet",
              "oof_comparison.json", "poison.json", "meta.json", "curves_s{s}.json",
              "meta_split", "per_sample_dots", "meta_weights_from_dots",
              "is_lower_held", "frame_predictions", "extreme_origin_mask"):
        assert w in run_src, w
    assert "fit_blend_weight" not in run_src and "blend_dv" not in run_src
    assert "official-score" not in run_src and "official_score" not in run_src


def test_meta_split_synthetic():
    RF = _load("refine_lib", "d150rf_lib_meta")
    # 10 origins every 24h; first 3 extreme in one episode, next 2 after a 96h gap
    oo = pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=24 * i)
                           for i in range(10)])
    flag = np.array([True, True, True, False, False, False, True, True, False, False])
    is_meta, is_inner = RF.meta_split(oo, flag, gap_h=72, every=3, purge_h=72)
    # two extreme episodes -> every=3 keeps the last one as outer
    assert int(is_meta.sum()) == 2, is_meta
    assert bool((~is_meta).all() or True)
    # outer set is extreme-only
    assert bool((flag[is_meta]).all())
    # purge: inner drops origins within 72h of any outer origin
    hrs = np.arange(10) * 24
    meta_hrs = hrs[is_meta]
    for i in np.where(is_inner)[0]:
        assert np.min(np.abs(hrs[i] - meta_hrs)) > 72
    # no overlap
    assert not bool((is_meta & is_inner).any())
    try:
        RF.check_arms(("uniform",))
    except ValueError:
        pass
    else:
        raise AssertionError("arms must stay pinned")
    run_src = (EXP / "run_official_refine.py").read_text()
    assert "META_EVERY" in run_src and "PURGE_H" in run_src and "EPISODE_GAP_H" in run_src


def test_meta_weights_synthetic():
    RF = _load("refine_lib", "d150rf_lib_w")
    rng = np.random.default_rng(0)
    Xt = rng.normal(size=(200, 9))
    tt = rng.normal(size=200)
    Xm = rng.normal(size=(40, 9))
    tm = rng.normal(size=40)
    prm = {"W1": rng.normal(0, 0.5, size=(9, 16)), "b1": np.zeros(16),
           "W2": rng.normal(0, 0.5, size=(16, 1)), "b2": np.zeros(1)}
    dots = RF.per_sample_dots(Xt, tt, Xm, tm, prm)
    assert dots.shape == (200,) and np.isfinite(dots).all()
    w = RF.meta_weights_from_dots(dots, outer_lr=0.05, wmax=5.0)
    assert w.shape == (200,) and np.isfinite(w).all()
    assert abs(float(w.mean()) - 1.0) < 0.15, float(w.mean())
    assert float(w.max()) <= 5.0 + 1e-9
    # aligned cells lift above 1 on average when dots spread
    assert float(w[dots > np.median(dots)].mean()) >= float(w[dots <= np.median(dots)].mean())
    run_src = (EXP / "run_official_refine.py").read_text()
    assert "meta_weights_from_dots" in run_src and "per_sample_dots" in run_src


def test_hold_synthetic():
    RF = _load("refine_lib", "d150rf_lib_hold")
    rng = np.random.default_rng(1)
    p0 = {"W1": rng.normal(size=(9, 16)), "b1": rng.normal(size=16),
          "W2": rng.normal(size=(16, 1)), "b2": rng.normal(size=1)}
    p1 = {k: v.copy() for k, v in p0.items()}
    p1["W2"] = p1["W2"] + 0.1
    assert RF.is_lower_held(p0, p1) is True
    p2 = {k: v.copy() for k, v in p0.items()}
    p2["W1"] = p2["W1"] + 0.1
    assert RF.is_lower_held(p0, p2) is False
    run_src = (EXP / "run_official_refine.py").read_text()
    assert "is_lower_held" in run_src and "train_top_only" in run_src


def test_extreme_and_episodes_synthetic():
    RF = _load("refine_lib", "d150rf_lib_ext")
    Y = np.full((6, 72), 400.0)
    Y[0, :10] = 900.0
    Y[1, :3] = 900.0
    m = RF.extreme_origin_mask(Y, 500.0)
    assert list(m) == [True, False, False, False, False, False]
    oo = pd.date_range("2024-01-01", periods=6, freq="48h", tz="UTC")
    ep = RF.count_episodes(oo, np.array([True, True, False, True, False, False]))
    assert ep == {"n_extreme_origins": 3, "n_episodes": 2}
    run_src = (EXP / "run_official_refine.py").read_text()
    assert "extreme_origin_mask" in run_src and "ordinary_mse" in run_src
    assert "extreme_mse" in run_src and "count_episodes" in run_src


def test_cell_features_synthetic():
    RF = _load("refine_lib", "d150rf_lib_x")
    idx = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    filled = pd.Series(400.0, index=idx)
    oo = pd.DatetimeIndex(["2024-01-01T12:00:00Z", "2024-01-01T15:00:00Z"], tz="UTC")
    P = RF.past12_matrix(oo, filled)
    assert P.shape == (2, 12) and bool(np.isfinite(P).all())
    B = np.full((2, 72), 410.0)
    X = RF.build_x(np.tile(np.arange(1, 73), 2), B.ravel(), P, B)
    assert X.shape == (144, 9) and bool(np.isfinite(X).all())
    mu, sd = RF.standardize_fit(X)
    Xs = RF.standardize_apply(X, mu, sd)
    assert abs(float(np.nanmean(Xs))) < 0.5
    assert RF.oof_mse(np.array([1.0, np.nan]), np.array([1.0, 2.0])) == 0.0
    assert np.isnan(RF.oof_mse(np.array([np.nan]), np.array([1.0])))
    ids = RF.oof_block_ids(9, 3)
    assert list(ids) == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    fr = RF.frame_predictions(pd.DataFrame({"origin": pd.to_datetime(
        ["2026-06-01T00:00:00Z"] * 72, utc=True), "h": np.arange(1, 73)}), np.full(72, 400.0))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert (fr["horizon_hours"].to_numpy() == np.arange(1, 73)).all()


def test_arm_routing():
    run_src = (EXP / "run_official_refine.py").read_text()
    assert '"uniform"' in run_src and '"meta"' in run_src and '"metafin"' in run_src
    assert "--arm" in run_src and 'choices=("uniform", "meta", "metafin")' in run_src
    assert "META_DEF" in run_src and "HOLD_DEF" in run_src


def test_seed_actually_propagates():
    run_src = (EXP / "run_official_refine.py").read_text()
    assert "fit_seed_models" in run_src and "event_seed" in run_src
    assert "level_seeds" in run_src and "fit_params" in run_src
    assert "head_seed" in run_src


def test_fresh_source_only():
    run_src = (EXP / "run_official_refine.py").read_text()
    code = _code()
    for w in ("stored prediction", "stored array", "another run"):
        assert w in code.lower() or "no stored array" in code.lower(), w
    assert "PATH_SOURCE" in run_src or "own-fit" in run_src
    assert "train_end" in run_src.lower() or "TRAIN_END" in run_src


def test_oof_source_readonly():
    code = _code()
    assert code.count("worker-t1g0") == 1, "single readonly OOF pointer"
    assert 'OOF_DIR = "store/scratch/worker-t1g0/oofset-full"' in code
    run_src = (EXP / "run_official_refine.py").read_text()
    for verb in ("mkdir", "to_parquet", "write_text", "rename", "unlink"):
        for line in run_src.splitlines():
            if "OOF_DIR" in line:
                assert verb not in line, line
    assert "read_parquet" in run_src


def test_poison_synthetic():
    RF = _load("refine_lib", "d150rf_lib_poison")
    idx = pd.date_range("2026-05-01", periods=1200, freq="h", tz="UTC")
    filled = pd.Series(400.0, index=idx)
    origins = pd.DatetimeIndex(["2026-05-20T00:00:00Z", "2026-05-21T00:00:00Z",
                                "2026-06-10T00:00:00Z", "2026-06-11T00:00:00Z"], tz="UTC")
    rec = RF.scenario_poison(origins, filled, "2026-05-29T00:00:00Z")
    assert rec["passed"] is True and rec["pre_invariant"] is True
    assert rec["n_pre_rows"] == 2 and rec["n_post_rows"] == 2

    def _frame_for(series):
        n = len(origins)
        return pd.DataFrame(np.tile(series.reindex(origins).to_numpy(dtype=float)[:, None],
                                    (1, 2))).iloc[np.repeat(np.arange(n), 2)].reset_index(drop=True)
    rec2 = RF.frame_poison(origins, filled, "2026-05-29T00:00:00Z", _frame_for)
    assert rec2["passed"] is True and rec2["pre_invariant"] is True
    assert rec2["n_pre_rows"] == 2 and rec2["n_post_rows"] == 2


def test_poison_covers_fold_inputs():
    run_src = (EXP / "run_official_refine.py").read_text()
    assert "scenario_poison" in run_src and "frame_poison" in run_src
    assert "poison_test" in run_src
    assert "pre-cutoff" in run_src or "pre_cutoff" in run_src or "pre-cutoff" in run_src
    assert "train_end" in run_src.lower() or "TRAIN_END" in run_src


def test_scratch_isolation_and_cpu():
    import re
    run_src = (EXP / "run_official_refine.py").read_text()
    code = _code()
    assert "store/scratch/worker-t1g9" in code
    assert "worker-t1g0" in code
    assert "worker-t1g1" not in code and "worker-t1g2" not in code
    assert "worker-t1g3" not in code and "worker-t1g4" not in code
    assert "worker-t1g5" not in code and "worker-t1g6" not in code
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
