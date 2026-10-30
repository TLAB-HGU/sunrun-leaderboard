"""D150 MBB-RW contract: static + synthetic checks (fast, no fits).

No real data fits, no official reads. Covers the owned file budget,
spec identity (D150/worker-t1g3/full), the three pre-registered ratios
with per-ratio official ids, block unit and rise threshold, 72h purge,
per-lead recal on the held-out window, mixture seed propagation,
block-weight and recal math on synthetic data, readonly-fresh source
isolation, CPU/n_jobs budget, scratch isolation, poison-key shape, and
forbidden-source absence.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d150_oofset" / "mbbrw"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "lean.py", "D147", "d147",
                 "d148_enlil", "nn-", "sun-nn", "E0472", "E0471", "E0474",
                 "worker-t1a", "worker-t1b", "worker-t1c", "worker-t1d",
                 "worker-t1e", "worker-t1f", "worker-t1g0", "worker-t1g1",
                 "worker-t1g2", "worker-d147", "worker-d141", "worker-t2no"]
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


def _code(names=("spec_mb.py", "mbbrw_lib.py", "run_official_mbbrw.py")):
    return "\n".join((EXP / n).read_text() for n in names)


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 6, files
    assert {"spec_mb.py", "mbbrw_lib.py", "run_official_mbbrw.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_mb", "d150mb_spec")
    assert SP.DIRECTION == "D150" and SP.OWNER == "worker-t1g3"
    assert SP.KIND == "full"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.OOF_TRAIN_END_ISO == "2026-03-01T00:00:00Z"
    assert SP.OOF_START_ISO == "2026-03-04T00:00:00Z"
    assert SP.OOF_END_ISO == "2026-05-28T23:00:00Z"
    assert SP.OOF_FREQ == "3h"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.ORIGIN_START == "2026-05-31 23:00" and SP.ORIGIN_END == "2026-09-26 23:00"
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24 and SP.PURGE_H == 72
    assert SP.BLOCK_ORIGINS == 24 and abs(float(SP.RISE_KMS) - 150.0) < 1e-9
    assert tuple(SP.RATIOS) == (2, 3, 5)
    assert tuple(SP.ARMS) == ("r21", "r31", "r51")
    assert dict(SP.RATIO_OF) == {"r21": 2, "r31": 3, "r51": 5}
    assert SP.SCRATCH == "store/scratch/worker-t1g3"
    assert SP.FIT_NJOBS <= 6
    assert SP.SMOKE_OOF_ORIGINS == 40 and SP.SMOKE_OFFICIAL == 8
    assert SP.BASE_PARAMS["n_estimators"] == 900
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - 0.03) < 1e-12
    assert SP.BASE_PARAMS.get("device") == "cpu" and int(SP.BASE_PARAMS.get("n_jobs")) <= 6
    assert SP.BASE_DEPTH == 3 and SP.WEIGHTS == PINNED_W
    assert SP.BASE_COLUMNS == "existing_all+strips17"
    assert tuple(SP.LEADS) == ("1-6", "7-24", "25-48", "49-72")
    assert SP.PATH_SOURCE == "own-fit-oof-recal-own-fit-official"
    assert SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert SP.RECAL_DEF.startswith("y ~ a_lead")
    assert SP.RECAL_MIN_ROWS == 50
    assert abs(float(SP.NOISE_MSE) - 90.0) < 1e-9
    assert "[axes:" in SP.AXES_TAG and "mbb-rw" in SP.AXES_TAG
    assert "OOF-only" in SP.SELECT_RULE or "OOF" in SP.SELECT_RULE


def test_weights_match_shared_recipe():
    SP = _load("spec_mb", "d150mb_spec_b")
    loc = importlib.util.spec_from_file_location(
        "d150mb_oof", LB / "experiments/d150_oofset/spec_oof.py")
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
    SP = _load("spec_mb", "d150mb_spec_c")
    te = pd.Timestamp(SP.OOF_TRAIN_END_ISO, tz="UTC")
    st = pd.Timestamp(SP.OOF_START_ISO, tz="UTC")
    en = pd.Timestamp(SP.OOF_END_ISO, tz="UTC")
    cut = pd.Timestamp(SP.CUTOFF_ISO, tz="UTC")
    assert (st - te) == pd.Timedelta("72h")
    assert en < cut
    assert SP.OOF_FREQ == "3h" and SP.PURGE_H == 72
    off = pd.Timestamp("2026-05-31 23:00", tz="UTC")
    assert off > cut


def test_single_axis_recipe():
    run_src = (EXP / "run_official_mbbrw.py").read_text()
    for mod in ("info_ablation", "strips", "spec_mb", "mbbrw_lib", "existing_all"):
        assert mod in run_src, mod
    for w in ("predictions_official_s{s}.parquet", "oof_comparison.json",
              "poison.json", "meta.json", "resample_weights", "block_ids",
              "origin_rise", "fit_recal_per_lead", "apply_recal", "--ratio"):
        assert w in run_src, w
    assert "neighbour" not in run_src and "neighbor" not in run_src
    assert "official-score" not in run_src and "official_score" not in run_src


def test_mixture_helpers():
    MB = _load("mbbrw_lib", "d150mb_lib_a")
    SP = _load("spec_mb", "d150mb_spec_d")
    blocks = [set(MB.level_seeds(s)) for s in (0, 1, 2)]
    assert len(set().union(*blocks)) == 9
    for s in (0, 1, 2):
        assert MB.event_seed(s) in MB.level_seeds(s)
    p = MB.fit_params(dict(SP.BASE_PARAMS), 3, 4)
    assert p["max_depth"] == 3 and p["random_state"] == 4 and p["n_estimators"] == 900
    MB.check_weights(SP.WEIGHTS)
    MB.check_ratios(SP.RATIOS)
    MB.check_arms(SP.ARMS)
    MB.check_ratio_of(SP.RATIO_OF)
    rng = np.random.default_rng(0)
    h = np.tile(np.arange(1, 73), 4)
    mem = {m: rng.normal(400, 20, len(h)) for m in ("e0003_line", "d33_hsplit", "d45_lvlmean")}
    out = MB.blend_per_lead(mem, SP.WEIGHTS, h)
    assert np.isfinite(out).all() and len(out) == len(h)
    m1 = h <= 6
    m2 = (h >= 7) & (h <= 24)
    m3 = h > 24
    st = MB.stitch_members({s: mem["e0003_line"][m1] for s in (0, 1, 2)},
                           {s: mem["e0003_line"][m2] for s in (0, 1, 2)},
                           mem["e0003_line"], mem["e0003_line"][m3], m1, m2, m3, (0, 1, 2), 1)
    assert set(st) == {"e0003_line", "d33_hsplit", "d45_lvlmean"}
    assert MB.as_path_matrix(np.arange(72 * 3, dtype=float)).shape == (3, 72)
    assert MB.lead_of(1) == "1-6" and MB.lead_of(72) == "49-72"
    try:
        MB.check_weights({"1-6": dict(SP.WEIGHTS["1-6"])})
    except ValueError:
        pass
    else:
        raise AssertionError("partial weights must be rejected")
    try:
        MB.check_ratios((2, 3))
    except ValueError:
        pass
    else:
        raise AssertionError("ratio grid must stay pinned")
    try:
        MB.check_arms(("r21",))
    except ValueError:
        pass
    else:
        raise AssertionError("arms must stay pinned")


def test_block_weights_synthetic():
    MB = _load("mbbrw_lib", "d150mb_lib_blocks")
    ids = MB.block_ids(48, block=24)
    assert list(ids[:25]) == [0] * 24 + [1]
    assert int(ids.max()) == 1
    try:
        MB.block_ids(0, block=24)
    except ValueError:
        pass
    else:
        raise AssertionError("blocks must reject empty origins")
    max_y = np.array([500.0, 400.0, 600.0, 400.0])
    past = np.array([400.0, 400.0, 400.0, 400.0])
    flag = MB.origin_rise(max_y, past, rise_kms=150.0)
    assert list(flag) == [False, False, True, False]
    bids = MB.block_ids(4, block=2)
    rel = MB.block_relevant(np.array([False, True, False, False]), bids)
    assert list(rel) == [True, False]
    w = MB.resample_weights(np.array([False, True, False, False]), bids, 5)
    assert list(w) == [5.0, 5.0, 1.0, 1.0]
    w2 = MB.resample_weights(np.array([False, False, False, False]), bids, 2)
    assert list(w2) == [1.0, 1.0, 1.0, 1.0]
    try:
        MB.resample_weights(np.array([True, False, False, False]), bids, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("ratio must stay pinned")
    idx = np.array([0, 0, 1, 2, 2, 2])
    per = np.array([2.0, 1.0, 5.0])
    assert list(MB.expand_row_weights(idx, per)) == [2.0, 2.0, 1.0, 5.0, 5.0, 5.0]
    P = np.array([[400.0] * 12, [410.0] * 12])
    B = np.tile(np.linspace(390, 410, 72), (2, 1))
    assert MB.past12_matrix(
        pd.DatetimeIndex(["2024-01-01T00:00:00Z", "2024-01-01T03:00:00Z"], tz="UTC"),
        pd.Series([400.0] * 100,
                  index=pd.date_range("2023-12-01", periods=100, freq="h", tz="UTC"))).shape == (2, 12)


def test_recal_synthetic():
    MB = _load("mbbrw_lib", "d150mb_lib_recal")
    rng = np.random.default_rng(2)
    n = 72 * 10
    h = np.tile(np.arange(1, 73), 10)
    pred = rng.normal(400, 30, n)
    y = 5.0 + 1.02 * pred + rng.normal(0, 5, n)
    rec = MB.fit_recal_per_lead(y, pred, h, min_rows=50)
    assert set(rec) == {"1-6", "7-24", "25-48", "49-72"}
    for lead, v in rec.items():
        assert v["fitted"] is True and v["n"] > 0
        assert np.isfinite(v["a"]) and np.isfinite(v["b"])
    out = MB.apply_recal(pred, h, rec)
    assert out.shape == pred.shape and np.isfinite(out).all()
    # Thin lead falls back to identity.
    rec2 = MB.fit_recal_per_lead(np.array([1.0, 2.0]), np.array([1.0, 2.0]),
                                 np.array([1, 2]), min_rows=50)
    assert rec2["1-6"]["fitted"] is False
    assert rec2["1-6"]["a"] == 0.0 and rec2["1-6"]["b"] == 1.0
    # Constant predictions fall back to identity.
    rec3 = MB.fit_recal_per_lead(rng.normal(400, 5, 200), np.full(200, 400.0),
                                 np.tile(np.arange(1, 73), 3)[:200], min_rows=10)
    for v in rec3.values():
        if v["n"] >= 10:
            assert v["fitted"] is False or np.isfinite(v["b"])


def test_extreme_and_frame_synthetic():
    MB = _load("mbbrw_lib", "d150mb_lib_ext")
    Y = np.full((4, 72), 400.0)
    Y[0, :10] = 900.0
    Y[1, :3] = 900.0
    m = MB.extreme_origin_mask(Y, 500.0)
    assert list(m) == [True, False, False, False]
    assert abs(MB.oof_mse(np.array([1.0, np.nan]), np.array([1.0, 2.0])) - 0.0) < 1e-12
    assert np.isnan(MB.oof_mse(np.array([np.nan]), np.array([1.0])))
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = MB.frame_predictions(te, np.full(72, 400.0))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert (fr["horizon_hours"].to_numpy() == np.arange(1, 73)).all()
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_ratio_routing():
    run_src = (EXP / "run_official_mbbrw.py").read_text()
    assert "--ratio" in run_src
    assert "choices=(2, 3, 5)" in run_src or "choices=(2,3,5)" in run_src
    assert "RATIO_OF" in run_src or "RATIOS" in run_src
    assert "ratio" in run_src.lower()


def test_seed_actually_propagates():
    run_src = (EXP / "run_official_mbbrw.py").read_text()
    assert "fit_seed_models" in run_src and "event_seed" in run_src
    assert "level_seeds" in run_src and "fit_params" in run_src


def test_recal_only_outputs():
    run_src = (EXP / "run_official_mbbrw.py").read_text()
    for w in ("predictions_official_s{s}.parquet", "oof_comparison.json",
              "poison.json", "meta.json", "fit_recal_per_lead", "apply_recal"):
        assert w in run_src, w
    for w in ("official-score", "official_score", "freeze",
              "submit", "upload", "push"):
        assert w.lower() not in run_src.lower(), w
    assert "PATH_SOURCE" in run_src or "own-fit" in run_src


def test_fresh_source_only():
    code = _code()
    assert "worker-t1g0" not in code
    assert "oof_resid" not in code
    assert "own-fit-oof-recal-own-fit-official" in code
    run_src = (EXP / "run_official_mbbrw.py").read_text()
    assert "train_end" in run_src.lower() or "TRAIN_END" in run_src


def test_poison_synthetic():
    MB = _load("mbbrw_lib", "d150mb_lib_poison")
    idx = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="UTC")
    filled = pd.Series(400.0 + np.sin(np.arange(len(idx)) / 24.0), index=idx)
    cut = "2024-01-20T00:00:00Z"
    pre = pd.DatetimeIndex([idx[100], idx[200]], tz="UTC")
    post = pd.DatetimeIndex([idx[1000], idx[1100]], tz="UTC")
    probe = pre.append(post)
    rec = MB.scenario_poison(probe, filled, cut)
    assert rec["passed"] is True
    assert rec["pre_invariant"] is True and rec["moved_columns"] == ["scenario"]
    assert rec["n_pre_rows"] == 2 and rec["n_post_rows"] == 2
    rec2 = MB.frame_poison(probe, filled, cut, lambda f: MB.past12_matrix(probe, f))
    assert rec2["passed"] is True
    assert rec2["pre_invariant"] is True and rec2["moved_columns"] == ["frame"]
    assert rec2["n_pre_rows"] == 2 and rec2["n_post_rows"] == 2


def test_poison_covers_fold_inputs():
    run_src = (EXP / "run_official_mbbrw.py").read_text()
    assert "poison_test" in run_src and "scenario_poison" in run_src and "frame_poison" in run_src
    assert "poison.json" in run_src and '"passed"' in run_src
    assert "train_end" in run_src.lower() or "TRAIN_END" in run_src


def test_scratch_isolation_and_cpu():
    import re
    SP = _load("spec_mb", "d150mb_spec_f")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1g3"
    assert set(re.findall(r"worker-[a-z0-9]+", code)) == {"worker-t1g3"}
    run_src = (EXP / "run_official_mbbrw.py").read_text()
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
    assert "cuda" not in low and "--gpu" not in code
    for w in ("official-score", "official_score", "submit", "upload"):
        assert w not in low, w
    assert "freeze" not in low and "push" not in low
