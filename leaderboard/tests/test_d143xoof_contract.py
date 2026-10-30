"""D143 XOOF contract: static + synthetic checks (fast, no fits, no data)."""
import importlib.util
import sys

import numpy as np
import pandas as pd
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d143_axisA_analog" / "reshead"
XOOF = EXP / "xoof"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_IDS = ["e0147-abl", "d141_e0147", "ablate", "D141", "E0443", "E0446", "E0414", "E0438",
             "worker-d76", "worker-t1d7",
             "worker-t1a1", "worker-t1a2", "worker-t1a5", "worker-t1a6", "worker-t1a7",
             "worker-t1a8", "worker-t1b0", "worker-t1b1", "worker-t1b2", "worker-t1b3",
             "worker-t1c0", "worker-t1c1", "worker-t1c2", "worker-d135", "worker-d133",
             "worker-t2no", "nn-"]
BANNED_SRC = ["GONG", "WSA", "ADAPT", "CME", "DONKI"]


def _load_xoof(name):
    loc = importlib.util.spec_from_file_location(f"d143xoof_{name}", XOOF / f"{name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _load_parent(name):
    loc = importlib.util.spec_from_file_location(f"d143parent_{name}", EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(XOOF.glob("*.py")))


def test_owned_file_budget():
    parent_files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in parent_files if f.endswith(".py")]
    assert {"spec_reshead.py", "reshead_lib.py",
            "run_official_reshead.py"} <= set(pys), parent_files
    files = sorted(p.name for p in XOOF.iterdir() if p.is_file())
    xpys = [f for f in files if f.endswith(".py")]
    assert 2 <= len(xpys) <= 6, files
    assert {"spec_xoof.py", "run_official_xoof.py"} <= set(xpys), files


def test_spec_identity_and_scope():
    SP = _load_xoof("spec_xoof")
    assert SP.DIRECTION == "D143" and SP.OWNER == "worker-t1b4"
    assert SP.KIND == "full" and SP.REFERENCE == "E0421"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h"
    assert SP.SCRATCH == "store/scratch/worker-t1b4"
    assert SP.FIT_NJOBS <= 6 and SP.FIT_NJOBS + SP.HEAD_NJOBS <= 20
    assert SP.K == 15 and SP.WEIGHTING == "uniform"
    assert SP.SCENARIO_H == 12 and SP.ERR_H == 12 and SP.PATH_H == 72
    assert SP.SCENARIO_DIM == 96
    assert tuple(SP.ARMS) == ("uniform-mean", "summary-median", "xgb-head")
    assert tuple(SP.HEAD_FEATURES) == ("dv_mean", "dv_median", "dv_std", "dv_q10",
                                       "dv_q25", "dv_q75", "dv_q90", "dv_skew",
                                       "dv_var", "d2mean", "h")
    assert SP.HEAD_TARGET == "dv"
    assert SP.SUMMARY_STAT == "median"
    assert int(SP.OOF_NFOLD) == 3
    assert int(SP.HEAD_MIN_TRAIN_ROWS) >= 50
    assert abs(SP.CARRINGTON_H - 654.48) < 1e-9
    assert len(SP.BLEND_GRID) == 21 and SP.BLEND_GRID[0] == 0.0 and SP.BLEND_GRID[-1] == 1.0
    assert SP.BASE_DEPTH == 3 and SP.BASE_COLUMNS == "existing_all+strips17"
    assert SP.BASE_PARAMS.get("n_estimators") == 900
    assert abs(float(SP.BASE_PARAMS.get("learning_rate")) - 0.03) < 1e-12
    assert SP.PATH_SOURCE == "oof-fit"
    assert SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert "[axes:" in SP.AXES_TAG and "residual" in SP.AXES_TAG


def test_head_and_base_fixed_vs_parent():
    SP = _load_xoof("spec_xoof")
    EP = _load_parent("spec_reshead")
    assert dict(SP.BASE_PARAMS) == dict(EP.BASE_PARAMS)
    assert dict(SP.HEAD_PARAMS) == dict(EP.HEAD_PARAMS)
    assert tuple(SP.HEAD_FEATURES) == tuple(EP.HEAD_FEATURES)
    assert tuple(SP.ARMS) == tuple(EP.ARMS)
    assert SP.WEIGHTS == EP.WEIGHTS
    assert SP.K == EP.K and SP.WEIGHTING == EP.WEIGHTING
    assert SP.SCENARIO_DIM == EP.SCENARIO_DIM
    assert SP.SUMMARY_DEF == EP.SUMMARY_DEF and SP.HEAD_DEF == EP.HEAD_DEF
    assert SP.RESIDUAL_DEF == EP.RESIDUAL_DEF and SP.HEAD_TARGET == EP.HEAD_TARGET
    assert SP.PATH_SOURCE == EP.PATH_SOURCE and SP.SUMMARY_STAT == EP.SUMMARY_STAT
    assert SP.OOF_TRAIN_END_ISO == EP.OOF_TRAIN_END_ISO
    assert SP.OOF_START_ISO == EP.OOF_START_ISO and SP.OOF_END_ISO == EP.OOF_END_ISO


def test_windows_prespecified_and_causal():
    SP = _load_xoof("spec_xoof")
    wins = list(SP.XOOF_WINDOWS)
    assert len(wins) == 3
    assert [w["name"] for w in wins] == ["W0-early", "W1-mid", "W2-main"]
    assert SP.MAIN_WINDOW == "W2-main"
    assert wins[-1]["train_end"] == SP.OOF_TRAIN_END_ISO == "2026-03-01T00:00:00Z"
    assert wins[-1]["oof_start"] == SP.OOF_START_ISO
    assert wins[-1]["oof_end"] == SP.OOF_END_ISO
    for w in wins:
        assert w["freq"] == "3h"
        assert w["train_end"] <= w["oof_start"] < w["oof_end"] < SP.CUTOFF_ISO
    for a, b in zip(wins, wins[1:]):
        assert a["train_end"] < b["train_end"], "trains must expand"
        assert a["oof_end"] < b["oof_start"], "test periods must not overlap"
    oo = pd.date_range(wins[0]["oof_start"], wins[0]["oof_end"], freq="3h", tz="UTC")
    assert len(oo) > 100, "early window must hold enough origins for a stable MSE"
    oo1 = pd.date_range(wins[1]["oof_start"], wins[1]["oof_end"], freq="3h", tz="UTC")
    assert len(oo1) > 100
    text = (XOOF / "run_official_xoof.py").read_text()
    assert "mean-across-windows" in text or "mean across windows" in text


def test_xoof_single_axis_reuse():
    SP = _load_xoof("spec_xoof")
    text = (XOOF / "run_official_xoof.py").read_text()
    assert "import spec_xoof as SP" in text
    assert "import reshead_lib as RL" in text
    assert "worker-t1b4" in text
    assert "predictions_official_s{s}.parquet" in text
    assert "resid_summary_s{s}.parquet" in text
    assert "residual_for_origins_with_summary" in text
    assert "head_feature_matrix" in text
    assert "oof_fold_ids" in text
    assert "fit_blend_weight_dv" in text
    assert "blend_head" in text
    assert "XOOF_WINDOWS" in text and "MAIN_WINDOW" in text
    for dup in ("def summary_for_curves", "def head_feature_matrix",
                "def residual_for_origins_with_summary", "def oof_fold_ids",
                "def scenario_distance3"):
        assert dup not in text, f"xoof must reuse the parent lib, not duplicate {dup}"
    sys.path.insert(0, str(LB / "experiments/headroom_audit"))
    import info_ablation as ia
    assert "max_depth" not in SP.BASE_PARAMS, "depth comes only from BASE_DEPTH"
    for k, v in ia.PARAMS.items():
        if k in ("max_depth", "n_jobs", "random_state", "n_estimators", "learning_rate"):
            continue
        assert SP.BASE_PARAMS.get(k) == v, k
    assert SP.BASE_PARAMS.get("tree_method") == "hist"
    assert SP.BASE_PARAMS.get("device") == "cpu"
    assert SP.HEAD_PARAMS.get("device") == "cpu"
    assert int(SP.HEAD_PARAMS.get("n_estimators")) <= 300
    assert int(SP.HEAD_PARAMS.get("max_depth")) <= 3
    sys.path.insert(0, str(EXP))
    import reshead_lib as RL
    RL.check_weights(SP.WEIGHTS)
    RL.check_head_features(SP.HEAD_FEATURES)
    RL.check_head_params(SP.HEAD_PARAMS)
    RL.check_arms(SP.ARMS)


def test_no_forbidden_leak_or_banned_source():
    code = _code()
    for s in FORBIDDEN:
        assert s not in code, s
    for s in BANNED_SRC:
        assert s not in code, s
    for s in OTHER_IDS:
        assert s not in code, s
    for f in sorted(XOOF.glob("*.py")):
        t = f.read_text()
        assert "core.freeze" not in t and "official-score" not in t
        assert "--gpu" not in t, f.name
        for w in ("freeze", "submit", "upload"):
            assert w not in t.lower(), (f.name, w)


def test_seed_actually_propagates():
    sys.path.insert(0, str(EXP))
    import reshead_lib as RL
    SP = _load_xoof("spec_xoof")
    params = [RL.fit_params(dict(SP.BASE_PARAMS), SP.BASE_DEPTH, s) for s in SP.SEEDS]
    assert len({p["random_state"] for p in params}) == 3
    assert all(p["max_depth"] == 3 for p in params)
    assert RL.level_seeds(1) == (3, 4, 5) and RL.event_seed(1) == 4
    assert len({RL.head_seed(s) for s in SP.SEEDS}) == 3
    assert RL.head_seed(0) not in set(RL.level_seeds(0)) | {RL.event_seed(0)}


def test_cross_window_selection_rule():
    means = {"uniform-mean": [10.0, 12.0, 11.0],
             "summary-median": [9.0, 13.0, 11.5],
             "xgb-head": [8.0, 8.5, 9.0]}
    avg = {a: float(np.mean(v)) for a, v in means.items()}
    assert avg["xgb-head"] < avg["uniform-mean"] < avg["summary-median"]
    assert int(np.nanargmin(np.array([avg["uniform-mean"], avg["summary-median"],
                                      avg["xgb-head"]]))) == 2
    bad = np.array([1.0, 2.0, float("nan")])
    assert int(np.nanargmin(bad)) in (0, 1)


def test_output_frame_schema():
    sys.path.insert(0, str(EXP))
    import reshead_lib as RL
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = RL.frame_predictions(te, np.full(72, 400.0))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert (fr["horizon_hours"].to_numpy() == np.arange(1, 73)).all()
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_poison_keys_and_invariance():
    sys.path.insert(0, str(EXP))
    import reshead_lib as RL
    idx = pd.date_range("2026-05-01", periods=24 * 70, freq="h", tz="UTC")
    filled = pd.Series(400.0 + np.sin(np.arange(len(idx)) / 24.0), index=idx)
    cut = pd.Timestamp("2026-05-29T00:00:00Z", tz="UTC")
    pre = pd.DatetimeIndex([idx[100], idx[200]], tz="UTC")
    post = pd.DatetimeIndex([idx[1200], idx[1300]], tz="UTC")
    probe = pre.append(post)
    rec = RL.scenario_poison(probe, filled, cut)
    assert rec["passed"] is True
    assert rec["pre_invariant"] is True and rec["moved_columns"] == ["scenario"]
    assert rec["n_pre_rows"] == 2 and rec["n_post_rows"] == 2
