"""D153 cascade contract: static plus synthetic checks (fast, no fits)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d153_shifts/cascade"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs", "predictions_official", "origin_last_input_utc",
             "pred_kms", "official_score", "official-score",
             "from scorer", "import scorer", "from expos", "import expos",
             "freeze", "submit", "isMostAccurate", "versionId"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "lean.py",
                 "D141", "D147", "D148", "D151", "D152",
                 "d141", "d147", "d148", "d151", "d152",
                 "d151_e0472_audit", "d151_enlil_ext", "d148_enlil",
                 "E0471", "E0472", "e0471", "e0472", "E0512",
                 "nn-", "sun-nn", "worker-t1a", "worker-t1b",
                 "worker-t1c", "worker-t1d", "worker-t1e", "worker-t1f",
                 "worker-t1g", "worker-t1h0", "worker-t1h1", "worker-t1h2",
                 "worker-t1h3", "worker-t1h4", "worker-t2no",
                 "donki-live"]
BANNED_SRC = ["GONG", "CDAW", "LASCO"]
CELL_LABELS = ["low_low_high", "low_high_high", "high_low_high", "high_high_high",
               "hlh", "llh", "lhh", "hhh"]
OFFICIAL_TOKENS = ["2026-05-31", "2026-09-26", "N_OFFICIAL_ORIGINS",
                   "ORIGIN_START", "ORIGIN_END", "lb-eval", "lb-best"]

sys.path.insert(0, str(EXP))


def _load(mod_name, alias):
    loc = importlib.util.spec_from_file_location(alias, EXP / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code(names=("spec_cascade.py", "cascade_lib.py", "run_diag_cascade.py")):
    return "\n".join((EXP / n).read_text() for n in names)


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 4, files
    assert {"spec_cascade.py", "cascade_lib.py", "run_diag_cascade.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_cascade", "d153cas_spec")
    assert SP.DIRECTION == "D153" and SP.OWNER == "worker-t1h5"
    assert SP.KIND == "diag" and SP.REFERENCE == "E0408"
    assert tuple(SP.SEEDS) == (0,) and SP.SMOKE_SEED == 0
    assert SP.SCRATCH == "store/scratch/worker-t1h5"
    assert int(SP.N_JOBS) == 4 and int(SP.N_JOBS) <= 6
    assert int(SP.FIT_NJOBS) <= 6
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.TRAIN_FREQ == "3h" and int(SP.SPLIT_H) == 24
    assert int(SP.BASE_DEPTH) == 3
    assert int(SP.BASE_PARAMS.get("n_estimators")) == 900
    assert abs(float(SP.BASE_PARAMS.get("learning_rate")) - 0.03) < 1e-12
    assert SP.BASE_PARAMS.get("device") == "cpu"
    assert int(SP.BASE_PARAMS.get("n_jobs")) <= 6
    assert SP.BASE_COLUMNS == "existing_all+strips17"
    assert SP.MAIN_WINDOW == "W2-main"
    assert SP.OOF_TRAIN_END_ISO == "2026-03-01T00:00:00Z"
    assert SP.OOF_START_ISO == "2026-03-04T00:00:00Z"
    assert SP.OOF_END_ISO == "2026-05-28T23:00:00Z"
    assert SP.OOF_FREQ == "3h"
    assert abs(float(SP.THRESHOLD_Q) - 0.95) < 1e-12
    assert abs(float(SP.EXTREME_FRAC) - 0.05) < 1e-12
    assert "P95" in SP.THRESHOLD_RULE and "5pct" in SP.THRESHOLD_RULE
    assert "fixed" in SP.EPISODE_DEF.lower() and "extreme" in SP.EPISODE_DEF.lower()
    assert SP.PATH_SOURCE == "oof-fit"
    assert SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert tuple(SP.AMPLITUDE_STAGES) == ("oracle", "oos")
    assert "oracle" in SP.AMPLITUDE_DEF.lower() and "oos" in SP.AMPLITUDE_DEF.lower()
    assert "retention" in SP.OOS_DEF.lower()
    assert tuple(float(x) for x in SP.TIMING_SCALES) == (1.0, 0.5, 0.25, 0.0)
    assert "pre-cutoff" in SP.TIMING_DEF.lower() or "pre cutoff" in SP.TIMING_DEF.lower()
    assert "before cutoff" in SP.PAIR_RULE.lower() and "at or before event" in SP.PAIR_RULE.lower()
    assert "6" in SP.PAIR_RULE and "200" in SP.PAIR_RULE
    assert tuple(float(x) for x in SP.DETECTION_P) == (0.5, 0.75, 1.0)
    assert "mixture" in SP.DETECTION_DEF.lower() or "p" in SP.DETECTION_DEF.lower()
    assert "interpolation" in SP.BREAKEVEN_DEF.lower()
    assert abs(float(SP.CLOSE_MAE_H) - 11.0) < 1e-12
    assert "close" in SP.CLOSE_RULE.lower() and "11" in SP.CLOSE_RULE
    assert abs(float(SP.NOISE_FLOOR) - 90.0) < 1e-12
    assert "targets <" in SP.VINTAGE_RULE and "<= origin" in SP.VINTAGE_RULE
    assert abs(float(SP.REF_MSE) - 3281.31) < 1e-9
    assert SP.DECISION == "diagnostic-only-no-adoption"
    assert "no registration" in SP.DIAG_NOTE and "no adoption" in SP.DIAG_NOTE
    assert "ShiftSafe" in SP.DIFFERENCE and "gate" in SP.DIFFERENCE.lower()
    assert "E0408" in SP.DIFFERENCE
    assert SP.NO_OFFICIAL and SP.NO_SELECTION and SP.NO_REGISTRATION
    assert int(SP.SMOKE_ORIGINS) == 8
    assert SP.POISON_CUTOFF_ISO == "2026-04-16T00:00:00Z"
    assert tuple(SP.COND_OBS) == ("v0", "half", "ambient", "run_count", "revision_count", "lead")
    assert "vintage-safe" in SP.COND_DEF.lower() and "no catalogue" in SP.COND_DEF.lower()
    assert int(SP.COND_BLOCKS) == 4 and int(SP.COND_PURGE_H) == 72
    assert int(SP.COND_MIN_N) == 10
    assert abs(float(SP.COND_SHARE_MIN) - 0.20) < 1e-12
    assert abs(float(SP.COND_SSE_MIN) - 0.10) < 1e-12
    assert "close" in SP.COND_CLOSE_RULE.lower() and "arrival" in SP.COND_CLOSE_RULE.lower()


def test_windows_cover_w0_w2():
    SP = _load("spec_cascade", "d153cas_spec_b")
    assert [w["name"] for w in SP.XOOF_WINDOWS] == ["W0-early", "W1-mid", "W2-main"]
    assert SP.XOOF_WINDOWS[0]["train_end"] == "2026-01-01T00:00:00Z"
    assert SP.XOOF_WINDOWS[0]["oof_start"] == "2026-01-04T00:00:00Z"
    assert SP.XOOF_WINDOWS[0]["oof_end"] == "2026-01-31T23:00:00Z"
    assert SP.XOOF_WINDOWS[1]["train_end"] == "2026-02-01T00:00:00Z"
    assert SP.XOOF_WINDOWS[2]["train_end"] == "2026-03-01T00:00:00Z"
    assert SP.XOOF_WINDOWS[2]["oof_end"] == "2026-05-28T23:00:00Z"
    for w in SP.XOOF_WINDOWS:
        assert w["freq"] == "3h"
        assert pd.Timestamp(w["oof_end"], tz="UTC") < pd.Timestamp(SP.CUTOFF_ISO, tz="UTC")
        assert pd.Timestamp(w["train_end"], tz="UTC") < pd.Timestamp(w["oof_start"], tz="UTC")
    for a, b in zip(SP.XOOF_WINDOWS, SP.XOOF_WINDOWS[1:]):
        assert pd.Timestamp(b["train_end"], tz="UTC") > pd.Timestamp(a["train_end"], tz="UTC")


def test_episode_and_insertion_synthetic():
    CL = _load("cascade_lib", "d153cas_lib_ep")
    thr = CL.threshold_p95(np.array([300.0, 400.0, 500.0, 600.0]), q=0.95)
    assert 500.0 < thr <= 600.0
    m = np.zeros((2, 72))
    m[0, :4] = 9999.0
    flags = CL.extreme_origin_mask(m, 500.0, frac=0.05)
    assert list(flags) == [True, False]
    try:
        CL.extreme_origin_mask(np.zeros((2, 71)), 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-72 width must be rejected")
    r = np.arange(72, dtype=float)
    s0 = CL.shift_residual(r, 0.0)
    assert np.array_equal(s0, r)
    s1 = CL.shift_residual(r, 1.0)
    assert s1[0] == 0.0 and s1[1] == r[0] and s1[71] == r[70]
    sm1 = CL.shift_residual(r, -1.0)
    assert sm1[0] == r[1] and sm1[71] == 0.0
    yb = np.full(72, 400.0)
    y_ins = CL.apply_insertion(yb, r, 0.0)
    assert np.array_equal(y_ins, yb + r)
    ym = CL.mixture_prediction(yb, y_ins, 1.0)
    assert np.array_equal(ym, y_ins)
    ym2 = CL.mixture_prediction(yb, y_ins, 0.5)
    assert np.allclose(ym2, 0.5 * y_ins + 0.5 * yb)
    try:
        CL.mixture_prediction(yb, y_ins, 1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("p outside 0..1 must be rejected")
    mm = CL.mse(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert mm["n"] == 2 and abs(mm["mse"]) < 1e-12
    yt = np.stack([np.linspace(400, 800, 72), np.linspace(400, 600, 72)])
    yb2 = np.stack([np.linspace(400, 500, 72), np.linspace(400, 450, 72)])
    ret = CL.oos_retention(yt, yb2)
    assert 0.0 <= ret <= 2.0 and ret < 1.0
    assert abs(CL.timing_mae(np.array([1.0, -2.0, 3.0])) - 2.0) < 1e-12
    be = CL.breakeven_mae([0.0, 5.0, 10.0], [-100.0, -20.0, 50.0])
    assert 5.0 < be < 10.0
    assert CL.breakeven_mae([0.0, 5.0], [10.0, 20.0]) == 0.0
    assert CL.breakeven_mae([0.0, 5.0], [-10.0, -5.0]) == float("inf")
    g = CL.decide_gate(5.0, 500.0, close_mae=11.0, floor=90.0)
    assert g["decision"] == "CLOSE"
    g2 = CL.decide_gate(15.0, 500.0, close_mae=11.0, floor=90.0)
    assert g2["decision"] == "OPEN"
    g3 = CL.decide_gate(15.0, 10.0, close_mae=11.0, floor=90.0)
    assert g3["decision"] == "CLOSE"


def test_conditional_synthetic():
    CL = _load("cascade_lib", "d153cas_lib_cond")
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    lo, mid, hi = CL.quartile_bounds(v)
    assert lo < mid < hi
    g = CL.assign_quartile(np.array([0.0, lo, mid, hi, 99.0, np.nan]), (lo, mid, hi))
    assert list(g) == [0, 1, 2, 3, 3, -1]
    try:
        CL.quartile_bounds(np.array([1.0, 1.0, 1.0, 1.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("degenerate quartiles must be rejected")
    blocks = CL.block_indices(8, 4)
    assert len(blocks) == 4 and sum(len(b) for b in blocks) == 8
    try:
        CL.block_indices(3, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("too few rows must be rejected")
    t = np.array([0, 10, 20, 30, 100, 110, 120, 130], dtype=np.int64) * 3600000000000
    keep = CL.purge_train_mask(t, np.array([2, 3]), 72 * 3600000000000)
    assert keep[0] is np.True_ or bool(keep[0]) is False
    assert bool(keep[2]) is False and bool(keep[3]) is False
    qm = CL.quartile_mae(np.array([1.0, -2.0, 3.0, -4.0]), np.array([0, 0, 1, 1]))
    assert qm[0]["n"] == 2 and abs(qm[0]["mae"] - 1.5) < 1e-12
    e1 = CL.decide_ensemble_gate(8.0, 11.0, 0.25, 0.15)
    assert e1["decision"] == "OPEN"
    e2 = CL.decide_ensemble_gate(12.0, 11.0, 0.25, 0.15)
    assert e2["decision"] == "CLOSE"
    e3 = CL.decide_ensemble_gate(8.0, 11.0, 0.10, 0.15)
    assert e3["decision"] == "CLOSE"


def test_poison_synthetic():
    CL = _load("cascade_lib", "d153cas_lib_pois")
    idx = pd.date_range("2023-05-01T00:00:00Z", "2023-07-01T00:00:00Z", freq="h", tz="UTC")
    filled = pd.Series(400.0 + np.sin(np.arange(len(idx)) / 24.0), index=idx)
    cut = "2023-05-29T00:00:00Z"
    origins = pd.DatetimeIndex(["2023-05-10T00:00:00Z", "2023-05-20T00:00:00Z",
                                "2023-06-05T00:00:00Z", "2023-06-10T00:00:00Z"], tz="UTC")
    sc = CL.scenario_poison(origins, filled, cut)
    assert sc["passed"] is True and sc["pre_invariant"] is True and sc["moved"] is True
    assert sc["n_pre_rows"] == 2 and sc["n_post_rows"] == 2

    def _frame_for(series):
        o = pd.DatetimeIndex(origins, tz="UTC")
        return np.tile(series.reindex(o, method="ffill").to_numpy(dtype=float)[:, None], (1, 4))

    fp = CL.frame_poison(origins, filled, cut, _frame_for)
    assert fp["passed"] is True and fp["pre_invariant"] is True


def test_diag_only_outputs():
    run_src = (EXP / "run_diag_cascade.py").read_text()
    for w in ("diag_cascade.json", "causality.json", "meta.json"):
        assert w in run_src, w
    for w in ("NO_OFFICIAL", "NO_SELECTION", "NO_REGISTRATION"):
        assert w in run_src, w
    for w in ("scenario_poison", "frame_poison", '"passed"', "cutoff", "n_jobs"):
        assert w in run_src, w
    for w in ("conditional_report", "quartile", "purge", "ambient", "run_count"):
        assert w in run_src, w
    for w in ("predictions_official", "origin_last_input_utc", "pred_kms",
              "official_score", "official-score", "isMostAccurate", "--gpu",
              "freeze", "submit", "lb-eval", "lb-best",
              "2026-05-31", "2026-09-26", "N_OFFICIAL_ORIGINS",
              "ORIGIN_START", "ORIGIN_END"):
        assert w not in run_src, w


def test_scratch_isolation_and_cpu():
    import re
    SP = _load("spec_cascade", "d153cas_spec_e")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1h5"
    assert len(set(re.findall(r"worker-[a-z0-9]+", code))) == 1
    assert "worker-t1h5" in code
    assert int(SP.N_JOBS) <= 6
    assert int(SP.BASE_PARAMS.get("n_jobs")) <= 6
    assert SP.BASE_PARAMS.get("device") == "cpu"
    assert "--gpu" not in code


def test_vintage_and_difference():
    SP = _load("spec_cascade", "d153cas_spec_f")
    run_src = (EXP / "run_diag_cascade.py").read_text()
    assert "targets <" in SP.VINTAGE_RULE and "<= origin" in SP.VINTAGE_RULE
    assert "cutoff" in run_src.lower()
    assert "ShiftSafe" in SP.DIFFERENCE
    assert "E0408" in SP.DIFFERENCE
    assert "strips" in run_src and "select_slots" in run_src
    assert "strip_features" in run_src or "strip_columns" in run_src
    assert "store/donki-cme" in run_src
    assert "CME_" in run_src and "IPS_" in run_src
    assert "vintage-safe" in SP.COND_DEF.lower()


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
    for w in OFFICIAL_TOKENS:
        assert w not in code, w
    assert "cuda" not in low and "--gpu" not in code
