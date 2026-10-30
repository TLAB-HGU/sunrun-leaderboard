"""D153 lag contract: static plus synthetic checks (fast, no fits)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d153_shifts/lagdiag"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs", "predictions_official", "origin_last_input_utc",
             "pred_kms", "official_score", "official-score",
             "from scorer", "import scorer", "from expos", "import expos",
             "freeze", "submit", "isMostAccurate"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "lean.py",
                 "D141", "D147", "D148", "D151", "D152",
                 "d141", "d147", "d148", "d151", "d152",
                 "d151_e0472_audit", "d151_enlil_ext", "d148_enlil",
                 "E0471", "E0472", "e0471", "e0472", "E0512",
                 "nn-", "sun-nn", "worker-t1a", "worker-t1b",
                 "worker-t1c", "worker-t1d", "worker-t1e", "worker-t1f",
                 "worker-t1g", "worker-t1h0", "worker-t1h1", "worker-t1h2",
                 "worker-t1h3", "worker-t1h4", "worker-t1h5", "worker-t2no",
                 "donki-live"]
BANNED_SRC = ["GONG", "CDAW"]
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


def _code(names=("spec_lag.py", "lag_lib.py", "run_diag_lag.py")):
    return "\n".join((EXP / n).read_text() for n in names)


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 4, files
    assert {"spec_lag.py", "lag_lib.py", "run_diag_lag.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_lag", "d153lag_spec")
    assert SP.DIRECTION == "D153" and SP.OWNER == "worker-t1h6"
    assert SP.KIND == "diag" and SP.REFERENCE == "E0408"
    assert tuple(SP.SEEDS) == (0,) and SP.SMOKE_SEED == 0
    assert SP.SCRATCH == "store/scratch/worker-t1h6"
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
    assert "shock" in SP.EPISODE_DEF.lower() and "quiet" in SP.EPISODE_DEF.lower()
    assert SP.PATH_SOURCE == "oof-fit"
    assert int(SP.LAG_MAX_H) == 12
    assert "12" in SP.LAG_DEF and "late" in SP.LAG_DEF.lower()
    assert abs(float(SP.SHOCK_DV) - 50.0) < 1e-12
    assert abs(float(SP.HSS_AMP) - 100.0) < 1e-12
    assert "truth" in SP.GROUP_DEF.lower()
    assert "first differences" in SP.DERIV_DEF.lower()
    assert "pre-cutoff" in SP.TIMING_DEF.lower() or "pre cutoff" in SP.TIMING_DEF.lower()
    assert "before cutoff" in SP.PAIR_RULE.lower() and "at or before event" in SP.PAIR_RULE.lower()
    assert "6" in SP.PAIR_RULE and "200" in SP.PAIR_RULE
    assert abs(float(SP.CLOSE_NONSHOCK_LAG_H) - 1.0) < 1e-12
    assert abs(float(SP.CLOSE_ENLIL_TOL_H) - 3.0) < 1e-12
    assert int(SP.MIN_SHOCK_N) == 20
    assert "close" in SP.CLOSE_RULE.lower() and "3 h" in SP.CLOSE_RULE.lower()
    assert abs(float(SP.NOISE_FLOOR) - 90.0) < 1e-12
    assert "targets <" in SP.VINTAGE_RULE and "<= origin" in SP.VINTAGE_RULE
    assert abs(float(SP.REF_MSE) - 3281.31) < 1e-9
    assert SP.DECISION == "diagnostic-only-no-adoption"
    assert "no registration" in SP.DIAG_NOTE and "no adoption" in SP.DIAG_NOTE
    assert "ShiftSafe" in SP.DIFFERENCE and "gate" in SP.DIFFERENCE.lower()
    assert "E0408" in SP.DIFFERENCE and "lag" in SP.DIFFERENCE.lower()
    assert SP.NO_OFFICIAL and SP.NO_SELECTION and SP.NO_REGISTRATION
    assert int(SP.SMOKE_ORIGINS) == 8
    assert SP.POISON_CUTOFF_ISO == "2026-04-16T00:00:00Z"


def test_windows_cover_w0_w2():
    SP = _load("spec_lag", "d153lag_spec_b")
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


def _ramp(at=20, height=200.0, base=400.0):
    y = np.full(72, base, dtype=float)
    y[at:] = base + height * np.minimum(1.0, (np.arange(72)[at:] - at + 1) / 2.0)
    return y


def test_groups_lag_deriv_synthetic():
    LL = _load("lag_lib", "d153lag_lib_ep")
    thr = LL.threshold_p95(np.array([300.0, 400.0, 500.0, 600.0]), q=0.95)
    assert 500.0 < thr <= 600.0
    m = np.zeros((2, 72))
    m[0, :4] = 9999.0
    flags = LL.extreme_origin_mask(m, 500.0, frac=0.05)
    assert list(flags) == [True, False]
    try:
        LL.extreme_origin_mask(np.zeros((2, 71)), 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-72 width must be rejected")
    d = LL.first_diff(np.arange(72, dtype=float))
    assert d[0] == 0.0 and d[1] == 1.0 and d[71] == 1.0
    try:
        LL.first_diff(np.zeros(71))
    except ValueError:
        pass
    else:
        raise AssertionError("non-72 path must be rejected")
    assert LL.classify_episode(_ramp(at=20), 50.0, 100.0) == "shock"
    gradual = np.full(72, 400.0)
    gradual[10:] = 400.0 + np.minimum(250.0, (np.arange(72)[10:] - 10) * 4.0)
    assert LL.classify_episode(gradual, 50.0, 100.0) == "hss"
    assert LL.classify_episode(np.full(72, 400.0), 50.0, 100.0) == "quiet"
    assert LL.classify_episode(np.full(72, np.nan), 50.0, 100.0) == "quiet"
    yt = _ramp(at=20)
    same = LL.xcorr_lag(yt, yt.copy(), 12)
    assert same["lag"] == 0 and abs(same["rbest"] - 1.0) < 1e-9
    late = np.full(72, 400.0)
    late[3:] = yt[:69]
    got = LL.xcorr_lag(yt, late, 12)
    assert got["lag"] == 3, got
    assert got["rbest"] > got["r0"]
    early = np.full(72, 400.0)
    early[:70] = yt[2:]
    got2 = LL.xcorr_lag(yt, early, 12)
    assert got2["lag"] == -2, got2
    try:
        LL.xcorr_lag(np.zeros(71), np.zeros(72), 12)
    except ValueError:
        pass
    else:
        raise AssertionError("non-72 paths must be rejected")
    dm0 = LL.deriv_mse(yt, yt.copy())
    assert dm0["n"] == 72 and abs(dm0["mse"]) < 1e-12
    dm1 = LL.deriv_mse(late, yt)
    assert dm1["mse"] > 0.0
    mm = LL.mse(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert mm["n"] == 2 and abs(mm["mse"]) < 1e-12
    s = LL.summarize(np.array([1.0, 2.0, 3.0]))
    assert s["n"] == 3 and abs(s["median"] - 2.0) < 1e-12 and abs(s["iqr"] - 1.0) < 1e-12
    assert abs(LL.timing_mae(np.array([1.0, -2.0, 3.0])) - 2.0) < 1e-12
    g = LL.decide_gate(50, 0.0, 1.0, 3.0, 3.0, tol_nonshock=1.0, tol_ref=3.0, min_n=20)
    assert g["decision"] == "CLOSE"
    g2 = LL.decide_gate(50, 0.0, 3.0, 3.0, 3.0, tol_nonshock=1.0, tol_ref=3.0, min_n=20)
    assert g2["decision"] == "OPEN"
    g3 = LL.decide_gate(50, 0.0, 0.0, 10.0, 3.0, tol_nonshock=1.0, tol_ref=3.0, min_n=20)
    assert g3["decision"] == "OPEN"
    g4 = LL.decide_gate(3, 0.0, 0.0, 3.0, 3.0, tol_nonshock=1.0, tol_ref=3.0, min_n=20)
    assert g4["decision"] == "CLOSE"


def test_poison_synthetic():
    LL = _load("lag_lib", "d153lag_lib_pois")
    idx = pd.date_range("2023-05-01T00:00:00Z", "2023-07-01T00:00:00Z", freq="h", tz="UTC")
    filled = pd.Series(400.0 + np.sin(np.arange(len(idx)) / 24.0), index=idx)
    cut = "2023-05-29T00:00:00Z"
    origins = pd.DatetimeIndex(["2023-05-10T00:00:00Z", "2023-05-20T00:00:00Z",
                                "2023-06-05T00:00:00Z", "2023-06-10T00:00:00Z"], tz="UTC")
    sc = LL.scenario_poison(origins, filled, cut)
    assert sc["passed"] is True and sc["pre_invariant"] is True and sc["moved"] is True
    assert sc["n_pre_rows"] == 2 and sc["n_post_rows"] == 2

    def _frame_for(series):
        o = pd.DatetimeIndex(origins, tz="UTC")
        return np.tile(series.reindex(o, method="ffill").to_numpy(dtype=float)[:, None], (1, 4))

    fp = LL.frame_poison(origins, filled, cut, _frame_for)
    assert fp["passed"] is True and fp["pre_invariant"] is True


def test_diag_only_outputs():
    run_src = (EXP / "run_diag_lag.py").read_text()
    for w in ("diag_lag.json", "causality.json", "meta.json"):
        assert w in run_src, w
    for w in ("NO_OFFICIAL", "NO_SELECTION", "NO_REGISTRATION"):
        assert w in run_src, w
    for w in ("scenario_poison", "frame_poison", '"passed"', "cutoff", "n_jobs"):
        assert w in run_src, w
    for w in ("predictions_official", "origin_last_input_utc", "pred_kms",
              "official_score", "official-score", "isMostAccurate", "--gpu",
              "freeze", "submit", "lb-eval", "lb-best",
              "2026-05-31", "2026-09-26", "N_OFFICIAL_ORIGINS",
              "ORIGIN_START", "ORIGIN_END"):
        assert w not in run_src, w


def test_scratch_isolation_and_cpu():
    import re
    SP = _load("spec_lag", "d153lag_spec_e")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1h6"
    assert len(set(re.findall(r"worker-[a-z0-9]+", code))) == 1
    assert "worker-t1h6" in code
    assert int(SP.N_JOBS) <= 6
    assert int(SP.BASE_PARAMS.get("n_jobs")) <= 6
    assert SP.BASE_PARAMS.get("device") == "cpu"
    assert "--gpu" not in code


def test_vintage_and_difference():
    SP = _load("spec_lag", "d153lag_spec_f")
    run_src = (EXP / "run_diag_lag.py").read_text()
    assert "targets <" in SP.VINTAGE_RULE and "<= origin" in SP.VINTAGE_RULE
    assert "cutoff" in run_src.lower()
    assert "ShiftSafe" in SP.DIFFERENCE
    assert "E0408" in SP.DIFFERENCE
    assert "strips" in run_src and "select_slots" in run_src
    assert "strip_features" in run_src or "strip_columns" in run_src
    assert "store/donki-cme" in run_src
    assert "CME_" in run_src and "IPS_" in run_src


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
