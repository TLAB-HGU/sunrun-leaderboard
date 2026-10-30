"""D153 shifts diagnostic contract: static plus synthetic checks (fast, no fits)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d153_shifts"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs", "predictions_official", "origin_last_input_utc",
             "pred_kms", "official_score", "official-score",
             "from scorer", "import scorer", "from expos", "import expos"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "lean.py",
                 "D141", "D147", "D148", "D151", "D152",
                 "d141", "d147", "d148", "d151", "d152",
                 "d151_e0472_audit", "d151_enlil_ext", "d148_enlil",
                 "E0471", "E0472", "e0471", "e0472", "E0512",
                 "nn-", "sun-nn", "worker-t1a", "worker-t1b",
                 "worker-t1c", "worker-t1d", "worker-t1e", "worker-t1f",
                 "worker-t1g", "worker-t1h0", "worker-t1h1", "worker-t1h2",
                 "worker-t2no", "existing_all", "info_ablation",
                 "donki-live", "isMostAccurate"]
BANNED_SRC = ["GONG", "CDAW"]
CELL_LABELS = ["low_low_high", "low_high_high", "high_low_high", "high_high_high", "hlh"]
OFFICIAL_TOKENS = ["2026-05-31", "2026-09-26", "N_OFFICIAL_ORIGINS",
                   "ORIGIN_START", "ORIGIN_END", "lb-eval", "lb-best"]

sys.path.insert(0, str(EXP))


def _load(mod_name, alias):
    loc = importlib.util.spec_from_file_location(alias, EXP / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code(names=("spec_shifts.py", "shifts_lib.py", "run_diag_shifts.py")):
    return "\n".join((EXP / n).read_text() for n in names)


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 4, files
    assert {"spec_shifts.py", "shifts_lib.py", "run_diag_shifts.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_shifts", "d153_spec")
    assert SP.DIRECTION == "D153" and SP.OWNER == "worker-t1h3"
    assert SP.KIND == "diag" and SP.REFERENCE == "none-diag"
    assert tuple(SP.SEEDS) == (0,) and SP.SMOKE_SEED == 0
    assert SP.SCRATCH == "store/scratch/worker-t1h3"
    assert int(SP.N_JOBS) == 4 and int(SP.N_JOBS) <= 6
    assert tuple(int(x) for x in SP.TARGET_LEADS) == (24, 48, 72)
    assert int(SP.STRIP_H) == 24
    assert SP.ORIGIN_CEILING_ISO == "2026-03-01T00:00:00Z"
    assert SP.ORIGIN_FREQ == "6h"
    assert len(SP.FEATURES) == 13
    assert set(SP.FEATURE_FAMILIES) == {"plasma", "imf", "epam", "ch", "strips"}
    assert abs(float(SP.STABLE_ABS_R) - 0.25) < 1e-12
    assert int(SP.MIN_N) == 200
    assert abs(float(SP.RANK_RHO_MIN) - 0.6) < 1e-12
    assert abs(float(SP.DRIFT_DMEAN) - 0.5) < 1e-12
    assert abs(float(SP.DRIFT_STD_LO) - 0.7) < 1e-12
    assert abs(float(SP.DRIFT_STD_HI) - 1.4) < 1e-12
    assert abs(float(SP.DRIFT_DR) - 0.2) < 1e-12
    assert "global" in SP.NORMALIZATION.lower() and "per fold" in SP.NORMALIZATION.lower()
    assert "[axes:" in SP.AXES_TAG and "diag" in SP.AXES_TAG
    assert SP.NO_OFFICIAL and SP.NO_BLEND and SP.NO_SELECTION and SP.NO_REGISTRATION
    assert "targets <" in SP.VINTAGE_RULE and "<= origin" in SP.VINTAGE_RULE
    assert abs(float(SP.REF_MSE) - 3281.31) < 1e-9
    assert SP.DECISION == "diagnostic-only-no-adoption"
    assert "no registration" in SP.DIAG_NOTE and "no adoption" in SP.DIAG_NOTE
    assert "ShiftSafe" in SP.DIFFERENCE and "gate" in SP.DIFFERENCE.lower()
    assert ("pre" in SP.DIFFERENCE.lower() and "before" in SP.DIFFERENCE.lower())
    assert "close" in SP.CLOSE_RULE.lower()
    assert int(SP.SMOKE_FOLDS) == 1 and int(SP.SMOKE_ORIGINS) == 8


def test_folds_cover_2023_to_2025():
    SP = _load("spec_shifts", "d153_spec_b")
    SL = _load("shifts_lib", "d153_lib_folds")
    assert SL.check_folds(SP.FOLDS, SP.ORIGIN_CEILING_ISO) is True
    assert [w["name"] for w in SP.FOLDS] == ["Y2023", "Y2024", "Y2025"]
    first = pd.Timestamp(SP.FOLDS[0]["start"], tz="UTC")
    last = pd.Timestamp(SP.FOLDS[-1]["end"], tz="UTC")
    assert first == pd.Timestamp("2023-01-01T00:00:00Z", tz="UTC")
    assert last == pd.Timestamp("2025-12-31T21:00:00Z", tz="UTC")
    for w in SP.FOLDS:
        st = pd.Timestamp(w["start"], tz="UTC")
        en = pd.Timestamp(w["end"], tz="UTC")
        assert en < pd.Timestamp(SP.ORIGIN_CEILING_ISO, tz="UTC")
        assert st < en
    for a, b in zip(SP.FOLDS, SP.FOLDS[1:]):
        assert pd.Timestamp(b["start"], tz="UTC") > pd.Timestamp(a["end"], tz="UTC")
    assert SP.ORIGIN_FREQ == "6h"
    SL.check_leads(SP.TARGET_LEADS)
    try:
        SL.check_leads((24, 48))
    except ValueError:
        pass
    else:
        raise AssertionError("short lead grid must be rejected")
    try:
        SL.check_folds(({"name": "Y2023", "start": "2023-01-01T00:00:00Z",
                         "end": "2026-06-01T00:00:00Z"},), SP.ORIGIN_CEILING_ISO)
    except ValueError:
        pass
    else:
        raise AssertionError("post ceiling fold must be rejected")


def test_features_cover_families():
    SP = _load("spec_shifts", "d153_spec_c")
    SL = _load("shifts_lib", "d153_lib_feat")
    assert SL.check_features(SP.FEATURES, SP.FEATURE_FAMILIES) is True
    text = " ".join(SP.FEATURES)
    for tok in ("density", "temperature", "bt_nt", "bz_gsm", "epam", "ch_t035", "strip_v550"):
        assert tok in text, tok
    try:
        SL.check_features(SP.FEATURES[:5], SP.FEATURE_FAMILIES)
    except ValueError:
        pass
    else:
        raise AssertionError("short feature list must be rejected")


def test_correlation_helpers_synthetic():
    SL = _load("shifts_lib", "d153_lib_corr")
    c = SL.pearson_corr(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 4.0]))
    assert c["n"] == 4 and abs(float(c["r"]) - 1.0) < 1e-12
    c2 = SL.pearson_corr(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0]))
    assert abs(float(c2["r"]) + 1.0) < 1e-12
    s = SL.spearman_corr(np.array([1.0, 2.0, 3.0, 4.0]), np.array([10.0, 20.0, 30.0, 40.0]))
    assert s["n"] == 4 and abs(float(s["rho"]) - 1.0) < 1e-12
    assert SL.sign_stable([0.5, 0.3, 0.2]) is True
    assert SL.sign_stable([0.5, -0.3, 0.2]) is False
    assert SL.sign_stable([0.5, float("nan"), 0.2]) is False
    assert abs(float(SL.rank_rho(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))) - 1.0) < 1e-12
    sh = SL.norm_shift(1.0, 1.0, 0.0, 1.0)
    assert abs(float(sh["dmean"]) - 1.0) < 1e-12 and abs(float(sh["std_ratio"]) - 1.0) < 1e-12
    g = SL.decide_gate({24: 2, 48: 0, 72: 0}, True, 0.7, stable_need=2, rho_min=0.6)
    assert g["decision"] == "OPEN"
    g2 = SL.decide_gate({24: 1, 48: 0, 72: 0}, True, 0.7, stable_need=2, rho_min=0.6)
    assert g2["decision"] == "CLOSE"
    g3 = SL.decide_gate({24: 3, 48: 0, 72: 0}, False, 0.8, stable_need=2, rho_min=0.6)
    assert g3["decision"] == "CLOSE"
    g4 = SL.decide_gate({24: 3, 48: 0, 72: 0}, True, 0.2, stable_need=2, rho_min=0.6)
    assert g4["decision"] == "CLOSE"


def test_poison_synthetic():
    SL = _load("shifts_lib", "d153_lib_pois")
    idx = pd.date_range("2023-05-01T00:00:00Z", "2023-07-01T00:00:00Z", freq="h", tz="UTC")
    filled = pd.Series(400.0 + np.sin(np.arange(len(idx)) / 24.0), index=idx)
    cut = "2023-05-29T00:00:00Z"
    origins = pd.DatetimeIndex(["2023-05-10T00:00:00Z", "2023-05-20T00:00:00Z",
                                "2023-06-05T00:00:00Z", "2023-06-10T00:00:00Z"], tz="UTC")
    sc = SL.scenario_poison(origins, filled, cut)
    assert sc["passed"] is True and sc["pre_invariant"] is True and sc["moved"] is True
    assert sc["n_pre_rows"] == 2 and sc["n_post_rows"] == 2

    def _frame_for(series):
        o = pd.DatetimeIndex(origins, tz="UTC")
        return np.tile(series.reindex(o, method="ffill").to_numpy(dtype=float)[:, None], (1, 4))

    fp = SL.frame_poison(origins, filled, cut, _frame_for)
    assert fp["passed"] is True and fp["pre_invariant"] is True


def test_diag_only_outputs():
    run_src = (EXP / "run_diag_shifts.py").read_text()
    for w in ("diag_shifts.json", "causality.json", "meta.json"):
        assert w in run_src, w
    for w in ("NO_OFFICIAL", "NO_BLEND", "NO_SELECTION", "NO_REGISTRATION"):
        assert w in run_src, w
    for w in ("scenario_poison", "frame_poison", '"passed"', "ceiling", "n_jobs"):
        assert w in run_src, w
    for w in ("predictions_official", "origin_last_input_utc", "pred_kms",
              "official_score", "official-score", "isMostAccurate", "--gpu",
              "freeze", "submit", "lb-eval", "lb-best",
              "2026-05-31", "2026-09-26", "N_OFFICIAL_ORIGINS",
              "ORIGIN_START", "ORIGIN_END"):
        assert w not in run_src, w


def test_scratch_isolation_and_cpu():
    import re
    SP = _load("spec_shifts", "d153_spec_e")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1h3"
    assert len(set(re.findall(r"worker-[a-z0-9]+", code))) == 1
    assert "worker-t1h3" in code
    assert int(SP.N_JOBS) <= 6
    assert "--gpu" not in code


def test_vintage_and_difference():
    SP = _load("spec_shifts", "d153_spec_f")
    run_src = (EXP / "run_diag_shifts.py").read_text()
    assert "targets <" in SP.VINTAGE_RULE and "<= origin" in SP.VINTAGE_RULE
    assert "ceiling" in run_src.lower() and "<= origin" in _code() or "<= origin" in SP.VINTAGE_RULE
    assert "ShiftSafe" in SP.DIFFERENCE
    assert "strips" in run_src and "select_slots" in run_src
    assert "strip_single_frame" in run_src or "strip_v550" in run_src


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
