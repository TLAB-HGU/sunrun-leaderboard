"""D153 APT contract: static plus synthetic checks (fast, no fits)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d153_shifts/apt"
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
                 "worker-t1h3", "worker-t2no",
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


def _code(names=("spec_apt.py", "apt_lib.py", "run_diag_apt.py")):
    return "\n".join((EXP / n).read_text() for n in names)


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 4, files
    assert {"spec_apt.py", "apt_lib.py", "run_diag_apt.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_apt", "d153apt_spec")
    assert SP.DIRECTION == "D153" and SP.OWNER == "worker-t1h4"
    assert SP.KIND == "diag" and SP.REFERENCE == "E0421"
    assert tuple(SP.SEEDS) == (0,) and SP.SMOKE_SEED == 0
    assert SP.SCRATCH == "store/scratch/worker-t1h4"
    assert int(SP.N_JOBS) == 4 and int(SP.N_JOBS) <= 6
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.POISON_CUTOFF_ISO == "2026-04-16T00:00:00Z"
    assert SP.OOF_TRAIN_END_ISO == "2026-03-01T00:00:00Z"
    assert SP.OOF_START_ISO == "2026-03-04T00:00:00Z"
    assert SP.OOF_END_ISO == "2026-05-28T23:00:00Z"
    assert pd.Timestamp(SP.OOF_START_ISO, tz="UTC") < pd.Timestamp(SP.POISON_CUTOFF_ISO, tz="UTC") < pd.Timestamp(SP.OOF_END_ISO, tz="UTC")
    assert pd.Timestamp(SP.POISON_CUTOFF_ISO, tz="UTC") < pd.Timestamp(SP.CUTOFF_ISO, tz="UTC")
    assert SP.OOF_FREQ == "3h" and SP.TRAIN_FREQ == "3h"
    assert int(SP.SPLIT_H) == 24
    assert int(SP.N_PROTO) == 3 and tuple(SP.PROTO_NAMES) == ("k0", "k1", "k2")
    assert "tercile" in SP.PROTO_DEF.lower() and "month" in SP.PROTO_DEF.lower()
    assert abs(float(SP.SHRINK) - 0.25) < 1e-12
    assert tuple(tuple(x) for x in SP.BANDS) == ((1, 6), (7, 24), (25, 72))
    assert tuple(SP.BAND_NAMES) == ("1-6", "7-24", "25-72")
    assert tuple(SP.ARMS) == ("identity", "single-affine", "proto-affine")
    assert int(SP.MIN_N) == 200
    assert abs(float(SP.NOISE_FLOOR) - 90.0) < 1e-12
    assert "[axes:" in SP.AXES_TAG and "apt" in SP.AXES_TAG
    assert SP.NO_OFFICIAL and SP.NO_SELECTION and SP.NO_REGISTRATION
    assert "targets <" in SP.VINTAGE_RULE and "<= origin" in SP.VINTAGE_RULE
    assert "pre cutoff" in SP.VINTAGE_RULE
    assert abs(float(SP.REF_MSE) - 3637.84) < 1e-9
    assert SP.DECISION == "diagnostic-only-no-adoption"
    assert "no registration" in SP.DIAG_NOTE and "no adoption" in SP.DIAG_NOTE
    assert "ShiftSafe" in SP.DIFFERENCE and "gate" in SP.DIFFERENCE.lower()
    assert "E0421" in SP.DIFFERENCE
    assert "close" in SP.CLOSE_RULE.lower() and "90" in SP.CLOSE_RULE
    assert int(SP.SMOKE_ORIGINS) == 8
    assert int(SP.BASE_DEPTH) == 3
    assert int(SP.BASE_PARAMS.get("n_estimators")) == 900
    assert abs(float(SP.BASE_PARAMS.get("learning_rate")) - 0.03) < 1e-12
    assert SP.BASE_PARAMS.get("device") == "cpu"
    assert int(SP.BASE_PARAMS.get("n_jobs")) <= 6
    assert SP.BASE_COLUMNS == "existing_all+strips17"
    assert SP.PATH_SOURCE == "oof-fit"
    assert "a_k" in SP.AFFINE_DEF and "shrunk" in SP.AFFINE_DEF.lower()


def test_prototypes_prespecified_and_separated():
    SP = _load("spec_apt", "d153apt_spec_b")
    AL = _load("apt_lib", "d153apt_lib_proto")
    assert int(SP.N_PROTO) == 3
    assert len(set(SP.PROTO_NAMES)) == 3
    b = np.array([300.0, 350.0, 400.0, 450.0, 500.0, 550.0])
    lo, hi = AL.tercile_bounds(b)
    assert lo < hi and 350.0 <= lo <= 450.0 and 350.0 <= hi <= 500.0
    p = AL.assign_prototype(np.array([lo - 1.0, (lo + hi) / 2.0, hi + 1.0, np.nan]), (lo, hi))
    assert list(p) == [0, 1, 2, -1]
    try:
        AL.tercile_bounds(np.array([1.0, 1.0, 1.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("degenerate terciles must be rejected")
    AL.check_weights(SP.WEIGHTS)
    assert AL.level_seeds(0) == (0, 1, 2) and AL.event_seed(0) == 1
    assert AL.level_seeds(1) == (3, 4, 5) and AL.event_seed(1) == 4


def test_affine_helpers_synthetic():
    AL = _load("apt_lib", "d153apt_lib_aff")
    rng = np.random.default_rng(0)
    base = 400.0 + 50.0 * rng.standard_normal(500)
    y = 1.0 * base + 0.0 + 10.0 * rng.standard_normal(500)
    r = AL.fit_affine_shrunk(base, y, shrink=0.25)
    assert abs(r["a"] - (1.0 + 0.25 * (r["a_ols"] - 1.0))) < 1e-12
    assert abs(r["b"] - 0.25 * r["b_ols"]) < 1e-12
    assert r["n"] == 500
    p = AL.apply_affine(base, r["a"], r["b"])
    assert p.shape == base.shape and np.isfinite(p).all()
    proto = np.tile([0, 1, 2], 200)[:500]
    per_k = AL.fit_prototype_affine(base, y, proto, shrink=0.25)
    assert set(per_k) == {"k0", "k1", "k2"}
    for k in ("k0", "k1", "k2"):
        assert abs(per_k[k]["a"] - (1.0 + 0.25 * (per_k[k]["a_ols"] - 1.0))) < 1e-12
    q = AL.apply_prototype_affine(base, proto, per_k)
    assert q.shape == base.shape and np.isfinite(q).all()
    m = AL.mse(base, y)
    assert m["n"] == 500 and m["mse"] > 0
    h = np.tile(np.arange(1, 73), 7)[:500]
    bm = AL.band_mse(base, y, h)
    assert set(bm) == {"1-6", "7-24", "25-72"}
    assert all(v["n"] >= 0 for v in bm.values())
    g = AL.decide_gate(1000.0, 990.0, 800.0, {"1-6": True, "7-24": True, "25-72": False},
                       {"s0": True, "s1": True, "s2": False}, floor=90.0)
    assert g["decision"] == "OPEN"
    g2 = AL.decide_gate(1000.0, 990.0, 950.0, {"1-6": True, "7-24": True, "25-72": True},
                        {"s0": True, "s1": True, "s2": True}, floor=90.0)
    assert g2["decision"] == "CLOSE"
    g3 = AL.decide_gate(1000.0, 800.0, 900.0, {"1-6": True, "7-24": True, "25-72": True},
                        {"s0": True, "s1": True, "s2": True}, floor=90.0)
    assert g3["decision"] == "CLOSE"
    g4 = AL.decide_gate(1000.0, 990.0, 800.0, {"1-6": True, "7-24": False, "25-72": False},
                        {"s0": True, "s1": True, "s2": True}, floor=90.0)
    assert g4["decision"] == "CLOSE"


def test_poison_synthetic():
    AL = _load("apt_lib", "d153apt_lib_pois")
    idx = pd.date_range("2023-05-01T00:00:00Z", "2023-07-01T00:00:00Z", freq="h", tz="UTC")
    filled = pd.Series(400.0 + np.sin(np.arange(len(idx)) / 24.0), index=idx)
    cut = "2023-05-29T00:00:00Z"
    origins = pd.DatetimeIndex(["2023-05-10T00:00:00Z", "2023-05-20T00:00:00Z",
                                "2023-06-05T00:00:00Z", "2023-06-10T00:00:00Z"], tz="UTC")
    sc = AL.scenario_poison(origins, filled, cut)
    assert sc["passed"] is True and sc["pre_invariant"] is True and sc["moved"] is True
    assert sc["n_pre_rows"] == 2 and sc["n_post_rows"] == 2

    def _frame_for(series):
        o = pd.DatetimeIndex(origins, tz="UTC")
        return np.tile(series.reindex(o, method="ffill").to_numpy(dtype=float)[:, None], (1, 4))

    fp = AL.frame_poison(origins, filled, cut, _frame_for)
    assert fp["passed"] is True and fp["pre_invariant"] is True


def test_diag_only_outputs():
    run_src = (EXP / "run_diag_apt.py").read_text()
    for w in ("diag_apt.json", "causality.json", "meta.json"):
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
    SP = _load("spec_apt", "d153apt_spec_e")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1h4"
    assert len(set(re.findall(r"worker-[a-z0-9]+", code))) == 1
    assert "worker-t1h4" in code
    assert int(SP.N_JOBS) <= 6
    assert int(SP.BASE_PARAMS.get("n_jobs")) <= 6
    assert SP.BASE_PARAMS.get("device") == "cpu"
    assert "--gpu" not in code


def test_vintage_and_difference():
    SP = _load("spec_apt", "d153apt_spec_f")
    run_src = (EXP / "run_diag_apt.py").read_text()
    assert "targets <" in SP.VINTAGE_RULE and "<= origin" in SP.VINTAGE_RULE
    assert "cutoff" in run_src.lower()
    assert "ShiftSafe" in SP.DIFFERENCE
    assert "strips" in run_src and "select_slots" in run_src
    assert "strip_features" in run_src or "strip_columns" in run_src


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
