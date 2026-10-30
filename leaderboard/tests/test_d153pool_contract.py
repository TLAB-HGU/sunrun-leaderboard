"""D153 pooling contract: static plus synthetic checks (fast, no fits)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d153_shifts/pooling"
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
                 "worker-t1h3", "worker-t1h4", "worker-t1h5", "worker-t1h6",
                 "worker-t1h8", "worker-t2no",
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


def _code(names=("spec_pool.py", "pool_lib.py", "run_diag_pool.py")):
    return "\n".join((EXP / n).read_text() for n in names)


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 5, files
    assert {"spec_pool.py", "pool_lib.py", "run_diag_pool.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_pool", "d153pool_spec")
    assert SP.DIRECTION == "D153" and SP.OWNER == "worker-t1h7"
    assert SP.KIND == "diag" and SP.REFERENCE == "none-diag"
    assert tuple(SP.SEEDS) == (0,) and SP.SMOKE_SEED == 0
    assert SP.SCRATCH == "store/scratch/worker-t1h7"
    assert int(SP.N_JOBS) == 4 and int(SP.N_JOBS) <= 6
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.TRAIN_END_ISO == "2025-01-01T00:00:00Z"
    assert SP.TEST_START_ISO == "2025-01-04T00:00:00Z"
    assert int(SP.PURGE_H) == 72 and int(SP.BLOCKS) == 4
    assert tuple(SP.SOURCES) == ("enlil-latest", "enlil-earliest", "cone-ballistic")
    assert "ballistic" in SP.SOURCE_DEF.lower() and "earliest" in SP.SOURCE_DEF.lower()
    assert abs(float(SP.SIGMA_FLOOR_H) - 1.0) < 1e-12
    assert abs(float(SP.WEIGHT_STEP) - 0.05) < 1e-12
    assert "log score" in SP.WEIGHT_DEF.lower() and "simplex" in SP.WEIGHT_DEF.lower()
    assert abs(float(SP.AU_KM) - 149597870.7) < 1e-3
    assert abs(float(SP.TRANSIT_MIN_H) - 6.0) < 1e-12
    assert abs(float(SP.TRANSIT_MAX_H) - 200.0) < 1e-12
    assert "before event" in SP.PAIR_RULE.lower() and "before" in SP.PAIR_RULE.lower()
    assert "6" in SP.PAIR_RULE and "200" in SP.PAIR_RULE
    assert "episode" in SP.EPISODE_DEF.lower() and "72" in SP.EPISODE_DEF
    assert "72" in SP.PURGE_DEF
    assert int(SP.MIN_N) == 40 and int(SP.TRAIN_MIN_N) == 60
    assert abs(float(SP.CRPS_FLOOR_H) - 0.5) < 1e-12
    assert "close" in SP.CLOSE_RULE.lower() and "unless" in SP.CLOSE_RULE.lower()
    assert "close" in SP.CLOSE_RULE.lower() and "noise" in SP.CLOSE_RULE.lower()
    assert "block" in SP.NOISE_DEF.lower() and "4" in SP.NOISE_DEF
    assert "targets <" in SP.VINTAGE_RULE and "<= origin" in SP.VINTAGE_RULE
    assert abs(float(SP.REF_MSE) - 3281.31) < 1e-9
    assert SP.DECISION == "diagnostic-only-no-adoption"
    assert "no registration" in SP.DIAG_NOTE and "no adoption" in SP.DIAG_NOTE
    assert "ShiftSafe" in SP.DIFFERENCE and "gate" in SP.DIFFERENCE.lower()
    assert "pool" in SP.DIFFERENCE.lower()
    assert SP.NO_OFFICIAL and SP.NO_SELECTION and SP.NO_REGISTRATION
    assert SP.POISON_CUTOFF_ISO == "2026-04-16T00:00:00Z"
    assert int(SP.SMOKE_PAIRS) == 16
    assert "[axes:" in SP.AXES_TAG and "pool" in SP.AXES_TAG


def test_gauss_crps_weights_synthetic():
    PL = _load("pool_lib", "d153pool_lib_gauss")
    assert abs(float(PL.norm_cdf(0.0)) - 0.5) < 1e-12
    assert abs(float(PL.norm_pdf(0.0)) - 1.0 / np.sqrt(2.0 * np.pi)) < 1e-12
    c1 = PL.gauss_crps(np.array([0.0]), np.array([0.0]), 2.0)
    assert c1.shape == (1,) and float(c1[0]) > 0.0
    c2 = PL.gauss_crps(np.array([0.0]), np.array([0.0]), 4.0)
    assert abs(float(c2[0]) / float(c1[0]) - 2.0) < 1e-9
    bad = PL.gauss_crps(np.array([10.0]), np.array([0.0]), 2.0)
    assert float(bad[0]) > float(c1[0])
    try:
        PL.gauss_crps(np.array([0.0]), np.array([0.0]), 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive sigma must be rejected")
    nll_good = PL.gauss_nll(np.array([0.1]), np.array([0.0]), 2.0)
    nll_bad = PL.gauss_nll(np.array([10.0]), np.array([0.0]), 2.0)
    assert float(nll_good[0]) < float(nll_bad[0])
    # Linear pool of one sharp good source and two flat bad ones.
    rng = np.random.default_rng(0)
    y = rng.standard_normal(60) * 2.0
    mu = np.stack([y + rng.standard_normal(60) * 0.2,
                   y + rng.standard_normal(60) * 8.0,
                   y + rng.standard_normal(60) * 8.0]).T
    fit = PL.fit_linear_pool(y, mu, np.array([2.0, 8.0, 8.0]), step=0.1)
    w = np.asarray(fit["weights"])
    assert abs(w.sum() - 1.0) < 1e-9 and w[0] > 0.8
    assert fit["n"] == 60 and np.isfinite(fit["mean_nll"])
    try:
        PL.check_weights(np.array([0.5, 0.5, 0.5]), k=3)
    except ValueError:
        pass
    else:
        raise AssertionError("weights not summing to 1 must be rejected")
    med = PL.mixture_median(np.array([0.0]), np.array([2.0]), np.array([1.0]))
    assert abs(med) < 0.05
    assert abs(PL.mixture_cdf_at(med, np.array([0.0]), np.array([2.0]),
                                 np.array([1.0])) - 0.5) < 1e-6
    mc = PL.mixture_crps(0.0, np.array([0.0]), np.array([2.0]), np.array([1.0]))
    assert abs(mc - float(c1[0])) / float(c1[0]) < 0.02
    assert abs(PL.timing_mae(np.array([1.0, -2.0, 3.0])) - 2.0) < 1e-12


def test_episode_purge_synthetic():
    PL = _load("pool_lib", "d153pool_lib_ep")
    H = 3_600_000_000_000
    t = np.array([0, 10, 100, 110, 500, 510], dtype=np.int64) * H
    lab = PL.assign_episodes(t, 72 * H)
    assert list(lab) == [0, 0, 1, 1, 2, 2]
    try:
        PL.assign_episodes(np.array([], dtype=np.int64), 72 * H)
    except ValueError:
        pass
    else:
        raise AssertionError("empty times must be rejected")
    keep = PL.purge_test_mask(t, lab, 10 * H, 72 * H)
    assert not bool(keep[0]) and not bool(keep[1])
    assert bool(keep[2]) and bool(keep[3])
    assert bool(keep[4]) and bool(keep[5])
    bl = PL.block_labels_timeordered(8, 4)
    assert list(bl) == [0, 0, 1, 1, 2, 2, 3, 3]
    try:
        PL.block_labels_timeordered(3, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("too few rows must be rejected")
    se = PL.block_se(np.array([1.0, 1.2, 0.8, 1.1, 0.9, 1.0, 1.3, 0.7]), n_blocks=4)
    assert se["n"] == 8 and se["n_blocks"] == 4 and se["se"] >= 0.0
    assert abs(se["mean"] - 1.0) < 1e-12


def test_gate_synthetic():
    PL = _load("pool_lib", "d153pool_lib_gate")
    g = PL.decide_gate(100, 60, 10.0, 8.0, 0.3, 12.0, 11.0, floor=0.5, min_n=40)
    assert g["decision"] == "OPEN"
    g2 = PL.decide_gate(100, 60, 10.0, 9.9, 0.3, 12.0, 11.0, floor=0.5, min_n=40)
    assert g2["decision"] == "CLOSE"
    g3 = PL.decide_gate(100, 60, 10.0, 8.0, 0.3, 12.0, 13.0, floor=0.5, min_n=40)
    assert g3["decision"] == "CLOSE"
    g4 = PL.decide_gate(100, 10, 10.0, 8.0, 0.3, 12.0, 11.0, floor=0.5, min_n=40)
    assert g4["decision"] == "CLOSE"
    g5 = PL.decide_gate(100, 60, 10.0, 8.0, 0.0, 12.0, 11.0, floor=0.5, min_n=40)
    assert g5["decision"] == "CLOSE"


def test_poison_synthetic():
    PL = _load("pool_lib", "d153pool_lib_pois")
    idx = pd.date_range("2023-05-01T00:00:00Z", "2023-07-01T00:00:00Z", freq="h", tz="UTC")
    filled = pd.Series(400.0 + np.sin(np.arange(len(idx)) / 24.0), index=idx)
    cut = "2023-05-29T00:00:00Z"
    origins = pd.DatetimeIndex(["2023-05-10T00:00:00Z", "2023-05-20T00:00:00Z",
                                "2023-06-05T00:00:00Z", "2023-06-10T00:00:00Z"], tz="UTC")
    sc = PL.scenario_poison(origins, filled, cut)
    assert sc["passed"] is True and sc["pre_invariant"] is True and sc["moved"] is True
    assert sc["n_pre_rows"] == 2 and sc["n_post_rows"] == 2

    def _frame_for(series):
        o = pd.DatetimeIndex(origins, tz="UTC")
        return np.tile(series.reindex(o, method="ffill").to_numpy(dtype=float)[:, None], (1, 4))

    fp = PL.frame_poison(origins, filled, cut, _frame_for)
    assert fp["passed"] is True and fp["pre_invariant"] is True


def test_diag_only_outputs():
    run_src = (EXP / "run_diag_pool.py").read_text()
    for w in ("diag_pool.json", "causality.json", "meta.json"):
        assert w in run_src, w
    for w in ("NO_OFFICIAL", "NO_SELECTION", "NO_REGISTRATION"):
        assert w in run_src, w
    for w in ("scenario_poison", "frame_poison", '"passed"', "cutoff", "n_jobs"):
        assert w in run_src, w
    for w in ("linear", "crps", "purge", "episode", "weight"):
        assert w.lower() in run_src.lower(), w
    for w in ("predictions_official", "origin_last_input_utc", "pred_kms",
              "official_score", "official-score", "isMostAccurate", "--gpu",
              "freeze", "submit", "lb-eval", "lb-best",
              "2026-05-31", "2026-09-26", "N_OFFICIAL_ORIGINS",
              "ORIGIN_START", "ORIGIN_END"):
        assert w not in run_src, w


def test_scratch_isolation_and_cpu():
    import re
    SP = _load("spec_pool", "d153pool_spec_e")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1h7"
    assert len(set(re.findall(r"worker-[a-z0-9]+", code))) == 1
    assert "worker-t1h7" in code
    assert int(SP.N_JOBS) <= 6
    assert '"n_jobs"' in code or "'n_jobs'" in code or '"device"' in code
    assert "--gpu" not in code


def test_vintage_and_difference():
    SP = _load("spec_pool", "d153pool_spec_f")
    run_src = (EXP / "run_diag_pool.py").read_text()
    assert "targets <" in SP.VINTAGE_RULE and "<= origin" in SP.VINTAGE_RULE
    assert "cutoff" in run_src.lower()
    assert "ShiftSafe" in SP.DIFFERENCE
    assert "pool" in SP.DIFFERENCE.lower()
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
