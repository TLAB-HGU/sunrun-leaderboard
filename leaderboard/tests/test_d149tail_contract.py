"""D149 tail-gate contract: static + synthetic checks (fast, no fits)."""
import importlib.util
import re
from pathlib import Path

import numpy as np

LB = Path(__file__).resolve().parents[1]
DIAG = LB / "experiments/d149_enlil_cells/diag_tail"
FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "predictions_official", "research/expos/runs",
             "datasets--tlabtlab--sunrun-lb-store",
             "EXPOS_OUT", "isMostAccurate", "--gpu",
             "from scorer", "import scorer", "from expos", "import expos",
             "official-score", "official_score"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "lean.py", "D147", "d147",
                 "d148_enlil", "nn-", "sun-nn", "E0471", "e0471", "E0472", "e0472",
                 "donki-live", "ips_oracle", "spec_oracle", "run_oracle",
                 "spec_ec", "cells_lib", "run_ec",
                 "diag_rise", "diag_dis", "spec_rise", "spec_dis",
                 "rise_lib", "disagree_lib", "run_rise", "run_dis",
                 "worker-t1f0", "worker-t1f1", "worker-t1e", "worker-t1a",
                 "worker-t1b", "worker-t1c", "worker-t1d", "worker-d147",
                 "worker-d141", "worker-d76", "worker-t2no",
                 "worker-t1f2", "worker-t1f3",
                 "enlil_next", "enlil_dt", "ENLIL_COLS"]
BANNED_SRC = ["GONG", "CDAW"]
CELL_LABELS = ["low_low_high", "low_high_high", "high_low_high", "high_high_high"]
OFFICIAL_TOKENS = ["2026-05-31", "2026-09-26", "N_OFFICIAL_ORIGINS",
                   "ORIGIN_START", "ORIGIN_END", "lb-eval", "lb-best"]
XFORMER_TOKENS = ["xformer", "XFormer", "transformer"]
PINNED_W = {"1-6": {"e0003_line": 0.21536233464375223, "d33_hsplit": 0.3899845818930272,
                    "d45_lvlmean": 0.3946530834632206},
            "7-24": {"e0003_line": 0.3152013131693896, "d33_hsplit": 0.3436571856823209,
                     "d45_lvlmean": 0.3411415011482895},
            "25-48": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
                      "d45_lvlmean": 0.3333333333333333},
            "49-72": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
                      "d45_lvlmean": 0.3333333333333333}}


def _load(name):
    loc = importlib.util.spec_from_file_location(f"d149tail_{name}", DIAG / f"{name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(DIAG.glob("*.py")))


def test_owned_file_budget():
    files = sorted(p.name for p in DIAG.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 4, files
    assert {"spec_tail.py", "tail_lib.py", "run_tail.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_tail")
    assert SP.DIRECTION == "D149" and SP.OWNER == "worker-t1f4"
    assert SP.KIND == "diag" and SP.REFERENCE == "E0408"
    assert tuple(SP.SEEDS) == (0,) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24
    assert SP.SCRATCH == "store/scratch/worker-t1f4"
    assert int(SP.FIT_NJOBS) <= 6
    assert SP.BASE_DEPTH == 3
    assert SP.DIAG_PARAMS["n_estimators"] == 150
    assert abs(float(SP.DIAG_PARAMS["learning_rate"]) - 0.03) < 1e-12
    assert SP.DIAG_PARAMS.get("device") == "cpu" and int(SP.DIAG_PARAMS.get("n_jobs")) <= 6
    assert SP.DIAG_PARAMS["tree_method"] == "hist"
    assert SP.BASE_COLUMNS == "existing_all+strips17"
    assert int(SP.DIAG_MAX_TRAIN) == 60000
    assert SP.WEIGHTS == PINNED_W
    assert [w["name"] for w in SP.WINDOWS] == ["W0", "W1", "W2"]
    assert SP.MAIN_WINDOW == "W2"
    for w in SP.WINDOWS:
        assert w["oof_end"] < SP.CUTOFF_ISO
    assert tuple(SP.SHOCK_PREDS) == ("ace_density_cm3_last", "ace_temperature_k_last",
                                     "ace_bt_nt_last", "ace_speed_kms_last",
                                     "speed_d24", "epam_p1_d6")
    assert "OOF" in SP.SHOCK_DEF
    assert tuple(float(x) for x in SP.BOUND_QUANTS) == (0.90, 0.95)
    assert abs(float(SP.WORST_FRAC) - 0.05) < 1e-12
    assert int(SP.PURGE_H) == 72
    assert abs(float(SP.NOISE_MSE) - 90.0) < 1e-12
    assert abs(float(SP.REF_MSE) - 3281.3) < 1e-6
    assert abs(float(SP.CLOSE_SHARE) - 0.15) < 1e-12
    assert tuple(SP.METRICS) == ("share", "bias", "episodes", "linear")
    assert "level" in SP.MEMBER_RULE and "mixture" in SP.MEMBER_RULE
    assert "boundary" in SP.BOUND_RULE and "72h" in SP.BOUND_RULE
    assert "excess" in SP.TAIL_DEF and "OOF" in SP.TAIL_DEF
    assert "0.15" in SP.STOP_RULE and "noise" in SP.STOP_RULE
    assert SP.PATH_SOURCE == "oof-fit" and SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert "[axes:" in SP.AXES_TAG and "tail-gate" in SP.AXES_TAG
    assert SP.DECISION == "diagnostic-only-no-adoption"
    assert "no adoption" in SP.DIAG_NOTE and "no registration" in SP.DIAG_NOTE


def test_single_axis_vs_recipe():
    run_src = (DIAG / "run_tail.py").read_text()
    for mod in ("info_ablation", "strips", "spec_tail", "tail_lib"):
        assert mod in run_src, mod
    assert "150" in run_src and "0.03" in run_src
    assert "oof-fit" in run_src and "dv = y - yhat_base" in run_src
    assert "existing_all" in run_src and "strip_columns" in run_src
    assert "105" in run_src
    assert "level_seeds" in run_src
    assert "fit_params" in run_src
    assert "blend_members" in run_src and "mixture" in run_src
    assert "boundary" in run_src and "share" in run_src
    assert "bias" in run_src and "excess" in run_src
    assert "episode" in run_src and "premax" in run_src
    assert "oof_start" in run_src and "oof_end" in run_src
    assert "tail_report.json" in run_src


def test_tail_math_identity():
    TL = _load("tail_lib")
    assert tuple(TL.level_seeds(0)) == (0, 1, 2)
    assert tuple(TL.level_seeds(2)) == (6, 7, 8)
    assert TL.event_seed(1) in TL.level_seeds(1)
    p = TL.fit_params({"n_estimators": 150}, 3, 7)
    assert p["max_depth"] == 3 and p["random_state"] == 7
    assert TL.lead_of(1) == "1-6" and TL.lead_of(24) == "7-24"
    assert TL.lead_of(25) == "25-48" and TL.lead_of(72) == "49-72"
    TL.check_weights(PINNED_W)
    assert abs(TL.col_quantile(np.arange(10, dtype=float), 0.90) - 8.1) < 1e-9
    assert abs(TL.check_bound_quant(0.95) - 0.95) < 1e-12
    try:
        TL.check_bound_quant(1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("bound quant must reject outside (0,1)")
    tr = {"a": np.arange(100, dtype=float), "b": np.arange(100, dtype=float) * 2.0}
    th = TL.train_thresholds(tr, (0.90, 0.95))
    assert set(th[0.90]) == {"a", "b"}
    assert th[0.95]["a"] > th[0.90]["a"]
    med = TL.train_medians(tr)
    assert abs(med["a"] - 49.5) < 1e-9
    sc = TL.train_scales(tr, 0.95)
    assert all(v > 0 for v in sc.values())
    xo = {"a": np.array([0.0, 200.0]), "b": np.array([0.0, 0.0])}
    b = TL.boundary_flag(xo, th[0.90])
    assert b.tolist() == [False, True]
    e = TL.standardized_excess(xo, th[0.90], med, sc)
    assert e[0] == 0.0 and e[1] > 0.0
    se = np.array([100.0, 200.0, 300.0, 400.0])
    m = np.array([True, True, False, False])
    assert abs(TL.se_share(se, m) - 0.3) < 1e-12
    assert abs(TL.mean_signed_error(np.array([410.0, 510.0]), np.array([400.0, 500.0])) - 10.0) < 1e-12
    assert abs(TL.mse(np.array([410.0, 510.0]), np.array([400.0, 500.0])) - 100.0) < 1e-12
    lab = TL.worst_label(np.arange(100, dtype=float), 0.05)
    assert 0 < int(lab.sum()) <= 10
    se40 = np.array([1.0] * 30 + [100.0] * 10)
    b40 = np.array([False] * 30 + [True] * 10)
    assert abs(TL.bulk_share_in_worst(se40, b40, 0.05) - 0.0) < 1e-12
    b40b = np.array([True] * 30 + [False] * 10)
    assert abs(TL.bulk_share_in_worst(se40, b40b, 0.05) - 1.0) < 1e-12
    assert abs(TL.premax_exceed_rate(np.array([400.0, 500.0, 600.0]), 450.0) - 2.0 / 3.0) < 1e-12
    exc = np.array([0.0, 1.0, 2.0, 3.0])
    res = np.array([0.0, 2.0, 4.0, 6.0])
    mm = np.array([True, True, True, True])
    a, bb = TL.linear_tail_fit(exc, res, mm)
    assert abs(a) < 1e-9 and abs(bb - 2.0) < 1e-9
    corr = TL.linear_tail_correct(np.array([400.0, 400.0]), np.array([0.0, 1.0]), 1.0, 2.0,
                                  np.array([True, False]))
    assert abs(corr[0] - 401.0) < 1e-9 and abs(corr[1] - 400.0) < 1e-9
    oo = [np.datetime64("2026-01-04T00:00"), np.datetime64("2026-01-04T03:00"),
          np.datetime64("2026-01-10T00:00")]
    ne, non = TL.count_episodes(oo, np.array([True, True, True]), 72)
    assert (ne, non) == (2, 3)
    tab = TL.summarize_boundary(np.array([400.0, 500.0, 600.0, 700.0]),
                                np.array([390.0, 510.0, 590.0, 710.0]),
                                np.array([True, True, False, False]),
                                np.array([0.0, 1.0, 0.0, 0.0]), 650.0)
    assert tab["n_rows"] == 4 and tab["n_boundary"] == 2
    assert 0.0 <= tab["se_share_boundary"] <= 1.0
    assert np.isfinite(tab["bias_boundary"]) and np.isfinite(tab["tail_b"])
    assert np.isfinite(tab["oracle_debias_cap"])


def test_blend_helpers():
    TL = _load("tail_lib")
    out = TL.blend_members(np.full(72, 400.0), np.full(72, 400.0), np.full(72, 400.0), np.arange(1, 73))
    assert np.allclose(out, 400.0)


def test_scratch_isolation_and_cpu():
    SP = _load("spec_tail")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1f4"
    assert len(set(re.findall(r"worker-t1f[0-9]", code))) == 1
    assert "worker-t1f4" in code
    run_src = (DIAG / "run_tail.py").read_text()
    assert 'device") == "cpu"' in run_src and "n_jobs" in run_src
    assert "--gpu" not in code
    assert "cuda" not in code.lower()


def test_no_forbidden_leak_banned_source_or_later():
    code = _code()
    low = code.lower()
    for w in FORBIDDEN:
        assert w.lower() not in low, w
    for w in OTHER_CONTACT:
        assert w not in code, w
    for w in BANNED_SRC:
        assert w.lower() not in low, w
    for w in CELL_LABELS:
        assert w not in code, w
    for w in OFFICIAL_TOKENS:
        assert w not in code, w
    for w in XFORMER_TOKENS:
        assert w.lower() not in low, w
    assert "freeze" not in low and "submit" not in low
