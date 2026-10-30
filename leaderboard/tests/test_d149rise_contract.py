"""D149 rise-premise contract: static + synthetic checks (fast, no fits)."""
import importlib.util
import re
from pathlib import Path

import numpy as np

LB = Path(__file__).resolve().parents[1]
DIAG = LB / "experiments/d149_enlil_cells/diag_rise"
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
                 "worker-t1f0", "worker-t1f1", "worker-t1e", "worker-t1a",
                 "worker-t1b", "worker-t1c", "worker-t1d", "worker-d147",
                 "worker-d141", "worker-d76", "worker-t2no",
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
    loc = importlib.util.spec_from_file_location(f"d149rise_{name}", DIAG / f"{name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(DIAG.glob("*.py")))


def test_owned_file_budget():
    files = sorted(p.name for p in DIAG.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 4, files
    assert {"spec_rise.py", "rise_lib.py", "run_rise.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_rise")
    assert SP.DIRECTION == "D149" and SP.OWNER == "worker-t1f3"
    assert SP.KIND == "diag" and SP.REFERENCE == "E0408"
    assert tuple(SP.SEEDS) == (0,) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24
    assert SP.SCRATCH == "store/scratch/worker-t1f3"
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
    assert int(SP.RISE_H) == 72 and abs(float(SP.RISE_THRESH) - 150.0) < 1e-12
    assert "150" in SP.RISE_DEF and "72h" in SP.RISE_DEF
    assert "cell" in SP.RISE_DEF.lower()
    assert "OOF" in SP.RISE_DEF
    assert "recal" in SP.RECAL_DEF.lower() and "OOF" in SP.RECAL_DEF
    assert tuple(SP.METRICS) == ("share", "bias", "recal")
    assert "level" in SP.MEMBER_RULE and "mixture" in SP.MEMBER_RULE
    assert SP.PATH_SOURCE == "oof-fit" and SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert "[axes:" in SP.AXES_TAG and "rise-premise" in SP.AXES_TAG
    assert SP.DECISION == "diagnostic-only-no-adoption"
    assert "no adoption" in SP.DIAG_NOTE and "no registration" in SP.DIAG_NOTE


def test_single_axis_vs_recipe():
    run_src = (DIAG / "run_rise.py").read_text()
    for mod in ("info_ablation", "strips", "spec_rise", "rise_lib"):
        assert mod in run_src, mod
    assert "150" in run_src and "0.03" in run_src
    assert "oof-fit" in run_src and "dv = y - yhat_base" in run_src
    assert "existing_all" in run_src and "strip_columns" in run_src
    assert "105" in run_src
    assert "level_seeds" in run_src
    assert "fit_params" in run_src
    assert "blend_members" in run_src and "mixture" in run_src
    assert "rise" in run_src.lower() and "share" in run_src
    assert "bias" in run_src and "recal" in run_src.lower()
    assert "oof_start" in run_src and "oof_end" in run_src
    assert "rise_report.json" in run_src


def test_rise_math_identity():
    RL = _load("rise_lib")
    assert tuple(RL.level_seeds(0)) == (0, 1, 2)
    assert tuple(RL.level_seeds(2)) == (6, 7, 8)
    assert RL.event_seed(1) in RL.level_seeds(1)
    p = RL.fit_params({"n_estimators": 150}, 3, 7)
    assert p["max_depth"] == 3 and p["random_state"] == 7
    assert RL.lead_of(1) == "1-6" and RL.lead_of(24) == "7-24"
    assert RL.lead_of(25) == "25-48" and RL.lead_of(72) == "49-72"
    RL.check_weights(PINNED_W)
    assert abs(RL.check_rise_thresh(150.0) - 150.0) < 1e-12
    try:
        RL.check_rise_thresh(0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("rise thresh must reject non-positive")
    assert RL.origin_rise_flag(np.array([400.0, 450.0, 560.0]), 400.0, 150.0) is True
    assert RL.origin_rise_flag(np.array([400.0, 450.0, 500.0]), 400.0, 150.0) is False
    assert RL.origin_rise_flag(np.array([400.0, np.nan, 560.0]), 400.0, 150.0) is True
    assert RL.origin_rise_flag(np.array([400.0, 500.0]), np.nan, 150.0) is False
    origins = ["o1"] * 3 + ["o2"] * 3
    y = np.array([400.0, 420.0, 560.0, 400.0, 410.0, 420.0])
    v0 = {"o1": 400.0, "o2": 400.0}
    flags, cell = RL.flag_origins(origins, y, v0, 150.0)
    assert flags == {"o1": True, "o2": False}
    assert cell.tolist() == [True, True, True, False, False, False]
    se = np.array([100.0, 200.0, 300.0, 400.0])
    m = np.array([True, True, False, False])
    assert abs(RL.se_share(se, m) - 0.3) < 1e-12
    assert abs(RL.mean_signed_error(np.array([410.0, 510.0]), np.array([400.0, 500.0])) - 10.0) < 1e-12
    assert abs(RL.mse(np.array([410.0, 510.0]), np.array([400.0, 500.0])) - 100.0) < 1e-12
    a, b = RL.linear_recal_fit(np.array([400.0, 500.0, 600.0]), np.array([400.0, 500.0, 600.0]))
    assert abs(a) < 1e-6 and abs(b - 1.0) < 1e-6
    assert np.allclose(RL.linear_recal_apply(np.array([400.0, 500.0]), a, b), [400.0, 500.0])
    tab = RL.summarize_window(np.array([400.0, 500.0, 600.0, 700.0]),
                              np.array([390.0, 510.0, 590.0, 710.0]),
                              np.array([True, True, False, False]))
    assert tab["n_rows"] == 4 and tab["n_rise_cells"] == 2
    assert 0.0 <= tab["se_share_rise"] <= 1.0
    assert np.isfinite(tab["bias_rise"]) and np.isfinite(tab["recal_b"])


def test_blend_helpers():
    RL = _load("rise_lib")
    out = RL.blend_members(np.full(72, 400.0), np.full(72, 400.0), np.full(72, 400.0), np.arange(1, 73))
    assert np.allclose(out, 400.0)


def test_scratch_isolation_and_cpu():
    SP = _load("spec_rise")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1f3"
    assert len(set(re.findall(r"worker-t1f[0-9]", code))) == 1
    assert "worker-t1f3" in code
    run_src = (DIAG / "run_rise.py").read_text()
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
