"""D148 error-decomp contract: cheap diagnostic only (fast, no fits)."""
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

LB = Path(__file__).resolve().parents[1]
DIAG = LB / "experiments/d148_cme_timing/diag"
D142 = LB / "experiments/d142_e0147depth"
FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "baseline-replay.parquet", "official-predictions",
             "predictions_official", "from scorer", "import scorer",
             "from expos", "import expos", "official-score", "official_score",
             "research/expos/runs", "datasets--tlabtlab--sunrun-lb-store",
             "EXPOS_OUT", "isMostAccurate", "--gpu"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "lean.py", "D147", "d147",
                 "nn-", "nn_", "sun-nn", "E0471", "e0471", "E0472", "e0472",
                 "donki-live", "d148_enlil", "ips_oracle", "spec_oracle",
                 "run_oracle", "worker-t1e0", "worker-t1e1", "worker-t1e2",
                 "worker-t1e3", "worker-t1e4", "worker-t1a", "worker-t1b",
                 "worker-t1c", "worker-t1d", "worker-d141", "worker-d147",
                 "worker-d76", "worker-t2no"]
BANNED_SRC = ["GONG", "CDAW", "ADAPT", "WSA", "ENLIL", "PFSS"]
CELL_LABELS = ["low_low_high", "low_high_high", "high_low_high", "high_high_high"]
OFFICIAL_TOKENS = ["2026-05-31", "2026-09-26", "N_OFFICIAL_ORIGINS",
                   "ORIGIN_START", "ORIGIN_END", "lb-eval", "lb-best"]
ARM_TOKENS = ["wPCC", "MDI", "stratified"]
PINNED_W = {"1-6": {"e0003_line": 0.21536233464375223, "d33_hsplit": 0.3899845818930272,
                    "d45_lvlmean": 0.3946530834632206},
            "7-24": {"e0003_line": 0.3152013131693896, "d33_hsplit": 0.3436571856823209,
                     "d45_lvlmean": 0.3411415011482895},
            "25-48": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
                      "d45_lvlmean": 0.3333333333333333},
            "49-72": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
                      "d45_lvlmean": 0.3333333333333333}}


def _load(name):
    loc = importlib.util.spec_from_file_location(f"d148diag_{name}", DIAG / f"{name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(DIAG.glob("*.py")))


def test_owned_file_budget():
    files = sorted(p.name for p in DIAG.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 4, files
    assert {"spec_diag.py", "run_diag.py", "error_decomp.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_diag")
    assert SP.DIRECTION == "D148" and SP.OWNER == "worker-t1e5"
    assert SP.KIND == "diag" and SP.REFERENCE == "E0408"
    assert tuple(SP.SEEDS) == (0,) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24
    assert SP.SCRATCH == "store/scratch/worker-t1e5"
    assert int(SP.FIT_NJOBS) <= 6
    assert SP.BASE_DEPTH == 3
    assert SP.BASE_PARAMS["n_estimators"] == 900
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - 0.03) < 1e-12
    assert SP.BASE_PARAMS.get("device") == "cpu" and int(SP.BASE_PARAMS["n_jobs"]) <= 6
    assert SP.BASE_PARAMS["tree_method"] == "hist"
    assert SP.WEIGHTS == PINNED_W
    assert SP.BASE_COLUMNS == "existing_all+strips17"
    assert SP.PATH_SOURCE == "oof-fit"
    assert SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert "[axes:" in SP.AXES_TAG and "diag" in SP.AXES_TAG
    assert SP.DECISION == "diagnostic-only-no-adoption"
    assert "no adoption" in SP.DIAG_NOTE and "no registration" in SP.DIAG_NOTE
    wins = list(SP.XOOF_WINDOWS)
    assert [w["name"] for w in wins] == ["W0-early", "W1-mid", "W2-main"]
    assert SP.MAIN_WINDOW == "W2-main"
    assert wins[0]["train_end"] == "2026-01-01T00:00:00Z"
    assert wins[1]["train_end"] == "2026-02-01T00:00:00Z"
    assert wins[2]["train_end"] == "2026-03-01T00:00:00Z"
    for w in wins:
        assert w["oof_end"] < SP.CUTOFF_ISO
    assert float(SP.THRESHOLD_Q) == 0.95 and float(SP.EXTREME_FRAC) == 0.05
    assert "P95" in SP.THRESHOLD_RULE


def test_single_axis_vs_recipe():
    SP = _load("spec_diag")
    run_src = (DIAG / "run_diag.py").read_text()
    for mod in ("depth_lib", "info_ablation", "strips", "spec_diag", "error_decomp"):
        assert mod in run_src, mod
    assert "900" in run_src and "0.03" in run_src
    assert "oof-fit" in run_src and "dv = y - yhat_base" in run_src
    assert "existing_all" in run_src and "strip_columns" in run_src
    assert "105" in run_src
    assert SP.BASE_PARAMS["subsample"] == 0.8 and SP.BASE_PARAMS["colsample_bytree"] == 0.7
    assert SP.BASE_PARAMS["min_child_weight"] == 20 and SP.BASE_PARAMS["reg_lambda"] == 5


def test_threshold_uses_train_y_only_and_no_cell_labels():
    run_src = (DIAG / "run_diag.py").read_text()
    assert "threshold_p95" in run_src
    assert "THRESHOLD_Q" in run_src and "EXTREME_FRAC" in run_src
    assert "frac" in run_src and "flags" in run_src
    code = _code()
    for c in CELL_LABELS:
        assert c not in code, c


def test_decompose_identity_and_recal_control():
    ED = _load("error_decomp")
    rng = np.random.default_rng(0)
    y = 400.0 + 30.0 * rng.standard_normal(500)
    p = y * 0.8 + 50.0 + 10.0 * rng.standard_normal(500)
    d = ED.decompose(y, p)
    assert d["n"] == 500
    assert abs(d["mse"] - (d["term_gap"] + d["term_var"] + d["term_corr"])) < 1e-6 * max(1.0, d["mse"])
    assert -1.0 <= d["rho"] <= 1.0
    assert abs(d["slope"] - d["rho"] * d["sd_y"] / d["sd_p"]) < 1e-9
    assert abs(d["intercept"] - (d["mean_y"] - d["slope"] * d["mean_p"])) < 1e-9
    assert d["recal_gain"] >= -1e-9
    assert d["recal_mse"] <= d["mse"] + 1e-9
    thr = ED.threshold_p95(np.arange(100.0), 0.95)
    assert abs(thr - 94.05) < 1e-9
    ym = np.full((2, 72), 400.0)
    ym[0, :4] = 9999.0
    ym[1, :3] = 9999.0
    m = ED.extreme_origin_mask(ym, 500.0, 0.05)
    assert m.tolist() == [True, False]


def test_scratch_isolation_and_cpu():
    SP = _load("spec_diag")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1e5"
    assert len(set(re.findall(r"worker-t1e[0-9]", code))) == 1
    assert "worker-t1e5" in code
    run_src = (DIAG / "run_diag.py").read_text()
    assert 'device") == "cpu"' in run_src and "n_jobs" in run_src
    assert "--gpu" not in code
    assert "cuda" not in code.lower()


def test_no_forbidden_leak_banned_source_or_official():
    code = _code()
    low = code.lower()
    for w in FORBIDDEN:
        assert w.lower() not in low, w
    for w in OTHER_CONTACT:
        assert w not in code, w
    for w in BANNED_SRC:
        assert w.lower() not in low, w
    for w in OFFICIAL_TOKENS:
        assert w not in code, w
    for w in ARM_TOKENS:
        assert w not in code, w
    assert "freeze" not in low and "submit" not in low
