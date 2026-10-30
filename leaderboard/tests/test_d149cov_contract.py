"""D149 covariance-diag contract: static + synthetic checks (fast, no fits)."""
import importlib.util
import re
from pathlib import Path

import numpy as np

LB = Path(__file__).resolve().parents[1]
DIAG = LB / "experiments/d149_enlil_cells/diag_cov"
FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "predictions_official", "research/expos/runs",
             "datasets--tlabtlab--sunrun-lb-store",
             "EXPOS_OUT", "isMostAccurate", "--gpu",
             "from scorer", "import scorer", "from expos", "import expos",
             "official-score", "official_score"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "lean.py", "D147", "d147",
                 "d148_enlil", "nn-", "sun-nn", "E0471", "e0471",
                 "ips_oracle", "spec_oracle", "run_oracle",
                 "spec_ec", "cells_lib", "run_ec",
                 "diag_tail", "diag_rise", "diag_dis",
                 "spec_tail", "spec_rise", "spec_dis",
                 "tail_lib", "rise_lib", "disagree_lib",
                 "run_tail", "run_rise", "run_dis",
                 "worker-t1f0", "worker-t1f1", "worker-t1f2", "worker-t1f3",
                 "worker-t1f4", "worker-t1e", "worker-t1a",
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
    loc = importlib.util.spec_from_file_location(f"d149cov_{name}", DIAG / f"{name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(DIAG.glob("*.py")))


def test_owned_file_budget():
    files = sorted(p.name for p in DIAG.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 4, files
    assert {"spec_cov.py", "cov_lib.py", "run_cov.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_cov")
    assert SP.DIRECTION == "D149" and SP.OWNER == "worker-t1f5"
    assert SP.KIND == "diag" and SP.REFERENCE == "E0472"
    assert tuple(SP.SEEDS) == (0,) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24
    assert SP.SCRATCH == "store/scratch/worker-t1f5"
    assert int(SP.FIT_NJOBS) <= 6
    assert SP.BASE_DEPTH == 3 and tuple(SP.DEPTH_VARIANTS) == (4, 5)
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
    assert float(SP.WORST_FRAC) == 0.05 and float(SP.TOP_FRAC) == 0.10
    assert abs(float(SP.REF_MSE) - 3281.3) < 1e-6
    assert "level" in SP.MEMBER_RULE and "depth" in SP.MEMBER_RULE
    assert "mixture" in SP.MEMBER_RULE and "single" in SP.MEMBER_RULE
    assert "fold" not in SP.MEMBER_RULE
    assert tuple(SP.INDICATORS) == ("cov_spread", "spread_1-6", "spread_7-24",
                                    "spread_25-48", "spread_49-72", "eig_timing",
                                    "quant_width", "resid_var")
    assert "AUC" in SP.STOP_RULE and "share" in SP.STOP_RULE
    assert "AUC" in SP.GATE_RULE and "share" in SP.GATE_RULE
    assert "modelCompletionTime" in SP.ENLIL_RULE and "multi-round" in SP.ENLIL_RULE
    assert SP.PATH_SOURCE == "oof-fit" and SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert "[axes:" in SP.AXES_TAG and "covariance" in SP.AXES_TAG
    assert SP.DECISION == "diagnostic-only-no-adoption"
    assert "no adoption" in SP.DIAG_NOTE and "no registration" in SP.DIAG_NOTE


def test_single_axis_vs_recipe():
    run_src = (DIAG / "run_cov.py").read_text()
    for mod in ("info_ablation", "strips", "spec_cov", "cov_lib"):
        assert mod in run_src, mod
    assert "150" in run_src and "0.03" in run_src
    assert "oof-fit" in run_src and "dv = y - yhat_base" in run_src
    assert "existing_all" in run_src and "strip_columns" in run_src
    assert "105" in run_src
    assert "level_seeds" in run_src and "event_seed" in run_src
    assert "fit_params" in run_src
    assert "mixture" in run_src and "single" in run_src
    assert "cov_spread" in run_src and "eig_timing" in run_src
    assert "quant_width" in run_src and "resid_var" in run_src
    assert "spearman" in run_src or "contrast" in run_src
    assert "path_stds" in run_src and "lead_spreads" in run_src
    assert "eig_timing_level" in run_src and "cov_spread" in run_src
    assert "arrival_spread" in run_src and "modelCompletionTime" in run_src
    assert "donki-live" in run_src and "vintage" in run_src
    assert "oof_start" in run_src and "oof_end" in run_src
    assert "cov_report.json" in run_src


def test_cov_math_identity():
    DL = _load("cov_lib")
    assert tuple(DL.level_seeds(0)) == (0, 1, 2)
    assert tuple(DL.level_seeds(2)) == (6, 7, 8)
    assert DL.event_seed(1) in DL.level_seeds(1)
    p = DL.fit_params({"n_estimators": 150}, 5, 7)
    assert p["max_depth"] == 5 and p["random_state"] == 7
    rng = np.random.default_rng(0)
    n = 400
    se = rng.chisquare(3, n) * 100.0
    score = se + rng.normal(0, 50, n)
    r = DL.spearman(score, se)
    assert 0.5 < r <= 1.0
    assert DL.spearman(np.ones(n), se) == 0.0
    lab = DL.worst_label(se, 0.05)
    assert 0 < int(lab.sum()) <= int(0.10 * n) + 2
    auc = DL.auc_score(score, lab)
    assert 0.5 < auc <= 1.0
    shr = DL.top_share(score, se, 0.10)
    assert shr > 0.10
    assert abs(DL.auc_score(np.arange(n, dtype=float),
                            DL.worst_label(np.arange(n, dtype=float), 0.05)) - 1.0) < 1e-9
    s = DL.summarize(score, se)
    assert s["n"] == n and -1.0 <= s["spearman"] <= 1.0
    assert 0.0 <= s["auc_worst5"] <= 1.0 and 0.0 <= s["top10_share"] <= 1.0
    tab = DL.contrast_table({"a": score, "b": np.zeros(n)}, se)
    assert set(tab) == {"a", "b"} and tab["a"]["n"] == n
    m = np.column_stack([score, score + 1.0, score - 1.0])
    assert np.all(DL.quantile_width(m) > 0)
    # path covariance structure on synthetic timing vs level disagreement
    t = np.arange(72, dtype=float)
    base = 400.0 + 0.5 * t
    shift = np.zeros(72)
    shift[36:] = 60.0
    paths_level = np.stack([base, base + 10.0, base - 10.0,
                            base + 5.0, base - 5.0], axis=0)
    paths_timing = np.stack([base, base + shift, base - 0.5 * shift,
                             base + 0.25 * shift, base - 0.25 * shift], axis=0)
    stds = DL.path_stds(paths_level)
    assert stds.shape == (72,)
    assert np.all(stds > 0)
    lsp = DL.lead_spreads(stds)
    assert set(lsp) == {"1-6", "7-24", "25-48", "49-72"}
    assert DL.cov_spread(paths_level) > 0
    c = DL.path_cov(paths_level)
    assert c.shape == (72, 72)
    e_level = DL.eig_timing_level(paths_level)
    assert abs(e_level["timing_corr"]) < 0.5
    assert e_level["level_score"] > 0.8
    e_timing = DL.eig_timing_level(paths_timing)
    assert abs(e_timing["timing_corr"]) > 0.5
    assert np.isfinite(DL.arrival_spread(np.array([10.0, 20.0, 30.0])))
    assert not np.isfinite(DL.arrival_spread(np.array([10.0])))


def test_scratch_isolation_and_cpu():
    SP = _load("spec_cov")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1f5"
    assert len(set(re.findall(r"worker-t1f[0-9]", code))) == 1
    assert "worker-t1f5" in code
    run_src = (DIAG / "run_cov.py").read_text()
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


def test_member_plumbing_covers_all_types():
    run_src = (DIAG / "run_cov.py").read_text()
    assert "DEPTH_VARIANTS" in run_src and "BASE_DEPTH" in run_src
    assert "blend_members" in run_src and "mixture" in run_src
    assert "single" in run_src
    assert "fit_full" in run_src
    assert "target_t" in run_src and "train_end" in run_src
    code = _code()
    assert code.count("fit_full") >= 2
    assert "lead_spreads" in run_src and "eig_timing_level" in run_src
    assert "live_runs_meta" in run_src and "arrival_spread_for" in run_src
    assert "cov_report.json" in run_src and '"gate"' in run_src
