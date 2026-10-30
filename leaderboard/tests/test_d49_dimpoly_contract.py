"""D49 dimpoly contract: cubic-Lasso standalone on dimming+sig only, E0003-free.
Fast: no data reads, no heavy fits beyond small synthetic Lasso (35 cols)."""

import sys
from pathlib import Path

import numpy as np

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d49_dimpoly"))
sys.path.insert(0, str(LB / "experiments" / "d24_dimming"))
sys.path.insert(0, str(LB / "experiments" / "d25_insitu_event"))

import dimpoly as DP  # noqa: E402
import dimming as DM  # noqa: E402
import event as EV  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]
E0003_CODE_TOKENS = ["from existing_all", "import existing_all", "XGBRegressor", "XGBClassifier",
                "import xgboost", "tree_method", "ia.arms", "arms(df",
                '["existing_all"]', "strip_v", "enlil_next", "v_lag0", "rec_mean",
                "ace_speed_kms_m24", "ch_ch_"]


def test_base_cols_exact_35_dimming_plus_sig():
    assert len(DM.dimming_columns()) == 23
    assert len(EV.sig_columns()) == 12
    assert len(DP.POLY_BASE_COLS) == 35 and len(set(DP.POLY_BASE_COLS)) == 35
    assert tuple(DP.POLY_BASE_COLS[:23]) == tuple(DM.dimming_columns())
    assert tuple(DP.POLY_BASE_COLS[23:]) == tuple(EV.sig_columns())
    assert DP.DEGREE == 3
    assert DP.n_poly_features(35, 3) == 8435
    assert DP.check_dimpoly()["n_expanded"] == 8435


def test_e0003_features_entirely_absent():
    for banned in ("v_lag", "v_m", "rec", "ace_", "ch_", "strip_", "enlil_"):
        assert not any(c.startswith(banned) for c in DP.POLY_BASE_COLS), banned
    assert "h" not in DP.POLY_BASE_COLS, "horizon is an E0003 speed-family col here; keep 35 NEW cols only"
    assert tuple(DP.LASSO_ALPHAS) == (0.5, 2.0, 8.0)
    assert DP.DEV_SPLIT == 0.7 and DP.PURGE_H == 72
    assert DP.F1_FLOOR == 0.41121495
    assert DP.PERIOD_JOBS == 3 and DP.MAX_TRAIN_ROWS == 20000
    assert "dev-OOF" in DP.SELECT_RULE and "selection/audit" in DP.SELECT_RULE
    assert "Son" in DP.SELECT_RULE and "72h purge" in DP.SELECT_RULE
    assert "X0066" in DP.SELECT_RULE and "X0069" in DP.SELECT_RULE and "X0090" in DP.SELECT_RULE
    assert "E0003" in DP.SELECT_RULE


def test_full_rule_prespecified_and_report_only():
    assert "full 3h" in DP.FULL_RULE and "E0001" in DP.FULL_RULE
    assert "F1" in DP.FULL_RULE and "0.41121495" in DP.FULL_RULE
    assert "no full run" in DP.FULL_RULE
    assert "freeze" in DP.FULL_RULE.lower() and "official" in DP.FULL_RULE.lower()


def _synthetic_frame(n=220, seed=0):
    rng = np.random.default_rng(seed)
    import pandas as pd

    data = {c: rng.standard_normal(n) * 2.0 for c in DP.POLY_BASE_COLS}
    df = pd.DataFrame(data)
    df["y"] = 400.0 + df[DP.POLY_BASE_COLS[0]] * 3.0 + rng.standard_normal(n) * 5.0
    return df


def test_poly_expansion_width_and_alpha_sparsity():
    df = _synthetic_frame()
    for a in DP.LASSO_ALPHAS:
        m = DP.build_poly(float(a))
        m.fit(df[list(DP.POLY_BASE_COLS)].to_numpy(), df["y"].to_numpy())
        poly = m.named_steps["poly"]
        assert poly.degree == 3 and poly.include_bias is False
        xt = poly.transform(df[list(DP.POLY_BASE_COLS)].to_numpy()[:4])
        assert xt.shape == (4, 8435)
        assert np.isfinite(m.predict(df[list(DP.POLY_BASE_COLS)].to_numpy()[:4])).all()
    nzs = []
    for a in DP.LASSO_ALPHAS:
        m = DP.build_poly(float(a))
        m.fit(df[list(DP.POLY_BASE_COLS)].to_numpy(), df["y"].to_numpy())
        coef = np.asarray(m.named_steps["lasso"].coef_)
        assert len(coef) == 8435
        nzs.append(int((np.abs(coef) > 1e-12).sum()))
    assert nzs[0] >= nzs[-1], f"stronger alpha should be sparser {nzs}"
    assert 0 < nzs[0] <= 8435


def test_select_alpha_and_subsample_rules():
    assert DP.select_alpha({0.5: 100.0, 2.0: 90.0, 8.0: 95.0}) == 2.0
    assert DP.select_alpha({0.5: 90.0, 2.0: 90.0, 8.0: 90.0}) == 0.5
    import pandas as pd

    df = pd.DataFrame({"target_t": pd.date_range("2025-01-01", periods=100, freq="h", tz="UTC"),
                       "y": np.arange(100, dtype=float)})
    sub = DP.subsample_train(df, max_rows=20)
    assert len(sub) == 20
    assert sub["target_t"].is_monotonic_increasing
    assert DP.subsample_train(df, max_rows=200) is df or len(DP.subsample_train(df, max_rows=200)) == 100


def test_owned_files_and_no_vendoring_or_forbidden():
    pkg = LB / "experiments" / "d49_dimpoly"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 5, files
    assert {"dimpoly.py", "runner.py"} <= set(files)
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"
        for s in E0003_CODE_TOKENS:
            assert s not in text, f"{f.name} leaks E0003/XGB code token {s!r}"
    runner = (pkg / "runner.py").read_text()
    assert "import dimming" in runner or "import DM" in runner
    assert "import event" in runner or "import EV" in runner
    assert "import dimpoly" in runner or "import DP" in runner
    assert "store/scratch/worker-d49" in runner
    for other in ("worker-d48", "worker-d50", "worker-d43", "worker-d24", "worker-d25",
                  "worker-d22", "worker-d23", "worker-d33", "worker-d45"):
        assert other not in runner


def test_runner_cpu_cache_poison_smoke_and_scratch():
    runner = (LB / "experiments" / "d49_dimpoly" / "runner.py").read_text()
    assert "dev_oof_select" in runner
    assert "DP.fit_poly" in runner and "DP.predict_poly" in runner and "DP.select_alpha" in runner
    assert "DP.subsample_train" in runner
    assert "poison_check" in runner and '"passed"' in runner
    assert "SystemExit" in runner
    assert "ThreadPoolExecutor" in runner and "max_workers=DP.PERIOD_JOBS" in runner
    assert "ST.build_manifest" in runner
    assert "fcache" in runner.lower() and "featcache" in runner
    assert "EXPOS_SMOKE" in runner
    assert "import info_ablation" in runner and "ia.build" in runner
    assert "X0066" in runner or "X0069" in runner or "Son" in runner
    assert str(DP.SCRATCH) == str(LB / "store" / "scratch" / "worker-d49")
