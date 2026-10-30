"""D16 combo contract: joint strips+precursors, import-only, dev-only, causal."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
PKG = LB / "experiments" / "d16_combo"
sys.path.insert(0, str(LB / "experiments/d1_longitude_strips"))
sys.path.insert(0, str(LB / "experiments/d7_shock_precursor"))
sys.path.insert(0, str(PKG))
import combo as CB  # noqa: E402
import shock as SH  # noqa: E402
import strips as ST  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]
COPY_BANS = ["def strip_weights", "def strip_features", "def select_slots",
             "def add_shock_features", "def lam(", "PARKER_WEST", "OVERLAP"]


def test_combo_columns_17_plus_15():
    assert len(ST.strip_columns()) == 17
    assert len(SH.shock_columns()) == 15
    extra = CB.combo_extra_columns()
    assert len(extra) == 32 and len(set(extra)) == 32
    assert extra == ST.strip_columns() + SH.shock_columns()
    assert all(c.startswith(("strip_", "img_")) for c in extra[:17])
    assert all(c.startswith("shock_") for c in extra[17:])


def test_import_only_no_copy():
    for f in ("combo.py", "runner.py"):
        text = (PKG / f).read_text()
        assert "import strips" in text or "import shock" in text, f
        for ban in COPY_BANS:
            assert ban not in text, f"{f} copies {ban}"


def test_shock_stays_trailing_only():
    rng = np.random.default_rng(0)
    n = 240
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    x = pd.DataFrame({
        "ace_epam_p_channel1": np.abs(rng.normal(5000, 1000, n)) + 1.0,
        "ace_epam_p_channel3": np.abs(rng.normal(500, 100, n)) + 1.0,
        "ace_epam_p_channel5": np.abs(rng.normal(50, 10, n)) + 1.0,
        "ace_bt_nt": np.abs(rng.normal(6, 2, n)) + 0.1,
        "ace_bz_gsm_nt": rng.normal(0, 3, n),
        "ace_density_cm3": np.abs(rng.normal(6, 2, n)) + 0.1,
        "ace_speed_kms": np.abs(rng.normal(430, 40, n)) + 1.0,
    }, index=idx)
    cut = x.index[150]
    full = SH.add_shock_features(x)
    trunc = SH.add_shock_features(x.loc[:cut])
    pre = full.index[full.index < cut - pd.Timedelta("30h")]
    for c in SH.shock_columns():
        assert np.allclose(full.loc[pre, c].to_numpy(float),
                           trunc.reindex(pre)[c].to_numpy(float), equal_nan=True), c


def test_runner_joint_cuda_pilot_purge_poison_curves():
    src = (PKG / "runner.py").read_text()
    assert '"cuda"' in src and "device" in src
    assert "--pilot" in src and '"6h"' in src
    assert "72h" in src
    assert "poison.json" in src and "pred_invariant" in src
    assert "positive_control" in src
    assert "importance.json" in src
    assert "get_score" in (PKG / "combo.py").read_text()
    assert "learning_curves.json" in src and "evals_result" in src
    assert "dev_internal_curves" in src
    assert "manifest.json" in src
    assert "store/scratch/worker-d16" in src
    assert "CB.add_combo" in src or "add_combo" in src
    assert "Optuna" not in src and "optuna" not in src


def test_no_selection_fed_choice():
    src = (PKG / "runner.py").read_text()
    for line in src.splitlines():
        low = line.lower()
        if "inverse-mse" in low or "oof selects pure" in low:
            raise AssertionError(line)
    assert "selection_rule" in src  # manifest states dev-only discipline


def test_no_forbidden_or_official_window():
    for f in ("combo.py", "runner.py"):
        text = (PKG / f).read_text()
        assert not [p for p in FORBIDDEN if p in text], f
        assert "2026-06" not in text, f


def test_owned_dir_is_small():
    files = sorted(p.name for p in PKG.iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"combo.py", "runner.py"} <= set(files)
