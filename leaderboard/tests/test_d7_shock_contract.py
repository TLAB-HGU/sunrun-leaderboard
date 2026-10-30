"""D7 shock-precursor contract: causality, E0003-independence, dev-only, guards."""
import inspect
import sys

import numpy as np
import pandas as pd
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "experiments" / "d7_shock_precursor"
sys.path.insert(0, str(PKG))
import shock as SH  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]
# Reuse of image maps or prior-experiment predictions would break the axis.
INDEPENDENCE_BANS = ["e0002", "strips", "suvi", "lonmap", "cnn.py"]


def _synth_hourly(n=240, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({
        "ace_epam_p_channel1": np.abs(rng.normal(5000, 1000, n)) + 1.0,
        "ace_epam_p_channel3": np.abs(rng.normal(500, 100, n)) + 1.0,
        "ace_epam_p_channel5": np.abs(rng.normal(50, 10, n)) + 1.0,
        "ace_bt_nt": np.abs(rng.normal(6, 2, n)) + 0.1,
        "ace_bz_gsm_nt": rng.normal(0, 3, n),
        "ace_density_cm3": np.abs(rng.normal(6, 2, n)) + 0.1,
        "ace_speed_kms": np.abs(rng.normal(430, 40, n)) + 1.0,
    }, index=idx)
    return df


def test_shock_columns_fixed_new_and_prefixed():
    cols = SH.shock_columns()
    assert len(cols) == 15 and len(set(cols)) == 15
    assert all(c.startswith("shock_") for c in cols)
    # disjoint from every existing_all family by construction (prefix match)
    for c in cols:
        assert not c.startswith(("v_", "rec", "ace_", "ch_", "enlil_"))
        assert c not in ("h", "speed_d24", "epam_p1_d6")


def test_features_causal_truncation_invariant():
    x = _synth_hourly()
    cut = x.index[150]
    full = SH.add_shock_features(x)
    trunc = SH.add_shock_features(x.loc[:cut])
    assert list(full.columns) == SH.shock_columns()
    assert list(trunc.columns) == SH.shock_columns()
    # origins strictly before the cut are bit-identical with/without future
    pre = full.index[full.index < cut - pd.Timedelta("30h")]
    for c in SH.shock_columns():
        a = full.loc[pre, c].to_numpy(float)
        b = trunc.reindex(pre)[c].to_numpy(float)
        assert np.allclose(a, b, equal_nan=True), c


def test_future_shock_moves_features_positive_control():
    x = _synth_hourly()
    y = x.copy()
    y.loc[y.index[100:120], "ace_epam_p_channel1"] *= 50.0
    a = SH.add_shock_features(x).iloc[110]
    b = SH.add_shock_features(y).iloc[110]
    assert any(abs(a[c] - b[c]) > 0 for c in SH.shock_columns()
               if np.isfinite(a[c]) and np.isfinite(b[c]))


def test_shock_features_takes_origins_only():
    assert list(inspect.signature(SH.shock_features).parameters) == ["origins"]
    src = (PKG / "shock.py").read_text().lower()
    for word in ("selection", "audit", "official"):
        assert word not in src


def test_no_forbidden_or_official_window_strings():
    for f in ("shock.py", "runner.py"):
        text = (PKG / f).read_text()
        assert not [p for p in FORBIDDEN if p in text], f
        assert "2026-06" not in text, f"{f} must not reference the official window"


def test_e0003_independent_axis():
    for f in ("shock.py", "runner.py"):
        text = (PKG / f).read_text().lower()
        assert not [b for b in INDEPENDENCE_BANS if b in text], f
    src = (PKG / "runner.py").read_text()
    assert "independent_of_E0003" in src  # manifest states the independent axis
    for line in src.splitlines():  # E0003 only in the standalone-axis flag
        if "E0003" in line:
            assert "independent_of_E0003" in line, line


def test_runner_uses_cuda_pilot_purge_poison_importance():
    src = (PKG / "runner.py").read_text()
    assert '"cuda"' in src and "device" in src  # XGBoost device=cuda
    assert "--pilot" in src and '"6h"' in src
    assert "72h" in src  # 72h purge kept
    assert "poison.json" in src and "pred_invariant" in src
    assert "positive_control" in src
    assert "importance.json" in src and "get_score" in src
    assert "raw.get(c" in src  # name-keyed gain mapping (f-channel fallback only)
    assert "manifest.json" in src


def test_owned_dir_is_small():
    files = sorted(p.name for p in PKG.iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"shock.py", "runner.py"} <= set(files)
