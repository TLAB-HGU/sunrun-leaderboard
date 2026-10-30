"""D10 CH-aligned analog contract: causality, dev-only keys, guards."""
import inspect
import sys

import numpy as np
import pandas as pd
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "experiments" / "d10_analog"
sys.path.insert(0, str(PKG))
import analog as AN  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]
WINDOW_DATES = ("2025-12-04", "2026-02-25", "2026-03-04", "2026-05-28",
                "2026-05-29", "2026-05-31", "2026-09-26")


def _synth_frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({"v": np.abs(rng.normal(430, 40, n)) + 200.0}, index=idx)
    for c in AN.CH_KEEP:
        df[c] = np.abs(rng.normal(0.01, 0.005, n))
    return df


def test_analog_columns_fixed_new_and_prefixed():
    cols = AN.analog_columns()
    assert cols == ["analog_best", "analog_mean3", "analog_dist", "analog_spread"]
    assert len(set(cols)) == 4
    assert all(c.startswith("analog_") for c in cols)
    for c in cols:
        assert not c.startswith(("v_", "rec", "ace_", "ch_", "enlil_", "shock_", "strip_"))
        assert c not in ("h", "speed_d24", "epam_p1_d6")


def test_key_sets_cover_ch_and_speed():
    assert set(AN.KEY_SETS) == {"CH6", "CH6V2", "CH3V3"}
    assert set(AN.METRICS) == {"l2", "l1"}
    assert len(AN.KEY_SETS["CH6"]) == 6 and len(AN.KEY_SETS["CH6V2"]) == 8
    assert all(k.startswith("s_") for k in AN.KEY_SETS["CH6V2"])
    assert any("v_" in k for k in AN.KEY_SETS["CH6V2"])


def test_state_causal_truncation_invariant():
    x = _synth_frame()
    cut = x.index[200]
    full = AN.add_state_features(x)
    trunc = AN.add_state_features(x.loc[:cut])
    assert list(full.columns) == list(trunc.columns)
    pre = full.index[full.index < cut - pd.Timedelta("30h")]
    for c in full.columns:
        a = full.loc[pre, c].to_numpy(float)
        b = trunc.reindex(pre)[c].to_numpy(float)
        assert np.allclose(a, b, equal_nan=True), c


def test_future_moves_state_positive_control():
    x = _synth_frame()
    y = x.copy()
    y.loc[y.index[100:120], "v"] += 200.0
    a = AN.add_state_features(x).iloc[110]
    b = AN.add_state_features(y).iloc[110]
    assert any(abs(a[c] - b[c]) > 0 for c in a.index
               if np.isfinite(a[c]) and np.isfinite(b[c]))


def test_analog_features_takes_origins_only():
    assert list(inspect.signature(AN.analog_features).parameters) == ["origins"]
    src = (PKG / "analog.py").read_text().lower()
    for word in ("selection", "audit", "official"):
        assert word not in src, f"analog.py must not contain {word!r}"


def test_select_config_takes_dev_only():
    assert list(inspect.signature(AN.select_config).parameters) == ["dev_scores"]
    w, m = AN.select_config({"CH6/l2": 100.0, "CH6V2/l2": 90.0, "CH3V3/l1": 95.0})
    assert (w, m) == ("CH6V2", "l2")


def test_lookup_window_strictly_before_origin():
    assert AN.WIN_LO_H == 480 and AN.WIN_HI_H == 840 and AN.K_TOP == 3
    origins = pd.DatetimeIndex(["2025-09-01T00:00:00Z", "2025-10-01T00:00:00Z"], tz="UTC")
    for o in origins:
        latest = AN.max_lookup_time(pd.DatetimeIndex([o]))
        assert latest < o
        assert (o - latest) >= pd.Timedelta(hours=AN.WIN_LO_H - 72)


def test_no_forbidden_or_window_dates_in_new_code():
    for name in ("analog.py", "runner.py"):
        text = (PKG / name).read_text()
        assert not [p for p in FORBIDDEN if p in text], name
        assert not [s for s in WINDOW_DATES if s in text], name
        assert "2026-06" not in text, f"{name} must not reference the official window"


def test_runner_uses_cuda_pilot_purge_poison_manifest():
    src = (PKG / "runner.py").read_text()
    assert '"cuda"' in src and "device" in src
    assert "--pilot" in src and '"6h"' in src
    assert "72h" in src
    assert "poison.json" in src and "pred_invariant" in src
    assert "positive_control" in src
    assert "manifest.json" in src and "selection.json" in src and "importance.json" in src
    assert "get_score" in src
    assert "raw.get(c" in src
    assert "select_config(" in src
    assert "dev rows only" in src


def test_runner_selects_keys_on_dev_only():
    src = (PKG / "runner.py").read_text()
    assert "dev_scores_for(" in src and "select_config(scores)" in src
    low = src.lower()
    assert "scores[\"selection\"]" not in low and "scores['selection']" not in low


def test_owned_dir_is_small():
    files = sorted(p.name for p in PKG.iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"analog.py", "runner.py"} <= set(files)
