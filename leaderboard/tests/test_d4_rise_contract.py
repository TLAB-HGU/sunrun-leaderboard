"""D4 rise-head contract: labeling, dev-only gate, gating math, causality guards."""
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1] / "experiments" / "d4_rise_head"
sys.path.insert(0, str(PKG))
import rise as RZ  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]


def _long(n_org=6, seed=0):
    rng = np.random.default_rng(seed)
    origins = pd.date_range("2025-08-04", periods=n_org, freq="6h", tz="UTC")
    rows = []
    for o in origins:
        v0 = 380.0 + rng.normal(0, 10)
        for h in range(1, 73):
            jump = 200.0 if (o.hour == 0 and h == 60) else rng.normal(0, 20)
            rows.append({"origin": o, "h": h, "y": v0 + jump, "v_lag0": v0,
                         "rec26": 400.0, "x": rng.normal()})
    return pd.DataFrame(rows)


def test_label_boundary_150():
    df = _long()
    lab = RZ.origin_labels(df)
    assert lab.sum() >= 1  # the h==60 +200 jumps on midnight origins
    one = df[df["origin"] == df["origin"].iloc[0]]
    assert bool(lab.loc[one["origin"].iloc[0]]) == bool((one["y"] - one["v_lag0"]).max() >= 150.0)


def test_label_nan_jump_is_not_rise():
    df = _long(n_org=1)
    df["y"] = df["v_lag0"] + 10.0
    df.loc[df["h"] == 72, "y"] = np.nan
    lab = RZ.origin_labels(df.dropna(subset=["y"]))
    assert bool(lab.iloc[0]) is False


def test_origin_cols_drop_per_row_only():
    cols = ["h", "rec26", "rec27", "rec28", "rec_mean", "v_lag0", "ace_speed_kms_last",
            "ch_ch_t035_area"]
    got = RZ.origin_cols(cols)
    assert got == ["v_lag0", "ace_speed_kms_last", "ch_ch_t035_area"]


def test_band_index_edges():
    assert RZ.band_index([1, 6, 7, 24, 25, 48, 49, 72]).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_select_gate_takes_dev_arrays_only():
    params = list(inspect.signature(RZ.select_gate).parameters)
    assert params == ["y", "base", "p", "h"]
    src = inspect.getsource(RZ.select_gate)
    for word in ("selection", "audit", "official", "folds"):
        assert word not in src


def test_select_gate_improves_and_clips():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.normal(400, 30, n)
    base = y - 60.0  # systematic underprediction
    p = np.where(np.arange(n) % 4 == 0, 0.9, 0.1)
    h = np.tile(np.arange(1, 73), n // 72 + 1)[:n]
    gate = RZ.select_gate(y, base, p, h)
    assert gate["tau"] in RZ.TAU_GRID and gate["tau"] < 1.0
    assert gate["dev_mse_gated"] < gate["dev_mse_base"]
    for d in gate["deltas"].values():
        assert 0.0 <= d <= 250.0
    assert len(gate["table"]) == len(RZ.TAU_GRID)


def test_select_gate_falls_back_to_never_gate():
    n = 2000
    y = np.arange(n, dtype=float)
    base = y.copy()  # perfect head: any correction can only hurt or tie
    p = np.linspace(0.01, 0.99, n)
    h = np.tile(np.arange(1, 73), n // 72 + 1)[:n]
    gate = RZ.select_gate(y, base, p, h)
    assert gate["tau"] == 1.0 and gate["dev_mse_gated"] == gate["dev_mse_base"]
    assert all(v == 0.0 for v in gate["deltas"].values())


def test_apply_gate_math_and_tau_frozen():
    base = np.array([400.0, 400.0, 400.0])
    h = np.array([3, 10, 60])
    p = np.array([0.9, 0.9, 0.1])
    deltas = {"1-6": 50.0, "7-24": 0.0, "25-48": 80.0, "49-72": 100.0}
    out = RZ.apply_gate(base, h, p, 0.5, deltas)
    assert out.tolist() == [450.0, 400.0, 400.0]
    assert RZ.apply_gate(base, h, p, 1.0, deltas).tolist() == base.tolist()


def test_train_classifier_needs_both_classes():
    import pandas as pd
    try:
        RZ.train_classifier(pd.DataFrame({"x": [1.0, 2.0]}), np.array([0, 0]),
                            params=dict(n_estimators=2, random_state=0))
    except ValueError:
        pass
    else:
        raise AssertionError("single-class fit must fail closed")


def test_no_forbidden_strings_in_owned_code():
    for f in ("rise.py", "runner.py"):
        text = (PKG / f).read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f} contains forbidden {s!r}"
        assert "2026-06" not in text, f"{f} must not reference the official window"


def test_owned_dir_is_small_and_runner_has_pilot_and_purge():
    files = sorted(p.name for p in PKG.iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    src = (PKG / "runner.py").read_text()
    assert "--pilot" in src and "72h" in src and "ia.build" in src
    assert "select_gate" in src and "poison" in src.lower()
