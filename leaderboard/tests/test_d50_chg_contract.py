"""Contract for D50 change-series (leaderboard/tests/test_d50_chg_contract.py).

Fast unit checks (synthetic metadata, no S3/DB): column contract, proxy rule
(all inputs t<=O), causality (poison passes on synthetic data), linearity of
deltas/slope, and a static scan for forbidden inputs / GPU / geometry reuse.
"""
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments/d1_longitude_strips"))
sys.path.insert(0, str(LB / "experiments/d50_chgseries"))
import chgseries as CH  # noqa: E402

SEP = chr(45)  # hyphen, kept out of literals so the static scan stays clean
FORBIDDEN = ["datasets" + SEP + SEP + "tlabtlab" + SEP + SEP + "sunrun-lb-store",
             "truth" + chr(46) + "parquet",
             "folds" + chr(46) + "parquet", "naive" + chr(46) + "parquet",
             "official" + SEP + "predictions",
             "baseline" + SEP + "replay" + chr(46) + "parquet",
             "official" + chr(95) + "check", "official" + SEP + "check",
             "predictions" + chr(95) + "official" + chr(95) + "truth"]


def _synthetic(n_slots=200, step=0.01, seed=0):
    rng = np.random.default_rng(seed)
    slots = pd.date_range("2025-01-01", periods=n_slots, freq="h", tz="UTC")
    t = np.arange(n_slots, dtype=float)
    meta = pd.DataFrame({
        "slot": slots,
        "obs_end": slots - pd.Timedelta(minutes=30),
        "s3_modified": slots + pd.Timedelta(minutes=50),
        "status": "ok",
        "nc_path": [f"/tmp/d50-test/{i}.nc" for i in range(n_slots)],
    })
    # Linear ramp + tiny noise so deltas/slope have known values.
    block = pd.DataFrame({
        "disk_log_median": 5.0 + step * t + rng.normal(0, 1e-9, n_slots),
        "disk_log_q10": 4.0 + step * t,
        "disk_dark_fraction": 0.1 + 0.0001 * t,
    }, index=slots)
    return meta, block


def _run(monkeypatch=None):
    meta, block = _synthetic()
    origins = pd.date_range("2025-01-05", periods=8, freq="6h", tz="UTC")
    with mock.patch("os.path.isfile", return_value=True), \
         mock.patch("os.path.getsize", return_value=100):
        feats, sel = CH.build_features(origins, meta, block)
        poison = CH.poison_test(origins, "2025-01-06T00:00:00Z", meta=meta, block=block)
    return feats, sel, poison, origins


def test_column_contract():
    cols = CH.chg_columns()
    assert len(cols) == 12, cols
    assert cols[-2:] == ["img_age_h", "img_stale"]
    assert any(c.startswith("chg_logmed_") for c in cols)
    assert "chg_slope_24" in cols


def test_proxy_rule_all_inputs_before_origin():
    feats, sel, _, origins = _run()
    assert (sel["slot_idx"] >= 0).all()
    margin = sel["origin"] - sel["slot"]
    assert (margin >= pd.Timedelta("1h")).all(), margin.min()
    # Every row of every (origin,h) prediction uses only the pre-origin slot.
    assert len(feats) == len(origins) * 72
    assert feats["img_stale"].notna().all()


def test_deltas_match_linear_ramp():
    feats, _, _, _ = _run()
    per_origin = feats.groupby("origin").first()
    # step=0.01/hour -> 6-slot delta ~= 0.06 (noise < 1e-6).
    assert np.allclose(per_origin["chg_logmed_d6"].to_numpy(), 0.06, atol=1e-6)
    assert np.allclose(per_origin["chg_slope_24"].to_numpy(), 0.01, atol=1e-6)
    assert (per_origin["img_stale"] == 0).all()


def test_poison_passes_with_positive_control():
    _, _, poison, _ = _run()
    assert poison["passed"] is True
    assert poison["pre_invariant"] is True
    assert poison["n_pre_rows"] > 0 and poison["n_post_rows"] > 0
    assert len(poison["moved_columns"]) > 0


def test_static_source_contract():
    for name in ("chgseries.py", "runner.py"):
        text = (LB / "experiments/d50_chgseries" / name).read_text()
        for pat in FORBIDDEN:
            assert pat not in text, (name, pat)
    chg = (LB / "experiments/d50_chgseries" / "chgseries.py").read_text()
    runner_text = (LB / "experiments/d50_chgseries" / "runner.py").read_text()
    for gpu_pat in ("tree_method=\"gpu\"", "gpu_lease", "--gpu", "prefer-gpu"):
        assert gpu_pat not in chg + runner_text, gpu_pat
    assert "no GPU" in runner_text  # CPU-only guard documented
    runner = (LB / "experiments/d50_chgseries" / "runner.py").read_text()
    assert "tree_method\") == \"hist\"" in runner
    assert "n_jobs" in runner
    chg = (LB / "experiments/d50_chgseries" / "chgseries.py").read_text()
    for geom in ("OMEGA", "TAU", "PARKER", "strip_weights"):
        assert geom not in chg, geom  # no geometry redefinition; disk aggregates only
    assert "mse20-gjc-rsi-amendment-20261005" in runner  # proxy-rule authority cited
