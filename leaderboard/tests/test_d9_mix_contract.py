"""D9 mixture-quantile contract: regimes, dev-only selection, mixing math, causality guards."""
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1] / "experiments" / "d9_mixture"
sys.path.insert(0, str(PKG))
import mix as MX  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]


def _dev(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    v0 = rng.normal(420, 60, n)
    h = np.tile(np.arange(1, 73), n // 72 + 1)[:n]
    y = v0 + rng.normal(0, 25, n)
    mean = y - 40.0  # systematic underprediction like X0008
    med = y - 35.0 + rng.normal(0, 5, n)
    q90 = y + 30.0 + rng.normal(0, 5, n)
    return y, mean, med, q90, v0, h


def test_quantile_params_share_recipe():
    base = dict(n_estimators=600, max_depth=6, device="cuda")
    q = MX.quantile_params(base, 0.9)
    assert q["objective"] == "reg:quantileerror" and q["quantile_alpha"] == 0.9
    assert q["n_estimators"] == 600 and q["device"] == "cuda"
    assert "objective" not in base  # input untouched


def test_regime_cuts_are_tertiles_and_frozen():
    v0 = np.arange(1, 301, dtype=float)
    cuts = MX.regime_cuts(v0)
    assert cuts[0] < cuts[1]
    r = MX.regime_index(v0, cuts)
    assert set(np.unique(r)) == {0, 1, 2}
    assert (r[:100] == 0).all() and (r[200:] == 2).all()


def test_regime_nan_maps_to_mid():
    v0 = np.array([100.0, np.nan, 900.0])
    assert MX.regime_index(v0, (300.0, 600.0)).tolist() == [0, 1, 2]


def test_band_index_edges():
    assert MX.band_index([1, 6, 7, 24, 25, 48, 49, 72]).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_select_takes_dev_arrays_only():
    params = list(inspect.signature(MX.select_mixture).parameters)
    assert params == ["y", "mean", "med", "q90", "v0", "h"]
    src = inspect.getsource(MX.select_mixture)
    for word in ("selection", "audit", "official", "folds"):
        assert word not in src


def test_select_improves_or_ties_and_weights_simplex():
    sel = MX.select_mixture(*_dev())
    assert sel["dev_mse_mix"] <= sel["dev_mse_base"]
    assert sel["dev_mse_mix"] < sel["dev_mse_base"]  # q90 must earn weight on biased heads
    assert len(sel["cells"]) == 12 and len(sel["table"]) > 0
    for b in sel["cells"].values():
        assert b["w_mean"] >= 0 and b["w_med"] >= 0 and b["w_q90"] >= 0
        assert abs(b["w_mean"] + b["w_med"] + b["w_q90"] - 1.0) < 1e-9


def test_select_falls_back_to_mean_when_quantiles_hurt():
    rng = np.random.default_rng(3)
    n = 3000
    y = rng.normal(400, 20, n)
    v0 = rng.normal(420, 40, n)
    h = np.tile(np.arange(1, 73), n // 72 + 1)[:n]
    sel = MX.select_mixture(y, y.copy(), y + 500.0, y + 800.0, v0, h)
    assert sel["dev_mse_mix"] == sel["dev_mse_base"]
    for b in sel["cells"].values():
        assert (b["w_mean"], b["w_med"], b["w_q90"]) == (1.0, 0.0, 0.0)


def test_apply_mixture_math_and_frozen_sel():
    sel = {"cuts": [300.0, 600.0], "n_regimes": 3, "bands": list(MX.BAND_NAMES),
           "cells": {str(c): {"w_mean": 1.0, "w_med": 0.0, "w_q90": 0.0} for c in range(12)}}
    sel["cells"]["0"] = {"w_mean": 0.6, "w_med": 0.3, "w_q90": 0.1}
    mean = np.array([400.0, 400.0])
    out = MX.apply_mixture(mean, np.array([410.0, 410.0]), np.array([500.0, 500.0]),
                           np.array([100.0, 800.0]), np.array([3, 60]), sel)
    assert out.tolist() == [0.6 * 400 + 0.3 * 410 + 0.1 * 500, 400.0]


def test_distribution_definition_reports_point_and_regimes():
    sel = MX.select_mixture(*_dev(n=1500, seed=1))
    d = MX.distribution_definition(sel)
    assert d["regimes"]["cuts_kms"] == sel["cuts"]
    assert "w_mean" in d["point_forecast"] and "q90" in d["point_forecast"]
    assert d["dev_mse_mix"] <= d["dev_mse_base"]


def test_no_forbidden_strings_in_owned_code():
    for f in ("mix.py", "runner.py"):
        text = (PKG / f).read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f} contains forbidden {s!r}"
        assert "2026-06" not in text, f"{f} must not reference the official window"


def test_owned_dir_is_small_and_runner_has_pilot_and_purge():
    files = sorted(p.name for p in PKG.iterdir() if p.suffix == ".py")
    assert len(files) <= 5, files
    assert set(files) == {"mix.py", "runner.py"}
    src = (PKG / "runner.py").read_text()
    assert "--pilot" in src and "72h" in src and "ia.build" in src
    assert "select_mixture" in src and "poison" in src.lower()
    assert "device" in src and "cuda" in src
