"""D26 TCN-newrep contract: causality, proxy discipline, dev-only, budget, guards."""
import inspect
import sys

import numpy as np
import pandas as pd
import torch
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "experiments" / "d26_tcn_newrep"
sys.path.insert(0, str(PKG))
import features as FE  # noqa: E402
import model as MO  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]


def _synth(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    v = 400 + np.cumsum(rng.normal(0, 2, n))
    speed = pd.DataFrame({"speed": v.astype(np.float32),
                          "miss": np.zeros(n, dtype=np.float32)}, index=idx)
    med = 3 + np.cumsum(rng.normal(0, 0.01, n))
    suvi = pd.DataFrame({"dim_drop": (-np.diff(med, prepend=med[0])).astype(np.float32),
                         "dim_miss": np.zeros(n, dtype=np.float32)}, index=idx)
    s3 = pd.Series(idx.to_numpy(), index=idx) - pd.Timedelta("10min")
    epam = pd.DataFrame({"epam_lp1": rng.normal(4, 0.3, n).astype(np.float32),
                         "bt": np.abs(rng.normal(5, 1, n)).astype(np.float32)}, index=idx)
    return speed, suvi, s3, epam


def test_history_matrix_shape_and_trailing():
    speed, suvi, s3, epam = _synth()
    oo = pd.DatetimeIndex([speed.index[200], speed.index[300]], tz="UTC")
    m = FE.matrix_from_hourly(speed, suvi, s3, epam, oo)
    assert m.shape == (2, FE.HIST, 6)
    assert FE.NCH == 6 and len(FE.channel_names()) == 6
    assert np.allclose(m[0, :, 0], speed["speed"].to_numpy()[200 - FE.HIST + 1:201])
    # dimming reads one hour earlier (proxy margin): last position == hour o-1h
    assert np.allclose(m[0, :, 2], suvi["dim_drop"].to_numpy()[200 - FE.HIST:200])
    assert np.allclose(m[1, :, 4], epam["epam_lp1"].to_numpy()[300 - FE.HIST + 1:301])
    assert np.allclose(m[1, :, 5], epam["bt"].to_numpy()[300 - FE.HIST + 1:301])


def test_s3_mask_kills_ineligible_dim_hours():
    speed, suvi, s3, epam = _synth()
    oo = pd.DatetimeIndex([speed.index[200]], tz="UTC")
    s3b = s3.copy()
    s3b.iloc[150:200] = s3b.iloc[150:200] + pd.Timedelta(days=30)  # s3 > origin
    m = FE.matrix_from_hourly(speed, suvi, s3b, epam, oo)
    assert np.isnan(m[0, -50:, 2]).all()
    assert (m[0, -50:, 3] == 1.0).all()
    assert np.isfinite(m[0, :-50, 2]).all()  # eligible hours untouched


def test_history_causal_truncation_invariant():
    speed, suvi, s3, epam = _synth()
    oo = pd.DatetimeIndex([speed.index[200]], tz="UTC")
    a = FE.matrix_from_hourly(speed.iloc[:250], suvi.iloc[:250], s3.iloc[:250],
                              epam.iloc[:250], oo)
    b = FE.matrix_from_hourly(speed, suvi, s3, epam, oo)
    assert np.allclose(a, b, equal_nan=True)


def test_past_change_moves_matrix_positive_control():
    speed, suvi, s3, epam = _synth()
    oo = pd.DatetimeIndex([speed.index[200]], tz="UTC")
    a = FE.matrix_from_hourly(speed, suvi, s3, epam, oo)
    y = speed.copy()
    y.iloc[150:201, 0] = y.iloc[150:201, 0] + 500.0
    b = FE.matrix_from_hourly(y, suvi, s3, epam, oo)
    assert not np.allclose(a[:, :, 0], b[:, :, 0], equal_nan=True)
    z = suvi.copy()
    z.iloc[150:200, 0] = z.iloc[150:200, 0] + 5.0
    c = FE.matrix_from_hourly(speed, z, s3, epam, oo)
    assert not np.allclose(a[:, :, 2], c[:, :, 2], equal_nan=True)


def test_history_takes_origins_only():
    assert list(inspect.signature(FE.history_matrix).parameters) == ["origins"]
    assert list(inspect.signature(FE.matrix_from_hourly).parameters) == [
        "speed", "suvi", "s3", "epam", "origins"]


def test_tcn_uses_left_causal_padding():
    src = (PKG / "model.py").read_text()
    assert "CausalBlock" in src and "Conv1d" in src
    enc = MO.TCNEncoder()
    h = torch.randn(4, FE.HIST, 6)
    c1 = enc(h).detach()
    h2 = h.clone()
    h2[:, -1, :] += 100.0
    c2 = enc(h2).detach()
    assert c1.shape == (4, 32)
    enc.eval()
    with torch.no_grad():
        x = h.transpose(1, 2)
        y = h2.transpose(1, 2)
        p1 = enc.b1(x)
        p2 = enc.b1(y)
        assert torch.allclose(p1[:, :, :-7], p2[:, :, :-7])


def test_net_forward_finite():
    net = MO.D26Net(n_tab=10)
    pred = net(torch.randn(8, FE.HIST, 6), torch.randn(8, 10),
               torch.rand(8)).detach().numpy()
    assert pred.shape == (8,) and np.isfinite(pred).all()
    assert MO.count_params(net) > 1000


def test_no_forbidden_or_official_window_strings():
    for f in ("features.py", "model.py", "runner.py"):
        text = (PKG / f).read_text()
        for word in FORBIDDEN:
            assert word not in text, f"{f} contains forbidden {word}"
        assert "2026-06" not in text, f"{f} must not reference the official window"


def test_dev_only_discipline_markers():
    src = (PKG / "runner.py").read_text()
    for marker in ("NORMALIZE-FROM-DEV-ONLY", "EARLY-STOP-DEV-ONLY",
                   "dev-train-tail-OOF-only", "frozen dev-OOF epoch"):
        assert marker in src, f"runner missing dev-only marker: {marker}"
    assert "frozen_epochs=int(best_epoch)" in src


def test_runner_uses_strips_purge_poison_manifest():
    src = (PKG / "runner.py").read_text()
    assert "strip_features" in src and "ST.strip_columns()" in src
    assert "72h" in src and "poison.json" in src and "manifest.json" in src
    assert "--pilot" in src and "cuda" in src


def test_owned_dir_is_small():
    files = sorted(p.name for p in PKG.iterdir() if p.suffix == ".py")
    assert len(files) <= 5, files
    assert {"features.py", "model.py", "runner.py"} <= set(files)


def _load_d26_runner():
    import importlib.util
    spec = importlib.util.spec_from_file_location("d26_runner", PKG / "runner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_budget_preregistered_markers():
    import re
    src = (PKG / "runner.py").read_text()
    assert "BUDGET-DEV-OOF-CONVERGENCE" in src
    assert "learning_curves.json" in src
    assert "GPU 30min" in src
    assert "FULL-IFF" in src
    m_pilot = re.search(r"MAX_EPOCHS_PILOT\s*=\s*(\d+)", src)
    m_full = re.search(r"MAX_EPOCHS_FULL\s*=\s*(\d+)", src)
    m_pat = re.search(r"^PATIENCE\s*=\s*(\d+)", src, re.M)
    assert m_pilot and m_full and m_pat, "budget constants missing"
    assert int(m_pilot.group(1)) >= 60 and int(m_full.group(1)) >= 60
    assert int(m_pat.group(1)) >= 8
    assert "patience=PATIENCE" in src
    assert "converged" in src


def test_fit_model_returns_convergence_curves():
    mod = _load_d26_runner()
    rng = np.random.default_rng(0)
    n, n_tab = 64, 4
    tab = rng.normal(size=(n, n_tab)).astype(np.float32)
    hist = rng.normal(size=(n, FE.HIST, 6)).astype(np.float32)
    h = rng.uniform(size=n).astype(np.float32)
    y = rng.normal(size=n).astype(np.float32)
    net, best_epoch, info = mod.fit_model(tab, hist, h, y, tab, hist, h, y,
                                          max_epochs=6, patience=2, device="cpu")
    assert best_epoch >= 1
    assert {"train_loss", "val_mse", "epochs_run", "converged"} <= set(info)
    assert len(info["train_loss"]) == info["epochs_run"] >= 1
    assert len(info["val_mse"]) == info["epochs_run"]
    assert all(np.isfinite(info["train_loss"])) and all(np.isfinite(info["val_mse"]))
    pred = net(torch.from_numpy(hist), torch.from_numpy(tab),
               torch.from_numpy(h)).detach().numpy()
    assert np.isfinite(pred).all()
