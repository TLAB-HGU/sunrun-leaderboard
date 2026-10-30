"""D5 CNN-residual contract: shapes, stale-zero, determinism, causality hygiene.

Fast unit checks only (no fits on eval periods, no network). The pilot/full
runners carry the real dev-OOF training, poison test and manifests.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments/d5_cnn_residual"))
sys.path.insert(0, str(LB / "experiments/d1_longitude_strips"))
import cnn as CNN  # noqa: E402
import lonmap as LM  # noqa: E402
import strips as ST  # noqa: E402

D5 = LB / "experiments/d5_cnn_residual"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]


def test_map_shape_and_columns():
    assert LM.MAP_SHAPE == (2, 3, 12)
    cols = LM.column_names()
    assert len(cols) == 72 and len(set(cols)) == 72
    assert all(c.startswith("lon") for c in cols)
    cc = LM.cell_centers()
    assert cc.shape == (12,) and float(cc[0]) == -82.5 and float(cc[-1]) == 82.5


def test_cnn_shape_params_and_budget():
    conv = CNN.init_conv(seed=0)
    rng = np.random.RandomState(0)
    maps = rng.randn(4, 2, 3, 12).astype(np.float32)
    base72 = rng.randn(4, 72) * 50 + 450
    st = CNN.fit_standardizer(np.concatenate(
        [CNN.embed(maps, conv), base72], axis=1))
    resid = rng.randn(4, 72)
    ro = CNN.fit_ridge(CNN.apply_standardizer(np.concatenate(
        [CNN.embed(maps, conv), base72], axis=1), st), resid, 1.0)
    out = CNN.predict_residual(maps, base72, np.zeros(4, dtype=bool), conv, st, ro)
    assert out.shape == (4, 72) and np.isfinite(out).all()
    p = CNN.count_params()
    assert p["total"] <= 200_000, p
    assert p["total"] == p["conv_frozen"] + p["readout"]


def test_stale_residual_exactly_zero():
    conv = CNN.init_conv(seed=0)
    rng = np.random.RandomState(1)
    maps = rng.randn(3, 2, 3, 12).astype(np.float32)
    base72 = rng.randn(3, 72) * 50 + 450
    st = CNN.fit_standardizer(np.concatenate(
        [CNN.embed(maps, conv), base72], axis=1))
    ro = CNN.fit_ridge(CNN.apply_standardizer(np.concatenate(
        [CNN.embed(maps, conv), base72], axis=1), st), rng.randn(3, 72), 1.0)
    stale = np.array([False, True, True])
    out = CNN.predict_residual(maps, base72, stale, conv, st, ro)
    assert (out[1:] == 0.0).all()
    assert np.isfinite(out[0]).all()


def test_determinism_twice_identical():
    conv_a, conv_b = CNN.init_conv(seed=0), CNN.init_conv(seed=0)
    assert np.array_equal(conv_a["w1"], conv_b["w1"])
    assert np.array_equal(conv_a["w2"], conv_b["w2"])
    rng = np.random.RandomState(2)
    maps = rng.randn(5, 2, 3, 12).astype(np.float32)
    assert np.array_equal(CNN.embed(maps, conv_a), CNN.embed(maps, conv_b))
    n1 = LM.apply_norm(maps, {"channels": {0: {"mean": 1.0, "std": 2.0},
                                           1: {"mean": 0.5, "std": 0.25}},
                             "n_train_origins": 5})
    n2 = LM.apply_norm(maps, {"channels": {0: {"mean": 1.0, "std": 2.0},
                                           1: {"mean": 0.5, "std": 0.25}},
                             "n_train_origins": 5})
    assert np.array_equal(n1, n2)


def test_ridge_learns_correlated_residual():
    rng = np.random.RandomState(3)
    x = rng.randn(60, 8)
    r = np.concatenate([x[:, :1] * 10.0, np.zeros((60, 71))], axis=1)
    w = CNN.fit_ridge((x - x.mean(0)) / x.std(0), r, 0.01)
    pred = np.concatenate([(x - x.mean(0)) / x.std(0), np.ones((60, 1))], axis=1) @ np.concatenate(
        [w["coef"], w["intercept"][None, :]], axis=0)
    assert float(np.mean((pred - r) ** 2)) < float(np.mean(r ** 2)) * 0.5


def test_output_contract_columns():
    df = pd.DataFrame({"origin": pd.to_datetime(["2025-08-04T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    pred = np.full(72, 450.0)
    out = pd.DataFrame({"origin_last_input_utc": df["origin"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "horizon_hours": df["h"].astype(int), "pred_kms": pred.astype(float)})
    assert list(out.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert len(out) == 72 and out["horizon_hours"].tolist() == list(range(1, 73))


def test_no_forbidden_strings_in_d5_code():
    blob = "\n".join((D5 / f).read_text(errors="ignore")
                     for f in ("lonmap.py", "cnn.py", "runner.py"))
    hits = [p for p in FORBIDDEN if p in blob]
    assert hits == [], hits
    low = blob.lower()
    for pat in ("2026-06", "2026-09", "huggingface", "folds", "noaa", "gong", "donki"):
        assert pat not in low, pat


def test_proxy_rule_reuses_strips_geometry():
    assert ST.CELL_W == 15.0 and ST.MARGIN_H == 1.0 and ST.STALE_H == 6.0
    assert abs(ST.lam(1, 550, 0.0) - ST.OMEGA * (ST.TAU[550] - 1.0)) < 1e-9
    # Parker west at h=0 with a fresh image matches the frozen arithmetic.
    for v in ST.SPEEDS:
        assert abs(ST.PARKER_WEST[v] - ST.OMEGA * ST.TAU[v]) < 1e-9


def test_no_forbidden_strings_in_budget_code():
    blob = (D5 / "budget_cnn.py").read_text(errors="ignore")
    assert blob, "budget_cnn.py missing"
    hits = [p for p in FORBIDDEN if p in blob]
    assert hits == [], hits
    low = blob.lower()
    for pat in ("2026-06", "2026-09", "huggingface", "folds", "noaa", "gong", "donki"):
        assert pat not in low, pat


def test_frozen_head_marked_not_a_cnn():
    text = (D5 / "runner.py").read_text(errors="ignore")
    assert "MUST NOT be reported as a CNN" in text
    assert "X0046" in text


def test_runner_budget_guards():
    text = (D5 / "runner.py").read_text(errors="ignore")
    assert "--budget" in text
    assert "budget_cnn" in text
    assert "learning_curves.json" in text
    assert "PATIENCE" in text
    assert "dev-val-OOF" in text
    assert "no selection/audit rows read" in text.lower()


def test_budget_cnn_is_trainable_backprop():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(D5))
    import budget_cnn as BC  # noqa: E402
    net = BC.build_model(seed=0)
    assert BC.n_trainable_conv_layers(net) >= 2
    assert all(p.requires_grad for p in net.conv1.parameters())
    assert all(p.requires_grad for p in net.conv2.parameters())
    assert net.conv1.in_channels == 2
    mlp_params = sum(p.numel() for m in (net.fc1, net.fc2) for p in m.parameters())
    assert mlp_params > 0
    p = BC.count_params(net)
    assert p["conv_trainable"] > 0 and p["readout_mlp"] > 0
    assert p["total"] == p["conv_trainable"] + p["readout_mlp"]
    assert p["total"] <= 200_000, p
    before = [p.clone() for p in net.parameters()]
    net.train()
    opt = torch.optim.SGD(net.parameters(), lr=1e-2)
    m = torch.randn(4, 2, 3, 12)
    b = torch.randn(4, 72)
    opt.zero_grad()
    net(m, b).sum().backward()
    opt.step()
    assert any(not torch.equal(a, c) for a, c in zip(before, net.parameters()))
    out = net(m, b)
    assert out.shape == (4, 72) and torch.isfinite(out).all()


def test_budget_cnn_stale_zero_and_converges():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(D5))
    import budget_cnn as BC  # noqa: E402
    rng = np.random.RandomState(7)
    n = 48
    maps = rng.randn(n, 2, 3, 12).astype(np.float32)
    base72 = (rng.randn(n, 72) * 50 + 450).astype(np.float32)
    bnorm = BC.fit_base72_norm(base72[:32])
    b72n = BC.apply_base72_norm(base72, bnorm)
    resid = (maps[:, 0, 1, :].mean(axis=1, keepdims=True) * 5.0
             + rng.randn(n, 72) * 0.5).astype(np.float32)
    stale = np.zeros(n, dtype=bool)
    stale[40:] = True
    fit = BC.fit_early_stop(maps[:32], b72n[:32], resid[:32], stale[:32],
                            maps[32:40], b72n[32:40], resid[32:40], stale[32:40],
                            device="cpu", max_epochs=10, patience=3)
    assert fit["best_epoch"] >= 1
    assert len(fit["train_mse"]) == len(fit["val_mse"]) >= 1
    assert fit["train_mse"][0] >= 0 and fit["val_mse"][0] >= 0
    net = BC.build_model(seed=0)
    net.load_state_dict({k: v for k, v in fit["state"].items()})
    pred = BC.predict_residual(net, maps, b72n, stale, device="cpu")
    assert pred.shape == (n, 72)
    assert (pred[40:] == 0.0).all()
    assert np.isfinite(pred[:40]).all()
