"""D25 in-situ event contract: causality, E0003-independence, dev-only, guards."""
import inspect
import sys

import numpy as np
import pandas as pd
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "experiments" / "d25_insitu_event"
LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
import event as EV  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]
INDEPENDENCE_BANS = ["e0002", "strips", "suvi", "lonmap", "cnn.py"]
EXTRA_BANS = ["noaa", "gong", "donki"]


def _synth_hourly(n=240, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({
        "ace_speed_kms": np.abs(rng.normal(430, 40, n)) + 1.0,
        "ace_density_cm3": np.abs(rng.normal(6, 2, n)) + 0.1,
        "ace_temperature_k": np.abs(rng.normal(1e5, 2e4, n)) + 1.0,
        "ace_bt_nt": np.abs(rng.normal(6, 2, n)) + 0.1,
        "ace_bz_gsm_nt": rng.normal(0, 3, n),
        "ace_by_gsm_nt": rng.normal(0, 3, n),
    }, index=idx)
    return df


def test_sig_columns_fixed_new_and_prefixed():
    cols = EV.sig_columns()
    assert len(cols) == 12 and len(set(cols)) == 12
    assert all(c.startswith("sig_") for c in cols)
    for c in cols:
        assert not c.startswith(("v_", "rec", "ace_", "ch_", "enlil_"))
        assert c not in ("h", "speed_d24", "epam_p1_d6")
    # signature covers the three required families
    assert any("v_" in c for c in cols)
    assert any("np_" in c for c in cols)
    assert any("bz_" in c or "by_" in c or "bt_" in c for c in cols)


def test_features_causal_truncation_invariant():
    x = _synth_hourly()
    cut = x.index[150]
    full = EV.add_sig_features(x)
    trunc = EV.add_sig_features(x.loc[:cut])
    assert list(full.columns) == EV.sig_columns()
    assert list(trunc.columns) == EV.sig_columns()
    pre = full.index[full.index < cut - pd.Timedelta("30h")]
    for c in EV.sig_columns():
        a = full.loc[pre, c].to_numpy(float)
        b = trunc.reindex(pre)[c].to_numpy(float)
        assert np.allclose(a, b, equal_nan=True), c


def test_future_shock_moves_features_positive_control():
    x = _synth_hourly()
    y = x.copy()
    y.loc[y.index[100:120], "ace_speed_kms"] *= 1.8
    y.loc[y.index[100:120], "ace_density_cm3"] *= 3.0
    y.loc[y.index[100:120], "ace_bz_gsm_nt"] += 8.0
    a = EV.add_sig_features(x).iloc[110]
    b = EV.add_sig_features(y).iloc[110]
    assert any(abs(a[c] - b[c]) > 0 for c in EV.sig_columns()
               if np.isfinite(a[c]) and np.isfinite(b[c]))


def test_sig_features_takes_origins_only():
    assert list(inspect.signature(EV.sig_features).parameters) == ["origins"]
    src = (PKG / "event.py").read_text().lower()
    for word in ("selection", "audit", "official"):
        assert word not in src


def test_no_forbidden_or_official_window_strings():
    for f in ("event.py", "runner.py"):
        text = (PKG / f).read_text()
        assert not [p for p in FORBIDDEN if p in text], f
        assert "2026-06" not in text, f"{f} must not reference the official window"


def test_e0003_independent_axis():
    for f in ("event.py", "runner.py"):
        text = (PKG / f).read_text().lower()
        assert not [b for b in INDEPENDENCE_BANS if b in text], f
        assert not [b for b in EXTRA_BANS if b in text], f
    src = (PKG / "runner.py").read_text()
    assert "independent_of_E0003" in src
    for line in src.splitlines():
        if "E0003" in line:
            assert "independent_of_E0003" in line, line


def test_gate_dev_only_and_bounds():
    import inspect as _insp
    assert callable(EV.select_gate) and callable(EV.apply_gate)
    src = _insp.getsource(EV.select_gate)
    assert "TAU_GRID" in src or "tau" in src
    # deltas bounded 0..250, tau=1.0 never gates
    base = np.full(1000, 400.0)
    y = base.copy()
    p = np.linspace(0, 1, 1000)
    h = np.tile(np.arange(1, 73), 14)[:1000]
    g = EV.select_gate(y, base, p, h)
    assert g["tau"] == 1.0
    assert all(v == 0.0 for v in g["deltas"].values())
    out = EV.apply_gate(base, h, p, 1.0, g["deltas"])
    np.testing.assert_allclose(out, base)


def test_runner_uses_cpu_purge_poison_manifest():
    src = (PKG / "runner.py").read_text()
    assert "tree_method" in src and "hist" in src
    for gpu_pat in ('device="cuda', "device='cuda", "torch.cuda", "cupy", "EXPOS_GPUS",
                    "prefer-gpu", "cuda:0", "cuda:1"):
        assert gpu_pat not in src.lower(), gpu_pat
    assert "--pilot" in src and '"6h"' in src
    assert "72h" in src
    assert "poison.json" in src and "pred_invariant" in src
    assert "positive_control" in src
    assert "manifest.json" in src and "gate.json" in src
    assert "EV.SCRATCH" in src
    assert "from existing_all import write" in src
    assert "write(te" in src


def test_owned_dir_is_small_and_scratch_isolation():
    files = sorted(p.name for p in PKG.iterdir() if p.suffix == ".py")
    assert len(files) <= 5, files
    assert {"event.py", "runner.py"} <= set(files)
    assert str(EV.SCRATCH) == str(LB / "store" / "scratch" / "worker-d25")
    text = (PKG / "runner.py").read_text()
    assert "worker-d25" in text
    for other in ("worker-d23", "worker-d24", "worker-d26", "worker-d27", "worker-d28"):
        assert other not in text
