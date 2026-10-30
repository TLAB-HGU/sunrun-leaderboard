"""Contract for D53 residual MLP on XGBoost predictions (experiments/d53_residnn/)."""
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

D53 = Path(__file__).resolve().parents[1] / "experiments" / "d53_residnn"
sys.path.insert(0, str(D53))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "headroom_audit"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "expos_runners"))

import features as FE  # noqa: E402
import model as MO  # noqa: E402
import spec as SP  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]


def _src(name):
    return (D53 / name).read_text()


def test_hypothesis_dual_tag_leading_and_citations():
    SP.check_tags()
    assert SP.FAMILY_TAG == "[family=nn]"
    assert SP.AXES_TAG == "[axes: model=residual-nn]"
    head = SP.HYPOTHESIS.splitlines()[0]
    assert head.startswith("[family=nn]") and "[axes: model=residual-nn]" in SP.HYPOTHESIS
    for cite in ("E0121", "X0110"):
        assert cite in SP.HYPOTHESIS, f"missing citation {cite}"
    assert "72" in SP.HYPOTHESIS and "6h" in SP.HYPOTHESIS


def test_full_rule_preregistered_and_scratch_owned():
    assert "poison" in SP.FULL_RULE and "1/3" in SP.FULL_RULE and "exp-add" in SP.FULL_RULE
    assert "converged" in SP.FULL_RULE and "rc 0" in SP.FULL_RULE
    assert "curves" in SP.FULL_RULE and "budget" in SP.FULL_RULE
    assert str(SP.SCRATCH).endswith("store/scratch/worker-d53")
    for f in ("spec.py", "features.py", "model.py", "runner.py"):
        assert "worker-d52" not in _src(f) and "worker-d55" not in _src(f)
        assert "worker-d56" not in _src(f) and "worker-d57" not in _src(f)


def test_no_forbidden_or_window_refs():
    for f in ("spec.py", "features.py", "model.py", "runner.py"):
        text = _src(f)
        for s in FORBIDDEN:
            assert s not in text, f"{f} must not reference {s}"
        assert "2026-06" not in text, f"{f} must not reference the June window"
        assert "official" not in text.lower(), f"{f} must not write frozen-window predictions"


def test_probe_dates_only_in_probes():
    assert set(SP.PROBES) == {"dev", "selection", "audit"}
    for f in ("features.py", "model.py"):
        assert "2025-12-04" not in _src(f) and "2026-03-04" not in _src(f)
    n_dev = sum(_src("runner.py").count(s) for s in ("2025-09-15", "2026-01-05", "2026-04-06"))
    assert n_dev == 0, "probe dates must live in spec.PROBES only"


def test_origins_only_signatures():
    assert list(inspect.signature(FE.feature_rows).parameters)[:1] == ["origins"]


def test_zero_init_preserves_base_and_forward_shape():
    n_in = 91
    net = MO.ResidMLP(n_in=n_in)
    assert net.out.out_features == 1
    assert torch.equal(net.out.weight, torch.zeros_like(net.out.weight))
    assert torch.equal(net.out.bias, torch.zeros_like(net.out.bias))
    x = torch.randn(8, n_in)
    out = net(x)
    assert out.shape == (8,) and torch.equal(out, torch.zeros(8)), \
        "zero-init correction must be exactly 0 so final == base at init"
    assert MO.count_params(net) > 10000, "residual head must be a real backprop net"


def test_feature_rows_alone_vs_batched_identical():
    o1 = pd.DatetimeIndex(["2025-09-15T00:00:00Z"])
    o2 = pd.DatetimeIndex(["2025-09-15T00:00:00Z", "2025-09-16T00:00:00Z"])
    alone = FE.feature_rows(o1)
    batched = FE.feature_rows(o2)
    shared = [c for c in alone.columns if c in batched.columns]
    a = batched[batched["origin"] == o1[0]].reset_index(drop=True)
    assert len(a) == 72 and len(alone) == 72
    for c in shared:
        assert np.allclose(a[c].to_numpy(dtype=np.float64),
                           alone[c].to_numpy(dtype=np.float64), equal_nan=True), f"col {c} unstable"


def test_budget_and_dev_only_markers():
    src = _src("runner.py")
    assert "BUDGET-DEV-OOF-CONVERGENCE" in src
    assert "NORMALIZE-FROM-DEV-ONLY" in src and "EARLY-STOP-DEV-ONLY" in src
    assert "frozen_epochs=int(best_epoch)" in src
    assert "learning_curves.json" in src and "converged" in src
    assert "MAX_TRAIN_S_PILOT" in src or "30min" in src
    assert "--pilot" in src and "cuda" in src
    assert "poison.json" in src and '"passed": true' not in src.lower()
    assert "positive_control" in src and "bypassed" in src
    assert 3 * SP.XGB_NJOBS <= 20


def test_predictions_three_kinds():
    src = _src("runner.py")
    assert "write(te, final, name)" in src
    for name in ("dev", "selection", "audit"):
        assert name in src


def test_owned_dir_is_small():
    files = sorted(p.name for p in D53.iterdir() if p.suffix == ".py")
    assert len(files) <= 6, files
    assert {"spec.py", "features.py", "model.py", "runner.py"} <= set(files)


def _load_d53_runner():
    import importlib.util
    loc = str(D53 / "runner.py")
    for name, mod in list(sys.modules.items()):
        if getattr(mod, "__file__", None) == loc:
            return mod
    spec = importlib.util.spec_from_file_location("d53_runner", loc)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["d53_runner"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_train_rows_match_purge_and_drop_missing_labels():
    mod = _load_d53_runner()
    origins = pd.date_range("2025-07-20", periods=4, freq="6h", tz="UTC")
    df = pd.DataFrame({"origin": np.repeat(origins, 72), "h": np.tile(np.arange(1, 73), 4)})
    df["target_t"] = df["origin"] + pd.to_timedelta(df["h"], unit="h")
    df["y"] = 500.0
    m = mod.train_rows(df, pd.Timestamp("2025-08-04T00:00:00Z"))
    assert len(m) == 4 * 72
    assert (m["target_t"] < pd.Timestamp("2025-08-01T00:00:00Z", tz="UTC")).all()
    df2 = df.copy()
    df2.loc[df2.index[:10], "y"] = np.nan
    m2 = mod.train_rows(df2, pd.Timestamp("2025-08-04T00:00:00Z"))
    assert len(m2) == len(m) - 10


def test_oof_base_honest_and_finite():
    mod = _load_d53_runner()
    rng = np.random.default_rng(0)
    origins = pd.date_range("2025-07-01", periods=40, freq="6h", tz="UTC")
    n = len(origins) * 4
    tr = pd.DataFrame({"origin": np.repeat(origins, 4),
                       "f1": rng.normal(0, 1, n).astype(np.float32),
                       "f2": rng.normal(400, 50, n).astype(np.float32)})
    tr["y"] = (2 * tr["f1"] + 0.01 * tr["f2"] + rng.normal(0, 1, n)).astype(np.float32)
    params = {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1,
              "tree_method": "hist", "n_jobs": 1, "random_state": 0}
    base = mod.oof_base(tr, ["f1", "f2"], params, n_est=5)
    assert base.shape == (n,) and np.isfinite(base).all()
    assert float(np.mean((base - tr["y"].to_numpy()) ** 2)) > 0


def test_feature_rows_match_ia_build():
    import info_ablation as ia  # noqa: E402
    df = ia.build(freq="6h", start="2025-08-04", end="2025-08-04 12:00")
    origins = pd.DatetimeIndex(df["origin"].unique()).sort_values()
    got = FE.feature_rows(origins)
    ref = df.set_index(["origin", "h"]).sort_index()
    shared = [c for c in got.columns if c in ref.columns and c not in ("origin", "h")]
    assert len(shared) > 50
    g = got.set_index(["origin", "h"]).sort_index()
    common_idx = ref.index.intersection(g.index)
    assert len(common_idx) > 100
    for c in shared:
        a = g.loc[common_idx, c].to_numpy(dtype=np.float64)
        b = ref.loc[common_idx, c].to_numpy(dtype=np.float64)
        assert np.allclose(a, b, equal_nan=True), f"feature col {c} drifts from ia.build"


def test_existing_all_cols_cover_e0001_set():
    import info_ablation as ia  # noqa: E402
    cols = FE.existing_all_cols(
        pd.Index(["origin", "h", "y", "v_lag0", "rec26", "ace_speed_kms_last",
                  "ch_area_total", "enlil_next_arr_h", "speed_d24", "epam_p1_d6"]))
    assert "h" in cols and "v_lag0" in cols and "ace_speed_kms_last" in cols
    assert "enlil_next_arr_h" not in cols, "existing_all excludes ENLIL (E0001 parity)"
