"""D27 dimming-gate contract: gate-only dimming, E0003 lines, causality guards."""
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments/d1_longitude_strips"))
sys.path.insert(0, str(LB / "experiments/d24_dimming"))
sys.path.insert(0, str(LB / "experiments/d27_dimgate"))
sys.path.insert(0, str(LB / "experiments/headroom_audit"))
import strips as ST  # noqa: E402
import dimming as DM  # noqa: E402
import gate as GT  # noqa: E402

PKG = LB / "experiments" / "d27_dimgate"
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


def _dim_long(n_org=3, seed=0):
    rng = np.random.default_rng(seed)
    origins = pd.date_range("2025-08-04", periods=n_org, freq="6h", tz="UTC")
    rows = []
    for o in origins:
        for h in range(1, 73):
            row = {"origin": o, "h": h}
            for v in (350, 450, 550, 650, 750):
                row[f"dim_drop_v{v}"] = float(max(0.0, rng.normal(0.1, 0.05)))
                row[f"dim_area_v{v}"] = float(max(0.0, rng.normal(0.1, 0.05)))
                row[f"flare_up_v{v}"] = float(max(0.0, rng.normal(0.05, 0.03)))
                row[f"flare_q10_v{v}"] = float(max(0.0, rng.normal(0.05, 0.03)))
            row["dim_dt_h"] = 1.0
            row["img_age_h"] = 2.0
            row["img_stale"] = 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def test_gate_columns_43():
    cols = GT.gate_feature_columns()
    assert len(cols) == 43
    assert cols[-3:] == ["dim_dt_h", "img_age_h", "img_stale"]
    assert "dim_drop_v350_mean" in cols and "flare_q10_v750_max" in cols


def test_dim_origin_features_schema():
    df = GT.dim_origin_features(_dim_long())
    assert list(df["h"]) if "h" in df else True
    assert len(df) == 3
    assert list(df.columns) == ["origin"] + GT.gate_feature_columns()
    assert np.isfinite(df["dim_drop_v350_mean"].to_numpy()).all()


def test_dim_origin_features_stale_nan():
    df = _dim_long(n_org=1)
    for c in DM.dimming_columns():
        if c not in ("dim_dt_h", "img_age_h", "img_stale"):
            df[c] = np.nan
    df["img_stale"] = 1.0
    out = GT.dim_origin_features(df)
    assert out["img_stale"].iloc[0] == 1.0
    assert out["dim_drop_v350_mean"].isna().all()


def test_label_boundary_150():
    df = _long()
    lab = GT.origin_labels(df)
    assert lab.sum() >= 1
    one = df[df["origin"] == df["origin"].iloc[0]]
    assert bool(lab.loc[one["origin"].iloc[0]]) == bool((one["y"] - one["v_lag0"]).max() >= 150.0)


def test_band_index_edges():
    assert GT.band_index([1, 6, 7, 24, 25, 48, 49, 72]).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_select_gate_takes_oof_arrays_only():
    params = list(inspect.signature(GT.select_gate).parameters)
    assert params == ["y", "base", "p", "h"]
    src = inspect.getsource(GT.select_gate)
    for word in ("selection", "audit", "official", "folds"):
        assert word not in src


def test_select_gate_improves_and_clips():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.normal(400, 30, n)
    base = y - 60.0
    p = np.where(np.arange(n) % 4 == 0, 0.9, 0.1)
    h = np.tile(np.arange(1, 73), n // 72 + 1)[:n]
    gate = GT.select_gate(y, base, p, h)
    assert gate["tau"] in GT.TAU_GRID and gate["tau"] < 1.0
    assert gate["dev_mse_gated"] < gate["dev_mse_base"]
    for d in gate["deltas"].values():
        assert 0.0 <= d <= 250.0
    assert len(gate["table"]) == len(GT.TAU_GRID)


def test_select_gate_falls_back_to_never_gate():
    n = 2000
    y = np.arange(n, dtype=float)
    base = y.copy()
    p = np.linspace(0.01, 0.99, n)
    h = np.tile(np.arange(1, 73), n // 72 + 1)[:n]
    gate = GT.select_gate(y, base, p, h)
    assert gate["tau"] == 1.0 and gate["dev_mse_gated"] == gate["dev_mse_base"]
    assert all(v == 0.0 for v in gate["deltas"].values())


def test_apply_gate_math_and_tau_frozen():
    base = np.array([400.0, 400.0, 400.0])
    h = np.array([3, 10, 60])
    p = np.array([0.9, 0.9, 0.1])
    deltas = {"1-6": 50.0, "7-24": 0.0, "25-48": 80.0, "49-72": 100.0}
    out = GT.apply_gate(base, h, p, 0.5, deltas)
    assert out.tolist() == [450.0, 400.0, 400.0]
    assert GT.apply_gate(base, h, p, 1.0, deltas).tolist() == base.tolist()


def test_train_classifier_needs_both_classes():
    try:
        GT.train_classifier(pd.DataFrame({"x": [1.0, 2.0]}), np.array([0, 0]),
                            params=dict(n_estimators=2, random_state=0))
    except ValueError:
        pass
    else:
        raise AssertionError("single-class fit must fail closed")


def test_dimming_import_only_no_geometry_redefine():
    for f in ("gate.py", "runner.py"):
        src = (PKG / f).read_text()
        assert "import dimming" in src or "from dimming" in src or "import DM" in src or f == "gate.py", \
            f"{f} must import d24 dimming"
    rsrc = (PKG / "runner.py").read_text()
    assert "OMEGA = " not in rsrc and "TAU = " not in rsrc
    gsrc = (PKG / "gate.py").read_text()
    assert "OMEGA = " not in gsrc and "TAU = " not in gsrc
    assert GT.SCRATCH == LB / "store" / "scratch" / "worker-d27"


def test_level_is_e0003_lines_dimming_never_in_regressor():
    src = (PKG / "runner.py").read_text()
    assert "existing_all" in src and "strip_columns" in src
    assert "dimming must never enter the direct head" in src
    assert "DM.build_features" in src
    assert "GT.dim_origin_features" in src or "dim_origin_features" in src
    assert "GT.select_gate" in src or "select_gate" in src


def test_no_forbidden_strings_in_owned_code():
    for f in ("gate.py", "runner.py"):
        text = (PKG / f).read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f} contains forbidden {s!r}"
        assert "2026-06" not in text, f"{f} must not reference the official window"


def test_runner_contract_cpu_no_gpu_no_forbidden():
    text = (PKG / "runner.py").read_text()
    low = text.lower()
    assert "tree_method" in text and "hist" in text
    for gpu_pat in ('device="cuda', "device='cuda", "torch.cuda", "cupy", "expos_gpus",
                    "prefer-gpu", "cuda:0", "cuda:1"):
        assert gpu_pat not in low, gpu_pat
    assert "--gpu" not in text
    for bad in ("truth.parquet", "folds.parquet", "naive.parquet", "baseline-replay",
                "official-predictions", "official_check", "noaa", "gong", "donki"):
        assert bad not in low, bad
    for need in ("predictions_", "poison.json", "gate.json", "availability_manifest.json",
                 "--pilot", "72h", "ia.build", "GT.SCRATCH"):
        assert need in text, need
    assert "write(te" in text
    import info_ablation as ia
    assert ia.PARAMS["n_jobs"] <= 8
    assert ia.PARAMS["tree_method"] == "hist"
    assert GT.CLF_PARAMS["tree_method"] == "hist"
    assert int(GT.CLF_PARAMS.get("n_jobs", 0)) <= 8


def test_scratch_isolation():
    assert str(GT.SCRATCH) == str(LB / "store" / "scratch" / "worker-d27")
    text = (PKG / "runner.py").read_text()
    assert "GT.SCRATCH" in text
    assert "worker-d24" not in text
    assert "worker-d28" not in text
    assert "worker-d25" not in text


def test_owned_dir_is_small():
    files = sorted(p.name for p in PKG.iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"gate.py", "runner.py"} <= set(files)
