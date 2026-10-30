"""D12 bagged-strips contract: seeds, cuda, dev-only weights, import-only reuse.

CPU-only, synthetic numerics only. No fits, no GPU, no network, no data reads.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1] / "experiments" / "d12_bagged"
sys.path.insert(0, str(PKG))
import bag as BG  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]


def test_seeds_fixed_five():
    assert tuple(BG.SEEDS) == (0, 1, 2, 3, 4) and BG.N_SEEDS == 5
    assert BG.check_seeds([0, 1, 2, 3, 4]) == (0, 1, 2, 3, 4)
    with pytest.raises(ValueError):
        BG.check_seeds([0, 1, 2, 3])
    with pytest.raises(ValueError):
        BG.check_seeds([0, 1, 2, 3, 5])


def test_params_cuda_e0003_recipe():
    import info_ablation as ia
    for s in BG.SEEDS:
        p = BG.seed_params(s)
        assert p["device"] == "cuda" and p["random_state"] == s
        for k, v in ia.PARAMS.items():
            if k == "random_state":
                continue
            assert p[k] == v, k
    assert BG.DEVICE == "cuda"


def test_inverse_mse_weights_dev_only():
    mse = {0: 8000.0, 1: 8100.0, 2: 7900.0, 3: 8050.0, 4: 7950.0}
    w, method = BG.select_weights({"dev": mse})
    assert method == "dev-inverse-mse"
    assert abs(sum(w.values()) - 1.0) < 1e-12
    assert set(w) == set(BG.SEEDS)
    # lower MSE -> larger weight; near-equal MSE -> near-uniform
    assert w[2] > w[1] and max(w.values()) - min(w.values()) < 0.02


def test_selection_or_audit_weighting_aborts():
    mse = {s: 8000.0 + 10 * s for s in BG.SEEDS}
    with pytest.raises(ValueError):
        BG.select_weights({"dev": mse, "selection": mse})
    with pytest.raises(ValueError):
        BG.select_weights({"selection": mse})
    with pytest.raises(ValueError):
        BG.select_weights({"dev": mse, "audit": mse})
    with pytest.raises(ValueError):
        BG.select_weights({})


def test_degenerate_dev_mse_falls_back_uniform_recorded():
    bad = {s: 8000.0 for s in BG.SEEDS}
    bad[2] = float("nan")
    w, method = BG.select_weights({"dev": bad})
    assert method.startswith("uniform-fallback:")
    assert all(v == pytest.approx(0.2) for v in w.values())
    zero = {s: (0.0 if s == 1 else 8000.0) for s in BG.SEEDS}
    w2, m2 = BG.select_weights({"dev": zero})
    assert m2.startswith("uniform-fallback:")


def test_average_is_weighted_mean():
    rng = np.random.default_rng(0)
    base = rng.normal(400, 50, 100)
    preds = {s: base + rng.normal(0, 1, 100) for s in BG.SEEDS}
    w, _ = BG.select_weights({"dev": {s: 8000.0 for s in BG.SEEDS}})
    out = BG.average(preds, w)
    assert out.shape == (100,) and np.isfinite(out).all()
    assert np.allclose(out, sum(preds[s] * w[s] for s in BG.SEEDS))
    with pytest.raises(ValueError):
        BG.average({s: preds[s] for s in BG.SEEDS[:4]}, w)


def test_dev_mse_of_shapes():
    y = np.arange(10, dtype=float)
    preds = {s: y + s for s in BG.SEEDS}
    mse = BG.dev_mse_of(y, preds)
    assert mse[0] == pytest.approx(0.0) and mse[2] == pytest.approx(4.0)
    with pytest.raises(ValueError):
        BG.dev_mse_of(y, {s: np.zeros(9) for s in BG.SEEDS})


def test_no_forbidden_or_window_dates_in_new_code():
    for name in ("bag.py", "runner.py"):
        text = (PKG / name).read_text()
        assert not [p for p in FORBIDDEN if p in text], name
        assert "2026-06" not in text, f"{name} must not reference the window after 2026-05"
        assert "2026-05-29" not in text and "2026-05-31" not in text, name


def test_runner_import_only_and_markers():
    src = (PKG / "runner.py").read_text()
    # E0003 reuse is import-only: no copied strip geometry or grid builders
    assert "import strips as ST" in src and "from existing_all import" in src
    assert "import bag as BG" in src
    for token in ("OMEGA", "PARKER_WEST", "strip_weights", "def lam(", "def full_grid",
                  "def speed_features", "def enlil_features"):
        assert token not in src, f"runner must not copy E0003 code: {token}"
    assert "ST.strip_features" in src and "ST.strip_columns()" in src
    assert "ST.poison_test" in src
    # cuda + pilot + purge + outputs + dev-only markers
    assert '"cuda"' in src or "'cuda'" in src or "device" in src
    assert "--pilot" in src and '"6h"' in src and '"3h"' in src
    assert "72h" in src
    assert "poison.json" in src and "manifest.json" in src and "selection.json" in src
    assert "predictions_{dev,selection,audit}" in src or "for name in PERIODS" in src
    assert "select_weights" in src and '{"dev": dev_mse}' in src
    low = src.lower()
    assert 'scores["selection"]' not in low and "scores['selection']" not in low
    assert "selection" in low  # period loop necessarily names it; weights must not use it


def test_owned_dir_is_small():
    files = sorted(p.name for p in PKG.iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"bag.py", "runner.py"} <= set(files)
