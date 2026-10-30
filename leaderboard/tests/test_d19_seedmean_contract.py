"""D19 seed-mean contract: fresh seeds, E0003-recipe CPU params, import-only.

CPU-only, synthetic numerics only. No fits, no GPU, no network, no data reads.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

LB = Path(__file__).resolve().parents[1]
PKG = LB / "experiments" / "d19_seedmean"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
import seedmean as SM  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]


def test_seeds_fresh_fixed_and_disjoint():
    assert tuple(SM.SEEDS) == (2, 3)
    assert SM.N_SEEDS == 2
    assert SM.E0003_SEED == 0 and SM.E0041_SEED == 1
    assert set(SM.SEEDS).isdisjoint({SM.E0003_SEED, SM.E0041_SEED})
    assert SM.check_seeds([2, 3]) == (2, 3)
    with pytest.raises(ValueError):
        SM.check_seeds([0, 1])
    with pytest.raises(ValueError):
        SM.check_seeds([2])
    with pytest.raises(ValueError):
        SM.check_seeds([2, 3, 4])
    with pytest.raises(ValueError):
        SM.seed_params(0)
    with pytest.raises(ValueError):
        SM.seed_params(1)


def test_seed_params_e0003_recipe_cpu_only():
    import info_ablation as ia
    assert ia.PARAMS["random_state"] == 0
    for s in SM.SEEDS:
        p = SM.seed_params(s)
        assert p["random_state"] == s
        assert "device" not in p and "cuda" not in str(p.values())
        assert p["tree_method"] == "hist"
        for k, v in ia.PARAMS.items():
            if k == "random_state":
                continue
            assert p[k] == v, k


def test_fixed_uniform_mean_numerics():
    rng = np.random.default_rng(11)
    a = rng.normal(400, 50, 240)
    b = rng.normal(400, 50, 240)
    out = SM.mean_predictions({2: a, 3: b})
    assert out.shape == (240,) and np.isfinite(out).all()
    assert np.allclose(out, 0.5 * a + 0.5 * b)
    assert SM.WEIGHTS == {2: 0.5, 3: 0.5}
    with pytest.raises(ValueError):
        SM.mean_predictions({2: a})
    with pytest.raises(ValueError):
        SM.mean_predictions({0: a, 1: b})
    with pytest.raises(ValueError):
        SM.mean_predictions({2: a, 3: b[:100]})


def test_no_forbidden_or_external_recollection():
    for name in ("seedmean.py", "runner.py"):
        text = (PKG / name).read_text()
        assert not [p for p in FORBIDDEN if p in text], name
        for token in ("NOAA", "GONG", "DONKI", "noaa-goes18.s3", "s3.amazonaws"):
            assert token not in text, f"{name}:{token}"
        assert "2026-06" not in text, name
        assert "2026-05-29" not in text and "2026-05-31" not in text, name


def test_no_e0003_e0041_prediction_reuse():
    text = (PKG / "runner.py").read_text()
    assert "runs/E00" not in text and "research/expos/runs" not in text
    low = text.lower()
    assert "shutil" not in low
    assert "read_parquet" not in low  # fresh refits only; predictions are written, never read
    assert "E0041" not in text  # no repro-file provenance needed; seeds + rule are self-contained


def test_runner_import_only_and_causal_markers():
    src = (PKG / "runner.py").read_text()
    assert "import strips as ST" in src and "from existing_all import" in src
    assert "import seedmean as SM" in src
    for token in ("OMEGA", "PARKER_WEST", "strip_weights", "def lam(", "def full_grid",
                  "def speed_features", "def enlil_features", "def select_slots",
                  "def strip_features"):
        assert token not in src, f"runner must not copy E0003 code: {token}"
    assert "ST.strip_features" in src and "ST.strip_columns()" in src
    assert "ST.poison_test" in src and "ST.build_manifest" in src
    assert "SM.seed_params" in src and "SM.mean_predictions" in src and "SM.check_seeds" in src
    assert "xgb.XGBRegressor" in src
    assert "--pilot" in src and '"6h"' in src and '"3h"' in src
    assert "72h" in src
    assert "poison.json" in src and '"passed"' in src
    assert "availability_manifest.json" in src and "selection.json" in src
    assert "for name, (lo, hi) in PERIODS" in src
    assert "EXPOS_OUT" in src


def test_owned_dir_is_small_and_scratch_isolated():
    files = sorted(p.name for p in PKG.iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"seedmean.py", "runner.py"} <= set(files)
    for name in ("seedmean.py", "runner.py"):
        text = (PKG / name).read_text()
        assert "store/scratch/worker-d" not in text  # runner writes only $EXPOS_OUT
        assert "predictions_official" not in text
