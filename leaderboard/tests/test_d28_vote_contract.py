"""D28 seed-vote contract: fresh seeds, E0003-recipe CPU params, surge-majority vote.

CPU-only, synthetic numerics only. No fits, no GPU, no network, no data reads.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

LB = Path(__file__).resolve().parents[1]
PKG = LB / "experiments" / "d28_seedvote"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
import vote as VM  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]


def _preds(n=120, seed=0, loc=400.0, scale=50.0):
    rng = np.random.default_rng(seed)
    return rng.normal(loc, scale, n)


def test_seeds_fresh_fixed_and_disjoint():
    assert tuple(VM.SEEDS) == (10, 11, 12, 13, 14)
    assert VM.N_SEEDS == 5
    assert VM.MAJORITY == 3
    assert VM.SURGE_THRESHOLD == 600.0
    assert tuple(VM.EARLY_SEEDS) == (0, 1, 2, 3, 4)
    assert set(VM.SEEDS).isdisjoint(set(VM.EARLY_SEEDS))
    assert VM.check_seeds([10, 11, 12, 13, 14]) == (10, 11, 12, 13, 14)
    with pytest.raises(ValueError):
        VM.check_seeds([0, 1, 2, 3, 4])
    with pytest.raises(ValueError):
        VM.check_seeds([10, 11, 12, 13])
    with pytest.raises(ValueError):
        VM.check_seeds([10, 11, 12, 13, 14, 15])
    with pytest.raises(ValueError):
        VM.seed_params(0)
    with pytest.raises(ValueError):
        VM.seed_params(4)


def test_seed_params_e0003_recipe_cpu_only():
    import info_ablation as ia
    assert ia.PARAMS["random_state"] == 0
    for s in VM.SEEDS:
        p = VM.seed_params(s)
        assert p["random_state"] == s
        assert "device" not in p and "cuda" not in str(p.values())
        assert p["tree_method"] == "hist"
        for k, v in ia.PARAMS.items():
            if k == "random_state":
                continue
            assert p[k] == v, k


def test_unanimous_vote_is_global_mean():
    n = 96
    base = {s: _preds(n, seed=s, loc=400.0) for s in VM.SEEDS}
    out = VM.voted_predictions(base)
    assert out.shape == (n,) and np.isfinite(out).all()
    assert np.allclose(out, np.mean(np.stack([base[s] for s in VM.SEEDS]), axis=0))
    hot = {s: _preds(n, seed=100 + s, loc=700.0, scale=5.0) for s in VM.SEEDS}
    out_hot = VM.voted_predictions(hot)
    assert np.allclose(out_hot, np.mean(np.stack([hot[s] for s in VM.SEEDS]), axis=0))


def test_surge_majority_takes_surge_side_mean():
    n = 64
    surge = [np.full(n, 700.0), np.full(n, 710.0), np.full(n, 720.0)]
    calm = [np.full(n, 400.0), np.full(n, 410.0)]
    pred = {s: v for s, v in zip(VM.SEEDS, surge + calm)}
    out = VM.voted_predictions(pred)
    assert np.allclose(out, np.full(n, (700.0 + 710.0 + 720.0) / 3))
    assert VM.majority_surge_mask(pred).all()
    assert VM.surge_votes(pred).tolist() == [3] * n


def test_calm_majority_takes_calm_side_mean():
    n = 64
    surge = [np.full(n, 700.0), np.full(n, 710.0)]
    calm = [np.full(n, 400.0), np.full(n, 410.0), np.full(n, 420.0)]
    pred = {s: v for s, v in zip(VM.SEEDS, surge + calm)}
    out = VM.voted_predictions(pred)
    assert np.allclose(out, np.full(n, (400.0 + 410.0 + 420.0) / 3))
    assert not VM.majority_surge_mask(pred).any()
    stats = VM.vote_stats(pred)
    assert stats["surge_majority_frac"] == pytest.approx(0.0)
    assert stats["split_3_2_frac"] == pytest.approx(1.0)


def test_threshold_boundary_counts_as_surge_and_single_outlier_loses():
    n = 32
    at = [np.full(n, 600.0), np.full(n, 600.0), np.full(n, 599.9)]
    calm = [np.full(n, 400.0), np.full(n, 410.0)]
    pred = {s: v for s, v in zip(VM.SEEDS, at + calm)}
    assert VM.surge_votes(pred).tolist() == [2] * n
    assert not VM.majority_surge_mask(pred).any()
    out = VM.voted_predictions(pred)
    assert np.allclose(out, np.full(n, (599.9 + 400.0 + 410.0) / 3))
    spike = {s: np.full(n, 400.0) for s in VM.SEEDS}
    spike[VM.SEEDS[0]] = np.full(n, 900.0)
    out2 = VM.voted_predictions(spike)
    assert np.allclose(out2, np.full(n, 400.0))


def test_vote_rejects_bad_inputs():
    a = _preds(50, seed=1)
    b = _preds(50, seed=2)
    with pytest.raises(ValueError):
        VM.voted_predictions({10: a, 11: b})
    with pytest.raises(ValueError):
        VM.voted_predictions({s: _preds(50, seed=s) for s in (0, 1, 2, 3, 4)})
    bad = {s: _preds(50, seed=s) for s in VM.SEEDS}
    bad[VM.SEEDS[0]] = bad[VM.SEEDS[0]][:10]
    with pytest.raises(ValueError):
        VM.voted_predictions(bad)
    with pytest.raises(ValueError):
        VM.surge_votes({s: _preds(40, seed=s) for s in (10, 11, 12, 13)})


def test_no_forbidden_or_external_recollection():
    for name in ("vote.py", "runner.py"):
        text = (PKG / name).read_text()
        assert not [p for p in FORBIDDEN if p in text], name
        for token in ("NOAA", "GONG", "DONKI", "noaa-goes18.s3", "s3.amazonaws"):
            assert token not in text, f"{name}:{token}"
        assert "2026-06" not in text, name
        assert "2026-05-29" not in text and "2026-05-31" not in text, name


def test_no_prior_prediction_reuse():
    text = (PKG / "runner.py").read_text()
    assert "runs/E00" not in text and "research/expos/runs" not in text
    low = text.lower()
    assert "shutil" not in low
    assert "read_parquet" not in low  # fresh refits only; predictions are written, never read
    assert "E0041" not in text  # no repro-file provenance needed; seeds + rule are self-contained


def test_runner_import_only_and_causal_markers():
    src = (PKG / "runner.py").read_text()
    assert "import strips as ST" in src and "from existing_all import" in src
    assert "import vote as VM" in src
    for token in ("OMEGA", "PARKER_WEST", "strip_weights", "def lam(", "def full_grid",
                  "def speed_features", "def enlil_features", "def select_slots",
                  "def strip_features"):
        assert token not in src, f"runner must not copy E0003 code: {token}"
    assert "ST.strip_features" in src and "ST.strip_columns()" in src
    assert "ST.poison_test" in src and "ST.build_manifest" in src
    assert "VM.seed_params" in src and "VM.voted_predictions" in src and "VM.check_seeds" in src
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
    assert {"vote.py", "runner.py"} <= set(files)
    for name in ("vote.py", "runner.py"):
        text = (PKG / name).read_text()
        assert "predictions_official" not in text
    runner_text = (PKG / "runner.py").read_text()
    assert "store/scratch/worker-d28" in runner_text
    for other in ("worker-d5", "worker-d6", "worker-d7", "worker-d8", "worker-d10",
                  "worker-d12", "worker-d13", "worker-d16", "worker-d17", "worker-d19",
                  "worker-d20", "worker-d21", "worker-d22", "worker-d23", "worker-d24",
                  "worker-d26", "worker-repro", "worker-final", "worker-img-a", "worker-hpo"):
        assert other not in runner_text, f"scratch must stay in worker-d28 only: {other}"
    assert "600.0" in (PKG / "vote.py").read_text()
