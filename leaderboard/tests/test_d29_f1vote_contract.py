"""D29 F1-tuned vote contract: pinned grid, dev-only F1-first, import-only.

CPU-only, synthetic numerics only. No fits, no GPU, no network, no data reads.
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

LB = Path(__file__).resolve().parents[1]
PKG = LB / "experiments" / "d29_f1vote"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(LB / "experiments" / "d28_seedvote"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
import f1tune as FT  # noqa: E402
import vote as VM  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]


def _preds(n=120, seed=0, loc=400.0, scale=50.0):
    rng = np.random.default_rng(seed)
    return rng.normal(loc, scale, n)


def test_grid_pinned_15_combos_and_floor():
    assert tuple(FT.THRESHOLDS) == (500.0, 550.0, 600.0, 650.0, 700.0)
    assert tuple(FT.MAJORITIES) == (3, 4, 5)
    assert len(FT.GRID) == 15 and FT.N_GRID == 15
    assert (600.0, 3) in FT.GRID and (500.0, 5) in FT.GRID
    assert FT.F1_FLOOR == pytest.approx(0.41121495)
    assert FT.FIXED == (600.0, 3)
    assert "0.41121495" in FT.SELECT_RULE
    assert tuple(FT.SEEDS) == tuple(VM.SEEDS) == (10, 11, 12, 13, 14)


def test_check_candidate_pins_grid():
    assert FT.check_candidate(600.0, 3) == (600.0, 3)
    with pytest.raises(ValueError):
        FT.check_candidate(625.0, 3)
    with pytest.raises(ValueError):
        FT.check_candidate(600.0, 2)
    with pytest.raises(ValueError):
        FT.check_candidate(600.0, 6)


def test_voted_with_majority3_matches_d28_fixed():
    n = 96
    pred = {s: _preds(n, seed=s) for s in FT.SEEDS}
    assert np.allclose(FT.voted_with(pred, 600.0, 3), VM.voted_predictions(pred))
    assert np.allclose(FT.voted_with(pred, 600.0, 3),
                       VM.voted_predictions(pred, threshold=600.0))


def test_voted_with_majority_ladder():
    n = 32
    # exactly 3 surge votes: maj3 surge, maj4/5 calm
    cells = [np.full(n, 700.0)] * 3 + [np.full(n, 400.0)] * 2
    pred = {s: v for s, v in zip(FT.SEEDS, cells)}
    assert np.allclose(FT.voted_with(pred, 600.0, 3), np.full(n, 700.0))
    assert np.allclose(FT.voted_with(pred, 600.0, 4), np.full(n, 400.0))
    assert np.allclose(FT.voted_with(pred, 600.0, 5), np.full(n, 400.0))
    # exactly 4 surge votes: maj3/4 surge, maj5 calm
    cells4 = [np.full(n, 700.0)] * 4 + [np.full(n, 400.0)]
    pred4 = {s: v for s, v in zip(FT.SEEDS, cells4)}
    assert np.allclose(FT.voted_with(pred4, 600.0, 3), np.full(n, 700.0))
    assert np.allclose(FT.voted_with(pred4, 600.0, 4), np.full(n, 700.0))
    assert np.allclose(FT.voted_with(pred4, 600.0, 5), np.full(n, 400.0))
    # unanimous surge: all majorities take the surge mean
    hot = {s: np.full(n, 700.0 + s) for s in FT.SEEDS}
    for maj in (3, 4, 5):
        assert np.allclose(FT.voted_with(hot, 600.0, maj),
                           np.mean(np.stack([hot[s] for s in FT.SEEDS]), axis=0))


def test_voted_with_threshold_boundary_and_single_outlier():
    n = 24
    at = [np.full(n, 600.0)] * 3 + [np.full(n, 400.0)] * 2
    pred = {s: v for s, v in zip(FT.SEEDS, at)}
    assert np.allclose(FT.voted_with(pred, 600.0, 3), np.full(n, 600.0))
    spike = {s: np.full(n, 400.0) for s in FT.SEEDS}
    spike[FT.SEEDS[0]] = np.full(n, 900.0)
    for maj in (3, 4, 5):
        assert np.allclose(FT.voted_with(spike, 600.0, maj), np.full(n, 400.0))


def test_voted_with_rejects_bad_inputs():
    good = {s: _preds(40, seed=s) for s in FT.SEEDS}
    with pytest.raises(ValueError):
        FT.voted_with({s: _preds(40, seed=s) for s in (10, 11, 12, 13)}, 600.0, 3)
    with pytest.raises(ValueError):
        FT.voted_with({s: _preds(40, seed=s) for s in (0, 1, 2, 3, 4)}, 600.0, 3)
    bad = dict(good)
    bad[FT.SEEDS[0]] = bad[FT.SEEDS[0]][:10]
    with pytest.raises(ValueError):
        FT.voted_with(bad, 600.0, 3)
    with pytest.raises(ValueError):
        FT.voted_with(good, 625.0, 3)
    with pytest.raises(ValueError):
        FT.voted_with(good, 600.0, 2)


def _scores(f1_top=(550.0, 3), feasible_all=True):
    guard = 8000.0
    mse = {k: (7900.0 if feasible_all else 8100.0) for k in FT.GRID}
    f1 = {k: 0.30 for k in FT.GRID}
    if feasible_all:
        mse[f1_top] = 7950.0
        f1[f1_top] = 0.45
    return mse, f1, guard


def test_select_picks_highest_feasible_f1():
    mse, f1, guard = _scores()
    w, info = FT.select_threshold(mse, f1, guard)
    assert w == (550.0, 3) and info["abstained"] is False
    assert info["floor_pass"] is True


def test_select_guard_blocks_mse_blowup_to_fixed():
    mse = {k: 8100.0 for k in FT.GRID}
    f1 = {k: 0.50 for k in FT.GRID}
    w, info = FT.select_threshold(mse, f1, 8000.0)
    assert w == FT.FIXED and info["abstained"] is True
    assert info["feasible"] == []
    assert "abstains to fixed" in info["reason"]


def test_select_floor_is_aspiration_only():
    mse = {k: 7900.0 for k in FT.GRID}
    f1 = {k: 0.30 for k in FT.GRID}
    f1[(650.0, 4)] = 0.35
    w, info = FT.select_threshold(mse, f1, 8000.0)
    assert w == (650.0, 4) and info["abstained"] is False
    assert info["floor_pass"] is False
    assert "aspiration" in info["reason"]


def test_select_undefined_f1_skipped_or_falls_back():
    mse = {k: 7900.0 for k in FT.GRID}
    f1 = {k: None for k in FT.GRID}
    f1[(500.0, 3)] = 0.42
    w, info = FT.select_threshold(mse, f1, 8000.0)
    assert w == (500.0, 3) and info["abstained"] is False
    w2, info2 = FT.select_threshold(mse, {k: None for k in FT.GRID}, 8000.0)
    assert w2 == FT.FIXED and info2["abstained"] is True


def test_select_rejects_wrong_grid_or_guard():
    mse, f1, guard = _scores()
    bad_mse = dict(mse)
    del bad_mse[(500.0, 3)]
    with pytest.raises(ValueError):
        FT.select_threshold(bad_mse, f1, guard)
    with pytest.raises(ValueError):
        FT.select_threshold(mse, {k: 0.3 for k in list(FT.GRID)[:5]}, guard)
    with pytest.raises(ValueError):
        FT.select_threshold(mse, f1, float("nan"))


def test_select_dev_only_by_signature():
    assert list(inspect.signature(FT.select_threshold).parameters) == [
        "dev_mse", "dev_f1", "e0003_dev_mse"]
    src = (PKG / "f1tune.py").read_text()
    assert "selection" in src and "selection/audit never" in src
    assert "audit" in src  # only as the never-used window name


def test_no_forbidden_or_window_dates_in_owned_code():
    for name in ("f1tune.py", "runner.py"):
        text = (PKG / name).read_text()
        assert not [p for p in FORBIDDEN if p in text], name
        for token in ("NOAA", "GONG", "DONKI", "noaa-goes18.s3", "s3.amazonaws"):
            assert token not in text, f"{name}:{token}"
        assert "2026-06" not in text, name
        assert "2026-05-29" not in text and "2026-05-31" not in text, name
        assert "predictions_official" not in text, name


def test_no_prior_prediction_reuse():
    text = (PKG / "runner.py").read_text()
    assert "runs/E00" not in text and "research/expos/runs" not in text
    low = text.lower()
    assert "shutil" not in low
    assert "E0041" not in text
    assert "unique_event_report" in text  # dev-OOF scoring only
    assert "dev_origins" in text
    assert 'scores["selection"]' not in low and "scores['selection']" not in low


def test_runner_import_only_and_causal_markers():
    src = (PKG / "runner.py").read_text()
    assert "import strips as ST" in src and "from existing_all import" in src
    assert "import vote as VM" in src and "import f1tune as FT" in src
    for token in ("OMEGA", "PARKER_WEST", "strip_weights", "def lam(", "def full_grid",
                  "def speed_features", "def enlil_features", "def select_slots",
                  "def strip_features"):
        assert token not in src, f"runner must not copy E0003 code: {token}"
    assert "ST.strip_features" in src and "ST.strip_columns()" in src
    assert "ST.poison_test" in src and "ST.build_manifest" in src
    assert "VM.seed_params" in src and "VM.check_seeds" in src
    assert "FT.voted_with" in src and "FT.select_threshold" in src
    assert "xgb.XGBRegressor" in src
    assert "--pilot" in src and '"6h"' in src and '"3h"' in src
    assert "72h" in src
    assert "poison.json" in src and '"passed"' in src
    assert "availability_manifest.json" in src and "selection.json" in src
    assert "for name, (lo, hi) in PERIODS" in src
    assert "EXPOS_OUT" in src
    assert "tree_method" in src and "n_jobs" in src
    assert "cuda" in src.lower()  # only as the no-GPU assertion


def test_runner_dev_only_selection_markers():
    src = (PKG / "runner.py").read_text()
    assert "te_dev" in src and "dev-OOF" in src
    assert "e0003_dev_mse" in src or "e0003_mse" in src
    assert "unique_event_report" in src
    low = src.lower()
    assert "selection/audit never" in low


def test_owned_dir_is_small_and_scratch_isolated():
    files = sorted(p.name for p in PKG.iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"f1tune.py", "runner.py"} <= set(files)
    runner_text = (PKG / "runner.py").read_text()
    assert "store/scratch/worker-d29" in runner_text
    for other in ("worker-d5", "worker-d6", "worker-d7", "worker-d8", "worker-d10",
                  "worker-d12", "worker-d13", "worker-d16", "worker-d17", "worker-d19",
                  "worker-d20", "worker-d21", "worker-d22", "worker-d23", "worker-d24",
                  "worker-d25", "worker-d26", "worker-d27", "worker-d28",
                  "worker-repro", "worker-final", "worker-img-a", "worker-hpo"):
        assert other not in runner_text, f"scratch must stay in worker-d29 only: {other}"
    f1tune_text = (PKG / "f1tune.py").read_text()
    assert "worker-" not in f1tune_text, "scratch lives in runner.py only"
