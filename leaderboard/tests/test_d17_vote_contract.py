"""D17 extreme-vote contract: seeds, cuda recipe, dev-only weights+alphas, import-only.

CPU-only, synthetic numerics only. No fits, no GPU, no network, no data reads.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1] / "experiments" / "d17_extreme_vote"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "d12_bagged"))
import vote as VT  # noqa: E402
import f1first_vote as F1V  # noqa: E402
import bag as BG  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]


def test_seeds_fixed_main_and_alt():
    assert tuple(VT.SEEDS) == (0, 1, 2, 3, 4)
    assert tuple(VT.ALT_SEEDS) == (5, 6, 7, 8, 9)
    assert set(VT.SEEDS).isdisjoint(VT.ALT_SEEDS)
    assert VT.resolve_seeds() == (0, 1, 2, 3, 4)
    assert VT.resolve_seeds(alt=True) == (5, 6, 7, 8, 9)
    assert VT.check_seeds([0, 1, 2, 3, 4]) == (0, 1, 2, 3, 4)
    with pytest.raises(ValueError):
        VT.check_seeds([0, 1, 2, 3])
    with pytest.raises(ValueError):
        VT.check_seeds([0, 1, 2, 3, 5])
    with pytest.raises(ValueError):
        VT.check_seeds([5, 6, 7, 8, 9])


def test_params_cuda_e0003_recipe_and_bag_parity():
    import info_ablation as ia
    for s in (*VT.SEEDS, *VT.ALT_SEEDS):
        p = VT.seed_params(s)
        assert p["device"] == "cuda" and p["random_state"] == s
        for k, v in ia.PARAMS.items():
            if k == "random_state":
                continue
            assert p[k] == v, k
    for s in BG.SEEDS:
        assert VT.seed_params(s) == BG.seed_params(s)
    assert VT.DEVICE == "cuda"


def test_weights_dev_only_and_abort():
    mse = {s: 8000.0 + 10 * s for s in VT.SEEDS}
    w, method = VT.select_weights({"dev": mse}, VT.SEEDS)
    assert method == "dev-inverse-mse" and abs(sum(w.values()) - 1.0) < 1e-12
    assert w[0] > w[4]
    with pytest.raises(ValueError):
        VT.select_weights({"dev": mse, "selection": mse}, VT.SEEDS)
    with pytest.raises(ValueError):
        VT.select_weights({"selection": mse}, VT.SEEDS)
    with pytest.raises(ValueError):
        VT.select_weights({"dev": mse, "audit": mse}, VT.SEEDS)
    with pytest.raises(ValueError):
        VT.select_weights({}, VT.SEEDS)


def test_degenerate_dev_mse_falls_back_uniform_recorded():
    bad = {s: 8000.0 for s in VT.SEEDS}
    bad[VT.SEEDS[2]] = float("nan")
    w, method = VT.select_weights({"dev": bad}, VT.SEEDS)
    assert method.startswith("uniform-fallback:")
    assert all(v == pytest.approx(1 / 5) for v in w.values())


def test_weighted_mean_and_extreme_max():
    rng = np.random.default_rng(0)
    base = rng.normal(400, 50, 120)
    preds = {s: base + rng.normal(0, 1, 120) for s in VT.SEEDS}
    w, _ = VT.select_weights({"dev": {s: 8000.0 for s in VT.SEEDS}}, VT.SEEDS)
    out = VT.weighted_mean(preds, w)
    assert out.shape == (120,) and np.isfinite(out).all()
    assert np.allclose(out, sum(preds[s] * w[s] for s in VT.SEEDS))
    mx = VT.extreme_max(preds)
    assert np.allclose(mx, np.max(np.stack([preds[s] for s in VT.SEEDS]), axis=0))
    assert (mx >= out).all()
    with pytest.raises(ValueError):
        VT.weighted_mean({s: preds[s] for s in VT.SEEDS[:4]}, w)


def test_band_index_and_blend():
    h = np.array([1, 6, 7, 24, 25, 48, 49, 72])
    assert VT.band_index(h).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    mean = np.full(8, 400.0)
    extreme = np.full(8, 500.0)
    out = VT.blend_predictions(mean, extreme, h, (0.0, 0.0, 0.0, 0.5))
    assert out[:6].tolist() == [400.0] * 6
    assert out[6:].tolist() == [450.0, 450.0]
    with pytest.raises(ValueError):
        VT.blend_predictions(mean, extreme, h, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        VT.band_index(np.array([0, 73]))


def _synth(noise_std):
    rng = np.random.default_rng(7)
    h = np.tile(np.arange(1, 73), 4)
    y = 500.0 + rng.normal(0, noise_std, h.shape)
    preds = {}
    for s in VT.SEEDS:
        off = rng.normal(0, 1, h.shape)
        preds[s] = 500.0 + off + (0.6 if s == VT.SEEDS[-1] else 0.0)
    # force a clean spread: last seed is the row max everywhere
    for s in VT.SEEDS[:-1]:
        preds[s] = np.minimum(preds[s], preds[VT.SEEDS[-1]] - 0.2)
    return h, y, preds


def _f1_rewarding_blend(mean):
    base = float(np.sum(np.asarray(mean, dtype=float)))
    def f(blended):
        return 0.40 if float(np.sum(np.asarray(blended, dtype=float))) > base + 1e-9 else 0.30
    return f


def test_select_alphas_picks_smallest_gain_within_guard():
    h, y, preds = _synth(noise_std=5.0)
    w, _ = VT.select_weights({"dev": VT.dev_mse_of(y, preds)}, VT.SEEDS)
    mean = VT.weighted_mean(preds, w)
    alphas, info = VT.select_alphas(h, y, preds, w, _f1_rewarding_blend(mean))
    assert tuple(alphas) == (0.0, 0.0, 0.0, 0.15)
    assert info["combined"]["f1"] >= info["baseline"]["f1"]
    assert info["combined"]["global_rel"] <= VT.GLOBAL_MSE_MAX


def test_select_alphas_guard_blocks_mse_blowup():
    h, y, preds = _synth(noise_std=5.0)
    preds[VT.SEEDS[-1]] = preds[VT.SEEDS[-1]] + 40.0
    w, _ = VT.select_weights({"dev": VT.dev_mse_of(y, preds)}, VT.SEEDS)
    mean = VT.weighted_mean(preds, w)
    alphas, _ = VT.select_alphas(h, y, preds, w, _f1_rewarding_blend(mean))
    assert tuple(alphas) == (0.0, 0.0, 0.0, 0.0)


def test_select_alphas_never_picks_f1_worse():
    h, y, preds = _synth(noise_std=5.0)
    w, _ = VT.select_weights({"dev": VT.dev_mse_of(y, preds)}, VT.SEEDS)
    alphas, info = VT.select_alphas(h, y, preds, w, lambda b: 0.10)
    assert tuple(alphas) == (0.0, 0.0, 0.0, 0.0)
    assert info["combined"]["f1"] == pytest.approx(info["baseline"]["f1"])


def test_select_alphas_undefined_f1_falls_back_to_mean():
    h, y, preds = _synth(noise_std=5.0)
    w, _ = VT.select_weights({"dev": VT.dev_mse_of(y, preds)}, VT.SEEDS)
    alphas, info = VT.select_alphas(h, y, preds, w, lambda b: None)
    assert tuple(alphas) == (0.0, 0.0, 0.0, 0.0)
    assert "falls back to pure mean" in info["reason"]


def test_no_forbidden_or_window_dates_in_new_code():
    for name in ("vote.py", "runner.py"):
        text = (PKG / name).read_text()
        assert not [p for p in FORBIDDEN if p in text], name
        assert "2026-06" not in text, f"{name} must not reference the window after 2026-05"
        assert "2026-05-29" not in text and "2026-05-31" not in text, name


def test_runner_import_only_and_markers():
    src = (PKG / "runner.py").read_text()
    assert "import strips as ST" in src and "from existing_all import" in src
    assert "import vote as VT" in src
    for token in ("OMEGA", "PARKER_WEST", "strip_weights", "def lam(", "def full_grid",
                  "def speed_features", "def enlil_features"):
        assert token not in src, f"runner must not copy E0003 code: {token}"
    assert "ST.strip_features" in src and "ST.strip_columns()" in src
    assert "ST.poison_test" in src
    assert "unique_event_report" in src
    assert '"cuda"' in src or "'cuda'" in src or "device" in src
    assert "--pilot" in src and '"6h"' in src and '"3h"' in src
    assert "--alt-seeds" in src
    assert "72h" in src
    assert "poison.json" in src and "manifest.json" in src and "selection.json" in src
    assert "for name in PERIODS" in src
    assert "select_weights" in src and '{"dev": dev_mse}' in src
    assert "select_alphas" in src
    low = src.lower()
    assert 'scores["selection"]' not in low and "scores['selection']" not in low
    assert "selection" in low


def test_owned_dir_is_small():
    files = sorted(p.name for p in PKG.iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"vote.py", "runner.py"} <= set(files)


def _f1first_spiked(noise_std=5.0, spike=30.0, seed=11):
    """Synthetic where the row-max extreme helps both MSE and F1 on band 49-72.

    y carries spikes on band-3 rows; every seed under-predicts them except the
    last seed (row max), which nails them. Larger band-3 alphas then improve
    both dev MSE and a blend-rewarding F1, so F1-first must pick the grid max
    while MSE-first greedy stops at the first gain.
    """
    rng = np.random.default_rng(seed)
    h = np.tile(np.arange(1, 73), 6)
    y = 500.0 + rng.normal(0, noise_std, h.shape)
    y[h >= 49] += spike
    preds = {}
    for s in VT.SEEDS:
        preds[s] = 500.0 + rng.normal(0, 1, h.shape)
    preds[VT.SEEDS[-1]] = np.where(h >= 49, y, preds[VT.SEEDS[-1]])
    for s in VT.SEEDS[:-1]:
        preds[s] = np.minimum(preds[s], preds[VT.SEEDS[-1]] - 0.2)
    return h, y, preds


def _f1_sums(mean):
    base = float(np.sum(np.asarray(mean, dtype=float)))

    def f(blended):
        return 0.30 + 0.05 * float(np.sum(np.asarray(blended, dtype=float)) - base)
    return f


def test_f1first_pins_floor_grid_and_strict_guard():
    assert F1V.F1_FLOOR == pytest.approx(0.41121495)
    assert tuple(F1V.GRID) == (0.0, 0.15, 0.30, 0.50)
    assert F1V.N_GRID == 256
    assert F1V.GLOBAL_MSE_MAX == 0.0
    assert F1V.BAND_MSE_MAX == 0.01
    assert "0.41121495" in F1V.VOTE_RULE


def test_f1first_maximizes_not_first_gain():
    h, y, preds = _f1first_spiked()
    w, _ = VT.select_weights({"dev": VT.dev_mse_of(y, preds)}, VT.SEEDS)
    mean = VT.weighted_mean(preds, w)
    alphas, info = F1V.select_alphas_f1first(h, y, preds, w, _f1_sums(mean))
    assert tuple(alphas)[3] == 0.50
    greedy, _ = VT.select_alphas(h, y, preds, w, _f1_sums(mean))
    assert tuple(greedy)[3] == 0.15
    assert info["combined"]["f1"] >= info["baseline"]["f1"]
    assert info["combined"]["global_rel"] <= F1V.GLOBAL_MSE_MAX
    assert info["combined"]["floor_pass"] is True
    assert info["search"]["n_mse_feasible"] >= 2


def test_f1first_strict_mse_rejects_f1_gain():
    h, y, preds = _synth(noise_std=5.0)
    preds[VT.SEEDS[-1]] = preds[VT.SEEDS[-1]] + 40.0
    w, _ = VT.select_weights({"dev": VT.dev_mse_of(y, preds)}, VT.SEEDS)
    mean = VT.weighted_mean(preds, w)
    alphas, info = F1V.select_alphas_f1first(h, y, preds, w, _f1_rewarding_blend(mean))
    assert tuple(alphas) == (0.0, 0.0, 0.0, 0.0)
    assert info["combined"]["global_rel"] == pytest.approx(0.0)


def test_f1first_floor_abstains_to_mean_when_below_bar():
    h, y, preds = _synth(noise_std=5.0)
    w, _ = VT.select_weights({"dev": VT.dev_mse_of(y, preds)}, VT.SEEDS)
    mean = VT.weighted_mean(preds, w)

    def f(blended):
        return 0.40 if float(np.sum(np.asarray(blended, dtype=float))) > float(
            np.sum(mean)) + 1e-9 else 0.30

    alphas, info = F1V.select_alphas_f1first(h, y, preds, w, f)
    assert tuple(alphas) == (0.0, 0.0, 0.0, 0.0)
    assert info["combined"]["floor_pass"] is False
    assert info["combined"]["f1"] == pytest.approx(info["baseline"]["f1"])
    assert "abstains to pure mean" in info["reason"]


def test_f1first_never_picks_f1_worse_and_undefined_falls_back():
    h, y, preds = _synth(noise_std=5.0)
    w, _ = VT.select_weights({"dev": VT.dev_mse_of(y, preds)}, VT.SEEDS)
    alphas, info = F1V.select_alphas_f1first(h, y, preds, w, lambda b: 0.10)
    assert tuple(alphas) == (0.0, 0.0, 0.0, 0.0)
    assert info["combined"]["f1"] == pytest.approx(info["baseline"]["f1"])
    alphas, info = F1V.select_alphas_f1first(h, y, preds, w, lambda b: None)
    assert tuple(alphas) == (0.0, 0.0, 0.0, 0.0)
    assert "falls back to pure mean" in info["reason"]


def test_f1first_dev_only_by_construction():
    import inspect

    assert list(inspect.signature(F1V.select_alphas_f1first).parameters) == [
        "h", "y", "pred_by_seed", "weights", "f1_of"]
    src = (PKG / "f1first_vote.py").read_text()
    assert "import vote as VT" in src
    for token in ("def full_grid", "def speed_features", "def enlil_features",
                  "OMEGA", "PARKER_WEST"):
        assert token not in src, f"f1first_vote must not copy E0003 code: {token}"
    for name in ("vote.py", "runner.py", "f1first_vote.py"):
        text = (PKG / name).read_text()
        assert not [p for p in FORBIDDEN if p in text], name
        assert "2026-06" not in text, name
        assert "2026-05-29" not in text and "2026-05-31" not in text, name


def test_runner_uses_f1first_vote():
    src = (PKG / "runner.py").read_text()
    assert "import f1first_vote as F1V" in src
    assert "select_alphas_f1first" in src
    assert "F1V.VOTE_RULE" in src
    assert "floor_pass" in src
    assert "import vote as VT" in src


def test_owned_dir_still_small_with_f1first():
    files = sorted(p.name for p in PKG.iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"vote.py", "runner.py", "f1first_vote.py"} <= set(files)
