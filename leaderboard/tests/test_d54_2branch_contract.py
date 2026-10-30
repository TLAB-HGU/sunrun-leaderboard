"""Contract for D54 2-branch SUVI-map + speed-history model (experiments/d54_2branch/)."""
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

D54 = Path(__file__).resolve().parents[1] / "experiments" / "d54_2branch"
sys.path.insert(0, str(D54))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "headroom_audit"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "expos_runners"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "d1_longitude_strips"))

import features as FE  # noqa: E402
import model as MO  # noqa: E402
import spec as SP  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def _src(name):
    return (D54 / name).read_text()


def test_hypothesis_dual_tag_leading_and_citations():
    SP.check_tags()
    assert SP.FAMILY_TAG == "[family=nn]"
    assert SP.AXES_TAG == "[axes: model=2branch]"
    head = SP.HYPOTHESIS.splitlines()[0]
    assert head.startswith("[family=nn]") and "[axes: model=2branch]" in SP.HYPOTHESIS
    for cite in ("E0121", "X0110", "E0124"):
        assert cite in SP.HYPOTHESIS, f"missing citation {cite}"
    assert "72" in SP.HYPOTHESIS and "6h" in SP.HYPOTHESIS


def test_full_rule_preregistered_and_scratch_owned():
    assert "poison" in SP.FULL_RULE and "1/3" in SP.FULL_RULE and "exp-add" in SP.FULL_RULE
    assert "converged" in SP.FULL_RULE and "rc 0" in SP.FULL_RULE
    assert "curves" in SP.FULL_RULE and "budget" in SP.FULL_RULE
    assert str(SP.SCRATCH).endswith("store/scratch/worker-d54")
    for f in ("spec.py", "features.py", "model.py", "runner.py"):
        assert "worker-d52" not in _src(f) and "worker-d53" not in _src(f)
        assert "worker-d59" not in _src(f) and "worker-d61" not in _src(f)


def test_no_forbidden_official_or_copy():
    for f in ("spec.py", "features.py", "model.py", "runner.py"):
        text = _src(f)
        for s in FORBIDDEN:
            assert s not in text, f"{f} must not reference {s}"
        assert "2026-06" not in text, f"{f} must not reference the official window"
        assert "official" not in text.lower(), f"{f} must not write official predictions"
        for ban in COPY_BANS:
            assert ban not in text, f"{f} must import strips, never vendor it"


def test_probe_dates_only_in_probes():
    assert set(SP.PROBES) == {"dev", "selection", "audit"}
    for f in ("features.py", "model.py"):
        assert "2025-12-04" not in _src(f) and "2026-03-04" not in _src(f)
    n_dev = sum(_src("runner.py").count(s) for s in ("2025-09-15", "2026-01-05", "2026-04-06"))
    assert n_dev == 0, "probe dates must live in spec.PROBES only"


def test_origins_only_signatures():
    assert list(inspect.signature(FE.speed_history).parameters) == ["x", "origins"]
    assert list(inspect.signature(FE.map_frame).parameters) == ["origins", "use_cache"]
    assert list(inspect.signature(FE.static_frame).parameters) == ["origins"]


def _synth_speed(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({FE.SPEED_COL: rng.normal(400, 50, n).astype(np.float32)}, index=idx)


def test_speed_history_trailing_and_causal_truncation_invariant():
    x = _synth_speed()
    origins = pd.DatetimeIndex(["2025-01-15T00:00:00Z", "2025-01-16T12:00:00Z"])
    m = FE.speed_history(x, origins)
    assert m.shape == (2, FE.HIST, 1)
    o0 = pd.DatetimeIndex(["2025-01-15T00:00:00Z"])
    assert np.allclose(m[0], FE.speed_history(x, o0)[0], equal_nan=True)
    # last step equals the origin-hour value (trailing, no lookahead)
    assert m[0, -1, 0] == pytest.approx(x[FE.SPEED_COL].loc[o0[0]], rel=1e-5)
    # appending future hours must not change rows for fixed origins
    future = _synth_speed(n=500, seed=0)
    x_same = future.iloc[:400].copy()
    x_same.index = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
    assert np.allclose(FE.speed_history(x_same, origins),
                       FE.speed_history(future, origins), equal_nan=True)


def test_map_columns_are_17_strips():
    import strips as ST  # noqa: E402

    assert FE.MAP_COLS == ST.strip_columns()
    assert len(FE.MAP_COLS) == SP.N_MAP == 17


def test_twobranch_separate_encoders_and_72_head():
    src = _src("model.py")
    assert "bidirectional=False" in src and "bidirectional\": False" in src
    assert "h[-1]" in src
    assert "bidirectional=True" not in src and "Attention" not in src and "Transformer" not in src
    # branches must not interact before the head: single cat in D54Net.forward only
    assert src.count("torch.cat") == 1
    net = MO.D54Net(n_static=70)
    assert net.head[-1].out_features == 72
    assert MO.count_params(net) > 20000, "branches must be real (no shallow stub)"
    mp = torch.randn(4, SP.N_MAP)
    sq = torch.randn(4, SP.HIST, SP.N_SPEED)
    st = torch.randn(4, 70)
    assert net.map_br(mp).shape == (4, SP.MAP_HIDDEN[-1])
    assert net.spd_br(sq).shape == (4, SP.SPEED_HIDDEN)
    out = net(mp, sq, st)
    assert out.shape == (4, 72) and torch.isfinite(out).all()


def test_budget_and_dev_only_markers():
    src = _src("runner.py")
    assert "BUDGET-DEV-OOF-CONVERGENCE" in src
    assert "NORMALIZE-FROM-DEV-ONLY" in src and "EARLY-STOP-DEV-ONLY" in src
    assert "frozen_epochs=int(best_epoch)" in src
    assert "learning_curves.json" in src and "converged" in src
    assert "MAX_TRAIN_S_PILOT" in src or "30min" in src
    assert "--pilot" in src and "cuda" in src
    assert "poison.json" in src and '"passed": true' not in src.lower()  # passed comes from poison_check
    assert "positive_control" in src and "use_cache=False" in src
    assert SP.TORCH_THREADS + SP.PREFETCH_THREADS <= 20


def test_predictions_three_kinds_no_official():
    src = _src("runner.py")
    assert "write(te, pred.reshape(-1)" in src
    for name in ("dev", "selection", "audit"):
        assert name in src


def test_owned_dir_is_small():
    files = sorted(p.name for p in D54.iterdir() if p.suffix == ".py")
    assert len(files) <= 6, files
    assert {"spec.py", "features.py", "model.py", "runner.py"} <= set(files)


def _load_d54_runner():
    import importlib.util
    loc = str(D54 / "runner.py")
    for name, mod in list(sys.modules.items()):
        if getattr(mod, "__file__", None) == loc:
            return mod
    spec = importlib.util.spec_from_file_location("d54_runner", loc)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["d54_runner"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_valid_mask_matches_purge_and_drops_missing_targets():
    mod = _load_d54_runner()
    origins = pd.DatetimeIndex(["2025-08-10T00:00:00Z", "2025-08-10T06:00:00Z"])
    full = np.full((2, 72), 500.0, dtype=np.float32)
    m = mod.valid_mask(origins, pd.Timestamp("2025-09-01T00:00:00Z"))
    assert m.shape == (2, 72) and bool(m.all())
    m_old = mod.valid_mask(origins, pd.Timestamp("2025-08-04T00:00:00Z"))
    assert not bool(m_old.any()), "72h purge must exclude near-window origins"
    ymat = full.copy()
    ymat[0, :10] = np.nan
    m2 = mod.valid_mask(origins, pd.Timestamp("2025-09-01T00:00:00Z"), ymat)
    assert not bool(m2[0, :10].any()) and bool(m2[0, 10:].all()) and bool(m2[1].all())


def test_static_frame_matches_ia_build():
    import info_ablation as ia  # noqa: E402
    df = ia.build(freq="6h", start="2025-08-04", end="2025-08-04 12:00")
    origins = pd.DatetimeIndex(df["origin"].unique()).sort_values()
    got = FE.static_frame(origins)
    ref = df.drop_duplicates("origin").set_index("origin").reindex(origins)
    shared = [c for c in got.columns if c in ref.columns]
    assert len(shared) == len(got.columns) and len(shared) > 50
    for c in shared:
        a = got[c].to_numpy(dtype=np.float64)
        b = ref[c].to_numpy(dtype=np.float64)
        assert np.allclose(a, b, equal_nan=True), f"static col {c} drifts from ia.build"
