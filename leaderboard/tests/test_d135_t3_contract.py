"""Contract for D135 multi-seed GRU strip+ACE sequence model (experiments/d135_t3seq/)."""
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

D135 = Path(__file__).resolve().parents[1] / "experiments" / "d135_t3seq"
sys.path.insert(0, str(D135))
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
    return (D135 / name).read_text()


def test_hypothesis_dual_tag_leading_and_e0121():
    SP.check_tags()
    assert SP.FAMILY_TAG == "[family=nn]"
    assert SP.AXES_TAG == "[axes: model=gru, ens=seed-mean]"
    head = SP.HYPOTHESIS.splitlines()[0]
    assert head.startswith("[family=nn]") and "[axes: model=gru, ens=seed-mean]" in SP.HYPOTHESIS
    for cite in ("E0121", "X0044", "X0046", "X0047", "X0052", "E0003"):
        assert cite in SP.HYPOTHESIS, f"missing citation {cite}"
    assert "72" in SP.HYPOTHESIS and "seed-mean" in SP.HYPOTHESIS


def test_multiseed_preregistered_and_scratch_owned():
    assert tuple(SP.SEEDS) == (0, 1, 2)
    assert SP.MIN_EVIDENCE_S == 600
    assert "MIN-10MIN-RULE" in SP.__doc__ and "rocv" in SP.__doc__.lower()
    assert str(SP.SCRATCH).endswith("store/scratch/worker-d135")
    for f in ("spec.py", "features.py", "model.py", "runner.py", "rocv.py"):
        assert "worker-d52" not in _src(f), f"{f} must own the worker-d135 scratch"
        assert "worker-d59" not in _src(f) and "worker-d61" not in _src(f)


def test_no_forbidden_or_copy():
    for f in ("spec.py", "features.py", "model.py", "runner.py", "rocv.py"):
        text = _src(f)
        for s in FORBIDDEN:
            assert s not in text, f"{f} must not reference {s}"
        assert "2026-06" not in text, f"{f} must not reference the June window"
        assert "official" not in text.lower(), f"{f} must not write frozen-window predictions"
        for ban in COPY_BANS:
            assert ban not in text, f"{f} must import strips, never vendor it"


def test_probe_dates_only_in_probes():
    assert set(SP.PROBES) == {"dev", "selection", "audit"}
    for f in ("features.py", "model.py", "rocv.py"):
        assert "2025-12-04" not in _src(f) and "2026-03-04" not in _src(f)
    n_dev = sum(_src("runner.py").count(s) for s in ("2025-09-15", "2026-01-05", "2026-04-06"))
    assert n_dev == 0, "probe dates must live in spec.PROBES only"


def test_origins_only_signatures():
    assert list(inspect.signature(FE.sequence_matrix).parameters)[:1] == ["origins"]
    assert list(inspect.signature(FE.matrix_from_hourly).parameters) == ["x", "origins"]
    assert list(inspect.signature(FE.static_frame).parameters) == ["origins"]
    assert list(inspect.signature(FE.seq_from_frames).parameters) == [
        "strip_hist", "plasma", "origins"]


def _synth_plasma(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    data = {c: rng.normal(400 if "speed" in c else 0, 10, n).astype(np.float32)
            for c in FE.PLASMA_COLS}
    return pd.DataFrame(data, index=idx)


def _synth_strips(n=400, seed=1):
    import strips as ST  # noqa: E402
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    data = {c: rng.normal(0, 1, n).astype(np.float32) for c in ST.strip_columns()}
    return pd.DataFrame(data, index=idx)


def test_sequence_trailing_and_causal_truncation_invariant():
    x = _synth_plasma()
    origins = pd.DatetimeIndex(["2025-01-15T00:00:00Z", "2025-01-16T12:00:00Z"])
    m = FE.matrix_from_hourly(x, origins)
    assert m.shape == (2, FE.HIST, len(FE.PLASMA_COLS))
    o0 = pd.DatetimeIndex(["2025-01-15T00:00:00Z"])
    assert np.allclose(m[0], FE.matrix_from_hourly(x, o0)[0], equal_nan=True)
    future = _synth_plasma(n=500, seed=0)
    x_same_stream = future.iloc[:400].copy()
    x_same_stream.index = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
    m = FE.matrix_from_hourly(x_same_stream, origins)
    m2 = FE.matrix_from_hourly(future, origins)
    assert np.allclose(m, m2, equal_nan=True)


def test_seq_channels_and_static_width():
    sh, pl = _synth_strips(), _synth_plasma()
    origins = pd.DatetimeIndex(["2025-01-15T00:00:00Z"])
    seq = FE.seq_from_frames(sh, pl, origins)
    assert seq.shape == (1, SP.HIST, SP.N_SEQ) and SP.N_SEQ == 30
    assert seq.shape[2] == SP.N_STRIP + SP.N_PLASMA


def test_gru_unidirectional_last_state_and_72_head():
    src = _src("model.py")
    assert "bidirectional=False" in src and "bidirectional\": False" in src
    assert "h[:, -1, :]" in src
    assert "bidirectional=True" not in src and "Attention" not in src and "Transformer" not in src
    net = MO.D135Net(n_static=70)
    assert net.head[-1].out_features == 72
    assert MO.count_params(net) > 50000, "X0044 forbids shallow nets"
    seq = torch.randn(4, SP.HIST, SP.N_SEQ)
    stat = torch.randn(4, 70)
    out = net(seq, stat)
    assert out.shape == (4, 72) and torch.isfinite(out).all()


def test_budget_and_dev_only_markers():
    src = _src("runner.py")
    assert "BUDGET-DEV-OOF-CONVERGENCE" in src
    assert "NORMALIZE-FROM-DEV-ONLY" in src and "EARLY-STOP-DEV-ONLY" in src
    assert "MIN-10MIN-RULE" in src
    assert "frozen_epochs=int(best_ep[sd])" in src
    assert "for sd in seeds" in src and "np.mean([seed_preds[sd]" in src
    assert "learning_curves.json" in src and "converged" in src
    assert "cuda" in src
    assert "poison.json" in src and '"passed": true' not in src.lower()  # passed comes from poison_check
    assert "positive_control" in src and "use_cache=False" in src
    assert SP.TORCH_THREADS + SP.PREFETCH_THREADS <= 20


def test_predictions_three_kinds_seedmean():
    src = _src("runner.py")
    for name in ("dev", "selection", "audit"):
        assert name in src
    assert "seed-mean" in src or "seed_mean" in src


def test_owned_dir_is_small():
    files = sorted(p.name for p in D135.iterdir() if p.suffix == ".py")
    assert len(files) <= 6, files
    assert {"spec.py", "features.py", "model.py", "runner.py", "rocv.py"} <= set(files)


def _load_d135_runner():
    import importlib.util
    loc = str(D135 / "runner.py")
    for name, mod in list(sys.modules.items()):
        if getattr(mod, "__file__", None) == loc:
            return mod
    spec = importlib.util.spec_from_file_location("d135_runner", loc)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["d135_runner"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_valid_mask_matches_purge_and_drops_missing_targets():
    mod = _load_d135_runner()
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
