"""Contract for D135-R6 multiscale-input single-axis redesign (experiments/d135_t3seq/r6-msinput/)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

R6 = Path(__file__).resolve().parents[1] / "experiments" / "d135_t3seq" / "r6-msinput"
D135 = R6.parent

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]
RECOMB_BANS = ["predictions_", "blend", "stacking", "moe", "expos/runs"]

IDENTITY_KEYS = ("SEEDS", "HIST", "N_STRIP", "N_PLASMA",
                 "GRU_HIDDEN", "GRU_LAYERS", "MLP_HIDDEN", "N_OUT",
                 "LR", "BATCH", "VAL_FRAC", "PATIENCE", "MAX_EPOCHS_FULL",
                 "MIN_EVIDENCE_S", "TORCH_THREADS", "PREFETCH_THREADS",
                 "PURGE_H", "DEV", "STATIC_DROP", "PROBES")

OTHER_SCRATCHES = ("worker-d135", "worker-t3b", "worker-t3c", "worker-t3d",
                   "worker-d52", "worker-d59", "worker-d61")


def _load(name, path):
    loc = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(loc)
    sys.modules[name] = mod
    loc.loader.exec_module(mod)
    return mod


def _src(name):
    return (R6 / name).read_text()


def _r6_modules():
    SP = sys.modules.get("d135r6_spec") or _load("d135r6_spec", R6 / "spec.py")
    old = dict(sys.modules)
    try:
        sys.modules["spec"] = SP
        MS = _load("d135r6_msinput", R6 / "msinput.py")
        return SP, MS
    finally:
        for k in [k for k in sys.modules if k not in old]:
            del sys.modules[k]
        if "spec" in old:
            sys.modules["spec"] = old["spec"]
        else:
            sys.modules.pop("spec", None)


def test_hypothesis_tags_and_cites():
    SP = _load("d135r6_spec", R6 / "spec.py")
    SP.check_tags()
    assert SP.FAMILY_TAG == "[family=nn]"
    assert SP.AXES_TAG == "[axes: model=gru, feat=multiscale, ens=seed-mean]"
    assert SP.HYPOTHESIS.startswith("[family=nn]")
    for cite in ("E0208", "E0210", "E0213", "E0215", "E0121",
                 "X0044", "X0046", "X0047", "X0052", "E0003"):
        assert cite in SP.HYPOTHESIS, f"missing citation {cite}"
    assert "seed-mean" in SP.HYPOTHESIS
    assert "multiscale" in SP.HYPOTHESIS.lower() or "multi-scale" in SP.HYPOTHESIS.lower()
    assert tuple(SP.MS_SCALES) == (1, 6, 24)
    assert SP.N_SEQ == 90


def test_single_axis_input_only_backbone_identical():
    SP = sys.modules.get("d135r6_spec") or _load("d135r6_spec", R6 / "spec.py")
    P = _load("d135_parent_spec", D135 / "spec.py")
    for k in IDENTITY_KEYS:
        a, b = getattr(SP, k), getattr(P, k)
        assert a == b, f"{k} must be identical to E0208 ({a!r} != {b!r})"
    assert (SP.GRU_HIDDEN, SP.GRU_LAYERS, tuple(SP.MLP_HIDDEN)) == (96, 2, (128, 64))
    assert (P.GRU_HIDDEN, P.GRU_LAYERS, tuple(P.MLP_HIDDEN)) == (96, 2, (128, 64))
    assert getattr(P, "MS_SCALES", None) is None, "E0208 has no multiscale axis"
    assert getattr(P, "HEAD_BANDS", None) is None, "E0208 has no band-head axis"
    assert getattr(P, "POOL", None) is None, "E0208 has no readout-pool axis"
    assert P.N_SEQ == 30 and SP.N_SEQ == 90
    assert str(SP.SCRATCH).endswith("store/scratch/worker-t3f")
    assert str(P.SCRATCH).endswith("store/scratch/worker-d135")


def test_scratch_owned_and_separated():
    SP = sys.modules.get("d135r6_spec") or _load("d135r6_spec", R6 / "spec.py")
    assert SP.SCRATCH.name == "worker-t3f"
    assert (SP.LB / "expos" / "core.py").is_file(), "r6 LB must resolve to leaderboard/"
    assert SP.SCRATCH.parent.name == "scratch"
    assert str(SP.SCRATCH) == str(SP.LB / "store/scratch/worker-t3f")
    for f in ("spec.py", "msinput.py", "runner.py", "rocv.py"):
        text = _src(f)
        for other in OTHER_SCRATCHES:
            assert other not in text, f"{f} must not touch scratch {other}"


def test_no_forbidden_or_copy_or_window():
    for f in ("spec.py", "msinput.py", "runner.py", "rocv.py"):
        text = _src(f)
        for s in FORBIDDEN:
            assert s not in text, f"{f} must not reference {s}"
        assert "2026-06" not in text, f"{f} must not reference the June window"
        assert "official" not in text.lower(), f"{f} must not write frozen-window predictions"
        for ban in COPY_BANS:
            assert ban not in text, f"{f} must reuse parent/strips, never vendor it"


def test_no_stored_prediction_reuse():
    for f in ("spec.py", "msinput.py", "runner.py", "rocv.py"):
        text = _src(f)
        for s in RECOMB_BANS:
            assert s not in text, f"{f} must not recombine stored predictions ({s})"


def test_probe_dates_only_in_probes():
    SP = sys.modules.get("d135r6_spec") or _load("d135r6_spec", R6 / "spec.py")
    assert set(SP.PROBES) == {"dev", "selection", "audit"}
    for f in ("runner.py", "rocv.py"):
        assert "2025-12-04" not in _src(f) and "2026-03-04" not in _src(f)
        assert "2025-09-15" not in _src(f) and "2026-01-05" not in _src(f)


def test_runner_reuses_parent_training_logic():
    src = _src("runner.py")
    for banned in ("def fit_gru", "def valid_mask", "def predict", "def poison_check",
                   "def standardize_fit", "def apply_std", "class ", "GRU("):
        assert banned not in src, f"runner must reuse parent logic ({banned})"
    assert "R.fit_gru" in src and "R.valid_mask" in src and "R.predict" in src
    assert "R.poison_check" in src and "R.prep_origin_arrays" in src
    assert "R.standardize_fit" in src and "R.apply_std" in src
    assert "seq_multiscale" in src and "extended_need" in src
    assert "check_tags" in src and "d135_parent_runner" in src
    assert "--cpu" in src


def test_msinput_causal_and_nan_aware():
    SP, MS = _r6_modules()
    idx = pd.date_range("2025-01-01", periods=100, freq="h", tz="UTC")
    frame = pd.DataFrame({"a": np.arange(100, dtype=float),
                          "b": np.ones(100)}, index=idx)
    rolled = MS.trailing_means(frame)
    assert set(rolled) == {1, 6, 24}
    pd.testing.assert_frame_equal(rolled[1], frame.sort_index())
    assert rolled[6]["a"].iloc[5] == pytest.approx(2.5)  # mean of 0..5
    assert rolled[6]["a"].iloc[99] == pytest.approx(96.5)
    assert rolled[24]["a"].iloc[23] == pytest.approx(11.5)
    # early rows use short windows, never future data
    assert rolled[24]["a"].iloc[0] == pytest.approx(0.0)
    assert rolled[6]["b"].iloc[50] == pytest.approx(1.0)
    # NaN-aware: partial windows skip NaN, fully-NaN windows stay NaN
    frame2 = frame.copy()
    frame2.loc[idx[10:20], "a"] = np.nan
    r2 = MS.trailing_means(frame2)
    assert np.isnan(r2[6]["a"].iloc[15])  # window 10..15 all-NaN stays NaN
    assert r2[6]["a"].iloc[10] == pytest.approx(7.0)  # mean of 5..9, NaN at 10 skipped
    assert r2[6]["a"].iloc[20] == pytest.approx(20.0)  # window 15..20: only 20.0 valid
    # extended_need covers the trailing window plus the smoothing lookback
    origins = pd.DatetimeIndex([idx[50], idx[60]], tz="UTC")
    need = MS.extended_need(origins)
    assert need[0] == idx[50] - pd.Timedelta(hours=SP.HIST - 1 + SP.MS_PAD)
    assert need[-1] == idx[60]
    assert (need[1:] - need[:-1] == pd.Timedelta("1h")).all()


def test_msinput_truncation_invariance_and_shape():
    SP, MS = _r6_modules()
    LB = SP.LB
    sys.path.insert(0, str(LB / "experiments/d1_longitude_strips"))
    sys.path.insert(0, str(D135))
    import strips as ST  # noqa: E402
    import features as PFE  # noqa: E402
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-02-01", periods=300, freq="h", tz="UTC")
    strip = pd.DataFrame(rng.normal(size=(300, len(ST.strip_columns()))),
                         index=idx, columns=ST.strip_columns())
    plasma = pd.DataFrame(rng.normal(size=(300, len(PFE.PLASMA_COLS))),
                          index=idx, columns=PFE.PLASMA_COLS)
    origins = pd.DatetimeIndex(idx[100:110].to_numpy(), tz="UTC")
    seq = MS.seq_multiscale(strip, plasma, origins, seq_from_frames=PFE.seq_from_frames)
    assert seq.shape == (len(origins), SP.HIST, SP.N_SEQ)
    assert np.isfinite(seq).all()
    base = PFE.seq_from_frames(strip, plasma, origins)
    np.testing.assert_array_equal(seq[:, :, :30], base.astype(np.float32))
    # 6h block step j = nan-mean of base steps j-5..j
    j = 40
    expect6 = np.nanmean(base[3, j - 5:j + 1, :], axis=0)
    np.testing.assert_allclose(seq[3, j, 30:60], expect6.astype(np.float32), rtol=1e-5)
    expect24 = np.nanmean(base[3, j - 23:j + 1, :], axis=0)
    np.testing.assert_allclose(seq[3, j, 60:], expect24.astype(np.float32), rtol=1e-5)
    # truncating frames after the last origin leaves earlier rows identical
    cut = idx[105]
    seq_tr = MS.seq_multiscale(strip.loc[:cut], plasma.loc[:cut], origins[origins <= cut],
                               seq_from_frames=PFE.seq_from_frames)
    seq_full_early = MS.seq_multiscale(strip, plasma, origins[origins <= cut],
                                       seq_from_frames=PFE.seq_from_frames)
    np.testing.assert_array_equal(seq_tr, seq_full_early)


def test_model_wiring_parent_backbone_at_90_inputs():
    SP = sys.modules.get("d135r6_spec") or _load("d135r6_spec", R6 / "spec.py")
    P = sys.modules.get("d135_parent_spec") or _load("d135_parent_spec", D135 / "spec.py")
    old = dict(sys.modules)
    try:
        sys.modules["spec"] = SP
        MO = _load("d135r6_parent_model", D135 / "model.py")
        assert MO.ARCH["gru"]["hidden"] == 96 and MO.ARCH["gru"]["layers"] == 2
        assert MO.ARCH["gru"]["bidirectional"] is False
        assert MO.ARCH["n_seq"] == 90
        assert MO.ARCH["head"]["out"] == 72
        net = MO.D135Net(n_static=50)
        out = net(torch.randn(2, SP.HIST, SP.N_SEQ), torch.randn(2, 50))
        assert out.shape == (2, 72) and torch.isfinite(out).all()
        sys.modules["spec"] = P
        PMO = _load("d135r6_ref_model", D135 / "model.py")
        pnet = PMO.D135Net(n_static=50)
        delta = MO.count_params(net) - PMO.count_params(pnet)
        assert delta == 3 * 96 * (90 - 30), "only the GRU input projection may grow"
        assert 120000 < MO.count_params(net) < 160000
    finally:
        for k in [k for k in sys.modules if k not in old]:
            del sys.modules[k]
        sys.modules.update({k: v for k, v in old.items() if k in ("spec",)})


def test_seeds_real_in_every_learning_part():
    src = _src("rocv.py")
    assert "for s in SEEDS" in src and "f{k}_s{s}.parquet" in src
    assert "torch.manual_seed" in (D135 / "runner.py").read_text()
    SP = sys.modules.get("d135r6_spec") or _load("d135r6_spec", R6 / "spec.py")
    assert tuple(SP.SEEDS) == (0, 1, 2)


def test_rocv_contract_ready_but_deferred():
    src = _src("rocv.py")
    assert "f{k}_s{s}.parquet" in src and "EXPOS_OUT" in src and '"rocv"' in src
    assert "len(rocv.FOLDS)" in src and 'freq="3h"' in src
    assert "valid_mask(origins, lo" in src, "fold-72h purge must gate fold training rows"
    assert "rocv_curves.json" in src
    assert "parents[2]" in src, "r6 rocv LB must resolve to leaderboard/"
    assert "screen" in src.lower(), "rocv must be gated on the 3-period screen"
    assert "seq_multiscale" in src, "rocv must train on the multiscale input"
    SP = sys.modules.get("d135r6_spec") or _load("d135r6_spec", R6 / "spec.py")
    assert tuple(SP.SEEDS) == (0, 1, 2)


def test_budget_and_threads():
    SP = sys.modules.get("d135r6_spec") or _load("d135r6_spec", R6 / "spec.py")
    assert SP.MIN_EVIDENCE_S == 600
    assert "MIN-10MIN-RULE" in (R6 / "spec.py").read_text()
    assert SP.TORCH_THREADS + SP.PREFETCH_THREADS <= 20


def test_owned_subdir_is_small():
    files = sorted(p.name for p in R6.iterdir() if p.suffix == ".py")
    assert len(files) <= 5, files
    assert set(files) == {"spec.py", "msinput.py", "runner.py", "rocv.py"}
