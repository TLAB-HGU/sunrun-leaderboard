"""Contract for D135-R2 capacity single-axis redesign (experiments/d135_t3seq/r2/)."""
import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

R2 = Path(__file__).resolve().parents[1] / "experiments" / "d135_t3seq" / "r2"
D135 = R2.parent

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]
RECOMB_BANS = ["predictions_", "blend", "stacking", "moe", "expos/runs"]

IDENTITY_KEYS = ("SEEDS", "HIST", "N_STRIP", "N_PLASMA", "N_SEQ", "N_OUT",
                 "LR", "BATCH", "VAL_FRAC", "PATIENCE", "MAX_EPOCHS_FULL",
                 "MIN_EVIDENCE_S", "TORCH_THREADS", "PREFETCH_THREADS",
                 "PURGE_H", "DEV", "STATIC_DROP", "PROBES")
CAPACITY_KEYS = ("GRU_HIDDEN", "GRU_LAYERS", "MLP_HIDDEN")


def _load(name, path):
    loc = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(loc)
    sys.modules[name] = mod
    loc.loader.exec_module(mod)
    return mod


def _src(name):
    return (R2 / name).read_text()


def test_hypothesis_tags_and_cites():
    SP = _load("d135r2_spec", R2 / "spec.py")
    SP.check_tags()
    assert SP.FAMILY_TAG == "[family=nn]"
    assert SP.AXES_TAG == "[axes: model=gru-small, ens=seed-mean]"
    assert SP.HYPOTHESIS.startswith("[family=nn]")
    for cite in ("E0208", "E0121", "X0044", "X0046", "X0047", "X0052", "E0003"):
        assert cite in SP.HYPOTHESIS, f"missing citation {cite}"
    assert "seed-mean" in SP.HYPOTHESIS and "capacity" in SP.HYPOTHESIS.lower()


def test_single_axis_capacity_only():
    SP = sys.modules.get("d135r2_spec") or _load("d135r2_spec", R2 / "spec.py")
    P = _load("d135_parent_spec", D135 / "spec.py")
    for k in IDENTITY_KEYS:
        a, b = getattr(SP, k), getattr(P, k)
        assert a == b, f"{k} must be identical to E0208 ({a!r} != {b!r})"
    assert (SP.GRU_HIDDEN, SP.GRU_LAYERS, tuple(SP.MLP_HIDDEN)) == (32, 1, (32,))
    assert (P.GRU_HIDDEN, P.GRU_LAYERS, tuple(P.MLP_HIDDEN)) == (96, 2, (128, 64))
    assert str(SP.SCRATCH).endswith("store/scratch/worker-t3b")
    assert str(P.SCRATCH).endswith("store/scratch/worker-d135")


def test_scratch_owned_and_separated():
    SP = sys.modules.get("d135r2_spec") or _load("d135r2_spec", R2 / "spec.py")
    assert SP.SCRATCH.name == "worker-t3b"
    assert (SP.LB / "expos" / "core.py").is_file(), "r2 LB must resolve to leaderboard/"
    assert SP.SCRATCH.parent.name == "scratch"
    assert str(SP.SCRATCH) == str(SP.LB / "store/scratch/worker-t3b")
    for f in ("spec.py", "runner.py", "rocv.py"):
        text = _src(f)
        assert "worker-d135" not in text, f"{f} must not touch the E0208 scratch"
        assert "worker-d52" not in text and "worker-d59" not in text and "worker-d61" not in text


def test_no_forbidden_or_copy_or_window():
    for f in ("spec.py", "runner.py", "rocv.py"):
        text = _src(f)
        for s in FORBIDDEN:
            assert s not in text, f"{f} must not reference {s}"
        assert "2026-06" not in text, f"{f} must not reference the June window"
        assert "official" not in text.lower(), f"{f} must not write frozen-window predictions"
        for ban in COPY_BANS:
            assert ban not in text, f"{f} must reuse parent/strips, never vendor it"


def test_no_stored_prediction_reuse():
    for f in ("spec.py", "runner.py", "rocv.py"):
        text = _src(f)
        for s in RECOMB_BANS:
            assert s not in text, f"{f} must not recombine stored predictions ({s})"


def test_probe_dates_only_in_probes():
    SP = sys.modules.get("d135r2_spec") or _load("d135r2_spec", R2 / "spec.py")
    assert set(SP.PROBES) == {"dev", "selection", "audit"}
    for f in ("runner.py", "rocv.py"):
        assert "2025-12-04" not in _src(f) and "2026-03-04" not in _src(f)
        assert "2025-09-15" not in _src(f) and "2026-01-05" not in _src(f)


def test_wrapper_thin_and_pinned_to_r2_spec():
    src = _src("runner.py")
    for banned in ("def fit_gru", "def valid_mask", "def predict", "class ", "def poison_check"):
        assert banned not in src, f"wrapper must reuse parent logic ({banned})"
    assert "--cpu" in src and "check_tags" in src and "d135_parent_runner" in src
    mod = _load("d135r2_runner", R2 / "runner.py")
    assert mod.SP.GRU_HIDDEN == 32 and mod.SP.GRU_LAYERS == 1
    assert mod._PARENT.SP is mod.SP, "parent training code must run under the R2 spec"


def test_model_wiring_small_capacity_direct_72():
    src = (D135 / "model.py").read_text()
    assert "bidirectional=False" in src and "h[:, -1, :]" in src
    SP = sys.modules.get("d135r2_spec") or _load("d135r2_spec", R2 / "spec.py")
    old = dict(sys.modules)
    try:
        sys.modules["spec"] = SP
        MO = _load("d135r2_model", D135 / "model.py")
        assert MO.ARCH["gru"]["hidden"] == 32 and MO.ARCH["gru"]["layers"] == 1
        assert MO.ARCH["gru"]["bidirectional"] is False
        assert MO.ARCH["head"]["out"] == 72
        net = MO.D135Net(n_static=50)
        assert MO.count_params(net) < 20000, "R2 must cut E0208 129k params ~10x"
        out = net(torch.randn(2, SP.HIST, SP.N_SEQ), torch.randn(2, 50))
        assert out.shape == (2, 72) and torch.isfinite(out).all()
    finally:
        for k in [k for k in sys.modules if k not in old]:
            del sys.modules[k]
        sys.modules.update({k: v for k, v in old.items() if k in ("spec",)})


def test_rocv_contract_ready_but_deferred():
    src = _src("rocv.py")
    assert "f{k}_s{s}.parquet" in src and "EXPOS_OUT" in src and '"rocv"' in src
    assert "len(rocv.FOLDS)" in src and 'freq="3h"' in src
    assert "valid_mask(origins, lo" in src, "fold-72h purge must gate fold training rows"
    assert "rocv_curves.json" in src
    SP = sys.modules.get("d135r2_spec") or _load("d135r2_spec", R2 / "spec.py")
    assert tuple(SP.SEEDS) == (0, 1, 2)


def test_budget_and_threads():
    SP = sys.modules.get("d135r2_spec") or _load("d135r2_spec", R2 / "spec.py")
    assert SP.MIN_EVIDENCE_S == 600
    assert "MIN-10MIN-RULE" in (R2 / "spec.py").read_text()
    assert SP.TORCH_THREADS + SP.PREFETCH_THREADS <= 20


def test_owned_subdir_is_small():
    files = sorted(p.name for p in R2.iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert set(files) == {"spec.py", "runner.py", "rocv.py"}
