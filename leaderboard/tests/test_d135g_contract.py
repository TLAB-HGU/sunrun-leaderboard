"""Contract for D135-R12-reg Transformer official-window runner (experiments/d135_t3seq/r12-reg/).

Single-axis regularization (axis 2): E0234-identical backbone, dropout 0.0 -> 0.1
prespecified. Every other constant (arch, LR, budget, cutoff, early-stop) must
match the E0234 (r11-budgetup) spec exactly.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

R12 = Path(__file__).resolve().parents[1] / "experiments" / "d135_t3seq" / "r12-reg"
D135 = R12.parent
R11 = D135 / "r11-budgetup"

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]
# Stored-prediction recombination is stopped: the runner may WRITE per-seed
# official files but must never READ stored predictions back.
RECOMB_BANS = ["predictions.*read_parquet", "read_parquet.*predictions", "glob.*predictions",
               "blend", "stacking", "moe", "expos/runs"]
FOREIGN_SCRATCH = ["worker-d135", "worker-t3b", "worker-t3c", "worker-t3d",
                   "worker-t3e", "worker-t3f", "worker-t3g", "worker-t3h",
                   "worker-t3j", "worker-t3k", "worker-t3l",
                   "worker-d52", "worker-d59", "worker-d61"]


def _load(name, path):
    loc = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(loc)
    sys.modules[name] = mod
    loc.loader.exec_module(mod)
    return mod


def _src(name):
    return (R12 / name).read_text()


def test_hypothesis_tags_and_cites():
    SP = _load("d135g_spec", R12 / "spec.py")
    SP.check_tags()
    assert SP.FAMILY_TAG == "[family=nn]"
    assert SP.AXES_TAG == "[axes: model=transformer, opt=low-lr]"
    assert SP.HYPOTHESIS.startswith("[family=nn]")
    for cite in ("E0234", "E0233", "E0232", "X0228", "X0213", "X0044", "E0003", "E0147"):
        assert cite in SP.HYPOTHESIS, f"missing citation {cite}"
    assert "2026-05-29" in SP.HYPOTHESIS
    assert "dropout" in SP.HYPOTHESIS and "0.1" in SP.HYPOTHESIS
    assert "transformer" in SP.HYPOTHESIS.lower() and "encod" in SP.HYPOTHESIS.lower()
    assert (SP.D_MODEL, SP.NHEAD, SP.NLAYERS, SP.DIM_FF) == (64, 4, 2, 128)
    assert SP.DROPOUT == 0.1 and SP.POS == "learned-72"
    assert tuple(SP.SEEDS) == (0, 1, 2)


def test_official_window_constants():
    SP = _load("d135g_spec", R12 / "spec.py")
    assert SP.CUTOFF == pd.Timestamp("2026-05-29", tz="UTC")
    assert SP.OFFICIAL_ORIGINS == ("2026-05-31 23:00", "2026-09-26 23:00")
    assert SP.OFFICIAL_FREQ == "h"
    grid = SP.official_origins()
    assert len(grid) == SP.OFFICIAL_N == 2833
    assert grid[0] == pd.Timestamp("2026-05-31 23:00", tz="UTC")
    assert (grid[1:] - grid[:-1] == pd.Timedelta("1h")).all()
    assert (grid + pd.to_timedelta(72, unit="h") >= pd.Timestamp("2026-06-01", tz="UTC")).all()


def test_scratch_owned_and_separated():
    SP = _load("d135g_spec", R12 / "spec.py")
    assert SP.SCRATCH.name == "worker-t3m"
    assert str(SP.SCRATCH) == str(SP.LB / "store/scratch/worker-t3m")
    assert (SP.LB / "expos" / "core.py").is_file(), "r12 LB must resolve to leaderboard/"
    for f in ("spec.py", "model.py", "runner.py"):
        text = _src(f)
        for other in FOREIGN_SCRATCH:
            assert other not in text, f"{f} must not touch the {other} scratch"


def test_no_forbidden_or_copy():
    for f in ("spec.py", "model.py", "runner.py"):
        text = _src(f)
        for s in FORBIDDEN:
            assert s not in text, f"{f} must not reference {s}"
        for ban in COPY_BANS:
            assert ban not in text, f"{f} must reuse strips, never vendor it"
        assert "GONG" not in text.upper().replace("NO GONG", ""), f"{f} must not use GONG features"
        assert "DONKI" not in text, f"{f} must not use CME catalogs"


def test_no_stored_prediction_reuse():
    import re
    for f in ("spec.py", "model.py", "runner.py"):
        text = _src(f)
        for s in RECOMB_BANS:
            assert not re.search(s, text), f"{f} must not recombine stored predictions ({s})"
        assert "predictions_official_s" in text or f != "runner.py", \
            "runner must write per-seed official files"


def test_per_seed_files_not_blended():
    src = _src("runner.py")
    assert "official_s{sd}" in src or "official_s" in src
    assert "for sd in seeds" in src
    assert "torch.manual_seed" in src or "manual_seed(seed" in src or "seed=sd" in src
    # Each seed keeps its own predictions_official_s{seed}.parquet; a single
    # predictions_official.parquet must never be written.
    assert "predictions_official.parquet" not in src
    assert ', "official")' not in src


def test_cutoff_and_internal_split_discipline():
    src = _src("runner.py")
    assert "2026-05-29" in src or "CUTOFF" in src
    assert "official_train_mask" in src
    assert "val_hi < SP.CUTOFF" in src or "strictly before the cutoff" in src
    assert "poison_check" in src and "use_cache=False" in src
    assert "R.poison_check" in src, "must reuse the real future-poison test (with positive control)"
    assert "learning_curves.json" in src and "poison.json" in src and "manifest.json" in src


def test_threads_and_budget():
    SP = _load("d135g_spec", R12 / "spec.py")
    assert SP.MIN_EVIDENCE_S == 5400
    assert SP.TORCH_THREADS + SP.PREFETCH_THREADS <= 20
    assert SP.MAX_EPOCHS_FULL == 7500 and SP.PATIENCE == 7500
    assert SP.SEED_TIME_FLOOR_S == 1800
    assert SP.LR == 3e-4
    assert "REAL-BUDGET-CURVES" in _src("runner.py")
    assert "INTERNAL-SPLIT-EARLY-STOP" in _src("spec.py")


def test_owned_subdir_is_small():
    files = sorted(p.name for p in R12.iterdir() if p.suffix == ".py")
    assert len(files) <= 5, files
    assert set(files) == {"spec.py", "model.py", "runner.py"}


def test_single_axis_only_dropout_vs_e0234():
    """The ONLY behavioral delta versus the E0234 (r11-budgetup) spec is DROPOUT."""
    SP12 = _load("d135g_spec", R12 / "spec.py")
    SP11 = _load("d135r11_spec_ref", R11 / "spec.py")
    assert SP11.DROPOUT == 0.0, "E0234 baseline must be dropout-free"
    assert SP12.DROPOUT == 0.1
    for attr in ("SEEDS", "HIST", "N_STRIP", "N_PLASMA", "N_SEQ", "D_MODEL", "NHEAD",
                 "NLAYERS", "DIM_FF", "POS", "MLP_HIDDEN", "N_OUT", "LR", "BATCH",
                 "VAL_FRAC", "PATIENCE", "MAX_EPOCHS_FULL", "SEED_TIME_FLOOR_S",
                 "MIN_EVIDENCE_S", "TORCH_THREADS", "PREFETCH_THREADS",
                 "OFFICIAL_ORIGINS", "OFFICIAL_FREQ", "OFFICIAL_N",
                 "STATIC_DROP", "PROBES", "FAMILY_TAG", "AXES_TAG"):
        assert getattr(SP12, attr) == getattr(SP11, attr), f"{attr} must stay E0234-identical"
    assert SP12.CUTOFF == SP11.CUTOFF == pd.Timestamp("2026-05-29", tz="UTC")
    assert not hasattr(SP12, "WEIGHT_DECAY") and not hasattr(SP12, "WD"), \
        "weight decay must stay 0 (dropout is the single axis)"
    parent_opt = (D135 / "runner.py").read_text()
    assert "weight_decay" not in parent_opt, "optimizer must stay Adam(lr) without weight decay"
    assert "AdamW" not in _src("model.py") and "AdamW" not in _src("runner.py")


def test_train_mask_matches_cutoff_rule():
    pytest.importorskip("torch")
    SP = _load("d135g_spec", R12 / "spec.py")
    origins = pd.DatetimeIndex(["2026-05-28T00:00:00Z", "2026-05-28T03:00:00Z"])
    ymat = np.full((2, 72), 500.0, dtype=np.float32)
    sys.path.insert(0, str(D135))
    sys.path.insert(0, str(R12))
    old = dict(sys.modules)
    try:
        sys.modules["spec"] = SP
        sys.modules.pop("model", None)
        run = _load("d135g_runner_mask", R12 / "runner.py")
    finally:
        for k in [k for k in sys.modules if k not in old]:
            del sys.modules[k]
        sys.modules.update({k: v for k, v in old.items() if k in ("spec", "model")})
        sys.path.remove(str(R12))
        sys.path.remove(str(D135))
    m = run.official_train_mask(origins, ymat)
    assert m.shape == (2, 72)
    # origin + h is supervised iff target < 2026-05-29: h<=23 for the first
    # origin (indices :23), h<=20 for the second (indices :20).
    assert bool(m[0, :23].all()) and not bool(m[0, 23:].any())
    assert bool(m[1, :20].all()) and not bool(m[1, 20:].any())
    bad = ymat.copy()
    bad[0, 0] = np.nan
    assert not run.official_train_mask(origins, bad)[0, 0]


def _load_r12_model():
    torch = pytest.importorskip("torch")
    SP = _load("d135g_spec", R12 / "spec.py")
    sys.path.insert(0, str(R12))
    old = dict(sys.modules)
    try:
        sys.modules["spec"] = SP
        sys.modules.pop("model", None)
        return _load("d135g_model", R12 / "model.py"), SP, torch
    except Exception:
        for k in [k for k in sys.modules if k not in old]:
            del sys.modules[k]
        sys.modules.update({k: v for k, v in old.items() if k in ("spec", "model")})
        sys.path.remove(str(R12))
        raise


def test_causal_transformer_last_step_and_72_head():
    MO, SP, torch = _load_r12_model()
    try:
        assert MO.ARCH["encoder"]["kind"] == "causal-transformer"
        assert MO.ARCH["encoder"]["mask"] == "strict-upper-triangular-subsequent"
        assert MO.ARCH["encoder"]["readout"] == "last-step"
        assert MO.ARCH["encoder"]["dropout"] == 0.1
        assert "bidirectional" not in _src("model.py")
        assert "Attention" not in _src("model.py") or "Transformer" in _src("model.py")
        net = MO.D135Net(n_static=70)
        assert net.head[-1].out_features == 72
        assert MO.count_params(net) > 50000
        seq = torch.randn(4, SP.HIST, SP.N_SEQ)
        stat = torch.randn(4, 70)
        out = net(seq, stat)
        assert out.shape == (4, 72) and torch.isfinite(out).all()
        # The causal mask is strictly upper-triangular: position t never
        # attends to positions > t.
        mask = net.enc.causal_mask
        expected = torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1)
        assert torch.equal(torch.isinf(mask), expected)
    finally:
        sys.path.remove(str(R12))
        for k in ("d135g_model", "d135g_spec"):
            sys.modules.pop(k, None)


def test_seeds_really_vary_model_init():
    MO, SP, torch = _load_r12_model()
    try:
        outs = []
        for sd in (0, 1, 2):
            torch.manual_seed(sd)
            net = MO.D135Net(n_static=20)
            torch.manual_seed(999)
            net.eval()
            outs.append(net(torch.randn(2, SP.HIST, SP.N_SEQ),
                            torch.randn(2, 20)).detach())
        assert not torch.equal(outs[0], outs[1])
        assert not torch.equal(outs[1], outs[2])
    finally:
        sys.path.remove(str(R12))
        for k in ("d135g_model", "d135g_spec"):
            sys.modules.pop(k, None)


def test_dropout_live_in_train_only():
    """Dropout 0.1 must perturb train-mode forwards and vanish in eval mode."""
    MO, SP, torch = _load_r12_model()
    try:
        torch.manual_seed(0)
        net = MO.D135Net(n_static=20)
        seq = torch.randn(2, SP.HIST, SP.N_SEQ)
        stat = torch.randn(2, 20)
        net.train()
        a = net(seq, stat).detach()
        b = net(seq, stat).detach()
        assert not torch.equal(a, b), "dropout must be live in train mode"
        net.eval()
        c = net(seq, stat).detach()
        d = net(seq, stat).detach()
        assert torch.equal(c, d), "dropout must be off in eval mode"
    finally:
        sys.path.remove(str(R12))
        for k in ("d135g_model", "d135g_spec"):
            sys.modules.pop(k, None)
