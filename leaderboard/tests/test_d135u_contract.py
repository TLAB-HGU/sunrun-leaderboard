"""Contract for D135-R11-budgetup (experiments/d135_t3seq/r11-budgetup/).

E0233-identical Transformer official-window runner with the single
prespecified budget-increase axis: MAX_EPOCHS 200 -> 7500, patience
12 -> 7500, per-seed GPU time floor 1800s (예산증량). LR stays 3e-4; no
warmup, no EMA, no ensembling; every seed keeps its own
predictions_official_s{seed}.parquet plus its curves_s{seed}.json.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

R11 = Path(__file__).resolve().parents[1] / "experiments" / "d135_t3seq" / "r11-budgetup"
D135 = R11.parent
R10 = D135 / "r10-stab"

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
                   "worker-t3j", "worker-t3k", "worker-d52", "worker-d59", "worker-d61"]
# Structural keys that must match r10-stab exactly (budget keys excluded).
STRUCT_KEYS = ("SEEDS", "HIST", "N_STRIP", "N_PLASMA", "N_SEQ", "D_MODEL",
               "NHEAD", "NLAYERS", "DIM_FF", "DROPOUT", "POS", "MLP_HIDDEN",
               "N_OUT", "LR", "BATCH", "VAL_FRAC",
               "TORCH_THREADS", "PREFETCH_THREADS",
               "CUTOFF", "OFFICIAL_ORIGINS", "OFFICIAL_FREQ", "OFFICIAL_N",
               "STATIC_DROP", "PROBES", "FAMILY_TAG", "AXES_TAG")


def _load(name, path):
    loc = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(loc)
    sys.modules[name] = mod
    loc.loader.exec_module(mod)
    return mod


def _src(name):
    return (R11 / name).read_text()


def test_hypothesis_tags_and_prespecified_budget_only():
    SP = _load("d135u11_spec", R11 / "spec.py")
    try:
        SP.check_tags()
        assert SP.FAMILY_TAG == "[family=nn]"
        assert SP.AXES_TAG == "[axes: model=transformer, opt=low-lr]"
        assert SP.HYPOTHESIS.startswith("[family=nn]")
        for cite in ("E0233", "X0213", "X0044", "X0046", "X0047"):
            assert cite in SP.HYPOTHESIS, f"missing citation {cite}"
        assert "2026-05-29" in SP.HYPOTHESIS
        assert SP.LR == 3e-4
        low = SP.HYPOTHESIS.lower()
        assert "3e-4" in SP.HYPOTHESIS
        assert "no warmup" in low and "no ema" in low and "no ensembl" in low
        assert "7500" in SP.HYPOTHESIS and "1800" in SP.HYPOTHESIS
        assert tuple(SP.SEEDS) == (0, 1, 2)
    finally:
        sys.modules.pop("d135u11_spec", None)


def test_structure_identical_to_e0233_except_budget():
    r10 = _load("d135u11_r10spec", R10 / "spec.py")
    sp = _load("d135u11_spec2", R11 / "spec.py")
    try:
        for k in STRUCT_KEYS:
            assert getattr(sp, k) == getattr(r10, k), f"structural drift on {k}"
        assert sp.LR == r10.LR == 3e-4, "LR must stay at the E0233 value"
        assert sp.MAX_EPOCHS_FULL > r10.MAX_EPOCHS_FULL, "budget must exceed E0233"
        assert sp.PATIENCE > r10.PATIENCE, "patience must exceed E0233"
        assert sp.SEED_TIME_FLOOR_S >= 1800
        assert not hasattr(r10, "SEED_TIME_FLOOR_S"), "floor is the new budget key"
        assert sp.MIN_EVIDENCE_S == 5400
        assert str(sp.SCRATCH).endswith("worker-t3l")
        assert str(r10.SCRATCH).endswith("worker-t3k")
    finally:
        sys.modules.pop("d135u11_r10spec", None)
        sys.modules.pop("d135u11_spec2", None)


def test_official_window_constants():
    SP = _load("d135u11_spec3", R11 / "spec.py")
    try:
        assert SP.CUTOFF == pd.Timestamp("2026-05-29", tz="UTC")
        assert SP.OFFICIAL_ORIGINS == ("2026-05-31 23:00", "2026-09-26 23:00")
        assert SP.OFFICIAL_FREQ == "h"
        grid = SP.official_origins()
        assert len(grid) == SP.OFFICIAL_N == 2833
        assert grid[0] == pd.Timestamp("2026-05-31 23:00", tz="UTC")
        assert (grid[1:] - grid[:-1] == pd.Timedelta("1h")).all()
    finally:
        sys.modules.pop("d135u11_spec3", None)


def test_scratch_owned_and_separated():
    SP = _load("d135u11_spec4", R11 / "spec.py")
    try:
        assert SP.SCRATCH.name == "worker-t3l"
        assert str(SP.SCRATCH) == str(SP.LB / "store/scratch/worker-t3l")
        assert (SP.LB / "expos" / "core.py").is_file()
        for f in ("spec.py", "model.py", "runner.py"):
            text = _src(f)
            for other in FOREIGN_SCRATCH:
                assert other not in text, f"{f} must not touch the {other} scratch"
    finally:
        sys.modules.pop("d135u11_spec4", None)


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
    assert "seed=sd" in src
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
    SP = _load("d135u11_spec5", R11 / "spec.py")
    try:
        assert SP.MIN_EVIDENCE_S == 5400
        assert SP.SEED_TIME_FLOOR_S == 1800
        assert SP.TORCH_THREADS + SP.PREFETCH_THREADS <= 20
        assert "REAL-BUDGET-CURVES" in _src("runner.py")
        assert "INTERNAL-SPLIT-EARLY-STOP" in _src("spec.py")
        assert "SP.LR == 3e-4" in _src("runner.py")
        assert "curves_s{sd}" in _src("runner.py"), "runner must write per-seed curves_s files"
        assert "time_budget_s" in _src("runner.py"), "runner must enforce the per-seed time floor"
    finally:
        sys.modules.pop("d135u11_spec5", None)


def test_owned_subdir_is_small():
    files = sorted(p.name for p in R11.iterdir() if p.suffix == ".py")
    assert len(files) <= 5, files
    assert set(files) == {"spec.py", "model.py", "runner.py"}


def test_train_mask_matches_cutoff_rule():
    pytest.importorskip("torch")
    SP = _load("d135u11_spec6", R11 / "spec.py")
    origins = pd.DatetimeIndex(["2026-05-28T00:00:00Z", "2026-05-28T03:00:00Z"])
    ymat = np.full((2, 72), 500.0, dtype=np.float32)
    sys.path.insert(0, str(D135))
    sys.path.insert(0, str(R11))
    old = dict(sys.modules)
    try:
        sys.modules["spec"] = SP
        sys.modules.pop("model", None)
        run = _load("d135u11_runner_mask", R11 / "runner.py")
    finally:
        for k in [k for k in sys.modules if k not in old]:
            del sys.modules[k]
        sys.modules.update({k: v for k, v in old.items() if k in ("spec", "model")})
        sys.path.remove(str(R11))
        sys.path.remove(str(D135))
    m = run.official_train_mask(origins, ymat)
    assert m.shape == (2, 72)
    assert bool(m[0, :23].all()) and not bool(m[0, 23:].any())
    assert bool(m[1, :20].all()) and not bool(m[1, 20:].any())
    bad = ymat.copy()
    bad[0, 0] = np.nan
    assert not run.official_train_mask(origins, bad)[0, 0]


def _load_r11_model():
    torch = pytest.importorskip("torch")
    SP = _load("d135u11_spec7", R11 / "spec.py")
    sys.path.insert(0, str(R11))
    old = dict(sys.modules)
    try:
        sys.modules["spec"] = SP
        sys.modules.pop("model", None)
        return _load("d135u11_model", R11 / "model.py"), SP, torch
    except Exception:
        for k in [k for k in sys.modules if k not in old]:
            del sys.modules[k]
        sys.modules.update({k: v for k, v in old.items() if k in ("spec", "model")})
        sys.path.remove(str(R11))
        raise


def test_causal_transformer_last_step_and_72_head():
    MO, SP, torch = _load_r11_model()
    try:
        assert MO.ARCH["encoder"]["kind"] == "causal-transformer"
        assert MO.ARCH["encoder"]["mask"] == "strict-upper-triangular-subsequent"
        assert MO.ARCH["encoder"]["readout"] == "last-step"
        assert MO.ARCH["family"] == "Transformer-enc-laststep-direct-72h"
        assert "bidirectional" not in _src("model.py")
        net = MO.D135Net(n_static=70)
        assert net.head[-1].out_features == 72
        assert MO.count_params(net) > 50000
        seq = torch.randn(4, SP.HIST, SP.N_SEQ)
        stat = torch.randn(4, 70)
        out = net(seq, stat)
        assert out.shape == (4, 72) and torch.isfinite(out).all()
        mask = net.enc.causal_mask
        expected = torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1)
        assert torch.equal(torch.isinf(mask), expected)
    finally:
        sys.path.remove(str(R11))
        for k in ("d135u11_model", "d135u11_spec7"):
            sys.modules.pop(k, None)


def test_seeds_really_vary_model_init():
    MO, SP, torch = _load_r11_model()
    try:
        outs = []
        for sd in (0, 1, 2):
            torch.manual_seed(sd)
            net = MO.D135Net(n_static=20)
            torch.manual_seed(999)
            outs.append(net(torch.randn(2, SP.HIST, SP.N_SEQ),
                            torch.randn(2, 20)).detach())
        assert not torch.equal(outs[0], outs[1])
        assert not torch.equal(outs[1], outs[2])
    finally:
        sys.path.remove(str(R11))
        for k in ("d135u11_model", "d135u11_spec7"):
            sys.modules.pop(k, None)
