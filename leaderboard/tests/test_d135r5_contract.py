"""Contract for D135-R5 encoder single-axis redesign (experiments/d135_t3seq/r5-trans/)."""
import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

R5 = Path(__file__).resolve().parents[1] / "experiments" / "d135_t3seq" / "r5-trans"
D135 = R5.parent

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]
RECOMB_BANS = ["predictions_", "blend", "stacking", "moe", "expos/runs"]

IDENTITY_KEYS = ("SEEDS", "HIST", "N_STRIP", "N_PLASMA", "N_SEQ", "N_OUT",
                 "MLP_HIDDEN",
                 "LR", "BATCH", "VAL_FRAC", "PATIENCE", "MAX_EPOCHS_FULL",
                 "MIN_EVIDENCE_S", "TORCH_THREADS", "PREFETCH_THREADS",
                 "PURGE_H", "DEV", "STATIC_DROP", "PROBES")


def _load(name, path):
    loc = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(loc)
    sys.modules[name] = mod
    loc.loader.exec_module(mod)
    return mod


def _src(name):
    return (R5 / name).read_text()


def test_hypothesis_tags_and_cites():
    SP = _load("d135r5_spec", R5 / "spec.py")
    SP.check_tags()
    assert SP.FAMILY_TAG == "[family=nn]"
    assert SP.AXES_TAG == "[axes: model=transformer, ens=seed-mean]"
    assert SP.HYPOTHESIS.startswith("[family=nn]")
    for cite in ("E0215", "E0208", "E0210", "E0213", "E0121", "X0044", "X0046",
                 "X0047", "X0052", "E0003"):
        assert cite in SP.HYPOTHESIS, f"missing citation {cite}"
    assert "seed-mean" in SP.HYPOTHESIS
    assert "transformer" in SP.HYPOTHESIS.lower() and "encod" in SP.HYPOTHESIS.lower()
    assert (SP.D_MODEL, SP.NHEAD, SP.NLAYERS, SP.DIM_FF) == (64, 4, 2, 128)
    assert SP.DROPOUT == 0.0 and SP.POS == "learned-72"


def test_single_axis_encoder_only():
    SP = sys.modules.get("d135r5_spec") or _load("d135r5_spec", R5 / "spec.py")
    P = _load("d135_parent_spec", D135 / "spec.py")
    for k in IDENTITY_KEYS:
        a, b = getattr(SP, k), getattr(P, k)
        assert a == b, f"{k} must be identical to E0208 ({a!r} != {b!r})"
    assert (P.GRU_HIDDEN, P.GRU_LAYERS) == (96, 2), "E0208 encoder baseline"
    assert not hasattr(SP, "GRU_HIDDEN") and not hasattr(SP, "GRU_LAYERS")
    assert getattr(P, "POOL", None) is None, "E0208 has no readout-pool axis"
    assert getattr(P, "HEAD_BANDS", None) is None, "E0208 has no band-head axis"
    assert getattr(SP, "POOL", None) is None, "R5 keeps the last-step readout"
    assert getattr(SP, "HEAD_BANDS", None) is None, "R5 keeps the single head"
    assert tuple(SP.MLP_HIDDEN) == (128, 64)
    assert str(SP.SCRATCH).endswith("store/scratch/worker-t3e")
    assert str(P.SCRATCH).endswith("store/scratch/worker-d135")


def test_scratch_owned_and_separated():
    SP = sys.modules.get("d135r5_spec") or _load("d135r5_spec", R5 / "spec.py")
    assert SP.SCRATCH.name == "worker-t3e"
    assert (SP.LB / "expos" / "core.py").is_file(), "r5 LB must resolve to leaderboard/"
    assert SP.SCRATCH.parent.name == "scratch"
    assert str(SP.SCRATCH) == str(SP.LB / "store/scratch/worker-t3e")
    for f in ("spec.py", "model.py", "runner.py", "rocv.py"):
        text = _src(f)
        assert "worker-d135" not in text, f"{f} must not touch the E0208 scratch"
        assert "worker-t3b" not in text, f"{f} must not touch the R2 scratch"
        assert "worker-t3c" not in text, f"{f} must not touch the R3 scratch"
        assert "worker-t3d" not in text, f"{f} must not touch the R4 scratch"
        assert "worker-d52" not in text and "worker-d59" not in text and "worker-d61" not in text


def test_no_forbidden_or_copy_or_window():
    for f in ("spec.py", "model.py", "runner.py", "rocv.py"):
        text = _src(f)
        for s in FORBIDDEN:
            assert s not in text, f"{f} must not reference {s}"
        assert "2026-06" not in text, f"{f} must not reference the June window"
        assert "official" not in text.lower(), f"{f} must not write frozen-window predictions"
        for ban in COPY_BANS:
            assert ban not in text, f"{f} must reuse parent/strips, never vendor it"


def test_no_stored_prediction_reuse():
    for f in ("spec.py", "model.py", "runner.py", "rocv.py"):
        text = _src(f)
        for s in RECOMB_BANS:
            assert s not in text, f"{f} must not recombine stored predictions ({s})"


def test_probe_dates_only_in_probes():
    SP = sys.modules.get("d135r5_spec") or _load("d135r5_spec", R5 / "spec.py")
    assert set(SP.PROBES) == {"dev", "selection", "audit"}
    for f in ("runner.py", "rocv.py"):
        assert "2025-12-04" not in _src(f) and "2026-03-04" not in _src(f)
        assert "2025-09-15" not in _src(f) and "2026-01-05" not in _src(f)


def test_wrapper_thin_and_pinned_to_r5_spec_and_model():
    src = _src("runner.py")
    for banned in ("def fit_gru", "def valid_mask", "def predict", "class ", "def poison_check"):
        assert banned not in src, f"wrapper must reuse parent logic ({banned})"
    assert "--cpu" in src and "check_tags" in src and "d135_parent_runner" in src
    assert "transformer" in src.lower()
    # Pin the bare `import spec` / `import model` inside the wrapper to the R5
    # modules: another contract test in the same pytest process may have left
    # a foreign `spec`/`model` in sys.modules (import cache wins over path).
    _SP = sys.modules.get("d135r5_spec") or _load("d135r5_spec", R5 / "spec.py")
    _old_spec, _old_model = sys.modules.get("spec"), sys.modules.get("model")
    try:
        sys.modules["spec"] = _SP
        sys.modules.pop("model", None)
        _load("d135r5_runner", R5 / "runner.py")
    finally:
        if _old_spec is None:
            sys.modules.pop("spec", None)
        else:
            sys.modules["spec"] = _old_spec
        if _old_model is None:
            sys.modules.pop("model", None)
        else:
            sys.modules["model"] = _old_model
    mod = sys.modules["d135r5_runner"]
    assert (mod.SP.D_MODEL, mod.SP.NHEAD, mod.SP.NLAYERS) == (64, 4, 2)
    assert mod._PARENT.SP is mod.SP, "parent training code must run under the R5 spec"
    assert mod._PARENT.MO is mod.MO, "parent training loop must optimize the R5 Transformer model"
    assert mod.MO.ARCH["encoder"]["kind"] == "causal-transformer"


def test_model_wiring_transformer_causal_direct_72():
    src = _src("model.py")
    assert "nn.GRU" not in src and "GRUEncoder" not in src, "R5 must not contain any recurrent encoder"
    assert "AttnPool" not in src and "ModuleList" not in src and "band" not in src.lower()
    assert "TransformerEncoder" in src
    assert "torch.triu" in src, "causal mask must be an upper-triangular construction"
    assert "h[:, -1, :]" in src or "[:, -1, :]" in src, "R5 keeps the last-step readout"
    assert "bidirectional" not in src
    SP = sys.modules.get("d135r5_spec") or _load("d135r5_spec", R5 / "spec.py")
    old = dict(sys.modules)
    try:
        sys.modules["spec"] = SP
        MO = _load("d135r5_model", R5 / "model.py")
        assert MO.ARCH["encoder"]["d_model"] == 64 and MO.ARCH["encoder"]["nhead"] == 4
        assert MO.ARCH["encoder"]["layers"] == 2 and MO.ARCH["encoder"]["dim_ff"] == 128
        assert MO.ARCH["encoder"]["mask"] == "strict-upper-triangular-subsequent"
        assert MO.ARCH["encoder"]["readout"] == "last-step"
        assert MO.ARCH["head"]["out"] == 72
        net = MO.D135Net(n_static=50)
        assert 80000 < MO.count_params(net) < 120000
        out = net(torch.randn(2, SP.HIST, SP.N_SEQ), torch.randn(2, 50))
        assert out.shape == (2, 72) and torch.isfinite(out).all()
        # Causal mask values: -inf strictly above the diagonal, 0 on/below.
        m = net.enc.causal_mask
        assert m.shape == (SP.HIST, SP.HIST)
        assert torch.isinf(m[0, 1:]).all() and (m[0, 1:] < 0).all()
        assert (m.diag() == 0).all() and (m[1:, 0] == 0).all()
        # Seeds govern every learning part: same seed -> identical init.
        torch.manual_seed(0)
        a = MO.D135Net(n_static=10)
        torch.manual_seed(0)
        b = MO.D135Net(n_static=10)
        for (k1, v1), (k2, v2) in zip(a.state_dict().items(), b.state_dict().items()):
            assert k1 == k2 and torch.equal(v1, v2), f"seed 0 must reproduce {k1}"
        torch.manual_seed(1)
        c = MO.D135Net(n_static=10)
        assert any(not torch.equal(v1, v3)
                   for v1, v3 in zip(a.state_dict().values(), c.state_dict().values()))
        # Functional causality: perturbing hour 5 moves steps >= 5 only.
        enc = MO.TransEncoder()
        enc.eval()
        with torch.no_grad():
            x1 = torch.randn(1, SP.HIST, SP.N_SEQ)
            x2 = x1.clone()
            x2[0, 5, :] += 10.0
            h1, h2 = enc.encode(x1), enc.encode(x2)
            assert torch.allclose(h1[:, :5, :], h2[:, :5, :], atol=1e-5)
            assert not torch.allclose(h1[:, 6:, :], h2[:, 6:, :], atol=1e-5)
    finally:
        for k in [k for k in sys.modules if k not in old]:
            del sys.modules[k]
        sys.modules.update({k: v for k, v in old.items() if k in ("spec",)})


def test_seeds_real_in_every_learning_part():
    src = _src("rocv.py")
    assert "for s in SEEDS" in src and "f{k}_s{s}.parquet" in src
    assert "torch.manual_seed" in (D135 / "runner.py").read_text()
    SP = sys.modules.get("d135r5_spec") or _load("d135r5_spec", R5 / "spec.py")
    assert tuple(SP.SEEDS) == (0, 1, 2)


def test_rocv_contract_ready_but_deferred():
    src = _src("rocv.py")
    assert "f{k}_s{s}.parquet" in src and "EXPOS_OUT" in src and '"rocv"' in src
    assert "len(rocv.FOLDS)" in src and 'freq="3h"' in src
    assert "valid_mask(origins, lo" in src, "fold-72h purge must gate fold training rows"
    assert "rocv_curves.json" in src
    assert "parents[2]" in src, "r5 rocv LB must resolve to leaderboard/"
    assert "screen" in src.lower(), "rocv must be gated on the 3-period screen"
    SP = sys.modules.get("d135r5_spec") or _load("d135r5_spec", R5 / "spec.py")
    assert tuple(SP.SEEDS) == (0, 1, 2)


def test_budget_and_threads():
    SP = sys.modules.get("d135r5_spec") or _load("d135r5_spec", R5 / "spec.py")
    assert SP.MIN_EVIDENCE_S == 600
    assert "MIN-10MIN-RULE" in (R5 / "spec.py").read_text()
    assert SP.TORCH_THREADS + SP.PREFETCH_THREADS <= 20


def test_owned_subdir_is_small():
    files = sorted(p.name for p in R5.iterdir() if p.suffix == ".py")
    assert len(files) <= 5, files
    assert set(files) == {"spec.py", "model.py", "runner.py", "rocv.py"}
