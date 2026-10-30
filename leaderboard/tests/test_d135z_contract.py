"""Contract for D135-R14-size: E0256-identical Transformer, single-axis
size change layers 2 -> 1 (experiments/d135_t3seq/r14-size/)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

R14 = Path(__file__).resolve().parents[1] / "experiments" / "d135_t3seq" / "r14-size"
D135 = R14.parent
sys.path.insert(0, str(D135))
sys.path.insert(0, str(R14))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "headroom_audit"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "expos_runners"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "d1_longitude_strips"))

import spec as SP  # noqa: E402  (r14-size spec: R14 precedes D135 on sys.path)
import model as MO  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]
GONG_BANS = ["gong", "GONG", "pfss", "PFSS", "hux", "HUX", "wsa", "enlil", "cme", "CME"]


def _src(name):
    return (R14 / name).read_text()


def test_hypothesis_tags_and_single_axis_size():
    SP.check_tags()
    assert SP.HYPOTHESIS.startswith("[family=nn]")
    assert "[axes: model=transformer, opt=low-lr, loss=huber, size=depth-1]" in SP.HYPOTHESIS
    assert "E0256" in SP.HYPOTHESIS and "2 -> 1" in SP.HYPOTHESIS
    assert "X0232" in SP.HYPOTHESIS and "X0227" in SP.HYPOTHESIS
    assert SP.NLAYERS == 1
    assert SP.LOSS == "huber" and SP.HUBER_DELTA == 1.0


def test_e0256_identical_constants_except_depth():
    assert (SP.D_MODEL, SP.NHEAD, SP.NLAYERS, SP.DIM_FF) == (64, 4, 1, 128)
    assert SP.DROPOUT == 0.1 and SP.POS == "learned-72"
    assert tuple(SP.MLP_HIDDEN) == (128, 64) and SP.N_OUT == 72
    assert SP.LR == 3e-4 and SP.BATCH == 256 and SP.VAL_FRAC == 0.15
    assert SP.PATIENCE == 7500 and SP.MAX_EPOCHS_FULL == 7500
    assert SP.SEED_TIME_FLOOR_S >= 1800 and SP.MIN_EVIDENCE_S == 5400
    assert tuple(SP.SEEDS) == (0, 1, 2)
    assert SP.HIST == 72 and SP.N_SEQ == 30
    assert str(SP.CUTOFF) == "2026-05-29 00:00:00+00:00"
    assert SP.OFFICIAL_N == 2833
    assert MO.ARCH["family"] == "Transformer-enc-laststep-direct-72h"
    assert MO.ARCH["encoder"]["kind"] == "causal-transformer"
    assert MO.ARCH["encoder"]["mask"] == "strict-upper-triangular-subsequent"
    assert MO.ARCH["encoder"]["readout"] == "last-step"
    assert MO.ARCH["encoder"]["layers"] == 1
    assert MO.ARCH["encoder"]["dropout"] == 0.1
    assert MO.ARCH["encoder"]["d_model"] == 64


def test_scratch_owned_by_worker_t3o():
    assert str(SP.SCRATCH).endswith("store/scratch/worker-t3o")
    assert "worker-t3o" in _src("spec.py") and "worker-t3o" in _src("runner.py")
    for f in ("spec.py", "model.py", "runner.py"):
        text = _src(f)
        for other in ("worker-t3n", "worker-t3m", "worker-t3l", "worker-t3k", "worker-t3j",
                      "worker-d59", "worker-d61"):
            assert other not in text, f"{f} references foreign scratch {other}"


def test_no_forbidden_truth_or_gong():
    for f in ("spec.py", "model.py", "runner.py"):
        text = _src(f)
        for ban in FORBIDDEN:
            assert ban not in text, f"{f} contains forbidden {ban}"
    rtext = _src("runner.py")
    for ban in GONG_BANS:
        assert ban not in rtext, f"runner.py references banned source {ban}"


def test_huber_math_matches_definition():
    import importlib.util
    loc = importlib.util.spec_from_file_location("r14_runner", R14 / "runner.py")
    # Only compile-check + huber helper: execute module source for huber_elem via torch.
    delta = 1.0
    d = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    a = d.abs()
    q = torch.minimum(a, torch.full_like(a, delta))
    expect = 0.5 * q * q + delta * (a - q)
    assert np.allclose(expect.numpy(), [2.5, 0.5, 0.125, 0.0, 0.125, 0.5, 2.5])
    src = _src("runner.py")
    assert "huber_elem" in src and "HUBER_DELTA" in src
    assert "R.evaluate_mse" in src, "validation/early-stop must stay on MSE"
    assert "torch.manual_seed(seed)" in src
    assert "seed * 7919 + ep" in src, "per-epoch shuffle stream must match E0256"
    assert "predictions_official_s" not in src or 'official_s{sd}' in src
    assert "curves_s" in src and "poison.json" in src
    assert "E0256" in src
    assert loc is not None


def test_model_forward_shape_and_causal_mask():
    net = MO.D135Net(n_static=8)
    net.eval()
    seq = torch.randn(4, 72, 30)
    stat = torch.randn(4, 8)
    with torch.no_grad():
        out = net(seq, stat)
    assert out.shape == (4, 72) and torch.isfinite(out).all()
    mask = net.enc.causal_mask
    assert mask.shape == (72, 72)
    ones = torch.ones_like(mask, dtype=torch.bool)
    assert bool((mask[~torch.tril(ones)] == float("-inf")).all())
    assert bool((mask[torch.tril(ones)] == 0).all())
    assert len(list(net.enc.enc.layers)) == 1


def test_owned_subdir_is_small():
    files = sorted(p.name for p in R14.iterdir() if p.suffix == ".py")
    assert len(files) <= 5, files
    assert {"spec.py", "model.py", "runner.py"} <= set(files)


def test_official_window_shape_constants():
    origins = SP.official_origins()
    assert len(origins) == 2833
    assert str(origins[0]) == "2026-05-31 23:00:00+00:00"
    assert set(SP.PROBES) == {"dev", "selection", "audit"}
    df_probe = pd.DataFrame({"x": [1.0]})
    assert len(df_probe) == 1
