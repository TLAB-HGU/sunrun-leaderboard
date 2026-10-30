"""D142 depth-axis contract: static + synthetic checks (fast, no fits, no data)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d142_e0147depth"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_IDS = ["e0147-abl", "d141_e0147", "ablate", "worker-d76", "worker-t2no",
             "worker-d135", "worker-d133", "nn-"]
PINNED_W = {"1-6": {"e0003_line": 0.21536233464375223, "d33_hsplit": 0.3899845818930272,
                    "d45_lvlmean": 0.3946530834632206},
            "7-24": {"e0003_line": 0.3152013131693896, "d33_hsplit": 0.3436571856823209,
                     "d45_lvlmean": 0.3411415011482895},
            "25-48": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
                      "d45_lvlmean": 0.3333333333333333},
            "49-72": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
                      "d45_lvlmean": 0.3333333333333333}}


def _load(name):
    loc = importlib.util.spec_from_file_location(f"d142_{name}", EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(EXP.glob("*.py")))


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    assert len(files) <= 6, files
    assert {"spec.py", "depth_lib.py", "run_official.py"} <= set(files), files


def test_spec_identity_and_scope():
    SP = _load("spec")
    assert SP.DIRECTION == "D142" and SP.OWNER == "worker-t1d7"
    assert SP.KIND == "full" and SP.REFERENCE == "E0001"
    assert 1 <= len(SP.DEPTHS) <= 3 and tuple(SP.DEPTHS) == (3,)
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24
    assert SP.SCRATCH == "store/scratch/worker-t1d7"
    assert SP.FIT_NJOBS <= 20 and SP.WEIGHTS == PINNED_W


def test_single_axis_vs_recipe():
    SP = _load("spec")
    assert "max_depth" not in SP.BASE_PARAMS, "depth comes only from DEPTHS"
    import sys
    sys.path.insert(0, str(LB / "experiments/headroom_audit"))
    import info_ablation as ia
    for k, v in ia.PARAMS.items():
        if k in ("max_depth", "n_jobs", "random_state"):
            continue
        assert SP.BASE_PARAMS.get(k) == v, k
    assert SP.BASE_PARAMS.get("tree_method") == "hist"
    assert SP.BASE_PARAMS.get("device") == "cpu"


def test_no_forbidden_leak_or_banned_source():
    code = _code()
    for s in FORBIDDEN:
        assert s not in code, s
    assert "GONG" not in code
    for s in OTHER_IDS:
        assert s not in code, s
    for f in sorted(EXP.glob("*.py")):
        t = f.read_text()
        assert "core.freeze" not in t and "official-score" not in t
        for w in ("freeze", "submit", "upload"):
            assert w not in t.lower(), (f.name, w)


def test_seed_actually_propagates():
    DL = _load("depth_lib")
    SP = _load("spec")
    blocks = [DL.level_seeds(s) for s in SP.SEEDS]
    assert len({r for b in blocks for r in b}) == 9, blocks
    assert [DL.event_seed(s) for s in SP.SEEDS] == [b[1] for b in blocks]
    params = [DL.fit_params(SP.BASE_PARAMS, SP.DEPTHS[0], r) for b in blocks for r in b]
    assert all(p["max_depth"] == 3 for p in params)
    assert len({p["random_state"] for p in params}) == 9


def test_blend_convex_and_stitch_partition():
    DL = _load("depth_lib")
    DL.check_weights(_load("spec").WEIGHTS)
    rng = np.random.default_rng(0)
    n = 72 * 4
    h = np.tile(np.arange(1, 73), 4)
    mem = {m: rng.normal(400, 100, n) for m in DL.MEMBERS}
    out = DL.blend_per_lead(mem, PINNED_W, h)
    assert out.shape == (n,) and np.isfinite(out).all()
    band = (h >= 7) & (h <= 24)
    w = PINNED_W["7-24"]
    assert np.allclose(out[band], sum(w[m] * mem[m][band] for m in DL.MEMBERS))
    m1, m2, m3 = h <= 6, (h >= 7) & (h <= 24), h > 24
    s1 = {s: rng.normal(size=int(m1.sum())) for s in (0, 1, 2)}
    s2 = {s: rng.normal(size=int(m2.sum())) for s in (0, 1, 2)}
    long_all = rng.normal(size=n)
    tail = rng.normal(size=int(m3.sum()))
    st = DL.stitch_members(s1, s2, long_all, tail, m1, m2, m3, (0, 1, 2), 1)
    assert np.array_equal(st["e0003_line"], long_all)
    assert np.array_equal(st["d33_hsplit"][m1], s1[1])
    assert np.array_equal(st["d45_lvlmean"][m2], sum(s2[s] for s in (0, 1, 2)) / 3)
    assert np.array_equal(st["d45_lvlmean"][m3], tail)


def test_output_frame_schema():
    DL = _load("depth_lib")
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = DL.frame_predictions(te, np.full(72, 400.0))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert (fr["horizon_hours"].to_numpy() == np.arange(1, 73)).all()
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"
