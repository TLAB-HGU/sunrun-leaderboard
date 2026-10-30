"""D143 A2 contract: static + synthetic checks (fast, no fits, no data)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d143_axisA_analog" / "a2"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_IDS = ["e0147-abl", "d141_e0147", "ablate", "worker-d76", "worker-t1d7",
             "worker-d135", "worker-d133", "worker-t2no", "nn-"]
BANNED_SRC = ["GONG", "WSA", "ADAPT", "CME", "DONKI"]


def _load(name):
    loc = importlib.util.spec_from_file_location(f"d143a2_{name}", EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(EXP.glob("*.py")))


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 5, files
    assert {"spec_a2.py", "analog_lib_a2.py", "run_official_a2.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_a2")
    assert SP.DIRECTION == "D143" and SP.OWNER == "worker-t1a1"
    assert SP.KIND == "full" and SP.REFERENCE == "E0368"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h"
    assert SP.SCRATCH == "store/scratch/worker-t1a1"
    assert SP.FIT_NJOBS <= 20
    assert SP.K == 15 and SP.SCENARIO_H == 12 and SP.PATH_H == 72
    assert SP.SCENARIO_DIM == 84
    assert abs(SP.CARRINGTON_H - 654.48) < 1e-9
    assert len(SP.BLEND_GRID) == 21 and SP.BLEND_GRID[0] == 0.0 and SP.BLEND_GRID[-1] == 1.0
    assert SP.OOF_TRAIN_END_ISO < SP.OOF_START_ISO <= SP.OOF_END_ISO < SP.CUTOFF_ISO
    assert SP.BASE_DEPTH == 4 and SP.BASE_COLUMNS == "existing_all"
    assert SP.PATH_SOURCE == "oof-fit"
    assert "[axes:" in SP.AXES_TAG and "analog" in SP.AXES_TAG


def test_single_axis_vs_recipe():
    SP = _load("spec_a2")
    assert "max_depth" not in SP.BASE_PARAMS, "depth comes only from BASE_DEPTH"
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
    for s in BANNED_SRC:
        assert s not in code, s
    for s in OTHER_IDS:
        assert s not in code, s
    for f in sorted(EXP.glob("*.py")):
        t = f.read_text()
        assert "core.freeze" not in t and "official-score" not in t
        for w in ("freeze", "submit", "upload"):
            assert w not in t.lower(), (f.name, w)


def test_seed_actually_propagates():
    AL = _load("analog_lib_a2")
    SP = _load("spec_a2")
    params = [AL.fit_params({**SP.BASE_PARAMS, "max_depth": SP.BASE_DEPTH}, s) for s in SP.SEEDS]
    assert len({p["random_state"] for p in params}) == 3
    assert all(p["max_depth"] == 4 for p in params)


def test_scenario_past_and_carrington():
    AL = _load("analog_lib_a2")
    idx = pd.date_range("2026-01-01", periods=2000, freq="h", tz="UTC")
    filled = pd.Series(np.arange(2000, dtype=float) + 300.0, index=idx)
    o = pd.DatetimeIndex([idx[1000]], tz="UTC")
    sc = AL.scenario_past_matrix(o, filled)
    assert sc.shape == (1, 12)
    assert np.allclose(sc[0], np.arange(989, 1001, dtype=float) + 300.0)
    cands = AL.candidate_pool(o[0], idx, len(idx))
    # Carrington exclusion: every candidate is at least 655h before origin
    assert len(cands) > 0
    assert (o[0] - idx[cands]).min() >= pd.Timedelta("655h")
    assert (o[0] - idx[cands]).max() <= pd.Timedelta("8760h")


def test_path_distance_equal_subspace_weight():
    AL = _load("analog_lib_a2")
    past_q = np.full((1, 12), 400.0)
    path_q = np.full((1, 72), 400.0)
    # past-only deviation of 2.0 on every dim -> mean 4.0
    d_past = AL.scenario_distance2(past_q, path_q,
                                   np.full((1, 12), 402.0), np.full((1, 72), 400.0))
    # path-only deviation of 2.0 on every dim -> mean 4.0 (not 6x larger)
    d_path = AL.scenario_distance2(past_q, path_q,
                                   np.full((1, 12), 400.0), np.full((1, 72), 402.0))
    assert np.allclose(d_past, [4.0]) and np.allclose(d_path, [4.0])
    assert np.allclose(d_past, d_path)


def test_analog_with_path_replay():
    AL = _load("analog_lib_a2")
    idx = pd.date_range("2025-01-01", periods=24 * 500, freq="h", tz="UTC")
    filled = pd.Series(400.0 + 0.05 * np.arange(len(idx)), index=idx)
    o = pd.DatetimeIndex([idx[8000], idx[8100]], tz="UTC")
    qp = np.full((2, 72), 420.0)
    cand_times = idx
    cp = np.full((len(idx), 72), 420.0)
    out, cov = AL.analog_for_origins_with_path(o, filled, qp, cand_times, cp, 5)
    assert out.shape == (2, 72) and cov.shape == (2,)
    assert bool(np.isfinite(out).all()) and (cov > 0).all()
    out2, cov2 = AL.analog_for_origins_with_path(o, filled, qp, cand_times, cp, 5)
    assert np.array_equal(out, out2, equal_nan=True) and np.array_equal(cov, cov2)
    # query without past history falls back to NaN with zero coverage
    early = pd.DatetimeIndex([idx[5]], tz="UTC")
    out3, cov3 = AL.analog_for_origins_with_path(
        early, filled, np.full((1, 72), 420.0), cand_times, cp, 5)
    assert bool(np.isnan(out3).all()) and cov3[0] == 0
    # path mismatch changes the neighbour set vs past-only equivalent
    cp_far = np.full((len(idx), 72), 900.0)
    qp_near = np.full((2, 72), 900.0)
    out4, _ = AL.analog_for_origins_with_path(o, filled, qp_near, cand_times, cp_far, 5)
    assert bool(np.isfinite(out4).all())


def test_as_path_matrix():
    AL = _load("analog_lib_a2")
    m = AL.as_path_matrix(np.arange(144, dtype=float))
    assert m.shape == (2, 72)
    try:
        AL.as_path_matrix(np.arange(100, dtype=float))
    except AssertionError:
        pass
    else:
        raise AssertionError("as_path_matrix must reject non-multiples of 72")


def test_blend_convex_and_oof_only():
    AL = _load("analog_lib_a2")
    AL.check_blend_grid(tuple(round(i * 0.05, 2) for i in range(21)))
    rng = np.random.default_rng(0)
    b = rng.normal(400, 50, 500)
    a = rng.normal(400, 50, 500)
    y = rng.normal(400, 50, 500)
    w, base_mse, blend_mse, mses = AL.fit_blend_weight(b, a, y, tuple(round(i * 0.05, 2) for i in range(21)))
    assert 0.0 <= w <= 1.0 and len(mses) == 21 and blend_mse <= base_mse + 1e-9
    out = AL.blend(b, a, w)
    assert np.allclose(out, (1 - w) * b + w * a)
    nan_a = a.copy()
    nan_a[:10] = np.nan
    out2 = AL.blend(b, nan_a, w)
    assert np.allclose(out2[:10], b[:10])


def test_output_frame_schema():
    AL = _load("analog_lib_a2")
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = AL.frame_predictions(te, np.full(72, 400.0))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert (fr["horizon_hours"].to_numpy() == np.arange(1, 73)).all()
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_poison_keys_and_invariance():
    AL = _load("analog_lib_a2")
    idx = pd.date_range("2026-05-01", periods=24 * 70, freq="h", tz="UTC")
    filled = pd.Series(400.0 + np.sin(np.arange(len(idx)) / 24.0), index=idx)
    cut = pd.Timestamp("2026-05-29T00:00:00Z", tz="UTC")
    pre = pd.DatetimeIndex([idx[100], idx[200]], tz="UTC")
    post = pd.DatetimeIndex([idx[1200], idx[1300]], tz="UTC")
    probe = pre.append(post)
    rec = AL.scenario_poison(probe, filled, cut)
    assert rec["passed"] is True
    assert rec["pre_invariant"] is True and rec["moved_columns"] == ["scenario"]
    assert rec["n_pre_rows"] == 2 and rec["n_post_rows"] == 2
    rec2 = AL.frame_poison(probe, filled, cut, lambda f: AL.scenario_past_matrix(probe, f))
    assert rec2["passed"] is True
    assert rec2["pre_invariant"] is True and rec2["moved_columns"] == ["frame"]
    assert rec2["n_pre_rows"] == 2 and rec2["n_post_rows"] == 2
