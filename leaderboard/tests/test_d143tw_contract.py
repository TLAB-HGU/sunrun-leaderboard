"""D143 TW contract: static + synthetic checks (fast, no fits, no data)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d143_axisA_analog" / "twokey"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_IDS = ["e0147-abl", "d141_e0147", "ablate", "worker-d76", "worker-t1d7",
             "worker-t1a1", "worker-t1a2", "worker-t1a3", "worker-t1a5",
             "worker-t1a6", "worker-t1a7", "worker-d135", "worker-d133",
             "worker-t2no", "nn-"]
BANNED_SRC = ["GONG", "WSA", "ADAPT", "CME", "DONKI"]


def _load(name):
    loc = importlib.util.spec_from_file_location(f"d143tw_{name}", EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(EXP.glob("*.py")))


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 6, files
    assert {"spec_tw.py", "resid_lib_tw.py", "run_official_tw.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_tw")
    assert SP.DIRECTION == "D143" and SP.OWNER == "worker-t1a8"
    assert SP.KIND == "full" and SP.REFERENCE == "E0421"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h"
    assert SP.SCRATCH == "store/scratch/worker-t1a8"
    assert SP.FIT_NJOBS <= 20
    assert SP.K == 15
    assert SP.POINT_STAT == "mean"
    assert SP.SCENARIO_H == 12 and SP.ERR_H == 12 and SP.PATH_H == 72
    assert SP.SCENARIO_DIM == 96
    assert abs(SP.CARRINGTON_H - 654.48) < 1e-9
    assert len(SP.BLEND_GRID) == 21 and SP.BLEND_GRID[0] == 0.0 and SP.BLEND_GRID[-1] == 1.0
    assert len(SP.ALPHA_GRID) >= 2
    assert all(0.0 < float(a) < 1.0 for a in SP.ALPHA_GRID)
    assert any(abs(float(a) - 1.0 / 3.0) < 1e-9 for a in SP.ALPHA_GRID)
    assert "shape" in SP.KEY_SHAPE_DEF.lower() and "72" in SP.KEY_SHAPE_DEF
    assert "error" in SP.KEY_STATE_DEF.lower() or "state" in SP.KEY_STATE_DEF.lower()
    assert SP.OOF_TRAIN_END_ISO < SP.OOF_START_ISO <= SP.OOF_END_ISO < SP.CUTOFF_ISO
    assert SP.BASE_DEPTH == 3 and SP.BASE_COLUMNS == "existing_all+strips17"
    assert SP.BASE_PARAMS.get("n_estimators") == 900
    assert abs(float(SP.BASE_PARAMS.get("learning_rate")) - 0.03) < 1e-12
    assert SP.PATH_SOURCE == "oof-fit"
    assert SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert "[axes:" in SP.AXES_TAG and "twokey" in SP.AXES_TAG


def test_tw_single_axis_vs_e0421():
    SP = _load("spec_tw")
    # Base recipe identical to E0421 (existing_all+17 strips, CPU hist).
    assert "max_depth" not in SP.BASE_PARAMS, "depth comes only from BASE_DEPTH"
    assert SP.BASE_PARAMS.get("tree_method") == "hist"
    assert SP.BASE_PARAMS.get("device") == "cpu"
    assert SP.WEIGHTS["1-6"]["e0003_line"] == 0.21536233464375223
    assert SP.WEIGHTS["7-24"]["d33_hsplit"] == 0.3436571856823209
    assert SP.K == 15 and SP.POINT_STAT == "mean"
    assert SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert SP.PATH_SOURCE == "oof-fit"
    import sys
    sys.path.insert(0, str(LB / "experiments/headroom_audit"))
    import info_ablation as ia
    for k, v in ia.PARAMS.items():
        if k in ("max_depth", "n_jobs", "random_state", "n_estimators", "learning_rate"):
            continue
        assert SP.BASE_PARAMS.get(k) == v, k
    RL = _load("resid_lib_tw")
    RL.check_weights(SP.WEIGHTS)
    RL.check_alpha_grid(SP.ALPHA_GRID)
    RL.check_blend_grid(SP.BLEND_GRID)
    text = (EXP / "run_official_tw.py").read_text()
    assert "import spec_tw as SP" in text
    assert "worker-t1a8" in text
    assert "predictions_official_s{s}.parquet" in text
    assert "ALPHA_GRID" in text and "pick_best_alpha" in text
    assert "fit_blend_weight_dv" in text


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
        assert "--gpu" not in t, f.name
        for w in ("freeze", "submit", "upload"):
            assert w not in t.lower(), (f.name, w)


def test_seed_actually_propagates():
    RL = _load("resid_lib_tw")
    SP = _load("spec_tw")
    params = [RL.fit_params(dict(SP.BASE_PARAMS), SP.BASE_DEPTH, s) for s in SP.SEEDS]
    assert len({p["random_state"] for p in params}) == 3
    assert all(p["max_depth"] == 3 for p in params)
    assert RL.level_seeds(1) == (3, 4, 5) and RL.event_seed(1) == 4


def test_mixture_blend_helpers():
    RL = _load("resid_lib_tw")
    SP = _load("spec_tw")
    h = np.arange(1, 73)
    members = {m: np.full(72, 400.0) for m in RL.MEMBERS}
    out = RL.blend_per_lead(members, SP.WEIGHTS, h)
    assert np.allclose(out, 400.0)
    st = RL.stitch_members({0: np.full(6, 1.0)}, {0: np.full(18, 2.0)},
                            np.full(72, 420.0), np.full(48, 3.0),
                            h <= 6, (h >= 7) & (h <= 24), h > 24, (0,), 0)
    assert np.allclose(st["e0003_line"][:6], 420.0)


def test_scenario_past_and_carrington():
    RL = _load("resid_lib_tw")
    idx = pd.date_range("2025-01-01", periods=24 * 60, freq="h", tz="UTC")
    filled = pd.Series(400.0, index=idx)
    o = pd.DatetimeIndex([idx[1000]], tz="UTC")
    m = RL.scenario_past_matrix(o, filled)
    assert m.shape == (1, 12) and np.allclose(m, 400.0)
    cands = RL.candidate_pool(o[0], idx, len(idx))
    assert len(cands) > 0
    assert (o[0] - idx[cands]).min() >= pd.Timedelta("655h")
    assert (o[0] - idx[cands]).max() <= pd.Timedelta("8760h")


def test_error_matrix():
    RL = _load("resid_lib_tw")
    past = np.full((2, 12), 420.0)
    hind = np.full((2, 12), 410.0)
    err = RL.scenario_error_matrix(past, hind)
    assert err.shape == (2, 12) and np.allclose(err, 10.0)


def test_twokey_distance_split_and_nesting():
    RL = _load("resid_lib_tw")
    SP = _load("spec_tw")
    RL.check_alpha_grid(SP.ALPHA_GRID)
    past_q = np.full((1, 12), 400.0)
    err_q = np.full((1, 12), 10.0)
    path_q = np.full((1, 72), 400.0)
    past_c = np.full((3, 12), 402.0)
    err_c = np.full((3, 12), 10.0)
    path_c = np.full((3, 72), 400.0)
    d_state, d_shape = RL.scenario_distance_parts(past_q, err_q, path_q, past_c, err_c, path_c)
    # past differs by 2 -> past-mean 4, error identical -> state = 0.5*4+0.5*0 = 2; shape 0.
    assert np.allclose(d_state, [2.0]) and np.allclose(d_shape, [0.0])
    d_tw = RL.scenario_distance_tw(past_q, err_q, path_q, past_c, err_c, path_c, 0.5)
    assert np.allclose(d_tw, 0.5 * d_shape + 0.5 * d_state)
    # alpha=1/3 nests the equal-per-subspace ranking (E0421/3).
    rng = np.random.default_rng(0)
    pq, eq, hq = rng.normal(400, 20, (1, 12)), rng.normal(0, 10, (1, 12)), rng.normal(400, 30, (1, 72))
    pc, ec, hc = rng.normal(400, 20, (7, 12)), rng.normal(0, 10, (7, 12)), rng.normal(400, 30, (7, 72))
    d3 = RL.scenario_distance3(pq, eq, hq, pc, ec, hc)
    dt = RL.scenario_distance_tw(pq, eq, hq, pc, ec, hc, 1.0 / 3.0)
    assert np.allclose(dt, d3 / 3.0)
    assert np.array_equal(np.argsort(d3, kind="stable"), np.argsort(dt, kind="stable"))
    # shape-heavy alpha reacts more to a path-only move than state-heavy alpha.
    d_lo = RL.scenario_distance_tw(pq, eq, hq, pc, ec, hc + 50.0, 0.2)
    d_hi = RL.scenario_distance_tw(pq, eq, hq, pc, ec, hc + 50.0, 0.8)
    assert bool(((d_hi - dt) > (d_lo - dt)).all())
    for bad in (0.0, 1.0, -0.1, 1.5, float("nan")):
        try:
            RL.check_alpha(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"check_alpha must reject {bad}")
    try:
        RL.check_alpha_grid((0.2, 0.5))
    except AssertionError:
        pass
    else:
        raise AssertionError("check_alpha_grid must reject a grid without 1/3")


def test_pick_best_alpha_stable():
    RL = _load("resid_lib_tw")
    a, m = RL.pick_best_alpha({0.2: 100.0, 1.0 / 3.0: 90.0, 0.5: 95.0})
    assert abs(a - 1.0 / 3.0) < 1e-12 and m == 90.0
    a2, _ = RL.pick_best_alpha({0.2: 90.0, 0.5: 90.0, 0.8: 95.0})
    assert a2 == 0.2, "ties keep the first minimum"
    try:
        RL.pick_best_alpha({0.5: 1.0})
    except ValueError:
        pass
    else:
        raise AssertionError("pick_best_alpha must reject a single cell")


def test_as_path_matrix():
    RL = _load("resid_lib_tw")
    m = RL.as_path_matrix(np.arange(144, dtype=float))
    assert m.shape == (2, 72)
    try:
        RL.as_path_matrix(np.arange(100, dtype=float))
    except AssertionError:
        pass
    else:
        raise AssertionError("as_path_matrix must reject non-multiples of 72")


def test_residual_replay_twokey():
    RL = _load("resid_lib_tw")
    idx = pd.date_range("2025-01-01", periods=24 * 500, freq="h", tz="UTC")
    filled = pd.Series(400.0 + 0.05 * np.arange(len(idx)), index=idx)
    o = pd.DatetimeIndex([idx[8000], idx[8100]], tz="UTC")
    qp = np.full((2, 72), 420.0)
    qh = np.full((2, 72), 410.0)
    cand_times = idx
    cp = np.full((len(idx), 72), 420.0)
    out, cov = RL.residual_for_origins(o, filled, qp, qh, cand_times, cp, 5, 0.5)
    assert out.shape == (2, 72) and cov.shape == (2,)
    assert bool(np.isfinite(out).all()) and (cov > 0).all()
    out2, cov2 = RL.residual_for_origins(o, filled, qp, qh, cand_times, cp, 5, 0.5)
    assert np.array_equal(out, out2, equal_nan=True) and np.array_equal(cov, cov2)
    out3, _ = RL.residual_for_origins(o, filled, qp, qh, cand_times, cp, 5, 0.8)
    assert out3.shape == (2, 72) and bool(np.isfinite(out3).all())
    far = pd.Timestamp("2027-01-01T00:00:00Z", tz="UTC")
    outf, covf = RL.residual_for_origins(o, filled, qp, qh, cand_times, cp, 5, 0.5, cutoff=far)
    assert np.array_equal(out, outf, equal_nan=True) and np.array_equal(cov, covf)
    early = pd.DatetimeIndex([idx[5]], tz="UTC")
    oute, cove = RL.residual_for_origins(
        early, filled, np.full((1, 72), 420.0), np.full((1, 72), 410.0),
        cand_times, cp, 5, 0.5)
    assert bool(np.isnan(oute).all()) and cove[0] == 0


def test_blend_dv_convex_and_oof_only():
    RL = _load("resid_lib_tw")
    RL.check_blend_grid(tuple(round(i * 0.05, 2) for i in range(21)))
    rng = np.random.default_rng(0)
    b = rng.normal(400, 50, 500)
    dv = rng.normal(0, 10, 500)
    y = rng.normal(400, 50, 500)
    w, base_mse, blend_mse, mses = RL.fit_blend_weight_dv(b, dv, y, tuple(round(i * 0.05, 2) for i in range(21)))
    assert 0.0 <= w <= 1.0 and len(mses) == 21 and blend_mse <= base_mse + 1e-9
    out = RL.blend_dv(b, dv, w)
    assert np.allclose(out, b + w * dv)
    nan_dv = dv.copy()
    nan_dv[:10] = np.nan
    out2 = RL.blend_dv(b, nan_dv, w)
    assert np.allclose(out2[:10], b[:10])


def test_output_frame_schema():
    RL = _load("resid_lib_tw")
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = RL.frame_predictions(te, np.full(72, 400.0))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert (fr["horizon_hours"].to_numpy() == np.arange(1, 73)).all()
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_poison_keys_and_invariance():
    RL = _load("resid_lib_tw")
    idx = pd.date_range("2026-05-01", periods=24 * 70, freq="h", tz="UTC")
    filled = pd.Series(400.0 + np.sin(np.arange(len(idx)) / 24.0), index=idx)
    cut = pd.Timestamp("2026-05-29T00:00:00Z", tz="UTC")
    pre = pd.DatetimeIndex([idx[100], idx[200]], tz="UTC")
    post = pd.DatetimeIndex([idx[1200], idx[1300]], tz="UTC")
    probe = pre.append(post)
    rec = RL.scenario_poison(probe, filled, cut)
    assert rec["passed"] is True
    assert rec["pre_invariant"] is True and rec["moved_columns"] == ["scenario"]
    assert rec["n_pre_rows"] == 2 and rec["n_post_rows"] == 2
    rec2 = RL.frame_poison(probe, filled, cut, lambda f: RL.scenario_past_matrix(probe, f))
    assert rec2["passed"] is True
    assert rec2["pre_invariant"] is True and rec2["moved_columns"] == ["frame"]
    assert rec2["n_pre_rows"] == 2 and rec2["n_post_rows"] == 2
