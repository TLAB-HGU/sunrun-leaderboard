"""D143 MED contract: static + synthetic checks (fast, no fits, no data)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d143_axisA_analog" / "median"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_IDS = ["e0147-abl", "d141_e0147", "ablate", "worker-d76", "worker-t1d7",
             "worker-t1a1", "worker-t1a2", "worker-t1a5", "worker-d135", "worker-d133",
             "worker-t2no", "nn-"]
BANNED_SRC = ["GONG", "WSA", "ADAPT", "CME", "DONKI"]


def _load(name):
    loc = importlib.util.spec_from_file_location(f"d143med_{name}", EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(EXP.glob("*.py")))


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 5, files
    assert {"spec_med15.py", "spec_tm15.py", "resid_lib_med.py",
            "run_official_med15.py", "run_official_tm15.py"} <= set(pys), files


def test_spec_identity_and_scope():
    for name, stat, trim in (("spec_med15", "median", 0.0),
                             ("spec_tm15", "trimmed-mean", 0.2)):
        SP = _load(name)
        assert SP.DIRECTION == "D143" and SP.OWNER == "worker-t1a7", name
        assert SP.KIND == "full" and SP.REFERENCE == "E0421", name
        assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0, name
        assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z", name
        assert SP.N_OFFICIAL_ORIGINS == 2833, name
        assert SP.TRAIN_FREQ == "3h", name
        assert SP.SCRATCH == "store/scratch/worker-t1a7", name
        assert SP.FIT_NJOBS <= 20, name
        assert SP.K == 15, name
        assert SP.SCENARIO_H == 12 and SP.ERR_H == 12 and SP.PATH_H == 72, name
        assert SP.SCENARIO_DIM == 96, name
        assert tuple(SP.STAT_SEP_GRID) == ("median", "trimmed-mean"), name
        assert SP.POINT_STAT == stat and abs(float(SP.TRIM_FRAC) - trim) < 1e-12, name
        assert SP.POINT_STAT in SP.STAT_SEP_GRID, name
        assert abs(SP.CARRINGTON_H - 654.48) < 1e-9, name
        assert len(SP.BLEND_GRID) == 21 and SP.BLEND_GRID[0] == 0.0 and SP.BLEND_GRID[-1] == 1.0, name
        assert SP.OOF_TRAIN_END_ISO < SP.OOF_START_ISO <= SP.OOF_END_ISO < SP.CUTOFF_ISO, name
        assert SP.BASE_DEPTH == 3 and SP.BASE_COLUMNS == "existing_all+strips17", name
        assert SP.BASE_PARAMS.get("n_estimators") == 900, name
        assert abs(float(SP.BASE_PARAMS.get("learning_rate")) - 0.03) < 1e-12, name
        assert SP.PATH_SOURCE == "oof-fit", name
        assert SP.RESIDUAL_DEF == "dv = y - yhat_base", name
        assert "[axes:" in SP.AXES_TAG and "residual" in SP.AXES_TAG, name


def test_med_single_axis_separation():
    SM = _load("spec_med15")
    ST = _load("spec_tm15")
    # Only the point statistic differs between the two separation cells.
    assert SM.K == ST.K == 15
    assert {SM.POINT_STAT, ST.POINT_STAT} == {"median", "trimmed-mean"}
    assert float(SM.TRIM_FRAC) == 0.0 and abs(float(ST.TRIM_FRAC) - 0.2) < 1e-12
    assert SM.SCENARIO_DIM == ST.SCENARIO_DIM == 96
    assert SM.BASE_PARAMS == ST.BASE_PARAMS and SM.WEIGHTS == ST.WEIGHTS
    assert SM.BASE_DEPTH == ST.BASE_DEPTH == 3
    assert SM.PATH_SOURCE == ST.PATH_SOURCE == "oof-fit"
    assert SM.BLEND_GRID == ST.BLEND_GRID
    # Each run pins its own spec; no shared-config rerun.
    for run_name, spec_name, stat in (("run_official_med15", "spec_med15", "median"),
                                      ("run_official_tm15", "spec_tm15", "trimmed-mean")):
        text = (EXP / f"{run_name}.py").read_text()
        assert f"import {spec_name} as SP" in text, run_name
        assert f'POINT_STAT == "{stat}"' in text, run_name
        assert "worker-t1a7" in text, run_name
        assert "predictions_official_s{s}.parquet" in text, run_name
    # Base recipe identical to E0421 (existing_all+17 strips, CPU hist).
    import sys
    sys.path.insert(0, str(LB / "experiments/headroom_audit"))
    import info_ablation as ia
    for SP in (SM, ST):
        assert "max_depth" not in SP.BASE_PARAMS, "depth comes only from BASE_DEPTH"
        for k, v in ia.PARAMS.items():
            if k in ("max_depth", "n_jobs", "random_state", "n_estimators", "learning_rate"):
                continue
            assert SP.BASE_PARAMS.get(k) == v, k
        assert SP.BASE_PARAMS.get("tree_method") == "hist"
        assert SP.BASE_PARAMS.get("device") == "cpu"
    RL = _load("resid_lib_med")
    RL.check_weights(SM.WEIGHTS)
    RL.check_weights(ST.WEIGHTS)
    RL.check_point_stat(SM.POINT_STAT, SM.TRIM_FRAC)
    RL.check_point_stat(ST.POINT_STAT, ST.TRIM_FRAC)


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
    RL = _load("resid_lib_med")
    SP = _load("spec_med15")
    params = [RL.fit_params(dict(SP.BASE_PARAMS), SP.BASE_DEPTH, s) for s in SP.SEEDS]
    assert len({p["random_state"] for p in params}) == 3
    assert all(p["max_depth"] == 3 for p in params)
    assert RL.level_seeds(1) == (3, 4, 5) and RL.event_seed(1) == 4


def test_mixture_blend_helpers():
    RL = _load("resid_lib_med")
    SP = _load("spec_med15")
    h = np.arange(1, 73)
    members = {m: np.full(72, 400.0) for m in RL.MEMBERS}
    out = RL.blend_per_lead(members, SP.WEIGHTS, h)
    assert np.allclose(out, 400.0)
    st = RL.stitch_members({0: np.full(6, 1.0)}, {0: np.full(18, 2.0)},
                            np.full(72, 420.0), np.full(48, 3.0),
                            h <= 6, (h >= 7) & (h <= 24), h > 24, (0,), 0)
    assert np.allclose(st["e0003_line"][:6], 420.0)


def test_scenario_past_and_carrington():
    RL = _load("resid_lib_med")
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
    RL = _load("resid_lib_med")
    past = np.full((2, 12), 420.0)
    hind = np.full((2, 12), 410.0)
    err = RL.scenario_error_matrix(past, hind)
    assert err.shape == (2, 12) and np.allclose(err, 10.0)


def test_path_distance_equal_subspace_weight():
    RL = _load("resid_lib_med")
    past_q = np.full((1, 12), 400.0)
    err_q = np.full((1, 12), 10.0)
    path_q = np.full((1, 72), 400.0)
    d_past = RL.scenario_distance3(past_q, err_q, path_q,
                                   np.full((1, 12), 402.0), np.full((1, 12), 10.0), np.full((1, 72), 400.0))
    d_err = RL.scenario_distance3(past_q, err_q, path_q,
                                  np.full((1, 12), 400.0), np.full((1, 12), 12.0), np.full((1, 72), 400.0))
    d_path = RL.scenario_distance3(past_q, err_q, path_q,
                                   np.full((1, 12), 400.0), np.full((1, 12), 10.0), np.full((1, 72), 402.0))
    assert np.allclose(d_past, [4.0]) and np.allclose(d_err, [4.0]) and np.allclose(d_path, [4.0])
    assert np.allclose(d_past, d_err) and np.allclose(d_err, d_path)


def test_point_stat_helpers():
    RL = _load("resid_lib_med")
    RL.check_point_stat("median", 0.0)
    RL.check_point_stat("trimmed-mean", 0.2)
    for stat, trim in (("mean", 0.0), ("median", 0.1), ("trimmed-mean", 0.0),
                       ("trimmed-mean", 0.5), ("trimmed-mean", -0.1)):
        try:
            RL.check_point_stat(stat, trim)
        except ValueError:
            pass
        else:
            raise AssertionError(f"check_point_stat must reject {(stat, trim)}")
    try:
        RL.aggregate_curves(np.zeros((0, 72)), "median", 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("aggregate_curves must reject empty curves")
    # median ignores a far outlier; uniform mean would move.
    curves = np.array([[400.0] * 72, [402.0] * 72, [398.0] * 72,
                       [401.0] * 72, [399.0] * 72, [900.0] * 72])
    med = RL.aggregate_curves(curves, "median", 0.0)
    assert np.allclose(med, np.median(curves, axis=0))
    assert abs(float(med[0]) - 400.5) < 1e-9
    # trimmed-mean 0.2 on 5 values cuts 1 per tail, means the middle 3.
    c5 = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [100.0, 50.0]])
    tm = RL.aggregate_curves(c5, "trimmed-mean", 0.2)
    assert np.allclose(tm, [3.0, 30.0])
    # NaN entries are skipped per horizon.
    cn = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    assert np.allclose(RL.aggregate_curves(cn, "median", 0.0), [3.0, 5.0])
    assert np.allclose(RL.aggregate_curves(cn, "trimmed-mean", 0.2), [3.0, 5.0])
    allnan = np.full((3, 72), np.nan)
    assert bool(np.isnan(RL.aggregate_curves(allnan, "median", 0.0)).all())


def test_as_path_matrix():
    RL = _load("resid_lib_med")
    m = RL.as_path_matrix(np.arange(144, dtype=float))
    assert m.shape == (2, 72)
    try:
        RL.as_path_matrix(np.arange(100, dtype=float))
    except AssertionError:
        pass
    else:
        raise AssertionError("as_path_matrix must reject non-multiples of 72")


def test_residual_replay_point_stat():
    RL = _load("resid_lib_med")
    idx = pd.date_range("2025-01-01", periods=24 * 500, freq="h", tz="UTC")
    filled = pd.Series(400.0 + 0.05 * np.arange(len(idx)), index=idx)
    o = pd.DatetimeIndex([idx[8000], idx[8100]], tz="UTC")
    qp = np.full((2, 72), 420.0)
    qh = np.full((2, 72), 410.0)
    cand_times = idx
    cp = np.full((len(idx), 72), 420.0)
    out, cov = RL.residual_for_origins(o, filled, qp, qh, cand_times, cp, 5,
                                       point_stat="median", trim_frac=0.0)
    assert out.shape == (2, 72) and cov.shape == (2,)
    assert bool(np.isfinite(out).all()) and (cov > 0).all()
    out2, cov2 = RL.residual_for_origins(o, filled, qp, qh, cand_times, cp, 5,
                                         point_stat="median", trim_frac=0.0)
    assert np.array_equal(out, out2, equal_nan=True) and np.array_equal(cov, cov2)
    outt, _ = RL.residual_for_origins(o, filled, qp, qh, cand_times, cp, 5,
                                      point_stat="trimmed-mean", trim_frac=0.2)
    assert outt.shape == (2, 72) and bool(np.isfinite(outt).all())
    far = pd.Timestamp("2027-01-01T00:00:00Z", tz="UTC")
    outf, covf = RL.residual_for_origins(o, filled, qp, qh, cand_times, cp, 5,
                                         cutoff=far, point_stat="median", trim_frac=0.0)
    assert np.array_equal(out, outf, equal_nan=True) and np.array_equal(cov, covf)
    early = pd.DatetimeIndex([idx[5]], tz="UTC")
    out3, cov3 = RL.residual_for_origins(
        early, filled, np.full((1, 72), 420.0), np.full((1, 72), 410.0),
        cand_times, cp, 5, point_stat="median", trim_frac=0.0)
    assert bool(np.isnan(out3).all()) and cov3[0] == 0


def test_blend_dv_convex_and_oof_only():
    RL = _load("resid_lib_med")
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
    RL = _load("resid_lib_med")
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = RL.frame_predictions(te, np.full(72, 400.0))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert (fr["horizon_hours"].to_numpy() == np.arange(1, 73)).all()
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_poison_keys_and_invariance():
    RL = _load("resid_lib_med")
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
