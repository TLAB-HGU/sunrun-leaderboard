"""D143 PL-DIST contract: static + synthetic checks (fast, no fits, no data)."""
import importlib.util

import numpy as np
import pandas as pd
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d143_axisA_analog" / "perlead"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_IDS = ["e0147-abl", "d141_e0147", "ablate", "worker-d76", "worker-t1d7",
             "worker-t1a1", "worker-t1a2", "worker-t1a5", "worker-t1a6", "worker-t1a7",
             "worker-t1c1", "worker-t1c2", "worker-d135", "worker-d133",
             "worker-t2no", "nn-"]
BANNED_SRC = ["GONG", "WSA", "ADAPT", "CME", "DONKI"]


def _load(name):
    loc = importlib.util.spec_from_file_location(f"d143pl_{name}", EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(EXP.glob("*.py")))


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 6, files
    assert {"spec_pl.py", "resid_lib_pl.py",
            "run_official_pl.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_pl")
    assert SP.DIRECTION == "D143" and SP.OWNER == "worker-t1b0"
    assert SP.KIND == "full" and SP.REFERENCE == "E0421"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h"
    assert SP.SCRATCH == "store/scratch/worker-t1b0"
    assert SP.FIT_NJOBS <= 20
    assert SP.K == 15
    assert SP.SCENARIO_H == 12 and SP.ERR_H == 12 and SP.PATH_H == 72
    assert SP.SCENARIO_DIM == 96
    assert tuple(SP.STAT_SEP_GRID) == ("median", "trimmed-mean")
    assert SP.POINT_STAT == "median" and abs(float(SP.TRIM_FRAC)) < 1e-12
    assert SP.POINT_STAT in SP.STAT_SEP_GRID
    assert tuple(SP.DIST_QUANTS) == (10, 50, 90)
    assert SP.DIST_DEF and "d2mean" in SP.DIST_DEF and "resid_var" in SP.DIST_DEF
    assert abs(SP.CARRINGTON_H - 654.48) < 1e-9
    assert len(SP.BLEND_GRID) == 21 and SP.BLEND_GRID[0] == 0.0 and SP.BLEND_GRID[-1] == 1.0
    assert SP.OOF_TRAIN_END_ISO < SP.OOF_START_ISO <= SP.OOF_END_ISO < SP.CUTOFF_ISO
    assert SP.BASE_DEPTH == 3 and SP.BASE_COLUMNS == "existing_all+strips17"
    assert SP.BASE_PARAMS.get("n_estimators") == 900
    assert abs(float(SP.BASE_PARAMS.get("learning_rate")) - 0.03) < 1e-12
    assert SP.PATH_SOURCE == "oof-fit"
    assert SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert "[axes:" in SP.AXES_TAG and "residual" in SP.AXES_TAG


def test_pl_single_axis_pinned():
    SP = _load("spec_pl")
    text = (EXP / "run_official_pl.py").read_text()
    assert "import spec_pl as SP" in text
    assert "import resid_lib_pl as RL" in text
    assert 'POINT_STAT == "median"' in text
    assert "worker-t1b0" in text
    assert "predictions_official_s{s}.parquet" in text
    assert "resid_dist_s{s}.parquet" in text
    assert "return_dist=True" in text
    assert "dist_quants=SP.DIST_QUANTS" in text
    assert "point must equal stored q50" in text
    # Base recipe identical to E0421 (existing_all+17 strips, CPU hist).
    import sys
    sys.path.insert(0, str(LB / "experiments/headroom_audit"))
    import info_ablation as ia
    assert "max_depth" not in SP.BASE_PARAMS, "depth comes only from BASE_DEPTH"
    for k, v in ia.PARAMS.items():
        if k in ("max_depth", "n_jobs", "random_state", "n_estimators", "learning_rate"):
            continue
        assert SP.BASE_PARAMS.get(k) == v, k
    assert SP.BASE_PARAMS.get("tree_method") == "hist"
    assert SP.BASE_PARAMS.get("device") == "cpu"
    RL = _load("resid_lib_pl")
    RL.check_weights(SP.WEIGHTS)
    RL.check_point_stat(SP.POINT_STAT, SP.TRIM_FRAC)
    RL.check_dist_quants(SP.DIST_QUANTS)


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
    RL = _load("resid_lib_pl")
    SP = _load("spec_pl")
    params = [RL.fit_params(dict(SP.BASE_PARAMS), SP.BASE_DEPTH, s) for s in SP.SEEDS]
    assert len({p["random_state"] for p in params}) == 3
    assert all(p["max_depth"] == 3 for p in params)
    assert RL.level_seeds(1) == (3, 4, 5) and RL.event_seed(1) == 4


def test_mixture_blend_helpers():
    RL = _load("resid_lib_pl")
    SP = _load("spec_pl")
    h = np.arange(1, 73)
    members = {m: np.full(72, 400.0) for m in RL.MEMBERS}
    out = RL.blend_per_lead(members, SP.WEIGHTS, h)
    assert np.allclose(out, 400.0)
    st = RL.stitch_members({0: np.full(6, 1.0)}, {0: np.full(18, 2.0)},
                            np.full(72, 420.0), np.full(48, 3.0),
                            h <= 6, (h >= 7) & (h <= 24), h > 24, (0,), 0)
    assert np.allclose(st["e0003_line"][:6], 420.0)


def test_scenario_past_and_carrington():
    RL = _load("resid_lib_pl")
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
    RL = _load("resid_lib_pl")
    past = np.full((2, 12), 420.0)
    hind = np.full((2, 12), 410.0)
    err = RL.scenario_error_matrix(past, hind)
    assert err.shape == (2, 12) and np.allclose(err, 10.0)


def test_path_distance_equal_subspace_weight():
    RL = _load("resid_lib_pl")
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
    RL = _load("resid_lib_pl")
    RL.check_point_stat("median", 0.0)
    for stat, trim in (("mean", 0.0), ("median", 0.1), ("trimmed-mean", 0.0),
                       ("trimmed-mean", 0.5), ("trimmed-mean", -0.1)):
        try:
            RL.check_point_stat(stat, trim)
        except ValueError:
            pass
        else:
            raise AssertionError(f"check_point_stat must reject {(stat, trim)}")
    curves = np.array([[400.0] * 72, [402.0] * 72, [398.0] * 72,
                       [401.0] * 72, [399.0] * 72, [900.0] * 72])
    med = RL.aggregate_curves(curves, "median", 0.0)
    assert np.allclose(med, np.median(curves, axis=0))
    assert abs(float(med[0]) - 400.5) < 1e-9


def test_dist_helpers():
    RL = _load("resid_lib_pl")
    RL.check_dist_quants((10, 50, 90))
    for bad in ((5, 50, 95), (10, 50), (10, 50, 90, 99), ("a", "b", "c")):
        try:
            RL.check_dist_quants(bad)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"check_dist_quants must reject {bad}")
    curves = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])
    dd = RL.dist_for_curves(curves)
    assert set(dd) == {"q10", "q50", "q90", "mean"}
    assert np.allclose(dd["q50"], [3.0, 30.0])
    assert np.allclose(dd["mean"], [3.0, 30.0])
    assert bool((dd["q10"] <= dd["q50"]).all() and (dd["q50"] <= dd["q90"]).all())
    assert np.allclose(dd["q10"], [np.quantile([1, 2, 3, 4, 5], 0.1), np.quantile([10, 20, 30, 40, 50], 0.1)])
    vv = RL.dispersion_for_curves(curves)
    assert np.allclose(vv, [np.var([1, 2, 3, 4, 5]), np.var([10, 20, 30, 40, 50])])
    assert float(vv[0]) >= 0.0
    # NaN entries are skipped per horizon; <2 finite entries give NaN variance.
    cn = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    ddn = RL.dist_for_curves(cn)
    assert np.allclose(ddn["q50"], [3.0, 5.0])
    vvn = RL.dispersion_for_curves(cn)
    assert np.isfinite(vvn).all()
    one = np.array([[1.0, np.nan], [np.nan, np.nan]])
    assert bool(np.isnan(RL.dispersion_for_curves(one)[1]))
    assert bool(np.isnan(RL.dispersion_for_curves(one)[0]))
    allnan = np.full((3, 72), np.nan)
    assert bool(np.isnan(RL.dist_for_curves(allnan)["q50"]).all())
    try:
        RL.dist_for_curves(np.zeros((0, 72)))
    except ValueError:
        pass
    else:
        raise AssertionError("dist_for_curves must reject empty curves")


def test_residual_dist_replay():
    RL = _load("resid_lib_pl")
    idx = pd.date_range("2025-01-01", periods=24 * 500, freq="h", tz="UTC")
    filled = pd.Series(400.0 + 0.05 * np.arange(len(idx)), index=idx)
    o = pd.DatetimeIndex([idx[8000], idx[8100]], tz="UTC")
    qp = np.full((2, 72), 420.0)
    qh = np.full((2, 72), 410.0)
    cand_times = idx
    cp = np.full((len(idx), 72), 420.0)
    out, cov, dist = RL.residual_for_origins(o, filled, qp, qh, cand_times, cp, 5,
                                             point_stat="median", trim_frac=0.0,
                                             return_dist=True, dist_quants=(10, 50, 90))
    assert out.shape == (2, 72) and cov.shape == (2,)
    assert bool(np.isfinite(out).all()) and (cov > 0).all()
    for k in ("q10", "q50", "q90", "dv_mean", "d2mean", "resid_var"):
        assert k in dist, k
    assert dist["q10"].shape == (2, 72) and dist["d2mean"].shape == (2,)
    assert bool((dist["q10"] <= dist["q50"]).all() and (dist["q50"] <= dist["q90"]).all())
    assert np.array_equal(out, dist["q50"], equal_nan=True), "point must equal stored q50"
    assert bool(np.isfinite(dist["d2mean"]).all() and (dist["d2mean"] >= 0).all())
    assert bool(np.nan_to_num(dist["resid_var"], nan=0.0).min() >= 0.0)
    # default call keeps the old 2-tuple contract and matches the point.
    out2, cov2 = RL.residual_for_origins(o, filled, qp, qh, cand_times, cp, 5,
                                         point_stat="median", trim_frac=0.0)
    assert np.array_equal(out, out2, equal_nan=True) and np.array_equal(cov, cov2)


def test_blend_dv_convex_and_oof_only():
    RL = _load("resid_lib_pl")
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
    RL = _load("resid_lib_pl")
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = RL.frame_predictions(te, np.full(72, 400.0))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert (fr["horizon_hours"].to_numpy() == np.arange(1, 73)).all()
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_poison_keys_and_invariance():
    RL = _load("resid_lib_pl")
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
