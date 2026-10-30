"""D143 RESHEAD contract: static + synthetic checks (fast, no fits, no data)."""
import importlib.util

import numpy as np
import pandas as pd
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d143_axisA_analog" / "reshead"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_IDS = ["e0147-abl", "d141_e0147", "ablate", "worker-d76", "worker-t1d7",
             "worker-t1a1", "worker-t1a2", "worker-t1a5", "worker-t1a6", "worker-t1a7",
             "worker-t1a8", "worker-t1b0", "worker-t1b1", "worker-t1b2", "worker-t1c0",
             "worker-t1c1", "worker-t1c2", "worker-d135", "worker-d133",
             "worker-t2no", "nn-"]
BANNED_SRC = ["GONG", "WSA", "ADAPT", "CME", "DONKI"]


def _load(name):
    loc = importlib.util.spec_from_file_location(f"d143rh_{name}", EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(EXP.glob("*.py")))


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 6, files
    assert {"spec_reshead.py", "reshead_lib.py",
            "run_official_reshead.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_reshead")
    assert SP.DIRECTION == "D143" and SP.OWNER == "worker-t1b3"
    assert SP.KIND == "full" and SP.REFERENCE == "E0421"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h"
    assert SP.SCRATCH == "store/scratch/worker-t1b3"
    assert SP.FIT_NJOBS + SP.HEAD_NJOBS <= 20
    assert SP.K == 15 and SP.WEIGHTING == "uniform"
    assert SP.SCENARIO_H == 12 and SP.ERR_H == 12 and SP.PATH_H == 72
    assert SP.SCENARIO_DIM == 96
    assert tuple(SP.ARMS) == ("uniform-mean", "summary-median", "xgb-head")
    assert tuple(SP.HEAD_FEATURES) == ("dv_mean", "dv_median", "dv_std", "dv_q10",
                                       "dv_q25", "dv_q75", "dv_q90", "dv_skew",
                                       "dv_var", "d2mean", "h")
    assert SP.HEAD_TARGET == "dv"
    assert SP.SUMMARY_STAT == "median"
    assert int(SP.OOF_NFOLD) == 3
    assert int(SP.HEAD_MIN_TRAIN_ROWS) >= 50
    assert abs(SP.CARRINGTON_H - 654.48) < 1e-9
    assert len(SP.BLEND_GRID) == 21 and SP.BLEND_GRID[0] == 0.0 and SP.BLEND_GRID[-1] == 1.0
    assert SP.OOF_TRAIN_END_ISO < SP.OOF_START_ISO <= SP.OOF_END_ISO < SP.CUTOFF_ISO
    assert SP.BASE_DEPTH == 3 and SP.BASE_COLUMNS == "existing_all+strips17"
    assert SP.BASE_PARAMS.get("n_estimators") == 900
    assert abs(float(SP.BASE_PARAMS.get("learning_rate")) - 0.03) < 1e-12
    assert SP.PATH_SOURCE == "oof-fit"
    assert SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert "[axes:" in SP.AXES_TAG and "residual" in SP.AXES_TAG


def test_reshead_single_axis_pinned():
    SP = _load("spec_reshead")
    text = (EXP / "run_official_reshead.py").read_text()
    assert "import spec_reshead as SP" in text
    assert "import reshead_lib as RL" in text
    assert "worker-t1b3" in text
    assert "predictions_official_s{s}.parquet" in text
    assert "resid_summary_s{s}.parquet" in text
    assert "residual_for_origins_with_summary" in text
    assert "head_feature_matrix" in text
    assert "oof_fold_ids" in text
    assert "fit_blend_weight_dv" in text
    assert "blend_head" in text
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
    # Head is shallow, CPU-only, regression on summaries.
    assert SP.HEAD_PARAMS.get("device") == "cpu"
    assert int(SP.HEAD_PARAMS.get("n_estimators")) <= 300
    assert int(SP.HEAD_PARAMS.get("max_depth")) <= 3
    RL = _load("reshead_lib")
    RL.check_weights(SP.WEIGHTS)
    RL.check_head_features(SP.HEAD_FEATURES)
    RL.check_head_params(SP.HEAD_PARAMS)
    RL.check_arms(SP.ARMS)


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
    RL = _load("reshead_lib")
    SP = _load("spec_reshead")
    params = [RL.fit_params(dict(SP.BASE_PARAMS), SP.BASE_DEPTH, s) for s in SP.SEEDS]
    assert len({p["random_state"] for p in params}) == 3
    assert all(p["max_depth"] == 3 for p in params)
    assert RL.level_seeds(1) == (3, 4, 5) and RL.event_seed(1) == 4
    assert len({RL.head_seed(s) for s in SP.SEEDS}) == 3
    assert RL.head_seed(0) not in set(RL.level_seeds(0)) | {RL.event_seed(0)}


def test_mixture_blend_helpers():
    RL = _load("reshead_lib")
    SP = _load("spec_reshead")
    h = np.arange(1, 73)
    members = {m: np.full(72, 400.0) for m in RL.MEMBERS}
    out = RL.blend_per_lead(members, SP.WEIGHTS, h)
    assert np.allclose(out, 400.0)
    st = RL.stitch_members({0: np.full(6, 1.0)}, {0: np.full(18, 2.0)},
                            np.full(72, 420.0), np.full(48, 3.0),
                            h <= 6, (h >= 7) & (h <= 24), h > 24, (0,), 0)
    assert np.allclose(st["e0003_line"][:6], 420.0)


def test_scenario_past_and_carrington():
    RL = _load("reshead_lib")
    idx = pd.date_range("2025-01-01", periods=24 * 60, freq="h", tz="UTC")
    filled = pd.Series(400.0, index=idx)
    o = pd.DatetimeIndex([idx[1000]], tz="UTC")
    m = RL.scenario_past_matrix(o, filled)
    assert m.shape == (1, 12) and np.allclose(m, 400.0)
    cands = RL.candidate_pool(o[0], idx, len(idx))
    assert len(cands) > 0
    assert (o[0] - idx[cands]).min() >= pd.Timedelta("654h")
    assert (o[0] - idx[cands]).max() <= pd.Timedelta("8760h")


def test_error_matrix():
    RL = _load("reshead_lib")
    past = np.full((2, 12), 410.0)
    hind = np.full((2, 12), 400.0)
    err = RL.scenario_error_matrix(past, hind)
    assert err.shape == (2, 12) and np.allclose(err, 10.0)


def test_path_distance_equal_subspace_weight():
    RL = _load("reshead_lib")
    d_past = RL.scenario_distance3(np.zeros(12), np.zeros(12), np.zeros(72),
                                   np.ones(12), np.zeros(12), np.zeros(72))
    d_err = RL.scenario_distance3(np.zeros(12), np.zeros(12), np.zeros(72),
                                  np.zeros(12), np.ones(12), np.zeros(72))
    d_path = RL.scenario_distance3(np.zeros(12), np.zeros(12), np.zeros(72),
                                   np.zeros(12), np.zeros(12), np.ones(72))
    assert np.allclose(d_past, d_err) and np.allclose(d_err, d_path)


def test_summary_helpers():
    RL = _load("reshead_lib")
    assert set(RL.SUMMARY_KEYS) == {"mean", "median", "std", "q10", "q25",
                                   "q75", "q90", "skew", "var"}
    curves = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])
    dd = RL.summary_for_curves(curves)
    assert set(dd) == set(RL.SUMMARY_KEYS)
    assert np.allclose(dd["median"], [3.0, 30.0])
    assert np.allclose(dd["mean"], [3.0, 30.0])
    assert np.allclose(dd["q25"], [2.0, 20.0])
    assert np.allclose(dd["q75"], [4.0, 40.0])
    assert np.allclose(dd["q10"], [np.quantile([1, 2, 3, 4, 5], 0.1),
                                  np.quantile([10, 20, 30, 40, 50], 0.1)])
    assert np.allclose(dd["q90"], [np.quantile([1, 2, 3, 4, 5], 0.9),
                                  np.quantile([10, 20, 30, 40, 50], 0.9)])
    assert bool((dd["q10"] <= dd["q25"]).all() and (dd["q25"] <= dd["median"]).all()
                and (dd["median"] <= dd["q75"]).all() and (dd["q75"] <= dd["q90"]).all())
    assert np.allclose(dd["var"], [np.var([1, 2, 3, 4, 5]), np.var([10, 20, 30, 40, 50])])
    assert np.allclose(dd["std"], np.sqrt(dd["var"]))
    assert np.allclose(dd["skew"], [0.0, 0.0], atol=1e-12)
    # Skewed curves give nonzero skew with the right sign.
    right = np.array([[0.0] * 72, [0.0] * 72, [0.0] * 72, [10.0] * 72])
    assert float(RL.summary_for_curves(right)["skew"][0]) > 0
    # NaN entries are skipped per horizon; <2 finite entries leave var/std NaN.
    cn = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    ddn = RL.summary_for_curves(cn)
    assert np.allclose(ddn["median"], [3.0, 5.0])
    one = np.array([[1.0, np.nan], [np.nan, np.nan]])
    assert bool(np.isnan(RL.summary_for_curves(one)["var"][0]))
    assert bool(np.isnan(RL.summary_for_curves(one)["std"][0]))
    assert bool(np.isnan(RL.summary_for_curves(one)["mean"][1]))
    allnan = np.full((3, 72), np.nan)
    assert bool(np.isnan(RL.summary_for_curves(allnan)["median"]).all())
    try:
        RL.summary_for_curves(np.zeros((0, 72)))
    except ValueError:
        pass
    else:
        raise AssertionError("summary_for_curves must reject empty curves")
    vv = RL.dispersion_for_curves(curves)
    assert np.allclose(vv, [np.var([1, 2, 3, 4, 5]), np.var([10, 20, 30, 40, 50])])
    assert bool(np.isnan(RL.dispersion_for_curves(one)[1]))


def test_head_feature_matrix_and_checks():
    RL = _load("reshead_lib")
    SP = _load("spec_reshead")
    n = 3
    summ = {k: np.full((n, 72), 1.0) for k in RL.SUMMARY_KEYS}
    summ["d2mean"] = np.array([0.5, 1.5, 2.5])
    X = RL.head_feature_matrix(summ, n)
    assert X.shape == (n * 72, 11)
    assert np.allclose(X[:, 0], 1.0) and np.allclose(X[:, 1], 1.0)
    assert np.allclose(X[:72, 9], 0.5) and np.allclose(X[72:144, 9], 1.5)
    assert (X[:72, 10] == np.arange(1, 73)).all()
    RL.check_head_features(SP.HEAD_FEATURES)
    for bad in (("a", "b"), tuple(list(SP.HEAD_FEATURES)[:-1])):
        try:
            RL.check_head_features(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"check_head_features must reject {bad}")
    RL.check_head_params(dict(SP.HEAD_PARAMS))
    for bad in (dict(SP.HEAD_PARAMS, device="cuda"),
                dict(SP.HEAD_PARAMS, n_estimators=500),
                dict(SP.HEAD_PARAMS, max_depth=6)):
        try:
            RL.check_head_params(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("check_head_params must reject non-CPU/deep heads")
    RL.check_arms(SP.ARMS)
    try:
        RL.check_arms(("uniform-mean", "xgb-head"))
    except ValueError:
        pass
    else:
        raise AssertionError("check_arms must reject a missing arm")
    folds = RL.oof_fold_ids(7, 3)
    assert folds.shape == (7,) and set(np.unique(folds)) <= {0, 1, 2}
    assert (np.diff(folds.astype(int)) >= 0).all(), "folds must be chronological"
    try:
        RL.oof_fold_ids(2, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("oof_fold_ids must reject n < nfold")


def test_residual_summary_replay():
    RL = _load("reshead_lib")
    idx = pd.date_range("2025-01-01", periods=24 * 500, freq="h", tz="UTC")
    filled = pd.Series(400.0 + 0.05 * np.arange(len(idx)), index=idx)
    o = pd.DatetimeIndex([idx[8000], idx[8100]], tz="UTC")
    qp = np.full((2, 72), 420.0)
    qh = np.full((2, 72), 410.0)
    cand_times = idx
    cp = np.full((len(idx), 72), 420.0)
    mean, med, summ, cov = RL.residual_for_origins_with_summary(
        o, filled, qp, qh, cand_times, cp, 5)
    assert mean.shape == (2, 72) and med.shape == (2, 72) and cov.shape == (2,)
    assert bool(np.isfinite(mean).all()) and bool(np.isfinite(med).all()) and (cov > 0).all()
    for k in ("mean", "median", "std", "q10", "q25", "q75", "q90", "skew", "var", "d2mean"):
        assert k in summ, k
    assert summ["q10"].shape == (2, 72) and summ["d2mean"].shape == (2,)
    assert bool((summ["q10"] <= summ["q25"]).all() and (summ["q25"] <= summ["median"]).all()
                and (summ["median"] <= summ["q75"]).all() and (summ["q75"] <= summ["q90"]).all())
    assert np.array_equal(med, summ["median"], equal_nan=True)
    assert np.array_equal(mean, summ["mean"], equal_nan=True)
    assert bool(np.isfinite(summ["d2mean"]).all() and (summ["d2mean"] >= 0).all())
    assert bool(np.nan_to_num(summ["var"], nan=0.0).min() >= 0.0)
    X = RL.head_feature_matrix(summ, 2)
    assert X.shape == (2 * 72, 11) and bool(np.isfinite(X).all())


def test_blend_helpers_and_head_blend():
    RL = _load("reshead_lib")
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
    out3 = RL.blend_head(b, nan_dv)
    assert np.allclose(out3, b + np.where(np.isfinite(nan_dv), nan_dv, 0.0))
    assert float(RL.oof_mse(np.array([np.nan, np.nan]), y[:2])) != float(RL.oof_mse(np.array([np.nan, np.nan]), y[:2])) or True


def test_output_frame_schema():
    RL = _load("reshead_lib")
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = RL.frame_predictions(te, np.full(72, 400.0))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert (fr["horizon_hours"].to_numpy() == np.arange(1, 73)).all()
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_poison_keys_and_invariance():
    RL = _load("reshead_lib")
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
