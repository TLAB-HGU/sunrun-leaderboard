"""D143 ShiftSafe contract: static + synthetic checks (fast, no fits, no data)."""
import importlib.util
import sys

import numpy as np
import pandas as pd
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d143_axisA_analog" / "shiftsafe"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_IDS = ["e0147-abl", "d141_e0147", "ablate", "D141", "E0443", "E0446", "E0414", "E0438",
             "worker-d76", "worker-t1d7",
             "worker-t1a1", "worker-t1a2", "worker-t1a3", "worker-t1a5",
             "worker-t1a6", "worker-t1a7", "worker-t1a8",
             "worker-t1b0", "worker-t1b1", "worker-t1b2", "worker-t1b3",
             "worker-t1b4", "worker-t1b5",
             "worker-t1c0", "worker-t1c1", "worker-t1c2", "worker-t1c3",
             "worker-d135", "worker-d133", "worker-d141",
             "worker-t2no", "nn-", "sun-nn"]
BANNED_SRC = ["GONG", "WSA", "ADAPT", "CME", "DONKI"]
CELL_LABELS = ["low_low_high", "low_high_high", "high_low_high", "high_high_high"]


def _load(name):
    loc = importlib.util.spec_from_file_location(f"d143ss_{name}", EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(EXP.glob("*.py")))


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 6, files
    assert {"spec_shiftsafe.py", "shiftsafe_lib.py", "run_official_shiftsafe.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_shiftsafe")
    assert SP.DIRECTION == "D143" and SP.OWNER == "worker-t1b6"
    assert SP.KIND == "full" and SP.REFERENCE == "E0421"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h"
    assert SP.SCRATCH == "store/scratch/worker-t1b6"
    assert int(SP.FIT_NJOBS) <= 6
    assert int(SP.FIT_NJOBS) + int(SP.GATE_NJOBS) <= 20
    assert SP.K == 15 and SP.WEIGHTING == "uniform"
    assert SP.SCENARIO_H == 12 and SP.ERR_H == 12 and SP.PATH_H == 72
    assert SP.SCENARIO_DIM == 96
    assert abs(SP.CARRINGTON_H - 654.48) < 1e-9
    assert len(SP.BLEND_GRID) == 21 and SP.BLEND_GRID[0] == 0.0 and SP.BLEND_GRID[-1] == 1.0
    assert SP.OOF_TRAIN_END_ISO < SP.OOF_START_ISO <= SP.OOF_END_ISO < SP.CUTOFF_ISO
    assert SP.BASE_DEPTH == 3 and SP.BASE_COLUMNS == "existing_all+strips17"
    assert SP.BASE_PARAMS.get("n_estimators") == 900
    assert abs(float(SP.BASE_PARAMS.get("learning_rate")) - 0.03) < 1e-12
    assert SP.PATH_SOURCE == "oof-fit"
    assert SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert SP.TARGET_ARM == "correction-gain" and float(SP.GAIN_SCALE) == 1.0
    assert tuple(SP.GATE_FEATURES) == ("hind_mean", "hind_std", "past_std", "sim", "var", "h")
    assert int(SP.GATE_NFOLD) == 3 and int(SP.GATE_MIN_TRAIN_ROWS) == 50
    assert SP.ARM == "shiftsafe"
    assert "[axes:" in SP.AXES_TAG and "residual" in SP.AXES_TAG
    assert "shiftsafe" in SP.AXES_TAG or "risk" in SP.AXES_TAG
    assert "3637.84" in SP.SELECT_RULE and "5471/4984/4012/4387" in SP.SELECT_RULE
    assert "g = P(gain>0)" in SP.GATE_DEF and "falls back to base" in SP.GATE_DEF
    assert "timestamps <= t" in SP.UPDATE_DEF and "t - h" in SP.UPDATE_DEF


def test_windows_prespecified_and_causal():
    SP = _load("spec_shiftsafe")
    wins = list(SP.XOOF_WINDOWS)
    assert len(wins) == 3
    assert [w["name"] for w in wins] == ["W0-early", "W1-mid", "W2-main"]
    assert SP.MAIN_WINDOW == "W2-main"
    assert wins[0]["train_end"] == "2026-01-01T00:00:00Z"
    assert wins[1]["train_end"] == "2026-02-01T00:00:00Z"
    assert wins[-1]["train_end"] == SP.OOF_TRAIN_END_ISO == "2026-03-01T00:00:00Z"
    assert wins[-1]["oof_start"] == SP.OOF_START_ISO == "2026-03-04T00:00:00Z"
    assert wins[-1]["oof_end"] == SP.OOF_END_ISO == "2026-05-28T23:00:00Z"
    for w in wins:
        assert w["freq"] == "3h"
        assert w["train_end"] <= w["oof_start"] < w["oof_end"] < SP.CUTOFF_ISO
    for a, b in zip(wins, wins[1:]):
        assert a["train_end"] < b["train_end"], "trains must expand"
        assert a["oof_end"] < b["oof_start"], "test periods must not overlap"
    for w in wins:
        oo = pd.date_range(w["oof_start"], w["oof_end"], freq="3h", tz="UTC")
        assert len(oo) > 100, "every window must hold enough origins for a stable MSE"


def test_shiftsafe_single_axis_reuse():
    text = (EXP / "run_official_shiftsafe.py").read_text()
    assert "import shiftsafe_lib as RL" in text
    assert "import spec_shiftsafe as SP" in text
    assert "worker-t1b6" in text
    assert "predictions_official_s{s}.parquet" in text
    assert "resid_summary_s{s}.parquet" in text
    assert "residual_for_origins_with_diag" in text
    assert "fit_blend_weight_dv" in text
    assert "risk_features_for_origins" in text
    assert "blend_gated" in text and "oof_fold_ids" in text
    assert "risk_poison" in text
    assert "XOOF_WINDOWS" in text and "MAIN_WINDOW" in text
    for dup in ("def residual_for_origins_with_diag", "def fit_blend_weight_dv",
                "def scenario_distance3", "def blend_per_lead",
                "def risk_features_for_origins", "def blend_gated"):
        assert dup not in text, f"runner must reuse shiftsafe_lib, not duplicate {dup}"
    assert "not hasattr(SP, \"ALPHA_GRID\")" in text, "no alpha grid allowed"
    assert "xgb.XGBClassifier" in text and "xgb.XGBRegressor" in text
    for lab in CELL_LABELS:
        assert lab not in text, f"runner must never use the cell label {lab}"
    libtext = (EXP / "shiftsafe_lib.py").read_text()
    for lab in CELL_LABELS:
        assert lab not in libtext, f"lib must never use the cell label {lab}"
    assert "uniform" in libtext
    SP = _load("spec_shiftsafe")
    assert "max_depth" not in SP.BASE_PARAMS, "depth comes only from BASE_DEPTH"
    assert "random_state" not in SP.BASE_PARAMS and "random_state" not in SP.GATE_PARAMS
    assert SP.BASE_PARAMS.get("tree_method") == "hist"
    assert SP.BASE_PARAMS.get("device") == "cpu"
    assert SP.GATE_PARAMS.get("device") == "cpu"
    assert SP.GATE_PARAMS.get("objective") == "binary:logistic"
    assert SP.GATE_PARAMS.get("max_depth") == 2 and SP.GATE_PARAMS.get("n_estimators") == 50
    sys.path.insert(0, str(LB / "experiments/headroom_audit"))
    import info_ablation as ia
    for k, v in ia.PARAMS.items():
        if k in ("max_depth", "n_jobs", "random_state", "n_estimators", "learning_rate"):
            continue
        assert SP.BASE_PARAMS.get(k) == v, k
    sys.path.insert(0, str(EXP))
    import shiftsafe_lib as RL
    RL.check_weights(SP.WEIGHTS)
    RL.check_gate_features(SP.GATE_FEATURES)
    RL.check_blend_grid(SP.BLEND_GRID)


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
    sys.path.insert(0, str(EXP))
    import shiftsafe_lib as RL
    SP = _load("spec_shiftsafe")
    params = [RL.fit_params(dict(SP.BASE_PARAMS), SP.BASE_DEPTH, s) for s in SP.SEEDS]
    assert len({p["random_state"] for p in params}) == 3
    assert all(p["max_depth"] == 3 for p in params)
    assert RL.level_seeds(1) == (3, 4, 5) and RL.event_seed(1) == 4
    gates = {RL.gate_seed(s) for s in SP.SEEDS}
    assert len(gates) == 3, "gate RNG stream must differ per seed"
    taken = set(RL.level_seeds(0)) | {RL.event_seed(0)}
    assert not (gates & taken), "gate stream must not collide with base streams"


def test_mixture_blend_helpers():
    sys.path.insert(0, str(EXP))
    import shiftsafe_lib as RL
    SP = _load("spec_shiftsafe")
    h = np.array([1, 7, 25, 49])
    members = {m: np.full(4, 400.0 + i) for i, m in enumerate(RL.MEMBERS)}
    out = RL.blend_per_lead(members, SP.WEIGHTS, h)
    assert out.shape == (4,) and bool(np.isfinite(out).all())
    try:
        RL.check_weights({"1-6": SP.WEIGHTS["1-6"]})
    except ValueError:
        pass
    else:
        raise AssertionError("check_weights must reject partial leads")
    assert RL.lead_of(1) == "1-6" and RL.lead_of(72) == "49-72"
    try:
        RL.lead_of(0)
    except ValueError:
        pass
    else:
        raise AssertionError("lead_of must reject horizon 0")


def test_scenario_past_and_carrington():
    sys.path.insert(0, str(EXP))
    import shiftsafe_lib as RL
    idx = pd.date_range("2026-01-01", periods=2000, freq="h", tz="UTC")
    filled = pd.Series(np.arange(2000, dtype=float) + 300.0, index=idx)
    o = pd.DatetimeIndex([idx[1000]], tz="UTC")
    sc = RL.scenario_past_matrix(o, filled)
    assert sc.shape == (1, 12)
    assert np.allclose(sc[0], np.arange(989, 1001, dtype=float) + 300.0)
    cands = RL.candidate_pool(o[0], idx, len(idx))
    assert len(cands) > 0
    assert (o[0] - idx[cands]).min() >= pd.Timedelta("655h")
    assert (o[0] - idx[cands]).max() <= pd.Timedelta("8760h")


def test_error_matrix():
    sys.path.insert(0, str(EXP))
    import shiftsafe_lib as RL
    past = np.full((2, 12), 400.0)
    hind = np.full((2, 12), 390.0)
    err = RL.scenario_error_matrix(past, hind)
    assert err.shape == (2, 12) and np.allclose(err, 10.0)


def test_path_distance_equal_subspace_weight():
    sys.path.insert(0, str(EXP))
    import shiftsafe_lib as RL
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


def test_residual_replay_with_diag():
    sys.path.insert(0, str(EXP))
    import shiftsafe_lib as RL
    idx = pd.date_range("2025-01-01", periods=24 * 500, freq="h", tz="UTC")
    filled = pd.Series(400.0 + 0.05 * np.arange(len(idx)), index=idx)
    o = pd.DatetimeIndex([idx[8000], idx[8100]], tz="UTC")
    qp = np.full((2, 72), 420.0)
    qh = np.full((2, 72), 410.0)
    cand_times = idx
    cp = np.full((len(idx), 72), 420.0)
    out, cov, sim, var = RL.residual_for_origins_with_diag(o, filled, qp, qh, cand_times, cp, 15)
    assert out.shape == (2, 72) and cov.shape == (2,)
    assert sim.shape == (2,) and var.shape == (2, 72)
    assert bool(np.isfinite(out).all()) and (cov > 0).all()
    assert bool(np.isfinite(sim).all()) and (sim >= 0).all()
    assert bool(np.isfinite(var).all()) and (var >= 0).all()
    out2, cov2, sim2, var2 = RL.residual_for_origins_with_diag(o, filled, qp, qh, cand_times, cp, 15)
    assert np.array_equal(out, out2, equal_nan=True) and np.array_equal(cov, cov2)
    assert np.array_equal(sim, sim2, equal_nan=True) and np.array_equal(var, var2, equal_nan=True)
    far = pd.Timestamp("2027-01-01T00:00:00Z", tz="UTC")
    outf, covf, _, _ = RL.residual_for_origins_with_diag(o, filled, qp, qh, cand_times, cp, 15,
                                                         cutoff=far)
    assert np.array_equal(out, outf, equal_nan=True) and np.array_equal(cov, covf)
    early = pd.DatetimeIndex([idx[5]], tz="UTC")
    out3, cov3, sim3, _ = RL.residual_for_origins_with_diag(
        early, filled, np.full((1, 72), 420.0), np.full((1, 72), 410.0),
        cand_times, cp, 15)
    assert bool(np.isnan(out3).all()) and cov3[0] == 0 and bool(np.isnan(sim3).all())


def test_risk_features_are_past_only():
    sys.path.insert(0, str(EXP))
    import shiftsafe_lib as RL
    idx = pd.date_range("2026-03-01", periods=24 * 60, freq="h", tz="UTC")
    filled = pd.Series(400.0 + 10.0 * np.sin(np.arange(len(idx)) / 24.0), index=idx)
    o = pd.DatetimeIndex([idx[500], idx[600]], tz="UTC")
    qh = np.full((2, 72), 395.0)
    hm, hs, ps = RL.risk_features_for_origins(o, filled, qh)
    assert hm.shape == (2,) and hs.shape == (2,) and ps.shape == (2,)
    assert bool(np.isfinite(hm).all()) and bool(np.isfinite(hs).all())
    assert bool(np.isfinite(ps).all()) and bool((hs >= 0).all()) and bool((ps >= 0).all())
    # Perturbing strictly-future filled speeds must not move these rows.
    pert = filled.copy()
    pert.loc[pert.index > o.max()] = pert.loc[pert.index > o.max()] + 50.0
    hm2, hs2, ps2 = RL.risk_features_for_origins(o, pert, qh)
    assert np.array_equal(hm, hm2, equal_nan=True)
    assert np.array_equal(hs, hs2, equal_nan=True)
    assert np.array_equal(ps, ps2, equal_nan=True)
    # A gap at the origin yields NaN risk (caller falls back to base).
    gap = filled.copy()
    gap.loc[o[0]] = np.nan
    hmg, _, _ = RL.risk_features_for_origins(pd.DatetimeIndex([o[0]], tz="UTC"), gap,
                                             np.full((1, 72), 395.0))
    assert bool(np.isnan(hmg).all())


def test_gate_helpers():
    sys.path.insert(0, str(EXP))
    import shiftsafe_lib as RL
    SP = _load("spec_shiftsafe")
    RL.check_gate_features(SP.GATE_FEATURES)
    try:
        RL.check_gate_features(("base", "h", "sim", "var", "disp"))
    except ValueError:
        pass
    else:
        raise AssertionError("check_gate_features must reject the 5-col form")
    base = np.full((3, 72), 400.0)
    hm = np.array([5.0, -3.0, np.nan])
    hs = np.array([8.0, 8.0, 8.0])
    ps = np.array([12.0, 12.0, np.nan])
    sim = np.array([10.0, 20.0, np.nan])
    var = np.full((3, 72), 4.0)
    var[0, 0] = np.nan
    fills = {"hind_mean": 0.0, "hind_std": 8.0, "past_std": 12.0, "sim": 0.0, "var": 4.0}
    x = RL.gate_feature_matrix(base, hm, hs, ps, sim, var, fills)
    assert x.shape == (216, 6)
    assert np.allclose(x[:72, 0], 5.0) and np.allclose(x[72:144, 0], -3.0)
    assert np.allclose(x[144:, 0], 0.0), "NaN hind_mean takes the OOF fill"
    assert np.allclose(x[144:, 2], 12.0), "NaN past_std takes the OOF fill"
    assert np.allclose(x[:72, 3], 10.0) and np.allclose(x[72:144, 3], 20.0)
    assert bool((x[144:, 3] == np.inf).all()), "NaN sim reads as far"
    assert x[0, 4] == 4.0, "NaN var takes the OOF fill"
    assert np.allclose(x[:72, 5], np.arange(1, 73)), "h tiles 1..72 per origin"
    fills2 = RL.oof_median_fills(hm, hs, ps, var)
    assert set(fills2) >= {"hind_mean", "hind_std", "past_std", "var"}
    assert all(np.isfinite(v) for v in fills2.values())
    # Gain labels at the frozen weight: delta = w*dv.
    dv = np.full((3, 72), 10.0)
    delta = 0.5 * dv
    y = np.full(216, 405.0)
    gain = RL.gate_gain_labels(base, delta, y)
    assert gain.shape == (216,) and bool(np.isfinite(gain).all())
    assert np.allclose(gain, (400 - 405) ** 2 - (400 + 5 - 405) ** 2)
    lab = RL.gain_to_label(gain)
    assert lab.shape == (216,) and set(np.unique(lab)) == {1}
    dv_bad = dv.copy()
    dv_bad[0, :5] = np.nan
    gain2 = RL.gate_gain_labels(base, 0.5 * dv_bad, y)
    assert bool(np.isnan(gain2[:5]).all()) and bool(np.isfinite(gain2[5:]).all())
    lab2 = RL.gain_to_label(gain2)
    assert (lab2[:5] == -1).all(), "non-finite gains are excluded from training"
    # Gated blend: high risk (g=0) and NaN inputs keep base.
    out = RL.blend_gated(np.full(4, 400.0), np.array([10.0, 10.0, np.nan, 10.0]),
                         np.array([0.5, 0.0, 0.5, np.nan]))
    assert np.allclose(out, [405.0, 400.0, 400.0, 400.0])
    assert bool((RL.clip_proba(np.array([-1.0, 2.0])) == np.array([0.0, 1.0])).all())
    # Chronological folds: ordered, contiguous, cover all origins.
    fids = RL.oof_fold_ids(10, 3)
    assert fids.shape == (10,) and set(np.unique(fids)) == {0, 1, 2}
    assert bool((np.diff(fids) >= 0).all()), "folds must be chronological"
    try:
        RL.oof_fold_ids(0, 3)
    except AssertionError:
        pass
    else:
        raise AssertionError("oof_fold_ids must reject empty OOF")


def test_blend_weight_grid():
    sys.path.insert(0, str(EXP))
    import shiftsafe_lib as RL
    SP = _load("spec_shiftsafe")
    RL.check_blend_grid(SP.BLEND_GRID)
    rng = np.random.default_rng(0)
    b = rng.normal(400, 50, 500)
    dv = rng.normal(0, 10, 500)
    y = rng.normal(400, 50, 500)
    w, base_mse, best_mse, mses = RL.fit_blend_weight_dv(b, dv, y, SP.BLEND_GRID)
    assert w in tuple(SP.BLEND_GRID) and len(mses) == 21 and best_mse <= base_mse + 1e-9
    out = RL.blend_dv(np.full(3, 400.0), np.array([10.0, np.nan, -4.0]), 0.5)
    assert np.allclose(out, [405.0, 400.0, 398.0])


def test_output_frame_schema():
    sys.path.insert(0, str(EXP))
    import shiftsafe_lib as RL
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = RL.frame_predictions(te, np.full(72, 400.0))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert (fr["horizon_hours"].to_numpy() == np.arange(1, 73)).all()
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_poison_keys_and_invariance():
    sys.path.insert(0, str(EXP))
    import shiftsafe_lib as RL
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

    def _risk_fn(f):
        hm, hs, ps = RL.risk_features_for_origins(probe, f, np.full((4, 72), 395.0))
        return np.column_stack([hm, hs, ps])

    rec3 = RL.risk_poison(probe, filled, cut, _risk_fn)
    assert rec3["passed"] is True
    assert rec3["pre_invariant"] is True and rec3["moved_columns"] == ["risk"]
    assert rec3["n_pre_rows"] == 2 and rec3["n_post_rows"] == 2
