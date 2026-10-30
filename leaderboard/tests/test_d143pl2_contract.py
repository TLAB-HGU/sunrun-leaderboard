"""D143 PIR-lite contract: static + synthetic checks (fast, no fits, no data)."""
import importlib.util

import numpy as np
import pandas as pd
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d143_axisA_analog" / "pirlite"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_IDS = ["e0147-abl", "d141_e0147", "ablate", "worker-d76", "worker-t1d7",
             "worker-t1a1", "worker-t1a2", "worker-t1a3", "worker-t1a5",
             "worker-t1a6", "worker-t1a7", "worker-t1a8",
             "worker-d135", "worker-d133", "worker-t2no", "nn-"]
BANNED_SRC = ["GONG", "WSA", "ADAPT", "CME", "DONKI"]


def _load(name):
    loc = importlib.util.spec_from_file_location(f"d143pl2_{name}", EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(EXP.glob("*.py")))


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 6, files
    assert {"spec_pirlite.py", "pirlite_lib.py", "run_official_pirlite.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_pirlite")
    assert SP.DIRECTION == "D143" and SP.OWNER == "worker-t1c0"
    assert SP.KIND == "full" and SP.REFERENCE == "E0421"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h"
    assert SP.SCRATCH == "store/scratch/worker-t1c0"
    assert SP.FIT_NJOBS <= 20 and int(SP.FIT_NJOBS) + int(SP.GATE_NJOBS) <= 20
    assert SP.K == 15 and SP.WEIGHTING == "uniform"
    assert SP.SCENARIO_H == 12 and SP.ERR_H == 12 and SP.PATH_H == 72
    assert SP.SCENARIO_DIM == 96
    assert SP.TARGET_ARM == "correction-gain"
    assert float(SP.GAIN_SCALE) == 1.0
    assert not hasattr(SP, "ALPHA_GRID"), "PIR-lite must not carry a coarse alpha grid"
    assert tuple(SP.GATE_FEATURES) == ("base", "h", "sim", "var", "disp")
    assert abs(SP.CARRINGTON_H - 654.48) < 1e-9
    assert SP.OOF_TRAIN_END_ISO < SP.OOF_START_ISO <= SP.OOF_END_ISO < SP.CUTOFF_ISO
    assert SP.BASE_DEPTH == 3 and SP.BASE_COLUMNS == "existing_all+strips17"
    assert SP.BASE_PARAMS.get("n_estimators") == 900
    assert abs(float(SP.BASE_PARAMS.get("learning_rate")) - 0.03) < 1e-12
    assert SP.GATE_PARAMS.get("objective") == "binary:logistic"
    assert SP.GATE_PARAMS.get("n_estimators") == 50 and SP.GATE_PARAMS.get("max_depth") == 2
    assert SP.PATH_SOURCE == "oof-fit"
    assert SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert "[axes:" in SP.AXES_TAG and "pirlite" in SP.AXES_TAG
    assert "3637.84" in SP.SELECT_RULE and "5471/4984/4012/4387" in SP.SELECT_RULE
    assert "P(gain>0)" in SP.GATE_DEF and "disp" in SP.GATE_DEF and "sim" in SP.GATE_DEF


def test_pirlite_single_axis_vs_e0421():
    SP = _load("spec_pirlite")
    assert SP.K == 15 and SP.WEIGHTING == "uniform", "E0421 residual form is K=15 uniform"
    assert SP.SCENARIO_DIM == 96 and SP.PATH_SOURCE == "oof-fit"
    assert SP.BASE_PARAMS.get("tree_method") == "hist"
    assert SP.BASE_PARAMS.get("device") == "cpu"
    assert SP.GATE_PARAMS.get("device") == "cpu" and SP.GATE_PARAMS.get("tree_method") == "hist"
    assert "max_depth" not in SP.BASE_PARAMS, "depth comes only from BASE_DEPTH"
    assert "random_state" not in SP.BASE_PARAMS and "random_state" not in SP.GATE_PARAMS
    text = (EXP / "run_official_pirlite.py").read_text()
    assert "import spec_pirlite as SP" in text
    assert "SP.K == 15" in text
    assert "worker-t1c0" in text
    assert "XGBClassifier" in text and "predict_proba" in text
    assert "ALPHA_GRID" not in text or "not hasattr(SP, \"ALPHA_GRID\")" in text
    import sys
    sys.path.insert(0, str(LB / "experiments/headroom_audit"))
    import info_ablation as ia
    for k, v in ia.PARAMS.items():
        if k in ("max_depth", "n_jobs", "random_state", "n_estimators", "learning_rate"):
            continue
        assert SP.BASE_PARAMS.get(k) == v, k
    RL = _load("pirlite_lib")
    RL.check_weights(SP.WEIGHTS)


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
    RL = _load("pirlite_lib")
    SP = _load("spec_pirlite")
    params = [RL.fit_params(dict(SP.BASE_PARAMS), SP.BASE_DEPTH, s) for s in SP.SEEDS]
    assert len({p["random_state"] for p in params}) == 3
    assert all(p["max_depth"] == 3 for p in params)
    assert RL.level_seeds(1) == (3, 4, 5) and RL.event_seed(1) == 4
    assert RL.level_seeds(0) == (0, 1, 2) and RL.event_seed(0) == 1
    assert RL.level_seeds(2) == (6, 7, 8) and RL.event_seed(2) == 7
    gates = {RL.gate_seed(s) for s in SP.SEEDS}
    assert len(gates) == 3, "gate RNG stream must differ per seed"


def test_mixture_blend_helpers():
    RL = _load("pirlite_lib")
    SP = _load("spec_pirlite")
    h = np.arange(1, 73)
    members = {m: np.full(72, 400.0) for m in RL.MEMBERS}
    out = RL.blend_per_lead(members, SP.WEIGHTS, h)
    assert np.allclose(out, 400.0)
    s1 = {s: np.full(6, 400.0) for s in (0, 1, 2)}
    s2 = {s: np.full(18, 410.0) for s in (0, 1, 2)}
    m1 = h <= 6
    m2 = (h >= 7) & (h <= 24)
    m3 = h > 24
    st = RL.stitch_members(s1, s2, np.full(72, 420.0), np.full(48, 430.0), m1, m2, m3, (0, 1, 2), 1)
    assert set(st.keys()) == set(RL.MEMBERS)
    assert np.allclose(st["e0003_line"][:6], 420.0)


def test_scenario_past_and_carrington():
    RL = _load("pirlite_lib")
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
    RL = _load("pirlite_lib")
    past = np.full((2, 12), 400.0)
    hind = np.full((2, 12), 390.0)
    err = RL.scenario_error_matrix(past, hind)
    assert err.shape == (2, 12) and np.allclose(err, 10.0)


def test_path_distance_equal_subspace_weight():
    RL = _load("pirlite_lib")
    d_past = RL.scenario_distance3(np.full((1, 12), 400.0), np.full((1, 12), 10.0), np.full((1, 72), 402.0),
                                   np.full((1, 12), 400.0), np.full((1, 12), 10.0), np.full((1, 72), 402.0))
    assert np.allclose(d_past, [0.0])
    d2 = RL.scenario_distance3(np.zeros((1, 12)), np.zeros((1, 12)), np.zeros((1, 72)),
                               np.full((1, 12), 2.0), np.full((1, 12), 2.0), np.full((1, 72), 2.0))
    d_past_only = RL.scenario_distance3(np.zeros((1, 12)), np.zeros((1, 12)), np.zeros((1, 72)),
                                        np.full((1, 12), 2.0), np.zeros((1, 12)), np.zeros((1, 72)))
    d_err_only = RL.scenario_distance3(np.zeros((1, 12)), np.zeros((1, 12)), np.zeros((1, 72)),
                                       np.zeros((1, 12)), np.full((1, 12), 2.0), np.zeros((1, 72)))
    d_path_only = RL.scenario_distance3(np.zeros((1, 12)), np.zeros((1, 12)), np.zeros((1, 72)),
                                        np.zeros((1, 12)), np.zeros((1, 12)), np.full((1, 72), 2.0))
    assert np.allclose(d2, [12.0])
    assert np.allclose(d_past_only, d_err_only) and np.allclose(d_err_only, d_path_only)


def test_as_path_matrix():
    RL = _load("pirlite_lib")
    m = RL.as_path_matrix(np.arange(144, dtype=float))
    assert m.shape == (2, 72)
    try:
        RL.as_path_matrix(np.arange(100, dtype=float))
    except AssertionError:
        pass
    else:
        raise AssertionError("as_path_matrix must reject non-multiples of 72")


def test_residual_replay_with_diag_and_disp():
    RL = _load("pirlite_lib")
    idx = pd.date_range("2025-01-01", periods=24 * 500, freq="h", tz="UTC")
    filled = pd.Series(400.0 + 0.05 * np.arange(len(idx)), index=idx)
    o = pd.DatetimeIndex([idx[8000], idx[8100]], tz="UTC")
    qp = np.full((2, 72), 420.0)
    qh = np.full((2, 72), 410.0)
    cand_times = idx
    cp = np.full((len(idx), 72), 420.0)
    out, cov, sim, var, disp = RL.residual_for_origins_with_diag(o, filled, qp, qh, cand_times, cp, 15)
    assert out.shape == (2, 72) and cov.shape == (2,)
    assert sim.shape == (2,) and var.shape == (2, 72) and disp.shape == (2, 72)
    assert bool(np.isfinite(out).all()) and (cov > 0).all()
    assert bool(np.isfinite(sim).all()) and (sim >= 0).all()
    assert bool(np.isfinite(var).all()) and (var >= 0).all()
    assert bool(np.isfinite(disp).all()) and (disp >= 0).all()
    assert bool((disp <= 6 * np.sqrt(np.maximum(var, 0)) + 1e-6).all()), "IQR must be bounded by spread"
    out2, cov2, sim2, var2, disp2 = RL.residual_for_origins_with_diag(o, filled, qp, qh, cand_times, cp, 15)
    assert np.array_equal(out, out2, equal_nan=True) and np.array_equal(cov, cov2)
    assert np.array_equal(sim, sim2, equal_nan=True) and np.array_equal(var, var2, equal_nan=True)
    assert np.array_equal(disp, disp2, equal_nan=True)
    far = pd.Timestamp("2027-01-01T00:00:00Z", tz="UTC")
    outf, covf, _, _, _ = RL.residual_for_origins_with_diag(o, filled, qp, qh, cand_times, cp, 15,
                                                            cutoff=far)
    assert np.array_equal(out, outf, equal_nan=True) and np.array_equal(cov, covf)
    early = pd.DatetimeIndex([idx[5]], tz="UTC")
    out3, cov3, sim3, _, disp3 = RL.residual_for_origins_with_diag(
        early, filled, np.full((1, 72), 420.0), np.full((1, 72), 410.0),
        cand_times, cp, 15)
    assert bool(np.isnan(out3).all()) and cov3[0] == 0 and bool(np.isnan(sim3).all())
    assert bool(np.isnan(disp3).all())


def test_gate_helpers_continuous():
    RL = _load("pirlite_lib")
    SP = _load("spec_pirlite")
    base = np.full((3, 72), 400.0)
    d2m = np.array([10.0, 20.0, np.nan])
    var = np.full((3, 72), 4.0)
    var[0, 0] = np.nan
    disp = np.full((3, 72), 2.0)
    disp[1, 0] = np.nan
    x = RL.gate_feature_matrix(base, d2m, var, disp, 4.0, 2.0)
    assert x.shape == (216, 5)
    assert np.allclose(x[:72, 1], np.arange(1, 73)), "h tiles 1..72 per origin"
    assert np.allclose(x[:72, 2], 10.0) and np.allclose(x[72:144, 2], 20.0)
    assert bool((x[144:, 2] == np.inf).all()), "NaN sim maps to +inf"
    assert x[0, 3] == 4.0, "NaN var takes the OOF fill"
    assert x[72, 4] == 2.0, "NaN disp takes the OOF fill"
    y = np.full(216, 405.0)
    dv = np.full((3, 72), 10.0)
    gain = RL.gate_gain_labels(base, dv, y)
    assert gain.shape == (216,) and bool(np.isfinite(gain).all())
    assert np.allclose(gain, (400 - 405) ** 2 - (400 + 10.0 - 405) ** 2)
    dv_bad = dv.copy()
    dv_bad[0, :5] = np.nan
    gain2 = RL.gate_gain_labels(base, dv_bad, y)
    assert bool(np.isnan(gain2[:5]).all()) and bool(np.isfinite(gain2[5:]).all())
    lab = RL.gain_to_label(gain)
    assert set(np.unique(lab)).issubset({0, 1})
    lab2 = RL.gain_to_label(gain2)
    assert (lab2[:5] == -1).all() and set(np.unique(lab2[5:])).issubset({0, 1})
    neg = RL.gain_to_label(np.full(10, -3.0))
    assert (neg == 0).all()
    g = RL.clip_proba(np.array([-0.5, 0.25, 1.5]))
    assert np.allclose(g, [0.0, 0.25, 1.0])
    out = RL.blend_continuous(np.full(4, 400.0), np.array([10.0, 10.0, np.nan, 10.0]),
                              np.array([0.5, 0.0, 0.9, 1.0]))
    assert np.allclose(out, [405.0, 400.0, 400.0, 410.0]), "g scales dv; NaN-dv stays at base"
    out2 = RL.blend_continuous(np.full(3, 400.0), np.full(3, 10.0), np.array([0.0, 0.0, 0.0]))
    assert np.allclose(out2, 400.0), "g=0 recovers base exactly"


def test_output_frame_schema():
    RL = _load("pirlite_lib")
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = RL.frame_predictions(te, np.full(72, 400.0))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert (fr["horizon_hours"].to_numpy() == np.arange(1, 73)).all()
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_poison_keys_and_invariance():
    RL = _load("pirlite_lib")
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
