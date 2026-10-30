"""D143 PRR contract: static + synthetic checks (fast, no data reads)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d143_axisA_analog" / "prr"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_IDS = ["e0147-abl", "d141_e0147", "ablate", "worker-d76", "worker-t1d7",
             "worker-t1a1", "worker-t1a2", "worker-t1a3", "worker-t1a5",
             "worker-t1a6", "worker-t1a7", "worker-t1a8", "worker-t1b0",
             "worker-t1b1", "worker-t1b3", "worker-t1b4", "worker-t1c0",
             "worker-t1c1", "worker-t1c2", "worker-d135", "worker-d133",
             "worker-t2no", "nn-"]
BANNED_SRC = ["GONG", "WSA", "ADAPT", "CME", "DONKI"]


def _load(name):
    loc = importlib.util.spec_from_file_location(f"d143prr_{name}", EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(EXP.glob("*.py")))


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 6, files
    assert {"spec_prr.py", "prr_lib.py", "run_official_prr.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_prr")
    assert SP.DIRECTION == "D143" and SP.OWNER == "worker-t1b2"
    assert SP.KIND == "full" and SP.REFERENCE == "E0421"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h"
    assert SP.SCRATCH == "store/scratch/worker-t1b2"
    assert SP.FIT_NJOBS <= 20
    assert SP.K == 15 and SP.COARSE_K == 50
    assert SP.SCENARIO_H == 12 and SP.ERR_H == 12 and SP.PATH_H == 72
    assert SP.SCENARIO_DIM == 96
    assert abs(SP.CARRINGTON_H - 654.48) < 1e-9
    assert len(SP.BLEND_GRID) == 21 and SP.BLEND_GRID[0] == 0.0 and SP.BLEND_GRID[-1] == 1.0
    assert SP.RERANK_TRAIN_END_ISO < SP.OOF_START_ISO
    assert SP.OOF_TRAIN_END_ISO < SP.OOF_START_ISO <= SP.OOF_END_ISO < SP.CUTOFF_ISO
    assert SP.RERANK_TRAIN_END_ISO < SP.CUTOFF_ISO
    assert SP.BASE_DEPTH == 3 and SP.BASE_COLUMNS == "existing_all+strips17"
    assert SP.BASE_PARAMS.get("n_estimators") == 900
    assert abs(float(SP.BASE_PARAMS.get("learning_rate")) - 0.03) < 1e-12
    assert SP.PATH_SOURCE == "oof-fit"
    assert SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert SP.RANK_DEF.startswith("rank:pairwise")
    assert "uniform mean" in SP.RANK_DEF
    assert "retrieval" in SP.RETRIEVAL_DEF.lower()
    assert "[axes:" in SP.AXES_TAG and "prr" in SP.AXES_TAG
    assert tuple(SP.RANK_FEATURES) == tuple(_load("prr_lib").RANK_FEATURE_NAMES)
    assert len(SP.RANK_FEATURES) == 29
    RP = dict(SP.RANK_PARAMS)
    assert RP.get("objective") == "rank:pairwise"
    assert RP.get("device") == "cpu"
    assert int(RP.get("n_jobs")) <= 20
    assert int(RP.get("max_depth")) <= 4
    assert int(RP.get("n_estimators")) <= 300


def test_prr_single_axis_vs_e0421():
    SP = _load("spec_prr")
    assert "max_depth" not in SP.BASE_PARAMS, "depth comes only from BASE_DEPTH"
    assert SP.BASE_PARAMS.get("tree_method") == "hist"
    assert SP.BASE_PARAMS.get("device") == "cpu"
    assert SP.WEIGHTS["1-6"]["e0003_line"] == 0.21536233464375223
    assert SP.WEIGHTS["7-24"]["d33_hsplit"] == 0.3436571856823209
    assert SP.K == 15
    assert SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert SP.PATH_SOURCE == "oof-fit"
    import sys
    sys.path.insert(0, str(LB / "experiments/headroom_audit"))
    import info_ablation as ia
    for k, v in ia.PARAMS.items():
        if k in ("max_depth", "n_jobs", "random_state", "n_estimators", "learning_rate"):
            continue
        assert SP.BASE_PARAMS.get(k) == v, k
    RL = _load("prr_lib")
    RL.check_weights(SP.WEIGHTS)
    RL.check_blend_grid(SP.BLEND_GRID)
    RL.check_coarse_final(SP.COARSE_K, SP.K)
    RL.check_rank_params(SP.RANK_PARAMS)
    text = (EXP / "run_official_prr.py").read_text()
    assert "import spec_prr as SP" in text
    assert "import prr_lib as RL" in text
    assert "worker-t1b2" in text
    assert "predictions_official_s{s}.parquet" in text
    assert "reranked_residual_for_origins" in text
    assert "train_ranker" in text
    assert "retrieval_mse" in text
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
    RL = _load("prr_lib")
    SP = _load("spec_prr")
    params = [RL.fit_params(dict(SP.BASE_PARAMS), SP.BASE_DEPTH, s) for s in SP.SEEDS]
    assert len({p["random_state"] for p in params}) == 3
    assert all(p["max_depth"] == 3 for p in params)
    assert RL.level_seeds(1) == (3, 4, 5) and RL.event_seed(1) == 4
    rp = [RL.rank_params_for_seed(SP.RANK_PARAMS, s)["random_state"] for s in SP.SEEDS]
    assert len(set(rp)) == 3


def test_mixture_blend_helpers():
    RL = _load("prr_lib")
    SP = _load("spec_prr")
    h = np.arange(1, 73)
    members = {m: np.full(72, 400.0) for m in RL.MEMBERS}
    out = RL.blend_per_lead(members, SP.WEIGHTS, h)
    assert np.allclose(out, 400.0)
    st = RL.stitch_members({0: np.full(6, 1.0)}, {0: np.full(18, 2.0)},
                            np.full(72, 420.0), np.full(48, 3.0),
                            h <= 6, (h >= 7) & (h <= 24), h > 24, (0,), 0)
    assert np.allclose(st["e0003_line"][:6], 420.0)


def test_scenario_past_and_carrington():
    RL = _load("prr_lib")
    idx = pd.date_range("2025-01-01", periods=24 * 60, freq="h", tz="UTC")
    filled = pd.Series(400.0, index=idx)
    o = pd.DatetimeIndex([idx[1000]], tz="UTC")
    m = RL.scenario_past_matrix(o, filled)
    assert m.shape == (1, 12) and np.allclose(m, 400.0)
    cands = RL.candidate_pool(o[0], idx, len(idx))
    assert len(cands) > 0
    assert (o[0] - idx[cands]).min() >= pd.Timedelta("655h")
    assert (o[0] - idx[cands]).max() <= pd.Timedelta("8760h")
    e = RL.scenario_error_matrix(np.full((1, 12), 400.0), np.full((1, 12), 390.0))
    assert np.allclose(e, 10.0)
    dp, de, dh = RL.scenario_distance_parts(
        np.full(12, 400.0), np.full(12, 10.0), np.full(72, 400.0),
        np.full((3, 12), 402.0), np.full((3, 12), 10.0), np.full((3, 72), 400.0))
    assert np.allclose(dp, [4.0]) and np.allclose(de, [0.0]) and np.allclose(dh, [0.0])
    RL.check_coarse_final(50, 15)
    try:
        RL.check_coarse_final(10, 15)
    except ValueError:
        pass
    else:
        raise AssertionError("check_coarse_final must reject coarse < final")


def test_pair_features_origin_only_and_label():
    RL = _load("prr_lib")
    SP = _load("spec_prr")
    rng = np.random.default_rng(0)
    q_past = rng.normal(400, 20, 12)
    q_err = rng.normal(0, 10, 12)
    q_path = rng.normal(400, 30, 72)
    c_past = rng.normal(400, 20, (5, 12))
    c_err = rng.normal(0, 10, (5, 12))
    c_path = rng.normal(400, 30, (5, 72))
    dp, de, dh = RL.scenario_distance_parts(q_past, q_err, q_path, c_past, c_err, c_path)
    X = RL.pair_feature_matrix(
        q_past, q_err, q_path, 400.0, 15.0, np.array([0.01, 0.005]),
        c_past, c_err, c_path, np.full(5, 400.0), np.full(5, 15.0),
        np.tile(np.array([0.01, 0.005]), (5, 1)),
        dp, de, dh, dp + de + dh, np.arange(1, 6, dtype=float))
    assert X.shape == (5, len(SP.RANK_FEATURES))
    assert np.allclose(X[:, 0], dp) and np.allclose(X[:, 3], dp + de + dh)
    # origin-only: shifting the query future must not change features;
    # moving one candidate past level moves only its row.
    X2 = RL.pair_feature_matrix(
        q_past, q_err, q_path, 400.0, 15.0, np.array([0.01, 0.005]),
        c_past + np.array([[50.0] + [0.0] * 11] + [[0.0] * 12] * 4),
        c_err, c_path, np.full(5, 400.0), np.full(5, 15.0),
        np.tile(np.array([0.01, 0.005]), (5, 1)),
        dp, de, dh, dp + de + dh, np.arange(1, 6, dtype=float))
    assert not np.allclose(X2[0], X[0])
    assert np.allclose(X2[1:], X[1:])
    curves = rng.normal(0, 10, (5, 72))
    true = rng.normal(0, 10, 72)
    lab = RL.label_neg_mse(curves, true)
    assert lab.shape == (5,) and bool(np.isfinite(lab).all())
    best = int(np.argmin(np.mean((curves - true) ** 2, axis=1)))
    assert int(np.argmax(lab)) == best
    assert RL.retrieval_mse(curves.mean(0), true) >= 0.0
    assert np.isnan(RL.retrieval_mse(np.full(72, np.nan), true))


def test_rank_params_cpu_pairwise_shallow():
    RL = _load("prr_lib")
    SP = _load("spec_prr")
    RL.check_rank_params(SP.RANK_PARAMS)
    p = RL.rank_params_for_seed(SP.RANK_PARAMS, 1)
    assert p["random_state"] == 1 and p["objective"] == "rank:pairwise"
    assert p["device"] == "cpu"
    try:
        RL.check_rank_params({**SP.RANK_PARAMS, "objective": "rank:ndcg"})
    except ValueError:
        pass
    else:
        raise AssertionError("check_rank_params must reject non-pairwise objective")
    try:
        RL.check_rank_params({**SP.RANK_PARAMS, "device": "cuda"})
    except ValueError:
        pass
    else:
        raise AssertionError("check_rank_params must reject GPU ranker")


def test_reranked_uniform_mean_and_fallback():
    RL = _load("prr_lib")
    idx = pd.date_range("2025-01-01", periods=24 * 500, freq="h", tz="UTC")
    filled = pd.Series(400.0 + 0.05 * np.arange(len(idx)), index=idx)
    o = pd.DatetimeIndex([idx[8000], idx[8100]], tz="UTC")
    qp = np.full((2, 72), 420.0)
    qh = np.full((2, 72), 410.0)
    cand_times = idx
    cp = np.full((len(idx), 72), 420.0)
    out, cov = RL.residual_for_origins(o, filled, qp, qh, cand_times, cp, 5)
    assert out.shape == (2, 72) and cov.shape == (2,)
    assert bool(np.isfinite(out).all()) and (cov > 0).all()

    class _TopFirst:
        def predict(self, X):
            return np.arange(len(X))[::-1].astype(float)

    tm, ts = RL.trailing_regime(filled)
    ch = pd.DataFrame({"ch_area": np.full(len(idx), 0.01),
                       "ch_eq": np.full(len(idx), 0.005)}, index=idx)
    out2, cov2 = RL.reranked_residual_for_origins(
        o, filled, qp, qh, cand_times, cp, 10, 5, _TopFirst(), tm, ts, ch)
    assert out2.shape == (2, 72) and cov2.shape == (2,)
    assert bool(np.isfinite(out2).all()) and (cov2 == 5).all()
    # uniform mean: reranked output with a constant ranker equals the coarse
    # top-K mean only when scores tie; check finiteness and determinism.
    out3, cov3 = RL.reranked_residual_for_origins(
        o, filled, qp, qh, cand_times, cp, 10, 5, _TopFirst(), tm, ts, ch)
    assert np.array_equal(out2, out3, equal_nan=True) and np.array_equal(cov2, cov3)


def test_blend_dv_convex_and_oof_only():
    RL = _load("prr_lib")
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
    RL = _load("prr_lib")
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = RL.frame_predictions(te, np.full(72, 400.0))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert (fr["horizon_hours"].to_numpy() == np.arange(1, 73)).all()
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_poison_keys_and_invariance():
    RL = _load("prr_lib")
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
