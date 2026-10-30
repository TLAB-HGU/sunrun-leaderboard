"""D143 FP contract: static + synthetic checks (fast, no fits, no data)."""
import importlib.util

import numpy as np
import pandas as pd
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d143_axisA_analog" / "frozenpool"
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
    loc = importlib.util.spec_from_file_location(f"d143fp_{name}", EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(EXP.glob("*.py")))


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 6, files
    assert {"spec_fp.py", "frozenpool_lib.py",
            "run_official_fp.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_fp")
    assert SP.DIRECTION == "D143" and SP.OWNER == "worker-t1b7"
    assert SP.KIND == "full" and SP.REFERENCE == "E0421"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h"
    assert SP.SCRATCH == "store/scratch/worker-t1b7"
    assert SP.FIT_NJOBS <= 6
    assert SP.K == 15
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
    assert "[axes:" in SP.AXES_TAG and "ens=frozen-pool" in SP.AXES_TAG


def test_fp_single_axis_pinned():
    SP = _load("spec_fp")
    text = (EXP / "run_official_fp.py").read_text()
    assert "import spec_fp as SP" in text
    assert "import frozenpool_lib as RL" in text
    assert "worker-t1b7" in text
    assert "predictions_official_s{s}.parquet" in text
    assert "pool_detail_s{s}.parquet" in text
    assert 'EXPERTS == ("E1-unbounded", "E2-capped50")' in text or "check_experts(SP.EXPERTS)" in text
    assert "apply_pool(e1_off, e2_off" in text
    assert "fit_pool_weights" in text and "shrink_pool_weights" in text
    assert tuple(SP.EXPERTS) == ("E1-unbounded", "E2-capped50")
    assert SP.EXPERT_DEF and "E1" in SP.EXPERT_DEF and "E2" in SP.EXPERT_DEF
    assert tuple(SP.BANDS) == ((1, 6), (7, 24), (25, 72))
    assert tuple(SP.BAND_NAMES) == ("1-6", "7-24", "25-72")
    assert float(SP.CAP) == 50.0 and int(SP.MASK_MIN_WINS) == 2
    assert len(SP.POOL_GRID) == 21 and SP.POOL_GRID[0] == 0.0 and SP.POOL_GRID[-1] == 1.0
    assert 0.0 <= float(SP.SHRINK_TO_E1) <= 1.0 and abs(float(SP.SHRINK_TO_E1) - 0.25) < 1e-12
    assert SP.POOL_DEF and "a_b" in SP.POOL_DEF
    assert SP.POOL_RULE and "SHRINK" in SP.POOL_RULE
    assert [w["name"] for w in SP.XOOF_WINDOWS] == ["W0-early", "W1-mid", "W2-main"]
    assert SP.MAIN_WINDOW == "W2-main"
    # Base recipe identical to parents (existing_all+17 strips, CPU hist).
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
    RL = _load("frozenpool_lib")
    RL.check_weights(SP.WEIGHTS)
    RL.check_bands(SP.BANDS)
    RL.check_band_names(SP.BAND_NAMES)
    RL.check_pool_grid(SP.POOL_GRID)
    RL.check_shrink(SP.SHRINK_TO_E1)
    RL.check_experts(SP.EXPERTS)


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
    RL = _load("frozenpool_lib")
    SP = _load("spec_fp")
    params = [RL.fit_params(dict(SP.BASE_PARAMS), SP.BASE_DEPTH, s) for s in SP.SEEDS]
    assert len({p["random_state"] for p in params}) == 3
    assert all(p["max_depth"] == 3 for p in params)
    assert RL.level_seeds(1) == (3, 4, 5) and RL.event_seed(1) == 4


def test_mixture_blend_helpers():
    RL = _load("frozenpool_lib")
    SP = _load("spec_fp")
    h = np.arange(1, 73)
    members = {m: np.full(72, 400.0) for m in RL.MEMBERS}
    out = RL.blend_per_lead(members, SP.WEIGHTS, h)
    assert np.allclose(out, 400.0)
    st = RL.stitch_members({0: np.full(6, 1.0)}, {0: np.full(18, 2.0)},
                            np.full(72, 420.0), np.full(48, 3.0),
                            h <= 6, (h >= 7) & (h <= 24), h > 24, (0,), 0)
    assert np.allclose(st["e0003_line"][:6], 420.0)


def test_scenario_past_and_carrington():
    RL = _load("frozenpool_lib")
    idx = pd.date_range("2025-01-01", periods=24 * 60, freq="h", tz="UTC")
    filled = pd.Series(400.0, index=idx)
    o = pd.DatetimeIndex([idx[1000]], tz="UTC")
    m = RL.scenario_past_matrix(o, filled)
    assert m.shape == (1, 12) and np.allclose(m, 400.0)
    cands = RL.candidate_pool(o[0], idx, len(idx))
    assert len(cands) > 0
    assert (o[0] - idx[cands]).max() >= pd.Timedelta("654.48h") - pd.Timedelta("1h")
    assert (o[0] - idx[cands]).max() <= pd.Timedelta("8760h")


def test_error_matrix():
    RL = _load("frozenpool_lib")
    past = np.full((2, 12), 420.0)
    hind = np.full((2, 12), 410.0)
    err = RL.scenario_error_matrix(past, hind)
    assert err.shape == (2, 12) and np.allclose(err, 10.0)


def test_band_cap_helpers():
    RL = _load("frozenpool_lib")
    SP = _load("spec_fp")
    assert RL.band_of(1) == "1-6" and RL.band_of(6) == "1-6"
    assert RL.band_of(7) == "7-24" and RL.band_of(24) == "7-24"
    assert RL.band_of(25) == "25-72" and RL.band_of(72) == "25-72"
    raw = np.array([-100.0, -50.0, 0.0, 30.0, 80.0])
    assert np.allclose(RL.clip_correction(raw, SP.CAP), [-50.0, -50.0, 0.0, 30.0, 50.0])
    base = np.full(72 * 2, 400.0)
    dv = np.full(72 * 2, 10.0)
    hh = np.tile(np.arange(1, 73), 2)
    fin, _, _, _, _ = RL.apply_capped_correction(base, dv, hh, 0.5, SP.CAP,
                                                 {"1-6": True, "7-24": True, "25-72": True})
    assert np.allclose(fin, 405.0)
    fin2, _, _, _, _ = RL.apply_capped_correction(base, dv, hh, 0.5, SP.CAP,
                                                  {"1-6": False, "7-24": False, "25-72": False})
    assert np.allclose(fin2, 400.0)
    RL.check_bands(SP.BANDS)
    RL.check_band_names(SP.BAND_NAMES)
    for bad in ((((1, 5), (7, 24), (25, 72)),), (("a", "b", "c"),)):
        try:
            RL.check_bands(bad[0] if isinstance(bad[0][0], tuple) else bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"check_bands must reject {bad}")


def test_pool_convex_shrink_and_oof_only():
    RL = _load("frozenpool_lib")
    SP = _load("spec_fp")
    RL.check_pool_grid(tuple(round(i * 0.05, 2) for i in range(21)))
    for bad in ((0.0, 0.5, 1.0), tuple(round(i * 0.1, 2) for i in range(11))):
        try:
            RL.check_pool_grid(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"check_pool_grid must reject {bad}")
    for bad in (-0.1, 1.5, float("nan")):
        try:
            RL.check_shrink(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"check_shrink must reject {bad}")
    try:
        RL.check_experts(("E1-unbounded", "E9-other"))
    except ValueError:
        pass
    else:
        raise AssertionError("check_experts must reject unknown experts")
    rng = np.random.default_rng(0)
    n = 720
    e1 = rng.normal(400, 30, n)
    e2 = e1 + rng.normal(0, 5, n)
    y = rng.normal(402, 30, n)
    hh = np.tile(np.arange(1, 73), n // 72 + 1)[:n]
    fit = RL.fit_pool_weights(e1, e2, y, hh, SP.POOL_GRID)
    assert set(fit) == {"1-6", "7-24", "25-72"}
    for b, r in fit.items():
        assert 0.0 <= r["a_best"] <= 1.0 and r["n"] >= 40
        assert r["mse_best"] <= min(r["e1_mse"], r["e2_mse"]) + 1e-9
        assert abs(r["mse_best"] - float(r["mses"][str(float(r["a_best"]))])) < 1e-9
    # Directional: a perfect E1 must win the grid, a perfect E2 must lose it.
    y2 = rng.normal(400, 10, 720)
    hh2 = np.tile(np.arange(1, 73), 11)[:720]
    e1p = y2.copy()
    e2n = y2 + rng.normal(0, 20, 720)
    fit_p = RL.fit_pool_weights(e1p, e2n, y2, hh2, SP.POOL_GRID)
    for b, r in fit_p.items():
        assert r["a_best"] == 1.0 and abs(r["mse_best"]) < 1e-9
    fit_q = RL.fit_pool_weights(e2n, e1p, y2, hh2, SP.POOL_GRID)
    for b, r in fit_q.items():
        assert r["a_best"] == 0.0 and abs(r["mse_best"]) < 1e-9
    fr = RL.shrink_pool_weights(fit, SP.SHRINK_TO_E1)
    for b in RL.BAND_NAMES:
        expect = 0.25 * 1.0 + 0.75 * fit[b]["a_best"]
        assert abs(fr[b] - expect) < 1e-12
        assert 0.0 <= fr[b] <= 1.0
    out = RL.apply_pool(e1, e2, hh, fr)
    bands = RL.band_mask_for_h(hh)
    for i in range(n):
        w = fr[str(bands[i])]
        assert abs(out[i] - (w * e1[i] + (1 - w) * e2[i])) < 1e-9
    # Convexity: frozen weights stay non-negative and sum to one per band.
    for b in RL.BAND_NAMES:
        assert 0.0 <= fr[b] <= 1.0 and 0.0 <= 1.0 - fr[b] <= 1.0


def test_blend_dv_convex_and_oof_only():
    RL = _load("frozenpool_lib")
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
    RL = _load("frozenpool_lib")
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = RL.frame_predictions(te, np.full(72, 400.0))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert (fr["horizon_hours"].to_numpy() == np.arange(1, 73)).all()
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_poison_keys_and_invariance():
    RL = _load("frozenpool_lib")
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
