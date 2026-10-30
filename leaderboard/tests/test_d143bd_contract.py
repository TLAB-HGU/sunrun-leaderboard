"""D143 BOUNDED contract: static + synthetic checks (fast, no fits, no data)."""
import importlib.util
import sys

import numpy as np
import pandas as pd
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d143_axisA_analog" / "bounded"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_IDS = ["e0147-abl", "d141_e0147", "ablate", "D141", "E0443", "E0446", "E0414", "E0438",
             "worker-d76", "worker-t1d7",
             "worker-t1a1", "worker-t1a2", "worker-t1a5", "worker-t1a6", "worker-t1a7",
             "worker-t1a8", "worker-t1b0", "worker-t1b1", "worker-t1b2", "worker-t1b3",
             "worker-t1b4", "worker-t1c0", "worker-t1c1", "worker-t1c2", "worker-t1c3",
             "worker-d135", "worker-d133",
             "worker-t2no", "nn-", "sun-nn"]
BANNED_SRC = ["GONG", "WSA", "ADAPT", "CME", "DONKI"]
CELL_LABELS = ["low_low_high", "low_high_high", "high_low_high", "high_high_high"]


def _load(name):
    loc = importlib.util.spec_from_file_location(f"d143bd_{name}", EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(EXP.glob("*.py")))


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert 4 <= len(pys) <= 6, files
    assert {"spec_cap25.py", "spec_cap50.py", "spec_cap75.py",
            "bounded_lib.py", "run_official_bounded.py"} <= set(pys), files


def test_spec_identity_and_scope():
    for name, cap, arm in (("spec_cap25", 25.0, "cap25"),
                           ("spec_cap50", 50.0, "cap50"),
                           ("spec_cap75", 75.0, "cap75")):
        SP = _load(name)
        assert SP.DIRECTION == "D143" and SP.OWNER == "worker-t1b5", name
        assert SP.KIND == "full" and SP.REFERENCE == "E0421", name
        assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0, name
        assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z", name
        assert SP.N_OFFICIAL_ORIGINS == 2833, name
        assert SP.TRAIN_FREQ == "3h", name
        assert SP.SCRATCH == "store/scratch/worker-t1b5", name
        assert SP.FIT_NJOBS <= 6, name
        assert SP.K == 15 and SP.WEIGHTING == "uniform", name
        assert SP.SCENARIO_H == 12 and SP.ERR_H == 12 and SP.PATH_H == 72, name
        assert SP.SCENARIO_DIM == 96, name
        assert tuple(tuple(b) for b in SP.BANDS) == ((1, 6), (7, 24), (25, 72)), name
        assert tuple(SP.BAND_NAMES) == ("1-6", "7-24", "25-72"), name
        assert abs(float(SP.CAP) - cap) < 1e-12 and SP.ARM == arm, name
        assert int(SP.MASK_MIN_WINS) == 2, name
        assert abs(SP.CARRINGTON_H - 654.48) < 1e-9, name
        assert len(SP.BLEND_GRID) == 21 and SP.BLEND_GRID[0] == 0.0 and SP.BLEND_GRID[-1] == 1.0, name
        assert SP.OOF_TRAIN_END_ISO < SP.OOF_START_ISO <= SP.OOF_END_ISO < SP.CUTOFF_ISO, name
        assert SP.BASE_DEPTH == 3 and SP.BASE_COLUMNS == "existing_all+strips17", name
        assert SP.BASE_PARAMS.get("n_estimators") == 900, name
        assert abs(float(SP.BASE_PARAMS.get("learning_rate")) - 0.03) < 1e-12, name
        assert SP.PATH_SOURCE == "oof-fit", name
        assert SP.RESIDUAL_DEF == "dv = y - yhat_base", name
        assert "[axes:" in SP.AXES_TAG and "residual" in SP.AXES_TAG and "bounded" in SP.AXES_TAG, name
        assert str(int(cap)) in SP.CAP_DEF, name


def test_three_caps_prespecified_and_separated():
    s25, s50, s75 = _load("spec_cap25"), _load("spec_cap50"), _load("spec_cap75")
    assert {float(s25.CAP), float(s50.CAP), float(s75.CAP)} == {25.0, 50.0, 75.0}
    assert len({s25.ARM, s50.ARM, s75.ARM}) == 3
    # Only the cap differs between the three arms.
    for SP in (s25, s50, s75):
        assert tuple(tuple(b) for b in SP.BANDS) == ((1, 6), (7, 24), (25, 72))
        assert int(SP.MASK_MIN_WINS) == 2
    assert s25.MASK_RULE == s50.MASK_RULE == s75.MASK_RULE
    assert s25.XOOF_WINDOWS == s50.XOOF_WINDOWS == s75.XOOF_WINDOWS
    assert s25.BASE_PARAMS == s50.BASE_PARAMS == s75.BASE_PARAMS
    assert s25.WEIGHTS == s50.WEIGHTS == s75.WEIGHTS
    assert s25.BLEND_GRID == s50.BLEND_GRID == s75.BLEND_GRID
    assert s25.AXES_TAG == s50.AXES_TAG == s75.AXES_TAG
    assert s25.RESIDUAL_DEF == s50.RESIDUAL_DEF == s75.RESIDUAL_DEF
    text25 = (EXP / "run_official_bounded.py").read_text()
    assert "BOUNDED_SPEC" in text25, "shared runner must select the pre-registered spec"


def test_windows_prespecified_and_causal():
    SP = _load("spec_cap50")
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


def test_bounded_single_axis_reuse():
    text = (EXP / "run_official_bounded.py").read_text()
    assert "import bounded_lib as RL" in text
    assert "BOUNDED_SPEC" in text
    assert "worker-t1b5" in text
    assert "predictions_official_s{s}.parquet" in text
    assert "resid_summary_s{s}.parquet" in text
    assert "residual_for_origins" in text
    assert "fit_blend_weight_dv" in text
    assert "oof_band_report" in text and "vote_band_masks" in text
    assert "apply_capped_correction" in text
    assert "XOOF_WINDOWS" in text and "MAIN_WINDOW" in text
    for dup in ("def residual_for_origins", "def fit_blend_weight_dv",
                "def scenario_distance3", "def blend_per_lead"):
        assert dup not in text, f"runner must reuse bounded_lib, not duplicate {dup}"
    assert "xgb.XGBRegressor" in text
    # No head, no gate on regime labels: the only per-horizon structure is
    # the pre-registered band mask voted from OOF gains.
    for lab in CELL_LABELS:
        assert lab not in text, f"runner must never use the cell label {lab}"
    libtext = (EXP / "bounded_lib.py").read_text()
    for lab in CELL_LABELS:
        assert lab not in libtext, f"lib must never use the cell label {lab}"
    assert "uniform" in libtext
    sys.path.insert(0, str(LB / "experiments/headroom_audit"))
    import info_ablation as ia
    SP = _load("spec_cap50")
    assert "max_depth" not in SP.BASE_PARAMS, "depth comes only from BASE_DEPTH"
    for k, v in ia.PARAMS.items():
        if k in ("max_depth", "n_jobs", "random_state", "n_estimators", "learning_rate"):
            continue
        assert SP.BASE_PARAMS.get(k) == v, k
    assert SP.BASE_PARAMS.get("tree_method") == "hist"
    assert SP.BASE_PARAMS.get("device") == "cpu"
    sys.path.insert(0, str(EXP))
    import bounded_lib as RL
    RL.check_weights(SP.WEIGHTS)
    RL.check_bands(SP.BANDS)
    RL.check_band_names(SP.BAND_NAMES)
    RL.check_cap(SP.CAP)


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
    import bounded_lib as RL
    SP = _load("spec_cap50")
    params = [RL.fit_params(dict(SP.BASE_PARAMS), SP.BASE_DEPTH, s) for s in SP.SEEDS]
    assert len({p["random_state"] for p in params}) == 3
    assert all(p["max_depth"] == 3 for p in params)
    assert RL.level_seeds(1) == (3, 4, 5) and RL.event_seed(1) == 4


def test_cap_and_mask_helpers():
    sys.path.insert(0, str(EXP))
    import bounded_lib as RL
    assert RL.band_of(1) == "1-6" and RL.band_of(6) == "1-6"
    assert RL.band_of(7) == "7-24" and RL.band_of(24) == "7-24"
    assert RL.band_of(25) == "25-72" and RL.band_of(72) == "25-72"
    try:
        RL.band_of(0)
    except ValueError:
        pass
    else:
        raise AssertionError("band_of must reject horizon 0")
    for cap in (25.0, 50.0, 75.0):
        RL.check_cap(cap)
    for bad in (0.0, -5.0, float("inf"), float("nan")):
        try:
            RL.check_cap(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"check_cap must reject {bad}")
    raw = np.array([-100.0, -25.0, 10.0, 60.0])
    assert np.allclose(RL.clip_correction(raw, 50.0), [-50.0, -25.0, 10.0, 50.0])
    # Masked application: OFF bands and NaN dv keep the base.
    base = np.full(6, 400.0)
    dv = np.array([100.0, 100.0, np.nan, -200.0, 10.0, 10.0])
    h = np.array([1, 2, 3, 8, 30, 40])
    final, r, c, applied, on = RL.apply_capped_correction(
        base, dv, h, 0.5, 25.0, {"1-6": True, "7-24": False, "25-72": True})
    assert np.allclose(r[:2], [50.0, 50.0])
    assert np.allclose(c[:2], [25.0, 25.0])
    assert final[0] == 425.0 and final[1] == 425.0
    assert final[2] == 400.0, "NaN dv falls back to base"
    assert final[3] == 400.0, "OFF band keeps base"
    assert final[4] == 405.0 and final[5] == 405.0
    # Mask vote: ON iff gain in at least 2 of 3 windows.
    def _rep(gains):
        return {b: {"gain": g, "base_mse": 2.0, "capped_mse": 1.0 if g else 3.0,
                    "n": 500, "n_finite_dv": 500, "abs_p50": 5.0, "abs_p90": 30.0,
                    "abs_p99": 60.0, "abs_max": 120.0,
                    "share_gt25": 0.2, "share_gt50": 0.05, "share_gt75": 0.01}
                for b, g in zip(("1-6", "7-24", "25-72"), gains)}
    votes = {"W0-early": _rep([True, False, True]),
             "W1-mid": _rep([True, False, False]),
             "W2-main": _rep([False, False, True])}
    assert RL.vote_band_masks(votes, 2) == {"1-6": True, "7-24": False, "25-72": True}
    try:
        RL.vote_band_masks({"W0-early": _rep([True] * 3)}, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("vote needs exactly 3 windows")


def test_oof_band_report_bites():
    sys.path.insert(0, str(EXP))
    import bounded_lib as RL
    rng = np.random.default_rng(7)
    n = 40
    h = np.tile(np.arange(1, 73), n)
    base = rng.normal(400, 30, len(h))
    dv = rng.normal(0, 120, len(h))  # wide enough that caps bite
    y = base + 0.3 * dv + rng.normal(0, 5, len(h))
    rep = RL.oof_band_report(base, dv, y, h, 0.3, 50.0)
    assert set(rep) == {"1-6", "7-24", "25-72"}
    for b, r in rep.items():
        assert r["n"] > 100 and r["n_finite_dv"] == r["n"]
        assert r["abs_p50"] < r["abs_p90"] < r["abs_max"]
        assert 0.0 <= r["share_gt25"] <= 1.0
        assert r["share_gt25"] >= r["share_gt50"] >= r["share_gt75"]
        assert r["share_gt25"] > 0.0, "test distribution must let the caps bite"
        assert isinstance(r["gain"], bool)


def test_output_frame_schema():
    sys.path.insert(0, str(EXP))
    import bounded_lib as RL
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = RL.frame_predictions(te, np.full(72, 400.0))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert (fr["horizon_hours"].to_numpy() == np.arange(1, 73)).all()
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_poison_keys_and_invariance():
    sys.path.insert(0, str(EXP))
    import bounded_lib as RL
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
