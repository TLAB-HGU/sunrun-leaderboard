"""D150 shift-scale diagnostic contract: static + synthetic checks (fast, no fits)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d150_oofset/diag_shift"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "lean.py", "D147", "d147",
                 "d148_enlil", "nn-", "sun-nn", "E0472", "E0474", "E0471",
                 "worker-t1a", "worker-t1b", "worker-t1c", "worker-t1d",
                 "worker-t1e", "worker-t1f", "worker-t1g0", "worker-t1g6",
                 "worker-t2no"]
BANNED_SRC = ["GONG", "CDAW"]
CELL_LABELS = ["low_low_high", "low_high_high", "high_low_high", "high_high_high", "hlh"]
PINNED_W = {"1-6": {"e0003_line": 0.21536233464375223, "d33_hsplit": 0.3899845818930272,
                    "d45_lvlmean": 0.3946530834632206},
            "7-24": {"e0003_line": 0.3152013131693896, "d33_hsplit": 0.3436571856823209,
                     "d45_lvlmean": 0.3411415011482895},
            "25-48": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
                      "d45_lvlmean": 0.3333333333333333},
            "49-72": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
                      "d45_lvlmean": 0.3333333333333333}}

sys.path.insert(0, str(EXP))


def _load(mod_name, alias):
    loc = importlib.util.spec_from_file_location(alias, EXP / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code(names=("spec_shift.py", "shift_lib.py", "run_diag_shift.py")):
    return "\n".join((EXP / n).read_text() for n in names)


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 4, files
    assert {"spec_shift.py", "shift_lib.py", "run_diag_shift.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_shift", "d150shift_spec")
    assert SP.DIRECTION == "D150" and SP.OWNER == "worker-t1g7"
    assert SP.KIND == "diag" and SP.REFERENCE == "E0421"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.SCRATCH == "store/scratch/worker-t1g7"
    assert SP.FIT_NJOBS <= 6
    assert SP.BASE_PARAMS["n_estimators"] == 900
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - 0.03) < 1e-12
    assert SP.BASE_PARAMS.get("device") == "cpu" and int(SP.BASE_PARAMS.get("n_jobs")) <= 6
    assert SP.BASE_DEPTH == 3 and SP.WEIGHTS == PINNED_W
    assert SP.BASE_COLUMNS == "existing_all+strips17"
    assert tuple(SP.LEADS) == ("1-6", "7-24", "25-48", "49-72")
    assert SP.PATH_SOURCE == "per-window-oof-fit" and SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert abs(float(SP.EXTREME_Q) - 0.95) < 1e-12
    assert abs(float(SP.EXTREME_FRAC) - 0.05) < 1e-12
    assert int(SP.EXTREME_MIN_FINITE) == 36 and int(SP.EPISODE_GAP_H) == 72
    assert tuple(int(x) for x in SP.SHIFT_KS) == (6, 12, 24) and int(SP.SHIFT_MAX) == 24
    assert tuple(float(x) for x in SP.SCALE_BOUNDS) == (0.5, 2.0)
    assert abs(float(SP.NOISE_MSE) - 90.0) < 1e-12
    assert SP.ORIGIN_CEILING_ISO == "2026-03-01T00:00:00Z"
    assert int(SP.SMOKE_WINDOWS) == 1 and int(SP.SMOKE_ORIGINS) == 8
    assert SP.NO_OFFICIAL and SP.NO_BLEND and SP.NO_SELECTION
    assert "[axes:" in SP.AXES_TAG and "diag=shift-scale-oracle" in SP.AXES_TAG
    assert "close" in SP.CLOSE_RULE.lower()


def test_windows_cover_2023_06_to_2026_02():
    SP = _load("spec_shift", "d150shift_spec_c")
    SL = _load("shift_lib", "d150shift_lib_folds")
    assert SL.check_windows(SP.WINDOWS, SP.ORIGIN_CEILING_ISO) is True
    assert [w["name"] for w in SP.WINDOWS] == ["W0", "W1", "W2"]
    first = pd.Timestamp(SP.WINDOWS[0]["start"], tz="UTC")
    last = pd.Timestamp(SP.WINDOWS[-1]["end"], tz="UTC")
    assert first == pd.Timestamp("2023-06-01T00:00:00Z", tz="UTC")
    assert last >= pd.Timestamp("2026-02-26T00:00:00Z", tz="UTC")
    for w in SP.WINDOWS:
        te = pd.Timestamp(w["train_end"], tz="UTC")
        st = pd.Timestamp(w["start"], tz="UTC")
        en = pd.Timestamp(w["end"], tz="UTC")
        assert (st - te) == pd.Timedelta("72h")
        assert en < pd.Timestamp(SP.ORIGIN_CEILING_ISO, tz="UTC")
    for a, b in zip(SP.WINDOWS, SP.WINDOWS[1:]):
        assert pd.Timestamp(b["start"], tz="UTC") > pd.Timestamp(a["end"], tz="UTC")
    assert SP.OOF_FREQ == "3h" and SP.TRAIN_FREQ == "3h" and SP.PURGE_H == 72
    SL.check_shift_ks(SP.SHIFT_KS)
    SL.check_scale_bounds(SP.SCALE_BOUNDS)
    SL.check_extreme(SP.EXTREME_Q, SP.EXTREME_FRAC, SP.EXTREME_MIN_FINITE, SP.EPISODE_GAP_H)


def test_single_axis_recipe():
    run_src = (EXP / "run_diag_shift.py").read_text()
    for mod in ("info_ablation", "strips", "spec_shift", "shift_lib", "existing_all"):
        assert mod in run_src, mod
    assert "fit_blend_weight" not in run_src and "blend_dv" not in run_src


def test_mixture_helpers():
    SP = _load("spec_shift", "d150shift_spec_d")
    SL = _load("shift_lib", "d150shift_lib_b")
    blocks = [set(SL.level_seeds(s)) for s in (0, 1, 2)]
    assert len(set().union(*blocks)) == 9
    for s in (0, 1, 2):
        assert SL.event_seed(s) in SL.level_seeds(s)
    p = SL.fit_params(dict(SP.BASE_PARAMS), 3, 4)
    assert p["max_depth"] == 3 and p["random_state"] == 4 and p["n_estimators"] == 900
    SL.check_weights(SP.WEIGHTS)
    rng = np.random.default_rng(0)
    h = np.tile(np.arange(1, 73), 4)
    mem = {m: rng.normal(400, 20, len(h)) for m in ("e0003_line", "d33_hsplit", "d45_lvlmean")}
    out = SL.blend_per_lead(mem, SP.WEIGHTS, h)
    assert np.isfinite(out).all() and len(out) == len(h)
    m1 = h <= 6
    m2 = (h >= 7) & (h <= 24)
    m3 = h > 24
    st = SL.stitch_members({s: mem["e0003_line"][m1] for s in (0, 1, 2)},
                           {s: mem["e0003_line"][m2] for s in (0, 1, 2)},
                           mem["e0003_line"], mem["e0003_line"][m3], m1, m2, m3, (0, 1, 2), 1)
    assert set(st) == {"e0003_line", "d33_hsplit", "d45_lvlmean"}
    assert SL.as_path_matrix(np.arange(72 * 3, dtype=float)).shape == (3, 72)
    assert SL.lead_of(1) == "1-6" and SL.lead_of(72) == "49-72"
    try:
        SL.check_weights({"1-6": dict(SP.WEIGHTS["1-6"])})
    except ValueError:
        pass
    else:
        raise AssertionError("partial weights must be rejected")
    try:
        SL.check_shift_ks((6, 12))
    except ValueError:
        pass
    else:
        raise AssertionError("short shift grid must be rejected")
    try:
        SL.check_scale_bounds((0.5, 1.5))
    except ValueError:
        pass
    else:
        raise AssertionError("narrow scale bounds must be rejected")


def test_oracle_synthetic():
    SL = _load("shift_lib", "d150shift_lib_oracle")
    rng = np.random.default_rng(2)
    # synthetic base shifted by +5h plus small noise; oracle should recover it
    n = 6
    Y = rng.normal(400, 30, (n, 72))
    s_true = 5
    idx = np.clip(np.arange(72) - s_true, 0, 71)
    B = Y[:, idx] + rng.normal(0, 2, (n, 72))
    rec = SL.oracle_episode(Y, B, ks=(6, 12, 24), scale_bounds=(0.5, 2.0))
    assert rec["base_mse"] > 0 and np.isfinite(rec["base_mse"])
    # shift oracle must improve over base and pick a shift near -5 (inverse)
    assert rec["shift"][6]["mse"] <= rec["base_mse"] + 1e-9
    assert rec["shift"][24]["mse"] <= rec["shift"][6]["mse"] + 1e-9
    assert abs(rec["s_star_24"] + s_true) <= 2
    assert rec["scale_mse"] <= rec["base_mse"] + 1e-9
    assert rec["both_mse"] <= min(rec["shift"][24]["mse"], rec["scale_mse"]) + 1e-9
    assert 0.5 <= float(rec["scale_a"]) <= 2.0
    pooled = SL.pooled_oracle([(rec, 1), (rec, 1)], ks=(6, 12, 24))
    assert pooled["shift_pm24h_mse"] <= pooled["base_mse"] + 1e-9
    assert pooled["scale_gain"] >= -1e-9 and pooled["both_gain"] >= -1e-9
    # extreme + episodes on synthetic paths
    p95 = float(np.nanquantile(Y.ravel(), 0.95))
    mask = SL.extreme_origin_mask(Y, p95)
    assert mask.dtype == bool and len(mask) == n
    origins = pd.DatetimeIndex(["2024-02-01T00:00:00Z", "2024-02-01T03:00:00Z",
                                "2024-02-10T00:00:00Z", "2024-02-10T03:00:00Z",
                                "2024-02-10T06:00:00Z", "2024-02-10T09:00:00Z"], tz="UTC")
    ids, info = SL.build_episodes(origins, np.array([True, True, False, True, True, True]))
    assert info["n_episodes"] == 2 and info["n_extreme_origins"] == 5
    c = SL.pearson_corr(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert c["n"] == 3 and abs(float(c["r"]) - 1.0) < 1e-12


def test_arrival_context_synthetic():
    SL = _load("shift_lib", "d150shift_lib_arr")
    runs = pd.DataFrame({
        "done": pd.to_datetime(["2024-02-01T00:00:00Z", "2024-02-02T00:00:00Z"], utc=True),
        "arrival": pd.to_datetime(["2024-02-03T00:00:00Z", "2024-02-03T06:00:00Z"], utc=True),
        "kp": [4.0, 5.0],
    })
    shocks = pd.DatetimeIndex(["2024-02-03T02:00:00Z"], tz="UTC")
    origins = pd.DatetimeIndex(["2024-02-02T12:00:00Z", "2024-01-01T00:00:00Z"], tz="UTC")
    ctx = SL.arrival_context(origins, runs, shocks, lookback_d=7, tol_past_h=6, past_window_h=96)
    assert list(ctx.columns) == ["origin", "next_arr_h", "lead_h", "n_pending",
                                 "revision_h", "obs_arr_h", "arr_err_h"]
    assert np.isfinite(ctx["next_arr_h"].iloc[0]) and ctx["n_pending"].iloc[0] >= 1
    assert not np.isfinite(ctx["next_arr_h"].iloc[1])


def test_poison_synthetic():
    SL = _load("shift_lib", "d150shift_lib_e")
    idx = pd.date_range("2023-05-01T00:00:00Z", "2023-07-01T00:00:00Z", freq="h", tz="UTC")
    filled = pd.Series(400.0 + np.sin(np.arange(len(idx)) / 24.0), index=idx)
    cut = "2023-05-29T00:00:00Z"
    origins = pd.DatetimeIndex(["2023-05-10T00:00:00Z", "2023-05-20T00:00:00Z",
                                "2023-06-05T00:00:00Z", "2023-06-10T00:00:00Z"], tz="UTC")
    sc = SL.scenario_poison(origins, filled, cut)
    assert sc["passed"] is True and sc["pre_invariant"] is True and sc["moved"] is True
    assert sc["n_pre_rows"] == 2 and sc["n_post_rows"] == 2

    def _frame_for(series):
        o = pd.DatetimeIndex(origins, tz="UTC")
        return np.tile(series.reindex(o).to_numpy(dtype=float)[:, None], (1, 72))

    fp = SL.frame_poison(origins, filled, cut, _frame_for)
    assert fp["passed"] is True and fp["pre_invariant"] is True


def test_seed_actually_propagates():
    run_src = (EXP / "run_diag_shift.py").read_text()
    assert "fit_seed_models" in run_src and "event_seed" in run_src
    assert "level_seeds" in run_src and "fit_params" in run_src


def test_diag_only_outputs():
    run_src = (EXP / "run_diag_shift.py").read_text()
    for w in ("diag_shift.json", "causality.json", "meta.json"):
        assert w in run_src, w
    for w in ("predictions_official", "origin_last_input_utc", "pred_kms",
              "SELECT_RULE", "official-score", "official_score", "freeze",
              "submit", "upload", "push"):
        assert w not in run_src, w
    assert "NO_SELECTION" in run_src and "NO_BLEND" in run_src and "NO_OFFICIAL" in run_src


def test_scratch_isolation_and_cpu():
    import re
    SP = _load("spec_shift", "d150shift_spec_f")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1g7"
    assert len(set(re.findall(r"worker-[a-z0-9]+", code))) == 1
    assert "worker-t1g7" in code
    run_src = (EXP / "run_diag_shift.py").read_text()
    assert 'device") == "cpu"' in run_src and "n_jobs" in run_src
    assert "--gpu" not in code


def test_poison_covers_window_inputs():
    run_src = (EXP / "run_diag_shift.py").read_text()
    assert "poison_test" in run_src and "scenario_poison" in run_src and "frame_poison" in run_src
    assert "causality.json" in run_src and '"passed"' in run_src
    assert "train_end" in run_src


def test_no_forbidden_leak_or_banned_source():
    code = _code()
    for w in FORBIDDEN:
        assert w not in code, w
    for w in OTHER_CONTACT:
        assert w not in code, w
    low = code.lower()
    for w in BANNED_SRC:
        assert w.lower() not in low, w
    for w in CELL_LABELS:
        assert w not in low, w
    assert "cuda" not in low and "--gpu" not in code
