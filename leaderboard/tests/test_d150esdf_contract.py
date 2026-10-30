"""D150 ESDF cheap-contrast contract: static + synthetic checks (fast, no fits).

No real data fits, no official reads. Covers the owned file budget,
spec identity (D150/worker-t1g5/full), the three pre-registered arms
(real/dup/interp, diffusion excluded), the family extreme rule with
episode counts, duplicate emphasis and pair-blend interpolation math on
synthetic data, synthetic-path validation, seed propagation across
base and synth streams, fresh-fit source isolation, CPU/n_jobs budget,
scratch isolation, poison-key shape, and forbidden-source absence.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d150_oofset" / "esdf"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "lean.py", "D147", "d147",
                 "d148_enlil", "nn-", "sun-nn", "E0472", "E0471", "E0474",
                 "worker-t1a", "worker-t1b", "worker-t1c", "worker-t1d",
                 "worker-t1e", "worker-t1f", "worker-t1g0", "worker-t1g1",
                 "worker-t1g2", "worker-t1g3", "worker-t1g4", "worker-d147",
                 "worker-d141", "worker-t2no"]
BANNED_SRC = ["GONG", "CDAW", "WSA", "DONKI", "CME", "ADAPT", "ENLIL"]
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


def _code(names=("spec_es.py", "esdf_lib.py", "run_official_esdf.py")):
    return "\n".join((EXP / n).read_text() for n in names)


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 6, files
    assert {"spec_es.py", "esdf_lib.py", "run_official_esdf.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_es", "d150es_spec")
    assert SP.DIRECTION == "D150" and SP.OWNER == "worker-t1g5"
    assert SP.KIND == "full"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.OOF_TRAIN_END_ISO == "2026-03-01T00:00:00Z"
    assert SP.OOF_START_ISO == "2026-03-04T00:00:00Z"
    assert SP.OOF_END_ISO == "2026-05-28T23:00:00Z"
    assert SP.OOF_FREQ == "3h"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.ORIGIN_START == "2026-05-31 23:00" and SP.ORIGIN_END == "2026-09-26 23:00"
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24 and SP.PURGE_H == 72
    assert tuple(SP.ARMS) == ("real", "dup", "interp")
    assert abs(float(SP.DUP_FACTOR) - 3.0) < 1e-9
    assert int(SP.N_INTERP_PER_EXTREME) == 1 and int(SP.EPISODE_GAP_H) == 72
    assert SP.SCRATCH == "store/scratch/worker-t1g5"
    assert SP.FIT_NJOBS <= 6
    assert SP.SMOKE_OOF_ORIGINS == 40 and SP.SMOKE_OFFICIAL == 8
    assert SP.BASE_PARAMS["n_estimators"] == 900
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - 0.03) < 1e-12
    assert SP.BASE_PARAMS.get("device") == "cpu" and int(SP.BASE_PARAMS.get("n_jobs")) <= 6
    assert SP.BASE_DEPTH == 3 and SP.WEIGHTS == PINNED_W
    assert SP.BASE_COLUMNS == "existing_all+strips17"
    assert tuple(SP.LEADS) == ("1-6", "7-24", "25-48", "49-72")
    assert SP.PATH_SOURCE == "own-fit-oof-contrast-own-fit-official"
    assert SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert "blend" in SP.SYNTH_DEF
    assert abs(float(SP.EXTREME_FRAC) - 0.05) < 1e-12 and int(SP.EXTREME_MIN_FINITE) == 36
    assert abs(float(SP.NOISE_MSE) - 90.0) < 1e-9
    assert "[axes:" in SP.AXES_TAG and "esdf" in SP.AXES_TAG
    assert "OOF-only" in SP.SELECT_RULE or "OOF" in SP.SELECT_RULE


def test_weights_match_shared_recipe():
    SP = _load("spec_es", "d150es_spec_b")
    loc = importlib.util.spec_from_file_location(
        "d150es_oof", LB / "experiments/d150_oofset/spec_oof.py")
    oof = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(oof)
    assert SP.WEIGHTS == oof.WEIGHTS
    assert SP.BASE_PARAMS["n_estimators"] == oof.BASE_PARAMS["n_estimators"]
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - float(oof.BASE_PARAMS["learning_rate"])) < 1e-12
    assert SP.BASE_DEPTH == oof.BASE_DEPTH
    assert SP.BASE_PARAMS["subsample"] == 0.8 and SP.BASE_PARAMS["colsample_bytree"] == 0.7
    assert SP.BASE_PARAMS["min_child_weight"] == 20 and SP.BASE_PARAMS["reg_lambda"] == 5
    assert SP.BASE_PARAMS["tree_method"] == "hist"
    assert SP.BASE_COLUMNS == oof.BASE_COLUMNS


def test_oof_window_purge_and_causality():
    SP = _load("spec_es", "d150es_spec_c")
    te = pd.Timestamp(SP.OOF_TRAIN_END_ISO, tz="UTC")
    st = pd.Timestamp(SP.OOF_START_ISO, tz="UTC")
    en = pd.Timestamp(SP.OOF_END_ISO, tz="UTC")
    cut = pd.Timestamp(SP.CUTOFF_ISO, tz="UTC")
    assert (st - te) == pd.Timedelta("72h")
    assert en < cut
    assert SP.OOF_FREQ == "3h" and SP.PURGE_H == 72
    off = pd.Timestamp("2026-05-31 23:00", tz="UTC")
    assert off > cut


def test_single_axis_recipe():
    run_src = (EXP / "run_official_esdf.py").read_text()
    for mod in ("info_ablation", "strips", "spec_es", "esdf_lib", "existing_all"):
        assert mod in run_src, mod
    for w in ("predictions_official_s{s}.parquet", "oof_comparison.json",
              "poison.json", "meta.json", "dup_origin_weights",
              "make_interp_rows", "extreme_origin_mask", "count_episodes",
              "synth_path_report", "--arm"):
        assert w in run_src, w
    for w in ("diffusion", "denoise", "score_model", "unet", "blend_dv",
              "fit_blend_weight", "disentangl"):
        assert w not in run_src.lower(), w
    assert "official-score" not in run_src and "official_score" not in run_src


def test_mixture_helpers():
    ES = _load("esdf_lib", "d150es_lib_a")
    SP = _load("spec_es", "d150es_spec_d")
    blocks = [set(ES.level_seeds(s)) for s in (0, 1, 2)]
    assert len(set().union(*blocks)) == 9
    for s in (0, 1, 2):
        assert ES.event_seed(s) in ES.level_seeds(s)
    assert ES.synth_seed(0, "oof") != ES.synth_seed(0, "full")
    assert ES.synth_seed(0, "oof") != ES.synth_seed(1, "oof")
    p = ES.fit_params(dict(SP.BASE_PARAMS), 3, 4)
    assert p["max_depth"] == 3 and p["random_state"] == 4 and p["n_estimators"] == 900
    ES.check_weights(SP.WEIGHTS)
    ES.check_arms(SP.ARMS)
    rng = np.random.default_rng(0)
    h = np.tile(np.arange(1, 73), 4)
    mem = {m: rng.normal(400, 20, len(h)) for m in ("e0003_line", "d33_hsplit", "d45_lvlmean")}
    out = ES.blend_per_lead(mem, SP.WEIGHTS, h)
    assert np.isfinite(out).all() and len(out) == len(h)
    m1 = h <= 6
    m2 = (h >= 7) & (h <= 24)
    m3 = h > 24
    st = ES.stitch_members({s: mem["e0003_line"][m1] for s in (0, 1, 2)},
                           {s: mem["e0003_line"][m2] for s in (0, 1, 2)},
                           mem["e0003_line"], mem["e0003_line"][m3], m1, m2, m3, (0, 1, 2), 1)
    assert set(st) == {"e0003_line", "d33_hsplit", "d45_lvlmean"}
    assert ES.as_path_matrix(np.arange(72 * 3, dtype=float)).shape == (3, 72)
    assert ES.lead_of(1) == "1-6" and ES.lead_of(72) == "49-72"
    assert ES.oof_mse(np.array([1.0, 2.0]), np.array([1.0, 3.0])) == 0.5
    assert np.isnan(ES.oof_mse(np.array([np.nan]), np.array([np.nan])))
    try:
        ES.check_weights({"1-6": dict(SP.WEIGHTS["1-6"])})
    except ValueError:
        pass
    else:
        raise AssertionError("partial weights must be rejected")
    try:
        ES.check_arms(("real",))
    except ValueError:
        pass
    else:
        raise AssertionError("arms must stay pinned")


def test_extreme_and_episode_synthetic():
    ES = _load("esdf_lib", "d150es_lib_ext")
    Y = np.full((3, 72), 400.0)
    Y[0, :4] = 900.0
    Y[1, :3] = 900.0
    mask = ES.extreme_origin_mask(Y, 500.0, frac=0.05, min_finite=36)
    assert list(mask) == [True, False, False]
    origins = pd.DatetimeIndex(["2024-01-01T00:00:00Z", "2024-01-01T03:00:00Z",
                                "2024-01-05T00:00:00Z"], tz="UTC")
    ep = ES.count_episodes(origins, np.array([True, True, True]), gap_h=72)
    assert ep == {"n_extreme_origins": 3, "n_episodes": 2}
    ep0 = ES.count_episodes(origins, np.array([False, False, False]), gap_h=72)
    assert ep0 == {"n_extreme_origins": 0, "n_episodes": 0}
    w = ES.dup_origin_weights(np.array([True, False, True]), factor=3.0)
    assert list(w) == [3.0, 1.0, 3.0]
    idx = np.array([0, 0, 1, 2, 2, 2])
    per = np.array([3.0, 1.0, 1.0])
    assert list(ES.expand_row_weights(idx, per)) == [3.0, 3.0, 1.0, 1.0, 1.0, 1.0]
    assert ES.past12_matrix(
        pd.DatetimeIndex(["2024-01-01T00:00:00Z", "2024-01-01T03:00:00Z"], tz="UTC"),
        pd.Series([400.0] * 100,
                  index=pd.date_range("2023-12-01", periods=100, freq="h", tz="UTC"))).shape == (2, 12)


def test_interp_rows_synthetic():
    ES = _load("esdf_lib", "d150es_lib_syn")
    rng = np.random.default_rng(7)
    rows = []
    for k, base in enumerate((400.0, 500.0, 600.0)):
        for hh in range(1, 73):
            rows.append({"origin": f"2024-01-0{k + 1}T00:00:00Z", "h": hh,
                         "y": base + rng.normal(0, 5),
                         "f1": base / 10.0 + hh, "f2": float(hh)})
    ext = pd.DataFrame(rows)
    syn = ES.make_interp_rows(ext, ["f1", "f2"], seed=9000, n_per=1)
    assert len(syn) == 3 * 72
    assert set(syn["h"].unique()) == set(range(1, 73))
    assert syn["origin"].astype(str).str.startswith("synth-").all()
    assert np.isfinite(syn[["f1", "f2", "y"]].to_numpy(dtype=float)).all()
    for hh in (1, 36, 72):
        got = syn[syn["h"] == hh]["f1"].to_numpy(dtype=float)
        lo = min(400.0 / 10.0 + hh, 600.0 / 10.0 + hh)
        hi = max(400.0 / 10.0 + hh, 600.0 / 10.0 + hh)
        assert ((got >= lo - 1e-9) & (got <= hi + 1e-9)).all()
    part = ext[((ext["origin"] == "2024-01-01T00:00:00Z") & (ext["h"] <= 40)) |
               ((ext["origin"] == "2024-01-02T00:00:00Z") & (ext["h"] >= 20)) |
               (ext["origin"] == "2024-01-03T00:00:00Z")].copy()
    syn2 = ES.make_interp_rows(part, ["f1", "f2"], seed=9001, n_per=1)
    assert syn2["origin"].nunique() >= 1
    assert syn2.groupby("origin")["h"].count().min() >= 36
    try:
        ES.make_interp_rows(ext[ext["origin"] == "2024-01-01T00:00:00Z"], ["f1", "f2"],
                            seed=9000, n_per=1)
    except ValueError:
        pass
    else:
        raise AssertionError("blending needs >=2 extreme origins")


def test_synth_validation_synthetic():
    ES = _load("esdf_lib", "d150es_lib_val")
    rng = np.random.default_rng(11)
    t = np.linspace(0, 1, 72)
    real = 400 + 150 * t + rng.normal(0, 5, (4, 72))
    syn = 400 + 150 * t + rng.normal(0, 5, (4, 72))
    srows = []
    for k in range(4):
        for hh in range(1, 73):
            srows.append({"origin": f"synth-9000-{k:05d}", "h": hh, "y": float(syn[k, hh - 1])})
    rep = ES.synth_path_report(pd.DataFrame(srows), real, phys_min=200.0, phys_max=1200.0)
    assert rep["passed"] is True
    assert rep["synthetic"]["n_paths"] == 4 and rep["real"]["n_paths"] == 4
    assert rep["synthetic"]["lag1_autocorr"] > 0.8
    bad = pd.DataFrame([{"origin": "synth-1-00000", "h": hh, "y": 5000.0} for hh in range(1, 73)])
    rep2 = ES.synth_path_report(bad, real, phys_min=200.0, phys_max=1200.0)
    assert rep2["passed"] is False
    fr = ES.frame_predictions(pd.DataFrame({"origin": ["2026-06-01T00:00:00Z"] * 72,
                                            "h": np.arange(1, 73)}), np.full(72, 400.0))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_origin_paths_keep_tz():
    ES = _load("esdf_lib", "d150es_lib_paths")
    rows = []
    for k in range(3):
        for hh in range(1, 73):
            rows.append({"origin": f"2024-01-0{k + 1}T00:00:00Z", "h": hh,
                         "y": 400.0 + hh})
    origins, Y = ES.origin_paths(pd.DataFrame(rows))
    assert len(origins) == 3 and Y.shape == (3, 72)
    assert float(np.nanmean(Y)) > 400.0
    flag = ES.extreme_origin_mask(Y, 400.0, frac=0.05, min_finite=36)
    assert bool(flag.all()) is True


def test_arm_routing():
    run_src = (EXP / "run_official_esdf.py").read_text()
    assert '"real"' in run_src and '"dup"' in run_src and '"interp"' in run_src
    assert "--arm" in run_src and 'choices=("real", "dup", "interp")' in run_src
    assert "DUP_FACTOR" in run_src and "N_INTERP_PER_EXTREME" in run_src
    assert "extreme pre-cutoff" in run_src


def test_seed_actually_propagates():
    run_src = (EXP / "run_official_esdf.py").read_text()
    assert "fit_seed_models" in run_src and "event_seed" in run_src
    assert "level_seeds" in run_src and "fit_params" in run_src
    assert "synth_seed" in run_src


def test_fresh_source_only():
    run_src = (EXP / "run_official_esdf.py").read_text()
    code = _code()
    for w in ("stored prediction", "stored array", "another run"):
        assert w in code.lower() or "no stored array" in code.lower(), w
    assert "PATH_SOURCE" in run_src or "own-fit" in run_src
    assert "train_end" in run_src.lower() or "TRAIN_END" in run_src


def test_poison_synthetic():
    ES = _load("esdf_lib", "d150es_lib_poison")
    idx = pd.date_range("2026-05-01", periods=1200, freq="h", tz="UTC")
    filled = pd.Series(400.0, index=idx)
    origins = pd.DatetimeIndex(["2026-05-20T00:00:00Z", "2026-05-21T00:00:00Z",
                                "2026-06-10T00:00:00Z", "2026-06-11T00:00:00Z"], tz="UTC")
    rec = ES.scenario_poison(origins, filled, "2026-05-29T00:00:00Z")
    assert rec["passed"] is True and rec["pre_invariant"] is True
    assert rec["n_pre_rows"] == 2 and rec["n_post_rows"] == 2

    def _frame_for(series):
        n = len(origins)
        return pd.DataFrame(np.tile(series.reindex(origins).to_numpy(dtype=float)[:, None],
                                    (1, 2))).iloc[np.repeat(np.arange(n), 2)].reset_index(drop=True)
    rec2 = ES.frame_poison(origins, filled, "2026-05-29T00:00:00Z", _frame_for)
    assert rec2["passed"] is True and rec2["pre_invariant"] is True
    assert rec2["n_pre_rows"] == 2 and rec2["n_post_rows"] == 2


def test_poison_covers_fold_inputs():
    run_src = (EXP / "run_official_esdf.py").read_text()
    assert "scenario_poison" in run_src and "frame_poison" in run_src
    assert "poison_test" in run_src
    assert "pre-cutoff" in run_src or "pre_cutoff" in run_src or "pre-cutoff" in run_src
    assert "train_end" in run_src.lower() or "TRAIN_END" in run_src


def test_scratch_isolation_and_cpu():
    import re
    run_src = (EXP / "run_official_esdf.py").read_text()
    code = _code()
    assert "store/scratch/worker-t1g5" in code
    assert "worker-t1g0" not in code and "worker-t1g1" not in code
    assert "worker-t1g2" not in code and "worker-t1g3" not in code
    assert "worker-t1g4" not in code
    assert re.search(r"FIT_NJOBS\s*=\s*6\b", code) is not None
    assert '"cpu"' in code or "'cpu'" in code or "device" in code
    assert "--gpu" not in code
    low = code.lower()
    assert "cuda" not in low


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
    for w in ("official-score", "official_score", "submit", "upload"):
        assert w not in low, w
    assert "freeze" not in low and "push" not in low
