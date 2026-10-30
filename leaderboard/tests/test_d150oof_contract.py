"""D150 OOF residual-set contract: static + synthetic checks (fast, no fits).

No real data fits, no arm judgment, no official reads. Covers the owned
file budget, spec identity (D150/worker-t1g0/infra/E0421), the expanding
fold table (2023-06..2026-02, 72h purge, pre-ceiling), the E0408 recipe
base (fresh per-fold fit), residual-only outputs (no official schema, no
blend, no selection), mixture seed propagation, summary/dist/threshold
helpers on synthetic data, per-fold causality coverage, CPU/n_jobs
budget, scratch isolation, and forbidden-source absence.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d150_oofset"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "lean.py", "D147", "d147",
                 "d148_enlil", "nn-", "sun-nn", "E0472", "E0474", "E0471",
                 "worker-t1a", "worker-t1b", "worker-t1c", "worker-t1d",
                 "worker-t1e", "worker-t1f", "worker-d147", "worker-d141",
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


def _code(names=("spec_oof.py", "oofset_lib.py", "run_oofset.py")):
    return "\n".join((EXP / n).read_text() for n in names)


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 6, files
    assert {"spec_oof.py", "oofset_lib.py", "run_oofset.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_oof", "d150_spec")
    assert SP.DIRECTION == "D150" and SP.OWNER == "worker-t1g0"
    assert SP.KIND == "infra" and SP.REFERENCE == "E0421"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.SCRATCH == "store/scratch/worker-t1g0"
    assert SP.FIT_NJOBS <= 6
    assert SP.BASE_PARAMS["n_estimators"] == 900
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - 0.03) < 1e-12
    assert SP.BASE_PARAMS.get("device") == "cpu" and int(SP.BASE_PARAMS.get("n_jobs")) <= 6
    assert SP.BASE_DEPTH == 3 and SP.WEIGHTS == PINNED_W
    assert SP.BASE_COLUMNS == "existing_all+strips17"
    assert tuple(SP.LEADS) == ("1-6", "7-24", "25-48", "49-72")
    assert SP.PATH_SOURCE == "per-fold-oof-fit" and SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert tuple(float(x) for x in SP.CLIP_CANDIDATES) == (25.0, 50.0, 75.0, 150.0)
    assert len(tuple(SP.QUANTILES)) == 9
    assert SP.ORIGIN_CEILING_ISO == "2026-03-01T00:00:00Z"
    assert SP.SMOKE_FOLDS == 2 and SP.SMOKE_ORIGINS == 8
    assert SP.NO_OFFICIAL and SP.NO_BLEND and SP.NO_SELECTION
    assert "[axes:" in SP.AXES_TAG and "infra=oof-residual-set" in SP.AXES_TAG


def test_weights_match_e0421_recipe():
    SP = _load("spec_oof", "d150_spec_b")
    loc = importlib.util.spec_from_file_location(
        "d150_a2b", LB / "experiments/d143_axisA_analog/a2b/spec_a2b.py")
    a2b = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(a2b)
    assert SP.WEIGHTS == a2b.WEIGHTS
    assert SP.BASE_PARAMS["n_estimators"] == a2b.BASE_PARAMS["n_estimators"]
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - float(a2b.BASE_PARAMS["learning_rate"])) < 1e-12
    assert SP.BASE_DEPTH == a2b.BASE_DEPTH
    assert SP.BASE_PARAMS["subsample"] == 0.8 and SP.BASE_PARAMS["colsample_bytree"] == 0.7
    assert SP.BASE_PARAMS["min_child_weight"] == 20 and SP.BASE_PARAMS["reg_lambda"] == 5
    assert SP.BASE_PARAMS["tree_method"] == "hist"


def test_folds_cover_2023_06_to_2026_02():
    SP = _load("spec_oof", "d150_spec_c")
    OL = _load("oofset_lib", "d150_lib_folds")
    assert OL.check_folds(SP.FOLDS, SP.ORIGIN_CEILING_ISO) is True
    assert [f["name"] for f in SP.FOLDS] == [f"F{i}" for i in range(8)]
    first = pd.Timestamp(SP.FOLDS[0]["start"], tz="UTC")
    last = pd.Timestamp(SP.FOLDS[-1]["end"], tz="UTC")
    assert first == pd.Timestamp("2023-06-01T00:00:00Z", tz="UTC")
    assert last >= pd.Timestamp("2026-02-26T00:00:00Z", tz="UTC")
    for f in SP.FOLDS:
        te = pd.Timestamp(f["train_end"], tz="UTC")
        st = pd.Timestamp(f["start"], tz="UTC")
        en = pd.Timestamp(f["end"], tz="UTC")
        assert (st - te) == pd.Timedelta("72h")
        assert en < pd.Timestamp(SP.ORIGIN_CEILING_ISO, tz="UTC")
        assert (en - st) >= pd.Timedelta("119D")
    for a, b in zip(SP.FOLDS, SP.FOLDS[1:]):
        assert pd.Timestamp(b["start"], tz="UTC") > pd.Timestamp(a["end"], tz="UTC")
        assert pd.Timestamp(b["start"], tz="UTC") <= pd.Timestamp(a["end"], tz="UTC") + pd.Timedelta("72h")
    assert SP.OOF_FREQ == "3h" and SP.TRAIN_FREQ == "3h" and SP.PURGE_H == 72


def test_single_axis_recipe():
    run_src = (EXP / "run_oofset.py").read_text()
    for mod in ("info_ablation", "strips", "spec_oof", "oofset_lib", "existing_all"):
        assert mod in run_src, mod
    assert "900" in run_src and "0.03" in run_src
    assert "strip_columns" in run_src and "select_slots" in run_src
    assert "frame_residuals" in run_src and "summarize_fold_residuals" in run_src
    assert "pool_distribution" in run_src and "propose_threshold_refs" in run_src
    low = (EXP / "oofset_lib.py").read_text().lower() + run_src.lower()
    assert "enlil" not in low
    assert "residual_for_origins" not in run_src
    assert "fit_blend_weight" not in run_src and "blend_dv" not in run_src


def test_mixture_helpers():
    OL = _load("oofset_lib", "d150_lib_b")
    SP = _load("spec_oof", "d150_spec_d")
    blocks = [set(OL.level_seeds(s)) for s in (0, 1, 2)]
    assert len(set().union(*blocks)) == 9
    for s in (0, 1, 2):
        assert OL.event_seed(s) in OL.level_seeds(s)
    p = OL.fit_params(dict(SP.BASE_PARAMS), 3, 4)
    assert p["max_depth"] == 3 and p["random_state"] == 4 and p["n_estimators"] == 900
    OL.check_weights(SP.WEIGHTS)
    rng = np.random.default_rng(0)
    h = np.tile(np.arange(1, 73), 4)
    mem = {m: rng.normal(400, 20, len(h)) for m in ("e0003_line", "d33_hsplit", "d45_lvlmean")}
    out = OL.blend_per_lead(mem, SP.WEIGHTS, h)
    assert np.isfinite(out).all() and len(out) == len(h)
    m1 = h <= 6
    m2 = (h >= 7) & (h <= 24)
    m3 = h > 24
    st = OL.stitch_members({s: mem["e0003_line"][m1] for s in (0, 1, 2)},
                           {s: mem["e0003_line"][m2] for s in (0, 1, 2)},
                           mem["e0003_line"], mem["e0003_line"][m3], m1, m2, m3, (0, 1, 2), 1)
    assert set(st) == {"e0003_line", "d33_hsplit", "d45_lvlmean"}
    assert OL.as_path_matrix(np.arange(72 * 3, dtype=float)).shape == (3, 72)
    assert OL.lead_of(1) == "1-6" and OL.lead_of(72) == "49-72"
    try:
        OL.check_weights({"1-6": dict(SP.WEIGHTS["1-6"])})
    except ValueError:
        pass
    else:
        raise AssertionError("partial weights must be rejected")


def test_residual_frame_schema():
    OL = _load("oofset_lib", "d150_lib_c")
    n = 72 * 2
    te_o = pd.to_datetime(["2023-06-01T00:00:00Z"] * n, utc=True)
    fr = OL.frame_residuals(te_o, np.tile(np.arange(1, 73), 2),
                            np.full(n, 400.0), np.full(n, 395.0))
    assert list(fr.columns) == ["origin_utc", "h", "y_kms", "base_kms", "dv_kms"]
    assert len(fr) == n and np.allclose(fr["dv_kms"], 5.0)
    assert "origin_last_input_utc" not in list(fr.columns) and "pred_kms" not in list(fr.columns)


def test_summary_dist_thresholds_synthetic():
    OL = _load("oofset_lib", "d150_lib_d")
    SP = _load("spec_oof", "d150_spec_e")
    rng = np.random.default_rng(1)
    n = 72 * 20
    h = np.tile(np.arange(1, 73), 20)
    y = rng.normal(400, 30, n)
    b = y + rng.normal(0, 10, n)
    s = OL.summarize_fold_residuals(h, y, b, SP.QUANTILES)
    assert set(s) == {"1-6", "7-24", "25-48", "49-72"}
    for lead, rec in s.items():
        assert rec["n"] > 0 and np.isfinite(rec["mse"]) and np.isfinite(rec["bias"])
        qs = [rec["quantiles"][str(q)] for q in SP.QUANTILES]
        assert qs == sorted(qs) and rec["q90_abs"] <= rec["q95_abs"] <= rec["q99_abs"]
    fr = pd.DataFrame({"h": h, "dv_kms": y - b})
    d = OL.pool_distribution([fr, fr], SP.QUANTILES)
    assert set(d) == set(s) and all(r["n"] == 2 * len(fr) // 4 or r["n"] > 0 for r in d.values())
    refs = OL.propose_threshold_refs(s, SP.CLIP_CANDIDATES)
    assert set(refs) == set(s)
    for lead, r in refs.items():
        assert r["clip_grid"] == [25.0, 50.0, 75.0, 150.0]
    OL.check_clip_candidates(SP.CLIP_CANDIDATES)
    try:
        OL.check_clip_candidates((50.0,))
    except ValueError:
        pass
    else:
        raise AssertionError("single clip candidate must be rejected")


def test_poison_synthetic():
    OL = _load("oofset_lib", "d150_lib_e")
    idx = pd.date_range("2023-05-01T00:00:00Z", "2023-07-01T00:00:00Z", freq="h", tz="UTC")
    filled = pd.Series(400.0 + np.sin(np.arange(len(idx)) / 24.0), index=idx)
    cut = "2023-05-29T00:00:00Z"
    origins = pd.DatetimeIndex(["2023-05-10T00:00:00Z", "2023-05-20T00:00:00Z",
                                "2023-06-05T00:00:00Z", "2023-06-10T00:00:00Z"], tz="UTC")
    sc = OL.scenario_poison(origins, filled, cut)
    assert sc["passed"] is True and sc["pre_invariant"] is True and sc["moved"] is True
    assert sc["n_pre_rows"] == 2 and sc["n_post_rows"] == 2

    def _frame_for(series):
        o = pd.DatetimeIndex(origins, tz="UTC")
        return np.tile(series.reindex(o).to_numpy(dtype=float)[:, None], (1, 72))

    fp = OL.frame_poison(origins, filled, cut, _frame_for)
    assert fp["passed"] is True and fp["pre_invariant"] is True


def test_seed_actually_propagates():
    run_src = (EXP / "run_oofset.py").read_text()
    assert "fit_seed_models" in run_src and "event_seed" in run_src
    assert "level_seeds" in run_src and "fit_params" in run_src


def test_residual_only_outputs():
    run_src = (EXP / "run_oofset.py").read_text()
    for w in ("oof_resid_", "residual_summary.json", "residual_dist.json",
              "thresholds.json", "causality.json", "meta.json"):
        assert w in run_src, w
    for w in ("predictions_official", "origin_last_input_utc", "pred_kms",
              "SELECT_RULE", "official-score", "official_score", "freeze",
              "submit", "upload", "push"):
        assert w not in run_src, w
    assert "NO_SELECTION" in run_src and "NO_BLEND" in run_src and "NO_OFFICIAL" in run_src


def test_scratch_isolation_and_cpu():
    import re
    SP = _load("spec_oof", "d150_spec_f")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1g0"
    assert len(set(re.findall(r"worker-[a-z0-9]+", code))) == 1
    assert "worker-t1g0" in code
    run_src = (EXP / "run_oofset.py").read_text()
    assert 'device") == "cpu"' in run_src and "n_jobs" in run_src
    assert "--gpu" not in code


def test_poison_covers_fold_inputs():
    run_src = (EXP / "run_oofset.py").read_text()
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
