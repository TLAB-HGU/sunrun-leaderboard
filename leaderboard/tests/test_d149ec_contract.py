"""D149 Enlil-cells contract: static + synthetic checks (fast, no fits).

No real data fits, no arm registration, no official reads. Covers the owned
file budget, spec identity (D149/worker-t1f0/full/E0472), the single-axis
rule (E0472 base identical plus OOF-frozen bandcap only), band helpers,
OOF cap selection, vintage enforcement on synthetic cases, poison coverage
of the new ENLIL input, CPU/n_jobs budget, seed propagation, output
schema, scratch isolation, and forbidden-source absence.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d149_enlil_cells"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "lean.py", "D147", "d147",
                 "d148_enlil", "nn-", "sun-nn", "E0471",
                 "worker-t1a", "worker-t1b", "worker-t1c", "worker-t1d",
                 "worker-t1e", "worker-d147", "worker-d141", "worker-d76",
                 "worker-t2no"]
BANNED_SRC = ["GONG", "CDAW"]
CELL_LABELS = ["low_low_high", "low_high_high", "high_low_high", "high_high_high"]
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


def _code(names=("spec_ec.py", "cells_lib.py", "run_ec.py")):
    return "\n".join((EXP / n).read_text() for n in names)


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 6, files
    assert {"spec_ec.py", "cells_lib.py", "run_ec.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_ec", "d149_spec")
    assert SP.DIRECTION == "D149" and SP.OWNER == "worker-t1f0"
    assert SP.KIND == "full" and SP.REFERENCE == "E0472"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24
    assert SP.SCRATCH == "store/scratch/worker-t1f0"
    assert SP.FIT_NJOBS <= 6
    assert SP.BASE_PARAMS["n_estimators"] == 900
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - 0.03) < 1e-12
    assert SP.BASE_PARAMS.get("device") == "cpu" and int(SP.BASE_PARAMS.get("n_jobs")) <= 6
    assert SP.BASE_DEPTH == 3 and SP.WEIGHTS == PINNED_W
    assert tuple(SP.BANDS) == ((1, 6), (7, 24), (25, 72))
    assert tuple(SP.BAND_NAMES) == ("1-6", "7-24", "25-72")
    assert tuple(float(x) for x in SP.CAP_CANDIDATES) == (25.0, 50.0, 75.0, 150.0)
    assert int(SP.MASK_MIN_WINS) == 2
    assert SP.MAIN_WINDOW == "W2"
    assert [w["name"] for w in SP.WINDOWS] == ["W0", "W1", "W2"]
    assert SP.PATH_SOURCE == "oof-fit" and SP.RESIDUAL_DEF == "dv = y - yhat_base"
    assert SP.K == 15 and SP.SCENARIO_DIM == 96
    assert "[axes:" in SP.AXES_TAG and "post=clipping" in SP.AXES_TAG
    rule = SP.SELECT_RULE
    assert "lb-eval" in rule and "F1" in rule and "3281" in rule
    for c in CELL_LABELS:
        assert c in rule, c


def test_single_axis_vs_recipe():
    SP = _load("spec_ec", "d149_spec_b")
    run_src = (EXP / "run_ec.py").read_text()
    for mod in ("info_ablation", "strips", "spec_ec", "cells_lib", "existing_all"):
        assert mod in run_src, mod
    assert "900" in run_src and "0.03" in run_src
    assert "ENLIL_COLS" in run_src and "enlil_dt_target" in run_src
    assert "single axis violated" in run_src
    assert "std_equal" not in run_src or "std" in run_src
    assert "residual_multi" in run_src and "pool_scales" in run_src
    assert "select_best_cap" in run_src and "vote_band_masks" in (EXP / "cells_lib.py").read_text()
    assert "apply_capped_correction" in run_src
    assert SP.BASE_PARAMS["subsample"] == 0.8 and SP.BASE_PARAMS["colsample_bytree"] == 0.7
    assert SP.BASE_PARAMS["min_child_weight"] == 20 and SP.BASE_PARAMS["reg_lambda"] == 5
    assert SP.BASE_PARAMS["tree_method"] == "hist"
    for c in CELL_LABELS:
        assert c not in run_src, c
        assert c not in (EXP / "cells_lib.py").read_text(), c


def test_band_cap_helpers():
    CL = _load("cells_lib", "d149_lib")
    SP = _load("spec_ec", "d149_spec_c")
    assert CL.band_of(1) == "1-6" and CL.band_of(6) == "1-6"
    assert CL.band_of(7) == "7-24" and CL.band_of(24) == "7-24"
    assert CL.band_of(25) == "25-72" and CL.band_of(72) == "25-72"
    raw = np.array([-100.0, -50.0, 0.0, 30.0, 80.0])
    assert np.allclose(CL.clip_correction(raw, 50.0), [-50.0, -50.0, 0.0, 30.0, 50.0])
    base = np.full(72 * 2, 400.0)
    dv = np.full(72 * 2, 10.0)
    hh = np.tile(np.arange(1, 73), 2)
    fin, _, _, _, _ = CL.apply_capped_correction(base, dv, hh, 0.5, 50.0,
                                                {"1-6": True, "7-24": True, "25-72": True})
    assert np.allclose(fin, 405.0)
    fin2, _, _, _, _ = CL.apply_capped_correction(base, dv, hh, 0.5, 50.0,
                                                  {"1-6": False, "7-24": False, "25-72": False})
    assert np.allclose(fin2, 400.0)
    CL.check_bands(SP.BANDS)
    CL.check_band_names(SP.BAND_NAMES)
    CL.check_cap_candidates(SP.CAP_CANDIDATES)


def test_cap_selection_is_oof_only():
    CL = _load("cells_lib", "d149_lib_b")
    rng = np.random.default_rng(0)
    wins = {}
    for wn in ("W0", "W1", "W2"):
        n = 72 * 10
        base = rng.normal(400, 30, n)
        dv = rng.normal(0, 5, n)
        y = base + 0.5 * dv + rng.normal(0, 1, n)
        hh = np.tile(np.arange(1, 73), 10)
        wins[wn] = {"base": base, "dv": dv, "y": y, "h": hh}
    best, scored = CL.select_best_cap(wins, 0.5, (25.0, 50.0, 75.0, 150.0))
    assert best in (25.0, 50.0, 75.0, 150.0)
    assert set(scored) == {25.0, 50.0, 75.0, 150.0}
    for cap, rec in scored.items():
        assert set(rec["masks"]) == {"1-6", "7-24", "25-72"}
        assert np.isfinite(rec["mse"])
    mses = [rec["mse"] for rec in scored.values()]
    assert scored[best]["mse"] == min(mses)
    try:
        CL.select_best_cap(wins, 0.5, (50.0,))
    except ValueError:
        pass
    else:
        raise AssertionError("single cap candidate must be rejected")


def test_vintage_enlil_values():
    RN = _load("run_ec", "d149_run")
    assert "modelCompletionTime" in (EXP / "run_ec.py").read_text()
    ev = pd.DataFrame({"done": pd.to_datetime(["2026-06-01T00:00:00Z", "2026-06-10T00:00:00Z"], utc=True),
                       "arrival": pd.to_datetime(["2026-06-02T00:00:00Z", "2026-06-11T00:00:00Z"], utc=True),
                       "kp": [5.0, 7.0], "glancing": [True, False]})
    import info_ablation as ia
    sys.path.insert(0, str(LB / "experiments/headroom_audit"))
    o_pre = pd.DatetimeIndex(["2026-06-01T00:00:00Z"], tz="UTC")
    f_pre = ia.enlil_features(o_pre, ev)
    assert float(f_pre["enlil_next_arr_h"].iloc[0]) == 24.0
    o_old = pd.DatetimeIndex(["2026-05-01T00:00:00Z"], tz="UTC")
    f_old = ia.enlil_features(o_old, ev)
    assert bool(f_old["enlil_next_arr_h"].isna().all())
    rec = RN.enlil_poison_probe(pd.DatetimeIndex(["2026-05-01T00:00:00Z", "2026-06-15T00:00:00Z"], tz="UTC"),
                                "2026-05-29T00:00:00Z", ev)
    assert rec["passed"] is True and rec["pre_invariant"] is True
    assert rec["n_pre_rows"] > 0 and rec["moved"] > 0


def test_real_store_loads_vintage_clean():
    RN = _load("run_ec", "d149_run_b")
    ev = RN.enlil_runs()
    assert len(ev) > 100
    assert bool((ev["done"].diff().dropna() >= pd.Timedelta(0)).all())
    assert ev["done"].max() >= pd.Timestamp("2026-06-01T00:00:00Z", tz="UTC")


def test_seed_actually_propagates():
    CL = _load("cells_lib", "d149_lib_c")
    run_src = (EXP / "run_ec.py").read_text()
    assert "fit_set" in run_src and "event_seed" in run_src
    assert "level_seeds" in run_src
    blocks = [set(CL.level_seeds(s)) for s in (0, 1, 2)]
    assert len(set().union(*blocks)) == 9
    for s in (0, 1, 2):
        assert CL.event_seed(s) in CL.level_seeds(s)


def test_output_frame_schema():
    CL = _load("cells_lib", "d149_lib_d")
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = CL.frame_predictions(te, np.arange(72, dtype=float))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert len(fr) == 72 and fr["horizon_hours"].tolist() == list(range(1, 73))
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_scratch_isolation_and_cpu():
    import re
    SP = _load("spec_ec", "d149_spec_d")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1f0"
    assert len(set(re.findall(r"worker-[a-z0-9]+", code))) == 1
    assert "worker-t1f0" in code
    run_src = (EXP / "run_ec.py").read_text()
    assert 'device") == "cpu"' in run_src and "n_jobs" in run_src
    assert "--gpu" not in code


def test_poison_covers_new_inputs():
    run_src = (EXP / "run_ec.py").read_text()
    assert "poison_test" in run_src and "enlil_poison_probe" in run_src
    assert "scenario_poison" in run_src and "frame_poison" in run_src
    assert '"passed"' in run_src and "poison.json" in run_src


def test_no_forbidden_leak_or_banned_source():
    code = _code()
    for w in FORBIDDEN:
        assert w not in code, w
    for w in OTHER_CONTACT:
        assert w not in code, w
    low = code.lower()
    for w in BANNED_SRC:
        assert w.lower() not in low, w
    assert "cuda" not in low and "--gpu" not in code
