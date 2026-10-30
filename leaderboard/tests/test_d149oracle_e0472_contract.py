"""D149 oracle-on-E0472 contract: static + synthetic checks (fast, no fits).

No real data fits, no arm registration, no official reads. Covers the owned
file budget, spec identity (D149/worker-t1f1/full/E0472), the single-axis
rule (E0472 recipe identical plus hindsight IPS columns only), the oracle
column definition (true arrival minus target time), poison coverage of the
new IPS input, CPU/n_jobs budget, scratch isolation, output schema, and
forbidden-source absence.
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
POSTHOC = ["isMostAccurate", "versionId", "submissionTime"]
BANNED_SRC = ["GONG", "CDAW"]
CELL_LABELS = ["low_low_high", "low_high_high", "high_low_high", "high_high_high"]

sys.path.insert(0, str(EXP))


def _load(mod_name, alias):
    loc = importlib.util.spec_from_file_location(alias, EXP / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code(names=("spec_oracle_e0472.py", "run_oracle_e0472.py")):
    return "\n".join((EXP / n).read_text() for n in names)


def _synth_arrivals():
    return pd.DatetimeIndex(["2026-06-05T00:00:00Z", "2026-06-10T12:00:00Z",
                             "2026-06-20T00:00:00Z"], tz="UTC")


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 6, files
    assert {"spec_ec.py", "cells_lib.py", "run_ec.py",
            "spec_oracle_e0472.py", "run_oracle_e0472.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_oracle_e0472", "d149_oracle_spec")
    assert SP.DIRECTION == "D149" and SP.OWNER == "worker-t1f1"
    assert SP.KIND == "full" and SP.REFERENCE == "E0472"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24
    assert SP.SCRATCH == "store/scratch/worker-t1f1"
    assert SP.FIT_NJOBS <= 6
    assert SP.BASE_PARAMS["n_estimators"] == 900
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - 0.03) < 1e-12
    assert SP.BASE_PARAMS.get("device") == "cpu" and int(SP.BASE_PARAMS.get("n_jobs")) <= 6
    assert SP.BASE_DEPTH == 3
    assert tuple(SP.IPS_COLS) == ("ips_next_arr_h", "ips_since_arr_h", "ips_dt_target")
    assert "[axes:" in SP.AXES_TAG and "ips-oracle" in SP.AXES_TAG
    assert float(SP.EXPECTED_LINE_MSE) == 90.0
    assert "3281" in SP.E0472_MSE.__str__() or SP.E0472_MSE == 3281.31
    rule = SP.SELECT_RULE
    assert "lb-eval" in rule and "3281" in rule and "90" in rule
    assert "not selection" in rule
    for c in CELL_LABELS:
        assert c in rule, c


def test_single_axis_vs_recipe():
    SP = _load("spec_oracle_e0472", "d149_oracle_spec_b")
    run_src = (EXP / "run_oracle_e0472.py").read_text()
    for mod in ("lean as L", "info_ablation", "strips as ST", "run_official_a2b",
                "spec_oracle_e0472"):
        assert mod in run_src, mod
    assert '"s3_std"' in run_src and "--arm" in run_src
    assert "single axis violated" in run_src
    assert "enlil_dt_target" in run_src
    assert "predictions_official_s" in run_src and "oracle.json" in run_src
    assert SP.BASE_PARAMS["subsample"] == 0.8 and SP.BASE_PARAMS["colsample_bytree"] == 0.7
    assert SP.BASE_PARAMS["min_child_weight"] == 20 and SP.BASE_PARAMS["reg_lambda"] == 5
    assert SP.BASE_PARAMS["tree_method"] == "hist"
    assert "read_only" in run_src or "read-only" in run_src


def test_oracle_columns_are_true_minus_target():
    RN = _load("run_oracle_e0472", "d149_oracle_run")
    arr = _synth_arrivals()
    grid = pd.DataFrame({"origin": np.repeat(pd.DatetimeIndex(["2026-06-09T00:00:00Z"], tz="UTC").to_numpy(), 72),
                         "h": np.arange(1, 73)})
    out = RN.add_ips_oracle(grid, arr)
    assert list(out.columns[-3:]) == ["ips_next_arr_h", "ips_since_arr_h", "ips_dt_target"]
    # next true arrival 2026-06-10T12:00Z is 36 h after the origin
    assert bool((out["ips_next_arr_h"] == 36.0).all())
    # ips_dt_target is true arrival minus target time
    assert np.allclose(out["ips_dt_target"].to_numpy(), 36.0 - np.arange(1, 73))
    # far-past origin: no pending arrival in 6 h + 96 h window for since-branch check
    grid2 = pd.DataFrame({"origin": np.repeat(pd.DatetimeIndex(["2026-06-01T00:00:00Z"], tz="UTC").to_numpy(), 72),
                          "h": np.arange(1, 73)})
    out2 = RN.add_ips_oracle(grid2, arr)
    assert bool((out2["ips_next_arr_h"] == 96.0).all())
    assert bool(out2["ips_since_arr_h"].isna().all())


def test_poison_probe_invariance_and_positive_control():
    RN = _load("run_oracle_e0472", "d149_oracle_run_b")
    arr = _synth_arrivals()
    origins = pd.DatetimeIndex(["2026-06-01T00:00:00Z", "2026-06-09T00:00:00Z"], tz="UTC")
    rec = RN.ips_poison_probe(origins, "2026-06-06T00:00:00Z", arr)
    assert rec["passed"] is True and rec["pre_invariant"] is True
    assert rec["moved"] > 0 and rec["n_pre_rows"] > 0


def test_ips_arrivals_use_event_times_only():
    RN = _load("run_oracle_e0472", "d149_oracle_run_c")
    src = (EXP / "run_oracle_e0472.py").read_text()
    for w in POSTHOC:
        assert w not in src, w
    arr = RN.ips_arrivals()
    assert len(arr) > 0
    assert str(arr.tz) == "UTC" and arr.duplicated().sum() == 0
    assert bool((arr.sort_values() == arr).all())


def test_recipe_parity_with_e0472():
    SP = _load("spec_oracle_e0472", "d149_oracle_spec_c")
    sys.path.insert(0, str(LB / "experiments/d143_axisA_analog/a2b"))
    import spec_a2b as RECIPE
    assert tuple(SP.SEEDS) == tuple(RECIPE.SEEDS)
    assert SP.SMOKE_SEED == RECIPE.SMOKE_SEED
    assert SP.CUTOFF_ISO == RECIPE.CUTOFF_ISO
    assert SP.N_OFFICIAL_ORIGINS == RECIPE.N_OFFICIAL_ORIGINS
    assert SP.TRAIN_FREQ == RECIPE.TRAIN_FREQ and SP.SPLIT_H == RECIPE.SPLIT_H
    assert SP.BASE_PARAMS == RECIPE.BASE_PARAMS
    assert SP.BASE_DEPTH == RECIPE.BASE_DEPTH


def test_output_frame_schema():
    sys.path.insert(0, str(LB / "experiments/d143_axisA_analog/a2b"))
    import resid_lib_a2b as RL
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = RL.frame_predictions(te[["origin", "h"]], np.arange(72, dtype=float))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert len(fr) == 72 and fr["horizon_hours"].tolist() == list(range(1, 73))
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_scratch_isolation_and_cpu():
    import re
    SP = _load("spec_oracle_e0472", "d149_oracle_spec_d")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1f1"
    assert len(set(re.findall(r"worker-[a-z0-9]+", code))) == 1
    assert "worker-t1f1" in code
    assert 'device") == "cpu"' in code and "n_jobs" in code
    assert "--gpu" not in code


def test_poison_covers_new_inputs():
    run_src = (EXP / "run_oracle_e0472.py").read_text()
    assert "poison_test" in run_src and "ips_poison_probe" in run_src
    assert '"passed"' in run_src and "poison.json" in run_src


def test_no_forbidden_leak_or_banned_source():
    code = _code()
    for w in FORBIDDEN:
        assert w not in code, w
    low = code.lower()
    for w in BANNED_SRC:
        assert w.lower() not in low, w
    assert "cuda" not in low and "--gpu" not in code
    assert "research/expos/runs" not in code
    assert "load_truth" not in code
