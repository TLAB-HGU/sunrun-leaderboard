"""D148 Step 1 oracle-headroom contract: static + synthetic checks (fast, no fits).

No real data fits, no arm registration, no official reads. Covers the owned
file budget, spec identity (D148/worker-t1e1/full/E0408), the single-axis
rule (E0408 base identical + hindsight IPS columns only), CPU/n_jobs
budget, seed propagation, output schema, scratch isolation, poison coverage
of the new IPS input, and forbidden-source absence (truth/folds/naive,
post-hoc flags, GONG/CDAW, GPU, other owners' ids and code).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d148_cme_timing"
D142 = LB / "experiments/d142_e0147depth"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_IDS = ["e0147-abl", "d141_e0147", "ablate", "lean.py", "D147", "d147",
             "nn-", "sun-nn", "E0471", "E0472", "donki-live", "d148_enlil",
             "worker-t1a", "worker-t1b", "worker-t1c", "worker-t1d",
             "worker-d141", "worker-d147", "worker-d76", "worker-t2no"]
BANNED_SRC = ["GONG", "CDAW", "ADAPT", "WSA", "ENLIL"]
POSTHOC = ["isMostAccurate", "versionId", "submissionTime"]
PINNED_W = {"1-6": {"e0003_line": 0.21536233464375223, "d33_hsplit": 0.3899845818930272,
                    "d45_lvlmean": 0.3946530834632206},
            "7-24": {"e0003_line": 0.3152013131693896, "d33_hsplit": 0.3436571856823209,
                     "d45_lvlmean": 0.3411415011482895},
            "25-48": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
                      "d45_lvlmean": 0.3333333333333333},
            "49-72": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
                      "d45_lvlmean": 0.3333333333333333}}

sys.path.insert(0, str(EXP))


def _load(exp_dir, mod_name, alias):
    loc = importlib.util.spec_from_file_location(alias, exp_dir / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(EXP.glob("*.py"))
                     if p.name in ("spec_oracle.py", "ips_oracle.py", "run_oracle.py"))


def _synth_arrivals():
    return pd.DatetimeIndex(["2026-06-05T00:00:00Z", "2026-06-10T12:00:00Z",
                             "2026-06-20T00:00:00Z"], tz="UTC")


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 9, files
    assert {"spec_oracle.py", "ips_oracle.py", "run_oracle.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load(EXP, "spec_oracle", "d148_spec")
    assert SP.DIRECTION == "D148" and SP.OWNER == "worker-t1e1"
    assert SP.KIND == "full" and SP.REFERENCE == "E0408"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24
    assert SP.SCRATCH == "store/scratch/worker-t1e1"
    assert SP.FIT_NJOBS <= 6
    assert SP.BASE_PARAMS["n_estimators"] == 900
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - 0.03) < 1e-12
    assert SP.BASE_PARAMS.get("device") == "cpu" and int(SP.BASE_PARAMS["n_jobs"]) <= 6
    assert SP.WEIGHTS == PINNED_W
    assert tuple(SP.IPS_COLS) == ("ips_next_arr_h", "ips_since_arr_h", "ips_dt_target")
    assert "[axes:" in SP.AXES_TAG
    assert float(SP.EXPECTED_LINE_MSE) == 90.0
    rule = SP.SELECT_RULE
    assert "lb-eval" in rule and "F1" in rule and "90" not in rule  # 90 lives in EXPECTED_LINE_MSE
    assert "low_low_high" in rule and "high_high_high" in rule


def test_single_axis_vs_recipe():
    SP = _load(EXP, "spec_oracle", "d148_spec_b")
    run_src = (EXP / "run_oracle.py").read_text()
    for mod in ("run_eta", "depth_lib", "info_ablation", "strips", "existing_all"):
        assert mod in run_src, mod
    assert "spec_oracle" in run_src and "ips_oracle" in run_src
    assert "900" in run_src and "0.03" in run_src
    assert "IPS_COLS" in run_src
    assert "single axis violated" in run_src
    assert SP.BASE_PARAMS["subsample"] == 0.8 and SP.BASE_PARAMS["colsample_bytree"] == 0.7
    assert SP.BASE_PARAMS["min_child_weight"] == 20 and SP.BASE_PARAMS["reg_lambda"] == 5
    assert SP.BASE_PARAMS["tree_method"] == "hist"


def test_oracle_columns_are_true_minus_target():
    IO = _load(EXP, "ips_oracle", "d148_io")
    assert tuple(IO.IPS_COLS) == ("ips_next_arr_h", "ips_since_arr_h", "ips_dt_target")
    arr = _synth_arrivals()
    df = pd.DataFrame({"origin": pd.to_datetime(["2026-06-04T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    out = IO.add_ips_oracle(df, arr)
    assert list(out.columns[-3:]) == list(IO.IPS_COLS)
    # next true arrival 2026-06-05T00:00Z: 24 h after origin; dt_target = arrival - target
    assert float(out["ips_next_arr_h"].iloc[0]) == 24.0
    assert float(out["ips_dt_target"].iloc[0]) == 24.0 - 1.0
    assert float(out["ips_dt_target"].iloc[71]) == 24.0 - 72.0
    # no future arrival -> NaN, never fabricated
    late = pd.DataFrame({"origin": pd.to_datetime(["2027-01-01T00:00:00Z"] * 4, utc=True),
                         "h": [1, 2, 3, 4]})
    out2 = IO.add_ips_oracle(late, arr)
    assert bool(out2["ips_dt_target"].isna().all())


def test_poison_probe_invariance_and_positive_control():
    IO = _load(EXP, "ips_oracle", "d148_io_b")
    arr = _synth_arrivals()
    origins = pd.date_range("2026-06-01", periods=10, freq="12h", tz="UTC")
    rec = IO.ips_poison_probe(origins, "2026-06-08T00:00:00Z", arr, shift_h=72.0)
    assert rec["passed"] is True and rec["pre_invariant"] is True
    assert rec["moved"] > 0 and rec["n_pre_rows"] > 0


def test_ips_arrivals_use_event_times_only():
    IO = _load(EXP, "ips_oracle", "d148_io_c")
    arr = IO.ips_arrivals()
    assert len(arr) > 200 and arr.is_monotonic_increasing
    assert str(arr.tz) == "UTC" and arr.duplicated().sum() == 0


def test_seed_actually_propagates():
    DL = _load(D142, "depth_lib", "d142_dl_ref")
    run_src = (EXP / "run_oracle.py").read_text()
    assert "fit_seed_models" in run_src and "event_seed" in run_src
    assert "level_seeds" in run_src or "level" in run_src
    blocks = [set(DL.level_seeds(s)) for s in (0, 1, 2)]
    assert len(set().union(*blocks)) == 9
    for s in (0, 1, 2):
        assert DL.event_seed(s) in DL.level_seeds(s)


def test_output_frame_schema():
    DL = _load(D142, "depth_lib", "d142_dl_schema")
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = DL.frame_predictions(te, np.arange(72, dtype=float))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert len(fr) == 72 and fr["horizon_hours"].tolist() == list(range(1, 73))
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_scratch_isolation_and_cpu():
    import re
    SP = _load(EXP, "spec_oracle", "d148_spec_d")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1e1"
    assert len(set(re.findall(r"worker-[a-z0-9]+", code))) == 1
    assert "worker-t1e1" in code
    run_src = (EXP / "run_oracle.py").read_text()
    assert 'device") == "cpu"' in run_src and "n_jobs" in run_src
    assert "--gpu" not in code


def test_poison_covers_new_inputs():
    run_src = (EXP / "run_oracle.py").read_text()
    assert "poison_test" in run_src and "ips_poison_probe" in run_src
    assert '"passed"' in run_src and "poison.json" in run_src


def test_no_forbidden_leak_or_banned_source():
    code = _code()
    for w in FORBIDDEN:
        assert w not in code, w
    for w in OTHER_IDS:
        assert w not in code, w
    low = code.lower()
    for w in BANNED_SRC:
        assert w.lower() not in low, w
    for w in POSTHOC:
        assert w not in code, w
    assert "cuda" not in low and "--gpu" not in code
    assert "truth" not in low or "official-window truth" in low
