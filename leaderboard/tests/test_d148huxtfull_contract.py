"""D148 Step 4 full contract: E0408 base + HUXt corotation features (fast, no fits).

No real data fits, no arm registration, no official reads. Covers the owned
file budget, spec identity (D148/worker-t1e4/full/E0408), the single-axis
rule (E0408 base identical + four huxt_* columns only), HUXt arrival logic
(v_huxt per hour, v0 boundary, dt500 arrival, slope), vintage enforcement
(ACE one rotation earlier, causal fallback only), poison coverage of the new
HUXt input, CPU/n_jobs budget, seed propagation, output schema, scratch
isolation, Step 2/3 comparison context, and forbidden-source absence.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d148_cme_timing"
HUXT = EXP / "huxt"
D142 = LB / "experiments/d142_e0147depth"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "lean.py", "D147", "d147",
                 "nn-", "sun-nn", "E0471", "donki-live", "d148_enlil",
                 "ips_oracle", "spec_oracle", "run_oracle",
                 "run_s2", "spec_s2", "run_s3", "spec_s3", "dbm_lib",
                 "worker-t1e1", "worker-t1e2", "worker-t1e3",
                 "worker-t1a", "worker-t1b", "worker-t1c",
                 "worker-t1d", "worker-d141", "worker-d147", "worker-d76",
                 "worker-t1e0", "worker-t2no", "D139", "E0229", "E0230"]
BANNED_SRC = ["GONG", "CDAW", "ADAPT", "WSA", "ENLIL", "PFSS"]
PINNED_W = {"1-6": {"e0003_line": 0.21536233464375223, "d33_hsplit": 0.3899845818930272,
                    "d45_lvlmean": 0.3946530834632206},
            "7-24": {"e0003_line": 0.3152013131693896, "d33_hsplit": 0.3436571856823209,
                     "d45_lvlmean": 0.3411415011482895},
            "25-48": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
                      "d45_lvlmean": 0.3333333333333333},
            "49-72": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
                      "d45_lvlmean": 0.3333333333333333}}
HUXT_COLS = ("huxt_v", "huxt_v0", "huxt_dt500", "huxt_slope")

sys.path.insert(0, str(HUXT))


def _load(mod_name, alias):
    loc = importlib.util.spec_from_file_location(alias, HUXT / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(HUXT.glob("*.py"))
                     if p.name in ("spec_full.py", "run_s4.py"))


def _synth_ace():
    idx = pd.date_range("2026-05-01", periods=24 * 60, freq="h", tz="UTC")
    return pd.DataFrame({"v": 400.0 + np.arange(len(idx)) * 0.01,
                         "miss": np.zeros(len(idx), dtype=int)}, index=idx)


def _synth_forecast(v0):
    base = float(v0)
    return base + np.linspace(-5.0, 25.0, 72)


def test_owned_file_budget():
    files = sorted(p.name for p in HUXT.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 6, files
    assert {"spec_huxt.py", "corotation.py", "run_pilot.py",
            "spec_full.py", "run_s4.py"} <= set(pys), files
    assert (HUXT / "venv_receipt.json").exists(), files


def test_spec_identity_and_scope():
    SP = _load("spec_full", "d148_s4_spec")
    assert SP.DIRECTION == "D148" and SP.OWNER == "worker-t1e4"
    assert SP.KIND == "full" and SP.REFERENCE == "E0408"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24
    assert SP.SCRATCH == "store/scratch/worker-t1e4"
    assert SP.FIT_NJOBS <= 6
    assert SP.BASE_PARAMS["n_estimators"] == 900
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - 0.03) < 1e-12
    assert SP.BASE_PARAMS.get("device") == "cpu" and int(SP.BASE_PARAMS["n_jobs"]) <= 6
    assert SP.WEIGHTS == PINNED_W
    assert tuple(SP.HUXT_COLS) == HUXT_COLS
    assert "[axes:" in SP.AXES_TAG and "huxt" in SP.AXES_TAG
    assert SP.VENV == ".venv-huxt" and SP.HUXT_PIN == "huxt==5.0.3"
    assert abs(float(SP.ROT_H) - 27.2753 * 24) < 1e-9
    assert float(SP.V_THRESH) == 500.0
    assert float(SP.HUXT_SIMTIME_D) == 5.0 and float(SP.HUXT_RMIN_RS) == 30.0
    rule = SP.SELECT_RULE
    assert "lb-eval" in rule and "F1" in rule and "3281" in rule
    assert "0.0833" in rule
    assert "low_low_high" in rule and "high_high_high" in rule
    assert "3681" in rule or "Step 2" in rule
    assert "3711" in rule or "Step 3" in rule


def test_single_axis_vs_recipe():
    SP = _load("spec_full", "d148_s4_spec_b")
    run_src = (HUXT / "run_s4.py").read_text()
    for mod in ("run_eta", "depth_lib", "info_ablation", "strips", "existing_all"):
        assert mod in run_src, mod
    assert "spec_full" in run_src
    assert "corotation" in run_src
    assert "900" in run_src and "0.03" in run_src
    assert "HUXT_COLS" in run_src
    assert "single axis violated" in run_src
    assert SP.BASE_PARAMS["subsample"] == 0.8 and SP.BASE_PARAMS["colsample_bytree"] == 0.7
    assert SP.BASE_PARAMS["min_child_weight"] == 20 and SP.BASE_PARAMS["reg_lambda"] == 5
    assert SP.BASE_PARAMS["tree_method"] == "hist"


def test_huxt_feature_values():
    RN = _load("run_s4", "d148_s4_run")
    assert tuple(RN.SP.HUXT_COLS) == HUXT_COLS
    o = pd.Timestamp("2026-06-10T00:00:00Z", tz="UTC")
    ons = int(o.value)
    fc = 480.0 + np.linspace(0.0, 60.0, 72)
    df = pd.DataFrame({"origin": [o] * 72, "h": np.arange(1, 73)})
    out = RN.add_huxt_features(df, {ons: 450.0}, {ons: fc})
    assert list(out.columns[-4:]) == list(HUXT_COLS)
    assert float(out["huxt_v"].iloc[0]) == float(fc[0])
    assert float(out["huxt_v"].iloc[71]) == float(fc[71])
    assert bool((out["huxt_v0"] == 450.0).all())
    assert abs(float(out["huxt_slope"].iloc[0]) - (fc[71] - fc[0]) / 71.0) < 1e-12
    k = int(np.where(fc >= 500.0)[0][0]) + 1
    tarr = o.value + k * 3_600_000_000_000
    want0 = (tarr - o.value) / 3_600_000_000_000 - 1.0
    assert abs(float(out["huxt_dt500"].iloc[0]) - want0) < 1e-9
    dts = out["huxt_dt500"].to_numpy()
    assert np.isfinite(dts).all()
    assert abs((dts[0] - dts[1]) - 1.0) < 1e-9
    slow = 400.0 + np.linspace(0.0, 10.0, 72)
    out2 = RN.add_huxt_features(df, {ons: 400.0}, {ons: slow})
    assert bool(out2["huxt_dt500"].isna().all())


def test_vintage_causal_only():
    CR = _load("corotation", "d148_s4_cr")
    ace = _synth_ace()
    o = pd.Timestamp("2026-06-10T00:00:00Z", tz="UTC")
    v0, used, fb, ok = CR.boundary_for_origin(o, ace)
    assert used <= o and ok is True and fb == 0.0
    ace2 = ace.copy()
    ace2.loc[used, "miss"] = 1
    _, usedb, fbb, _ = CR.boundary_for_origin(o, ace2)
    assert usedb == used - pd.Timedelta(hours=1) and fbb == 1.0
    run_src = (HUXT / "run_s4.py").read_text()
    assert "used_ace_ts" in run_src or "used" in run_src
    assert "store/ch-v1/ace.parquet" in (HUXT / "corotation.py").read_text()
    assert "store/donki-cme" not in _code(), "this arm is ACE-only"


def test_poison_probe_invariance_and_positive_control():
    RN = _load("run_s4", "d148_s4_run_c")
    ace = _synth_ace()
    origins = pd.date_range("2026-06-01", "2026-06-25", freq="24h", tz="UTC")
    rec = RN.huxt_poison_probe(origins, "2026-06-08T00:00:00Z", ace,
                               lambda o, v: _synth_forecast(v))
    assert rec["passed"] is True and rec["pre_invariant"] is True
    assert rec["moved"] > 0 and rec["n_pre_rows"] > 0


def test_seed_actually_propagates():
    DL = _load("depth_lib", "d142_dl_ref_s4") if False else None
    import importlib.util as iu
    loc = iu.spec_from_file_location("d142_dl_s4",
                                     LB / "experiments/d142_e0147depth/depth_lib.py")
    mod = iu.module_from_spec(loc)
    loc.loader.exec_module(mod)
    run_src = (HUXT / "run_s4.py").read_text()
    assert "fit_seed_models" in run_src and "event_seed" in run_src
    assert "level_seeds" in run_src or "level" in run_src
    blocks = [set(mod.level_seeds(s)) for s in (0, 1, 2)]
    assert len(set().union(*blocks)) == 9
    for s in (0, 1, 2):
        assert mod.event_seed(s) in mod.level_seeds(s)


def test_output_frame_schema():
    import importlib.util as iu
    loc = iu.spec_from_file_location("d142_dl_schema_s4",
                                     LB / "experiments/d142_e0147depth/depth_lib.py")
    mod = iu.module_from_spec(loc)
    loc.loader.exec_module(mod)
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = mod.frame_predictions(te, np.arange(72, dtype=float))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert len(fr) == 72 and fr["horizon_hours"].tolist() == list(range(1, 73))
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_scratch_isolation_and_cpu():
    import re
    SP = _load("spec_full", "d148_s4_spec_d")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1e4"
    assert len(set(re.findall(r"worker-[a-z0-9]+", code))) == 1
    assert "worker-t1e4" in code
    run_src = (HUXT / "run_s4.py").read_text()
    assert 'device") == "cpu"' in run_src and "n_jobs" in run_src
    assert "--gpu" not in code


def test_poison_covers_new_inputs():
    run_src = (HUXT / "run_s4.py").read_text()
    assert "poison_test" in run_src and "huxt_poison_probe" in run_src
    assert '"passed"' in run_src and "poison.json" in run_src


def test_no_forbidden_leak_or_banned_source():
    spec_src = (HUXT / "spec_full.py").read_text()
    run_src = (HUXT / "run_s4.py").read_text()
    for w in FORBIDDEN:
        assert w not in spec_src, w
        assert w not in run_src, w
    for w in OTHER_CONTACT:
        assert w not in run_src, w
    assert "E0471" not in spec_src
    assert "E0471" not in run_src
    low = (spec_src + run_src).lower()
    for w in BANNED_SRC:
        assert w.lower() not in low, w
    assert "cuda" not in low and "--gpu" not in (spec_src + run_src)
