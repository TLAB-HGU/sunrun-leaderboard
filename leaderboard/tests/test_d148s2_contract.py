"""D148 Step 2 contract: vintage-clean DONKI analysis features (fast, no fits).

No real data fits, no arm registration, no official reads. Covers the owned
file budget, spec identity (D148/worker-t1e2/full/E0408), the single-axis
rule (E0408 base identical + eight donki_* columns only), the pre-registered
past-6-day window, vintage enforcement on synthetic cases (submissionTime <=
origin, time21_5 in the trailing window, post-hoc flags ignored), poison
coverage of the new DONKI input, CPU/n_jobs budget, seed propagation, output
schema, scratch isolation, and forbidden-source absence.
"""
import importlib.util
import json
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
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "lean.py", "D147", "d147",
                 "nn-", "sun-nn", "E0471", "donki-live", "d148_enlil",
                 "ips_oracle", "spec_oracle", "run_oracle",
                 "worker-t1e1", "worker-t1a", "worker-t1b", "worker-t1c",
                 "worker-t1d", "worker-d141", "worker-d147", "worker-d76",
                 "worker-t2no"]
BANNED_SRC = ["GONG", "CDAW", "ADAPT", "WSA", "ENLIL"]
PINNED_W = {"1-6": {"e0003_line": 0.21536233464375223, "d33_hsplit": 0.3899845818930272,
                    "d45_lvlmean": 0.3946530834632206},
            "7-24": {"e0003_line": 0.3152013131693896, "d33_hsplit": 0.3436571856823209,
                     "d45_lvlmean": 0.3411415011482895},
            "25-48": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
                      "d45_lvlmean": 0.3333333333333333},
            "49-72": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
                      "d45_lvlmean": 0.3333333333333333}}
S2_COLS = ("donki_n6d", "donki_vmax", "donki_hamax", "donki_vearth",
           "donki_haearth", "donki_offmin", "donki_clearmin",
           "donki_dt21earth")

sys.path.insert(0, str(EXP))


def _load(exp_dir, mod_name, alias):
    loc = importlib.util.spec_from_file_location(alias, exp_dir / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(EXP.glob("*.py"))
                     if p.name in ("spec_s2.py", "run_s2.py"))


def _synth_A():
    # (submissionTime, time21_5, speed, halfAngle, lat, lon)
    rows = [("2026-06-01T00:00:00Z", "2026-06-02T00:00:00Z", 600.0, 30.0, 0.0, 0.0),
            ("2026-06-01T00:00:00Z", "2026-06-03T00:00:00Z", 1000.0, 20.0, 0.0, 90.0),
            ("2026-06-10T00:00:00Z", "2026-06-02T00:00:00Z", 2000.0, 60.0, 0.0, 0.0)]
    sub = np.array([pd.Timestamp(s, tz="UTC").value for s, _, _, _, _, _ in rows])
    t21 = np.array([pd.Timestamp(t, tz="UTC").value for _, t, _, _, _, _ in rows])
    spd = np.array([r[2] for r in rows])
    ha = np.array([r[3] for r in rows])
    la = np.array([r[4] for r in rows]) * np.pi / 180.0
    lo = np.array([r[5] for r in rows]) * np.pi / 180.0
    off = np.arccos(np.clip(np.cos(la) * np.cos(lo), -1.0, 1.0)) * 180.0 / np.pi
    order = np.argsort(t21, kind="stable")
    return {"sub": sub[order], "t21": t21[order], "speed": spd[order],
            "half": ha[order], "off": off[order]}


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) == 9, files
    assert {"spec_s2.py", "run_s2.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load(EXP, "spec_s2", "d148_s2_spec")
    assert SP.DIRECTION == "D148" and SP.OWNER == "worker-t1e2"
    assert SP.KIND == "full" and SP.REFERENCE == "E0408"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24
    assert SP.SCRATCH == "store/scratch/worker-t1e2"
    assert SP.FIT_NJOBS <= 6
    assert SP.BASE_PARAMS["n_estimators"] == 900
    assert abs(float(SP.BASE_PARAMS["learning_rate"]) - 0.03) < 1e-12
    assert SP.BASE_PARAMS.get("device") == "cpu" and int(SP.BASE_PARAMS["n_jobs"]) <= 6
    assert SP.WEIGHTS == PINNED_W
    assert tuple(SP.DONKI_S2_COLS) == S2_COLS
    assert "[axes:" in SP.AXES_TAG
    assert float(SP.WINDOW_H) == 144.0
    rule = SP.SELECT_RULE
    assert "lb-eval" in rule and "F1" in rule and "3281" in rule
    assert "low_low_high" in rule and "high_high_high" in rule


def test_single_axis_vs_recipe():
    SP = _load(EXP, "spec_s2", "d148_s2_spec_b")
    run_src = (EXP / "run_s2.py").read_text()
    for mod in ("run_eta", "depth_lib", "info_ablation", "strips", "existing_all"):
        assert mod in run_src, mod
    assert "spec_s2" in run_src
    assert "900" in run_src and "0.03" in run_src
    assert "DONKI_S2_COLS" in run_src
    assert "single axis violated" in run_src
    assert SP.BASE_PARAMS["subsample"] == 0.8 and SP.BASE_PARAMS["colsample_bytree"] == 0.7
    assert SP.BASE_PARAMS["min_child_weight"] == 20 and SP.BASE_PARAMS["reg_lambda"] == 5
    assert SP.BASE_PARAMS["tree_method"] == "hist"


def test_vintage_window_values():
    A = _synth_A()
    RN = _load(EXP, "run_s2", "d148_s2_run")
    assert tuple(RN.SP.DONKI_S2_COLS) == S2_COLS
    df = pd.DataFrame({"origin": pd.to_datetime(["2026-06-04T00:00:00Z"] * 3, utc=True),
                       "h": [1, 2, 3]})
    out = RN.add_donki_s2(df, A)
    assert list(out.columns[-8:]) == list(S2_COLS)
    r = out.iloc[0]
    # usable: rows 0-1 (submitted pre-origin, time21_5 in trailing 144 h);
    # row 2 excluded (submissionTime after origin despite fast speed).
    assert float(r["donki_n6d"]) == 2.0
    assert float(r["donki_vmax"]) == 1000.0 and float(r["donki_hamax"]) == 20.0
    assert float(r["donki_vearth"]) == 600.0 and float(r["donki_haearth"]) == 30.0
    assert float(r["donki_offmin"]) == 0.0
    assert float(r["donki_clearmin"]) == -30.0
    assert float(r["donki_dt21earth"]) == 48.0
    # stale time21_5 (older than 144 h) and future time21_5 are excluded.
    old = pd.DataFrame({"origin": pd.to_datetime(["2026-06-20T00:00:00Z"] * 2, utc=True),
                        "h": [1, 2]})
    out_old = RN.add_donki_s2(old, A)
    assert float(out_old["donki_n6d"].iloc[0]) == 0.0
    assert bool(out_old["donki_vmax"].isna().all())
    assert bool(out_old["donki_dt21earth"].isna().all())


def test_posthoc_flags_ignored_at_load(tmp_path):
    RN = _load(EXP, "run_s2", "d148_s2_run_b")
    base = {"submissionTime": "2026-06-01T00:00:00Z", "time21_5": "2026-06-02T00:00:00Z",
            "speed": 700.0, "halfAngle": 25.0, "latitude": 10.0, "longitude": -20.0,
            "link": "https://example.invalid/rev/2", "versionId": 7}
    rows = [dict(base, isMostAccurate=True), dict(base, isMostAccurate=False)]
    fa = tmp_path / "CMEAnalysis_2026-06.json"
    fa.write_text(json.dumps(rows))
    old = RN.DONKI
    try:
        RN.DONKI = tmp_path
        A = RN.load_analyses()
    finally:
        RN.DONKI = old
    assert len(A["t21"]) == 2
    assert A["speed"].tolist() == [700.0, 700.0]
    assert A["speed"].tolist() == [700.0, 700.0]


def test_loader_reads_allowed_keys_only():
    run_src = (EXP / "run_s2.py").read_text()
    assert "submissionTime" in run_src and "time21_5" in run_src
    assert "isMostAccurate" not in run_src and "versionId" not in run_src


def test_poison_probe_invariance_and_positive_control():
    RN = _load(EXP, "run_s2", "d148_s2_run_c")
    rows = [("2026-05-31T23:00:00Z", "2026-06-01T00:00:00Z", 500.0, 20.0, 0.0, 10.0),
            ("2026-06-04T23:00:00Z", "2026-06-05T00:00:00Z", 800.0, 30.0, 5.0, -5.0),
            ("2026-06-19T23:00:00Z", "2026-06-20T00:00:00Z", 900.0, 25.0, 0.0, 0.0)]
    sub = np.array([pd.Timestamp(s, tz="UTC").value for s, _, _, _, _, _ in rows])
    t21 = np.array([pd.Timestamp(t, tz="UTC").value for _, t, _, _, _, _ in rows])
    spd = np.array([r[2] for r in rows])
    ha = np.array([r[3] for r in rows])
    la = np.array([r[4] for r in rows]) * np.pi / 180.0
    lo = np.array([r[5] for r in rows]) * np.pi / 180.0
    off = np.arccos(np.clip(np.cos(la) * np.cos(lo), -1.0, 1.0)) * 180.0 / np.pi
    order = np.argsort(t21, kind="stable")
    A = {"sub": sub[order], "t21": t21[order], "speed": spd[order],
         "half": ha[order], "off": off[order]}
    origins = pd.date_range("2026-06-01", "2026-06-25", freq="24h", tz="UTC")
    rec = RN.donki_poison_probe(origins, "2026-06-08T00:00:00Z", A, shift_h=72.0)
    assert rec["passed"] is True and rec["pre_invariant"] is True
    assert rec["moved"] > 0 and rec["n_pre_rows"] > 0


def test_real_store_loads_vintage_clean():
    RN = _load(EXP, "run_s2", "d148_s2_run_d")
    A = RN.load_analyses()
    assert len(A["t21"]) > 4000
    assert bool((np.diff(A["t21"]) >= 0).all())
    assert bool((A["speed"] > 0).all()) and bool((A["half"] > 0).all())


def test_seed_actually_propagates():
    DL = _load(D142, "depth_lib", "d142_dl_ref_s2")
    run_src = (EXP / "run_s2.py").read_text()
    assert "fit_seed_models" in run_src and "event_seed" in run_src
    assert "level_seeds" in run_src or "level" in run_src
    blocks = [set(DL.level_seeds(s)) for s in (0, 1, 2)]
    assert len(set().union(*blocks)) == 9
    for s in (0, 1, 2):
        assert DL.event_seed(s) in DL.level_seeds(s)


def test_output_frame_schema():
    DL = _load(D142, "depth_lib", "d142_dl_schema_s2")
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = DL.frame_predictions(te, np.arange(72, dtype=float))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert len(fr) == 72 and fr["horizon_hours"].tolist() == list(range(1, 73))
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_scratch_isolation_and_cpu():
    import re
    SP = _load(EXP, "spec_s2", "d148_s2_spec_d")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1e2"
    assert len(set(re.findall(r"worker-[a-z0-9]+", code))) == 1
    assert "worker-t1e2" in code
    run_src = (EXP / "run_s2.py").read_text()
    assert 'device") == "cpu"' in run_src and "n_jobs" in run_src
    assert "--gpu" not in code


def test_poison_covers_new_inputs():
    run_src = (EXP / "run_s2.py").read_text()
    assert "poison_test" in run_src and "donki_poison_probe" in run_src
    assert '"passed"' in run_src and "poison.json" in run_src


def test_no_forbidden_leak_or_banned_source():
    spec_src = (EXP / "spec_s2.py").read_text()
    run_src = (EXP / "run_s2.py").read_text()
    for w in FORBIDDEN:
        assert w not in spec_src, w
        assert w not in run_src, w
    for w in OTHER_CONTACT:
        assert w not in run_src, w
    assert "E0471" not in spec_src
    low = (spec_src + run_src).lower()
    for w in BANNED_SRC:
        assert w.lower() not in low, w
    assert "cuda" not in low and "--gpu" not in (spec_src + run_src)
