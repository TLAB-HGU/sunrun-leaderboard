"""D67 family-ablation contract: leave-one-family-out, dev-OOF-only winner.

Fast: no data reads, no heavy fits (numpy synthetic only, tiny XGB).
"""

import sys
from pathlib import Path

import numpy as np

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d67_famabl"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import famspec as FS  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_dual_tag_and_experience_citation():
    assert FS.FAMILY_TAG == "[family=xgb]"
    assert FS.AXES_TAG.startswith("[axes:") and "sel=" in FS.AXES_TAG
    assert FS.HYPOTHESIS.startswith("[family=xgb] [axes:")
    for token in ("E0114", "X0105", "E0001"):
        assert token in FS.HYPOTHESIS, f"hypothesis must cite {token}"
    assert "dev-OOF" in FS.HYPOTHESIS or "dev-only" in FS.HYPOTHESIS
    assert "freeze" in FS.HYPOTHESIS.lower() and "official-score" in FS.HYPOTHESIS
    assert "6h" in FS.HYPOTHESIS and "gpu1" in FS.HYPOTHESIS


def test_families_partition_design_exactly_once():
    cols = (["h", "v_lag1", "v_m6", "rec26", "rec_mean"]
            + ["ace_speed", "speed_d24", "epam_p1_d6"]
            + ["ch_area", "ch_lat"]
            + ["strip_v350_logmed", "img_age_h", "img_stale"])
    fams = FS.families_of(cols)
    assert set(fams) == {"h", "speed", "plasma", "ch", "strips"}
    assert fams["h"] == ["h"]
    assert set(fams["speed"]) == {"v_lag1", "v_m6", "rec26", "rec_mean"}
    assert set(fams["plasma"]) == {"ace_speed", "speed_d24", "epam_p1_d6"}
    assert set(fams["ch"]) == {"ch_area", "ch_lat"}
    assert set(fams["strips"]) == {"strip_v350_logmed", "img_age_h", "img_stale"}
    flat = ["h"] + fams["speed"] + fams["plasma"] + fams["ch"] + fams["strips"]
    assert sorted(flat) == sorted(cols)


def test_arms_leave_one_family_out_and_keep_h():
    cols = (["h", "v_lag1", "rec26"] + ["ace_x", "speed_d24"]
            + ["ch_a"] + ["strip_v350_logmed", "img_stale"])
    arms = FS.arms_of(cols)
    assert tuple(arms) == ("full", "no_speed", "no_plasma", "no_ch", "no_strips")
    assert set(arms["full"]) == set(cols)
    for name, arm in arms.items():
        assert "h" in arm, f"{name} must keep h"
    assert "v_lag1" not in arms["no_speed"] and "rec26" not in arms["no_speed"]
    assert "ace_x" not in arms["no_plasma"] and "speed_d24" not in arms["no_plasma"]
    assert "ch_a" not in arms["no_ch"]
    assert "strip_v350_logmed" not in arms["no_strips"] and "img_stale" not in arms["no_strips"]
    # each arm drops exactly its family
    assert len(arms["no_speed"]) == len(cols) - 2
    assert len(arms["no_ch"]) == len(cols) - 1
    assert FS.rank_arms({"full": {"val_mse": 10.0}, "no_speed": {"val_mse": 9.0},
                         "no_plasma": {"val_mse": 11.0}, "no_ch": {"val_mse": 10.5},
                         "no_strips": {"val_mse": 9.5}})[0] == "no_speed"


def test_spec_and_dev_only_split():
    assert FS.check_spec()["arms"] == ["full", "no_speed", "no_plasma", "no_ch", "no_strips"]
    assert FS.VAL_LO_UTC == "2025-10-15T00:00:00Z" and FS.PURGE_H == 72 and FS.SEED == 0
    assert "dev-OOF" in FS.SELECT_RULE and "selection/audit" in FS.SELECT_RULE
    assert "never read" in FS.SELECT_RULE
    assert "E0114" in FS.SELECT_RULE and "X0105" in FS.SELECT_RULE
    import pandas as pd
    assert pd.Timestamp(FS.DEV_LO_UTC) < pd.Timestamp(FS.VAL_LO_UTC) < pd.Timestamp(FS.DEV_END_UTC)
    n = 200
    rng = np.random.default_rng(0)
    origins = pd.date_range("2025-08-04", "2025-11-27 23:00", freq="6h", tz="UTC")
    df = pd.DataFrame({"origin": rng.choice(origins, n),
                       "target_t": rng.choice(origins, n), "y": rng.normal(400, 50, n)})
    df = df.sort_values("origin").reset_index(drop=True)
    tr, va = FS.split_dev_oof(df)
    assert (va["origin"] >= pd.Timestamp(FS.DEV_LO_UTC)).all()
    assert (va["origin"] <= pd.Timestamp(FS.DEV_END_UTC)).all()
    assert (tr["target_t"] < pd.Timestamp(FS.VAL_LO_UTC) - pd.Timedelta(hours=72)).all()


def test_fit_params_match_e0003_recipe_infra_only():
    p = FS.fit_params("cuda:1")
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["tree_method"] == "hist"
    assert str(p["device"]).startswith("cuda")
    assert p["n_jobs"] == FS.FIT_NJOBS == 3
    assert FS.PERIOD_JOBS * FS.FIT_NJOBS <= 10


def test_devices_round_robin_visible_gpus(monkeypatch):
    monkeypatch.delenv("EXPOS_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert FS.device_for(0) == "cuda"
    monkeypatch.setenv("EXPOS_GPUS", "1")
    assert FS.device_for(0) == "cuda"
    monkeypatch.setenv("EXPOS_GPUS", "0,1")
    assert FS.device_for(0) == "cuda:0"
    assert FS.device_for(1) == "cuda:1"
    assert FS.device_for(2) == "cuda:0"


def test_full_rule_prespecified_and_report_only():
    assert "full 3h" in FS.FULL_RULE and "E0001" in FS.FULL_RULE
    assert "F1" in FS.FULL_RULE and "0.30" in FS.FULL_RULE
    assert "no full run" in FS.FULL_RULE
    assert "freeze" in FS.FULL_RULE.lower() and "official-score" in FS.FULL_RULE
    assert "E0003" in FS.FULL_RULE and "seed-repro" in FS.FULL_RULE


def test_owned_files_and_no_vendored_strips():
    pkg = LB / "experiments" / "d67_famabl"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 5, files
    assert {"famspec.py", "runner.py"} <= set(files)
    runner = (pkg / "runner.py").read_text()
    assert "import strips" in runner or "import ST" in runner
    assert "import famspec" in runner or "import FS" in runner
    assert "store/scratch/worker-d67" in runner
    assert "worker-d66" not in runner
    for other in ("worker-d59", "worker-d60", "worker-d61", "worker-d63",
                  "worker-d64", "worker-d65", "worker-d45", "worker-d33"):
        assert other not in runner
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"


def test_runner_ablation_poison_gpu_cache_and_smoke():
    runner = (LB / "experiments" / "d67_famabl" / "runner.py").read_text()
    assert "dev_oof_ablate" in runner
    assert "arms_of" in runner and "families_of" in runner
    assert "rank_arms" in runner
    assert "ablation_table" in runner
    assert "ST.poison_test" in runner and '"passed"' in runner
    assert "SystemExit" in runner
    assert "ThreadPoolExecutor" in runner
    assert "device" in runner and "cuda" in runner
    assert "for i, n in enumerate(names)" in runner
    assert "ST.build_manifest" in runner
    assert "fcache" in runner.lower() and "featcache" in runner
    assert "EXPOS_SMOKE" in runner
    assert "design_cols" in runner and "dict.fromkeys" in runner
    assert "E0114" in runner or "X0105" in runner
