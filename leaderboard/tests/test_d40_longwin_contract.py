"""D40 longwin contract: fixed 2x windows, E0003-identical params/rows except
infra (threads+GPU), no forbidden inputs, scratch separation.
Fast: no data reads, no fits."""

import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d40_longwin"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import longwin as LW  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]


def test_window_rule_fixed_and_causal():
    assert LW.SPEED_LONG_WINS == (12, 48, 144)
    assert LW.ACE_LONG_WINS == (12, 48)
    assert LW.CH_LONG_WINS == (48, 96)
    assert LW.SLOPE_SPAN_H == 48
    assert LW.check_windows()["window_def"] == LW.WINDOW_DEF
    assert "v_m144" in LW.WINDOW_DEF and "m96" in LW.WINDOW_DEF
    assert "t<=O" in LW.WINDOW_DEF
    assert "2x" in LW.SELECT_RULE or "doubled" in LW.SELECT_RULE
    assert "X0009" in LW.SELECT_RULE
    assert "72-96h" in LW.SELECT_RULE or "72-96" in LW.SELECT_RULE
    assert "72h purge" in LW.SELECT_RULE
    assert "No fitted choice" in LW.SELECT_RULE or "no fitted choice" in LW.SELECT_RULE.lower()


def test_longwin_columns_cover_speed_ace_ch():
    cols = LW.longwin_columns(["ch_a", "ch_b"])
    assert "v_m12" in cols and "v_m48" in cols and "v_m144" in cols
    assert "v_sd12" in cols and "v_d48" in cols
    assert "ace_speed_kms_m12" in cols and "ace_speed_kms_m48" in cols
    assert "ace_speed_kms_d48" in cols
    assert "ch_a_m48" in cols and "ch_a_m96" in cols and "ch_b_m96" in cols
    base = set(ia.arms(["h", "v_m6", "ace_speed_kms_m6", "ch_x"] + cols)["existing_all"])
    assert not (set(cols) & base - set(cols)), "longwin cols must be new"
    for c in cols:
        assert c not in ("h", "v_m6", "ace_speed_kms_m6"), c


def test_fit_params_match_e0003_recipe_infra_only():
    p = LW.fit_params("cuda:1")
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["tree_method"] == "hist"
    assert str(p["device"]).startswith("cuda")
    assert p["n_jobs"] == LW.FIT_NJOBS == 3
    assert LW.PERIOD_JOBS * LW.FIT_NJOBS <= 10


def test_devices_round_robin_visible_gpus(monkeypatch):
    monkeypatch.delenv("EXPOS_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert LW.device_for(0) == "cuda"
    monkeypatch.setenv("EXPOS_GPUS", "0,1")
    assert LW.device_for(0) == "cuda:0"
    assert LW.device_for(1) == "cuda:1"
    assert LW.device_for(2) == "cuda:0"


def test_full_rule_prespecified_and_no_fitted_choice():
    assert "full 3h" in LW.FULL_RULE and "E0001" in LW.FULL_RULE
    assert "F1" in LW.FULL_RULE
    assert "no full run" in LW.FULL_RULE
    assert "0.41121495" in LW.FULL_RULE  # adoption bar cited, report-only
    assert "freeze" in LW.FULL_RULE.lower()


def test_owned_files_and_no_vendored_code():
    pkg = LB / "experiments" / "d40_longwin"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 5, files
    assert {"longwin.py", "runner.py"} <= set(files)
    runner = (pkg / "runner.py").read_text()
    assert "import strips" in runner or "import ST" in runner
    assert "import longwin" in runner or "import LW" in runner
    assert "store/scratch/worker-d40" in runner
    for other in ("worker-d37", "worker-d38", "worker-d39", "worker-d35", "worker-d36",
                  "worker-d33", "worker-d34", "worker-d21", "worker-d28", "worker-d22", "worker-d23"):
        assert other not in runner
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()


def test_runner_writes_three_predictions_and_real_poison():
    runner = (LB / "experiments" / "d40_longwin" / "runner.py").read_text()
    assert "for i, n in enumerate(names)" in runner
    assert "poison.json" in runner and '"passed"' in runner
    assert "ST.poison_test" in runner
    assert "poison_longwin" in runner
    assert "compute_longwin" in runner
    assert "add_longwin" in runner
    assert "fcache" in runner.lower() and "featcache" in runner
    assert "device" in runner and "cuda" in runner
    assert "SystemExit" in runner  # fail-closed on poison/non-finite


def test_design_cols_deduped_arms_sweep():
    runner = (LB / "experiments" / "d40_longwin" / "runner.py").read_text()
    assert "design_cols" in runner
    assert "dict.fromkeys" in runner  # arms() sweeps longwin cols into families; dedupe
    assert "duplicate design columns" in runner
