"""D42 litfeat contract: fixed literature geometry, E0003-identical params/rows
except infra (threads+GPU), no forbidden inputs, scratch separation.
Fast: no data reads, no fits."""

import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d42_litfeat"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import litfeat as LF  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]


def test_feat_rule_fixed_and_causal():
    assert LF.CH_TGT_LAGS == (96, 120, 144, 168)
    assert LF.ROT_H == 648 and LF.FFILL_H == 6
    assert LF.CH_TGT_SERIES == "ch_t045_meridian30_equatorial_area"
    assert len(LF.NORTH_CELLS) == 12 and len(LF.SOUTH_CELLS) == 12
    assert LF.check_definition()["n_cols"] == 13
    assert len(LF.litfeat_columns()) == 13
    assert "t<=O" in LF.SELECT_RULE
    assert "X0009" in LF.SELECT_RULE
    assert "72-96h" in LF.SELECT_RULE
    assert "72h purge" in LF.SELECT_RULE
    assert "No fitted choice" in LF.SELECT_RULE or "no fitted choice" in LF.SELECT_RULE.lower()
    assert "sunspot" not in LF.SELECT_RULE.lower() or "no external" in LF.SELECT_RULE.lower()


def test_litfeat_columns_new_and_complete():
    cols = LF.litfeat_columns()
    assert "lit_b0" in cols and "lit_ns_asym" in cols and "lit_b0_x_asym" in cols
    assert "lit_north_area" in cols and "lit_south_area" in cols
    assert "lit_bright_now" in cols and "lit_bright_d27" in cols
    assert "lit_bright_mu648" in cols and "lit_bright_z648" in cols
    assert "lit_ch_tgt_m96" in cols and "lit_ch_tgt_m168" in cols
    base = set(ia.arms(["h", "v_m6", "ace_speed_kms_m6", "ch_x"] + cols)["existing_all"])
    assert not (set(cols) & base - set(cols)), "litfeat cols must be new"
    for c in cols:
        assert c.startswith("lit_"), c
        assert c not in ("h", "v_m6", "ace_speed_kms_m6"), c


def test_fit_params_match_e0003_recipe_infra_only():
    p = LF.fit_params("cuda:1")
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["tree_method"] == "hist"
    assert str(p["device"]).startswith("cuda")
    assert p["n_jobs"] == LF.FIT_NJOBS == 3
    assert LF.PERIOD_JOBS * LF.FIT_NJOBS <= 10


def test_devices_round_robin_visible_gpus(monkeypatch):
    monkeypatch.delenv("EXPOS_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert LF.device_for(0) == "cuda"
    monkeypatch.setenv("EXPOS_GPUS", "0,1")
    assert LF.device_for(0) == "cuda:0"
    assert LF.device_for(1) == "cuda:1"
    assert LF.device_for(2) == "cuda:0"


def test_full_rule_prespecified_and_no_fitted_choice():
    assert "full 3h" in LF.FULL_RULE and "E0001" in LF.FULL_RULE
    assert "F1" in LF.FULL_RULE
    assert "no full run" in LF.FULL_RULE
    assert "0.41121495" in LF.FULL_RULE  # adoption bar cited, report-only
    assert "freeze" in LF.FULL_RULE.lower()


def test_owned_files_and_no_vendored_code():
    pkg = LB / "experiments" / "d42_litfeat"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 5, files
    assert {"litfeat.py", "runner.py"} <= set(files)
    runner = (pkg / "runner.py").read_text()
    assert "import strips" in runner or "import ST" in runner
    assert "import litfeat" in runner or "import LF" in runner
    assert "store/scratch/worker-d42" in runner
    for other in ("worker-d37", "worker-d38", "worker-d39", "worker-d40", "worker-d41",
                  "worker-d35", "worker-d36", "worker-d33", "worker-d34",
                  "worker-d21", "worker-d28", "worker-d22", "worker-d23"):
        assert other not in runner
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()


def test_runner_writes_three_predictions_and_real_poison():
    runner = (LB / "experiments" / "d42_litfeat" / "runner.py").read_text()
    assert "for i, n in enumerate(names)" in runner
    assert "poison.json" in runner and '"passed"' in runner
    assert "ST.poison_test" in runner
    assert "poison_litfeat" in runner
    assert "compute_origin_litfeat" in runner
    assert "compute_target_lag" in runner
    assert "add_litfeat" in runner
    assert "fcache" in runner.lower() and "featcache" in runner
    assert "device" in runner and "cuda" in runner
    assert "SystemExit" in runner  # fail-closed on poison/non-finite


def test_design_cols_deduped_arms_sweep():
    runner = (LB / "experiments" / "d42_litfeat" / "runner.py").read_text()
    assert "design_cols" in runner
    assert "dict.fromkeys" in runner  # arms() sweep guard; dedupe
    assert "duplicate design columns" in runner
