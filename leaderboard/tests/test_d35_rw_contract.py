"""D35 riseweight contract: fixed causal weight, E0003-identical params/rows
except infra (threads+GPU), no forbidden inputs, scratch separation.
Fast: no data reads, no fits."""

import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d35_riseweight"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import riseweight as RW  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]


def test_weight_rule_fixed_and_causal():
    assert RW.RISE_WINDOW_H == 24
    assert RW.RISE_THRESH_KMS == 80.0
    assert RW.W_BASE == 1.0 and RW.W_RISE == 5.0
    assert RW.check_weights()["formula"] == RW.WEIGHT_FORMULA
    assert "80" in RW.WEIGHT_FORMULA and "5.0" in RW.WEIGHT_FORMULA
    assert "no fitted choice" in RW.SELECT_RULE.lower()
    assert "observations <= O" in RW.SELECT_RULE
    assert "selection" in RW.SELECT_RULE  # explicitly states no selection stats


def test_fit_params_match_e0003_recipe_infra_only():
    p = RW.fit_params("cuda:1")
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["tree_method"] == "hist"
    assert str(p["device"]).startswith("cuda")
    assert p["n_jobs"] == RW.FIT_NJOBS == 3
    assert RW.PERIOD_JOBS * RW.FIT_NJOBS <= 10  # twins concurrent <= 20


def test_devices_round_robin_visible_gpus(monkeypatch):
    monkeypatch.delenv("EXPOS_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert RW.device_for(0) == "cuda"
    monkeypatch.setenv("EXPOS_GPUS", "0,1")
    assert RW.device_for(0) == "cuda:0"
    assert RW.device_for(1) == "cuda:1"
    assert RW.device_for(2) == "cuda:0"


def test_full_rule_prespecified_and_no_fitted_choice():
    assert "full 3h" in RW.FULL_RULE and "E0001" in RW.FULL_RULE
    assert "pooled F1" in RW.FULL_RULE or "F1" in RW.FULL_RULE
    assert "no full run" in RW.FULL_RULE
    assert "0.41121495" in RW.FULL_RULE  # adoption bar cited, report-only
    assert "freeze" in RW.FULL_RULE.lower()


def test_owned_files_and_no_vendored_strips():
    pkg = LB / "experiments" / "d35_riseweight"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 4, files
    assert {"riseweight.py", "runner.py"} <= set(files)
    runner = (pkg / "runner.py").read_text()
    assert "import strips" in runner or "from strips" in runner or "import ST" in runner
    assert "import riseweight" in runner or "import RW" in runner
    assert "store/scratch/worker-d35" in runner
    for other in ("worker-d36", "worker-d33", "worker-d34", "worker-d21", "worker-d28", "worker-d22", "worker-d23"):
        assert other not in runner
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()


def test_runner_writes_three_predictions_and_real_poison():
    runner = (LB / "experiments" / "d35_riseweight" / "runner.py").read_text()
    assert "for i, n in enumerate(names)" in runner or "for n in PERIODS" in runner
    assert "poison.json" in runner and '"passed"' in runner
    assert "ST.poison_test" in runner
    assert "sample_weight" in runner
    assert "weights_for_frame" in runner
    assert "device" in runner and "cuda" in runner
    assert "SystemExit" in runner  # fail-closed on poison/non-finite
