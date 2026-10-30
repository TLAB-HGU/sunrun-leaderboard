"""D60 asymloss contract: fixed >=500 asymmetric penalty, E0003-identical params/rows
except infra (threads+GPU), no forbidden inputs, scratch separation.
Fast: no data reads, no fits."""

import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d60_asymloss"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import asymloss as AL  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_weight_rule_fixed_and_causal():
    assert AL.HIGH_THRESH_KMS == 500.0
    assert AL.W_BASE == 1.0 and AL.W_HIGH == 4.0
    assert AL.check_weights()["formula"] == AL.WEIGHT_FORMULA
    assert "500" in AL.WEIGHT_FORMULA and "4.0" in AL.WEIGHT_FORMULA
    assert "y>=500" in AL.WEIGHT_FORMULA.replace(" ", "")
    assert "no fitted choice" in AL.SELECT_RULE.lower()
    assert "own label" in AL.SELECT_RULE.lower()
    assert "X0008" in AL.SELECT_RULE
    assert "single loss" in AL.SELECT_RULE.lower() or "loss-axis" in AL.SELECT_RULE.lower()
    assert "selection" in AL.SELECT_RULE  # explicitly states no selection stats
    import numpy as np
    w = AL.weights_for_y(np.array([499.9, 500.0, 800.0, 100.0]))
    assert list(w) == [1.0, 4.0, 4.0, 1.0]


def test_fit_params_match_e0003_recipe_infra_only():
    p = AL.fit_params("cuda:1")
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["tree_method"] == "hist"
    assert str(p["device"]).startswith("cuda")
    assert p["n_jobs"] == AL.FIT_NJOBS == 2
    assert AL.PERIOD_JOBS * AL.FIT_NJOBS <= 10  # 3x2=6, three runners sum 18<=20


def test_devices_round_robin_visible_gpus(monkeypatch):
    monkeypatch.delenv("EXPOS_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert AL.device_for(0) == "cuda"
    monkeypatch.setenv("EXPOS_GPUS", "0,1")
    assert AL.device_for(0) == "cuda:0"
    assert AL.device_for(1) == "cuda:1"
    assert AL.device_for(2) == "cuda:0"


def test_full_rule_prespecified_and_no_fitted_choice():
    assert "full 3h" in AL.FULL_RULE and "E0001" in AL.FULL_RULE
    assert "pooled F1" in AL.FULL_RULE or "F1" in AL.FULL_RULE
    assert "no full run" in AL.FULL_RULE
    assert "0.41121495" in AL.FULL_RULE  # adoption bar cited, report-only
    assert "freeze" in AL.FULL_RULE.lower()
    assert "official scoring" in AL.FULL_RULE.lower()


def test_owned_files_and_no_vendored_strips():
    pkg = LB / "experiments" / "d60_asymloss"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 4, files
    assert {"asymloss.py", "runner.py"} <= set(files)
    runner = (pkg / "runner.py").read_text()
    assert "import strips" in runner or "from strips" in runner or "import ST" in runner
    assert "import asymloss" in runner or "import AL" in runner
    assert "store/scratch/worker-d60" in runner
    for other in ("worker-d59", "worker-d61", "worker-d35", "worker-d36",
                  "worker-d37", "worker-d39", "worker-d33", "worker-d34"):
        assert other not in runner
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"


def test_runner_writes_three_predictions_and_real_poison():
    runner = (LB / "experiments" / "d60_asymloss" / "runner.py").read_text()
    assert "for i, n in enumerate(names)" in runner or "for n in PERIODS" in runner
    assert "poison.json" in runner and '"passed"' in runner
    assert "ST.poison_test" in runner
    assert "sample_weight" in runner
    assert "weights_for_frame" in runner or "weights_for_y" in runner
    assert "ThreadPoolExecutor" in runner
    assert "fcache_key" in runner and "featcache" in runner
    assert "device" in runner and "cuda" in runner
    assert "SystemExit" in runner  # fail-closed on poison/non-finite
