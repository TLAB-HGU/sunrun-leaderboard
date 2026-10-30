"""D34 mirror contract: hard split tiles 1..72, E0003-identical params/rows
except infra (threads+GPU), no forbidden inputs, scratch separation.
Fast: no data reads, no fits."""

import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d34_mirror"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import hybrid as HY  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]


def test_split_tiles_72_exactly_once():
    assert HY.SHORT_MAX == 24
    assert HY.LONG_BANDS == ((25, 48), (49, 72))
    assert list(HY.LONG_NAMES) == ["25-48", "49-72"]
    assert HY.check_split() == {"short": "1-24", "long": ["25-48", "49-72"], "n_horizons": 72}


def test_fit_params_match_e0003_recipe_infra_only():
    p = HY.fit_params("cuda:1")
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["tree_method"] == "hist"
    assert str(p["device"]).startswith("cuda")
    assert p["n_jobs"] == HY.FIT_NJOBS == 3
    assert HY.PERIOD_JOBS * HY.FIT_NJOBS <= 10  # twins concurrent <= 20


def test_devices_round_robin_visible_gpus(monkeypatch):
    monkeypatch.delenv("EXPOS_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert HY.device_for(0) == "cuda"
    monkeypatch.setenv("EXPOS_GPUS", "0,1")
    assert HY.device_for(0) == "cuda:0"
    assert HY.device_for(1) == "cuda:1"
    assert HY.device_for(2) == "cuda:0"


def test_full_rule_prespecified_and_no_fitted_choice():
    assert "full 3h" in HY.FULL_RULE and "E0001" in HY.FULL_RULE
    assert "pooled F1" in HY.FULL_RULE or "F1" in HY.FULL_RULE
    assert "25-48" in HY.FULL_RULE and "49-72" in HY.FULL_RULE
    assert "no fitted choice" in HY.SELECT_RULE.lower()
    assert "24/25" in HY.SELECT_RULE


def test_owned_files_and_no_vendored_strips():
    pkg = LB / "experiments" / "d34_mirror"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 4, files
    assert {"hybrid.py", "runner.py"} <= set(files)
    runner = (pkg / "runner.py").read_text()
    assert "import strips" in runner or "from strips" in runner or "import ST" in runner
    assert "import hybrid" in runner or "import HY" in runner
    assert "store/scratch/worker-d34" in runner
    for other in ("worker-d32", "worker-d33", "worker-d21", "worker-d28", "worker-d22", "worker-d23"):
        assert other not in runner
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()


def test_runner_writes_three_predictions_and_real_poison():
    runner = (LB / "experiments" / "d34_mirror" / "runner.py").read_text()
    assert "for n in PERIODS" in runner or "for i, n in enumerate(names)" in runner
    assert "poison.json" in runner and '"passed"' in runner
    assert "ST.poison_test" in runner
    assert "device" in runner and "cuda" in runner
    assert "SystemExit" in runner  # fail-closed on poison/non-finite
