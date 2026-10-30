"""D36 bandweight contract: fixed pre-dev band weights, single cuda head,
no fitted choice, no forbidden inputs, scratch separation. Fast: no data reads, no fits."""

import math
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d36_bandweight"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import bandweight as BW  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_bands_tile_72_and_weights_fixed_sqrt_rule():
    assert BW.BANDS == ((1, 6), (7, 24), (25, 48), (49, 72))
    assert list(BW.BAND_NAMES) == ["1-6", "7-24", "25-48", "49-72"]
    assert BW.check_partition() == {"bands": ["1-6", "7-24", "25-48", "49-72"], "n_horizons": 72}
    assert set(BW.EVENT_RATES) == set(BW.BAND_NAMES) == set(BW.BAND_WEIGHTS)
    mean_sqrt = sum(math.sqrt(r) for r in BW.EVENT_RATES.values()) / 4
    for b in BW.BAND_NAMES:
        assert BW.BAND_WEIGHTS[b] == math.sqrt(BW.EVENT_RATES[b]) / mean_sqrt
    assert BW.weights_mean_one()["mean"] == sum(BW.BAND_WEIGHTS.values()) / 4
    assert abs(sum(BW.BAND_WEIGHTS.values()) / 4 - 1.0) < 1e-9
    for h in range(1, 73):
        assert BW.weight_of(h) == BW.BAND_WEIGHTS[BW.band_of(h)] > 0
    for bad in (0, 73):
        try:
            BW.weight_of(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"h={bad} must raise")


def test_fit_params_match_e0003_recipe_cuda_infra_only():
    p = BW.fit_params()
    assert p["device"] == "cuda"
    assert p["tree_method"] == "hist"
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["n_jobs"] == BW.FIT_NJOBS == 6
    assert BW.FIT_JOBS * BW.FIT_NJOBS <= 20


def test_full_rule_prespecified_and_no_fitted_choice():
    assert "full 3h" in BW.FULL_RULE and "E0001" in BW.FULL_RULE
    assert "E0003" in BW.FULL_RULE
    assert "pooled F1" in BW.FULL_RULE or "F1" in BW.FULL_RULE
    assert "0.41121495" in BW.FULL_RULE
    assert "no freeze" in BW.FULL_RULE.lower()
    assert "no official-score" in BW.FULL_RULE.lower()
    assert "no fitted choice" in BW.SELECT_RULE.lower()
    assert "dev이전" in BW.SELECT_RULE
    for token in ("selection/audit", "never consumed"):
        assert token in BW.SELECT_RULE


def test_owned_files_and_no_vendored_strips():
    pkg = LB / "experiments" / "d36_bandweight"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 4, files
    assert {"bandweight.py", "runner.py"} <= set(files)
    runner = (pkg / "runner.py").read_text()
    assert "import strips" in runner or "import ST" in runner
    assert "import bandweight" in runner or "import BW" in runner
    assert "store/scratch/worker-d36" in runner
    for other in ("worker-d21", "worker-d28", "worker-d30", "worker-d22", "worker-d23", "worker-d32",
                  "worker-d31", "worker-d33", "worker-d34", "worker-d35"):
        assert other not in runner
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"
        assert "predictions_dev.parquet" not in text.replace("predictions_{dev", "")


def test_runner_single_bandweighted_head_gpu_parallel():
    runner = (LB / "experiments" / "d36_bandweight" / "runner.py").read_text()
    assert "sample_weight" in runner and "weight_of" in runner
    assert runner.count("XGBRegressor(**params).fit") == 1  # single head, no experts
    assert '"device": "cuda"' in runner or "'device': 'cuda'" in runner or "device=cuda" in runner
    assert "ThreadPoolExecutor" in runner
    assert "blend(" not in runner.lower()
    assert "sample_weight=w" in runner.replace(" ", "")


def test_runner_writes_three_predictions_and_real_poison():
    runner = (LB / "experiments" / "d36_bandweight" / "runner.py").read_text()
    assert "for n in PERIODS" in runner
    assert "predictions_{dev,selection,audit}" in runner or "for n in PERIODS" in runner
    assert "poison.json" in runner and '"passed"' in runner
    assert "ST.poison_test" in runner
    assert "SystemExit" in runner  # fail-closed on poison/non-finite
