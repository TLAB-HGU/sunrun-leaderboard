"""D45 level-mean contract: reuses D33 switch rules, 3-seed level mean + fixed event.

Fast: no data reads, no fits."""

import sys
from pathlib import Path

import numpy as np

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d45_lvlmean"))
sys.path.insert(0, str(LB / "experiments" / "d33_hsplit"))
sys.path.insert(0, str(LB / "experiments" / "d32_bandexpert"))
sys.path.insert(0, str(LB / "experiments" / "d1_longitude_strips"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import bands as D32B  # noqa: E402  (read-only band definition)
import hsplit as HS  # noqa: E402  (read-only switch-rule reference)
import lvlmean as LM  # noqa: E402
import strips as ST  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_split_reuses_hsplit_at_24():
    assert LM.SPLIT_H == HS.SPLIT_H == 24
    assert LM.check_split() == HS.check_split() == {"split_h": 24, "short": ["1-6", "7-24"],
                                                    "long": ["25-48", "49-72"]}
    assert HS.BANDS == D32B.BANDS == ((1, 6), (7, 24), (25, 48), (49, 72))
    for h in range(1, 25):
        assert LM.side_of(h) == "short"
    for h in range(25, 73):
        assert LM.side_of(h) == "long"
    for bad in (0, 73):
        try:
            LM.side_of(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"h={bad} must raise")
    assert len(ST.strip_columns()) == 17


def test_level_seeds_and_fixed_event_seed():
    assert tuple(LM.LEVEL_SEEDS) == (0, 1, 2)
    assert LM.N_LEVEL == 3
    assert LM.EVENT_SEED == 1
    assert tuple(LM.check_seeds((0, 1, 2))) == (0, 1, 2)
    assert LM.check_event_seed(1) == 1
    for bad in ((0, 1), (0, 1, 2, 3), (10, 11, 12), (1, 2, 0)):
        try:
            LM.check_seeds(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"seeds={bad} must raise")
    try:
        LM.check_event_seed(0)
    except ValueError:
        pass
    else:
        raise AssertionError("event seed 0 must raise")
    try:
        LM.seed_params(7)
    except ValueError:
        pass
    else:
        raise AssertionError("seed 7 must raise")
    assert LM.seed_params(2)["random_state"] == 2
    assert LM.seed_params(1)["random_state"] == 1


def test_mean_is_uniform_thirds_and_fail_closed():
    a = np.array([600.0, 610.0])
    b = np.array([630.0, 610.0])
    c = np.array([660.0, 610.0])
    out = LM.mean_predictions({0: a, 1: b, 2: c})
    assert np.allclose(out, np.array([630.0, 610.0]))
    assert np.allclose(out, (a + b + c) / 3.0)
    for bad in ({0: a, 1: b}, {0: a, 1: b, 2: c[:1]},
                {0: np.array([np.nan, 1.0]), 1: b, 2: c}):
        try:
            LM.mean_predictions(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("bad seed predictions must raise")


def test_fit_params_match_e0003_recipe_infra_only():
    p = LM.fit_params()
    assert p["device"] == "cuda"
    assert p["tree_method"] == "hist"
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["n_jobs"] == LM.FIT_NJOBS == 4
    assert LM.FIT_JOBS * LM.FIT_NJOBS <= 20


def test_full_rule_prespecified_with_repro_and_fragility_note():
    assert "full 3h" in LM.FULL_RULE and "E0001" in LM.FULL_RULE
    assert "E0003" in LM.FULL_RULE
    assert "pooled F1" in LM.FULL_RULE or "F1" in LM.FULL_RULE
    assert "0.41121495" in LM.FULL_RULE
    assert "freeze" in LM.FULL_RULE.lower() or "official-score" in LM.FULL_RULE
    assert "repro" in LM.FULL_RULE.lower() or "repro" in LM.FULL_RULE
    assert "new id" in LM.FULL_RULE
    assert "E0102" in LM.FULL_RULE and "X0097" in LM.FULL_RULE
    assert "no fitted choice" in LM.SELECT_RULE.lower()
    assert "no averaging on the event side" in LM.SELECT_RULE.lower()
    for token in ("no weight", "no blending"):
        assert token in LM.SELECT_RULE.lower()


def test_owned_files_and_no_vendored_strips():
    pkg = LB / "experiments" / "d45_lvlmean"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 5, files
    assert {"lvlmean.py", "runner.py"} <= set(files)
    lsrc = (pkg / "lvlmean.py").read_text()
    assert "import hsplit" in lsrc or "from hsplit" in lsrc
    assert "SPLIT_H = 24" not in lsrc  # switch rule referenced, never copied
    runner = (pkg / "runner.py").read_text()
    assert "import strips" in runner or "import ST" in runner
    assert "import lvlmean" in runner or "import LM" in runner
    assert "import hsplit" in runner or "import HS" in runner
    assert "store/scratch/worker-d45" in runner
    assert "worker-d33" not in runner
    for other in ("worker-d21", "worker-d28", "worker-d30", "worker-d22", "worker-d23", "worker-d32",
                  "worker-d31", "worker-d34", "worker-d44", "worker-d46", "worker-d47"):
        assert other not in runner
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"
        assert "predictions_dev.parquet" not in text.replace("predictions_{dev", "")


def test_runner_level_mean_fixed_event_no_blending_and_gpu_parallel():
    runner = (LB / "experiments" / "d45_lvlmean" / "runner.py").read_text()
    assert "SPLIT_H" in runner and "hard switch" in runner.lower()
    assert "mean_predictions" in runner
    assert "EVENT_SEED" in runner
    assert "device=cuda" in runner
    assert "ThreadPoolExecutor" in runner
    assert "no blending" in runner.lower() and "no weight" in runner.lower()
    assert "blend(" not in runner.lower() and "weighted" not in runner.lower()
    assert "sample_weight" not in runner and "alpha" not in runner.lower()
    assert "--seed" not in runner  # fixed preregistered seeds, no CLI override
    assert "level_spread" in runner or "level-spread" in runner or "spread" in runner


def test_runner_writes_three_predictions_and_real_poison():
    runner = (LB / "experiments" / "d45_lvlmean" / "runner.py").read_text()
    assert "for n in PERIODS" in runner
    assert "predictions_{dev,selection,audit}" in runner or "for n in PERIODS" in runner
    assert "poison.json" in runner and '"passed"' in runner
    assert "ST.poison_test" in runner
    assert "SystemExit" in runner  # fail-closed on poison/non-finite
