"""D73 crossover contract: E0132 level-mean head x E0060 F1-first joint branch.

Hard switch h=24 delegates to hsplit; E0003-identical params/rows + cuda;
3-seed common verdict; winner from dev-OOF only; no blending; no forbidden
inputs; scratch separation; no vendored mechanism code.
Fast: no data reads, no fits."""

import sys
from pathlib import Path

import numpy as np

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d73_msef1x"))
sys.path.insert(0, str(LB / "experiments" / "d33_hsplit"))
sys.path.insert(0, str(LB / "experiments" / "d21_f1joint"))
sys.path.insert(0, str(LB / "experiments" / "d1_longitude_strips"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import hsplit as HS  # noqa: E402
import joint as JT  # noqa: E402
import f1first as F1  # noqa: E402
import strips as ST  # noqa: E402
import info_ablation as ia  # noqa: E402
import xover as XO  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W", "def build_state",
             "KEY_SETS", "ANALOG_COLS", "SHORT_BANDS", "LONG_BANDS"]


def test_split_delegates_to_hsplit_and_tiles_72():
    assert XO.SPLIT_H == HS.SPLIT_H == 24
    assert XO.check_split() == HS.check_split() == {
        "split_h": 24, "short": ["1-6", "7-24"], "long": ["25-48", "49-72"]}
    for h in range(1, 25):
        assert XO.side_of(h) == HS.side_of(h) == "short"
    for h in range(25, 73):
        assert XO.side_of(h) == HS.side_of(h) == "long"
    for bad in (0, 73):
        try:
            XO.side_of(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"h={bad} must raise")
    assert len(ST.strip_columns()) == 17


def test_three_seed_common_verdict_and_fixed_event_seed():
    assert XO.check_seeds((0, 1, 2)) == (0, 1, 2)
    assert XO.N_LEVEL == 3
    assert XO.check_event_seed(1) == 1
    for bad in ((0, 1), (0, 1, 2, 3), (1, 2, 3)):
        try:
            XO.check_seeds(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"seeds={bad} must raise")
    try:
        XO.check_event_seed(7)
    except ValueError:
        pass
    else:
        raise AssertionError("event seed 7 must raise")
    rng = np.random.default_rng(0)
    vecs = {s: rng.normal(size=24) for s in (0, 1, 2)}
    mean = XO.mean_predictions(vecs)
    assert np.allclose(mean, sum(vecs.values()) / 3.0)
    try:
        XO.mean_predictions({0: vecs[0], 1: vecs[1]})
    except ValueError:
        pass
    else:
        raise AssertionError("2-seed mean must raise")


def test_fit_params_match_e0003_recipe_infra_only():
    p = XO.fit_params()
    assert p["device"] == "cuda"
    assert p["tree_method"] == "hist"
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["n_jobs"] == XO.FIT_NJOBS == 4
    assert XO.FIT_JOBS * XO.FIT_NJOBS <= 20
    assert XO.seed_params(0)["random_state"] == 0
    assert XO.seed_params(1)["random_state"] == 1


def test_e0060_branch_rule_imported_not_vendored():
    assert len(F1.GRID) == 6
    assert F1.F1_FLOOR == 0.41121495
    assert "0.41121495" in F1.SELECT_RULE
    assert JT.extra_columns() == ST.strip_columns() + ["analog_best", "analog_mean3",
                                                       "analog_dist", "analog_spread"]
    assert "E0001" in XO.FULL_RULE and "E0003" in XO.FULL_RULE
    assert "full 3h" in XO.FULL_RULE
    assert "F1" in XO.FULL_RULE and "0.41121495" in XO.FULL_RULE
    assert "freeze" in XO.FULL_RULE.lower() or "official-score" in XO.FULL_RULE
    assert "no weight" in XO.SELECT_RULE.lower() and "no blending" in XO.SELECT_RULE.lower()
    assert "dev" in XO.SELECT_RULE.lower()


def test_owned_files_and_no_vendored_mechanism():
    pkg = LB / "experiments" / "d73_msef1x"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 5, files
    assert {"xover.py", "runner.py"} <= set(files)
    xsrc = (pkg / "xover.py").read_text()
    assert "import hsplit" in xsrc or "import HS" in xsrc
    assert "E0132" in xsrc and "E0060" in xsrc
    runner = (pkg / "runner.py").read_text()
    for mod in ("import strips", "import joint", "import f1first",
                "import hsplit", "import xover"):
        assert mod in runner or mod.replace("import ", "import ") in runner, mod
    assert "import analog" not in runner
    assert "store/scratch/worker-d73" in runner
    for other in ("worker-d21", "worker-d28", "worker-d30", "worker-d22", "worker-d23",
                  "worker-d32", "worker-d31", "worker-d34", "worker-d33", "worker-d66",
                  "worker-d69", "worker-d70"):
        assert other not in runner
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"
        assert "predictions_dev.parquet" not in text.replace("predictions_{dev", "")


def test_runner_hard_switch_no_blending_and_gpu_parallel():
    runner = (LB / "experiments" / "d73_msef1x" / "runner.py").read_text()
    assert "SPLIT_H" in runner and "hard switch" in runner.lower()
    assert "device=cuda" in runner
    assert "ThreadPoolExecutor" in runner
    assert "mean_predictions" in runner
    assert "select_config_f1first" in runner
    assert "no blending" in runner.lower() and "no weight" in runner.lower()
    assert "blend(" not in runner.lower() and "weighted" not in runner.lower()
    assert "sample_weight" not in runner and "alpha" not in runner.lower()


def test_runner_writes_three_predictions_and_real_poison():
    runner = (LB / "experiments" / "d73_msef1x" / "runner.py").read_text()
    assert "for n in PERIODS" in runner
    assert "predictions_{dev,selection,audit}" in runner or "for n in PERIODS" in runner
    assert "poison.json" in runner and '"passed"' in runner
    assert "ST.poison_test" in runner
    assert "analog_poison_check" in runner
    assert "SystemExit" in runner  # fail-closed on poison/non-finite
