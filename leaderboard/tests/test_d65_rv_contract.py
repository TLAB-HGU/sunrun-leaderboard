"""D65 rolling-origin contract: 6 chronological folds re-evaluate E0102/E0105.

Fast: no data reads, no fits."""

import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d65_rollval"))
sys.path.insert(0, str(LB / "experiments" / "d45_lvlmean"))
sys.path.insert(0, str(LB / "experiments" / "d33_hsplit"))
sys.path.insert(0, str(LB / "experiments" / "d32_bandexpert"))
sys.path.insert(0, str(LB / "experiments" / "d1_longitude_strips"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import bands as D32B  # noqa: E402  (read-only band definition)
import hsplit as HS  # noqa: E402  (read-only switch-rule reference)
import lvlmean as LM  # noqa: E402  (read-only seed-mean reference)
import rollval as RV  # noqa: E402
import strips as ST  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_dual_tag_and_experience_citation():
    assert RV.FAMILY_TAG == "[family=xgb]"
    assert RV.AXES_TAG.startswith("[axes:") and "val=" in RV.AXES_TAG
    assert RV.HYPOTHESIS.startswith("[family=xgb] [axes:")
    for token in ("E0102", "E0105", "X0096", "X0098"):
        assert token in RV.HYPOTHESIS, f"hypothesis must cite {token}"
    assert "freeze" in RV.HYPOTHESIS.lower() or "official-score" in RV.HYPOTHESIS
    assert "6" in RV.HYPOTHESIS


def test_folds_are_six_chronological_inside_periods():
    folds = list(RV.check_folds())
    assert len(folds) == RV.N_FOLDS == 6
    assert [f["name"] for f in folds] == ["F1-dev-A", "F2-dev-B", "F3-sel-A",
                                          "F4-sel-B", "F5-audit-A", "F6-audit-B"]
    import pandas as pd
    prev_hi = None
    for f in folds:
        lo, hi = pd.Timestamp(f["val_lo"], tz="UTC"), pd.Timestamp(f["val_hi"], tz="UTC")
        assert lo < hi
        if prev_hi is not None:
            assert lo > prev_hi
        prev_hi = hi
        assert f["period"] in ("dev", "selection", "audit")
    periods = {f["period"] for f in folds}
    assert periods == {"dev", "selection", "audit"}
    for bad in ([], folds[:5], folds + [folds[0]]):
        try:
            RV.check_folds(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("bad fold count must raise")
    swapped = [folds[1], folds[0]] + folds[2:]
    try:
        RV.check_folds(swapped)
    except ValueError:
        pass
    else:
        raise AssertionError("non-chronological folds must raise")


def test_split_and_seeds_reuse_d33_d45():
    assert RV.SPLIT_H == HS.SPLIT_H == 24
    assert RV.check_split() == HS.check_split()
    assert HS.BANDS == D32B.BANDS == ((1, 6), (7, 24), (25, 48), (49, 72))
    assert tuple(RV.LEVEL_SEEDS) == tuple(LM.LEVEL_SEEDS) == (0, 1, 2)
    assert RV.EVENT_SEED == LM.EVENT_SEED == 1
    assert len(ST.strip_columns()) == 17


def test_fit_params_match_e0003_recipe_infra_only():
    p = RV.fit_params()
    assert p["device"] == "cuda"
    assert p["tree_method"] == "hist"
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["n_jobs"] == RV.FIT_NJOBS == 4
    assert RV.PERIOD_JOBS * RV.FIT_NJOBS <= 20
    assert RV.FOLD_JOBS * RV.FIT_NJOBS <= 20


def test_select_and_full_rules_prespecified():
    assert "no fitted choice" in RV.SELECT_RULE.lower()
    assert "no weight" in RV.SELECT_RULE.lower()
    assert "no blending" in RV.SELECT_RULE.lower()
    assert "no averaging on the event side" in RV.SELECT_RULE.lower()
    assert "dev/OOF only" in RV.SELECT_RULE or "dev/oof" in RV.SELECT_RULE.lower()
    assert "never" in RV.SELECT_RULE.lower() and "selection" in RV.SELECT_RULE.lower()
    assert "full 3h" in RV.FULL_RULE and "E0001" in RV.FULL_RULE
    assert "E0003" in RV.FULL_RULE
    assert "F1" in RV.FULL_RULE
    assert "0.41121495" in RV.FULL_RULE
    assert "freeze" in RV.FULL_RULE.lower() or "official-score" in RV.FULL_RULE
    assert "new id" in RV.FULL_RULE
    assert "E0102" in RV.FULL_RULE and "X0097" in RV.FULL_RULE
    assert "6-fold" in RV.FULL_RULE or "6/6" in RV.FULL_RULE


def test_owned_files_and_no_vendored_strips():
    pkg = LB / "experiments" / "d65_rollval"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 5, files
    assert {"rollval.py", "runner.py"} <= set(files)
    rsrc = (pkg / "rollval.py").read_text()
    assert "import hsplit" in rsrc or "from hsplit" in rsrc
    assert "import lvlmean" in rsrc or "from lvlmean" in rsrc
    assert "SPLIT_H = 24" not in rsrc  # switch rule referenced, never copied
    assert "LEVEL_SEEDS = (0, 1, 2)" not in rsrc  # seeds referenced, never copied
    runner = (pkg / "runner.py").read_text()
    assert "import strips" in runner or "import ST" in runner
    assert "import rollval" in runner or "import RV" in runner
    assert "import hsplit" in runner or "import HS" in runner
    assert "import lvlmean" in runner or "import LM" in runner
    assert "store/scratch/worker-d65" in runner
    assert "worker-d33" not in runner
    for other in ("worker-d21", "worker-d28", "worker-d30", "worker-d22", "worker-d23", "worker-d32",
                  "worker-d31", "worker-d34", "worker-d44", "worker-d45", "worker-d46", "worker-d47",
                  "worker-d63", "worker-d64"):
        assert other not in runner
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"
        assert "predictions_dev.parquet" not in text.replace("predictions_{dev", "")


def test_runner_rolling_heads_no_blending_and_gpu_parallel():
    runner = (LB / "experiments" / "d65_rollval" / "runner.py").read_text()
    assert "SPLIT_H" in runner and "hard switch" in runner.lower()
    assert "mean_predictions" in runner
    assert "EVENT_SEED" in runner
    assert "device=cuda" in runner
    assert "ThreadPoolExecutor" in runner
    assert "FOLD_JOBS" in runner and "PERIOD_JOBS" in runner
    assert "fold_table" in runner
    assert "run_fold" in runner and "run_standard_period" in runner
    assert "no blending" in runner.lower() and "no weight" in runner.lower()
    assert "blend(" not in runner.lower() and "weighted" not in runner.lower()
    assert "sample_weight" not in runner and "alpha" not in runner.lower()
    assert "--seed" not in runner  # fixed preregistered seeds, no CLI override


def test_runner_writes_three_predictions_fold_table_and_real_poison():
    runner = (LB / "experiments" / "d65_rollval" / "runner.py").read_text()
    assert "for n in PERIODS" in runner
    assert "predictions_{dev,selection,audit}" in runner or "for n in PERIODS" in runner
    assert "fold_table.json" in runner
    assert "poison.json" in runner and '"passed"' in runner
    assert "ST.poison_test" in runner
    assert "SystemExit" in runner  # fail-closed on poison/non-finite
