"""D33 hard-split contract: split tiles 1..72, E0003-identical params/rows + cuda,
no blending, no forbidden inputs, scratch separation. Fast: no data reads, no fits."""

import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d33_hsplit"))
sys.path.insert(0, str(LB / "experiments" / "d32_bandexpert"))
sys.path.insert(0, str(LB / "experiments" / "d1_longitude_strips"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import bands as D32B  # noqa: E402  (read-only band definition)
import hsplit as HS  # noqa: E402
import strips as ST  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_split_tiles_72_and_reuses_d32_bands():
    assert HS.BANDS == D32B.BANDS == ((1, 6), (7, 24), (25, 48), (49, 72))
    assert list(HS.BAND_NAMES) == ["1-6", "7-24", "25-48", "49-72"]
    assert HS.SPLIT_H == 24
    assert HS.check_split() == {"split_h": 24, "short": ["1-6", "7-24"], "long": ["25-48", "49-72"]}
    for h in range(1, 25):
        assert HS.side_of(h) == "short"
    for h in range(25, 73):
        assert HS.side_of(h) == "long"
    for bad in (0, 73):
        try:
            HS.side_of(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"h={bad} must raise")
    assert len(ST.strip_columns()) == 17


def test_fit_params_match_e0003_recipe_infra_only():
    p = HS.fit_params()
    assert p["device"] == "cuda"
    assert p["tree_method"] == "hist"
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["n_jobs"] == HS.FIT_NJOBS == 6
    assert HS.FIT_JOBS * HS.FIT_NJOBS <= 20


def test_full_rule_prespecified_and_no_fitted_choice():
    assert "full 3h" in HS.FULL_RULE and "E0001" in HS.FULL_RULE
    assert "E0003" in HS.FULL_RULE
    assert "pooled F1" in HS.FULL_RULE or "F1" in HS.FULL_RULE
    assert "0.41121495" in HS.FULL_RULE
    assert "freeze" in HS.FULL_RULE.lower() or "official-score" in HS.FULL_RULE
    assert "no fitted choice" in HS.SELECT_RULE.lower()
    for token in ("no weight", "no blending"):
        assert token in HS.SELECT_RULE.lower()


def test_owned_files_and_no_vendored_strips():
    pkg = LB / "experiments" / "d33_hsplit"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 4, files
    assert {"hsplit.py", "runner.py"} <= set(files)
    hsrc = (pkg / "hsplit.py").read_text()
    assert "import bands" in hsrc or "from bands" in hsrc
    runner = (pkg / "runner.py").read_text()
    assert "import strips" in runner or "import ST" in runner
    assert "import hsplit" in runner or "import HS" in runner
    assert "store/scratch/worker-d33" in runner
    for other in ("worker-d21", "worker-d28", "worker-d30", "worker-d22", "worker-d23", "worker-d32",
                  "worker-d31", "worker-d34"):
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
    runner = (LB / "experiments" / "d33_hsplit" / "runner.py").read_text()
    assert "SPLIT_H" in runner and "hard switch" in runner.lower()
    assert '"device": "cuda"' in runner or "'device': 'cuda'" in runner or '"device":"cuda"' in runner \
        or 'device": "cuda"' in runner or "device=cuda" in runner
    assert "ThreadPoolExecutor" in runner
    assert "no blending" in runner.lower() and "no weight" in runner.lower()
    assert "blend(" not in runner.lower() and "weighted" not in runner.lower()
    # predictions are assigned per-side, never averaged: exactly three side masks
    assert runner.count("m_long") >= 3
    assert "sample_weight" not in runner and "alpha" not in runner.lower()


def test_runner_writes_three_predictions_and_real_poison():
    runner = (LB / "experiments" / "d33_hsplit" / "runner.py").read_text()
    assert "for n in PERIODS" in runner
    assert "predictions_{dev,selection,audit}" in runner or "for n in PERIODS" in runner
    assert "poison.json" in runner and '"passed"' in runner
    assert "ST.poison_test" in runner
    assert "SystemExit" in runner  # fail-closed on poison/non-finite
