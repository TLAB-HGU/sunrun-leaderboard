"""D32 band-expert contract: bands tile 1..72, E0003-identical params/rows,
no forbidden inputs, scratch separation. Fast: no data reads, no fits."""

import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d32_bandexpert"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import bands as BD  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]


def test_bands_tile_72_exactly_once():
    assert BD.BANDS == ((1, 6), (7, 24), (25, 48), (49, 72))
    assert list(BD.BAND_NAMES) == ["1-6", "7-24", "25-48", "49-72"]
    assert BD.check_partition() == {"bands": ["1-6", "7-24", "25-48", "49-72"], "n_horizons": 72}
    for h in range(1, 73):
        assert BD.band_of(h) in BD.BAND_NAMES
    try:
        BD.band_of(73)
    except ValueError:
        pass
    else:
        raise AssertionError("h=73 must raise")


def test_fit_params_match_e0003_recipe_thread_only():
    p = BD.fit_params()
    assert set(p) == set(ia.PARAMS)
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["tree_method"] == "hist" and "cuda" not in str(p).lower()
    assert p["n_jobs"] == BD.FIT_NJOBS == 6
    assert BD.PERIOD_JOBS * BD.FIT_NJOBS <= 20


def test_full_rule_prespecified_and_no_fitted_choice():
    assert "full 3h" in BD.FULL_RULE and "E0001" in BD.FULL_RULE
    assert "pooled F1" in BD.FULL_RULE or "F1" in BD.FULL_RULE
    assert "no fitted choice" in BD.SELECT_RULE.lower()


def test_owned_files_and_no_vendored_strips():
    pkg = LB / "experiments" / "d32_bandexpert"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 5, files
    assert {"bands.py", "runner.py"} <= set(files)
    runner = (pkg / "runner.py").read_text()
    assert "import strips" in runner or "from strips" in runner or "import ST" in runner
    assert "store/scratch/worker-d32" in runner
    for other in ("worker-d21", "worker-d28", "worker-d30", "worker-d22", "worker-d23"):
        assert other not in runner
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()


def test_runner_writes_three_predictions_and_real_poison():
    runner = (LB / "experiments" / "d32_bandexpert" / "runner.py").read_text()
    assert "predictions_{dev,selection,audit}" in runner or "for n in PERIODS" in runner
    assert "poison.json" in runner and '"passed"' in runner
    assert "ST.poison_test" in runner
    assert "SystemExit" in runner  # fail-closed on poison/non-finite
