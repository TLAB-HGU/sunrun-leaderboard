"""D41 distribution-mapping contract: train-OOF maps, dev-OOF min-alpha rule,
E0003-identical params/rows + cuda, no forbidden inputs, scratch separation.
Fast: no data reads, no fits (numpy-only synthetic checks)."""

import sys
from pathlib import Path

import numpy as np

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d41_distmap"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import distmap as DM  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_bands_tile_72_and_rule_constants():
    assert DM.BANDS == ((1, 6), (7, 24), (25, 48), (49, 72))
    assert tuple(DM.BAND_NAMES) == ("1-6", "7-24", "25-48", "49-72")
    assert tuple(DM.ALPHAS) == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert DM.F1_FLOOR == 0.41121495
    assert DM.OOF_FOLDS == 3 and DM.DEV_SPLIT == 0.7 and DM.PURGE_H == 72
    assert DM.check_distmap()["f1_floor"] == 0.41121495
    for h in range(1, 73):
        assert DM.band_of(h) in DM.BAND_NAMES
    for bad in (0, 73):
        try:
            DM.band_of(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"h={bad} must raise")


def test_fit_params_match_e0003_recipe_infra_only():
    p = DM.fit_params("cuda")
    assert p["device"] == "cuda"
    assert p["tree_method"] == "hist"
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["n_jobs"] == DM.FIT_NJOBS == 6
    assert DM.FIT_JOBS * DM.FIT_NJOBS <= 20


def test_rules_prespecified_train_oof_dev_oof_and_sources():
    assert "train-OOF" in DM.SELECT_RULE and "dev-OOF" in DM.SELECT_RULE
    assert "0.41121495" in DM.SELECT_RULE
    assert "selection/audit" in DM.SELECT_RULE
    assert "X0085" in DM.SELECT_RULE and "2410.05068" in DM.SELECT_RULE
    assert "full 3h" in DM.FULL_RULE and "E0001" in DM.FULL_RULE
    assert "0.41121495" in DM.FULL_RULE
    assert "freeze" in DM.FULL_RULE.lower() and "official" in DM.FULL_RULE.lower()
    assert "no full run" in DM.FULL_RULE
    assert "2410.05068" in DM.LITERATURE and DM.EXPERIENCE == "X0085"


def _synthetic_oof(seed=0):
    rng = np.random.default_rng(seed)
    n = 3000
    h = rng.integers(1, 73, size=n)
    obs = 400.0 + 120.0 * rng.standard_normal(n)  # wide observed distribution
    pred = 400.0 + 40.0 * rng.standard_normal(n)  # shrunk MSE-style predictions
    return h, pred, obs


def test_quantile_map_widens_shrunk_predictions_monotonic():
    h, pred, obs = _synthetic_oof()
    mapping = DM.fit_mapping(h, pred, obs)
    assert set(mapping) == set(DM.BAND_NAMES)
    base = np.linspace(320, 480, 9)
    hb = np.full_like(base, 12, dtype=int)
    mapped = DM.apply_mapping(base, hb, mapping)
    assert np.isfinite(mapped).all()
    assert float(np.diff(mapped).min()) >= 0, "map must be monotonic non-decreasing"
    assert float(np.std(mapped)) > float(np.std(base)) * 0.5


def test_blend_endpoints_and_select_alpha_min_rule():
    base = np.array([350.0, 500.0, 650.0])
    mapped = np.array([300.0, 520.0, 700.0])
    assert np.allclose(DM.blend(base, mapped, 0.0), base)
    assert np.allclose(DM.blend(base, mapped, 1.0), mapped)
    assert np.allclose(DM.blend(base, mapped, 0.5), 0.5 * base + 0.5 * mapped)
    assert DM.select_alpha({0.0: 0.30, 0.25: 0.40, 0.5: 0.42, 0.75: 0.43, 1.0: 0.44}) == 0.5
    assert DM.select_alpha({0.0: 0.50, 0.25: 0.50, 0.5: 0.50, 0.75: 0.50, 1.0: 0.50}) == 0.0
    assert DM.select_alpha({0.0: 0.30, 0.25: 0.35, 0.5: 0.40, 0.75: 0.41, 1.0: 0.41}) == 0.0
    assert DM.select_alpha({a: None for a in DM.ALPHAS}) == 0.0


def test_owned_files_and_no_vendored_strips():
    pkg = LB / "experiments" / "d41_distmap"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 5, files
    assert {"distmap.py", "runner.py"} <= set(files)
    dsrc = (pkg / "distmap.py").read_text()
    assert "import info_ablation" in dsrc
    runner = (pkg / "runner.py").read_text()
    assert "import strips" in runner or "import ST" in runner
    assert "import distmap" in runner or "import DM" in runner
    assert "store/scratch/worker-d41" in runner
    for other in ("worker-d21", "worker-d28", "worker-d30", "worker-d22", "worker-d23",
                  "worker-d32", "worker-d33", "worker-d34", "worker-d39", "worker-d40",
                  "worker-d42"):
        assert other not in runner
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"


def test_runner_oof_alpha_poison_and_gpu_parallel():
    runner = (LB / "experiments" / "d41_distmap" / "runner.py").read_text()
    assert "train_oof_mapping" in runner and "dev_oof_alpha" in runner
    assert "unique_event_report" in runner
    assert "select_alpha" in runner and "apply_mapping" in runner and "blend" in runner
    assert "alpha" in runner.lower()
    assert "sample_weight" not in runner
    assert "ThreadPoolExecutor" in runner
    assert "device" in runner and "cuda" in runner
    assert "ST.poison_test" in runner
    assert '"passed"' in runner
    assert "SystemExit" in runner  # fail-closed on poison/non-finite
    assert "for n in PERIODS" in runner
    assert "ST.build_manifest" in runner
    assert "X0085" in runner or "2410.05068" in runner
