"""D72 per-lead isotonic contract: train-OOF maps, dev-OOF on/off gate,
E0003-identical params/rows + cuda, no forbidden inputs, scratch separation.
Fast: no data reads, no fits on real data (numpy-only synthetic checks)."""

import sys
from pathlib import Path

import numpy as np

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d72_isotonic"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import iso as ISO  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_leads_tile_72_and_rule_constants():
    assert tuple(ISO.LEADS) == tuple(range(1, 73))
    assert ISO.OOF_FOLDS == 3 and ISO.DEV_SPLIT == 0.7 and ISO.PURGE_H == 72
    assert ISO.MIN_OOF_PER_LEAD == 100
    assert ISO.F1_FLOOR == 0.41121495
    assert ISO.check_iso()["leads"] == 72
    for h in range(1, 73):
        assert ISO.lead_of(h) == h
    for bad in (0, 73):
        try:
            ISO.lead_of(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"h={bad} must raise")


def test_fit_params_match_e0003_recipe_infra_only():
    p = ISO.fit_params("cuda")
    assert p["device"] == "cuda"
    assert p["tree_method"] == "hist"
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["n_jobs"] == ISO.FIT_NJOBS == 4
    assert ISO.FIT_JOBS * ISO.FIT_NJOBS <= 20


def test_rules_prespecified_train_oof_dev_oof_and_sources():
    assert "train-OOF" in ISO.SELECT_RULE and "dev-OOF" in ISO.SELECT_RULE
    assert "selection/audit" in ISO.SELECT_RULE
    assert "post=isotonic" in ISO.SELECT_RULE
    assert "E0097" in ISO.SELECT_RULE and "X0091" in ISO.SELECT_RULE
    assert "full 3h" in ISO.FULL_RULE and "E0001" in ISO.FULL_RULE
    assert "0.41121495" in ISO.FULL_RULE
    assert "freeze" in ISO.FULL_RULE.lower() and "official-score" in ISO.FULL_RULE
    assert "no full run" in ISO.FULL_RULE
    assert "Zadrozny" in ISO.LITERATURE and ISO.EXPERIENCE == "X0091"


def _synthetic_oof(seed=0, n=4000, lead=12):
    rng = np.random.default_rng(seed)
    h = np.full(n, lead, dtype=int)
    obs = 400.0 + 120.0 * rng.standard_normal(n)  # wide observed distribution
    pred = 400.0 + 0.4 * (obs - 400.0) + 15.0 * rng.standard_normal(n)  # shrunk base
    return h, pred, obs


def test_isotonic_map_monotonic_and_clip():
    h, pred, obs = _synthetic_oof()
    mapping = ISO.fit_mapping(h, pred, obs)
    assert len(mapping[12]["x"]) > 0, "dense lead must map"
    base = np.linspace(280, 520, 13)
    hb = np.full_like(base, 12, dtype=int)
    mapped = ISO.apply_mapping(base, hb, mapping)
    assert np.isfinite(mapped).all()
    assert float(np.diff(mapped).min()) >= 0, "map must be monotonic non-decreasing"
    # clip: far-outside base predictions stay at the fitted value range ends
    assert float(mapped[0]) >= float(obs.min()) - 1e-6
    assert float(mapped[-1]) <= float(obs.max()) + 1e-6
    # determinism: refit-free re-application is bitwise identical
    assert np.array_equal(mapped, ISO.apply_mapping(base, hb, mapping))


def test_identity_for_thin_leads():
    rng = np.random.default_rng(1)
    n = 30  # below MIN_OOF_PER_LEAD on lead 5, dense on lead 12
    h = np.array([5] * n + [12] * 4000, dtype=int)
    pred = 400.0 + 40.0 * rng.standard_normal(len(h))
    obs = 400.0 + 120.0 * rng.standard_normal(len(h))
    mapping = ISO.fit_mapping(h, pred, obs)
    assert len(mapping[5]["x"]) == 0, "thin lead must stay identity"
    base = np.array([350.0, 500.0, 650.0])
    out = ISO.apply_mapping(base, np.array([5, 5, 5]), mapping)
    assert np.allclose(out, base), "identity lead must pass base through bitwise"


def test_select_apply_gate():
    assert ISO.select_apply(100.0, 99.0, 0.40, 0.41) is True
    assert ISO.select_apply(100.0, 99.0, 0.40, 0.40) is True  # equality is no-regression
    assert ISO.select_apply(100.0, 101.0, 0.40, 0.45) is False  # MSE regresses
    assert ISO.select_apply(100.0, 99.0, 0.40, 0.39) is False  # F1 regresses
    assert ISO.select_apply(100.0, 99.0, None, 0.41) is False  # undefined fails closed
    assert ISO.select_apply(100.0, 99.0, 0.40, None) is False


def test_owned_files_and_no_vendored_strips():
    pkg = LB / "experiments" / "d72_isotonic"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 4, files
    assert {"iso.py", "runner.py"} <= set(files)
    isrc = (pkg / "iso.py").read_text()
    assert "import info_ablation" in isrc
    runner = (pkg / "runner.py").read_text()
    assert "import strips" in runner or "import ST" in runner
    assert "import iso" in runner or "import ISO" in runner
    assert "store/scratch/worker-d72" in runner
    for other in ("worker-d41", "worker-d21", "worker-d28", "worker-d30", "worker-d22", "worker-d23",
                  "worker-d32", "worker-d33", "worker-d34", "worker-d39", "worker-d40",
                  "worker-d42", "worker-d71", "worker-d73", "worker-d74"):
        assert other not in runner
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"


def test_runner_oof_gate_poison_and_gpu_parallel():
    runner = (LB / "experiments" / "d72_isotonic" / "runner.py").read_text()
    assert "train_oof_iso" in runner and "dev_oof_gate" in runner
    assert "unique_event_report" in runner
    assert "select_apply" in runner and "apply_mapping" in runner and "fit_mapping" in runner
    assert "sample_weight" not in runner
    assert "ThreadPoolExecutor" in runner
    assert "device" in runner and "cuda" in runner
    assert "ST.poison_test" in runner
    assert '"passed"' in runner
    assert "SystemExit" in runner  # fail-closed on poison/non-finite
    assert "for n in PERIODS" in runner
    assert "ST.build_manifest" in runner
    assert "E0097" in runner or "X0091" in runner
    assert "EXPOS_SMOKE" in runner
