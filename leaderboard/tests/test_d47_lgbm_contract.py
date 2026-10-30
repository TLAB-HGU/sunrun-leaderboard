"""D47 leaf-wise twin contract: E0003 twin spec + CPU-fallback engine record,
no blending, no forbidden inputs, scratch separation. Fast: no data reads, no fits."""

import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d47_lgbm"))
sys.path.insert(0, str(LB / "experiments" / "d1_longitude_strips"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import lgbm as DL  # noqa: E402
import strips as ST  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_twin_spec_same_rows_features_purge_and_x0037():
    assert DL.check_twin() == {"leaf_budget": 31, "strip_cols": 17, "seed": 0,
                               "engine_preference": ["lightgbm-gpu", "lightgbm-cpu",
                                                     "xgb-lossguide-cpu-fallback"]}
    assert len(ST.strip_columns()) == 17
    runner = (LB / "experiments" / "d47_lgbm" / "runner.py").read_text()
    assert "existing_all" in runner and "ia.build" in runner and "base_grid" in runner
    assert 'pd.Timedelta("72h")' in runner
    assert "X0037" in (LB / "experiments" / "d47_lgbm" / "lgbm.py").read_text()


def test_engine_fallback_cpu_with_reason_and_leafwise_params():
    engine, reason = DL.detect_engine()
    assert engine == "xgb-lossguide-cpu-fallback" == DL.FALLBACK_ENGINE
    assert "LightGBM absent" in reason and "must not be modified" in reason and "CPU" in reason
    assert reason == DL.FALLBACK_REASON
    p = DL.fit_params()
    assert p["grow_policy"] == "lossguide" and p["max_leaves"] == 31
    assert p["tree_method"] == "hist" and p["device"] == "cpu"
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["n_jobs"] == DL.FIT_NJOBS == 2
    assert DL.FIT_JOBS * DL.FIT_NJOBS <= 20
    lp = DL.lgbm_params()
    assert lp["num_leaves"] == 31 and lp["max_depth"] == 6
    assert lp["learning_rate"] == 0.05 and lp["n_estimators"] == 600


def test_full_rule_prespecified_and_no_fitted_choice_and_e0102_note():
    assert "full 3h" in DL.FULL_RULE and "E0001" in DL.FULL_RULE
    assert "E0003" in DL.FULL_RULE
    assert "F1" in DL.FULL_RULE
    assert "0.41121495" in DL.FULL_RULE
    assert "freeze" in DL.FULL_RULE.lower() or "official-score" in DL.FULL_RULE
    assert "no fitted choice" in DL.SELECT_RULE.lower()
    for token in ("no weight", "no blending", "dev", "OOF"):
        assert token in DL.SELECT_RULE
    assert "E0102" in DL.E0102_NOTE and "seed" in DL.E0102_NOTE.lower()
    assert "submit candidate only" in DL.E0102_NOTE and "no freeze" in DL.E0102_NOTE


def test_owned_files_and_no_vendored_strips():
    pkg = LB / "experiments" / "d47_lgbm"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 4, files
    assert {"lgbm.py", "runner.py"} <= set(files)
    runner = (pkg / "runner.py").read_text()
    assert "import strips" in runner or "import ST" in runner
    assert "import lgbm" in runner or "import DL" in runner
    assert "store/scratch/worker-d47" in runner
    for other in ("worker-d21", "worker-d28", "worker-d30", "worker-d22", "worker-d23", "worker-d32",
                  "worker-d31", "worker-d33", "worker-d34", "worker-d45", "worker-d46"):
        assert other not in runner
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"
        assert "predictions_dev.parquet" not in text.replace("predictions_{dev", "")


def test_runner_single_head_no_blending_and_parallel():
    runner = (LB / "experiments" / "d47_lgbm" / "runner.py").read_text()
    assert "single-head" in runner or "single head" in runner.lower()
    assert "no blending" in runner.lower() and "no weight" in runner.lower()
    assert "blend(" not in runner.lower() and "weighted" not in runner.lower()
    assert "ThreadPoolExecutor" in runner
    assert "sample_weight" not in runner and "alpha" not in runner.lower()


def test_runner_writes_three_predictions_and_real_poison_and_engine():
    runner = (LB / "experiments" / "d47_lgbm" / "runner.py").read_text()
    assert "for n in names" in runner and "list(PERIODS)" in runner
    assert "predictions_{dev,selection,audit}" in runner
    assert "poison.json" in runner and '"passed"' in runner
    assert "ST.poison_test" in runner
    assert "SystemExit" in runner  # fail-closed on poison/non-finite
    assert "engine.json" in runner
