"""D121 boundary-trim contract: dual tags, frozen rule, single data axis, isolation.

Fast static + unit checks only: no data reads, no fits beyond one small
synthetic CPU fit, no held-window contact. The pilot run itself (registered
separately) proves end-to-end behaviour through expos run/evaluate/poison.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d121_btrim"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
sys.path.insert(0, str(LB))

import btspec as SPEC  # noqa: E402
import info_ablation as ia  # noqa: E402
from expos import core  # noqa: E402

RUNNER_SRC = (LB / "experiments" / "d121_btrim" / "runner.py").read_text()
SPEC_SRC = (LB / "experiments" / "d121_btrim" / "btspec.py").read_text()

COPY_BANS = ["def speed_features(", "def ace_features(", "def ch_features(",
             "def enlil_features(", "def build(", "def arms(", "def full_grid(",
             "def write("]
LATER_DATES = ("2025-12-04", "2026-02-25", "2026-03-04", "2026-05-28",
               "2026-05-29", "2026-05-31", "2026-09-26")


def test_hypothesis_dual_tag_and_experience():
    assert SPEC.HYPOTHESIS.startswith("[family=xgb] [axes: data=boundary-trim]")
    assert SPEC.FAMILY_TAG == "[family=xgb]"
    assert SPEC.AXES_TAG.startswith("[axes:") and "data=" in SPEC.AXES_TAG
    assert "boundary-trim" in SPEC.AXES_TAG
    assert SPEC.AXES_QUOTE in SPEC.HYPOTHESIS  # AXES.md data-section quote
    assert "exclusion of bad-image periods" in SPEC.HYPOTHESIS
    assert SPEC.EXPERIENCE == ["X0010", "X0076", "X0159"]
    assert "X0010" in SPEC.HYPOTHESIS and "X0076" in SPEC.HYPOTHESIS
    assert "X0159" in SPEC.HYPOTHESIS
    assert "research/AXES.md" in SPEC.HYPOTHESIS and "data" in SPEC.HYPOTHESIS
    assert SPEC.DIRECTION == "D121" and SPEC.OWNER == "worker-d121"
    assert "label-based" in SPEC.HYPOTHESIS and "distinct" in SPEC.HYPOTHESIS
    assert "6h" in SPEC.HYPOTHESIS  # pilot grid stated
    assert "screening" in SPEC.HYPOTHESIS  # 6h F1 is screening only
    assert "no freeze" in SPEC.HYPOTHESIS.lower()
    assert "held window" in SPEC.HYPOTHESIS


def test_trim_rule_frozen_causal():
    assert set(SPEC.TRIM_H) == {1, 2, 71, 72} and SPEC.KEPT_H == 68
    m = SPEC.keep_mask(np.array([1, 2, 3, 36, 70, 71, 72]))
    assert list(m) == [False, False, True, True, True, False, False]
    full = np.tile(np.arange(1, 73), 4)
    assert abs(SPEC.trim_share(full) - 16 / 288) < 1e-12
    assert abs(SPEC.trim_share(np.arange(1, 73)) - 4 / 72) < 1e-12
    df = pd.DataFrame({"h": [1, 2, 3, 71, 72, 36]})
    kept = df.loc[SPEC.keep_mask(df["h"].to_numpy())]
    assert list(kept["h"]) == [3, 36]
    assert "SPEC.keep_mask" in RUNNER_SRC  # frozen rule, no fitted threshold
    assert "sorted(SPEC.TRIM_H)" in RUNNER_SRC
    assert "72h purge" in SPEC.SELECT_RULE
    assert SPEC.check_spec()["trim"] == [1, 2, 71, 72]
    assert SPEC.check_spec()["kept"] == 68
    # rule source mentions only the horizon index, never labels or jumps
    # rule source describes horizon-index-only filtering, never labels or jumps
    assert "horizon" in SPEC_SRC and "72h purge" in SPEC.SELECT_RULE


def test_single_axis_data_only():
    p = SPEC.fit_params()
    assert p["device"] == "cpu"
    assert p.get("objective", "reg:squarederror") == "reg:squarederror"
    assert "tweedie_variance_power" not in p
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0001 recipe"
    assert p["n_estimators"] == 600 and p["max_depth"] == 6
    assert "base_score" not in p
    assert "sample_weight" not in RUNNER_SRC  # exclusion, not weighting
    assert "EPAM_COLS" not in RUNNER_SRC and "epam" not in RUNNER_SRC.lower()
    assert "blend" not in RUNNER_SRC.lower()
    assert RUNNER_SRC.count(".fit(") == 1, "exactly one head fit (single head, no refit)"
    assert 'ia.arms(df.columns)["existing_all"]' in RUNNER_SRC  # no new columns
    assert "EVENT_THRESHOLD" not in RUNNER_SRC
    assert "SHOCK_JUMP" not in RUNNER_SRC and "shock_jump" not in RUNNER_SRC
    assert "HIWIND_LEVEL" not in RUNNER_SRC and "hiwind" not in RUNNER_SRC.lower()
    assert SPEC.check_spec()["axes"] == "[axes: data=boundary-trim]"


def test_e0001_twin_rows_purge_budget():
    assert SPEC.PURGE_HOURS == 72
    assert SPEC.FREQ_PILOT == "6h" and SPEC.FREQ_FULL == "3h"
    assert SPEC.N_ESTIMATORS == 600 and SPEC.SEED == 0
    assert "SPEC.PURGE_HOURS" in RUNNER_SRC
    assert "build_train" in RUNNER_SRC and "test_grid" in RUNNER_SRC
    assert "design_cols" in RUNNER_SRC


def test_cpu_budget_no_accelerator():
    assert SPEC.PERIOD_JOBS * SPEC.FIT_NJOBS <= 10
    assert SPEC.PERIOD_JOBS == 3, "three periods run in parallel"
    assert SPEC.FIT_NJOBS <= 2
    assert SPEC.fit_params()["device"] == "cpu"
    for pat in ("gpu_lease", "--gpu", "prefer-gpu", "EXPOS_GPUS",
                "CUDA_VISIBLE_DEVICES", "device_for"):
        assert pat not in RUNNER_SRC, pat
        assert pat not in SPEC_SRC, pat
    assert "CPU" in RUNNER_SRC
    assert "D119" in SPEC.CPU_REASON and "D120" in SPEC.CPU_REASON and "D121" in SPEC.CPU_REASON
    assert "gpu0" in SPEC.CPU_REASON and "gpu1" in SPEC.CPU_REASON
    assert "60s" in SPEC.CPU_REASON and "foreground" in SPEC.CPU_REASON
    assert "lease" in SPEC.CPU_REASON


def test_scratch_isolation():
    assert "store/scratch/worker-d121" in RUNNER_SRC
    others = [l for l in RUNNER_SRC.splitlines() if "worker-" in l and "worker-d121" not in l]
    assert others == [], f"scratch leak: {others[:3]}"
    assert "featcache" in RUNNER_SRC and "fcache_key" in RUNNER_SRC
    assert SPEC.SCRATCH_NAME == "worker-d121"


def test_readonly_no_vendored_copies():
    for s in COPY_BANS:
        assert s not in RUNNER_SRC, f"runner must import, not copy: {s!r}"
        assert s not in SPEC_SRC, f"spec must not vendor: {s!r}"
    assert "from existing_all import" in RUNNER_SRC


def test_no_forbidden_sources_or_later_windows():
    for name, src in (("btspec.py", SPEC_SRC), ("runner.py", RUNNER_SRC)):
        for tok in ("datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
                    "naive.parquet", "official-predictions", "baseline-replay",
                    "official_check", "official-check", "predictions_official_truth"):
            assert tok not in src, (name, tok)
        for s in LATER_DATES:
            assert s not in src, (name, f"later window {s}")
    for ban in COPY_BANS:
        assert ban not in RUNNER_SRC, f"runner must import, not copy: {ban}"
    assert core.scan(SPEC_SRC + "\n" + RUNNER_SRC + "\n" + SPEC.HYPOTHESIS) == []


def test_no_foreign_experiment_reads():
    # runner refits fresh from inputs; it must never read another experiment's outputs
    assert "_load_member" not in RUNNER_SRC
    assert "research/expos/runs/E" not in RUNNER_SRC
    assert "RUNS /" not in RUNNER_SRC and "RUNS/" not in RUNNER_SRC
    assert "E0003" not in RUNNER_SRC and "E0102" not in RUNNER_SRC and "E0193" not in RUNNER_SRC


def test_full_rule_preregistered_no_held_claim():
    assert "rc==0" in SPEC.FULL_RULE and "poison" in SPEC.FULL_RULE and "E0001" in SPEC.FULL_RULE
    assert "F1>=0.30" in SPEC.FULL_RULE and "no freeze" in SPEC.HYPOTHESIS.lower()
    assert "6h F1" in SPEC.FULL_RULE  # 6h F1 is screening only
    assert "3h" in SPEC.FULL_RULE  # F1 claimed only at 3h density
    assert "no freeze" in SPEC_SRC.lower()
    assert "official_score(" not in RUNNER_SRC and "core.freeze" not in RUNNER_SRC
    assert "official-score" in SPEC.FULL_RULE
    assert "0.41121495" in SPEC.FULL_RULE and "seed-repro" in SPEC.FULL_RULE


def test_predictions_three_kinds_and_poison():
    assert "predictions_{dev,selection,audit}" in RUNNER_SRC  # output contract
    assert "write(te, pred, name)" in RUNNER_SRC  # per-period full-grid write
    assert "predictions_official" not in RUNNER_SRC
    assert "official" not in RUNNER_SRC.lower()
    assert "freeze" not in RUNNER_SRC.lower()
    assert "test_grid" in RUNNER_SRC and "prediction_invariant" in RUNNER_SRC
    assert "poison_check" in RUNNER_SRC and "positive_control" in RUNNER_SRC
    assert "passed" in RUNNER_SRC and "poison.json" in RUNNER_SRC
    assert "SystemExit" in RUNNER_SRC  # fail-closed on poison/non-finite


def test_parallel_cache_and_budget():
    assert "ThreadPoolExecutor" in RUNNER_SRC
    assert "fcache_key" in RUNNER_SRC and "featcache" in RUNNER_SRC
    assert "SPEC.keep_mask" in RUNNER_SRC
    assert SPEC.PERIOD_JOBS == 3
    assert SPEC.PERIOD_JOBS * SPEC.FIT_NJOBS <= 20
    assert SPEC.FREQ_PILOT == "6h" and SPEC.FREQ_FULL == "3h"


def test_dev_only_selection_shape():
    assert "_dev_mses" not in RUNNER_SRC  # no dev-fit search; rule is frozen
    assert "no selection/audit statistic" in SPEC.SELECT_RULE
    assert "exclusion_source" in SPEC.SELECT_RULE
    assert "no candidate search" in SPEC.SELECT_RULE
    assert "72h purge" in SPEC.SELECT_RULE
    assert "fitted_choice" in RUNNER_SRC
    assert "no full run" in SPEC.FULL_RULE.lower() or "no full run" in SPEC.FULL_RULE


def test_owned_dir_is_small():
    files = sorted(p.name for p in (LB / "experiments" / "d121_btrim").iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"btspec.py", "runner.py"} <= set(files)


def test_trim_unit_and_fit_smoke():
    """Frozen edge trim drops 4/72 horizons and a small CPU head fits on kept rows."""
    import xgboost as xgb

    rng = np.random.default_rng(7)
    h = np.tile(np.arange(1, 73), 4)
    assert SPEC.trim_share(h) == 16 / 288
    X = rng.standard_normal((288, 6))
    y = 400.0 + 30.0 * X[:, 0] + rng.standard_normal(288) * 10.0
    mask = SPEC.keep_mask(h)
    assert mask.sum() == 272 and (~mask).sum() == 16
    assert set(h[~mask]) == {1, 2, 71, 72}
    p = SPEC.fit_params()
    m = xgb.XGBRegressor(**dict(p, n_estimators=20)).fit(X[mask], y[mask])
    pred = np.asarray(m.predict(X[mask]), dtype=float)
    assert np.isfinite(pred).all() and float(np.std(pred)) > 0
    dump = m.get_booster().get_dump(dump_format="text")
    assert sum(t.count("[f") for t in dump) > 0, "trees must split"
