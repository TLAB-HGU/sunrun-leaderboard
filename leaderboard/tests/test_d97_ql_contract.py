"""D97 quality-weight contract: hypothesis tags, unused-first, single-axis data, isolation.

Fast static + unit checks only: no data reads, no fits beyond small synthetic
CPU fits, no official contact. The pilot run itself (registered separately)
proves end-to-end behaviour through expos run/evaluate/poison.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d97_quality"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
sys.path.insert(0, str(LB))

import qlspec as QS  # noqa: E402
import info_ablation as ia  # noqa: E402
from expos import core  # noqa: E402

RUNNER_SRC = (LB / "experiments" / "d97_quality" / "runner.py").read_text()
SPEC_SRC = (LB / "experiments" / "d97_quality" / "qlspec.py").read_text()

COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_hypothesis_dual_tag_and_experience():
    assert QS.HYPOTHESIS.startswith("[family=xgb] [axes: data=quality-weight]")
    assert QS.EXPERIENCE == ["X0106", "X0119"]
    assert "X0106" in QS.HYPOTHESIS and "X0119" in QS.HYPOTHESIS
    assert "research/AXES.md" in QS.HYPOTHESIS and "data" in QS.HYPOTHESIS
    assert QS.AXES_QUOTE in QS.HYPOTHESIS  # AXES.md data-section quote
    assert "sample weights by label quality (was_missing)" in QS.HYPOTHESIS
    assert "E0001" in QS.HYPOTHESIS and "E0003" in QS.HYPOTHESIS
    assert QS.DIRECTION == "D97" and QS.OWNER == "worker-d97"
    assert "6h" in QS.HYPOTHESIS  # pilot grid stated
    assert "CPU" in QS.HYPOTHESIS
    assert "Unused" in QS.HYPOTHESIS or "unused" in QS.HYPOTHESIS
    assert "causal" in QS.HYPOTHESIS
    assert "screening" in QS.HYPOTHESIS  # 6h F1 screening only
    assert "3h" in QS.HYPOTHESIS  # F1 claimed only at 3h density
    assert "No freeze" in QS.HYPOTHESIS
    assert "official-score" in QS.HYPOTHESIS


def test_quality_weight_unused_first():
    """was_missing quality weighting must be absent from every prior weight path."""
    ia_src = (LB / "experiments" / "headroom_audit" / "info_ablation.py").read_text()
    assert "sample_weight" not in ia_src
    assert "was_missing" not in ia_src
    head_src = (LB / "experiments" / "headroom_audit" / "headroom.py").read_text()
    assert "sample_weight" not in head_src
    assert QS.W_OBSERVED == 1.0 and QS.W_INTERPOLATED == 0.25
    # frozen rule: observed 1.0, interpolated 0.25, deterministic
    assert list(QS.quality_weights([0, 1, 0, 1])) == [1.0, 0.25, 1.0, 0.25]
    assert list(QS.quality_weights([0, 1, 0])) == list(QS.quality_weights([0, 1, 0]))
    w = QS.quality_weights(np.array([0, 1, 0], dtype=np.int8))
    assert set(np.unique(w)) <= {1.0, 0.25}
    assert QS.check_spec()["w_observed"] == 1.0
    assert QS.check_spec()["w_interpolated"] == 0.25


def test_single_axis_data_only():
    p = QS.fit_params()
    assert p["device"] == "cpu"
    assert p.get("objective", "reg:squarederror") == "reg:squarederror"
    assert "tweedie_variance_power" not in p
    # E0001 recipe untouched except infra (data change is weight-side only)
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0001 recipe"
    assert p["n_estimators"] == 600 and p["max_depth"] == 6
    assert "base_score" not in p, "start point stays the E0001 default (one axis only)"
    # data-axis change is sample_weight with the frozen qw column only
    assert "sample_weight" in RUNNER_SRC
    assert 'tr["qw"]' in RUNNER_SRC or "tr['qw']" in RUNNER_SRC
    assert '"qw" not in cols' in RUNNER_SRC and '"qmiss" not in cols' in RUNNER_SRC
    assert "quality_weights" in RUNNER_SRC
    assert "W_OBSERVED" in RUNNER_SRC or "w_observed" in RUNNER_SRC
    assert type(__import__("xgboost").XGBRegressor(**p)).__name__ == "XGBRegressor"


def test_e0001_twin_rows_purge_budget():
    assert QS.PURGE_HOURS == 72
    assert QS.FREQ_PILOT == "6h" and QS.FREQ_FULL == "3h"
    assert QS.N_ESTIMATORS == 600 and QS.SEED == 0
    assert 'ia.arms(df.columns)["existing_all"]' in RUNNER_SRC
    assert "QS.PURGE_HOURS" in RUNNER_SRC
    assert "build_train_quality" in RUNNER_SRC
    assert "speed_features" in RUNNER_SRC and "enlil_features" in RUNNER_SRC
    assert RUNNER_SRC.count(".fit(") == 1, "exactly one head fit (single head with weights)"
    assert QS.check_spec()["axes"] == "[axes: data=quality-weight]"
    assert QS.check_spec()["experience"] == ["X0106", "X0119"]


def test_cpu_budget_no_accelerator():
    assert QS.PERIOD_JOBS * QS.FIT_NJOBS <= 10
    assert QS.PERIOD_JOBS == 3, "three periods run in parallel"
    assert QS.FIT_NJOBS <= 2
    assert QS.fit_params()["device"] == "cpu"
    blob = RUNNER_SRC + "\n" + SPEC_SRC
    for pat in ("cuda", "gpu_lease", "--gpu", "prefer-gpu", "EXPOS_GPUS",
                "CUDA_VISIBLE_DEVICES", "device_for"):
        assert pat not in blob, pat
    assert "CPU" in RUNNER_SRC  # CPU-only guard documented
    assert "D96" in QS.CPU_REASON and "D98" in QS.CPU_REASON  # lease reason recorded
    assert "D97" in QS.CPU_REASON
    assert "<=20" in QS.CPU_REASON


def test_scratch_isolation():
    assert "store/scratch/worker-d97" in RUNNER_SRC
    others = [l for l in RUNNER_SRC.splitlines() if "worker-" in l and "worker-d97" not in l]
    assert others == [], f"scratch leak: {others[:3]}"
    assert "featcache" in RUNNER_SRC and "fcache_key" in RUNNER_SRC


def test_readonly_admin_no_vendored_strips():
    for s in COPY_BANS:
        assert s not in RUNNER_SRC, f"runner vendors d1-strip code {s!r}"
        assert s not in SPEC_SRC, f"spec vendors d1-strip code {s!r}"
    assert "import strips" not in RUNNER_SRC  # E0001 twin needs no strips


def test_no_forbidden_paths_in_registered_code():
    assert core.scan(RUNNER_SRC + "\n" + SPEC_SRC) == []
    assert core.scan(QS.HYPOTHESIS) == []


def test_full_rule_preregistered_no_freeze_claim():
    assert "rc==0" in QS.FULL_RULE and "poison" in QS.FULL_RULE and "E0001" in QS.FULL_RULE
    assert "F1>=0.30" in QS.FULL_RULE and "No freeze" in QS.HYPOTHESIS
    assert "6h F1" in QS.FULL_RULE  # 6h F1 is screening only
    assert "3h" in QS.FULL_RULE  # F1 claimed only at 3h density
    assert "E0003" in QS.FULL_RULE and "seed-repro" in QS.FULL_RULE
    assert "No freeze" in SPEC_SRC
    assert "official_score(" not in RUNNER_SRC and "core.freeze" not in RUNNER_SRC
    assert "official-score" in QS.FULL_RULE


def test_selection_dev_oof_only():
    assert "dev/OOF" in QS.SELECT_RULE
    assert "selection/audit" in QS.SELECT_RULE
    assert "never read" in QS.SELECT_RULE
    assert "72h" in QS.SELECT_RULE or "72h purge" in QS.SELECT_RULE
    assert "X0106" in QS.SELECT_RULE or "X0119" in QS.SELECT_RULE


def test_predictions_three_kinds_and_poison():
    assert "predictions_{dev,selection,audit}" in RUNNER_SRC  # output contract
    assert "write(te, pred, name)" in RUNNER_SRC  # per-period full-grid write
    assert "predictions_official" not in RUNNER_SRC
    assert "test_grid" in RUNNER_SRC and "prediction_invariant" in RUNNER_SRC
    assert "poison_check" in RUNNER_SRC and "positive_control" in RUNNER_SRC
    assert "passed" in RUNNER_SRC and "poison.json" in RUNNER_SRC
    assert "SystemExit" in RUNNER_SRC  # fail-closed on poison/non-finite


def test_owned_dir_is_small():
    files = sorted(p.name for p in (LB / "experiments" / "d97_quality").iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"qlspec.py", "runner.py"} <= set(files)


def test_quality_weights_causal_frozen():
    """Weight rule is frozen, deterministic, and label-local (no cross-row fits)."""
    a = QS.quality_weights([0, 0, 1, 1, 0])
    b = QS.quality_weights([0, 0, 1, 1, 0])
    assert np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    c = QS.quality_weights([1, 1, 1, 1, 1])
    assert (np.asarray(c) == 0.25).all()
    d = QS.quality_weights([0, 0, 0, 0, 0])
    assert (np.asarray(d) == 1.0).all()
    # NaN flag maps to low weight (row dropped anyway when y is NaN)
    e = QS.quality_weights(np.array([np.nan, 0.0, 1.0]))
    assert list(e) == [0.25, 1.0, 0.25]
    # weights never depend on y values, only on the flag
    assert float(np.asarray(a).sum()) == 1.0 * 3 + 0.25 * 2


def test_weighted_tiny_fit_cpu():
    """Frozen weights feed sample_weight on synthetic data; weighted head fits and splits."""
    import xgboost as xgb

    rng = np.random.default_rng(11)
    n = 800
    X = rng.standard_normal((n, 4))
    y = 400.0 + 40.0 * X[:, 0] - 20.0 * X[:, 1] + 5.0 * rng.standard_normal(n)
    qmiss = (rng.uniform(size=n) < 0.2).astype(np.int8)
    w = QS.quality_weights(qmiss)
    assert set(np.unique(w)) <= {1.0, 0.25}
    assert (w[qmiss == 1] == 0.25).all() and (w[qmiss == 0] == 1.0).all()
    cols = [f"f{i}" for i in range(4)]
    df = pd.DataFrame(X, columns=cols)
    p = QS.fit_params()
    assert p["device"] == "cpu"
    m = xgb.XGBRegressor(**dict(p, n_estimators=20)).fit(df[cols], y, sample_weight=w)
    pred = np.asarray(m.predict(df[cols]), dtype=float)
    assert np.isfinite(pred).all() and float(np.std(pred)) > 0, "predictions must vary"
    dump = m.get_booster().get_dump(dump_format="text")
    assert sum(t.count("[f") for t in dump) > 0, "trees must split"
    null_mse = float(((y - y.mean()) ** 2).mean())
    assert float(((pred - y) ** 2).mean()) < null_mse, "must beat constant-mean"
