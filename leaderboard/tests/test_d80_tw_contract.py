"""D80 Tweedie contract: hypothesis tags, single-axis log-link loss, twin spec, isolation.

Fast static + unit checks only: no data reads, no fits beyond one small
synthetic CPU fit, no official contact. The pilot run itself (registered
separately) proves end-to-end behaviour through expos run/evaluate/poison.
"""
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d80_tweedie"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
sys.path.insert(0, str(LB))

import tweedie as TW  # noqa: E402
import info_ablation as ia  # noqa: E402
from expos import core  # noqa: E402

RUNNER_SRC = (LB / "experiments" / "d80_tweedie" / "runner.py").read_text()
SPEC_SRC = (LB / "experiments" / "d80_tweedie" / "tweedie.py").read_text()

COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_hypothesis_dual_tag_and_experience():
    assert TW.HYPOTHESIS.startswith("[family=xgb] [axes: loss=tweedie]")
    assert TW.EXPERIENCE == "X0104" and "X0104" in TW.HYPOTHESIS
    assert "research/AXES.md" in TW.HYPOTHESIS and "loss" in TW.HYPOTHESIS
    assert "Tweedie/gamma" in TW.HYPOTHESIS  # AXES.md loss-section quote
    assert "positive skew" in TW.HYPOTHESIS
    assert TW.DIRECTION == "D80" and TW.OWNER == "worker-d80"
    assert "6h" in TW.HYPOTHESIS  # pilot grid stated
    assert "tweedie_variance_power=1.5" in TW.HYPOTHESIS  # parameter prespecified
    assert "No freeze" in TW.HYPOTHESIS
    assert "official-score" in TW.HYPOTHESIS or "official_score" in TW.HYPOTHESIS


def test_single_axis_tweedie_loss_only():
    p = TW.fit_params()
    assert p["objective"] == "reg:tweedie"
    assert float(p["tweedie_variance_power"]) == 1.5
    assert p["device"] == "cpu"
    # E0001 recipe untouched except infra + preregistered objective
    assert set(p) == set(ia.PARAMS) | {"device", "objective", "tweedie_variance_power"}
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0001 recipe"
    assert p["n_estimators"] == 600 and p["max_depth"] == 6
    assert "base_score" not in p, "start point stays the E0001 default (one axis only)"
    assert "sample_weight" not in RUNNER_SRC
    assert type(__import__("xgboost").XGBRegressor(**p)).__name__ == "XGBRegressor"


def test_e0001_twin_rows_purge_budget():
    assert TW.PURGE_HOURS == 72
    assert TW.FREQ_PILOT == "6h" and TW.FREQ_FULL == "3h"
    assert TW.N_ESTIMATORS == 600 and TW.SEED == 0
    assert 'ia.arms(df.columns)["existing_all"]' in RUNNER_SRC
    assert "TW.PURGE_HOURS" in RUNNER_SRC
    assert RUNNER_SRC.count(".fit(") == 1, "exactly one head fit (single head, no refit)"
    assert TW.check_twin()["objective"] == "reg:tweedie"
    assert TW.check_twin()["experience"] == "X0104"


def test_cpu_budget_no_accelerator():
    assert TW.PERIOD_JOBS * TW.FIT_NJOBS <= 10
    assert TW.PERIOD_JOBS == 3, "three periods run in parallel"
    assert TW.FIT_NJOBS <= 2
    assert TW.fit_params()["device"] == "cpu"
    blob = RUNNER_SRC + "\n" + SPEC_SRC
    for pat in ("cuda", "gpu_lease", "--gpu", "prefer-gpu", "EXPOS_GPUS",
                "CUDA_VISIBLE_DEVICES", "device_for"):
        assert pat not in blob, pat
    assert "CPU" in RUNNER_SRC  # CPU-only guard documented
    assert "D78" in TW.CPU_REASON and "D79" in TW.CPU_REASON  # lease reason recorded


def test_scratch_isolation():
    assert "store/scratch/worker-d80" in RUNNER_SRC
    others = [l for l in RUNNER_SRC.splitlines() if "worker-" in l and "worker-d80" not in l]
    assert others == [], f"scratch leak: {others[:3]}"
    assert "featcache" in RUNNER_SRC and "fcache_key" in RUNNER_SRC


def test_readonly_admin_no_vendored_strips():
    for s in COPY_BANS:
        assert s not in RUNNER_SRC, f"runner vendors d1-strip code {s!r}"
        assert s not in SPEC_SRC, f"spec vendors d1-strip code {s!r}"
    assert "import strips" not in RUNNER_SRC  # E0001 twin needs no strips


def test_no_forbidden_paths_in_registered_code():
    assert core.scan(RUNNER_SRC + "\n" + SPEC_SRC) == []
    assert core.scan(TW.HYPOTHESIS) == []


def test_full_rule_preregistered_no_freeze_claim():
    assert "rc==0" in TW.FULL_RULE and "poison" in TW.FULL_RULE and "E0001" in TW.FULL_RULE
    assert "F1>=0.30" in TW.FULL_RULE and "No freeze" in TW.HYPOTHESIS
    assert "No freeze" in SPEC_SRC
    assert "official_score(" not in RUNNER_SRC and "core.freeze" not in RUNNER_SRC
    assert "official-score" in TW.FULL_RULE


def test_predictions_three_kinds_and_poison():
    assert "predictions_{dev,selection,audit}" in RUNNER_SRC  # output contract
    assert "write(te, pred, name)" in RUNNER_SRC  # per-period full-grid write
    assert "predictions_official" not in RUNNER_SRC
    assert "poison_check" in RUNNER_SRC and "positive_control" in RUNNER_SRC
    assert "passed" in RUNNER_SRC and "poison.json" in RUNNER_SRC
    assert "SystemExit" in RUNNER_SRC  # fail-closed on poison/non-finite


def test_owned_dir_is_small():
    files = sorted(p.name for p in (LB / "experiments" / "d80_tweedie").iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"tweedie.py", "runner.py"} <= set(files)


def test_tweedie_smoke_km_s_scale():
    """The preregistered Tweedie objective must split, stay positive and vary
    on raw km/s-scale targets (synthetic, CPU, no data reads); guards a
    D39/E0089-style silent constant-head failure."""
    import numpy as np
    import xgboost as xgb

    rng = np.random.default_rng(11)
    n = 3000
    X = rng.standard_normal((n, 5))
    y = np.clip(400.0 + 50.0 * X[:, 0] + 10.0 * rng.standard_normal(n), 150.0, None)
    p = TW.fit_params()
    m = xgb.XGBRegressor(**dict(p, n_estimators=100)).fit(X, y)
    pred = np.asarray(m.predict(X), dtype=float)
    assert np.isfinite(pred).all()
    assert bool((pred > 0).all()), "log-link predictions must stay positive"
    assert float(np.std(pred)) > 0, "predictions must vary"
    dump = m.get_booster().get_dump(dump_format="text")
    assert sum(t.count("[f") for t in dump) > 0, "trees must split"
    null_mse = float(((y - y.mean()) ** 2).mean())
    assert float(((pred - y) ** 2).mean()) < null_mse, "must beat constant-mean"
