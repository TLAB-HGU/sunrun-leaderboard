"""D62 quantile contract: hypothesis tags, single-axis median loss, twin spec, isolation.

Fast static + unit checks only: no data reads, no fits beyond one small
synthetic CPU fit, no official contact. The pilot run itself (registered
separately) proves end-to-end behaviour through expos run/evaluate/poison.
"""
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d62_quantile"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
sys.path.insert(0, str(LB))

import quantile as QU  # noqa: E402
import info_ablation as ia  # noqa: E402
from expos import core  # noqa: E402

RUNNER_SRC = (LB / "experiments" / "d62_quantile" / "runner.py").read_text()
QUANT_SRC = (LB / "experiments" / "d62_quantile" / "quantile.py").read_text()


def test_hypothesis_dual_tag_and_experience():
    assert QU.HYPOTHESIS.startswith("[family=xgb] [axes: loss=quantile]")
    assert QU.EXPERIENCE == "X0086" and "X0086" in QU.HYPOTHESIS
    assert "research/AXES.md" in QU.HYPOTHESIS and "loss" in QU.HYPOTHESIS
    assert "median" in QU.HYPOTHESIS
    assert QU.DIRECTION == "D62" and QU.OWNER == "worker-d62"
    assert "6h" in QU.HYPOTHESIS  # pilot grid stated
    assert "No freeze" in QU.HYPOTHESIS
    assert "official-score" in QU.HYPOTHESIS or "official_score" in QU.HYPOTHESIS


def test_single_axis_median_loss_only():
    p = QU.fit_params()
    assert p["objective"] == "reg:quantileerror"
    assert float(p["quantile_alpha"]) == 0.5
    assert p["device"] == "cpu"
    # E0001 recipe untouched except infra + preregistered objective
    assert set(p) == set(ia.PARAMS) | {"device", "objective", "quantile_alpha"}
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0001 recipe"
    assert p["n_estimators"] == 600 and p["max_depth"] == 6
    assert "sample_weight" not in RUNNER_SRC
    assert type(__import__("xgboost").XGBRegressor(**p)).__name__ == "XGBRegressor"


def test_e0001_twin_rows_purge_budget():
    assert QU.PURGE_HOURS == 72
    assert QU.FREQ_PILOT == "6h" and QU.FREQ_FULL == "3h"
    assert QU.N_ESTIMATORS == 600 and QU.SEED == 0
    assert 'ia.arms(df.columns)["existing_all"]' in RUNNER_SRC
    assert "QU.PURGE_HOURS" in RUNNER_SRC
    assert RUNNER_SRC.count(".fit(") == 1, "exactly one head fit (single head, no refit)"
    assert QU.check_twin()["objective"] == "reg:quantileerror"
    assert QU.check_twin()["experience"] == "X0086"


def test_cpu_budget_no_accelerator():
    assert QU.PERIOD_JOBS * QU.FIT_NJOBS <= 10
    assert QU.PERIOD_JOBS == 3, "three periods run in parallel"
    assert QU.FIT_NJOBS <= 2
    assert QU.fit_params()["device"] == "cpu"
    blob = RUNNER_SRC + "\n" + QUANT_SRC
    for pat in ("cuda", "gpu_lease", "--gpu", "prefer-gpu", "EXPOS_GPUS",
                "CUDA_VISIBLE_DEVICES", "device_for"):
        assert pat not in blob, pat
    assert "CPU" in RUNNER_SRC  # CPU-only guard documented
    assert "D54" in QU.CPU_REASON and "D58" in QU.CPU_REASON  # lease reason recorded


def test_scratch_isolation():
    assert "store/scratch/worker-d62" in RUNNER_SRC
    others = [l for l in RUNNER_SRC.splitlines() if "worker-" in l and "worker-d62" not in l]
    assert others == [], f"scratch leak: {others[:3]}"
    assert "featcache" in RUNNER_SRC and "fcache_key" in RUNNER_SRC


def test_no_forbidden_paths_in_registered_code():
    assert core.scan(RUNNER_SRC + "\n" + QUANT_SRC) == []
    assert core.scan(QU.HYPOTHESIS) == []


def test_full_rule_preregistered_no_freeze_claim():
    assert "rc==0" in QU.FULL_RULE and "poison" in QU.FULL_RULE and "E0001" in QU.FULL_RULE
    assert "F1>=0.30" in QU.FULL_RULE and "No freeze" in QU.HYPOTHESIS
    assert "No freeze" in QUANT_SRC
    assert "official_score(" not in RUNNER_SRC and "core.freeze" not in RUNNER_SRC
    assert "official-score" in QU.FULL_RULE


def test_predictions_three_kinds_and_poison():
    assert "predictions_{dev,selection,audit}" in RUNNER_SRC  # output contract
    assert "write(te, pred, name)" in RUNNER_SRC  # per-period full-grid write
    assert "predictions_official" not in RUNNER_SRC
    assert "poison_check" in RUNNER_SRC and "positive_control" in RUNNER_SRC
    assert "passed" in RUNNER_SRC and "poison.json" in RUNNER_SRC
    assert "SystemExit" in RUNNER_SRC  # fail-closed on poison/non-finite


def test_owned_dir_is_small():
    files = sorted(p.name for p in (LB / "experiments" / "d62_quantile").iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"quantile.py", "runner.py"} <= set(files)


def test_quantile_median_smoke_km_s_scale():
    """The preregistered median objective must split and vary on raw km/s-scale
    targets (synthetic, CPU, no data reads); guards a D39/E0089-style silent
    constant-head failure."""
    import numpy as np
    import xgboost as xgb

    rng = np.random.default_rng(7)
    n = 3000
    X = rng.standard_normal((n, 5))
    y = 400.0 + 50.0 * X[:, 0] + 10.0 * rng.standard_normal(n)
    p = QU.fit_params()
    m = xgb.XGBRegressor(**dict(p, n_estimators=30)).fit(X, y)
    pred = np.asarray(m.predict(X), dtype=float)
    assert np.isfinite(pred).all()
    assert float(np.std(pred)) > 0, "predictions must vary"
    dump = m.get_booster().get_dump(dump_format="text")
    assert sum(t.count("[f") for t in dump) > 0, "trees must split"
    med = float(np.median(y))
    null_mae = float(np.abs(y - med).mean())
    assert float(np.abs(pred - y).mean()) < null_mae, "must beat constant-median"
