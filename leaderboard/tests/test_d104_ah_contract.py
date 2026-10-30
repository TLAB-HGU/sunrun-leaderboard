"""D104 asym-Huber contract: hypothesis tags, single-axis directional-robust loss, twin spec, GPU isolation.

Fast static + unit checks only: no data reads, no fits beyond small
synthetic CPU fits, no official contact. The pilot run itself (registered
separately) proves end-to-end behaviour through expos run/evaluate/poison.
"""
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d104_asym"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
sys.path.insert(0, str(LB))

import asymhuber as AH  # noqa: E402
import info_ablation as ia  # noqa: E402
from expos import core  # noqa: E402

RUNNER_SRC = (LB / "experiments" / "d104_asym" / "runner.py").read_text()
SPEC_SRC = (LB / "experiments" / "d104_asym" / "asymhuber.py").read_text()


def test_hypothesis_dual_tag_and_experience():
    assert AH.HYPOTHESIS.startswith("[family=xgb] [axes: loss=asym-huber]")
    assert set(AH.EXPERIENCE) == {"X0106", "X0116"}
    assert "X0106" in AH.HYPOTHESIS and "X0116" in AH.HYPOTHESIS
    assert "research/AXES.md" in AH.HYPOTHESIS and "loss" in AH.HYPOTHESIS
    assert "underprediction" in AH.HYPOTHESIS.lower()
    assert AH.DIRECTION == "D104" and AH.OWNER == "worker-d104"
    assert "6h" in AH.HYPOTHESIS  # pilot grid stated
    assert "3h" in AH.HYPOTHESIS  # F1 claims require 3h density
    assert "screening-only" in AH.HYPOTHESIS or "screening" in AH.HYPOTHESIS
    assert "No freeze" in AH.HYPOTHESIS
    assert "official-score" in AH.HYPOTHESIS or "official_score" in AH.HYPOTHESIS


def test_single_axis_asymhuber_loss_only():
    p = AH.fit_params("cuda")
    assert callable(p["objective"])
    assert p["objective"] is AH.asym_huber_obj
    assert float(p["base_score"]) == 0.5
    assert str(p["device"]).startswith("cuda")
    # E0001 recipe untouched except infra + preregistered objective
    assert set(p) == set(ia.PARAMS) | {"device", "objective", "base_score"}
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0001 recipe"
    assert p["n_estimators"] == 600 and p["max_depth"] == 6
    assert "sample_weight" not in RUNNER_SRC
    assert type(__import__("xgboost").XGBRegressor(**{k: v for k, v in p.items() if k != "objective"},
                                                    objective="reg:squarederror")).__name__ == "XGBRegressor"


def test_e0001_twin_rows_purge_budget():
    assert AH.PURGE_HOURS == 72
    assert AH.FREQ_PILOT == "6h" and AH.FREQ_FULL == "3h"
    assert AH.N_ESTIMATORS == 600 and AH.SEED == 0
    assert AH.DELTA == 100.0 and AH.UNDER_WEIGHT == 2.0
    assert 'ia.arms(df.columns)["existing_all"]' in RUNNER_SRC
    assert "AH.PURGE_HOURS" in RUNNER_SRC
    assert RUNNER_SRC.count(".fit(") == 1, "exactly one head fit (single head, no refit)"
    assert AH.check_twin()["under_weight"] == 2.0
    assert AH.check_twin()["experience"] == ["X0106", "X0116"]


def test_gpu_budget_and_devices():
    assert AH.PERIOD_JOBS * AH.FIT_NJOBS <= 10
    assert AH.PERIOD_JOBS == 3, "three periods run in parallel"
    assert AH.FIT_NJOBS <= 2
    assert AH.fit_params("cuda")["device"] == "cuda"
    assert "cuda" in RUNNER_SRC
    assert "ThreadPoolExecutor" in RUNNER_SRC
    assert "gpu1" in AH.GPU_REASON and "D102" in AH.GPU_REASON and "D103" in AH.GPU_REASON
    assert "20" in AH.GPU_REASON  # shared thread budget recorded


def test_devices_round_robin_visible_gpus(monkeypatch):
    monkeypatch.delenv("EXPOS_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert AH.device_for(0) == "cuda"
    monkeypatch.setenv("EXPOS_GPUS", "0,1")
    assert AH.device_for(0) == "cuda:0"
    assert AH.device_for(1) == "cuda:1"
    assert AH.device_for(2) == "cuda:0"


def test_scratch_isolation():
    assert "store/scratch/worker-d104" in RUNNER_SRC
    others = [l for l in RUNNER_SRC.splitlines() if "worker-" in l and "worker-d104" not in l]
    assert others == [], f"scratch leak: {others[:3]}"
    assert "featcache" in RUNNER_SRC and "fcache_key" in RUNNER_SRC


def test_no_forbidden_paths_in_registered_code():
    assert core.scan(RUNNER_SRC + "\n" + SPEC_SRC) == []
    assert core.scan(AH.HYPOTHESIS) == []


def test_full_rule_preregistered_no_freeze_claim():
    assert "rc==0" in AH.FULL_RULE and "poison" in AH.FULL_RULE and "E0001" in AH.FULL_RULE
    assert "full 3h" in AH.FULL_RULE
    assert "no full run" in AH.FULL_RULE
    assert "0.41121495" in AH.FULL_RULE  # adoption bar cited, report-only
    assert "freeze" in AH.FULL_RULE.lower()
    assert "official scoring" in AH.FULL_RULE.lower()
    assert "No freeze" in AH.HYPOTHESIS
    assert "official_score(" not in RUNNER_SRC and "core.freeze" not in RUNNER_SRC
    assert "official-score" in AH.FULL_RULE or "official scoring" in AH.FULL_RULE


def test_predictions_three_kinds_and_poison():
    assert "predictions_{dev,selection,audit}" in RUNNER_SRC  # output contract
    assert "write(te, pred, name)" in RUNNER_SRC  # per-period full-grid write
    assert "predictions_official" not in RUNNER_SRC
    assert "poison_check" in RUNNER_SRC and "positive_control" in RUNNER_SRC
    assert "passed" in RUNNER_SRC and "poison.json" in RUNNER_SRC
    assert "SystemExit" in RUNNER_SRC  # fail-closed on poison/non-finite


def test_owned_dir_is_small():
    files = sorted(p.name for p in (LB / "experiments" / "d104_asym").iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"asymhuber.py", "runner.py"} <= set(files)


def test_asymhuber_grad_asymmetry():
    """Preregistered grad rule: underpredictions carry 2x Huber gradient."""
    import numpy as np

    g_under, h_under = AH.asym_huber_obj(np.array([500.0]), np.array([450.0]))
    g_over, h_over = AH.asym_huber_obj(np.array([400.0]), np.array([450.0]))
    assert float(g_under[0]) == -100.0  # 2.0 * clip(-50) inside delta
    assert float(g_over[0]) == 50.0
    assert abs(float(g_under[0])) == 2.0 * abs(float(g_over[0]))
    assert float(h_under[0]) == 2.0 and float(h_over[0]) == 1.0
    # linear region keeps positive curvature (no zero-hess stall)
    _, h_far = AH.asym_huber_obj(np.array([400.0]), np.array([700.0]))
    assert float(h_far[0]) > 0


def test_asymhuber_smoke_km_s_scale():
    """The preregistered asym-Huber objective must split and vary on raw
    km/s-scale targets (synthetic, CPU, no data reads); guards a D39/E0089-style
    silent constant-head failure."""
    import numpy as np
    import xgboost as xgb

    rng = np.random.default_rng(7)
    n = 3000
    X = rng.standard_normal((n, 5))
    y = 400.0 + 50.0 * X[:, 0] + 10.0 * rng.standard_normal(n)
    p = AH.fit_params("cuda")
    p = dict(p, device="cpu", n_estimators=100)
    m = xgb.XGBRegressor(**p).fit(X, y)
    pred = np.asarray(m.predict(X), dtype=float)
    assert np.isfinite(pred).all()
    assert float(np.std(pred)) > 0, "predictions must vary"
    dump = m.get_booster().get_dump(dump_format="text")
    assert sum(t.count("[f") for t in dump) > 0, "trees must split"
    null_mse = float(((y - y.mean()) ** 2).mean())
    assert float(((pred - y) ** 2).mean()) < null_mse, "must beat constant-mean"
