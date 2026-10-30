"""D39 huber contract: fixed pseudo-huber objective, E0003-identical
params/rows except infra (threads+GPU) + preregistered loss, no forbidden
inputs, scratch separation. Fast: no data reads; one small synthetic CPU fit
for the E0089 variance-smoke guard."""

import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d39_huber"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import huber as HU  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]


def test_huber_rule_fixed_and_causal():
    assert HU.HUBER_DELTA == 100.0
    assert HU.BASE_SCORE == 0.5  # E0089 retry pin: omitted auto-estimate diverges
    assert HU.HUBER_OBJECTIVE == "reg:pseudohubererror"
    assert HU.check_huber()["formula"] == HU.LOSS_FORMULA
    assert HU.check_huber()["base_score"] == 0.5
    assert "100" in HU.LOSS_FORMULA and "pseudohuber" in HU.LOSS_FORMULA
    assert "base_score" in HU.LOSS_FORMULA
    assert "no fitted choice" in HU.SELECT_RULE.lower()
    assert "no dev/selection" in HU.SELECT_RULE.lower() or "no dev/selection/audit" in HU.SELECT_RULE.lower()
    assert "X0008" in HU.SELECT_RULE
    assert "E0089" in HU.SELECT_RULE  # retry cites the constant-7078 evidence
    assert "sample_weight" not in HU.SELECT_RULE  # objective replaces weighting


def test_fit_params_match_e0003_recipe_infra_and_loss_only():
    p = HU.fit_params("cuda:1")
    assert set(p) == set(ia.PARAMS) | {"device", "objective", "huber_slope", "base_score"}
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["tree_method"] == "hist"
    assert str(p["device"]).startswith("cuda")
    assert p["objective"] == "reg:pseudohubererror"
    assert float(p["huber_slope"]) == 100.0
    assert float(p["base_score"]) == 0.5  # pinned start point, E0089 fix
    assert p["n_jobs"] == HU.FIT_NJOBS == 3
    assert HU.PERIOD_JOBS * HU.FIT_NJOBS <= 10


def test_devices_round_robin_visible_gpus(monkeypatch):
    monkeypatch.delenv("EXPOS_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert HU.device_for(0) == "cuda"
    monkeypatch.setenv("EXPOS_GPUS", "0,1")
    assert HU.device_for(0) == "cuda:0"
    assert HU.device_for(1) == "cuda:1"
    assert HU.device_for(2) == "cuda:0"


def test_full_rule_prespecified_and_no_fitted_choice():
    assert "full 3h" in HU.FULL_RULE and "E0001" in HU.FULL_RULE
    assert "pooled F1" in HU.FULL_RULE or "F1" in HU.FULL_RULE
    assert "no full run" in HU.FULL_RULE
    assert "0.41121495" in HU.FULL_RULE  # adoption bar cited, report-only
    assert "freeze" in HU.FULL_RULE.lower()


def test_owned_files_and_no_vendored_strips():
    pkg = LB / "experiments" / "d39_huber"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 4, files
    assert {"huber.py", "runner.py"} <= set(files)
    runner = (pkg / "runner.py").read_text()
    assert "import strips" in runner or "import ST" in runner
    assert "import huber" in runner or "import HU" in runner
    assert "riseweight" not in runner and "calweight" not in runner
    assert "weights_for_frame" not in runner and "sample_weight" not in runner
    assert "store/scratch/worker-d39b" in runner
    for other in ("worker-d35", "worker-d36", "worker-d37", "worker-d33", "worker-d34", "worker-d21", "worker-d28", "worker-d22", "worker-d23", "worker-d40"):
        assert other not in runner
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()


def test_runner_writes_three_predictions_and_real_poison():
    runner = (LB / "experiments" / "d39_huber" / "runner.py").read_text()
    assert "for i, n in enumerate(names)" in runner or "for n in PERIODS" in runner
    assert "poison.json" in runner and '"passed"' in runner
    assert "ST.poison_test" in runner
    assert "reg:pseudohubererror" in runner
    assert "huber_slope" in runner
    assert "device" in runner and "cuda" in runner
    assert "SystemExit" in runner  # fail-closed on poison/non-finite


def test_pseudohuber_variance_smoke_km_s_scale():
    """E0089 regression guard: the fixed objective must split and vary on raw
    km/s-scale targets (synthetic, CPU, no data reads). Fails on the buggy
    config (omitted base_score -> constant ~6210, zero splits) and on
    slope=1.0 (constant 0.5, zero splits even when pinned)."""
    import numpy as np
    import xgboost as xgb

    rng = np.random.default_rng(7)
    n = 3000
    X = rng.standard_normal((n, 5))
    y = 400.0 + 50.0 * X[:, 0] + 10.0 * rng.standard_normal(n)
    p = HU.fit_params("cuda")
    p["device"] = "cpu"
    p["n_jobs"] = 2
    m = xgb.XGBRegressor(**dict(p, n_estimators=30)).fit(X, y)
    pred = np.asarray(m.predict(X), dtype=float)
    assert np.isfinite(pred).all()
    assert float(np.std(pred)) > 0, "predictions must vary (E0089 was std 0)"
    dump = m.get_booster().get_dump(dump_format="text")
    assert sum(t.count("[f") for t in dump) > 0, "trees must split (E0089 had 0)"
    null_mse = float(((y - y.mean()) ** 2).mean())
    assert float(((pred - y) ** 2).mean()) < null_mse, "must beat constant-mean"
