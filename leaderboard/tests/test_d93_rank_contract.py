"""D93 rank-target contract: hypothesis tags, single-axis rank target, twin spec, isolation.

Fast static + unit checks only: no data reads, no fits beyond one small
synthetic CPU fit, no official contact. The pilot run itself (registered
separately) proves end-to-end behaviour through expos run/evaluate/poison.
"""
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d93_rank"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
sys.path.insert(0, str(LB))

import rankspec as RS  # noqa: E402
import info_ablation as ia  # noqa: E402
from expos import core  # noqa: E402

RUNNER_SRC = (LB / "experiments" / "d93_rank" / "runner.py").read_text()
SPEC_SRC = (LB / "experiments" / "d93_rank" / "rankspec.py").read_text()

COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_hypothesis_dual_tag_and_experience():
    assert RS.HYPOTHESIS.startswith("[family=xgb] [axes: target=rank]")
    assert RS.FAMILY_TAG == "[family=xgb]" and RS.AXES_TAG == "[axes: target=rank]"
    assert set(RS.EXPERIENCE) == {"X0104", "X0138"}
    assert "X0104" in RS.HYPOTHESIS and "X0138" in RS.HYPOTHESIS
    assert "research/AXES.md" in RS.HYPOTHESIS and "target" in RS.HYPOTHESIS
    assert "log/rank/quantile-transformed target" in RS.HYPOTHESIS
    assert "quantile inverse" in RS.HYPOTHESIS
    assert "E0001" in RS.HYPOTHESIS and "E0003" in RS.HYPOTHESIS
    assert RS.DIRECTION == "D93" and RS.OWNER == "worker-d93"
    assert "6h" in RS.HYPOTHESIS
    assert "3h" in RS.HYPOTHESIS
    assert "rc==0" in RS.HYPOTHESIS and "F1>=0.30" in RS.HYPOTHESIS
    assert "screening" in RS.HYPOTHESIS
    assert "gpu0" in RS.HYPOTHESIS
    assert "No freeze" in RS.HYPOTHESIS
    assert "official-score" in RS.HYPOTHESIS


def test_single_axis_rank_target_only():
    p = RS.fit_params("cpu")
    assert p["device"] == "cpu"
    assert RS.fit_params("cuda")["device"] == "cuda"
    assert p.get("objective", "reg:squarederror") == "reg:squarederror"
    assert "tweedie_variance_power" not in p
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k in ("n_jobs", "random_state"):
            continue
        assert p[k] == v, f"param {k} differs from E0001 recipe"
    assert p["n_estimators"] == 600 and p["max_depth"] == 6
    assert p["random_state"] == 0
    assert "base_score" not in p
    assert "sample_weight" not in RUNNER_SRC
    assert RS.TRANSFORM == "rank"
    assert type(__import__("xgboost").XGBRegressor(**p)).__name__ == "XGBRegressor"


def test_e0001_twin_rows_purge_budget():
    assert RS.PURGE_H == 72
    assert RS.FREQ_PILOT == "6h" and RS.FREQ_FULL == "3h"
    assert RS.N_ESTIMATORS == 600 and RS.SEED == 0
    assert 'ia.arms(df.columns)["existing_all"]' in RUNNER_SRC
    assert "RS.PURGE_H" in RUNNER_SRC
    assert RUNNER_SRC.count(".fit(") == 1, "exactly one head fit (single head, no refit)"
    assert RS.check_spec()["transform"] == "rank"


def test_gpu_budget_and_lease_reason():
    assert RS.PERIOD_JOBS * RS.FIT_NJOBS <= 10
    assert RS.PERIOD_JOBS == 3
    assert RS.FIT_NJOBS <= 2
    assert "cuda" in RUNNER_SRC
    assert "device_for" in RUNNER_SRC
    assert "D93" in RS.GPU_REASON and "gpu0" in RS.GPU_REASON
    assert "D94" in RS.GPU_REASON and "D95" in RS.GPU_REASON and "CPU" in RS.GPU_REASON
    assert "60s" in RS.GPU_REASON and "--prefer-gpu 0" in RS.GPU_REASON
    assert "n_jobs<=20" in RS.GPU_REASON
    assert "foreground" in RS.GPU_REASON


def test_scratch_isolation():
    assert "store/scratch/worker-d93" in RUNNER_SRC
    others = [l for l in RUNNER_SRC.splitlines() if "worker-" in l and "worker-d93" not in l]
    assert others == [], f"scratch leak: {others[:3]}"
    assert RS.SCRATCH_NAME == "worker-d93"
    assert "featcache" in RUNNER_SRC and "fcache_key" in RUNNER_SRC


def test_readonly_admin_no_vendored_strips():
    for s in COPY_BANS:
        assert s not in RUNNER_SRC, f"runner vendors d1-strip code {s!r}"
        assert s not in SPEC_SRC, f"spec vendors d1-strip code {s!r}"
    assert "import strips" not in RUNNER_SRC


def test_no_forbidden_paths_in_registered_code():
    assert core.scan(RUNNER_SRC + "\n" + SPEC_SRC) == []
    assert core.scan(RS.HYPOTHESIS) == []


def test_full_rule_preregistered_no_freeze_claim():
    assert "rc==0" in RS.FULL_RULE and "poison" in RS.FULL_RULE and "E0001" in RS.FULL_RULE
    assert "F1>=0.30" in RS.FULL_RULE
    assert "6h F1" in RS.FULL_RULE
    assert "3h" in RS.FULL_RULE
    assert "No freeze" in SPEC_SRC
    assert "official_score(" not in RUNNER_SRC and "core.freeze" not in RUNNER_SRC
    assert "official-score" in RS.FULL_RULE
    assert "E0003" in RS.FULL_RULE


def test_predictions_three_kinds_and_poison():
    assert "predictions_{dev,selection,audit}" in RUNNER_SRC
    assert "write(te, pred, name)" in RUNNER_SRC
    assert "predictions_official" not in RUNNER_SRC
    assert "poison_check" in RUNNER_SRC and "positive_control" in RUNNER_SRC
    assert "quantile_ref_frozen" in RUNNER_SRC
    assert "passed" in RUNNER_SRC and "poison.json" in RUNNER_SRC
    assert "SystemExit" in RUNNER_SRC


def test_owned_dir_is_small():
    files = sorted(p.name for p in (LB / "experiments" / "d93_rank").iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"rankspec.py", "runner.py"} <= set(files)


def test_rank_transform_smoke_km_s_scale():
    """The preregistered rank target must split, invert and vary on km/s-scale
    targets (synthetic, CPU, no data reads); guards a silent constant-head failure."""
    import numpy as np
    import xgboost as xgb

    rng = np.random.default_rng(11)
    n = 3000
    X = rng.standard_normal((n, 5))
    y = np.clip(400.0 + 50.0 * X[:, 0] + 10.0 * rng.standard_normal(n), 150.0, None)
    assert bool((y > 0).all())
    r = RS.to_ranks(y)
    assert r.shape == (n,) and np.isfinite(r).all()
    assert bool(((r > 0) & (r < 1)).all())
    back = RS.from_ranks(r, y)
    assert np.allclose(np.sort(back), np.sort(y)), "quantile inverse must recover the distribution"
    probe = np.array([350.0, 450.0, 700.0, 500.0])
    assert np.allclose(RS.from_ranks(RS.to_ranks(probe), probe), probe)
    p = RS.fit_params("cpu")
    m = xgb.XGBRegressor(**dict(p, n_estimators=100)).fit(X, r)
    pred = np.asarray(RS.from_ranks(np.asarray(m.predict(X), dtype=float), y), dtype=float)
    assert np.isfinite(pred).all()
    assert bool((pred > 0).all()), "quantile inverse predictions must stay positive"
    assert float(np.std(pred)) > 0, "predictions must vary"
    dump = m.get_booster().get_dump(dump_format="text")
    assert sum(t.count("[f") for t in dump) > 0, "trees must split"
    null_mse = float(((y - y.mean()) ** 2).mean())
    assert float(((pred - y) ** 2).mean()) < null_mse, "must beat constant-mean"
