"""D91 increments-target contract: hypothesis tags, single-axis delta target, twin spec, isolation.

Fast static + unit checks only: no data reads, no fits beyond one small
synthetic CPU fit, no official contact. The pilot run itself (registered
separately) proves end-to-end behaviour through expos run/evaluate/poison.
"""
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d91_incr"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
sys.path.insert(0, str(LB))

import incr as IN  # noqa: E402
import info_ablation as ia  # noqa: E402
from expos import core  # noqa: E402

RUNNER_SRC = (LB / "experiments" / "d91_incr" / "runner.py").read_text()
SPEC_SRC = (LB / "experiments" / "d91_incr" / "incr.py").read_text()

COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_hypothesis_dual_tag_and_experience():
    assert IN.HYPOTHESIS.startswith("[family=xgb] [axes: target=increments]")
    assert IN.EXPERIENCE == "X0104" and "X0104" in IN.HYPOTHESIS
    assert "research/AXES.md" in IN.HYPOTHESIS and "target" in IN.HYPOTHESIS
    assert "predict increments between horizons (smoothness)" in IN.HYPOTHESIS  # AXES.md target-section quote
    assert "smooth" in IN.HYPOTHESIS
    assert IN.DIRECTION == "D91" and IN.OWNER == "worker-d91"
    assert "6h" in IN.HYPOTHESIS  # pilot grid stated
    assert "cumsum" in IN.HYPOTHESIS  # reconstruction stated
    assert "No freeze" in IN.HYPOTHESIS
    assert "official-score" in IN.HYPOTHESIS or "official_score" in IN.HYPOTHESIS


def test_single_axis_increments_only():
    p = IN.fit_params()
    assert p["device"] == "cpu"
    assert p.get("objective", "reg:squarederror") == "reg:squarederror"
    assert "tweedie_variance_power" not in p
    # E0001 recipe untouched except infra (target change is data-side only)
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0001 recipe"
    assert p["n_estimators"] == 600 and p["max_depth"] == 6
    assert "base_score" not in p, "start point stays the E0001 default (one axis only)"
    assert "sample_weight" not in RUNNER_SRC
    assert IN.TRANSFORM == "increments"
    assert type(__import__("xgboost").XGBRegressor(**p)).__name__ == "XGBRegressor"


def test_e0001_twin_rows_purge_budget():
    assert IN.PURGE_HOURS == 72
    assert IN.FREQ_PILOT == "6h" and IN.FREQ_FULL == "3h"
    assert IN.N_ESTIMATORS == 600 and IN.SEED == 0
    assert 'ia.arms(raw.columns)["existing_all"]' in RUNNER_SRC or \
        'ia.arms(df.columns)["existing_all"]' in RUNNER_SRC
    assert "IN.PURGE_HOURS" in RUNNER_SRC
    assert RUNNER_SRC.count(".fit(") == 1, "exactly one head fit (single head, no refit)"
    assert IN.check_twin()["transform"] == "increments"
    assert IN.check_twin()["experience"] == "X0104"


def test_cpu_budget_no_accelerator():
    assert IN.PERIOD_JOBS * IN.FIT_NJOBS <= 10
    assert IN.PERIOD_JOBS == 3, "three periods run in parallel"
    assert IN.FIT_NJOBS <= 2
    assert IN.fit_params()["device"] == "cpu"
    blob = RUNNER_SRC + "\n" + SPEC_SRC
    for pat in ("cuda", "gpu_lease", "--gpu", "prefer-gpu", "EXPOS_GPUS",
                "CUDA_VISIBLE_DEVICES", "device_for"):
        assert pat not in blob, pat
    assert "CPU" in RUNNER_SRC  # CPU-only guard documented
    assert "D90" in IN.CPU_REASON and "D92" in IN.CPU_REASON  # lease reason recorded


def test_scratch_isolation():
    assert "store/scratch/worker-d91" in RUNNER_SRC
    others = [l for l in RUNNER_SRC.splitlines() if "worker-" in l and "worker-d91" not in l]
    assert others == [], f"scratch leak: {others[:3]}"
    assert "featcache" in RUNNER_SRC and "fcache_key" in RUNNER_SRC


def test_readonly_admin_no_vendored_strips():
    for s in COPY_BANS:
        assert s not in RUNNER_SRC, f"runner vendors d1-strip code {s!r}"
        assert s not in SPEC_SRC, f"spec vendors d1-strip code {s!r}"
    assert "import strips" not in RUNNER_SRC  # E0001 twin needs no strips


def test_no_forbidden_paths_in_registered_code():
    assert core.scan(RUNNER_SRC + "\n" + SPEC_SRC) == []
    assert core.scan(IN.HYPOTHESIS) == []


def test_full_rule_preregistered_no_freeze_claim():
    assert "rc==0" in IN.FULL_RULE and "poison" in IN.FULL_RULE and "E0001" in IN.FULL_RULE
    assert "F1>=0.30" in IN.FULL_RULE and "No freeze" in IN.HYPOTHESIS
    assert "6h F1" in IN.FULL_RULE  # 6h F1 is screening only
    assert "3h" in IN.FULL_RULE  # F1 claimed only at 3h density
    assert "No freeze" in SPEC_SRC
    assert "official_score(" not in RUNNER_SRC and "core.freeze" not in RUNNER_SRC
    assert "official-score" in IN.FULL_RULE


def test_predictions_three_kinds_and_poison():
    assert "predictions_{dev,selection,audit}" in RUNNER_SRC  # output contract
    assert "write(te, pred, name)" in RUNNER_SRC  # per-period full-grid write
    assert "predictions_official" not in RUNNER_SRC
    assert "_reconstruct" in RUNNER_SRC and "anchors_for" in RUNNER_SRC  # grid reconstruction
    assert "cumsum" in RUNNER_SRC  # increments are rebuilt to the full grid
    assert "poison_check" in RUNNER_SRC and "positive_control" in RUNNER_SRC
    assert "anchor_invariant" in RUNNER_SRC
    assert "passed" in RUNNER_SRC and "poison.json" in RUNNER_SRC
    assert "SystemExit" in RUNNER_SRC  # fail-closed on poison/non-finite


def test_owned_dir_is_small():
    files = sorted(p.name for p in (LB / "experiments" / "d91_incr").iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"incr.py", "runner.py"} <= set(files)


def test_increments_transform_smoke_km_s_scale():
    """The preregistered increments target must split, reconstruct and vary
    on km/s-scale levels (synthetic, CPU, no data reads); guards a
    D39/E0089-style silent constant-head failure."""
    import numpy as np
    import xgboost as xgb

    rng = np.random.default_rng(11)
    n = 3000
    X = rng.standard_normal((n, 5))
    anchor = 400.0
    y = np.clip(400.0 + 50.0 * X[:, 0] + 10.0 * rng.standard_normal(n), 150.0, None)
    assert bool(np.isfinite(y).all())
    d = IN.to_deltas(y.reshape(-1), anchor).reshape(-1) if y.ndim == 1 else IN.to_deltas(y, anchor)
    assert np.isfinite(d).all()
    # per-origin roundtrip on a fake 72-step trajectory
    traj = np.linspace(380.0, 460.0, 72) + rng.standard_normal(72)
    assert np.allclose(IN.from_deltas(IN.to_deltas(traj, 370.0), 370.0), traj)
    p = IN.fit_params()
    m = xgb.XGBRegressor(**dict(p, n_estimators=100)).fit(X, d)
    pred_d = np.asarray(m.predict(X), dtype=float)
    pred = anchor + pred_d  # single-step analogue of the cumsum reconstruction
    assert np.isfinite(pred).all()
    assert float(np.std(pred)) > 0, "predictions must vary"
    dump = m.get_booster().get_dump(dump_format="text")
    assert sum(t.count("[f") for t in dump) > 0, "trees must split"
    null_mse = float(((y - y.mean()) ** 2).mean())
    assert float(((pred - y) ** 2).mean()) < null_mse, "must beat constant-mean"
