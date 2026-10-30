"""D99 arrival-time contract: hypothesis tags, unused-first, single-axis target, isolation.

Fast static + unit checks only: no data reads, no fits beyond small synthetic
CPU fits, no official contact. The pilot run itself (registered separately)
proves end-to-end behaviour through expos run/evaluate/poison.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d99_arriv"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
sys.path.insert(0, str(LB))

import arspec as AR  # noqa: E402
import info_ablation as ia  # noqa: E402
from expos import core  # noqa: E402

RUNNER_SRC = (LB / "experiments" / "d99_arriv" / "runner.py").read_text()
SPEC_SRC = (LB / "experiments" / "d99_arriv" / "arspec.py").read_text()

COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_hypothesis_dual_tag_and_experience():
    assert AR.HYPOTHESIS.startswith("[family=xgb] [axes: target=arrival-time]")
    assert AR.FAMILY_TAG == "[family=xgb]" and AR.AXES_TAG == "[axes: target=arrival-time]"
    assert AR.EXPERIENCE == ("X0146", "X0147")
    assert "X0146" in AR.HYPOTHESIS and "X0147" in AR.HYPOTHESIS
    assert "research/AXES.md" in AR.HYPOTHESIS and "target" in AR.HYPOTHESIS
    assert AR.AXES_QUOTE in AR.HYPOTHESIS
    assert "arrival-time + amplitude parametrisation (event model)" in AR.HYPOTHESIS
    assert AR.LABEL_EQ in AR.HYPOTHESIS
    assert "pred(o,h)=clip(v0(o)+90*K(h-That(o)),200,900)" in AR.HYPOTHESIS
    assert "K(d)=exp(-0.5*(d/10)^2) for d<=0 else exp(-d/24)" in AR.HYPOTHESIS
    assert "E0001" in AR.HYPOTHESIS and "E0003" in AR.HYPOTHESIS
    assert AR.DIRECTION == "D99" and AR.OWNER == "worker-d99"
    assert "6h" in AR.HYPOTHESIS
    assert "gpu0" in AR.HYPOTHESIS
    assert "Unused" in AR.HYPOTHESIS or "unused" in AR.HYPOTHESIS
    assert "causal" in AR.HYPOTHESIS
    assert "screening" in AR.HYPOTHESIS
    assert "3h" in AR.HYPOTHESIS
    assert "rc==0" in AR.HYPOTHESIS and "F1>=0.30" in AR.HYPOTHESIS
    assert "No freeze" in AR.HYPOTHESIS
    assert "official-score" in AR.HYPOTHESIS


def test_arrival_label_unused_first():
    """Frozen single-hour arrival label must be absent from the E0001 label path."""
    ia_src = (LB / "experiments" / "headroom_audit" / "info_ablation.py").read_text()
    assert "sample_weight" not in ia_src
    for tok in ("arrival_labels", "That(o)", "smooth_arrival", "map_predictions", "K(d)", "K(h-"):
        assert tok not in ia_src, f"E0001 path already has arrival mapping {tok!r}"
    head_src = (LB / "experiments" / "headroom_audit" / "headroom.py").read_text()
    assert "arrival_labels" not in head_src and "smooth_arrival" not in head_src
    assert AR.THRESH == 500.0 and AR.AMP == 90.0 and AR.SIGMA == 10.0 and AR.TAU == 24.0
    assert (AR.V_MIN, AR.V_MAX) == (200.0, 900.0) and AR.V0_FALLBACK == 400.0
    assert AR.check_spec()["mapping"] == AR.MAPPING_EQ
    assert AR.check_spec()["axes"] == "[axes: target=arrival-time]"


def test_mapping_units():
    probe = pd.DataFrame(
        {"origin": pd.to_datetime(["2025-01-01"] * 4 + ["2025-01-02"] * 4, utc=True),
         "h": [1, 10, 20, 72, 1, 10, 20, 72],
         "y": [400.0, 520.0, 480.0, 410.0, 400.0, 410.0, 420.0, 430.0]})
    lab = AR.arrival_labels(probe)
    assert list(lab[:4]) == [10.0] * 4
    assert list(lab[4:]) == [72.0] * 4
    assert float(AR.kernel(0.0)) == 1.0
    assert abs(float(AR.kernel(-10.0)) - np.exp(-0.5)) < 1e-12
    assert abs(float(AR.kernel(24.0)) - np.exp(-1.0)) < 1e-12
    # asymmetric HSS-like bump: peak 1 at zero, monotone falloff on both sides
    assert float(AR.kernel(-5.0)) > float(AR.kernel(-24.0))  # rise side monotone
    assert float(AR.kernel(5.0)) > float(AR.kernel(24.0))  # decay side monotone
    that = AR.smooth_arrival(np.array([10.0, 12.0, 10.0, 12.0, 72.0, 72.0, 72.0, 72.0]),
                             probe["origin"].to_numpy())
    assert list(that[:4]) == [11.0] * 4 and list(that[4:]) == [72.0] * 4
    assert bool(((that >= 1.0) & (that <= 72.0)).all())
    mp = AR.map_predictions(np.array([12.0]), np.array([400.0]), np.array([12.0]))
    assert list(mp) == [490.0]
    mp2 = AR.map_predictions(np.array([12.0]), np.array([np.nan]), np.array([12.0]))
    assert list(mp2) == [490.0]
    far = AR.map_predictions(np.array([12.0]), np.array([400.0]), np.array([72.0]))
    assert bool((far >= 200.0).all() and (far <= 900.0).all())
    assert float(far[0]) < 490.0  # decayed far from the predicted arrival
    early = AR.map_predictions(np.array([72.0]), np.array([400.0]), np.array([1.0]))
    assert bool((early >= 200.0).all() and (early <= 900.0).all())
    # D90 contrast: frozen single-hour asymmetric form, not fitted (t0,amp,sigma)
    assert "amp * exp(-0.5 * ((h - t0) / sigma)" not in SPEC_SRC
    assert "AMP * kernel" in RUNNER_SRC or "AMP * kernel" in SPEC_SRC or "AMP *kernel" in (
        RUNNER_SRC + SPEC_SRC) or "AMP * kernel" in (RUNNER_SRC + SPEC_SRC).replace("  ", " ") \
        or "v0 + AMP" in (RUNNER_SRC + SPEC_SRC) or "v0+AMP" in (RUNNER_SRC + SPEC_SRC).replace(" ", "")


def test_single_axis_target_only():
    p = AR.fit_params("cuda")
    assert p["device"] == "cuda"
    assert p.get("objective", "reg:squarederror") == "reg:squarederror"
    assert "tweedie_variance_power" not in p
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k in ("n_jobs", "random_state"):
            continue
        assert p[k] == v, f"param {k} differs from E0001 recipe"
    assert p["n_estimators"] == 600 and p["max_depth"] == 6 and p["random_state"] == 0
    assert "base_score" not in p
    assert 'ia.arms(df.columns)["existing_all"]' in RUNNER_SRC
    assert '"A" not in cols' in RUNNER_SRC
    assert 'tr["A"]' in RUNNER_SRC or "tr['A']" in RUNNER_SRC
    assert "arrival_labels" in RUNNER_SRC and "smooth_arrival" in RUNNER_SRC
    assert "map_predictions" in RUNNER_SRC and "v_lag0" in RUNNER_SRC
    assert RUNNER_SRC.count(".fit(") == 1, "exactly one arrival head fit"
    assert type(__import__("xgboost").XGBRegressor(**p)).__name__ == "XGBRegressor"


def test_e0001_twin_rows_purge_budget():
    assert AR.PURGE_H == 72
    assert AR.FREQ_PILOT == "6h" and AR.FREQ_FULL == "3h"
    assert AR.N_ESTIMATORS == 600 and AR.SEED == 0
    assert "ia.build(freq=freq)" in RUNNER_SRC or "ia.build(freq" in RUNNER_SRC
    assert "AR.PURGE_H" in RUNNER_SRC
    assert "build_train" in RUNNER_SRC
    assert "speed_features" not in RUNNER_SRC  # ia.build owns features; runner must not re-derive
    assert AR.check_spec()["family"] == "[family=xgb]"
    assert list(AR.EXPERIENCE) == ["X0146", "X0147"]


def test_gpu_budget_no_cpu_only():
    assert AR.PERIOD_JOBS * AR.FIT_NJOBS <= 10
    assert AR.PERIOD_JOBS == 3
    assert AR.FIT_NJOBS <= 2
    assert AR.fit_params("cuda")["device"] == "cuda"
    assert AR.device_for(0) == "cpu" or True  # env-dependent; explicit below
    import os as _os
    cur = (_os.environ.get("EXPOS_GPUS"), _os.environ.get("CUDA_VISIBLE_DEVICES"))
    try:
        _os.environ.pop("EXPOS_GPUS", None)
        _os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        assert AR.device_for(0) == "cpu"
        _os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        assert AR.device_for(0) == "cuda"
    finally:
        _os.environ.pop("EXPOS_GPUS", None)
        _os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        if cur[0] is not None:
            _os.environ["EXPOS_GPUS"] = cur[0]
        if cur[1] is not None:
            _os.environ["CUDA_VISIBLE_DEVICES"] = cur[1]
    assert "cuda:1" not in RUNNER_SRC  # D99 is the gpu0 leg only
    assert "D99" in AR.GPU_REASON and "gpu0" in AR.GPU_REASON
    assert "D101" in AR.GPU_REASON and "gpu1" in AR.GPU_REASON
    assert "D100" in AR.GPU_REASON and "CPU" in AR.GPU_REASON
    assert "60s" in AR.GPU_REASON and "--prefer-gpu 0" in AR.GPU_REASON
    assert "n_jobs<=20" in AR.GPU_REASON and "foreground" in AR.GPU_REASON


def test_scratch_isolation():
    assert "store/scratch/worker-d99" in RUNNER_SRC
    others = [l for l in RUNNER_SRC.splitlines() if "worker-" in l and "worker-d99" not in l]
    assert others == [], f"scratch leak: {others[:3]}"
    assert "featcache" in RUNNER_SRC and "fcache_key" in RUNNER_SRC
    assert AR.SCRATCH_NAME == "worker-d99"


def test_readonly_admin_no_vendored_strips():
    for s in COPY_BANS:
        assert s not in RUNNER_SRC, f"runner vendors d1-strip code {s!r}"
        assert s not in SPEC_SRC, f"spec vendors d1-strip code {s!r}"
    assert "import strips" not in RUNNER_SRC


def test_no_forbidden_paths_in_registered_code():
    assert core.scan(RUNNER_SRC + "\n" + SPEC_SRC) == []
    assert core.scan(AR.HYPOTHESIS) == []


def test_outputs_poison_selection_rules():
    for tok in ('predictions_{dev,selection,audit}', "poison.json", '"passed"',
                "positive_control", "nan_mask", "pred_invariant",
                "arrival_range", "pred_range", "selection.json", "importance.json"):
        assert tok in RUNNER_SRC, f"runner must implement {tok}"
    assert "dev only" in AR.SELECT_RULE and "selection/audit" in AR.SELECT_RULE
    assert "prior prediction files never read" in AR.SELECT_RULE
    assert "rc==0" in AR.FULL_RULE and "E0001" in AR.FULL_RULE and "E0003" in AR.FULL_RULE
    assert "F1>=0.30" in AR.FULL_RULE and "6h F1" in AR.FULL_RULE
    assert "no freeze" in AR.FULL_RULE.lower() and "official-score" in AR.FULL_RULE
    blob = RUNNER_SRC + "\n" + SPEC_SRC
    assert "no other experiment" in blob.lower() or "prior prediction files are never read" in blob
    assert "research/expos/runs/E" not in blob
    assert "predictions_official" not in blob
    assert "official-score" not in RUNNER_SRC  # runner never scores official; spec states the ban
