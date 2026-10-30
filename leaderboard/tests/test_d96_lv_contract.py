"""D96 level-shape contract: hypothesis tags, single-axis split, twin spec, isolation.

Fast static + unit checks only: no data reads, no fits beyond two small
synthetic CPU fits, no official contact. The pilot run itself (registered
separately) proves end-to-end behaviour through expos run/evaluate/poison.
"""
import re
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d96_lvshape"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
sys.path.insert(0, str(LB))

import lvspec as LV  # noqa: E402
import info_ablation as ia  # noqa: E402
from expos import core  # noqa: E402

RUNNER_SRC = (LB / "experiments" / "d96_lvshape" / "runner.py").read_text()
SPEC_SRC = (LB / "experiments" / "d96_lvshape" / "lvspec.py").read_text()

COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W",
             "import strips", "import lvlmean", "import stack", "import hsplit",
             "def mean_predictions", "def smooth_origin_matrix"]


def test_hypothesis_dual_tag_and_experience():
    assert LV.HYPOTHESIS.startswith("[family=xgb] [axes: target=level-shape]")
    assert LV.FAMILY_TAG == "[family=xgb]" and LV.AXES_TAG == "[axes: target=level-shape]"
    assert set(LV.EXPERIENCE) == {"X0001", "X0006"}
    assert "X0001" in LV.HYPOTHESIS and "X0006" in LV.HYPOTHESIS
    assert "research/AXES.md" in LV.HYPOTHESIS and "target" in LV.HYPOTHESIS
    assert "level + shape decomposition" in LV.HYPOTHESIS
    assert "pred(o,h)=mean_h L_head(o,h)+S_head(o,h)" in LV.HYPOTHESIS
    assert "L(o)=mean_h y(o,h)" in LV.HYPOTHESIS
    assert "E0001" in LV.HYPOTHESIS and "E0003" in LV.HYPOTHESIS
    assert LV.DIRECTION == "D96" and LV.OWNER == "worker-d96"
    assert "6h" in LV.HYPOTHESIS
    assert "3h" in LV.HYPOTHESIS
    assert "rc==0" in LV.HYPOTHESIS and "F1>=0.30" in LV.HYPOTHESIS
    assert "screening" in LV.HYPOTHESIS
    assert "gpu0" in LV.HYPOTHESIS
    assert "No freeze" in LV.HYPOTHESIS
    assert "official-score" in LV.HYPOTHESIS


def test_single_axis_level_shape_only():
    p = LV.fit_params("cpu")
    assert p["device"] == "cpu"
    assert LV.fit_params("cuda")["device"] == "cuda"
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
    assert LV.TRANSFORM == "level-shape"
    assert LV.DECOMP_EQ.startswith("pred(o,h)=")
    assert type(__import__("xgboost").XGBRegressor(**p)).__name__ == "XGBRegressor"


def test_e0001_twin_rows_purge_budget():
    assert LV.PURGE_H == 72
    assert LV.FREQ_PILOT == "6h" and LV.FREQ_FULL == "3h"
    assert LV.N_ESTIMATORS == 600 and LV.SEED == 0
    assert 'ia.arms(df.columns)["existing_all"]' in RUNNER_SRC
    assert "LV.PURGE_H" in RUNNER_SRC
    assert RUNNER_SRC.count(".fit(") == 2, "exactly two head fits (level + shape, no extra)"
    assert LV.check_spec()["transform"] == "level-shape"


def test_decomp_exact_and_smooth():
    import numpy as np
    import pandas as pd
    origins = pd.to_datetime(["2025-01-01"] * 72 + ["2025-01-02"] * 72, utc=True)
    y = np.concatenate([np.linspace(300.0, 600.0, 72), np.linspace(400.0, 500.0, 72)])
    tr = pd.DataFrame({"origin": origins, "y": y})
    lvl, shp = LV.decompose_frame(tr)
    assert np.allclose((lvl.to_numpy() + shp.to_numpy()), y), "L+S must reconstruct y"
    assert abs(float(shp.groupby(tr["origin"], observed=True).mean().abs().max())) < 1e-9
    assert np.allclose(LV.combine_predictions(
        lvl.to_numpy(), shp.to_numpy(), tr["origin"].to_numpy()), y)
    sm = LV.smooth_level(np.arange(144, dtype=float), tr["origin"].to_numpy())
    assert bool((sm[:72] == sm[0]).all() and (sm[72:] == sm[72]).all())


def test_gpu_budget_and_lease_reason():
    assert LV.PERIOD_JOBS * LV.FIT_NJOBS <= 10
    assert LV.PERIOD_JOBS == 3
    assert LV.FIT_NJOBS <= 2
    assert "cuda" in RUNNER_SRC
    assert "device_for" in RUNNER_SRC
    assert "D96" in LV.GPU_REASON and "gpu0" in LV.GPU_REASON
    assert "D98" in LV.GPU_REASON and "gpu1" in LV.GPU_REASON
    assert "D97" in LV.GPU_REASON and "CPU" in LV.GPU_REASON
    assert "60s" in LV.GPU_REASON and "--prefer-gpu 0" in LV.GPU_REASON
    assert "n_jobs<=20" in LV.GPU_REASON
    assert "foreground" in LV.GPU_REASON


def test_scratch_isolation():
    assert "store/scratch/worker-d96" in RUNNER_SRC
    others = [l for l in RUNNER_SRC.splitlines() if "worker-" in l and "worker-d96" not in l]
    assert others == [], f"scratch leak: {others[:3]}"
    assert LV.SCRATCH_NAME == "worker-d96"
    assert "featcache" in RUNNER_SRC and "fcache_key" in RUNNER_SRC


def test_readonly_admin_no_vendored_code_no_foreign_ids():
    for s in COPY_BANS:
        assert s not in RUNNER_SRC, f"runner vendors foreign code {s!r}"
        assert s not in SPEC_SRC, f"spec vendors foreign code {s!r}"
    ids = set(re.findall(r"E00\d\d", RUNNER_SRC))
    assert ids <= {"E0001", "E0003"}, f"runner names foreign experiment ids: {ids}"
    assert "predictions_official" not in RUNNER_SRC


def test_no_forbidden_paths_in_registered_code():
    assert core.scan(RUNNER_SRC + "\n" + SPEC_SRC) == []
    assert core.scan(LV.HYPOTHESIS) == []


def test_full_rule_preregistered_no_freeze_claim():
    assert "rc==0" in LV.FULL_RULE and "poison" in LV.FULL_RULE and "E0001" in LV.FULL_RULE
    assert "F1>=0.30" in LV.FULL_RULE
    assert "6h F1" in LV.FULL_RULE
    assert "3h" in LV.FULL_RULE
    assert "No freeze" in SPEC_SRC
    assert "official_score(" not in RUNNER_SRC and "core.freeze" not in RUNNER_SRC
    assert "official-score" in LV.FULL_RULE
    assert "E0003" in LV.FULL_RULE


def test_predictions_three_kinds_and_poison():
    assert "predictions_{dev,selection,audit}" in RUNNER_SRC
    assert "write(te, pred, name)" in RUNNER_SRC
    assert "poison_check" in RUNNER_SRC and "positive_control" in RUNNER_SRC
    assert "level_smooth_validated" in RUNNER_SRC
    assert "passed" in RUNNER_SRC and "poison.json" in RUNNER_SRC
    assert "SystemExit" in RUNNER_SRC


def test_owned_dir_is_small():
    files = sorted(p.name for p in (LB / "experiments" / "d96_lvshape").iterdir() if p.suffix == ".py")
    assert len(files) <= 5, files
    assert {"lvspec.py", "runner.py"} <= set(files)


def test_level_shape_heads_split_learn_and_vary():
    """Both preregistered heads must split, combine, vary and beat a constant
    (synthetic, CPU, no data reads); guards a silent constant-head failure."""
    import numpy as np
    import pandas as pd
    import xgboost as xgb

    rng = np.random.default_rng(11)
    n_orig, per = 50, 60
    n = n_orig * per
    X = rng.standard_normal((n, 5))
    oidx = np.repeat(np.arange(n_orig), per)
    y = np.clip(400.0 + 30.0 * oidx + 50.0 * X[:, 0] + 10.0 * rng.standard_normal(n), 150.0, None)
    assert bool((y > 0).all())
    origins = pd.to_datetime("2025-01-01", utc=True) + pd.to_timedelta(oidx, unit="h")
    tr = pd.DataFrame({"origin": origins, "y": y})
    lvl, shp = LV.decompose_frame(tr)
    assert np.allclose((lvl.to_numpy() + shp.to_numpy()), y)
    p = LV.fit_params("cpu")
    m_lvl = xgb.XGBRegressor(**dict(p, n_estimators=60)).fit(X, lvl.to_numpy())
    m_shp = xgb.XGBRegressor(**dict(p, n_estimators=60)).fit(X, shp.to_numpy())
    pred = LV.combine_predictions(np.asarray(m_lvl.predict(X)), np.asarray(m_shp.predict(X)),
                                  origins.to_numpy())
    assert np.isfinite(pred).all()
    assert bool((pred > 0).all()), "level+shape predictions must stay positive"
    assert float(np.std(pred)) > 0, "predictions must vary"
    for m in (m_lvl, m_shp):
        dump = m.get_booster().get_dump(dump_format="text")
        assert sum(t.count("[f") for t in dump) > 0, "both heads must split"
    null_mse = float(((y - y.mean()) ** 2).mean())
    assert float(((pred - y) ** 2).mean()) < null_mse, "two-head sum must beat constant-mean"
