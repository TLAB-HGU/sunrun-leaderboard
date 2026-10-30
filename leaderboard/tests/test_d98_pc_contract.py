"""D98 polar-CH contract: hypothesis tags, unused-first, single-axis feat, isolation.

Fast static + unit checks only: no data reads, no fits beyond one small
synthetic CPU fit, no official contact. The pilot run itself (registered
separately) proves end-to-end behaviour through expos run/evaluate/poison.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d98_polarch"))
sys.path.insert(0, str(LB / "experiments" / "d69_chgeo"))
sys.path.insert(0, str(LB / "experiments" / "d42_litfeat"))
sys.path.insert(0, str(LB / "experiments" / "d75_physics"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
sys.path.insert(0, str(LB))

import pcfeatures as PC  # noqa: E402
import pcspec as PS  # noqa: E402
import info_ablation as ia  # noqa: E402
from expos import core  # noqa: E402

RUNNER_SRC = (LB / "experiments" / "d98_polarch" / "runner.py").read_text()
SPEC_SRC = (LB / "experiments" / "d98_polarch" / "pcspec.py").read_text()
FEAT_SRC = (LB / "experiments" / "d98_polarch" / "pcfeatures.py").read_text()

COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_hypothesis_dual_tag_and_experience():
    assert PS.HYPOTHESIS.startswith("[family=xgb] [axes: feat=polar-ch]")
    assert PS.FAMILY_TAG == "[family=xgb]"
    assert PS.AXES_TAG == "[axes: feat=polar-ch]"
    assert PS.EXPERIENCE == ["X0009", "X0031"]
    assert "X0009" in PS.HYPOTHESIS and "X0031" in PS.HYPOTHESIS
    assert "research/AXES.md" in PS.HYPOTHESIS and "feat" in PS.HYPOTHESIS
    assert PS.AXES_QUOTE in PS.HYPOTHESIS  # AXES.md feat-section quote
    assert "polar CH" in PS.HYPOTHESIS
    assert "E0001" in PS.HYPOTHESIS
    assert PS.DIRECTION == "D98" and PS.OWNER == "worker-d98"
    assert "6h" in PS.HYPOTHESIS  # pilot grid stated
    assert "gpu1" in PS.HYPOTHESIS
    assert "Unused" in PS.HYPOTHESIS or "unused" in PS.HYPOTHESIS
    assert "screening" in PS.HYPOTHESIS  # 6h F1 is screening only
    assert "No freeze" in PS.HYPOTHESIS
    assert "official-score" in PS.HYPOTHESIS or "official_score" in PS.HYPOTHESIS


def test_polar_ch_unused_first():
    """t055 per-hemisphere polar-core value must be absent from every prior input family."""
    head_src = (LB / "experiments" / "headroom_audit" / "headroom.py").read_text()
    low = head_src.lower()
    assert "polar" not in low and "asym" not in low and "frag" not in low
    assert "pch_" not in head_src
    for c in PS.PCH_COLS:
        assert c not in head_src, f"polar col already in headroom reference: {c}"
    import geospec  # noqa: E402  (adds d69_chgeo via path below)
    assert set(geospec.GEO_COLS) & set(PS.PCH_COLS) == set()
    assert geospec.THR != PS.THR, "threshold must differ from the D69 t035 form"
    import litfeat  # noqa: E402
    lit_cols = list(litfeat.LIT_ORIGIN_COLS) + list(litfeat.LIT_BRIGHT_COLS) + list(litfeat.LIT_TGT_COLS)
    assert set(lit_cols) & set(PS.PCH_COLS) == set()
    assert litfeat.CH_NS_THRESHOLD != f"ch_{PS.THR}", "threshold must differ from the D42 t045 form"
    phy_src = (LB / "experiments" / "d75_physics" / "phyfeatures.py").read_text()
    assert "pch_" not in phy_src and "polar" not in phy_src.lower()
    import physpec  # noqa: E402  (adds d75_physics via path below)
    assert set(physpec.PHY_COLS) & set(PS.PCH_COLS) == set()
    shock_src = (LB / "experiments" / "d7_shock_precursor" / "shock.py").read_text()
    for c in PS.PCH_COLS:
        assert c not in shock_src, f"duplicate polar col in D7 shock: {c}"
    assert len(PS.PCH_COLS) == 12 and len(set(PS.PCH_COLS)) == 12
    assert all(c.startswith("pch_") for c in PS.PCH_COLS)


def test_single_axis_feat_only():
    p = PS.fit_params("cuda")
    assert p["device"].startswith("cuda")
    assert p.get("objective", "reg:squarederror") == "reg:squarederror"
    assert "tweedie_variance_power" not in p
    # E0001 recipe untouched except infra (feat change is data-side only)
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0001 recipe"
    assert p["n_estimators"] == 600 and p["max_depth"] == 6
    assert p["random_state"] == 0
    assert "base_score" not in p, "start point stays the E0001 default (one axis only)"
    assert "sample_weight" not in RUNNER_SRC
    assert type(__import__("xgboost").XGBRegressor(**p)).__name__ == "XGBRegressor"


def test_e0001_twin_rows_purge_budget():
    assert PS.PURGE_HOURS == 72
    assert PS.FREQ_PILOT == "6h" and PS.FREQ_FULL == "3h"
    assert PS.N_ESTIMATORS == 600 and PS.SEED == 0
    assert 'ia.arms(df.columns)["existing_all"]' in RUNNER_SRC
    assert "PS.PURGE_HOURS" in RUNNER_SRC
    assert "PS.PCH_COLS" in RUNNER_SRC
    assert RUNNER_SRC.count(".fit(") == 1, "exactly one head fit (single head, no refit)"
    assert PS.check_spec()["axes"] == "[axes: feat=polar-ch]"
    assert PS.check_spec()["experience"] == ["X0009", "X0031"]


def test_gpu_budget_lease():
    assert PS.PERIOD_JOBS * PS.FIT_NJOBS <= 10
    assert PS.PERIOD_JOBS == 3, "three periods run in parallel"
    assert PS.FIT_NJOBS <= 2
    assert PS.fit_params("cuda")["device"].startswith("cuda")
    blob = RUNNER_SRC + "\n" + SPEC_SRC + "\n" + FEAT_SRC
    assert "device_for" in blob and "EXPOS_GPUS" in blob
    assert "PS.device_for" in RUNNER_SRC
    assert "cuda" in blob
    assert "gpu1" in PS.GPU_REASON and "60s" in PS.GPU_REASON
    assert "D96" in PS.GPU_REASON and "D97" in PS.GPU_REASON and "D98" in PS.GPU_REASON
    assert "foreground" in PS.GPU_REASON


def test_scratch_isolation():
    assert "store/scratch/worker-d98" in RUNNER_SRC
    others = [l for l in RUNNER_SRC.splitlines() if "worker-" in l and "worker-d98" not in l]
    assert others == [], f"scratch leak: {others[:3]}"
    assert "featcache" in RUNNER_SRC and "fcache_key" in RUNNER_SRC
    assert PS.SCRATCH_NAME == "worker-d98"


def test_readonly_admin_no_vendored_strips():
    for s in COPY_BANS:
        assert s not in RUNNER_SRC, f"runner vendors d1-strip code {s!r}"
        assert s not in SPEC_SRC, f"spec vendors d1-strip code {s!r}"
        assert s not in FEAT_SRC, f"features vendor d1-strip code {s!r}"
    assert "import strips" not in RUNNER_SRC  # E0001 twin needs no strips


def test_no_forbidden_paths_in_registered_code():
    assert core.scan(RUNNER_SRC + "\n" + SPEC_SRC + "\n" + FEAT_SRC) == []
    assert core.scan(PS.HYPOTHESIS) == []


def test_full_rule_preregistered_no_freeze_claim():
    assert "rc==0" in PS.FULL_RULE and "poison" in PS.FULL_RULE and "E0001" in PS.FULL_RULE
    assert "F1>=0.30" in PS.FULL_RULE and "No freeze" in PS.HYPOTHESIS
    assert "6h F1" in PS.FULL_RULE  # 6h F1 is screening only
    assert "3h" in PS.FULL_RULE  # F1 claimed only at 3h density
    assert "No freeze" in SPEC_SRC
    assert "official_score(" not in RUNNER_SRC and "core.freeze" not in RUNNER_SRC
    assert "official-score" in PS.FULL_RULE


def test_predictions_three_kinds_and_poison():
    assert "predictions_{dev,selection,audit}" in RUNNER_SRC  # output contract
    assert "write(te, pred, name)" in RUNNER_SRC  # per-period full-grid write
    assert "predictions_official" not in RUNNER_SRC
    assert "test_grid" in RUNNER_SRC and "prediction_invariant" in RUNNER_SRC
    assert "poison_check" in RUNNER_SRC and "positive_control" in RUNNER_SRC
    assert "passed" in RUNNER_SRC and "poison.json" in RUNNER_SRC
    assert "SystemExit" in RUNNER_SRC  # fail-closed on poison/non-finite


def test_owned_dir_is_small():
    files = sorted(p.name for p in (LB / "experiments" / "d98_polarch").iterdir() if p.suffix == ".py")
    assert len(files) <= 5, files
    assert {"pcspec.py", "pcfeatures.py", "runner.py"} <= set(files)


def _synth_bundle(seed=7, n=400):
    rng = np.random.default_rng(seed)
    slots = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    grid = np.abs(rng.normal(2.0, 3.0, (n, 24)))
    grid[rng.random((n, 24)) < 0.05] = np.nan
    total = np.abs(rng.normal(60.0, 20.0, n))
    lat1 = np.abs(rng.normal(3.0, 2.0, (n, 12)))
    return slots, grid, total, lat1


def test_causality_future_invariant_past_sensitive():
    """Trailing-only proof on synthetic data: post-origin slots must not move an origin row."""
    slots, grid, total, lat1 = _synth_bundle()
    o = pd.DatetimeIndex([slots[250]])
    base = PC.origin_polar(o, slots, grid, total, lat1).to_numpy(float)
    assert np.isfinite(base[:, :9]).all()
    fut = grid.copy()
    fut[251:] = np.nan  # erase the entire future
    assert np.allclose(PC.origin_polar(o, slots, fut, total, lat1).to_numpy(float), base, equal_nan=True)
    fut2 = grid.copy()
    fut2[251:] = 999.0  # extreme future values
    assert np.allclose(PC.origin_polar(o, slots, fut2, total, lat1).to_numpy(float), base, equal_nan=True)
    past = grid.copy()
    past[:250] = 50.0  # change the past
    assert not np.allclose(PC.origin_polar(o, slots, past, total, lat1).to_numpy(float), base, equal_nan=True)


def test_polar_features_vary_and_fit_smoke():
    """12 frozen cols vary on synthetic data and a small CPU head fits and splits."""
    import xgboost as xgb

    slots, grid, total, lat1 = _synth_bundle(n=700)
    per = PC.origin_polar(slots[200:], slots, grid, total, lat1)
    assert list(per.columns) == list(PS.PCH_COLS)
    assert all(per[c].nunique() > 10 for c in PS.PCH_COLS if c != "pch_missing")
    X = per.fillna(per.median()).to_numpy(float)
    rng = np.random.default_rng(3)
    y = 400.0 + 30.0 * X[:, 0] + 20.0 * X[:, 5] + 5.0 * rng.standard_normal(len(X))
    p = dict(PS.fit_params("cuda"), device="cpu", n_estimators=50)
    m = xgb.XGBRegressor(**p).fit(X, y)
    pred = np.asarray(m.predict(X), dtype=float)
    assert np.isfinite(pred).all() and float(np.std(pred)) > 0, "predictions must vary"
    dump = m.get_booster().get_dump(dump_format="text")
    assert sum(t.count("[f") for t in dump) > 0, "trees must split"
    null_mse = float(((y - y.mean()) ** 2).mean())
    assert float(((pred - y) ** 2).mean()) < null_mse, "must beat constant-mean"
