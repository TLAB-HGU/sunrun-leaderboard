"""D95 IMF-persist contract: hypothesis tags, unused-first, single-axis feat, isolation.

Fast static + unit checks only: no data reads, no fits beyond one small
synthetic CPU fit, no official contact. The pilot run itself (registered
separately) proves end-to-end behaviour through expos run/evaluate/poison.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d95_imf"))
sys.path.insert(0, str(LB / "experiments" / "d75_physics"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
sys.path.insert(0, str(LB))

import imffeatures as IM  # noqa: E402
import imfspec as PS  # noqa: E402
import info_ablation as ia  # noqa: E402
from expos import core  # noqa: E402

RUNNER_SRC = (LB / "experiments" / "d95_imf" / "runner.py").read_text()
SPEC_SRC = (LB / "experiments" / "d95_imf" / "imfspec.py").read_text()
FEAT_SRC = (LB / "experiments" / "d95_imf" / "imffeatures.py").read_text()

COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_hypothesis_dual_tag_and_experience():
    assert PS.HYPOTHESIS.startswith("[family=xgb] [axes: feat=imf-persist]")
    assert PS.EXPERIENCE == ["X0002", "X0043"]
    assert "X0002" in PS.HYPOTHESIS and "X0043" in PS.HYPOTHESIS
    assert "research/AXES.md" in PS.HYPOTHESIS and "feat" in PS.HYPOTHESIS
    assert PS.AXES_QUOTE in PS.HYPOTHESIS  # AXES.md feat-section quote
    assert "IMF Bz/By sign persistence and clock-angle" in PS.HYPOTHESIS
    assert PS.DIRECTION == "D95" and PS.OWNER == "worker-d95"
    assert "6h" in PS.HYPOTHESIS  # pilot grid stated
    assert "Unused" in PS.HYPOTHESIS or "unused" in PS.HYPOTHESIS
    assert "No freeze" in PS.HYPOTHESIS
    assert "official-score" in PS.HYPOTHESIS or "official_score" in PS.HYPOTHESIS


def test_imf_persist_unused_first():
    """Bz/By sign-persistence + clock-angle must be absent from every prior input family."""
    ace_src = (LB / "experiments" / "headroom_audit" / "headroom.py").read_text()
    low = ace_src.lower()
    assert "clock" not in low and "persist" not in low and "frac" not in low
    phy_src = (LB / "experiments" / "d75_physics" / "phyfeatures.py").read_text()
    assert "clock" not in phy_src.lower() and "persist" not in phy_src.lower()
    import physpec  # noqa: E402  (adds d75_physics via path below)
    assert set(physpec.PHY_COLS) & set(PS.IMF_COLS) == set()
    shock_src = (LB / "experiments" / "d7_shock_precursor" / "shock.py").read_text()
    assert "clock" not in shock_src.lower() and "persist" not in shock_src.lower()
    for c in PS.IMF_COLS:
        assert c not in shock_src, f"duplicate imf col in D7 shock: {c}"
    assert len(PS.IMF_COLS) == 8 and len(set(PS.IMF_COLS)) == 8


def test_single_axis_feat_only():
    p = PS.fit_params()
    assert p["device"] == "cpu"
    assert p.get("objective", "reg:squarederror") == "reg:squarederror"
    assert "tweedie_variance_power" not in p
    # E0001 recipe untouched except infra (feat change is data-side only)
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0001 recipe"
    assert p["n_estimators"] == 600 and p["max_depth"] == 6
    assert "base_score" not in p, "start point stays the E0001 default (one axis only)"
    assert "sample_weight" not in RUNNER_SRC
    assert type(__import__("xgboost").XGBRegressor(**p)).__name__ == "XGBRegressor"


def test_e0001_twin_rows_purge_budget():
    assert PS.PURGE_HOURS == 72
    assert PS.FREQ_PILOT == "6h" and PS.FREQ_FULL == "3h"
    assert PS.N_ESTIMATORS == 600 and PS.SEED == 0
    assert 'ia.arms(df.columns)["existing_all"]' in RUNNER_SRC
    assert "PS.PURGE_HOURS" in RUNNER_SRC
    assert "PS.IMF_COLS" in RUNNER_SRC
    assert RUNNER_SRC.count(".fit(") == 1, "exactly one head fit (single head, no refit)"
    assert PS.check_spec()["axes"] == "[axes: feat=imf-persist]"
    assert PS.check_spec()["experience"] == ["X0002", "X0043"]


def test_cpu_budget_no_accelerator():
    assert PS.PERIOD_JOBS * PS.FIT_NJOBS <= 10
    assert PS.PERIOD_JOBS == 3, "three periods run in parallel"
    assert PS.FIT_NJOBS <= 2
    assert PS.fit_params()["device"] == "cpu"
    blob = RUNNER_SRC + "\n" + SPEC_SRC + "\n" + FEAT_SRC
    for pat in ("cuda", "gpu_lease", "--gpu", "prefer-gpu", "EXPOS_GPUS",
                "CUDA_VISIBLE_DEVICES", "device_for"):
        assert pat not in blob, pat
    assert "CPU" in RUNNER_SRC  # CPU-only guard documented
    assert "D93" in PS.CPU_REASON and "D94" in PS.CPU_REASON  # lease reason recorded


def test_scratch_isolation():
    assert "store/scratch/worker-d95" in RUNNER_SRC
    others = [l for l in RUNNER_SRC.splitlines() if "worker-" in l and "worker-d95" not in l]
    assert others == [], f"scratch leak: {others[:3]}"
    assert "featcache" in RUNNER_SRC and "fcache_key" in RUNNER_SRC


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
    files = sorted(p.name for p in (LB / "experiments" / "d95_imf").iterdir() if p.suffix == ".py")
    assert len(files) <= 5, files
    assert {"imfspec.py", "imffeatures.py", "runner.py"} <= set(files)


def _synth_frame(seed=7, n=300):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "ace_bz_gsm_nt": rng.normal(-1.0, 4.0, n),
        "ace_by_gsm_nt": rng.normal(0.0, 4.0, n),
        "ace_bt_nt": rng.uniform(3.0, 12.0, n)}, index=idx)


def test_causality_future_invariant_past_sensitive():
    """Trailing-only proof on synthetic data: post-origin hours must not move an origin row."""
    fr = _synth_frame()
    o = pd.DatetimeIndex([fr.index[150]])
    base = IM.origin_imf(o, fr).to_numpy(float)
    assert np.isfinite(base).all()
    fut = fr.copy()
    fut.iloc[151:] = np.nan  # erase the entire future
    assert np.allclose(IM.origin_imf(o, fut).to_numpy(float), base, equal_nan=True)
    fut2 = fr.copy()
    fut2.iloc[151:] = 999.0  # extreme future values
    assert np.allclose(IM.origin_imf(o, fut2).to_numpy(float), base, equal_nan=True)
    past = fr.copy()
    past.iloc[:150] = 50.0  # change the past
    assert not np.allclose(IM.origin_imf(o, past).to_numpy(float), base, equal_nan=True)


def test_imf_features_vary_and_fit_smoke():
    """8 frozen cols vary on synthetic data and a small CPU head fits and splits."""
    import xgboost as xgb

    fr = _synth_frame(n=600)
    per = IM.origin_imf(fr.index[100:], fr)
    assert list(per.columns) == list(PS.IMF_COLS)
    assert all(per[c].nunique() > 10 for c in PS.IMF_COLS if c not in ("imf_miss24", "bz_run_24h",
                                                                        "bz_neg_frac_6h"))
    assert per["bz_neg_frac_6h"].nunique() > 5  # k/6 fractions vary
    assert per["bz_run_24h"].nunique() > 3  # integer run length varies
    X = per.fillna(per.median()).to_numpy(float)
    rng = np.random.default_rng(3)
    y = 400.0 + 30.0 * X[:, 0] + 20.0 * X[:, 4] + 5.0 * rng.standard_normal(len(X))
    p = PS.fit_params()
    m = xgb.XGBRegressor(**dict(p, n_estimators=50)).fit(X, y)
    pred = np.asarray(m.predict(X), dtype=float)
    assert np.isfinite(pred).all() and float(np.std(pred)) > 0, "predictions must vary"
    dump = m.get_booster().get_dump(dump_format="text")
    assert sum(t.count("[f") for t in dump) > 0, "trees must split"
    null_mse = float(((y - y.mean()) ** 2).mean())
    assert float(((pred - y) ** 2).mean()) < null_mse, "must beat constant-mean"
