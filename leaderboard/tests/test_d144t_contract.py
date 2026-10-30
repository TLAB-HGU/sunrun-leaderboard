"""D144-turb contract: pure-helper + static gates for the turbulence-proxy pilot.

No real data, no fits, no registration, no official reads. Covers the turb
proxies (values, NaN handling, causality), the surrogate null control, the
recipe constants (CPU, jobs budget, cutoff, grid, depth, seeds, pilot kind,
E0421 reference, scratch isolation, axes tag), the 1-6h-only 3-arm design
with 7-72h invariance, the stitch/blend helpers, and the runner's static
contract (output names, poison gate, forbidden-source absence).
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TURB_DIR = Path(__file__).resolve().parents[1] / "experiments" / "d144_axisB_physics" / "turb"
sys.path.insert(0, str(TURB_DIR))

import features_turb as FT  # noqa: E402
import lib_turb as DL  # noqa: E402
import spec_turb as SP  # noqa: E402

TURB_SOURCES = {f: (TURB_DIR / f).read_text() for f in
                ("spec_turb.py", "features_turb.py", "lib_turb.py", "run_turb.py")}


def test_turb_dir_has_at_most_six_files():
    files = [p for p in TURB_DIR.iterdir()
             if p.is_file() and p.suffix == ".py" and not p.name.startswith("test")]
    assert len(files) <= 6, [p.name for p in files]
    assert {p.name for p in files} == {"spec_turb.py", "features_turb.py",
                                      "lib_turb.py", "run_turb.py"}


def _ace_frame(n=120, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "ace_speed_kms": 400 + 30 * rng.standard_normal(n),
        "ace_density_cm3": 5 + rng.standard_normal(n),
        "ace_bt_nt": 5 + rng.standard_normal(n),
        "ace_bz_gsm_nt": rng.standard_normal(n)}, index=idx)


def test_roll_std_known_value():
    s = pd.Series([1.0, 2, 3, 4, 5, 6])
    assert FT.roll_std(s, 6, 4).iloc[-1] == pytest.approx(1.8708286933875112)


def test_roll_skew_symmetric_is_zero():
    s = pd.Series([1.0, 2, 3, 4, 5])
    assert FT.roll_skew(s, 5, 3).iloc[-1] == pytest.approx(0.0)


def test_rolling_preserves_nan_until_min_periods():
    s = pd.Series([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0])
    assert FT.roll_std(s, 6, 4).iloc[:3].isna().all()
    assert np.isfinite(FT.roll_std(s, 6, 4).iloc[-1])
    long = pd.Series(np.arange(1.0, 31.0))
    assert FT.roll_kurt(long, 24, 10).iloc[:9].isna().all()
    assert FT.roll_kurt(long, 24, 10).iloc[23:].notna().all()


def test_compute_turb_is_causal_on_synthetic_frame():
    ace = _ace_frame()
    base = FT.compute_turb(ace)
    assert list(base.columns) == list(SP.TURB_COLUMNS)
    pert = ace.copy()
    pert.iloc[100:, 0] = pert.iloc[100:, 0] + 500.0
    pert.iloc[100:, 1:] = pert.iloc[100:, 1:] + 50.0
    mov = FT.compute_turb(pert)
    early = slice(0, 80)
    b, m = base.iloc[early].to_numpy(), mov.iloc[early].to_numpy()
    assert (np.isnan(b) == np.isnan(m)).all()
    with np.errstate(invalid="ignore"):
        assert np.nanmax(np.abs(b - m)) <= 1e-9
    late = slice(100, 120)
    assert np.nanmax(np.abs(base.iloc[late].to_numpy() - mov.iloc[late].to_numpy())) > 1e-3


def test_turb_group_is_single_and_small():
    assert SP.N_TURB_ADDED == 8 and len(FT.turb_columns()) == 8
    assert len(set(FT.turb_columns())) == 8
    assert list(FT.turb_columns()) == list(SP.TURB_COLUMNS)
    for c in SP.TURB_COLUMNS:
        m = re.fullmatch(r"turb_(v|dens|bt|bz)_(std|skew|kurt)(6|12|24)", c)
        assert m, c
        assert 6 <= int(m.group(3)) <= 24
    vars_used = {re.fullmatch(r"turb_(.*)_(std|skew|kurt).*", c).group(1) for c in SP.TURB_COLUMNS}
    assert vars_used == {"v", "dens", "bt", "bz"}


def test_noise_surrogate_mirrors_turb_and_is_deterministic():
    assert len(SP.NOISE_COLUMNS) == 8 == len(set(SP.NOISE_COLUMNS))
    assert list(FT.noise_columns()) == list(SP.NOISE_COLUMNS)
    for t, n in zip(SP.TURB_COLUMNS, SP.NOISE_COLUMNS):
        assert n == "noise_" + t[len("turb_"):]
    scales = {c: 2.0 for c in SP.NOISE_COLUMNS}
    a = FT.surrogate_block(500, scales, 1000)
    b = FT.surrogate_block(500, scales, 1000)
    c = FT.surrogate_block(500, scales, 1001)
    assert a.equals(b)
    assert not a.equals(c)
    assert list(a.columns) == list(SP.NOISE_COLUMNS)


def test_recipe_and_pilot_registration_contract():
    assert SP.REFERENCE == "E0421" and SP.KIND == "pilot"
    assert SP.DEPTH == 4 and SP.BASE_PARAMS["max_depth"] == 4
    assert SP.DEVICE == "cpu" and SP.BASE_PARAMS["device"] == "cpu"
    assert SP.FIT_NJOBS <= 20 and SP.BASE_PARAMS["n_jobs"] == SP.FIT_NJOBS
    assert SP.BASE_PARAMS["n_estimators"] == 600
    assert SP.CUTOFF_ISO < "2026-05-29T00:00:01Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24
    assert SP.JUDGED_ARM == "turb" and tuple(SP.ARMS) == ("base", "turb", "noise")


def test_seeds_really_vary_the_fit():
    assert tuple(SP.SEEDS) == (0, 1, 2) and len(set(SP.SEEDS)) == 3
    blocks = [DL.level_seeds(s) for s in SP.SEEDS]
    assert len({seed for b in blocks for seed in b}) == 9
    events = [DL.event_seed(s) for s in SP.SEEDS]
    assert len(set(events)) == 3
    for s, ev in zip(SP.SEEDS, events):
        assert ev in DL.level_seeds(s)


def test_hypothesis_has_axes_tag_and_three_arm_claim():
    assert re.search(r"\[axes:\s*[^\]]+\]", SP.HYPOTHESIS, re.I)
    tags = SP.HYPOTHESIS.lower()
    assert "feat=turbulence-proxy" in tags
    for arm in ("arm base", "arm turb", "arm noise"):
        assert arm in tags


def test_scratch_isolation():
    assert SP.SCRATCH == "store/scratch/worker-t1b4"
    assert SP.OWNER == "worker-t1b4" and SP.DIRECTION == "D144"


def test_blend_weights_convex_and_pinned():
    DL.check_weights(SP.WEIGHTS)
    h = np.arange(1, 73)
    rng = np.random.default_rng(0)
    mem = {m: rng.normal(400, 100, 72) for m in DL.MEMBERS}
    out = DL.blend_per_lead(mem, SP.WEIGHTS, h)
    assert out.shape == (72,) and np.all(np.isfinite(out))
    lo, hi = min(v.min() for v in mem.values()), max(v.max() for v in mem.values())
    assert np.all((out >= lo) & (out <= hi))


def test_frame_schema():
    te = pd.DataFrame({"origin": pd.date_range("2026-06-01", periods=2, freq="h", tz="UTC").repeat(72),
                       "h": np.tile(np.arange(1, 73), 2)})
    fr = DL.frame_predictions(te, np.full(len(te), 400.0))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert len(fr) == 144
    assert (fr["horizon_hours"].to_numpy() == np.tile(np.arange(1, 73), 2)).all()


def test_runner_writes_judged_plus_arm_files_and_poison_gate():
    src = TURB_SOURCES["run_turb.py"]
    assert "predictions_official_s{s}.parquet" in src
    assert "arm_base_s{s}.parquet" in src and "arm_noise_s{s}.parquet" in src
    assert "arm_summary.json" in src and '"poison.json"' in src
    assert "refusing to present predictions as causal" in src
    assert "ST.poison_test" in src and "FT.poison_check" in src


def test_short_expert_only_and_late_invariance():
    src = TURB_SOURCES["run_turb.py"]
    assert "fit_short_models(tr_s2, cols_base" in src
    assert "fit(tr_long[cols_base]" in src
    assert "7-72h must be bit-invariant" in src
    assert "te[cols_t]" in src and "te_n[cols_n]" in src and "te[cols_base]" in src


def test_runner_touches_no_forbidden_source():
    blob = "\n".join(TURB_SOURCES.values())
    for banned in ("truth.parquet", "folds.parquet", "naive.parquet",
                   "official-predictions", "baseline-replay.parquet",
                   "official_check", "official-check",
                   "predictions_official_truth", "sunrun-lb-store",
                   "GONG", "ADAPT", "DONKI", "CME", "WSA"):
        assert banned not in blob, banned
    for mod in ("d141", "nn_", "expos_runners", "sun-nn"):
        assert f"import {mod}" not in blob, mod
    assert "--gpu" not in blob and "cuda" not in blob.lower()
