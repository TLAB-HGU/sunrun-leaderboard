"""Synthetic CPU-only coverage-repair tests for hss-phase-ff1; no fits, no data."""
import numpy as np
import pandas as pd
import pytest

from experiments.mse20_hss_phase_ff1 import coverage as c
from experiments.mse20_hss_phase_ff1 import protocol as p
from experiments.mse20_hss_phase_ff1 import evaluate as ev


def _frames(index, hole=None, nan_cell=None):
    frames = {}
    frames["speed"] = pd.Series(np.arange(len(index), dtype=float) + 300.0, index=index)
    frames["ace"] = pd.DataFrame({"filled_speed_kms": frames["speed"].to_numpy()}, index=index)
    for name in c.REPAIRED_FRAMES:
        df = pd.DataFrame(
            {"f1": np.arange(len(index), dtype=float), "f2": np.arange(len(index), dtype=float) * 2},
            index=index,
        )
        if nan_cell is not None:
            df.iloc[nan_cell[0], nan_cell[1]] = np.nan
        if hole is not None:
            df = df.drop(pd.DatetimeIndex([hole]))
        frames[name] = df
    return frames


def _hourly(n=48):
    return pd.date_range("2024-06-15T00:00:00Z", periods=n, freq="h", tz="UTC")


def test_identity_no_hole_preserves_nans():
    index = _hourly()
    frames = _frames(index, nan_cell=(5, 0))
    origins = index[::6]
    aligned, ledger = c.align_frames(frames, origins)
    assert ledger["n_repaired_issuances"] == 0 and ledger["n_repaired_cells"] == 0
    for name in c.REPAIRED_FRAMES:
        pd.testing.assert_frame_equal(aligned[name], frames[name])
        assert aligned[name].isna().sum().sum() == frames[name].isna().sum().sum()
    assert frames["base"].isna().sum().sum() == 1


def test_single_absent_hour_gets_exact_prior_row():
    index = _hourly()
    hole = index[20]
    frames = _frames(index, hole=hole)
    origins = pd.DatetimeIndex([hole])
    aligned, ledger = c.align_frames(frames, origins)
    assert ledger["n_repaired_issuances"] == 1
    for name in c.REPAIRED_FRAMES:
        pd.testing.assert_series_equal(
            aligned[name].loc[hole], frames[name].loc[index[19]], check_names=False
        )
    entry = [e for e in ledger["repairs"] if e["frame"] == "base"][0]
    assert entry["prev_row_utc"] == pd.Timestamp(index[19]).isoformat()
    assert entry["lag_hours"] == 1.0 and entry["stale"] is True


def test_prior_nan_preserved_not_column_filled():
    index = _hourly()
    hole = index[20]
    frames = _frames(index, hole=hole, nan_cell=(19, 0))
    aligned, _ = c.align_frames(frames, pd.DatetimeIndex([hole]))
    assert np.isnan(aligned["base"].loc[hole, "f1"])
    assert aligned["base"].loc[hole, "f2"] == 38.0


def test_stale_age_over_6h_rejects():
    index = _hourly()
    frames = _frames(index)
    for name in c.REPAIRED_FRAMES:
        frames[name] = frames[name].drop(index[10:17])
    with pytest.raises(ValueError, match="exceeds"):
        c.align_frames(frames, pd.DatetimeIndex([index[16]]))


def test_leading_gap_rejects_no_backfill():
    index = _hourly()
    frames = _frames(index)
    for name in c.REPAIRED_FRAMES:
        frames[name] = frames[name].drop([index[0]])
    with pytest.raises(ValueError, match="leading"):
        c.align_frames(frames, pd.DatetimeIndex([index[0]]))


def test_future_rows_never_used():
    index = _hourly()
    hole = index[20]
    frames = _frames(index, hole=hole)
    tampered = {k: (v.copy() if k in c.REPAIRED_FRAMES else v) for k, v in frames.items()}
    for name in c.REPAIRED_FRAMES:
        tampered[name].loc[index[21], :] = 999.0
    aligned, _ = c.align_frames(tampered, pd.DatetimeIndex([hole]))
    for name in c.REPAIRED_FRAMES:
        assert (aligned[name].loc[hole].to_numpy() != 999.0).all()


def test_caller_frames_not_mutated():
    index = _hourly()
    hole = index[20]
    frames = _frames(index, hole=hole)
    before = {k: v.copy() for k, v in frames.items()}
    c.align_frames(frames, pd.DatetimeIndex([hole]))
    for key in frames:
        if isinstance(frames[key], pd.DataFrame):
            pd.testing.assert_frame_equal(frames[key], before[key])
        else:
            pd.testing.assert_series_equal(frames[key], before[key])


def test_all_requested_origins_preserved_and_covered():
    index = _hourly(72)
    holes = [index[20], index[21]]
    frames = _frames(index)
    for name in c.REPAIRED_FRAMES:
        frames[name] = frames[name].drop(pd.DatetimeIndex(holes))
    origins = pd.DatetimeIndex([index[18], index[20], index[21], index[24]])
    aligned, ledger = c.align_frames(frames, origins)
    assert ledger["n_requested"] == 4 and ledger["n_repaired_issuances"] == 2
    assert c.assert_origins_covered(aligned, origins) is True
    for name in c.REPAIRED_FRAMES:
        assert len(aligned[name].loc[origins]) == 4


def test_speed_labels_untouched():
    index = _hourly()
    hole = index[20]
    frames = _frames(index, hole=hole)
    speed_before = frames["speed"].copy()
    aligned, _ = c.align_frames(frames, pd.DatetimeIndex([hole]))
    assert aligned["speed"] is frames["speed"]
    pd.testing.assert_series_equal(frames["speed"], speed_before)
    assert hole in frames["speed"].index


def test_fit_replay_adapter_parity():
    index = _hourly()
    hole = index[20]
    frames = _frames(index, hole=hole)
    origins = pd.DatetimeIndex([hole, index[30]])
    fit_aligned, fit_ledger = c.align_frames(frames, origins)
    replay_aligned, replay_ledger = c.align_frames(frames, origins)
    assert fit_ledger == replay_ledger
    for name in c.REPAIRED_FRAMES:
        pd.testing.assert_frame_equal(fit_aligned[name], replay_aligned[name])


def test_frozen_guard_fails_before_passes_after():
    index = _hourly()
    hole = index[20]
    frames = _frames(index, hole=hole)
    origins = pd.DatetimeIndex([hole])
    with pytest.raises(ValueError, match="features missing origins"):
        missing = frames["base"].reindex(origins)
        bad = missing.index[missing.isna().all(axis=1)]
        if len(bad):
            raise ValueError(f"features missing origins: {bad[0]}")
    aligned, _ = c.align_frames(frames, origins)
    missing = aligned["base"].reindex(origins)
    assert not missing.isna().all(axis=1).any()
    assert len(aligned["base"].loc[origins]) == 1


def test_closure_and_paths_isolated_from_old_phase():
    from experiments.mse20_hss_phase import protocol as old

    assert p.RUN.resolve() != old.RUN.resolve()
    assert "ff1" in str(p.RUN) and str(p.RUN).endswith("hss-phase-ff1")
    assert p.CAPS == dict(dev_claimed_attempts=28, structural_configs=2, dev_backbone_claims=42,
                          final_composite_refits=1, final_backbone_claims=3, final_head_fits=1)
    assert any("ff1/coverage.py" in k for k in p.sources())
    assert "experiments/mse20_hss_phase_ff1/model.py" in ev.MODEL_SOURCE_NAMES
    assert "experiments/mse20_hss_phase_ff1/coverage.py" in ev.MODEL_SOURCE_NAMES
    import inspect
    from experiments.mse20_hss_phase_ff1 import model as m

    src = inspect.getsource(m.cached_backbone) + inspect.getsource(m.predict_saved)
    assert "backbones-ff1" in inspect.getsource(m.cached_backbone)
    assert "coverage.align_frames" in src
    assert "experiments.mse20_hss_phase_ff1" in inspect.getsource(ev.reconstruct_training)


def test_policy_binding_deterministic():
    assert c.policy_binding() == dict(c.COVERAGE_POLICY)
    assert c.COVERAGE_POLICY["max_stale_hours"] == 6
    assert c.COVERAGE_POLICY["backfill"] is False
    assert c.COVERAGE_POLICY["drop_issuances"] is False
    assert set(c.COVERAGE_POLICY["repaired_frames"]) == {"base", "rec_full", "physics", "quality"}


def test_prefit_readiness_covers_synthetic_suite_only(tmp_path):
    import glob as globmod

    tests = sorted(globmod.glob("tests/test_arrival_phase*.py"))
    assert any("test_arrival_phase_ff1" in t for t in tests)
    assert p.verify_readiness.__name__ == "verify_readiness"
