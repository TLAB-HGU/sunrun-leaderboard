"""GPU HPO confirmation tests (Stage1, dev/selection only)."""
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.ch_breakthrough_v2 import common
from experiments.ch_breakthrough_v2.hpo import (
    SEARCH_SPLITS,
    SPLIT_CFG,
    candidate_grid,
    eval_origins,
)
from experiments.ch_breakthrough_v2.hpo_gpu import (
    is_event_feasible,
    source_hashes,
)

HPO_GPU = Path("store/ch-breakthrough-v2/validated/hpo-gpu")

EXPECTED_CUTOFFS = {
    "dev-A": "2024-12-31T23:00:00Z",
    "dev-B": "2025-04-30T23:00:00Z",
    "dev-C": "2025-08-31T23:00:00Z",
    "selection": "2025-11-30T23:00:00Z",
}


def _load_json(path):
    return json.loads(Path(path).read_text())


def test_cutoff_and_selection_boundaries():
    for split, cutoff in EXPECTED_CUTOFFS.items():
        assert SPLIT_CFG[split]["cutoff"] == cutoff
    assert SPLIT_CFG["selection"]["start"] == "2025-12-01T00:00:00Z"
    assert SPLIT_CFG["selection"]["end"] == "2026-02-28T23:00:00Z"
    # Purge >=72h and 6h eval grid evidenced in real manifests/parquets.
    for branch in ("recurrence", "physical"):
        for split in SEARCH_SPLITS:
            man = _load_json(HPO_GPU / f"{branch}_trial00_{split}.manifest.json")
            assert common.utc(man["train_cutoff"]) == common.utc(
                SPLIT_CFG[split]["cutoff"])
            assert man["sampling"]["step"] == "6h"
            assert man["sampling"]["kept"] <= 120
            assert man["extra"]["max_train_origins"] == 6000
            assert man["extra"]["train_cap"] == 6000
            assert man["extra"]["train_retained"] <= 6000
            assert man["extra"]["train_requested"] >= man["extra"]["train_retained"]
            frame = pd.read_parquet(HPO_GPU / f"{branch}_trial00_{split}.parquet")
            origins = pd.DatetimeIndex(common.utc(frame["origin_last_input_utc"])).unique()
            cutoff = common.utc(SPLIT_CFG[split]["cutoff"])
            assert (origins.min() - cutoff) >= pd.Timedelta(hours=72)
            assert set(frame["horizon_hours"]) == set(range(1, 73))
    # No holdout/official cutoffs anywhere in Stage1 artifacts; every
    # manifest must declare dev/selection-only use (which mentions the
    # forbidden words only to disclaim them).
    forbidden = {common.utc("2026-02-28T23:00:00Z"), common.utc("2026-05-31T23:00:00Z")}
    for man_path in glob.glob(str(HPO_GPU / "*.manifest.json")):
        man = _load_json(man_path)
        assert common.utc(man["train_cutoff"]) not in forbidden, man_path
        use = json.dumps(man.get("extra", {}).get("selection_use", "")).lower()
        alt = json.dumps(man.get("extra", {})).lower()
        assert ("official never used" in alt) or ("no holdout" in alt) or ("no_retune_after_official" in alt) or ("dev/selection only" in alt), man_path
        assert "official-ace" not in alt and "official-ch" not in alt, man_path


def test_deterministic_sampling_and_grid():
    g1, g2 = candidate_grid(18), candidate_grid(18)
    assert len(g1) == 18 == len(g2) and g1 == g2
    assert len({t["n_estimators"] for t in g1}) > 1
    idx = pd.date_range("2025-01-01", periods=30 * 24, freq="h", tz="UTC")
    k1, p1 = common.subsample_origins(idx, step="6h", max_origins=50, seed=42)
    k2, p2 = common.subsample_origins(idx, step="6h", max_origins=50, seed=42)
    assert k1.equals(k2) and p1 == p2 and p1["kept"] == 50
    # Same origins reused across trials: trial params identical across splits.
    for branch in ("recurrence", "physical"):
        params = []
        for split in SEARCH_SPLITS:
            man = _load_json(HPO_GPU / f"{branch}_trial00_{split}.manifest.json")
            params.append(man["params"])
        assert all(p == params[0] for p in params)
    # Synthetic eval grid determinism (no heavy reload).
    synth = pd.date_range("2025-01-01", "2025-03-31T23:00", freq="h", tz="UTC")
    e1, _ = eval_origins(synth, "dev-A", step="6h", max_origins=20, seed=42)
    e2, _ = eval_origins(synth, "dev-A", step="6h", max_origins=20, seed=42)
    assert e1.equals(e2)


def test_real_trial_count_evidence():
    summary = _load_json(HPO_GPU / "summary.json")
    for branch in ("recurrence", "physical"):
        info = summary["branches"][branch]
        assert info["n_trials_complete"] >= 16
        assert len(info["trials"]) == 18
        for trial in info["trials"]:
            if trial.get("status") != "complete":
                continue
            for split in SEARCH_SPLITS:
                assert split in trial["per_split"]
                assert np.isfinite(trial["per_split"][split]["mse"])
    # Per-trial files exist for every completed trial x 4 splits.
    for branch in ("recurrence", "physical"):
        info = summary["branches"][branch]
        for trial in info["trials"]:
            if trial.get("status") != "complete":
                continue
            t = trial["trial"]
            for split in SEARCH_SPLITS:
                for ext in ("parquet", "metrics.json", "manifest.json"):
                    assert (HPO_GPU / f"{branch}_trial{t:02d}_{split}.{ext}").is_file()
    # Spot-check all72 unique keys on two real parquets.
    for name in ("recurrence_trial00_dev-A", "physical_trial00_selection"):
        frame = pd.read_parquet(HPO_GPU / f"{name}.parquet")
        assert set(frame["horizon_hours"]) == set(range(1, 73))
        keys = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex(common.utc(frame["origin_last_input_utc"])),
             frame["horizon_hours"]])
        assert keys.is_unique and len(keys) == len(frame)
    frozen = _load_json(HPO_GPU / "frozen-choice.json")
    combined = _load_json(HPO_GPU / "combined-choice.json")
    for branch in ("recurrence", "physical"):
        entry = frozen["branches"][branch]
        assert entry["params"] is not None and entry["winner_trial"] is not None
        assert frozen["branches"][branch]["feasibility_label"] in (
            "event-feasible", "no-event-feasible")
    assert combined["params"] is not None and combined["picked_branch"] in (
        "recurrence", "physical")
    assert "dev/selection" in combined["param_basis"]
    assert "holdout" not in combined["param_basis"].lower() or "no holdout" in combined["param_basis"].lower()


def test_no_source_after_issuance():
    # Feature-level causality: post-origin sources never leak into the row.
    idx = pd.date_range("2024-01-01", periods=800, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    speed = 400 + 30 * np.sin(np.arange(800) / 24 * 2 * np.pi) + rng.normal(0, 5, 800)
    ace = pd.DataFrame({"timestamp_utc": idx, "filled_speed_kms": speed,
                        "was_missing": np.zeros(800, dtype=int)})
    cols = {}
    for tag in ("t035", "t045", "t055"):
        cols[f"ch_{tag}_area"] = np.abs(rng.normal(0.02, 0.005, 800))
        cols[f"ch_{tag}_top1_lon"] = rng.normal(0, 20, 800)
        cols[f"ch_{tag}_top1_lat"] = rng.normal(0, 15, 800)
        cols[f"ch_{tag}_top1_lon_width"] = np.abs(rng.normal(20, 5, 800))
        cols[f"ch_{tag}_equatorial_area"] = np.abs(rng.normal(0.01, 0.003, 800))
    ch = pd.DataFrame(cols, index=idx)
    ch.index.name = "slot"
    t = common.utc("2024-01-20T00:00:00Z")
    future = ace.copy()
    future.loc[future["timestamp_utc"] > t, "filled_speed_kms"] *= 3.0
    ch_future = ch.copy()
    ch_future.loc[ch_future.index > t] *= 5.0
    base = common.build_features(ace, ch)
    altered = common.build_features(future, ch_future)
    pd.testing.assert_frame_equal(base.loc[:t].iloc[[-1]], altered.loc[:t].iloc[[-1]])
    # Artifact-level: every eval origin needs a full 72h label window and
    # starts >=72h after the recorded fit cutoff.
    for branch in ("recurrence", "physical"):
        for split in SEARCH_SPLITS:
            man = _load_json(HPO_GPU / f"{branch}_trial00_{split}.manifest.json")
            cutoff = common.utc(man["train_cutoff"])
            frame = pd.read_parquet(HPO_GPU / f"{branch}_trial00_{split}.parquet")
            origins = pd.DatetimeIndex(common.utc(frame["origin_last_input_utc"])).unique()
            assert (origins.min() - cutoff) >= pd.Timedelta(hours=72)


def test_observed_mask():
    idx = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    speed = pd.Series([400.0, 410.0, np.nan, np.nan, 420.0, 430.0, np.nan, 440.0], index=idx)
    speed.attrs["observed"] = pd.Series(
        [True, True, False, False, True, True, False, True], index=idx)
    observed = common.observed_mask(speed, idx)
    assert observed.tolist() == [True, True, False, False, True, True, False, True]
    # Forward-filled truth stays non-observed: finite fallback differs.
    bare = pd.Series([400.0, np.nan, 410.0], index=idx[:3])
    assert common.observed_mask(bare, idx[:3]).tolist() == [True, False, True]
    # Real metrics carry observed support + genuinely-observed peak counts.
    m = _load_json(HPO_GPU / "recurrence_trial00_dev-A.metrics.json")
    assert "mse_observed" in m and "unique_observed_peak_timestamps" in m
    assert isinstance(m["unique_observed_peak_timestamps"], int)
    assert is_event_feasible({"unique_observed_peak_timestamps": 12,
                              "precision": 0.7, "recall": 0.65,
                              "height_mae_kms": 50.0}) is True
    assert is_event_feasible({"unique_observed_peak_timestamps": 9,
                              "precision": 0.9, "recall": 0.9,
                              "height_mae_kms": 10.0}) is False


def test_gpu_lease_and_device_evidence():
    with common.acquire_gpu_lease() as gpu:
        assert gpu in (0, 1)
        assert f"cuda:{gpu}" != "auto"
    import pytest
    with pytest.raises(Exception):
        common.ensure_device("cuda:99")
    devices = set()
    for man_path in glob.glob(str(HPO_GPU / "*trial*.manifest.json")):
        man = _load_json(man_path)
        actual = man.get("actual_device", "")
        assert str(actual).startswith("cuda"), man_path
        assert "auto" not in str(actual)
        devices.add(str(actual))
        assert man["extra"]["device_requested"].startswith("cuda")
        assert man["extra"]["lease_gpu"] in (0, 1)
        assert man["extra"]["threads_per_fit"] == 4
        assert man["extra"]["source_hashes"].keys() >= {
            "common", "hpo", "recurrence", "physical_quality", "hpo_gpu"}
        assert man["prediction"]["sha256"] and len(man["prediction"]["sha256"]) == 64
        assert man["inference_seconds_per_origin"] is not None
        if man.get("mode", "").startswith(("recurrence_trial", "physical_trial")):
            assert man["extra"]["fit_seconds"] is not None, man_path
        assert man["versions"]["xgboost"] and man["seed"] == 42
    assert "cuda:0" in devices and "cuda:1" in devices
    hashes = source_hashes()
    assert set(hashes) == {"common", "hpo", "recurrence", "physical_quality", "hpo_gpu"}
    for name, info in hashes.items():
        assert len(info["sha256"]) == 64
    # Prediction hash matches the real parquet bytes.
    man = _load_json(HPO_GPU / "recurrence_trial00_dev-A.manifest.json")
    digest = hashlib.sha256(
        (HPO_GPU / "recurrence_trial00_dev-A.parquet").read_bytes()).hexdigest()
    assert man["prediction"]["sha256"] == digest
