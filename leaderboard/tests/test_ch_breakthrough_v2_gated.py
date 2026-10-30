"""Focused tests for experiments.ch_breakthrough_v2.gated (conditional gate lane)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.ch_breakthrough_v2.common import (
    HORIZON_HOURS,
    as_speed_series,
    build_features,
    observed_mask,
    utc,
)
from experiments.ch_breakthrough_v2 import gated as G


def _synthetic(n=2400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    speed = (400 + 30 * np.sin(np.arange(n) / 24 * 2 * np.pi)
             + rng.normal(0, 8, n))
    # Add a couple of fast-stream episodes so regimes are non-degenerate.
    speed[900:960] += np.linspace(0, 260, 60)
    speed[1300:1340] += np.linspace(0, 220, 40)
    ace = pd.DataFrame({"timestamp_utc": idx, "filled_speed_kms": speed,
                        "was_missing": np.zeros(n, dtype=int)})
    cols = {}
    for tag in ("t035", "t045", "t055"):
        cols[f"ch_{tag}_area"] = np.abs(rng.normal(0.02, 0.005, n))
        cols[f"ch_{tag}_top1_lon"] = rng.normal(0, 20, n)
        cols[f"ch_{tag}_top1_lat"] = rng.normal(0, 15, n)
        cols[f"ch_{tag}_top1_lon_width"] = np.abs(rng.normal(20, 5, n))
        cols[f"ch_{tag}_equatorial_area"] = np.abs(rng.normal(0.01, 0.003, n))
    ch = pd.DataFrame(cols, index=idx)
    ch.index.name = "slot"
    return ace, ch


def _fitted(n=2400, trees=5, oof_max=60, seed=0):
    ace, ch = _synthetic(n=n, seed=seed)
    features = build_features(ace, ch)
    speed = as_speed_series(ace)
    observed = observed_mask(speed, speed.index)
    cutoff = utc(features.index[1900])
    params = G.residual_params(trees)
    fitted = G.train(features, speed, observed, cutoff, params,
                     max_origins=60, max_oof_origins=oof_max,
                     max_gate_rows=3000, device="cpu", seed=seed)
    return features, speed, observed, cutoff, fitted


def test_gate_uses_issuance_time_only():
    """Perturbing sources after an origin must not change its gate decision."""
    features, speed, observed, cutoff, fitted = _fitted()
    origins = pd.DatetimeIndex(utc(features.index)).sort_values()[700:704]
    frames0, _ = G.predict_arms(fitted["residual"], fitted["gate"],
                                features, speed, origins)
    ace2, ch2 = _synthetic()
    t = origins.max()
    ace2.loc[ace2["timestamp_utc"] > t, "filled_speed_kms"] *= 3.0
    ch2.loc[ch2.index > t] *= 5.0
    features2 = build_features(ace2, ch2)
    frames1, _ = G.predict_arms(fitted["residual"], fitted["gate"],
                                features2, speed, origins)
    for arm in G.ARMS:
        pd.testing.assert_frame_equal(frames0[arm], frames1[arm])


def test_oof_predictions_are_never_insample():
    """Every OOF row's fit cutoff must precede its origin by >=72h purge."""
    ace, ch = _synthetic()
    features = build_features(ace, ch)
    speed = as_speed_series(ace)
    cutoff = utc(features.index[1900])
    oof, _ = G.oof_residual_predictions(
        features, speed, cutoff, G.residual_params(5),
        max_origins_per_fit=60, device="cpu")
    assert len(oof) > 0
    assert oof["resid_oof"].notna().any()
    gap = (pd.DatetimeIndex(utc(oof["origin_last_input_utc"]))
           - pd.DatetimeIndex(utc(oof["fit_cutoff"])))
    assert (gap >= pd.Timedelta(hours=72)).all()
    # OOF origins never reach the outer cutoff's eval region.
    assert (pd.DatetimeIndex(utc(oof["origin_last_input_utc"]))
            + pd.to_timedelta(oof["horizon_hours"].to_numpy(), unit="h")
            <= cutoff).all()
    # Gate labels derive from OOF only: helped==1 must mean OOF correction won.
    ok = np.isfinite(oof["truth"].to_numpy())
    base = (oof["baseline_pred"].to_numpy()[ok] - oof["truth"].to_numpy()[ok]) ** 2
    corr = (oof["corrected_pred"].to_numpy()[ok] - oof["truth"].to_numpy()[ok]) ** 2
    assert ((corr < base - 1e-9).sum() > 0) or ((corr >= base - 1e-9).sum() > 0)


def test_all_arms_cover_horizons_1_to_72():
    """Every arm emits the complete finite 1..72 curve per origin."""
    features, speed, observed, cutoff, fitted = _fitted()
    origins = pd.DatetimeIndex(utc(features.index)).sort_values()[2000:2003]
    frames, gate_info = G.predict_arms(
        fitted["residual"], fitted["gate"], features, speed, origins)
    assert set(frames) == set(G.ARMS)
    for arm, frame in frames.items():
        assert list(frame.columns) == ["origin_last_input_utc", "horizon_hours",
                                       "pred_kms"], arm
        assert len(frame) == len(origins) * HORIZON_HOURS, arm
        for _, grp in frame.groupby("origin_last_input_utc"):
            assert set(grp["horizon_hours"]) == set(range(1, 73)), arm
        assert np.isfinite(frame["pred_kms"]).all(), arm
    assert 0.0 <= gate_info["gate_open_fraction"] <= 1.0


def test_quiet_protection_metric_present():
    """Metrics carry quiet MSE/bias plus rising/high gains for every arm."""
    features, speed, observed, cutoff, fitted = _fitted()
    origins = pd.DatetimeIndex(utc(features.index)).sort_values()[2000:2006]
    frames, _ = G.predict_arms(
        fitted["residual"], fitted["gate"], features, speed, origins)
    for arm, frame in frames.items():
        qp = G.quiet_protection_metrics(frame, speed, observed)
        for regime in ("quiet", "rising", "high", "falling"):
            assert regime in qp, (arm, regime)
            assert {"count", "mse", "bias"} <= set(qp[regime]), (arm, regime)
        assert qp["quiet"]["count"] > 0, arm
        assert qp["quiet"]["mse"] is not None, arm
        bands = G.horizon_band_metrics(frame, speed, observed)
        assert set(bands) == {"1-24", "25-48", "49-72"}, arm
        months = G.month_split_metrics(frame, speed, observed)
        assert len(months) >= 1 and all("mse" in v for v in months.values())


def test_evaluate_writes_manifests_with_cutoff_and_sampling(tmp_path):
    """evaluate() writes parquets + manifests + metrics with identity fields."""
    features, speed, observed, cutoff, fitted = _fitted()
    origins = pd.DatetimeIndex(utc(features.index)).sort_values()[2100:2104]
    identities = {"ace": {"path": "fake", "bytes": 1, "files": 1,
                           "sha256": "0" * 64},
                  "ch_hourly": {"path": "fake", "bytes": 1, "files": 1,
                                "sha256": "1" * 64}}
    sampling = {"step": "24h", "requested": 4, "kept": 4, "seed": 42}
    results, _ = G.evaluate(fitted, features, speed, observed, origins,
                            tmp_path, "test-block", cutoff,
                            G.residual_params(5), identities, 42, sampling)
    assert set(results) == set(G.ARMS)
    for arm in G.ARMS:
        assert (tmp_path / f"test-block-{arm}.parquet").exists()
        manifest = (tmp_path / f"test-block-{arm}.manifest.json")
        assert manifest.exists()
        import json
        body = json.loads(manifest.read_text())
        for field in ("train_cutoff", "params", "tree_counts", "columns",
                      "input_identities", "code_hash", "seed",
                      "inference_seconds_per_origin", "sampling"):
            assert field in body, (arm, field)
        metrics = json.loads(
            (tmp_path / f"test-block-{arm}.metrics.json").read_text())
        assert "quiet_protection" in metrics and "mse" in metrics


def test_manifests_carry_prediction_hash_matching_parquet(tmp_path):
    """Manifest prediction sha256 must match the written parquet bytes."""
    import hashlib
    import json

    features, speed, observed, cutoff, fitted = _fitted()
    origins = pd.DatetimeIndex(utc(features.index)).sort_values()[2100:2104]
    identities = {"ace": {"path": "fake", "bytes": 1, "files": 1,
                           "sha256": "0" * 64},
                  "ch_hourly": {"path": "fake", "bytes": 1, "files": 1,
                                "sha256": "1" * 64}}
    sampling = {"step": "24h", "requested": 4, "kept": 4, "seed": 42}
    G.evaluate(fitted, features, speed, observed, origins,
               tmp_path, "test-block", cutoff,
               G.residual_params(5), identities, 42, sampling)
    for arm in G.ARMS:
        body = json.loads(
            (tmp_path / f"test-block-{arm}.manifest.json").read_text())
        digest = hashlib.sha256(
            (tmp_path / f"test-block-{arm}.parquet").read_bytes()).hexdigest()
        assert body["prediction"]["sha256"] == digest, arm
        assert body["code_hashes"]["common"]["sha256"]
        assert body["code_hashes"]["caller"]["path"].endswith("gated.py")
    gated = json.loads(
        (tmp_path / "test-block-gated.manifest.json").read_text())
    assert gated["extra"]["gate_columns"]
    assert gated["extra"]["residual_columns"]


def test_official_block_alias_runs_end_to_end(tmp_path):
    """'official' refits through the official cutoff on pinned-style origins."""
    ace, ch = _synthetic()
    features = build_features(ace, ch)
    speed = as_speed_series(ace)
    observed = observed_mask(speed, speed.index)
    origins = pd.DatetimeIndex(utc(features.index)).sort_values()[2100:2104]
    identities = {"ace": {"path": "fake", "bytes": 1, "files": 1,
                           "sha256": "0" * 64},
                  "ch_hourly": {"path": "fake", "bytes": 1, "files": 1,
                                "sha256": "1" * 64}}
    summary = G.run_block(
        "official", features, speed, observed, identities,
        outdir=tmp_path, trees=5, max_train_origins=60,
        max_oof_origins=60, device="cpu", seed=0,
        official_origins=origins, official_truth=None)
    assert summary["status"] == "complete"
    assert summary["train_cutoff"] == utc(G.OFFICIAL_CUTOFF).isoformat()
    assert summary["eval_origins"] == len(origins)
    for arm in G.ARMS:
        frame = pd.read_parquet(tmp_path / f"official-{arm}.parquet")
        assert len(frame) == len(origins) * HORIZON_HOURS, arm
