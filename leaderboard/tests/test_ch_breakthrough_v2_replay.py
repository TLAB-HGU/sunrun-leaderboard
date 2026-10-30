"""Stage2 replay acceptance: frozen winners replayed into validated/tuned-replay/.

Artifact-based (no refit) except one tiny synthetic CPU probe for the
future-invariance contract. Run with the fusion interpreter only:
.venv-fusion/bin/python -m pytest tests/test_ch_breakthrough_v2_replay.py -q
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPLAY_DIR = Path("store/ch-breakthrough-v2/validated/tuned-replay")
HPO_GPU_DIR = Path("store/ch-breakthrough-v2/validated/hpo-gpu")
MODELS = ("winner-recurrence", "winner-physical", "combined")
SPLITS = ("holdout", "official")
N_OFFICIAL_KEYS = 2833 * 72
TARGET_MSE_50PCT = 2783.156128712626
PINNED_SNAPSHOT = Path(
    "/home/t-lab01/.cache/huggingface/hub/datasets--tlabtlab"
    "--sunrun-lb-store/snapshots"
    "/bcf5d5417d99eaa331ddca40581d48953e09a58e")


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(model, split):
    path = REPLAY_DIR / f"{model}__{split}.metrics.json"
    assert path.is_file(), f"missing metrics: {path}"
    return json.loads(path.read_text())


def _manifest(model, split):
    path = REPLAY_DIR / f"{model}__{split}.manifest.json"
    assert path.is_file(), f"missing manifest: {path}"
    return json.loads(path.read_text())


def _preds(model, split):
    path = REPLAY_DIR / f"{model}__{split}.parquet"
    assert path.is_file(), f"missing predictions: {path}"
    return pd.read_parquet(path)


def test_official_complete_203976_keys_from_folds():
    folds = pd.read_parquet(PINNED_SNAPSHOT / "folds.parquet")
    expected_origins = pd.DatetimeIndex(
        pd.to_datetime(folds["origin_last_input_utc"], utc=True))
    assert len(expected_origins.unique()) == 2833
    for model in MODELS:
        frame = _preds(model, "official")
        assert len(frame) == N_OFFICIAL_KEYS, (model, len(frame))
        assert set(frame["horizon_hours"]) == set(range(1, 73)), model
        keys = pd.MultiIndex.from_arrays(
            [pd.to_datetime(frame["origin_last_input_utc"], utc=True),
             frame["horizon_hours"]])
        assert keys.is_unique, f"{model}: duplicate official keys"
        got_origins = pd.DatetimeIndex(
            pd.to_datetime(frame["origin_last_input_utc"],
                           utc=True)).unique().sort_values()
        assert got_origins.equals(expected_origins.sort_values().unique()), (
            model, "official origins differ from folds.parquet")
        assert np.isfinite(frame["pred_kms"].to_numpy()).all(), model
        assert ((frame["pred_kms"] > 0)
                & (frame["pred_kms"] <= 3000)).all(), model


def test_holdout_predictions_present_complete_and_purged():
    cutoff = pd.Timestamp("2026-02-28T23:00:00Z")
    end = pd.Timestamp("2026-05-31T23:00:00Z")
    for model in MODELS:
        frame = _preds(model, "holdout")
        assert set(frame["horizon_hours"]) == set(range(1, 73)), model
        origins = pd.DatetimeIndex(
            pd.to_datetime(frame["origin_last_input_utc"], utc=True))
        assert len(frame) == len(origins.unique()) * 72, model
        keys = pd.MultiIndex.from_arrays([origins, frame["horizon_hours"]])
        assert keys.is_unique, f"{model}: duplicate holdout keys"
        assert (origins.unique().min() - cutoff) >= pd.Timedelta(hours=72), (
            model, "purge gap violated")
        assert (origins.unique().max() + pd.Timedelta(hours=72)) <= end, (
            model, "holdout label window exceeds split end")
        manifest = _manifest(model, "holdout")
        assert manifest["train_cutoff"] == "2026-02-28T23:00:00+00:00", model
        manifest_off = _manifest(model, "official")
        assert (manifest_off["train_cutoff"]
                == "2026-05-31T23:00:00+00:00"), model


def test_frozen_params_unchanged():
    frozen = json.loads((HPO_GPU_DIR / "frozen-choice.json").read_text())
    combined = json.loads((HPO_GPU_DIR / "combined-choice.json").read_text())
    summary = json.loads((REPLAY_DIR / "summary.json").read_text())
    assert summary["frozen_sources_unchanged"] is True
    assert all(summary["frozen_source_match"].values())
    for name in ("frozen-choice", "combined-choice"):
        recorded = summary["frozen_file_hashes"][name]
        current = REPLAY_DIR.parent.joinpath(
            "hpo-gpu", f"{name}.json")
        assert recorded["sha256"] == _sha256(current), name
    frozen_unchanged = json.loads(
        (REPLAY_DIR / "frozen_unchanged.json").read_text())
    assert frozen_unchanged["frozen_sources_unchanged"] is True
    for model, branch in (("winner-recurrence", "recurrence"),
                          ("winner-physical", "physical")):
        expected = frozen["branches"][branch]["params"]
        for split in SPLITS:
            assert _metrics(model, split)["params"] == expected, (model, split)
            assert _manifest(model, split)["params"] == expected, (model, split)
    for split in SPLITS:
        assert _metrics("combined", split)["params"] == combined["params"], \
            split
    # Lane sources are byte-identical to the Stage1 frozen run.
    deps = json.loads(
        (REPLAY_DIR / "source_dependency_manifest.json").read_text())
    for key in ("common", "hpo", "hpo_gpu", "recurrence", "physical_quality"):
        assert deps["dependencies"][key]["sha256"] == \
            frozen["source_hashes"][key]["sha256"], key
        assert deps["frozen_source_match"][key] is True
    # Single frozen config per model: holdout and official share params
    # (no tuning after official).
    for model in MODELS:
        assert _metrics(model, "holdout")["params"] == \
            _metrics(model, "official")["params"], model


def test_future_invariance_identical_prediction_for_same_origin():
    for model in MODELS:
        for split in SPLITS:
            invariance = _metrics(model, split)["invariance"]
            assert invariance["identical"] is True, (model, split)
            assert invariance["max_abs_diff_perturbation"] == 0.0, (model,
                                                                    split)
            assert invariance["max_abs_diff_absence"] == 0.0, (model, split)


def test_future_invariance_functional_probe_cpu():
    """Tiny synthetic check: post-origin changes never move predictions."""
    from experiments.ch_breakthrough_v2.common import (
        PooledResidualForecaster,
        as_speed_series,
        build_features,
        utc,
    )
    rng = np.random.default_rng(11)
    idx = pd.date_range("2024-01-01", periods=1600, freq="h", tz="UTC")
    speed_vals = 400 + 30 * np.sin(np.arange(1600) / 24 * 2 * np.pi) \
        + rng.normal(0, 5, 1600)
    ace = pd.DataFrame({"timestamp_utc": idx, "filled_speed_kms": speed_vals,
                        "was_missing": np.zeros(1600, dtype=int)})
    cols = {}
    for tag in ("t035", "t045", "t055"):
        cols[f"ch_{tag}_area"] = np.abs(rng.normal(0.02, 0.005, 1600))
        cols[f"ch_{tag}_top1_lon"] = rng.normal(0, 20, 1600)
        cols[f"ch_{tag}_top1_lat"] = rng.normal(0, 15, 1600)
        cols[f"ch_{tag}_top1_lon_width"] = np.abs(rng.normal(20, 5, 1600))
        cols[f"ch_{tag}_equatorial_area"] = np.abs(rng.normal(0.01, 0.003,
                                                             1600))
    ch = pd.DataFrame(cols, index=idx)
    ch.index.name = "slot"
    features = build_features(ace, ch)
    speed = as_speed_series(ace)
    cutoff = utc("2024-02-20T00:00:00Z")
    fore = PooledResidualForecaster(
        params={"n_estimators": 5, "max_depth": 2}, device="cpu")
    fore.fit(features, speed, cutoff, max_origins=60)
    origins = pd.date_range("2024-02-25", periods=2, freq="h", tz="UTC")
    base = fore.predict(features, speed, origins)
    perturbed = speed.copy()
    after = perturbed.index[perturbed.index > origins.max()]
    perturbed.loc[after] = perturbed.loc[after] * 5.0 + 200.0
    pd.testing.assert_frame_equal(
        base, fore.predict(features, perturbed, origins))
    truncated = speed.loc[:origins.max()].copy()
    pd.testing.assert_frame_equal(
        base, fore.predict(features, truncated, origins))


def test_observed_mask_correct():
    from experiments.ch_breakthrough_v2.common import (
        as_speed_series,
        load_ace_hourly,
        observed_mask,
    )
    ace = load_ace_hourly(["store/ch-v1/ace.parquet"])
    speed = as_speed_series(ace)
    observed = observed_mask(speed, speed.index)
    assert observed.dtype == bool
    assert not observed.all() and observed.any()
    truth = pd.read_parquet(PINNED_SNAPSHOT / "truth.parquet")
    pinned_imputed = float(np.mean(truth["was_missing"].to_numpy() == 1))
    for model in MODELS:
        metrics = _metrics(model, "official")
        assert metrics["imputed_fraction"] is not None, model
        assert abs(metrics["imputed_fraction"] - pinned_imputed) < 1e-9, (
            model, metrics["imputed_fraction"], pinned_imputed)
        assert metrics["mse_observed"] != metrics["mse"], model
        assert np.isfinite(metrics["rmse_observed"]), model
        assert metrics["mse_observed"] is not None, model


def test_dependency_and_prediction_hash_manifests():
    deps = json.loads(
        (REPLAY_DIR / "source_dependency_manifest.json").read_text())
    assert set(("common", "hpo", "hpo_gpu", "recurrence", "physical_quality",
                "newrunner")) <= set(deps["dependencies"]), deps["dependencies"]
    for name, record in deps["dependencies"].items():
        assert Path(record["path"]).is_file(), name
        assert record["sha256"] == _sha256(record["path"]), name
    recorded = json.loads(
        (REPLAY_DIR / "prediction_hashes.json").read_text())["predictions"]
    assert set(recorded) == {f"{m}__{s}" for m in MODELS for s in SPLITS}
    inputs = json.loads(
        (REPLAY_DIR / "input_manifest.json").read_text())["identities"]
    for model in MODELS:
        for split in SPLITS:
            manifest = _manifest(model, split)
            ppath = REPLAY_DIR / f"{model}__{split}.parquet"
            assert manifest["prediction"]["sha256"] == _sha256(ppath), (
                model, split)
            assert recorded[f"{model}__{split}"]["sha256"] == _sha256(ppath)
            for key, ident in inputs.items():
                assert manifest["input_identities"][key]["sha256"] == \
                    ident["sha256"], (model, split, key)
            assert "code_hashes" in manifest and "common" in manifest[
                "code_hashes"] and "caller" in manifest["code_hashes"]
            assert manifest["actual_device"].startswith("cuda"), (model, split)
            assert manifest["seed"] == 42


def test_verdicts_honest_mse_target_and_peak_gate_exploratory():
    summary = json.loads((REPLAY_DIR / "summary.json").read_text())
    assert summary["target_mse_50pct"] == TARGET_MSE_50PCT
    assert summary["truth_read_after_predictions"] is True
    assert summary["predictions_written_utc"] <= summary[
        "truth_first_read_utc"]
    assert "exploratory" in summary["label"]
    for model in MODELS:
        metrics = _metrics(model, "official")
        assert metrics["exploratory"] is True, model
        assert metrics["n_keys"] == N_OFFICIAL_KEYS, model
        pinned = metrics["pinned_truth"]
        assert pinned is not None, model
        assert pinned["pairs"] == N_OFFICIAL_KEYS, model
        assert pinned["n_keys"] == N_OFFICIAL_KEYS, model
        assert "scorer" in pinned and "by_regime" in pinned["scorer"], model
        assert pinned["scorer"]["n_pairs"] == N_OFFICIAL_KEYS, model
        verdicts = metrics["verdicts"]
        assert verdicts["target_mse_50pct"] == TARGET_MSE_50PCT, model
        assert verdicts["mse_for_gate"] == pinned["mse_pinned"], model
        assert verdicts["target_pass"] == (
            pinned["mse_pinned"] <= TARGET_MSE_50PCT), model
        assert verdicts["peak_gate_pass"] == (
            verdicts["unique_ge_10"] and verdicts["precision_ge_060"]
            and verdicts["recall_ge_060"]
            and verdicts["height_mae_le_60"]), model
        assert verdicts["both_pass"] == (
            verdicts["target_pass"] and verdicts["peak_gate_pass"]), model
        assert verdicts["success_claimed"] is False, model
        assert verdicts["exploratory"] is True, model
        assert metrics["month_breakdown"], model
        assert set(("h1_24", "h25_48", "h49_72", "mse_by_horizon")) <= set(
            metrics["horizon_breakdown"]), model
        assert metrics["regime_diagnostics"], model
        assert metrics["overlap_block_ci95_7day"] is not None, model
        assert summary["models"][model]["official"]["both_pass"] == \
            verdicts["both_pass"], model
