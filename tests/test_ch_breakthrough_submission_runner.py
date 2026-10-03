"""Focused fail-closed contract checks for frozen submission replay."""
import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.seongeun import ch_breakthrough_runner as runner


def config(branch="combined"):
    return {"data": {"feature_branch": branch, "feature_arm": runner.ARMS.get(branch),
                     "training_cutoff": "2026-05-31T23:00:00Z"},
            "hyperparameters": {"xgboost": {"n_estimators": 80},
                                "seed": 42, "max_train_origins": 6000}}


@pytest.mark.parametrize("branch", runner.ARMS)
def test_supported_branch_and_config_integrity(branch):
    cfg = config(branch)
    runner.validate_config(cfg, runner.config_digest(cfg))
    other = copy.deepcopy(cfg)
    other["hyperparameters"]["xgboost"]["n_estimators"] += 1
    with pytest.raises(ValueError, match="CONFIG sha256"):
        runner.validate_config(other, runner.config_digest(cfg))


def test_unsupported_arm_and_frozen_training_contract():
    cfg = config("other")
    with pytest.raises(ValueError, match="unsupported"):
        runner.validate_config(cfg)
    cfg = config()
    cfg["hyperparameters"]["max_train_origins"] = 3
    with pytest.raises(ValueError, match="cap/seed"):
        runner.validate_config(cfg)


def test_input_and_source_hashes_fail_closed(tmp_path):
    cfg = config()
    for name in runner.SOURCE_NAMES:
        (tmp_path / f"{name}.py").write_text(name)
    ace, ch, raw = [tmp_path / name for name in ("ace", "ch", "raw.parquet")]
    for path in (ace, ch, raw):
        path.write_text(path.name)
    cfg["data"]["source_sha256"] = {name: runner.file_digest(tmp_path / f"{name}.py")
                                     for name in runner.SOURCE_NAMES}
    cfg["data"]["input_sha256"] = {"ace": runner.file_digest(ace), "ch_hourly": runner.file_digest(ch),
        "raw_ace_years": hashlib.sha256(runner.file_digest(raw).encode()).hexdigest()}
    args = (cfg, [ace], ch, str(raw), tmp_path)
    runner.validate_hashes(*args)
    ace.write_text("changed")
    with pytest.raises(ValueError, match="input sha256 mismatch: ace"):
        runner.validate_hashes(*args)
    ace.write_text("ace")
    (tmp_path / "common.py").write_text("changed")
    with pytest.raises(ValueError, match="source sha256 mismatch: common"):
        runner.validate_hashes(*args)


def test_origin_selection_and_complete_horizons():
    origins = pd.date_range("2026-06-01", periods=4, freq="h", tz="UTC")
    folds = pd.DataFrame({"origin_last_input_utc": [origins[3], origins[0], origins[1], origins[0]]})
    selected = runner.select_origins(folds, origins, max_folds=2)
    assert selected.equals(origins[:2])
    with pytest.raises(ValueError, match="missing from features"):
        runner.select_origins(folds, origins[:2])
    with pytest.raises(ValueError, match="positive"):
        runner.select_origins(folds, origins, 0)
    preds = pd.DataFrame({"origin_last_input_utc": selected.repeat(72),
                          "horizon_hours": np.tile(np.arange(1, 73), 2), "pred_kms": 400.0})
    runner.validate_predictions(preds, selected)
    with pytest.raises(ValueError, match="72 horizons"):
        runner.validate_predictions(preds.iloc[:-1], selected)
    broken = preds.copy()
    broken.loc[1, "horizon_hours"] = 1
    with pytest.raises(ValueError, match="72 horizons"):
        runner.validate_predictions(broken, selected)


def test_smoke_parser_does_not_override_training_params():
    args = runner.parser().parse_args(["--folds", "folds.parquet", "--output", "p.parquet", "--max-folds", "3"])
    assert args.max_folds == 3
    assert not hasattr(args, "max_train_origins")
    assert not hasattr(args, "seed")


def test_run_refits_frozen_params_and_records_prediction_only_timing(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    origins = pd.date_range("2026-05-31T23:00:00Z", periods=3, freq="h")
    folds_path = tmp_path / "folds.parquet"
    official_folds = pd.DataFrame({"origin_last_input_utc": origins.strftime("%Y-%m-%dT%H:%M:%SZ")})
    official_folds.to_parquet(folds_path)
    features = pd.DataFrame({"x": 1.0}, index=origins)
    speed = pd.Series(400.0, index=origins)
    speed.attrs["observed"] = pd.Series(True, index=origins)
    frames = {"speed": speed}
    monkeypatch.setattr(runner, "validate_hashes", lambda *a: {"ace": "checked"})
    monkeypatch.setattr(runner.hpo, "load_inputs", lambda *a: frames)
    monkeypatch.setattr(runner.hpo, "_arm_features", lambda *a: (features, ["x"], True))
    calls = []

    def fake_fit(branch, arm, params, got_frames, got_speed, cutoff, got_origins, **kw):
        calls.append(kw)
        assert got_speed.attrs["observed"].all()
        assert len(got_origins) == 2
        fore = SimpleNamespace(actual_device="cpu", columns=["x"],
            model=SimpleNamespace(predict=lambda matrix: np.ones(len(matrix))))
        fore._build_predict_matrix = lambda *a: (got_origins, pd.DataFrame({"x": np.ones(144)}), np.full(144, 400.0))
        preds = pd.DataFrame({"origin_last_input_utc": got_origins.repeat(72),
            "horizon_hours": np.tile(np.arange(1, 73), 2), "pred_kms": 401.0})
        return preds, fore, 10.0, 0.5

    monkeypatch.setattr(runner.hpo, "fit_predict", fake_fit)
    cfg = config()
    cfg["data"]["source_sha256"] = {}
    output = tmp_path / "preds.parquet"
    assert runner.run(cfg, runner.config_digest(cfg), argv=["--folds", str(folds_path),
        "--output", str(output), "--max-folds", "2", "--device", "cpu"]) == 0
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert calls == [{"device": "cpu", "seed": 42, "max_train_origins": 6000}]
    assert manifest["n_rows"] == 144
    assert manifest["smoke"] and manifest["observed_metadata_preserved"]
    assert manifest["pipeline_predict_seconds_per_fold"] == 0.5
    assert manifest["inference_seconds_per_fold"] >= 0
    from scorer.scoring import validate
    exported = pd.read_parquet(output)
    assert validate(exported, official_folds.iloc[:2]) == []
    np.testing.assert_array_equal(exported["pred_kms"].to_numpy(), np.full(144, 401.0))
    assert exported["origin_last_input_utc"].iloc[0] == "2026-05-31T23:00:00Z"

    earlier = origins - pd.Timedelta(hours=1)
    monkeypatch.setattr(runner.hpo, "_arm_features", lambda *a: (features.reindex(earlier), ["x"], True))
    pd.DataFrame({"origin_last_input_utc": earlier}).to_parquet(folds_path)
    with pytest.raises(ValueError, match="must not precede"):
        runner.run(cfg, argv=["--folds", str(folds_path), "--output", str(output), "--device", "cpu"])


@pytest.mark.parametrize("experiment", [
    "xgb_ch_recurrence_physics_official",
    "xgb_ch_recurrence_official",
    "xgb_ch_physics_quality_official",
])
def test_published_wrapper_config_identity(experiment):
    """Preserve numeric JSON identity (1.0 is not interchangeable with 1)."""
    import runpy
    from submit import canonical_config

    directory = Path(__file__).resolve().parents[1] / "scripts" / "seongeun" / experiment
    wrappers = list(directory.glob("*.py"))
    assert len(wrappers) == 1, f"expected one frozen wrapper in {experiment}"
    wrapper = wrappers[0]
    namespace = runpy.run_path(str(wrapper), run_name="submission_identity_test")
    cfg = namespace["CONFIG"]
    assert (namespace["CONFIG_SHA256"] == wrapper.stem
            == runner.config_digest(cfg) == canonical_config(cfg)[1])
