"""Strict evaluator regression tests; no training or candidate selection."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.mse20_four_regimes import evaluate as ev


def passing():
    return {"mse": ev.TOTAL_MAX, "by_regime": {
        cell: {"mse": np.nextafter(3800., -np.inf), "reliable": True}
        for cell in ev.BASELINE_CELLS}}


def event():
    return {"precision": 1., "recall": 1 / 818, "passed": False,
            "height_mae_kms": 9999, "time_mae_hours": 99}


def test_exact_total_and_strict_individual_cells():
    official = passing()
    assert ev.acceptance_gates(official, event(), event())[0] == "pass"
    official["mse"] = np.nextafter(ev.TOTAL_MAX, np.inf)
    assert ev.acceptance_gates(official, event(), event())[0] == "unmet"
    for cell in ev.BASELINE_CELLS:
        official = passing()
        official["by_regime"][cell]["mse"] = 3800.
        status, gates = ev.acceptance_gates(official, event(), event())
        assert status == "unmet" and not gates[cell]["passed"]
        assert sum(g["passed"] for g in gates.values()) == 6


@pytest.mark.parametrize("value", [None, np.nan, np.inf, -np.inf])
def test_undefined_metrics_are_indeterminate(value):
    for name in ("precision", "recall"):
        candidate = event(); candidate[name] = value
        assert ev.acceptance_gates(passing(), candidate, event())[0] == "indeterminate"
        assert ev.acceptance_gates(passing(), event(), candidate)[0] == "indeterminate"
    official = passing(); official["mse"] = value
    assert ev.acceptance_gates(official, event(), event())[0] == "indeterminate"


def test_required_cells_missing_unreliable_nonfinite():
    for mutation in ("missing", "unreliable", "nonfinite"):
        official = passing(); cell = next(iter(ev.BASELINE_CELLS))
        if mutation == "missing": del official["by_regime"][cell]
        elif mutation == "unreliable": official["by_regime"][cell]["reliable"] = False
        else: official["by_regime"][cell]["mse"] = np.nan
        assert ev.acceptance_gates(official, event(), event())[0] == "indeterminate"


def test_precision_recall_no_compensation_and_legacy_gate_ignored():
    assert ev.acceptance_gates(passing(), event(), event())[0] == "pass"
    candidate = event(); candidate.update(precision=.5, recall=1., passed=True)
    assert ev.acceptance_gates(passing(), candidate, event())[0] == "unmet"
    candidate = event(); candidate["recall"] = np.nextafter(1 / 818, 0.)
    assert ev.acceptance_gates(passing(), candidate, event())[0] == "unmet"


@pytest.fixture
def frames():
    origins = pd.date_range("2026-06-01", periods=2, freq="h", tz="UTC")
    folds = pd.DataFrame({ev.KEY[0]: origins.strftime("%Y-%m-%dT%H:%M:%SZ"), "regime": "low_low_high"})
    pred = pd.DataFrame([(o, h, 400.) for o in folds[ev.KEY[0]] for h in range(1, 73)], columns=ev.KEY + ["pred_kms"])
    return folds, pred


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "extra", "float_horizon", "string_horizon", "nan", "inf", "zero", "negative", "too_fast", "naive_time", "offset_time", "subhour", "string_speed"])
def test_strict_forecast_support(frames, mutation):
    folds, pred = frames
    if mutation == "duplicate": pred = pd.concat([pred, pred.iloc[:1]])
    elif mutation == "missing": pred = pred.iloc[1:]
    elif mutation == "extra": pred.loc[0, "horizon_hours"] = 73
    elif mutation == "float_horizon": pred["horizon_hours"] = pred.horizon_hours.astype(float)
    elif mutation == "string_horizon": pred["horizon_hours"] = pred.horizon_hours.astype(str)
    elif mutation == "string_speed": pred["pred_kms"] = pred.pred_kms.astype(str)
    elif mutation == "naive_time": pred.loc[0, ev.KEY[0]] = "2026-06-01T00:00:00"
    elif mutation == "offset_time": pred.loc[0, ev.KEY[0]] = "2026-06-01T09:00:00+09:00"
    elif mutation == "subhour": pred.loc[0, ev.KEY[0]] = "2026-06-01T00:00:01Z"
    else: pred.loc[0, "pred_kms"] = {"nan": np.nan, "inf": np.inf, "zero": 0., "negative": -1., "too_fast": 3001.}[mutation]
    with pytest.raises(ev.IntegrityError): ev.validate_frame(pred, folds)


def test_truth_complete_and_finite(frames):
    folds, pred = frames
    truth = pred.rename(columns={"pred_kms": "target_kms"})
    ev.validate_frame(truth, folds, "target_kms")
    with pytest.raises(ev.IntegrityError): ev.validate_frame(truth.iloc[1:], folds, "target_kms")
    truth.loc[0, "target_kms"] = np.nan
    with pytest.raises(ev.IntegrityError): ev.validate_frame(truth, folds, "target_kms")


def test_truth_fill_preserves_real_observed_mask(frames):
    _, pred = frames
    truth = pred.rename(columns={"pred_kms": "target_kms"}); truth["was_missing"] = 0
    times = pd.to_datetime(truth[ev.KEY[0]], utc=True) + pd.to_timedelta(truth.horizon_hours, unit="h")
    truth.loc[times == times.iloc[10], "was_missing"] = 1
    target, observed = ev.truth_series(truth)
    assert np.isfinite(target).all() and not observed.iloc[10]
    assert int((~observed).sum()) == 1


@pytest.mark.parametrize("mutation", ["value_disagreement", "mask_disagreement", "missing_mask", "bad_mask", "leading_missing", "bad_fill"])
def test_truth_reconciliation_fails_closed(frames, mutation):
    _, pred = frames
    truth = pred.rename(columns={"pred_kms": "target_kms"}); truth["was_missing"] = 0
    if mutation == "value_disagreement": truth.loc[1, "target_kms"] = 500.
    elif mutation == "mask_disagreement": truth.loc[1, "was_missing"] = 1
    elif mutation == "missing_mask": truth = truth.drop(columns="was_missing")
    elif mutation == "bad_mask": truth.loc[0, "was_missing"] = 2
    elif mutation == "leading_missing": truth.loc[0, "was_missing"] = 1
    elif mutation == "bad_fill":
        times = pd.to_datetime(truth[ev.KEY[0]], utc=True) + pd.to_timedelta(truth.horizon_hours, unit="h")
        mask = times == times.iloc[10]
        truth.loc[mask, "was_missing"] = 1; truth.loc[mask, "target_kms"] = 500.
    with pytest.raises(ev.IntegrityError): ev.truth_series(truth)


def test_explicit_all_origins_known_event_counts(frames):
    folds, pred = frames
    truth = pred.rename(columns={"pred_kms": "target_kms"}); truth["was_missing"] = 0
    times = pd.to_datetime(truth[ev.KEY[0]], utc=True) + pd.to_timedelta(truth.horizon_hours, unit="h")
    truth.loc[times == times.iloc[30], "target_kms"] = 700.
    target, observed = ev.truth_series(truth)
    pred["pred_kms"] = truth.target_kms
    naive = pred.copy(); naive["pred_kms"] += 1.
    result = ev.metrics(pred, truth, folds, naive, target, observed)
    event = result["diagnostics"]["events"]
    assert (event["tp"], event["fp"], event["fn"]) == (2, 0, 0)
    assert event["evaluated_windows"] == 2  # Includes nonmidnight origin.
    assert len(result["diagnostics"]["true_observed_peak_timestamps"]) == 1


def test_skipped_event_window_rejected(frames, monkeypatch):
    folds, pred = frames
    truth = pred.rename(columns={"pred_kms": "target_kms"}); truth["was_missing"] = 0
    target, observed = ev.truth_series(truth)
    monkeypatch.setattr(ev, "score_forecast", lambda *a, **kw: {"events": {"requested_windows": 2, "evaluated_windows": 1, "missing_windows": 1}})
    with pytest.raises(ev.IntegrityError, match="skipped"):
        ev.metrics(pred, truth, folds, None, target, observed)


def test_prediction_parity_tolerance(frames):
    _, pred = frames
    assert ev.parity(pred, pred.iloc[::-1]) == 0.
    changed = pred.copy(); changed.loc[0, "pred_kms"] += 2e-8
    with pytest.raises(ev.IntegrityError, match="drift"): ev.parity(pred, changed)


def test_file_identity_detects_mutation(tmp_path):
    path = tmp_path / "evidence"; path.write_text("original")
    entry = {"path": str(path), "sha256": ev.sha256(path)}
    assert ev.verify_file(entry, "test") == path
    path.write_text("changed")
    with pytest.raises(ev.IntegrityError, match="sha256"): ev.verify_file(entry, "test")


def test_cli_failure_always_json_and_exit2(tmp_path):
    output = tmp_path / "failure.json"
    code = ev.main(["--mode", "baseline", "--run-dir", str(tmp_path), "--contract", str(tmp_path / "absent"), "--reference", "absent", "--output", str(output)])
    report = json.loads(output.read_text())
    assert code == report["exit_code"] == 2
    assert report["status"] == "indeterminate" and report["objective_achieved"] is False
    assert report["reasons"]


def test_baseline_zero_never_objective_achieved(tmp_path, monkeypatch):
    output = tmp_path / "baseline.json"
    monkeypatch.setattr(ev, "evaluate", lambda args: {"status": "baseline_verified", "objective_achieved": False, "exit_code": 0})
    assert ev.main(["--mode", "baseline", "--run-dir", str(tmp_path), "--contract", "unused", "--reference", "unused", "--output", str(output)]) == 0
    assert json.loads(output.read_text())["objective_achieved"] is False


@pytest.fixture
def artifact_bundle(tmp_path):
    from types import SimpleNamespace
    def put(name, data):
        p = tmp_path / name
        p.write_text(json.dumps(data))
        return {"path": str(p), "sha256": ev.sha256(p)}
    evidence = put("evidence.json", {"test": "passed"})
    summary = put("summary.json", {"passed": True, "files": {"log": evidence}})
    replay = put("replay.json", [])
    contract_hash = "contract-hash"
    contract = {"started_utc": "2026-10-03T06:00:00Z", "deadline_utc": "2026-10-03T14:00:00Z",
                "official_cutoff": "2026-05-31T23:00:00Z", "files": {"reference": {"sha256": "reference-hash"}}}
    baseline = put("baseline.json", {"mode": "baseline", "status": "baseline_verified", "objective_achieved": False,
        "exit_code": 0, "contract_sha256": contract_hash, "finished_utc": "2026-10-03T06:30:00Z",
        "reference_sha256": "reference-hash", "support": {"origins": 2833, "pairs": 203976},
        "evaluator_sha256": ev.sha256(ev.__file__), "replay": replay})
    frozen = put("frozen.json", {"schema_version": 1, "contract_sha256": contract_hash,
        "frozen_utc": "2026-10-03T08:00:00Z", "selection_completed_utc": "2026-10-03T07:59:00Z",
        "training_cutoff_utc": contract["official_cutoff"],
        "files": {"selection": evidence, "splits": evidence, "code": evidence},
        "recipe": {"model": "tree", "config": {}, "features": ["lag1"], "training_rule": "fixed", "seed": 42, "blend_weights": [1.]}})
    final = put("final-artifacts.json", {"schema_version": 1, "contract_sha256": contract_hash, "recipe_sha256": frozen["sha256"],
        "created_utc": "2026-10-03T10:00:00Z", "audit_started_utc": "2026-10-03T08:01:00Z",
        "official_generation_started_utc": "2026-10-03T09:00:00Z", "training_label_max_utc": contract["official_cutoff"],
        "files": {"predictions": replay, "baseline_verification": baseline, "test_summary": summary,
                  "causality_summary": summary, "model": evidence}, "model_files": ["model"], "transform_files": []})
    args = SimpleNamespace(frozen=frozen["path"], final_artifacts=final["path"], run_dir=str(tmp_path), candidate=replay["path"])
    return args, contract, contract_hash


def test_complete_final_identity(artifact_bundle):
    args, contract, digest = artifact_bundle
    report = ev.verify_acceptance_artifacts(args, contract, digest)
    assert report["candidate_sha256"] == ev.sha256(args.candidate)


@pytest.mark.parametrize("mutation", ["recipe_hash", "contract_hash", "prefreeze_audit", "prefreeze_official", "late_labels", "deadline", "missing_model", "different_candidate", "stale_baseline", "tests_failed", "regime_feature"])
def test_invalid_final_identity_rejected(artifact_bundle, mutation):
    args, contract, digest = artifact_bundle
    final_path = Path(args.final_artifacts); final = ev.read_json(final_path)
    if mutation == "recipe_hash": final["recipe_sha256"] = "different"
    elif mutation == "contract_hash": final["contract_sha256"] = "different"
    elif mutation == "prefreeze_audit": final["audit_started_utc"] = "2026-10-03T07:00:00Z"
    elif mutation == "prefreeze_official": final["official_generation_started_utc"] = "2026-10-03T07:00:00Z"
    elif mutation == "late_labels": final["training_label_max_utc"] = "2026-06-01T00:00:00Z"
    elif mutation == "deadline": final["created_utc"] = "2026-10-03T15:00:00Z"
    elif mutation == "missing_model": final["model_files"] = []
    elif mutation == "different_candidate": args.candidate = str(Path(args.run_dir) / "other")
    elif mutation in ("stale_baseline", "tests_failed"):
        name = "baseline_verification" if mutation == "stale_baseline" else "test_summary"
        path = Path(final["files"][name]["path"]); doc = ev.read_json(path)
        doc["contract_sha256" if mutation == "stale_baseline" else "passed"] = "different" if mutation == "stale_baseline" else False
        path.write_text(json.dumps(doc)); final["files"][name]["sha256"] = ev.sha256(path)
        if mutation == "tests_failed": final["files"]["causality_summary"]["sha256"] = ev.sha256(path)
    elif mutation == "regime_feature":
        frozen = ev.read_json(args.frozen); frozen["recipe"]["features"].append("regime")
        Path(args.frozen).write_text(json.dumps(frozen)); final["recipe_sha256"] = ev.sha256(args.frozen)
    final_path.write_text(json.dumps(final))
    with pytest.raises(ev.IntegrityError): ev.verify_acceptance_artifacts(args, contract, digest)


@pytest.mark.parametrize("status,code", [("pass", 0), ("unmet", 1), ("indeterminate", 2)])
def test_cli_acceptance_status_and_exit_are_consistent(tmp_path, monkeypatch, status, code):
    output = tmp_path / "acceptance.json"
    monkeypatch.setattr(ev, "evaluate", lambda args: {"status": status, "exit_code": code, "objective_achieved": status == "pass"})
    result = ev.main(["--mode", "acceptance", "--run-dir", str(tmp_path), "--contract", "unused",
                      "--reference", "unused", "--candidate", "unused", "--frozen", "unused", "--output", str(output)])
    report = ev.read_json(output)
    assert result == report["exit_code"] == code
    assert report["objective_achieved"] is (status == "pass")


@pytest.mark.parametrize("name,value", [("snapshot", "wrong"), ("baseline_id", "wrong"),
    ("expected_origins", 2832), ("expected_pairs", 203975), ("missing_policy", "strict"),
    ("gates", {"mse_max": ev.TOTAL_MAX + 1, "cell_mse_strict_max": 3800}),
    ("baseline", {}), ("deadline_utc", "2026-10-04T06:00:00Z"),
    ("official_cutoff", "2026-06-01T00:00:00Z"),
    ("official_cutoff", "2026-05-31T22:00:00Z")])
def test_contract_invariants_cannot_be_relaxed(tmp_path, name, value):
    contract = {"schema_version": 1, "run_dir": str(tmp_path), "snapshot": ev.SNAPSHOT,
                "baseline_id": ev.BASELINE_ID, "expected_origins": 2833, "expected_pairs": 203976,
                "missing_policy": "forward_fill", "gates": {"mse_max": ev.TOTAL_MAX, "cell_mse_strict_max": 3800},
                "baseline": {"mse": ev.BASELINE_MSE, "cells": ev.BASELINE_CELLS, "tp": 1, "fp": 0, "fn": 817},
                "started_utc": "2026-10-03T06:00:00Z", "deadline_utc": "2026-10-03T14:00:00Z",
                "official_cutoff": "2026-05-31T23:00:00Z"}
    contract[name] = value
    with pytest.raises(ev.IntegrityError): ev.verify_contract(contract, tmp_path)
