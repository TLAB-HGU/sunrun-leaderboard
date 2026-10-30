"""Synthetic-only tests for final_evaluate.py (no real IO, fits, or scoring).

Every stage-level check here exercises pure helpers or fail-closed refusal
paths on tmp dirs. Real stages (regression/score/evaluate with live data)
are root-only and never invoked.
"""

import json

import numpy as np
import pandas as pd
import pytest

from experiments.mse20_scope_expansion import (
    final_contract,
    final_evaluate,
    final_model,
    protocol,
)


def _truth_df(n_hours=80, start="2026-06-01T00:00:00Z", missing=()):
    idx = pd.date_range(protocol.ts_utc(start), periods=n_hours, freq="h", tz="UTC")
    origin0 = protocol.ts_utc("2026-05-31T23:00:00Z")
    hours = ((idx - origin0).total_seconds() / 3600).to_numpy(dtype=int)
    # Fold into one origin's 1..72 window plus a second origin tail; keep simple:
    origins = [origin0.isoformat()] * n_hours
    frame = pd.DataFrame({
        "origin_last_input_utc": origins,
        "horizon_hours": list(range(1, n_hours + 1)),
        "target_kms": np.full(n_hours, 400.0),
        "was_missing": np.zeros(n_hours, dtype=int),
    })
    for h in missing:
        frame.loc[frame["horizon_hours"] == h, "was_missing"] = 1
    return frame


# ----------------------------------------------------------------------------
# Pins / scope (declared bindings, never re-chosen here).
# ----------------------------------------------------------------------------

def test_pins_match_frozen_contract_and_producer():
    assert final_contract.FROZEN_SELECTION_RECEIPT_SHA256 == (
        "faa32862d366d8dd8e284e0b4ddffefbb8e042e7be82a5296985d11cb80697a2")
    assert final_contract.PINNED_SNAPSHOT_SHA256 == (
        "bcf5d5417d99eaa331ddca40581d48953e09a58e")
    assert final_contract.OFFICIAL_TOTAL_KEYS == 2833 * 72 == 203976
    assert final_contract.TOTAL_MSE_MAX == 3289.0890348547073
    assert final_evaluate.EVAL_TEST_FILE.endswith("test_scope_expansion_final_evaluate.py")


def test_only_vjepa_excluded_with_narrow_clock_plugin():
    assert final_evaluate.VJEPA_PREFIX == "test_vjepa"
    assert final_evaluate.CLOCK_PLUGIN == (
        "experiments.mse20_hss_phase_verify_v1.clock_plugin")
    assert "fake_run" not in final_evaluate.CLOCK_PLUGIN


def test_suite_inventory_excludes_only_vjepa_and_requires_new_suites():
    inv = final_evaluate.suite_inventory("/home/t-lab01/sunrun")
    names = [e["test_file"] for e in inv]
    assert all(not n.split("/")[-1].startswith("test_vjepa") for n in names)
    assert final_evaluate.EVAL_TEST_FILE in names
    assert final_model.FINAL_TEST_FILE in names
    assert all(len(e["sha256"]) == 64 for e in inv)


def test_report_dir_must_stay_under_final(tmp_path):
    run = tmp_path / "run"
    with pytest.raises(ValueError):
        final_evaluate.report_dir_for(run, tmp_path / "elsewhere")
    out = final_evaluate.report_dir_for(run, None)
    assert out.resolve().is_relative_to(final_model.final_dir(run).resolve())


def test_append_only_never_overwrites(tmp_path):
    p = tmp_path / "final_score_a.json"
    p.write_text("{}")
    q = final_evaluate._append_path(p)
    assert q != p and p.read_text() == "{}"


# ----------------------------------------------------------------------------
# Canonical numeric/event digest (unrounded, no timestamps).
# ----------------------------------------------------------------------------

def test_digest_deterministic_and_timestamp_free():
    cand = {"tp": 2, "fp": 1, "fn": 3, "f1": 0.5}
    champ = {"tp": 1, "fp": 0, "fn": 5, "f1": 0.25}
    cells = {c: 100.0 for c in final_contract.CELL_NAMES}
    a = final_evaluate.numeric_event_digest(3000.0, cells, cand, champ)
    b = final_evaluate.numeric_event_digest(3000.0, cells, dict(cand), dict(champ))
    assert a == b and len(a) == 64
    other = final_evaluate.numeric_event_digest(3000.0000001, cells, cand, champ)
    assert other != a  # unrounded inputs matter; never pre-round scores


def test_identical_score_gate_rejects_copied_or_malformed():
    good = "a" * 64
    final_contract.assert_two_identical_scores(good, good)
    with pytest.raises(ValueError):
        final_evaluate.numeric_event_digest  # sanity: helper is pure, gate lives in contract
        final_contract.assert_two_identical_scores(good, "b" * 64)
    with pytest.raises(ValueError):
        final_contract.assert_two_identical_scores("not-hex", "not-hex")


# ----------------------------------------------------------------------------
# Truth series derivation (duplicate consistency, causal fill only).
# ----------------------------------------------------------------------------

def test_truth_series_happy_path():
    target, observed = final_evaluate.derive_truth_series(_truth_df())
    assert len(target) == 80 and bool(observed.all())


def test_truth_series_rejects_inconsistent_duplicates():
    frame = _truth_df()
    dup = frame.iloc[[0]].copy()
    dup["target_kms"] = 999.0
    with pytest.raises(ValueError):
        final_evaluate.derive_truth_series(pd.concat([frame, dup], ignore_index=True))


def test_truth_series_rejects_gaps_and_bad_mask():
    frame = _truth_df().drop(index=5).reset_index(drop=True)
    with pytest.raises(ValueError):
        final_evaluate.derive_truth_series(frame)
    bad = _truth_df()
    bad.loc[0, "was_missing"] = 7
    with pytest.raises(ValueError):
        final_evaluate.derive_truth_series(bad)


def test_truth_series_rejects_leading_missing_without_seed():
    frame = _truth_df(missing=(1, 2, 3))
    with pytest.raises(ValueError):
        final_evaluate.derive_truth_series(frame)


def test_event_recompute_flat_series_is_undefined_not_forced():
    frame = _truth_df(80)
    target, observed = final_evaluate.derive_truth_series(frame)
    pred = pd.DataFrame({"origin_last_input_utc": frame["origin_last_input_utc"],
                         "horizon_hours": frame["horizon_hours"],
                         "pred_kms": np.full(len(frame), 400.0)})
    origins = pd.DatetimeIndex([protocol.ts_utc("2026-05-31T23:00:00Z")])
    out = final_evaluate.recompute_event_f1(pred, target, observed, origins)
    assert out["f1"] is None  # never hardcoded; evaluator reports the gap
    assert final_contract.check_f1_gate(out["f1"], 0.1)["ok"] is False


def test_gate_passthrough_cells_and_f1_ordering():
    cells = {c: 100.0 for c in final_contract.CELL_NAMES}
    assert final_contract.check_cells_gate(cells)["ok"] is True
    bad = dict(cells)
    bad["low_low_high"] = 3800.0  # strict: equality fails
    assert final_contract.check_cells_gate(bad)["ok"] is False
    assert final_contract.check_f1_gate(0.05, 0.10)["ok"] is False
    assert final_contract.check_native_parity(-1.0, 0.0)["ok"] is False


# ----------------------------------------------------------------------------
# Freeze-before-truth ordering + tamper detection (tmp dirs only).
# ----------------------------------------------------------------------------

def _bare_run(tmp_path):
    run = tmp_path / "run"
    (final_model.final_dir(run)).mkdir(parents=True)
    return run


def test_score_refuses_before_forecast_without_touching_truth(tmp_path, monkeypatch):
    run = _bare_run(tmp_path)
    monkeypatch.setenv(final_evaluate.SNAPSHOT_ENV_VAR, str(tmp_path / "no-snapshot"))
    with pytest.raises(protocol.NotReadyError, match="forecast"):
        final_evaluate.stage_score(run, "/home/t-lab01/sunrun")


def test_freeze_rejects_tampered_prediction_bytes(tmp_path):
    run = _bare_run(tmp_path)
    fdir = final_model.final_dir(run)
    frame = pd.DataFrame({"origin_last_input_utc": ["2026-05-31T23:00:00Z"],
                          "horizon_hours": [1], "pred_kms": [400.0]})
    frame.to_parquet(fdir / final_model.FINAL_PREDICTIONS_FILENAME, index=False)
    (fdir / final_model.FINAL_DIRECT_MODEL_FILENAME).write_bytes(b"dummy-model")
    model_sha = protocol.sha256_file(fdir / final_model.FINAL_DIRECT_MODEL_FILENAME)
    reg = {"winner": "router",
           "selection_sha256": final_contract.FROZEN_SELECTION_RECEIPT_SHA256,
           "snapshot_sha256": final_contract.PINNED_SNAPSHOT_SHA256,
           "user_start_utc": protocol.USER_START_UTC, "user_deadline_utc": None,
           "window_utc": list(final_contract.SNAPSHOT_WINDOW)}
    (fdir / final_model.FINAL_REGISTRATION_FILENAME).write_text(json.dumps(reg))
    (fdir / final_model.FINAL_PRED_META_FILENAME).write_text(
        json.dumps({"prediction_sha256": "0" * 64,
                    "prediction_digest": "0" * 64,
                    "logged_utc": "2026-10-05T00:00:00Z"}))
    (fdir / final_model.FINAL_VERIFY_FILENAME).write_text(
        json.dumps({"prediction_digest": "0" * 64, "fits_observed": 0,
                    "fit_executions_bound": 3, "direct_trees": 600}))
    (fdir / final_model.FINAL_FIT_FILENAME).write_text(
        json.dumps({"fit_executions": 3,
                    "direct_fit": {"model_sha256": model_sha}}))
    (fdir / final_model.FINAL_DIRECT_META_FILENAME).write_text(
        json.dumps({"params": dict(final_model.FINAL_PARAMS)}))
    (fdir / final_model.FINAL_ROUTER_META_FILENAME).write_text(
        json.dumps({"params": dict(final_model.FINAL_ROUTER_PARAMS)}))
    with pytest.raises(ValueError, match="prediction bytes changed"):
        final_evaluate._require_forecast_freeze(run, "/home/t-lab01/sunrun")


def test_freeze_requires_mandatory_600_rounds_and_model_sha(tmp_path):
    run = _bare_run(tmp_path)
    fdir = final_model.final_dir(run)
    frame = pd.DataFrame({"origin_last_input_utc": ["2026-05-31T23:00:00Z"],
                          "horizon_hours": [1], "pred_kms": [400.0]})
    frame.to_parquet(fdir / final_model.FINAL_PREDICTIONS_FILENAME, index=False)
    live_sha = protocol.sha256_file(fdir / final_model.FINAL_PREDICTIONS_FILENAME)
    reg = {"winner": "router",
           "selection_sha256": final_contract.FROZEN_SELECTION_RECEIPT_SHA256,
           "snapshot_sha256": final_contract.PINNED_SNAPSHOT_SHA256,
           "user_start_utc": protocol.USER_START_UTC, "user_deadline_utc": None,
           "window_utc": list(final_contract.SNAPSHOT_WINDOW)}
    (fdir / final_model.FINAL_REGISTRATION_FILENAME).write_text(json.dumps(reg))
    (fdir / final_model.FINAL_PRED_META_FILENAME).write_text(
        json.dumps({"prediction_sha256": live_sha,
                    "prediction_digest": "0" * 64,
                    "logged_utc": "2026-10-05T00:00:00Z"}))
    (fdir / final_model.FINAL_VERIFY_FILENAME).write_text(
        json.dumps({"prediction_digest": "0" * 64, "fits_observed": 0,
                    "fit_executions_bound": 3, "direct_trees": 600}))
    bad_params = dict(final_model.FINAL_PARAMS)
    bad_params["n_estimators"] = 500
    (fdir / final_model.FINAL_FIT_FILENAME).write_text(
        json.dumps({"fit_executions": 3, "direct_fit": {"model_sha256": live_sha}}))
    (fdir / final_model.FINAL_DIRECT_META_FILENAME).write_text(
        json.dumps({"params": bad_params}))
    (fdir / final_model.FINAL_ROUTER_META_FILENAME).write_text(
        json.dumps({"params": dict(final_model.FINAL_ROUTER_PARAMS)}))
    with pytest.raises(protocol.NotReadyError, match="600 trees mandatory"):
        final_evaluate._require_forecast_freeze(run, "/home/t-lab01/sunrun")


def test_evaluate_empty_run_is_invalid_with_gaps(tmp_path):
    run = _bare_run(tmp_path)
    code, report = final_evaluate.stage_evaluate(run, "/home/t-lab01/sunrun")
    assert code == final_contract.INVALID == 2
    assert len(report["gaps"]) > 0
    assert report["scores"] is None
    assert all(v is False for v in report["evidence"].values())


def test_evaluate_rejects_mismatched_score_digests(tmp_path):
    run = _bare_run(tmp_path)
    out_dir = final_evaluate.report_dir_for(run, None)
    body = {"total_mse": 3000.0,
            "cells": {c: 100.0 for c in final_contract.CELL_NAMES},
            "candidate_event": {"tp": 1, "fp": 0, "fn": 2, "f1": 0.5},
            "champion_event": {"tp": 1, "fp": 0, "fn": 2, "f1": 0.5},
            "official": {"n_pairs": 203976}, "logged_utc": "2026-10-05T00:00:00Z"}
    a = dict(body, stage="score-a", score_digest="a" * 64)
    b = dict(body, stage="score-b", score_digest="b" * 64)
    (out_dir / final_evaluate.EVAL_SCORE_A_FILENAME).write_text(json.dumps(a))
    (out_dir / final_evaluate.EVAL_SCORE_B_FILENAME).write_text(json.dumps(b))
    code, report = final_evaluate.stage_evaluate(run, "/home/t-lab01/sunrun")
    assert code == 2
    assert report["checks"]["two_scores"]["ok"] is False


def test_evaluate_rejects_forged_unrecomputed_digest(tmp_path):
    run = _bare_run(tmp_path)
    out_dir = final_evaluate.report_dir_for(run, None)
    body = {"total_mse": 3000.0,
            "cells": {c: 100.0 for c in final_contract.CELL_NAMES},
            "candidate_event": {"tp": 1, "fp": 0, "fn": 2, "f1": 0.5},
            "champion_event": {"tp": 1, "fp": 0, "fn": 2, "f1": 0.5},
            "official": {"n_pairs": 203976}, "logged_utc": "2026-10-05T00:00:00Z"}
    forged = "c" * 64  # well-formed hex but not recomputed from these numerics
    a = dict(body, stage="score-a", score_digest=forged)
    b = dict(body, stage="score-b", score_digest=forged)
    (out_dir / final_evaluate.EVAL_SCORE_A_FILENAME).write_text(json.dumps(a))
    (out_dir / final_evaluate.EVAL_SCORE_B_FILENAME).write_text(json.dumps(b))
    _, report = final_evaluate.stage_evaluate(run, "/home/t-lab01/sunrun")
    assert report["checks"]["two_scores"]["ok"] is False


def test_native_parity_gate_values():
    assert final_contract.check_native_parity(1e-10, 1e-7)["ok"] is True
    assert final_contract.check_native_parity(1e-8, 0.0)["ok"] is False
    assert final_contract.check_native_parity(0.0, 2e-6)["ok"] is False


def test_cutoff_rule_strict_first_origin_minus_72h():
    assert protocol.ts_utc(final_contract.training_cutoff_for_first_origin(
        "2026-05-31T23:00:00Z")).isoformat() == protocol.ts_utc(
            "2026-05-28T23:00:00Z").isoformat()
    with pytest.raises(ValueError):
        final_contract.assert_training_purge("2026-05-28T23:00:00Z",
                                             "2026-05-31T23:00:00Z")


# ----------------------------------------------------------------------------
# Gap override: any required check False (others green) forces INVALID.
# ----------------------------------------------------------------------------

def _green_checks():
    return {name: {"ok": True, "detail": "synthetic green"}
            for name in final_evaluate.REQUIRED_EVALUATE_CHECKS}


def test_gap_override_preserves_clean_pass_and_unmet():
    checks, gaps = _green_checks(), []
    for code in (final_contract.VALID, final_contract.UNMET):
        out_code, _ = final_evaluate._apply_gap_override(
            code, {"code": code}, checks, gaps)
        assert out_code == code


def test_gap_override_any_single_required_check_false_is_invalid():
    for name in final_evaluate.REQUIRED_EVALUATE_CHECKS:
        for code in (final_contract.VALID, final_contract.UNMET):
            checks = _green_checks()
            checks[name] = {"ok": False, "detail": "synthetic red"}
            out_code, report = final_evaluate._apply_gap_override(
                code, {"code": code}, checks, [f"{name}: synthetic red"])
            assert out_code == final_contract.INVALID == 2, name
            assert "gap override" in report["reason"]


def test_gap_override_gaps_nonempty_with_all_green_is_invalid():
    out_code, _ = final_evaluate._apply_gap_override(
        final_contract.VALID, {"code": 0}, _green_checks(), ["x: y"])
    assert out_code == final_contract.INVALID == 2


# ----------------------------------------------------------------------------
# Ledger: normalized 7-category counts, exact IDs, no slot reuse.
# ----------------------------------------------------------------------------

def _ledger_claims():
    claims = []
    plan = [("reference_replay", 3), ("mean_level", 3), ("oof_base", 6),
            ("router_eval", 3)]
    n = 0
    for cat, count in plan:
        for _ in range(count):
            n += 1
            claims.append({"claim_id": f"{cat}-{n}", "category": cat,
                           "status": "ok"})
    for cid in final_model.FINAL_CLAIM_IDS:
        claims.append({"claim_id": cid, "category": "final", "status": "ok"})
    return claims


def _write_ledger(run, claims):
    from experiments.mse20_scope_expansion.protocol import ClaimLedger
    path = __import__("pathlib").Path(ClaimLedger(str(run)).path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"claims": claims}))
    return path


def test_ledger_counts_exact_18_all_ok(tmp_path):
    run = _bare_run(tmp_path)
    _write_ledger(run, _ledger_claims())
    by_cat, total, _ = final_evaluate._ledger_counts(run)
    assert total == 18
    assert by_cat == {"reference_replay": 3, "mean_level": 3,
                      "source_pilot": 0, "full_source_confirm": 0,
                      "oof_base": 6, "router_eval": 3, "final": 3}


def test_ledger_rejects_unknown_category(tmp_path):
    run = _bare_run(tmp_path)
    claims = _ledger_claims() + [{"claim_id": "zz-1", "category": "zz",
                                  "status": "ok"}]
    _write_ledger(run, claims)
    by_cat, _, detail = final_evaluate._ledger_counts(run)
    assert by_cat is None and "unknown ledger category" in detail


def test_ledger_rejects_duplicate_claim_ids(tmp_path):
    run = _bare_run(tmp_path)
    claims = _ledger_claims()
    claims.append(dict(claims[0]))
    _write_ledger(run, claims)
    by_cat, _, _ = final_evaluate._ledger_counts(run)
    assert by_cat is None


def test_ledger_rejects_non_ok_status(tmp_path):
    run = _bare_run(tmp_path)
    claims = _ledger_claims()
    claims[0] = dict(claims[0], status="failed")
    _write_ledger(run, claims)
    by_cat, _, _ = final_evaluate._ledger_counts(run)
    assert by_cat is None


def test_ledger_rejects_wrong_final_claim_set(tmp_path):
    run = _bare_run(tmp_path)
    claims = [c for c in _ledger_claims() if c["category"] != "final"]
    claims += [{"claim_id": "final_extra", "category": "final", "status": "ok"},
               {"claim_id": "final_direct", "category": "final", "status": "ok"},
               {"claim_id": "final_level", "category": "final", "status": "ok"}]
    _write_ledger(run, claims)
    by_cat, _, detail = final_evaluate._ledger_counts(run)
    assert by_cat is None and "final claim set" in detail


def test_evaluate_rejects_unknown_ledger_category(tmp_path):
    run = _bare_run(tmp_path)
    _write_ledger(run, _ledger_claims() + [{"claim_id": "zz-1",
                                            "category": "zz",
                                            "status": "ok"}])
    code, report = final_evaluate.stage_evaluate(run, "/home/t-lab01/sunrun")
    assert code == 2
    assert report["checks"]["claim_budget_18"]["ok"] is False


# ----------------------------------------------------------------------------
# Poison: post-cutoff change allowed, pre-cutoff prefix identical.
# ----------------------------------------------------------------------------

def test_poison_allows_post_cutoff_change_with_controls(tmp_path):
    run = _bare_run(tmp_path)
    fdir = final_model.final_dir(run)
    poison = {"exact_index_validated": True, "nan_mask_validated": True,
              "changed_features": ["some_post_cutoff_feature"],
              "pre_cutoff_changed": [],
              "positive_control_changed": ["some_post_cutoff_feature"],
              "positive_control_pre_changed": [],
              "cutoff_utc": protocol.ts_utc(
                  final_model.PINNED_TRAIN_CUTOFF_UTC).isoformat(),
              "official_causality": {"checked": True, "future_invariant": True,
                                     "representative_origins": ["a", "b", "c"]}}
    (fdir / final_model.FINAL_POISON_FILENAME).write_text(json.dumps(poison))
    _, report = final_evaluate.stage_evaluate(run, "/home/t-lab01/sunrun")
    assert report["checks"]["poison_causality"]["ok"] is True


def test_poison_rejects_pre_cutoff_change(tmp_path):
    run = _bare_run(tmp_path)
    fdir = final_model.final_dir(run)
    poison = {"exact_index_validated": True, "nan_mask_validated": True,
              "changed_features": ["f"], "pre_cutoff_changed": ["f"],
              "positive_control_changed": ["f"],
              "positive_control_pre_changed": [],
              "cutoff_utc": protocol.ts_utc(
                  final_model.PINNED_TRAIN_CUTOFF_UTC).isoformat(),
              "official_causality": {"checked": True, "future_invariant": True,
                                     "representative_origins": ["a", "b", "c"]}}
    (fdir / final_model.FINAL_POISON_FILENAME).write_text(json.dumps(poison))
    _, report = final_evaluate.stage_evaluate(run, "/home/t-lab01/sunrun")
    assert report["checks"]["poison_causality"]["ok"] is False


# ----------------------------------------------------------------------------
# Score pair: actual-shaped synthetic fixture, tamper each receipt.
# ----------------------------------------------------------------------------

def _score_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    snap = tmp_path / "snap"
    snap.mkdir()
    for name in ("folds", "truth", "naive"):
        pd.DataFrame({"x": [1]}).to_parquet(snap / f"{name}.parquet",
                                            index=False)
    champ = repo / final_evaluate.CHAMPION_REL
    champ.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": [2]}).to_parquet(champ, index=False)
    monkeypatch.setenv(final_evaluate.SNAPSHOT_ENV_VAR, str(snap))
    run = tmp_path / "run"
    fdir = final_model.final_dir(run)
    fdir.mkdir(parents=True)
    pred = pd.DataFrame(
        {"origin_last_input_utc": ["2026-05-31T23:00:00Z"],
         "horizon_hours": [1], "pred_kms": [400.0]})
    pred.to_parquet(fdir / final_model.FINAL_PREDICTIONS_FILENAME, index=False)
    return repo, run


def _score_bodies(repo, run):
    live_shas = {n: protocol.sha256_file(p)
                 for n, p in final_evaluate.snapshot_paths(repo).items()}
    live_shas["champion_replay"] = protocol.sha256_file(repo / final_evaluate.CHAMPION_REL)
    live_shas["predictions"] = protocol.sha256_file(
        final_model.final_dir(run) / final_model.FINAL_PREDICTIONS_FILENAME)
    cells = {c: 100.0 for c in final_contract.CELL_NAMES}
    official = {"n_pairs": 203976, "n_folds": 2833, "mse": 3000.0,
                "by_regime": {c: {"mse": 100.0} for c in final_contract.CELL_NAMES}}
    cand = {"tp": 1, "fp": 0, "fn": 2, "f1": 0.5}
    champ = {"tp": 1, "fp": 0, "fn": 2, "f1": 0.5}
    digest = final_evaluate.numeric_event_digest(3000.0, cells, cand, champ)
    pred_meta = {"prediction_digest": "d" * 64,
                 "logged_utc": "2026-10-04T00:00:00Z"}
    binding = {"prediction_sha256": live_shas["predictions"],
               "prediction_digest": "d" * 64}
    base = {"total_mse": 3000.0, "cells": cells,
            "candidate_event": cand, "champion_event": champ,
            "official": official, "score_digest": digest,
            "file_shas": live_shas, "binding": binding}
    a = dict(base, stage="score-a", logged_utc="2026-10-05T00:00:00Z")
    b = dict(base, stage="score-b", logged_utc="2026-10-05T00:00:01Z")
    return pred_meta, a, b


def test_score_pair_genuine_passes_both_recomputed(tmp_path, monkeypatch):
    repo, run = _score_repo(tmp_path, monkeypatch)
    pred_meta, a, b = _score_bodies(repo, run)
    ok, _ = final_evaluate.validate_score_pair(a, b, pred_meta, run, repo)
    assert ok is True


def test_score_pair_rejects_tampered_b_official(tmp_path, monkeypatch):
    repo, run = _score_repo(tmp_path, monkeypatch)
    pred_meta, a, b = _score_bodies(repo, run)
    b = json.loads(json.dumps(b))
    b["official"]["mse"] = 3001.0  # copied digest no longer matches B content
    ok, detail = final_evaluate.validate_score_pair(a, b, pred_meta, run, repo)
    assert ok is False
    assert "differ" in detail


def test_score_pair_rejects_tampered_b_cells(tmp_path, monkeypatch):
    repo, run = _score_repo(tmp_path, monkeypatch)
    pred_meta, a, b = _score_bodies(repo, run)
    b = json.loads(json.dumps(b))
    b["official"]["by_regime"][final_contract.CELL_NAMES[0]] = {"mse": 101.0}
    ok, _ = final_evaluate.validate_score_pair(a, b, pred_meta, run, repo)
    assert ok is False


def test_score_pair_rejects_forged_shared_digest(tmp_path, monkeypatch):
    repo, run = _score_repo(tmp_path, monkeypatch)
    pred_meta, a, b = _score_bodies(repo, run)
    a = dict(a, score_digest="c" * 64)
    b = dict(b, score_digest="c" * 64)
    ok, _ = final_evaluate.validate_score_pair(a, b, pred_meta, run, repo)
    assert ok is False


def test_score_pair_rejects_truth_before_freeze_order(tmp_path, monkeypatch):
    repo, run = _score_repo(tmp_path, monkeypatch)
    pred_meta, a, b = _score_bodies(repo, run)
    b = dict(b, logged_utc="2026-10-03T00:00:00Z")  # before predict freeze
    ok, _ = final_evaluate.validate_score_pair(a, b, pred_meta, run, repo)
    assert ok is False


def test_score_pair_rejects_changed_snapshot_bytes(tmp_path, monkeypatch):
    repo, run = _score_repo(tmp_path, monkeypatch)
    pred_meta, a, b = _score_bodies(repo, run)
    snap = __import__("pathlib").Path(
        __import__("os").environ[final_evaluate.SNAPSHOT_ENV_VAR])
    pd.DataFrame({"x": [9]}).to_parquet(snap / "truth.parquet", index=False)
    ok, _ = final_evaluate.validate_score_pair(a, b, pred_meta, run, repo)
    assert ok is False


# ----------------------------------------------------------------------------
# Regression receipt: stored JSON never trusted without live bytes.
# ----------------------------------------------------------------------------

def test_regression_receipt_rejects_missing_log(tmp_path):
    regr = {"ok": True, "exit_code": 0, "timed_out": False,
            "inventory": [], "log_path": str(tmp_path / "absent.log"),
            "log_sha256": "0" * 64}
    ok, _ = final_evaluate.validate_regression_receipt(regr, "/home/t-lab01/sunrun")
    assert ok is False


def test_regression_receipt_rejects_stale_inventory(tmp_path):
    live = final_evaluate.suite_inventory("/home/t-lab01/sunrun")
    regr = {"ok": True, "exit_code": 0, "timed_out": False,
            "inventory": live[:-1] if len(live) > 1 else [],
            "log_path": "", "log_sha256": "0" * 64,
            "command": [], "interpreter": "", "env": {},
            "n_tests": -1, "started_utc": "2026-10-05T00:00:01Z",
            "ended_utc": "2026-10-05T00:00:00Z",
            "source_hashes": {}, "closure_after": {},
            "registration_sources_match": False,
            "clock_plugin": final_evaluate.CLOCK_PLUGIN}
    ok, _ = final_evaluate.validate_regression_receipt(regr, "/home/t-lab01/sunrun")
    assert ok is False


def test_versions_probe_uses_live_interpreter():
    versions = final_evaluate._versions_for(__import__("sys").executable)
    assert set(("python", "pandas", "numpy", "sklearn", "xgboost")) <= set(versions)


def test_eval_closure_binds_producer_and_contract_sources():
    closure = final_evaluate.eval_closure_hashes("/home/t-lab01/sunrun")
    for rel in (final_model.FINAL_SOURCES
                + final_evaluate.EVAL_SOURCES
                + final_evaluate.EVAL_CONTRACT_SOURCES):
        assert len(closure[rel]) == 64


def test_full_native_replay_fail_closed_without_models(tmp_path):
    run = _bare_run(tmp_path)
    with pytest.raises(Exception):
        final_evaluate._full_native_replay(
            run, tmp_path, {"direct": {"columns": []}})


def test_native_parity_fail_closed_on_empty_run(tmp_path):
    _, report = final_evaluate.stage_evaluate(tmp_path / "run",
                                              "/home/t-lab01/sunrun")
    assert report["checks"]["native_parity"]["ok"] is False
def test_native_replay_uses_full_horizon_feature_grid():
    import inspect
    from experiments.mse20_scope_expansion import final_evaluate

    source = inspect.getsource(final_evaluate._full_native_replay)
    assert "official_direct_feature_grid" in source
    assert "assert_inference_schema(direct_grid, cols)" in source


def test_final_registration_binds_evaluator_closure():
    import inspect
    from experiments.mse20_scope_expansion import final_model

    source = inspect.getsource(final_model.stage_register)
    assert '"evaluator_closure": final_evaluate.eval_closure_hashes(repo_root)' in source

