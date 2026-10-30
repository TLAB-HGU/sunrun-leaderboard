"""Hostile trust tests: forged perfect artifacts must NEVER exit 0.

Covers the 20261005 root-inspection gaps: provenance without publication
receipts, self-compared grids, score-JSON-trusted numerics/events, byte-copied
pred/replay without frozen-model proof, and self-reported regression suites.
All tests fail closed (exit 2, never 0) until actual native/instrumented
verifiers exist. Pure gate checks below are pure logic, not acceptance.
"""

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

PKG_PARENT = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(PKG_PARENT))

from mse20_image_expansion import evaluate, final_contract as ic, protocol  # noqa: E402
from mse20_scope_expansion import final_contract as fc  # noqa: E402
from mse20_scope_expansion import selection  # noqa: E402

HEX = "ab" * 32
BASE = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _origins(n=2833):
    out = []
    for i in range(n):
        o = (BASE + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        b = (BASE + timedelta(hours=i) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        out.append({"origin_utc": o, "file_id": f"f{i % 3}",
                    "obs_end_utc": b, "avail_utc": b, "status": "ok"})
    return out


def _score_doc(total=100.0, created="2026-10-02T00:00:00Z"):
    cells = {k: 100.0 for k in fc.CELL_NAMES}
    doc = {"total_mse": total, "cells": cells,
           "candidate_tp": [10, 2, 1], "candidate_fp": [2, 0, 1], "candidate_fn": [1, 1, 1],
           "champion_tp": [1, 0, 0], "champion_fp": [2, 2, 2], "champion_fn": [3, 3, 3],
           "created_utc": created}
    doc["digest"] = ic.canonical_numeric_digest(
        {"total_mse": total, "cells": cells,
         "candidate": [doc["candidate_tp"], doc["candidate_fp"], doc["candidate_fn"]],
         "champion": [doc["champion_tp"], doc["champion_fp"], doc["champion_fn"]]})
    return doc


def _forged_perfect_run(root: Path):
    """All 13 bound artifacts forged self-consistent: valid shapes/hashes,
    perfect gates, verified flags, byte-identical pred+replay."""
    pd = pytest.importorskip("pandas")
    run = root / "mse20-image-expansion" / "run-20261005"
    bound = run / "bound"
    bound.mkdir(parents=True, exist_ok=True)
    origins = _origins()
    frame = pd.DataFrame(
        {"origin_utc": [r["origin_utc"] for r in origins for _ in range(72)],
         "h": list(range(1, 73)) * len(origins),
         "y_pred": [float((i % 100) * 0.5 + (i % 72) * 0.01)
                    for i in range(len(origins) * 72)]})
    payload = frame.to_parquet()
    (bound / "predictions.parquet").write_bytes(payload)
    (bound / "replay.parquet").write_bytes(payload)  # copy-identical forgery
    pred_sha = hashlib.sha256(payload).hexdigest()
    keys = [[r["origin_utc"], h] for r in origins for h in range(1, 73)]
    key_digest = ic.assert_exact_image_grid(keys, keys)
    score = _score_doc()
    docs = {
        "freeze.json": {"freeze_utc": "2026-10-01T00:00:00Z",
                        "prediction_sha256": pred_sha, "key_digest": key_digest},
        "closure.json": {"plan_sha256": ic.PLAN_SHA256, "amendment_sha256": ic.AMENDMENT_SHA256,
                         "snapshot_sha256": ic.PINNED_SNAPSHOT_SHA256,
                         "evaluator_sha256": evaluate._evaluator_live_sha(),
                         "source_shas": {"suvi_fe195": HEX}},
        "ledger.json": {"new": {"a_per_period": 1, "a_oof_conditional": 0,
                                "cnn_conditional": 0, "final_constituents": 0},
                        "new_total": 1,
                        "old_ledger": {"total": 18, "status": "CLOSED"}},
        "eligibility.json": {"relatives_vs_direct_a": {"dev": -0.1, "selection": -0.1, "audit": -0.1},
                             "relatives_vs_direct_b": {"dev": -0.1, "selection": -0.1, "audit": -0.1},
                             "relatives_vs_a": {"dev": -0.05, "selection": -0.05, "audit": -0.05},
                             "candidate_tp": [10, 2, 1], "candidate_fp": [2, 0, 1],
                             "candidate_fn": [1, 1, 1],
                             "direct_tp": [5, 1, 0], "direct_fp": [4, 3, 2], "direct_fn": [5, 4, 3],
                             "f1_baseline_tag": "preofficial-direct",
                             "a_positive": True, "b_eligible": True},
        "causality.json": {"origins": origins},
        "provenance.json": {"files": [{"file_id": "f0", "obs_end_utc": "2026-05-01T00:00:00Z",
                                       "avail_utc": "2026-05-01T01:00:00Z", "sha256": HEX,
                                       "avail_source": "independent-verification",
                                       "verified_available": True}]},
        "support.json": {"snapshot_sha256": ic.PINNED_SNAPSHOT_SHA256, "total_keys": 203976},
        "truth_hash.json": {"snapshot_sha256": ic.PINNED_SNAPSHOT_SHA256,
                            "truth_sha256": HEX, "support_sha256": HEX},
        "score_a.json": dict(score),
        "score_b.json": dict(score),
        "regression.json": {"suite_total": 5, "suite_passed": 5,
                            "created_utc": "2026-10-03T00:00:00Z",
                            "numeric_digest": score["digest"]},
    }
    for nm, obj in docs.items():
        (bound / nm).write_text(json.dumps(obj), encoding="utf-8")
    return run


# --- Test A: forged-all-13 can never PASS ------------------------------------

def test_forged_all13_perfect_artifacts_cannot_exit_zero(tmp_path):
    run = _forged_perfect_run(tmp_path)
    out = evaluate.evaluate_run(str(run))
    assert out["exit"] != 0
    assert out["exit"] == 2
    assert "missing-required-verifier" in out["reason"]
    for needle in ("native-scoring", "native-event", "publication-receipt",
                   "frozen-model", "live-regression"):
        assert needle in out["reason"], f"missing proof not named: {needle}"


def test_forged_all13_cli_cannot_exit_zero(tmp_path):
    run = _forged_perfect_run(tmp_path)
    rc = evaluate.main(["--run-dir", str(run)])
    assert rc == 2


# --- Test B: missing-verifier gate fails closed -------------------------------

def test_production_verifiers_disallow_pass():
    gate = evaluate.check_production_verifiers()
    assert gate["pass_allowed"] is False
    assert len(gate["gaps"]) == 5
    blob = ";".join(gate["gaps"])
    for needle in ("native-scoring", "native-event", "publication-receipt",
                   "frozen-model", "live-regression"):
        assert needle in blob


def test_empty_run_dir_fails_closed_without_receipts(tmp_path):
    run = tmp_path / "mse20-image-expansion" / "run-20261005"
    run.mkdir(parents=True)
    out = evaluate.evaluate_run(str(run))
    assert out["exit"] == 2
    assert "missing-required-verifier" in out["reason"]


def test_missing_empty_inputs_never_crash():
    for bad in (None, "", 123, b"bytes"):
        out = evaluate.evaluate_run(bad)
        assert out["exit"] == 2
        assert "missing-required-verifier" in out["reason"]


def test_no_hidden_test_bypass_flag():
    src = Path(evaluate.__file__).read_text(encoding="utf-8")
    assert "--allow-synthetic" not in src
    assert "allow_test_bypass" not in src
    assert "test_only_pass" not in src


# --- pure gates preserved as pure logic (not acceptance) -----------------------

def test_pure_gates_still_pure():
    assert protocol.check_total_gate(100.0)["ok"] is True
    assert protocol.check_total_gate(float("nan"))["ok"] is False
    good = {k: 100.0 for k in fc.CELL_NAMES}
    assert protocol.check_cells_gate(good)["ok"] is True
    assert ic.pooled_f1 is selection.pooled_f1
