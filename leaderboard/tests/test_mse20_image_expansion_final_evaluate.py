"""Adversarial tests for image-expansion FINAL verifier (synthetic fixtures only).

No fits, no official reads, no network. Production CLI forbids test-only
bypass: fixtures live in tmp_path and never touch real runs.
"""
import hashlib
import json
import math
import subprocess
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


def _score_doc(total=100.0, created="2026-10-02T00:00:00Z"):
    cells = {k: 100.0 for k in fc.CELL_NAMES}
    body = {"total_mse": total, "cells": cells,
            "candidate": [[10, 2, 1], [2, 0, 1], [1, 1, 1]][0:1] and [[10, 2, 1]] or [[10, 2, 1]]}
    # counts stored flat below; digest over canonical body
    doc = {"total_mse": total, "cells": cells,
           "candidate_tp": [10, 2, 1], "candidate_fp": [2, 0, 1], "candidate_fn": [1, 1, 1],
           "champion_tp": [1, 0, 0], "champion_fp": [2, 2, 2], "champion_fn": [3, 3, 3],
           "created_utc": created}
    doc["digest"] = ic.canonical_numeric_digest(
        {"total_mse": total, "cells": cells,
         "candidate": [doc["candidate_tp"], doc["candidate_fp"], doc["candidate_fn"]],
         "champion": [doc["champion_tp"], doc["champion_fp"], doc["champion_fn"]]})
    return doc


def _scaffold(root: Path, **over):
    run = root / "mse20-image-expansion" / "run-20261005"
    bound = run / "bound"
    bound.mkdir(parents=True, exist_ok=True)
    base = datetime(2026, 6, 1, tzinfo=timezone.utc)
    origins = []
    for i in range(2833):
        o = (base + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        b = (base + timedelta(hours=i) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        origins.append({"origin_utc": o, "file_id": f"f{i % 3}",
                        "obs_end_utc": b, "avail_utc": b, "status": "ok"})
    score = _score_doc()
    docs = {
        "freeze.json": {"freeze_utc": "2026-10-01T00:00:00Z",
                        "prediction_sha256": HEX, "key_digest": "cd" * 32},
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
    docs.update(over.get("docs", {}))
    for nm, obj in docs.items():
        (bound / nm).write_text(json.dumps(obj), encoding="utf-8")
    for nm in ("predictions.parquet", "replay.parquet"):
        if nm not in over.get("parquet", {}):
            (bound / nm).write_bytes(b"dummy")
    for nm, payload in over.get("parquet", {}).items():
        (bound / nm).write_bytes(payload)
    return run


# --- CLI fail-closed ----------------------------------------------------------

def test_cli_exit2_empty_run(tmp_path):
    run = tmp_path / "mse20-image-expansion" / "run-20261005"
    out = evaluate.evaluate_run(str(run))
    assert out["exit"] == 2 and "missing-required-verifier" in out["reason"]


def test_cli_rejects_outside_namespace(tmp_path):
    out = evaluate.evaluate_run(str(tmp_path / "other-run"))
    assert out["exit"] == 2 and "outside new namespace" in out["reason"]


def test_traversal_and_symlink_rejected(tmp_path):
    run = _scaffold(tmp_path)
    with pytest.raises(ValueError):
        evaluate._resolve_bound(run, "../evil.json")
    with pytest.raises(ValueError):
        evaluate._resolve_bound(run, "/abs.json")
    link = run / "bound" / "predictions.parquet"
    link.unlink()
    target = tmp_path / "outside.parquet"
    target.write_bytes(b"x")
    link.symlink_to(target)
    with pytest.raises(ValueError):
        evaluate._resolve_bound(run, "predictions.parquet")


def test_malicious_ok_without_proof_fails(tmp_path):
    run = _scaffold(tmp_path, docs={"closure.json": {"ok": True}})
    out = evaluate.evaluate_run(str(run))
    assert out["exit"] == 2 and "boolean-only" in out["reason"]


def test_edited_closure_hash_fails(tmp_path):
    bad = {"plan_sha256": "00" * 32, "amendment_sha256": ic.AMENDMENT_SHA256,
           "snapshot_sha256": ic.PINNED_SNAPSHOT_SHA256,
           "evaluator_sha256": evaluate._evaluator_live_sha(),
           "source_shas": {"suvi_fe195": HEX}}
    run = _scaffold(tmp_path, docs={"closure.json": bad})
    out = evaluate.evaluate_run(str(run))
    assert out["exit"] == 2 and "plan" in out["reason"].lower()


def test_old_ledger_quota_rejected(tmp_path):
    bad_ledger = {"new": {"a_per_period": 3, "a_oof_conditional": 3,
                          "cnn_conditional": 3, "final_constituents": 2},
                  "new_total": 11, "old_ledger": {"total": 18, "status": "REUSED"}}
    assert ic.check_image_claim_budget(bad_ledger)["ok"] is False
    old_as_new = {"new": {"a_per_period": 3, "a_oof_conditional": 3,
                          "cnn_conditional": 3, "final_constituents": 9},
                  "new_total": 18, "old_ledger": {"total": 18, "status": "CLOSED"}}
    assert ic.check_image_claim_budget(old_as_new)["ok"] is False
    run = _scaffold(tmp_path, docs={"ledger.json": bad_ledger})
    out = evaluate.evaluate_run(str(run))
    assert out["exit"] == 2


def test_false_pubdate_rejected(tmp_path):
    bad = {"files": [{"file_id": "f0", "obs_end_utc": "2026-05-01T00:00:00Z",
                      "avail_utc": "2026-05-01T01:00:00Z", "sha256": HEX,
                      "avail_source": "mtime", "verified_available": True}]}
    run = _scaffold(tmp_path, docs={"provenance.json": bad})
    out = evaluate.evaluate_run(str(run))
    assert out["exit"] == 2 and "independently verified" in out["reason"]


def test_future_source_breach(tmp_path):
    assert protocol.image_eligible("2026-01-01T01:00:00Z", "2025-12-31T23:00:00Z",
                                  "2026-01-01T00:00:00Z") is False
    run = _scaffold(tmp_path)
    caus = json.loads((run / "bound" / "causality.json").read_text())
    caus["origins"][0] = {"origin_utc": caus["origins"][0]["origin_utc"], "file_id": "f9",
                          "obs_end_utc": "2027-01-01T00:00:00Z",
                          "avail_utc": caus["origins"][0]["avail_utc"], "status": "ok"}
    (run / "bound" / "causality.json").write_text(json.dumps(caus))
    out = evaluate.evaluate_run(str(run))
    assert out["exit"] == 1 and "causality" in out["reason"].lower()


def test_official_f1_mixup_rejected(tmp_path):
    run = _scaffold(tmp_path)
    elig = json.loads((run / "bound" / "eligibility.json").read_text())
    elig["f1_baseline_tag"] = "official-champion-48c28"
    (run / "bound" / "eligibility.json").write_text(json.dumps(elig))
    out = evaluate.evaluate_run(str(run))
    assert out["exit"] == 2 and "preofficial-direct" in out["reason"]


def test_freeze_after_score_rejected(tmp_path):
    run = _scaffold(tmp_path)
    fr = json.loads((run / "bound" / "freeze.json").read_text())
    fr["freeze_utc"] = "2026-10-05T00:00:00Z"
    (run / "bound" / "freeze.json").write_text(json.dumps(fr))
    out = evaluate.evaluate_run(str(run))
    assert out["exit"] in (1, 2) and "freeze" in out["reason"].lower()


def test_score_pair_mismatch_rejected(tmp_path):
    run = _scaffold(tmp_path)
    sb = json.loads((run / "bound" / "score_b.json").read_text())
    sb["total_mse"] = 999.0
    (run / "bound" / "score_b.json").write_text(json.dumps(sb))
    out = evaluate.evaluate_run(str(run))
    assert out["exit"] == 2


def test_parquet_count_mismatch_cli(tmp_path):
    pytest.importorskip("pandas")
    import pandas as pd
    df = pd.DataFrame({"origin_utc": ["2026-06-01T00:00:00Z"] * 4,
                       "h": [1, 2, 3, 4], "y_pred": [1.0, 2.0, 3.0, 4.0]})
    run = _scaffold(tmp_path, parquet={"predictions.parquet": df.to_parquet(),
                                       "replay.parquet": df.to_parquet()})
    out = evaluate.evaluate_run(str(run))
    assert out["exit"] == 2 and "203976" in out["reason"]


# --- pure-contract boundaries ---------------------------------------------------

def test_total_cells_boundaries_pure():
    assert ic.check_total_gate(3289.0890348547073)["ok"] is True
    assert ic.check_total_gate(math.nextafter(3289.0890348547073, math.inf))["ok"] is False
    bad = {k: 100.0 for k in fc.CELL_NAMES}
    bad["high_high_high"] = 3800.0
    assert ic.check_cells_gate(bad)["ok"] is False
    good = {k: 3799.999 for k in fc.CELL_NAMES}
    assert ic.check_cells_gate(good)["ok"] is True
    assert ic.check_total_gate(float("nan"))["ok"] is False


def test_grid_faults_pure():
    req = [[f"2026-06-{1 + (i // 72):02d}T00:00:00Z", (i % 72) + 1] for i in range(10)]
    with pytest.raises(ValueError):
        ic.assert_exact_image_grid(req, req)  # wrong count, not 203976
    dup = [["2026-06-01T00:00:00Z", 1], ["2026-06-01T00:00:00Z", 1]]
    with pytest.raises(ValueError):
        ic.assert_exact_image_grid(dup, dup)
    badh = [["2026-06-01T00:00:00Z", 0]]
    with pytest.raises(ValueError):
        ic.assert_exact_image_grid(badh, badh)


def test_contract_unmet_vs_invalid():
    keys = [["2026-06-01T00:00:00Z", 1]]
    proof = {"total_mse": 1e12, "cells": {k: 100.0 for k in fc.CELL_NAMES},
             "candidate_tp": [1, 0, 0], "candidate_fp": [0, 0, 0], "candidate_fn": [0, 0, 0],
             "champion_tp": [1, 0, 0], "champion_fp": [0, 0, 0], "champion_fn": [0, 0, 0],
             "pred_keys": keys, "requested_keys": keys, "snapshot_sha256": ic.PINNED_SNAPSHOT_SHA256,
             "plan_sha256": ic.PLAN_SHA256, "amendment_sha256": ic.AMENDMENT_SHA256,
             "evaluator_sha256": HEX, "source_shas": {"s": HEX},
             "freeze_utc": "2026-10-01T00:00:00Z", "score_utc_a": "2026-10-02T00:00:00Z",
             "score_utc_b": "2026-10-02T00:00:00Z", "score_digest_a": HEX, "score_digest_b": HEX,
             "ledger": {"new": dict.fromkeys(ic.IMAGE_FIT_BREAKDOWN, 0), "new_total": 0,
                        "old_ledger": {"total": 18, "status": "CLOSED"}},
             "feature_columns": list(protocol.IMAGE_FEATURE_SCHEMA),
             "max_abs_replay_diff": 0.0, "mse_replay_diff": 0.0}
    ev = {k: True for k in ("integrity_ok", "provenance_ok", "causality_ok",
                            "regression_ok", "two_scores_ok", "native_parity_ok")}
    code, _ = ic.evaluate_image_final_contract({"final_proof": True}, ev)
    assert code == 2
    thin = dict(proof)
    del thin["total_mse"]
    code2, _ = ic.evaluate_image_final_contract(thin, ev)
    assert code2 == 2


def test_f1_identity_and_no_bypass_flag(tmp_path):
    assert ic.pooled_f1 is selection.pooled_f1
    src = Path(evaluate.__file__).read_text()
    assert "--allow-synthetic" not in src and "allow_test_bypass" not in src
    run = tmp_path / "mse20-image-expansion" / "run-20261005"
    cmd = [sys.executable, "-m", "mse20_image_expansion.evaluate", "--run-dir", str(run)]
    env = {"PYTHONPATH": str(PKG_PARENT), "PATH": "/usr/bin:/bin"}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 2
