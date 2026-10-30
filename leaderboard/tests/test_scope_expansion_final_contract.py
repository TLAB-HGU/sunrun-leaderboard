"""Synthetic CPU-only tests for the prospective final acceptance contract.

No real data, no fits, no registration, no claims, no network, no official
reads (snapshot hash appears only as a pinned constant, never fetched).
Every case exercises the actual ``final_contract`` pure helpers with fully
synthetic numerics/keys/digests and tmp-free in-memory dicts.
"""
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

PKG_PARENT = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(PKG_PARENT))

from mse20_scope_expansion import final_contract as fc  # noqa: E402
from mse20_scope_expansion import protocol, selection  # noqa: E402


def _origins(n=3):
    return [f"2026-06-0{d + 1}T00:00:00Z" for d in range(n)]


def _official_origins():
    base = datetime(2026, 5, 31, 23, 0, 0, tzinfo=timezone.utc)
    out = []
    for i in range(fc.OFFICIAL_ORIGINS):
        t = base + timedelta(hours=i)
        out.append(t.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return out


def _full_keys(origins=None):
    origins = origins or _origins()
    return [(o, h) for o in origins for h in range(1, 73)]


def _official_keys():
    return _full_keys(_official_origins())


def _cells(value=100.0):
    return {k: float(value) for k in fc.CELL_NAMES}


def _full_ledger(**over):
    base = {"reference_replay": 3, "mean_level": 3, "source_pilot": 0,
            "full_source_confirm": 0, "oof_base": 6, "router_eval": 3,
            "final": 0}
    base.update(over)
    return base


def _proof(official=False, **over):
    if official:
        origins = _official_origins()
        first = origins[0]
        train_max = "2026-05-28T22:00:00Z"
    else:
        origins = _origins()
        first = origins[0]
        train_max = "2026-05-28T23:00:00Z"
    keys = _full_keys(origins)
    digest = fc.canonical_numeric_digest({"total": 1.0})
    base = {
        "total_mse": 100.0,
        "cells": _cells(100.0),
        "candidate_f1": 0.10,
        "champion_f1": fc.CHAMPION_F1,
        "pred_keys": list(keys),
        "requested_keys": list(keys),
        "train_label_max_utc": train_max,
        "first_origin_utc": first,
        "max_abs_replay_diff": 1e-10,
        "mse_replay_diff": 1e-7,
        "score_digest_a": digest,
        "score_digest_b": digest,
        "ledger_by_category": _full_ledger(),
        "ledger_total": 15,
        "feature_columns": ["v_lag1", "ace_speed_kms"],
        "recipe": "router",
        "snapshot_sha256": fc.PINNED_SNAPSHOT_SHA256,
        "selection_winner": fc.FROZEN_SELECTION_WINNER,
        "selection_receipt_sha256": fc.FROZEN_SELECTION_RECEIPT_SHA256,
        "amendment_sha256": protocol.AMENDMENT_SHA256,
        "predictions_finite": True,
    }
    base.update(over)
    return base


def _evidence(**over):
    base = {"integrity_ok": True, "provenance_ok": True, "causality_ok": True,
            "regression_ok": True, "two_scores_ok": True}
    base.update(over)
    return base


# Frozen bindings (no mutation, single-sourced where immutable).
def test_frozen_bindings():
    assert fc.FROZEN_SELECTION_WINNER == "router"
    assert fc.FROZEN_SELECTION_RECEIPT_SHA256 == (
        "faa32862d366d8dd8e284e0b4ddffefbb8e042e7be82a5296985d11cb80697a2")
    assert fc.PINNED_SNAPSHOT_SHA256 == (
        "bcf5d5417d99eaa331ddca40581d48953e09a58e")
    assert fc.OFFICIAL_ORIGINS * fc.OFFICIAL_HORIZONS == fc.OFFICIAL_TOTAL_KEYS == 203976
    assert fc.TOTAL_MSE_MAX == 3289.0890348547073
    assert fc.PARAMS == protocol.PARAMS
    assert fc.CLAIM_ALLOCATION == protocol.CLAIM_ALLOCATION
    assert sum(fc.CLAIM_ALLOCATION.values()) == 36
    assert fc.CLAIM_ALLOCATION["final"] == 3
    assert fc.CHAMPION_F1 == pytest.approx(0.08333333333333333)
    assert fc.FROZEN_AMENDMENT_SHA256 == protocol.AMENDMENT_SHA256
    assert fc.EARLIEST_ORIGIN_UTC == "2026-05-31T23:00:00Z"
    assert fc.LATEST_ORIGIN_UTC == "2026-09-26T23:00:00Z"


# Total boundary: equality passes, just-over fails.
def test_total_boundary_equality_passes():
    assert fc.check_total_gate(fc.TOTAL_MSE_MAX)["ok"] is True
    assert fc.check_total_gate(math.nextafter(fc.TOTAL_MSE_MAX, math.inf))["ok"] is False


def test_total_nan_missing_fail():
    assert fc.check_total_gate(float("nan"))["ok"] is False
    assert fc.check_total_gate(float("inf"))["ok"] is False
    assert fc.check_total_gate(None)["ok"] is False
    assert fc.check_total_gate(True)["ok"] is False


def test_total_negative_invalid():
    assert fc.check_total_gate(-0.1)["ok"] is False
    assert fc.check_total_gate(-1e9)["ok"] is False
    code, _ = fc.scientific_verdict(-5.0, _cells(100.0), 0.10, fc.CHAMPION_F1)
    assert code == fc.INVALID


# Cells: strict <3800 (equality fails), all four required.
def test_cells_all_below_pass_equality_fails():
    assert fc.check_cells_gate(_cells(3799.999))["ok"] is True
    bad = _cells(100.0)
    bad["high_high_high"] = 3800.0
    assert fc.check_cells_gate(bad)["ok"] is False


def test_cells_missing_or_nan_fail():
    assert fc.check_cells_gate({"low_low_high": 1.0})["ok"] is False
    bad = _cells(100.0)
    bad["low_low_high"] = float("nan")
    assert fc.check_cells_gate(bad)["ok"] is False


def test_cells_negative_invalid_and_verdict():
    bad = _cells(100.0)
    bad["low_low_high"] = -1.0
    assert fc.check_cells_gate(bad)["ok"] is False
    code, _ = fc.scientific_verdict(100.0, bad, 0.10, fc.CHAMPION_F1)
    assert code == fc.INVALID
    code, _ = fc.scientific_verdict(100.0, {"low_low_high": 1.0}, 0.10,
                                    fc.CHAMPION_F1)
    assert code == fc.INVALID


# F1: equality passes, below fails, undefined invalid.
def test_f1_equality_passes_below_fails():
    assert fc.check_f1_gate(fc.CHAMPION_F1, fc.CHAMPION_F1)["ok"] is True
    assert fc.check_f1_gate(fc.CHAMPION_F1 - 1e-9, fc.CHAMPION_F1)["ok"] is False


def test_f1_undefined_fails_and_pooled_counts():
    assert fc.check_f1_gate(None, fc.CHAMPION_F1)["ok"] is False
    assert fc.pooled_f1_from_counts([0], [0], [0]) is None
    assert fc.pooled_f1_from_counts([1], [1], [1]) == pytest.approx(0.5)
    got = fc.pooled_f1_from_counts([1, 1], [1, 1], [1, 1])
    avg_of_ratios_note = "pooling ratios, never mean-of-F1"
    assert got == pytest.approx(0.5), avg_of_ratios_note
    assert selection.pooled_f1([1], [1], [1]) == pytest.approx(0.5)


def test_f1_range_confined():
    assert fc.check_f1_gate(1.5, fc.CHAMPION_F1)["ok"] is False
    assert fc.check_f1_gate(-0.1, fc.CHAMPION_F1)["ok"] is False
    assert fc.check_f1_gate(0.10, 1.5)["ok"] is False
    assert fc.check_f1_gate(float("nan"), fc.CHAMPION_F1)["ok"] is False
    code, _ = fc.scientific_verdict(100.0, _cells(100.0), 1.5, fc.CHAMPION_F1)
    assert code == fc.INVALID
    code, _ = fc.scientific_verdict(100.0, _cells(100.0), -0.1, fc.CHAMPION_F1)
    assert code == fc.INVALID


# Grid: exact keys, integer 1..72h, no drop/intersection.
def test_exact_grid_pass_and_tamper_reject():
    keys = _full_keys()
    assert fc.assert_exact_final_grid(keys, list(keys))
    missing = keys[:-1]
    with pytest.raises(ValueError):
        fc.assert_exact_final_grid(missing, keys)
    with pytest.raises(ValueError):
        fc.assert_exact_final_grid(keys + [keys[0]], keys + [keys[0]][:0] + keys)
    tampered = list(keys)
    tampered[0] = ("2026-06-09T00:00:00Z", 1)
    with pytest.raises(ValueError):
        fc.assert_exact_final_grid(tampered, keys)
    dup = list(keys)
    dup[1] = dup[0]
    with pytest.raises(ValueError):
        fc.assert_exact_final_grid(dup, keys)


def test_horizon_support_rejects_noninteger_and_range():
    keys = _full_keys()
    bad = [(o, float(h)) for o, h in keys]
    with pytest.raises(ValueError):
        fc.assert_exact_final_grid(bad, keys)
    bad2 = list(keys)
    bad2[0] = (bad2[0][0], 0)
    with pytest.raises(ValueError):
        fc.assert_exact_final_grid(bad2, keys)
    bad3 = list(keys)
    bad3[0] = (bad3[0][0], 73)
    with pytest.raises(ValueError):
        fc.assert_exact_final_grid(bad3, keys)
    short = [(o, h) for o, h in keys if h <= 71]
    with pytest.raises(ValueError):
        fc.assert_horizon_support_per_origin(short, _origins())


def test_grid_rejects_bool_malformed_and_equivalent_duplicates():
    keys = _full_keys()
    bad = list(keys)
    bad[0] = (bad[0][0], True)
    with pytest.raises(ValueError):
        fc.assert_exact_final_grid(bad, keys)
    badt = list(keys)
    badt[0] = ("not-a-time", 1)
    with pytest.raises(ValueError):
        fc.assert_exact_final_grid(badt, keys)
    naive = list(keys)
    naive[0] = ("2026-06-01T00:00:00", 1)
    with pytest.raises(ValueError):
        fc.assert_exact_final_grid(naive, keys)
    # Equivalent Z/+00:00 spellings normalize to duplicates.
    dup = list(keys)
    alt = keys[0][0].replace("Z", "+00:00")
    assert alt != keys[0][0]
    dup[1] = (alt, keys[0][1])
    with pytest.raises(ValueError):
        fc.assert_exact_final_grid(dup, keys)


# Purge: strict first_origin - 72h (equality fails).
def test_training_purge_strict():
    first = "2026-06-01T00:00:00Z"
    cutoff = fc.training_cutoff_for_first_origin(first)
    assert str(cutoff.isoformat()) == "2026-05-29T00:00:00+00:00"
    fc.assert_training_purge("2026-05-28T23:00:00Z", first)
    with pytest.raises(ValueError):
        fc.assert_training_purge("2026-05-29T00:00:00Z", first)
    with pytest.raises(ValueError):
        fc.assert_training_purge(None, first)


# Native parity + two identical canonical digests (no rounded scores).
def test_native_parity_boundaries():
    assert fc.check_native_parity(1e-9, 1e-6)["ok"] is True
    assert fc.check_native_parity(1e-9 + 1e-12, 1e-6)["ok"] is False
    assert fc.check_native_parity(1e-9, 1e-6 + 1e-9)["ok"] is False


def test_native_parity_negative_fails():
    assert fc.check_native_parity(-1e-10, 1e-7)["ok"] is False
    assert fc.check_native_parity(1e-10, -1e-7)["ok"] is False
    assert fc.check_native_parity(float("nan"), 1e-7)["ok"] is False


def test_two_identical_scores_exact():
    a = fc.canonical_numeric_digest({"mse": 3289.0890348547073})
    b = fc.canonical_numeric_digest({"mse": 3289.0890348547073})
    fc.assert_two_identical_scores(a, b)
    rounded = fc.canonical_numeric_digest({"mse": 3289.089})
    with pytest.raises(ValueError):
        fc.assert_two_identical_scores(a, rounded)
    with pytest.raises(ValueError):
        fc.assert_two_identical_scores(a, None)


def test_score_digests_require_hex64():
    with pytest.raises(ValueError):
        fc.assert_two_identical_scores("same", "same")
    with pytest.raises(ValueError):
        fc.assert_two_identical_scores("0" * 63, "0" * 63)
    with pytest.raises(ValueError):
        fc.assert_two_identical_scores("G" * 64, "G" * 64)


# Proof structure: boolean-only / partial proofs never pass.
def test_boolean_only_and_partial_proof_invalid():
    code, _ = fc.evaluate_final_contract(True, _evidence())
    assert code == fc.INVALID
    code, _ = fc.evaluate_final_contract({"final_proof": True}, _evidence())
    assert code == fc.INVALID
    partial = _proof()
    del partial["cells"]
    code, _ = fc.evaluate_final_contract(partial, _evidence())
    assert code == fc.INVALID


def test_proof_requires_pins_champion_predictions():
    for key in ("champion_f1", "snapshot_sha256", "selection_winner",
                "selection_receipt_sha256", "amendment_sha256",
                "predictions_finite", "recipe"):
        p = _proof(official=True)
        del p[key]
        code, _ = fc.evaluate_final_contract(p, _evidence())
        assert code == fc.INVALID, key
    bad = _proof(official=True, snapshot_sha256="0" * 40)
    code, _ = fc.evaluate_final_contract(bad, _evidence())
    assert code == fc.INVALID
    bad = _proof(official=True, amendment_sha256="0" * 64)
    code, _ = fc.evaluate_final_contract(bad, _evidence())
    assert code == fc.INVALID
    bad = _proof(official=True, predictions_finite=False)
    code, _ = fc.evaluate_final_contract(bad, _evidence())
    assert code == fc.INVALID


# Failure claims consume; rejected source quota; no official features.
def test_claim_budget_failures_consume_no_retry():
    ok = fc.check_claim_budget(_full_ledger(final=3), 18)
    assert ok["ok"] is True
    over = fc.check_claim_budget(_full_ledger(final=4), 19)
    assert over["ok"] is False
    over_global = fc.check_claim_budget(
        {k: v for k, v in _full_ledger().items()} | {"final": 3}, 37)
    # Build an over-global ledger by bumping a category past allocation.
    bumped = _full_ledger(oof_base=6, router_eval=3, final=3)
    bumped["reference_replay"] = 4
    assert fc.check_claim_budget(bumped, 16)["ok"] is False
    mismatch = fc.check_claim_budget(_full_ledger(final=3), 16)
    assert mismatch["ok"] is False


def test_claim_budget_strict_categories():
    assert fc.check_claim_budget({"final": 3}, 3)["ok"] is False
    bad = _full_ledger()
    bad["mystery"] = 0
    assert fc.check_claim_budget(bad, 15)["ok"] is False
    bad = _full_ledger()
    bad["source_pilot"] = 1
    assert fc.check_claim_budget(bad, 16)["ok"] is False
    bad = _full_ledger()
    bad["oof_base"] = 7
    assert fc.check_claim_budget(bad, 16)["ok"] is False
    assert fc.check_claim_budget(_full_ledger(), 15.0)["ok"] is False
    assert fc.check_claim_budget(_full_ledger(), "15")["ok"] is False
    flot = _full_ledger()
    flot["final"] = 1.0
    assert fc.check_claim_budget(flot, 16)["ok"] is False
    neg = _full_ledger()
    neg["final"] = -1
    assert fc.check_claim_budget(neg, 14)["ok"] is False


def test_source_closure_rejects_official_and_mean_alone():
    assert fc.check_source_closure(["v_lag1"], "router")["ok"] is True
    assert fc.check_source_closure(["official_speed"], "router")["ok"] is False
    assert fc.check_source_closure(["enlil_x"], "router")["ok"] is False
    assert fc.check_source_closure(["v_lag1"], "mean")["ok"] is False
    # Router-internal mean component stays allowed.
    assert fc.check_source_closure(["traj_mean", "v_lag1"], "router")["ok"] is True


def test_source_closure_empty_unknown():
    assert fc.check_source_closure([], "router")["ok"] is False
    assert fc.check_source_closure(None, "router")["ok"] is False
    assert fc.check_source_closure(["v_lag1"], "mystery")["ok"] is False
    assert fc.check_source_closure(["v_lag1"], None)["ok"] is False
    assert fc.check_source_closure(["  "], "router")["ok"] is False


# Full verdicts: PASS / UNMET / INVALID separation (official 2833x72 scale).
def test_full_pass_requires_integrity_evidence():
    code, _ = fc.evaluate_final_contract(_proof(official=True), _evidence())
    assert code == fc.VALID
    # Same scientific pass but no real-IO evidence -> INVALID, not acceptance.
    code, rep = fc.evaluate_final_contract(_proof(official=True), {})
    assert code == fc.INVALID
    assert rep["scientific_code"] == fc.VALID


def test_tiny_grid_never_passes():
    code, _ = fc.evaluate_final_contract(_proof(official=False), _evidence())
    assert code == fc.INVALID


def test_integrity_complete_science_fail_is_unmet():
    bad = _proof(official=True, total_mse=fc.TOTAL_MSE_MAX + 1.0)
    code, rep = fc.evaluate_final_contract(bad, _evidence())
    assert code == fc.UNMET
    assert rep["scientific_code"] == fc.UNMET


def test_science_undefined_is_invalid_even_with_evidence():
    bad = _proof(official=True, total_mse=float("nan"))
    code, _ = fc.evaluate_final_contract(bad, _evidence())
    assert code == fc.INVALID
