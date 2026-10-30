"""Synthetic CPU-only tests for the v3 native verification overlay.

No real fits, registration, fetching, official reads, uploads, or full
real-run execution. The root runs the real stages; every case here uses
synthetic frames/dicts only.
"""

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PKG_PARENT = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(PKG_PARENT))

from mse20_scope_expansion import mean_level, protocol  # noqa: E402
from mse20_scope_expansion import router_verify_v3 as v3  # noqa: E402
from mse20_scope_expansion import router_v2 as v2  # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _synthetic_inputs(n_origins=24, origin0="2024-10-05"):
    origins = pd.date_range(origin0, periods=n_origins, freq="6h", tz="UTC")
    rng = np.random.default_rng(7)
    feats = pd.DataFrame(
        {
            "v_lag0": rng.normal(0, 1, len(origins)),
            "ace_speed_kms_last": 400.0 + rng.normal(0, 1, len(origins)),
        },
        index=origins,
    )
    t0 = (origins.min() - pd.Timedelta(days=30)).floor("h")
    t1 = origins.max() + pd.Timedelta(hours=72)
    idx = pd.date_range(t0, t1, freq="h", tz="UTC")
    vals = 400 + 10 * np.sin(np.arange(len(idx)) / 24 * 2 * np.pi)
    truth = pd.Series(vals, index=idx)
    return {"origins": origins, "features": feats, "truth_observed": truth, "ace_filled": truth.copy()}


def _claims(cats):
    return [{"claim_id": f"c{i}", "category": c, "status": "ok"} for i, c in enumerate(cats)]


# ----------------------------------------------------------------------------
# 1. legacy wrong count/max overlaid only when digests + both cutoffs agree
# ----------------------------------------------------------------------------
def test_legacy_wrong_count_max_overlaid_only_when_digests_and_both_cutoffs_agree():
    cutoff = protocol.ts_utc("2024-10-01T00:00:00Z") - pd.Timedelta(hours=72)
    legacy_max = (cutoff - pd.Timedelta(hours=5)).isoformat()
    actual_max = (cutoff - pd.Timedelta(hours=9)).isoformat()
    rec = {
        "n_train_rows": 322681,
        "train_label_max_utc": legacy_max,
        "train_feature_digest": "featA",
        "train_label_digest": "labA",
    }
    rep = v3.legacy_level_metadata_report(
        rec,
        actual_n_train_rows=5800,
        actual_train_label_max_utc=actual_max,
        actual_train_feature_digest="featA",
        actual_train_label_digest="labA",
        cutoff=cutoff,
        receipt_sha256="abc",
    )
    assert rep["ok"] is True
    assert rep["legacy_metadata_mismatch_declared"] is True
    assert rep["legacy_purge_ok"] and rep["actual_purge_ok"]
    assert rep["mechanical_metadata_bug_only"] is True
    assert rep["legacy_receipt_sha256"] == "abc"
    # Fitted-arg digest tamper must fail even with both purges passing.
    bad = dict(rec, train_feature_digest="tampered")
    rep2 = v3.legacy_level_metadata_report(
        bad,
        actual_n_train_rows=5800,
        actual_train_label_max_utc=actual_max,
        actual_train_feature_digest="featA",
        actual_train_label_digest="labA",
        cutoff=cutoff,
    )
    assert rep2["ok"] is False
    bad2 = dict(rec, train_label_digest="tampered")
    rep3 = v3.legacy_level_metadata_report(
        bad2,
        actual_n_train_rows=5800,
        actual_train_label_max_utc=actual_max,
        actual_train_feature_digest="featA",
        actual_train_label_digest="labA",
        cutoff=cutoff,
    )
    assert rep3["ok"] is False
    # Legacy purge failure must fail even with digests matching.
    late_legacy = dict(rec, train_label_max_utc=cutoff.isoformat())
    rep4 = v3.legacy_level_metadata_report(
        late_legacy,
        actual_n_train_rows=5800,
        actual_train_label_max_utc=actual_max,
        actual_train_feature_digest="featA",
        actual_train_label_digest="labA",
        cutoff=cutoff,
    )
    assert rep4["legacy_purge_ok"] is False and rep4["ok"] is False


def test_actual_future_cutoff_fails():
    cutoff = protocol.ts_utc("2025-01-01T00:00:00Z") - pd.Timedelta(hours=72)
    legacy_max = (cutoff - pd.Timedelta(hours=1)).isoformat()
    future_actual = (cutoff + pd.Timedelta(hours=1)).isoformat()
    rec = {
        "n_train_rows": 100,
        "train_label_max_utc": legacy_max,
        "train_feature_digest": "f",
        "train_label_digest": "l",
    }
    rep = v3.legacy_level_metadata_report(
        rec,
        actual_n_train_rows=100,
        actual_train_label_max_utc=future_actual,
        actual_train_feature_digest="f",
        actual_train_label_digest="l",
        cutoff=cutoff,
    )
    assert rep["actual_purge_ok"] is False
    assert rep["ok"] is False


def test_actual_manifest_derives_from_frozen_helper_not_constants():
    inputs = _synthetic_inputs()
    cutoff = inputs["origins"].min() + pd.Timedelta(hours=72 + 5)
    train, ledger = v3.actual_mean_training_manifest(inputs, cutoff)
    train2, ledger2 = mean_level.build_mean_training(inputs, cutoff)
    assert len(train) == len(train2) == int(ledger["n_eligible"])
    assert ledger["train_label_max_utc"] == ledger2["train_label_max_utc"]
    assert ledger["label_digest"] == ledger2["label_digest"]
    src = Path(v3.__file__).read_text()
    assert "mean_level.build_mean_training" in src
    assert "322681" not in src and "366913" not in src and "408913" not in src


# ----------------------------------------------------------------------------
# 2. model/pred/hash tamper + future mask
# ----------------------------------------------------------------------------
def test_observed_mask_strict_and_future_tamper():
    idx = pd.date_range("2024-10-05", periods=4, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {"y": [1.0, np.nan, 3.0, 4.0], "observed_mask": [True, False, True, True]}, index=idx
    )
    assert v3.check_observed_mask_strict(frame, 3) is True
    tampered = frame.copy()
    tampered.loc[tampered.index[1], "observed_mask"] = True
    assert v3.check_observed_mask_strict(tampered, 3) is False
    assert v3.check_observed_mask_strict(tampered, 4) is False  # future/unobserved marked
    assert v3.check_observed_mask_strict(frame.drop(columns=["observed_mask"]), 3) is False
    assert v3.check_observed_mask_strict(frame, 999) is False


def test_hash_tamper_fails_legacy_report():
    cutoff = protocol.ts_utc("2025-04-01T00:00:00Z") - pd.Timedelta(hours=72)
    mx = (cutoff - pd.Timedelta(hours=2)).isoformat()
    rec = {
        "n_train_rows": 10,
        "train_label_max_utc": mx,
        "train_feature_digest": "good-feat",
        "train_label_digest": "good-lab",
    }
    ok_rep = v3.legacy_level_metadata_report(
        rec,
        actual_n_train_rows=8,
        actual_train_label_max_utc=mx,
        actual_train_feature_digest="good-feat",
        actual_train_label_digest="good-lab",
        cutoff=cutoff,
    )
    assert ok_rep["ok"] is True
    for key in ("train_feature_digest", "train_label_digest"):
        bad = dict(rec, **{key: "tampered"})
        rep = v3.legacy_level_metadata_report(
            bad,
            actual_n_train_rows=8,
            actual_train_label_max_utc=mx,
            actual_train_feature_digest="good-feat",
            actual_train_label_digest="good-lab",
            cutoff=cutoff,
        )
        assert rep["ok"] is False


# ----------------------------------------------------------------------------
# 3. ledger gates: shared 12 global, 6 OOF, 3 router
# ----------------------------------------------------------------------------
def test_six_oof_and_twelve_global_claim_gates():
    good12 = _claims(["reference_replay"] * 3 + ["mean_level"] * 3 + ["oof_base"] * 6)
    assert v3.assert_six_oof_claims_ok(good12) is True
    assert v3.assert_twelve_global_claims_ok(good12) is True
    short_oof = _claims(["reference_replay"] * 3 + ["mean_level"] * 3 + ["oof_base"] * 5)
    with pytest.raises(ValueError):
        v3.assert_six_oof_claims_ok(short_oof)
    with pytest.raises(ValueError):
        v3.assert_twelve_global_claims_ok(short_oof)
    missing_mean = _claims(["reference_replay"] * 3 + ["mean_level"] * 2 + ["oof_base"] * 6)
    with pytest.raises(ValueError):
        v3.assert_twelve_global_claims_ok(missing_mean)


def test_three_router_claims_gate():
    good3 = _claims(["router_eval"] * 3)
    assert v3.assert_three_router_claims_ok(good3) is True
    with pytest.raises(ValueError):
        v3.assert_three_router_claims_ok(_claims(["router_eval"] * 2))
    with pytest.raises(ValueError):
        v3.assert_three_router_claims_ok(_claims(["router_eval"] * 4))


# ----------------------------------------------------------------------------
# 4. native identity helper at 1e-9 (never relaxed)
# ----------------------------------------------------------------------------
def test_native_identity_helper_exact_1e9():
    a = np.array([1.0, 2.0, 3.0])
    assert v3.assert_native_identity(a, a.copy()) is True
    assert v3.assert_native_identity(a, a + 5e-10) is True
    with pytest.raises(ValueError):
        v3.assert_native_identity(a, a + 5e-8)
    with pytest.raises(ValueError):
        v3.assert_native_identity(a, np.array([1.0, 2.0]))
    src = Path(v3.__file__).read_text()
    assert "1e-9" in src
    assert v3.REPLAY_VALUES_TOL == 1e-9 == v2.REPLAY_VALUES_TOL


# ----------------------------------------------------------------------------
# 5. AST: zero fit calls in any phase; alias discipline; provenance binding
# ----------------------------------------------------------------------------
def _fit_calls_by_function(path):
    tree = ast.parse(Path(path).read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "stage_"
        ):
            calls = [
                n
                for n in ast.walk(node)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in ("fit", "partial_fit")
            ]
            out[node.name] = len(calls)
    return out


def test_ast_zero_fit_calls_any_phase():
    counts = _fit_calls_by_function(v3.__file__)
    assert set(counts) == {"stage_oof_verify", "stage_all_verify"}
    assert all(n == 0 for n in counts.values())
    tree = ast.parse(Path(v3.__file__).read_text())
    bad = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in ("fit", "partial_fit")
    ]
    assert bad == []
    assert "stage_oof_fit" not in counts and "stage_router_fit" not in counts


def test_oof_never_creates_alias_all_only_if_unused():
    src = Path(v3.__file__).read_text()
    tree = ast.parse(src)
    oof_fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "stage_oof_verify"
    )
    oof_src = ast.get_source_segment(src, oof_fn)
    assert "V3_ALIAS_FILENAME" not in oof_src and "alias_of" not in oof_src
    assert v3.V3_OOF_FILENAME == "router_oof_verify_v3.json"
    assert v3.V3_VERIFY_FILENAME == "router_verify_v3.json"
    all_fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "stage_all_verify"
    )
    all_src = ast.get_source_segment(src, all_fn)
    assert "if not (run / V3_ALIAS_FILENAME).exists():" in all_src


def test_v3_binds_v2_failed_preclaim_and_sources():
    assert v3.V3_SOURCES == (
        "leaderboard/experiments/mse20_scope_expansion/router_verify_v3.py",
        "leaderboard/tests/test_scope_expansion_router_verify_v3.py",
    )
    assert v3.V3_TEST_FILE.endswith("test_scope_expansion_router_verify_v3.py")
    src = Path(v3.__file__).read_text()
    for token in (
        "v2.ROUTER_REGISTRATION_FILENAME",
        "supersedes_failed_oof_fit",
        "source_hashes",
        "input_inventory",
        "v2_registration_sha256",
        "v2_driver_sha256",
        "v2_test_sha256",
    ):
        assert token in src
    assert v2.FAILED_OOF_EXEC_SESSION == 93021
    assert v2.FAILED_OOF_EXIT_CODE == 2
    assert v2.ROUTER_VERIFY_ALIAS_FILENAME == v3.V3_ALIAS_FILENAME == "router_verify.json"
    inv = v3.v3_tests_inventory("/home/t-lab01/sunrun")
    assert inv["test_file"] == v3.V3_TEST_FILE and inv["n_tests"] >= 10
