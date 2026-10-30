"""Synthetic CPU-only tests for the prospective mean-level driver glue.

No real data, no scientific fits, no registration, no claims, no network, no
uploads. Every case exercises the actual ``mean_driver`` helpers or the frozen
APIs it relies on, with fully synthetic frames and tmp run dirs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PKG_PARENT = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(PKG_PARENT))
REPO_ROOT = Path(__file__).resolve().parents[2]

from mse20_scope_expansion import mean_driver, mean_level, protocol  # noqa: E402
from mse20_scope_expansion.protocol import (  # noqa: E402
    ClaimLedger,
    DuplicateClaimError,
    NotReadyError,
)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _synthetic_inputs(t0="2024-12-01", hours=60 * 24, n_origins=120,
                      origin0="2025-01-10"):
    idx = pd.date_range(t0, periods=hours, freq="h", tz="UTC")
    rng = np.random.default_rng(11)
    vals = 400 + 40 * np.sin(np.arange(len(idx)) / 24 * 2 * np.pi) \
        + rng.normal(0, 4, len(idx))
    ace = pd.Series(vals, index=idx)
    origins = pd.date_range(origin0, periods=n_origins, freq="3h", tz="UTC")
    feats = pd.DataFrame({
        "v_lag0": np.sin(np.arange(len(origins)) / 5.0),
        "v_m24": np.cos(np.arange(len(origins)) / 7.0),
        "ace_speed_kms_last": 400.0,
        "speed_d24": 0.5,
        "ch_x_area": 2.0,
    }, index=origins)
    return {"origins": origins, "features": feats,
            "truth_observed": ace.copy(), "ace_filled": ace}


def _direct_frame(origins, gap_origin=None):
    rows = []
    for i, o in enumerate(origins):
        prof = 10 * np.sin(np.arange(1, 73) / 72 * 2 * np.pi) + i
        for h, p in zip(range(1, 73), prof):
            y = np.nan if (gap_origin is not None and o == gap_origin and h > 36) \
                else 400.0 + h * 0.1
            rows.append((o, h, y, float(p)))
    df = pd.DataFrame(rows, columns=["origin", "h", "y", "y_pred"])
    df["target_t"] = df["origin"] + pd.to_timedelta(df["h"], unit="h")
    df["observed_mask"] = df["y"].notna().to_numpy()
    return df


# ----------------------------------------------------------------------------
# 1. fixed claim plan: exactly three mean_level fits, same ledger
# ----------------------------------------------------------------------------
def test_claim_plan_exactly_three_mean_level():
    assert mean_driver.MEAN_CLAIM_CATEGORY == "mean_level"
    assert mean_driver.MEAN_PERIODS == protocol.PERIOD_ORDER
    assert sorted(mean_driver.MEAN_CLAIM_IDS) == ["audit", "dev", "selection"]
    assert set(mean_driver.MEAN_CLAIM_IDS.values()) == {
        "mean_level_dev", "mean_level_selection", "mean_level_audit"}
    assert protocol.CLAIM_ALLOCATION["mean_level"] == 3
    assert list(mean_driver.STAGES) == [
        "prepare", "poison-check", "register", "fit", "verify"]


def test_ledger_reuses_active_reference_run(tmp_path):
    assert mean_driver.ledger_path(tmp_path) == tmp_path / "claims" / "ledger.json"
    assert mean_driver.ledger_path(tmp_path) == ClaimLedger(str(tmp_path)).path


def test_cutoff_is_period_first_minus_72h():
    for p in protocol.PERIOD_ORDER:
        first = protocol.ts_utc(protocol.PERIODS[p][0])
        assert mean_driver.cutoff_for_period(p) == first - pd.Timedelta(hours=72)
        assert mean_driver.claim_id_for(p) == f"mean_level_{p}"
    with pytest.raises(ValueError):
        mean_driver.cutoff_for_period("nope")
    with pytest.raises(ValueError):
        mean_driver.claim_id_for("nope")


def test_schema_params_pinned_to_frozen_protocol():
    assert mean_driver.MEAN_PARAMS == protocol.PARAMS
    assert mean_driver.SUBMODULE_PROTOCOL_VERSION == protocol.PROTOCOL_VERSION
    assert mean_level.NORMALIZER is None


def test_mean_receipt_names_never_collide_with_frozen(tmp_path):
    frozen = {"prepare.json", "poison_diagnostic.json", "fit.json", "verify.json",
              "registration.json", "tests_proof.json",
              *(f"fit_{p}.json" for p in protocol.PERIOD_ORDER),
              *(f"predictions_existing_all_{p}.parquet"
                for p in protocol.PERIOD_ORDER),
              *(f"xgb_existing_all_{p}.ubj" for p in protocol.PERIOD_ORDER)}
    own = {mean_driver.MEAN_PREPARE_FILENAME, mean_driver.MEAN_POISON_FILENAME,
           mean_driver.MEAN_REGISTRATION_FILENAME, mean_driver.MEAN_FIT_SUMMARY_FILENAME,
           mean_driver.MEAN_VERIFY_FILENAME, mean_driver.MEAN_TESTS_PROOF_FILENAME,
           *(mean_driver.MEAN_FIT_FILENAME_TEMPLATE.format(period=p)
             for p in protocol.PERIOD_ORDER),
           *(mean_driver.MEAN_PRED_FILENAME_TEMPLATE.format(period=p)
             for p in protocol.PERIOD_ORDER)}
    assert not (own & frozen)


# ----------------------------------------------------------------------------
# 2. strict cutoff on synthetic labels (glue + frozen training API)
# ----------------------------------------------------------------------------
def test_mean_training_labels_strictly_before_driver_cutoff():
    inputs = _synthetic_inputs()
    cutoff = inputs["origins"][60]  # synthetic analogue of period_first - 72h
    train, ledger = mean_level.build_mean_training(inputs, cutoff)
    assert protocol.ts_utc(ledger["train_label_max_utc"]) < protocol.ts_utc(cutoff)
    assert len(train) == ledger["n_eligible"]
    assert ledger["n_eligible"] + ledger["n_excluded"] == ledger["n_requested"]


# ----------------------------------------------------------------------------
# 3. failed-claim retry guard (same ledger, no second budget)
# ----------------------------------------------------------------------------
def test_failed_mean_claim_blocks_any_retry(tmp_path):
    ledger = ClaimLedger(str(tmp_path))
    ledger.open_claim("mean_level_dev", "mean_level", tests_passed=True)
    ledger.close_claim("mean_level_dev", "failed", "boom")
    with pytest.raises(DuplicateClaimError):
        mean_driver.assert_no_mean_claims_yet(tmp_path)


def test_existing_mean_receipt_blocks_overwrite(tmp_path):
    (tmp_path / mean_driver.MEAN_FIT_FILENAME_TEMPLATE.format(period="dev")) \
        .write_text("{}")
    with pytest.raises(DuplicateClaimError):
        mean_driver.assert_no_mean_claims_yet(tmp_path)


def test_empty_run_has_no_mean_claims(tmp_path):
    assert mean_driver.assert_no_mean_claims_yet(tmp_path) is None


# ----------------------------------------------------------------------------
# 4. mismatched run/ledger and stale readiness fail closed (real-data-free)
# ----------------------------------------------------------------------------
def test_mismatched_run_without_reference_proof_refused(tmp_path):
    with pytest.raises(NotReadyError):
        mean_driver.require_active_reference_run(tmp_path, REPO_ROOT)


def test_stale_readiness_guard_requires_proof(tmp_path):
    with pytest.raises(NotReadyError):
        protocol.require_tests_passed(str(tmp_path), REPO_ROOT)


def test_cli_prepare_fail_closed_without_registration(tmp_path):
    rc = mean_driver.main(["--stage", "prepare", "--run-dir", str(tmp_path),
                           "--repo-root", str(REPO_ROOT)])
    assert rc == 2
    assert not (tmp_path / mean_driver.MEAN_PREPARE_FILENAME).exists()


# ----------------------------------------------------------------------------
# 5. exact-grid and incomplete-profile guards (frozen APIs the driver uses)
# ----------------------------------------------------------------------------
def test_poison_exact_index_guard_rejects_extra_and_missing_keys():
    clean = [("t", 1), ("t", 2)]
    with pytest.raises((ValueError, SystemExit, NotReadyError)):
        protocol.assert_exact_requested_index(clean, [("t", 1), ("t", 2), ("t", 3)])
    with pytest.raises((ValueError, SystemExit, NotReadyError)):
        protocol.assert_exact_requested_index(clean, [("t", 1)])


def test_incomplete_direct_profile_rejected():
    origins = pd.date_range("2025-02-01", periods=2, freq="3h", tz="UTC")
    level = pd.Series(400.0, index=origins)
    good = _direct_frame(origins)[["origin", "h", "y_pred"]]
    with pytest.raises(ValueError):
        mean_level.decompose_predictions(level, good[good["h"] != 72])
    with pytest.raises(ValueError):
        mean_level.decompose_predictions(
            level, pd.concat([good, good.iloc[:1]]))


# ----------------------------------------------------------------------------
# 6. prediction frame: full requested grid, separate observed mask, no drops
# ----------------------------------------------------------------------------
def test_prediction_frame_keeps_full_grid_and_mask():
    origins = pd.date_range("2025-02-01", periods=3, freq="3h", tz="UTC")
    direct = _direct_frame(origins, gap_origin=origins[1])
    level = pd.Series([380.0, 420.0, 460.0], index=origins, name="level_hat")
    frame = mean_driver.build_mean_prediction_frame(level, direct)
    assert len(frame) == len(direct) == 3 * 72  # no dropped missing origins
    assert list(frame["h"].to_numpy()) == list(direct["h"].to_numpy())
    assert frame["observed_mask"].sum() == direct["observed_mask"].sum() < len(frame)
    assert np.allclose(frame.groupby(frame["origin"])["y_pred"].mean().to_numpy(float),
                       level.to_numpy(float))
    diag = mean_driver.summarize_diagnostics(frame)
    assert diag["n_observed"] == int(frame["observed_mask"].sum())
    assert np.isfinite(diag["mean_mse"]) and np.isfinite(diag["level_bias"])


def test_prediction_frame_rejects_reordered_grid():
    origins = pd.date_range("2025-02-01", periods=2, freq="3h", tz="UTC")
    direct = _direct_frame(origins)
    level = pd.Series([400.0, 410.0], index=origins, name="level_hat")
    with pytest.raises(ValueError):
        mean_driver.build_mean_prediction_frame(
            level, direct.iloc[::-1].reset_index(drop=True))


# ----------------------------------------------------------------------------
# 7. small diagnostics only: horizon buckets, deferred jumps, no event metric
# ----------------------------------------------------------------------------
def test_diagnostics_buckets_and_deferred_jumps():
    origins = pd.date_range("2025-02-01", periods=4, freq="3h", tz="UTC")
    direct = _direct_frame(origins)
    level = pd.Series(405.0, index=origins, name="level_hat")
    frame = mean_driver.build_mean_prediction_frame(level, direct)
    diag = mean_driver.summarize_diagnostics(frame)
    assert set(diag["horizon_mse"]) == {"1-6h", "7-24h", "25-48h", "49-72h"}
    assert all(np.isfinite(v) for v in diag["horizon_mse"].values())
    assert "jump_buckets" in diag and "deferred" in str(diag["jump_buckets"])
    for fabricated in ("hss", "unique_f1", "unique_event_f1", "event_f1"):
        assert fabricated not in diag


def test_diagnostics_empty_observed_fails_closed():
    origins = pd.date_range("2025-02-01", periods=1, freq="3h", tz="UTC")
    direct = _direct_frame(origins)
    direct["y"] = np.nan
    direct["observed_mask"] = False
    level = pd.Series([400.0], index=origins, name="level_hat")
    frame = mean_driver.build_mean_prediction_frame(level, direct)
    with pytest.raises(ValueError):
        mean_driver.summarize_diagnostics(frame)


def test_owned_driver_test_inventory_counts_this_file():
    inv = mean_driver.mean_tests_inventory(REPO_ROOT)
    assert inv["test_file"] == mean_driver.MEAN_TEST_FILE
    assert inv["n_tests"] >= 10
# ----------------------------------------------------------------------------
# 8. MEAN readiness proof: fresh successful execution, never a bare flag
# ----------------------------------------------------------------------------
def _write_mean_log(path, n, failed=False):
    body = ("platform linux -- Python 3.12, pytest-8\n"
            "collected 1 item\n")
    if failed:
        body += "FFAILED\n1 failed, %d passed in 0.42s\n" % (n - 1)
    else:
        body += ".\n%d passed in 0.42s\n" % n
    path.write_text(body)


def _mean_run_with_registration(tmp_path):
    protocol.write_registration(str(tmp_path), str(REPO_ROOT))
    return tmp_path


def test_mean_proof_mark_and_require_roundtrip(tmp_path):
    _mean_run_with_registration(tmp_path)
    inv = mean_driver.mean_tests_inventory(REPO_ROOT)
    log = tmp_path / "mean_tests.log"
    _write_mean_log(log, inv["n_tests"])
    out = mean_driver.mark_mean_tests_passed(
        str(tmp_path), log, exit_code=0,
        command=mean_driver.MEAN_READY_COMMAND, repo_root=str(REPO_ROOT))
    assert out == mean_driver.mean_tests_proof_path(tmp_path)
    proof = mean_driver.require_mean_tests_passed(str(tmp_path), str(REPO_ROOT))
    assert proof["exit_code"] == 0
    assert proof["passed_count"] == inv["n_tests"]
    assert proof["log_sha256"] == protocol.sha256_file(log)
    assert proof["mean_tests_inventory"] == inv


def test_mean_proof_rejects_missing_empty_failed_log(tmp_path):
    _mean_run_with_registration(tmp_path)
    inv = mean_driver.mean_tests_inventory(REPO_ROOT)
    with pytest.raises(NotReadyError):
        mean_driver.mark_mean_tests_passed(str(tmp_path), tmp_path / "nope.log",
                                           repo_root=str(REPO_ROOT))
    empty = tmp_path / "empty.log"
    empty.write_text("")
    with pytest.raises(NotReadyError):
        mean_driver.mark_mean_tests_passed(str(tmp_path), empty,
                                           repo_root=str(REPO_ROOT))
    bad = tmp_path / "bad.log"
    _write_mean_log(bad, inv["n_tests"], failed=True)
    with pytest.raises(NotReadyError):
        mean_driver.mark_mean_tests_passed(str(tmp_path), bad, exit_code=0,
                                           command=mean_driver.MEAN_READY_COMMAND,
                                           repo_root=str(REPO_ROOT))
    good = tmp_path / "good.log"
    _write_mean_log(good, inv["n_tests"])
    with pytest.raises(NotReadyError):  # nonzero exit never marks
        mean_driver.mark_mean_tests_passed(str(tmp_path), good, exit_code=1,
                                           command=mean_driver.MEAN_READY_COMMAND,
                                           repo_root=str(REPO_ROOT))
    with pytest.raises(NotReadyError):  # wrong suite command never marks
        mean_driver.mark_mean_tests_passed(str(tmp_path), good, exit_code=0,
                                           command=protocol.ROOT_READY_COMMAND,
                                           repo_root=str(REPO_ROOT))
    with pytest.raises(NotReadyError):  # filename-suffix spoof never marks
        mean_driver.mark_mean_tests_passed(
            str(tmp_path), good, exit_code=0,
            command="pytest tests/prefix_test_scope_expansion_mean_driver.py -q",
            repo_root=str(REPO_ROOT))
    assert not mean_driver.mean_tests_proof_path(tmp_path).exists()


def test_mean_proof_rejects_count_mismatch_and_stale_log(tmp_path):
    _mean_run_with_registration(tmp_path)
    inv = mean_driver.mean_tests_inventory(REPO_ROOT)
    short = tmp_path / "short.log"
    _write_mean_log(short, inv["n_tests"] - 1)
    with pytest.raises(NotReadyError):
        mean_driver.mark_mean_tests_passed(str(tmp_path), short, exit_code=0,
                                           command=mean_driver.MEAN_READY_COMMAND,
                                           repo_root=str(REPO_ROOT))
    stale = tmp_path / "stale.log"
    _write_mean_log(stale, inv["n_tests"])
    old = (tmp_path / "registration.json").stat().st_mtime - 100
    import os
    os.utime(stale, (old, old))
    with pytest.raises(NotReadyError):
        mean_driver.mark_mean_tests_passed(str(tmp_path), stale, exit_code=0,
                                           command=mean_driver.MEAN_READY_COMMAND,
                                           repo_root=str(REPO_ROOT))


def test_mean_proof_no_overwrite_and_tamper_fails_closed(tmp_path):
    _mean_run_with_registration(tmp_path)
    inv = mean_driver.mean_tests_inventory(REPO_ROOT)
    log = tmp_path / "mean_tests.log"
    _write_mean_log(log, inv["n_tests"])
    mean_driver.mark_mean_tests_passed(str(tmp_path), log, exit_code=0,
                                      command=mean_driver.MEAN_READY_COMMAND,
                                      repo_root=str(REPO_ROOT))
    with pytest.raises(DuplicateClaimError):
        mean_driver.mark_mean_tests_passed(str(tmp_path), log, exit_code=0,
                                          command=mean_driver.MEAN_READY_COMMAND,
                                          repo_root=str(REPO_ROOT))
    log.write_text(log.read_text() + "# tampered\n")  # log drift
    with pytest.raises(NotReadyError):
        mean_driver.require_mean_tests_passed(str(tmp_path), str(REPO_ROOT))


def test_mean_proof_rejects_spoofed_source_binding(tmp_path):
    _mean_run_with_registration(tmp_path)
    inv = mean_driver.mean_tests_inventory(REPO_ROOT)
    log = tmp_path / "mean_tests.log"
    _write_mean_log(log, inv["n_tests"])
    mean_driver.mark_mean_tests_passed(str(tmp_path), log, exit_code=0,
                                      command=mean_driver.MEAN_READY_COMMAND,
                                      repo_root=str(REPO_ROOT))
    p = mean_driver.mean_tests_proof_path(tmp_path)
    forged = json.loads(p.read_text())
    forged["mean_sources"][mean_driver.MEAN_TEST_FILE] = "0" * 64
    p.write_text(json.dumps(forged, indent=1, sort_keys=True))
    with pytest.raises(NotReadyError):
        mean_driver.require_mean_tests_passed(str(tmp_path), str(REPO_ROOT))


def test_mean_proof_required_before_any_claim(tmp_path):
    _mean_run_with_registration(tmp_path)
    with pytest.raises(NotReadyError):
        mean_driver.require_mean_tests_passed(str(tmp_path), str(REPO_ROOT))
    ledger = ClaimLedger(str(tmp_path))  # bare True is not how the driver claims
    with pytest.raises(NotReadyError):
        ledger.open_claim("mean_level_dev", "mean_level", tests_passed=False)


def test_cli_mark_tests_passed_flag(tmp_path):
    import json as _json
    _mean_run_with_registration(tmp_path)
    inv = mean_driver.mean_tests_inventory(REPO_ROOT)
    log = tmp_path / "mean_tests.log"
    _write_mean_log(log, inv["n_tests"])
    rc = mean_driver.main(["--mark-tests-passed", str(log), "--run-dir", str(tmp_path),
                           "--repo-root", str(REPO_ROOT)])
    assert rc == 0
    assert mean_driver.mean_tests_proof_path(tmp_path).is_file()
    proof = _json.loads(mean_driver.mean_tests_proof_path(tmp_path).read_text())
    assert proof["passed_count"] == inv["n_tests"]
    rc = mean_driver.main(["--mark-tests-passed", str(tmp_path / "missing.log"),
                           "--run-dir", str(tmp_path),
                           "--repo-root", str(REPO_ROOT)])
    assert rc == 2


# ----------------------------------------------------------------------------
# 9. prepare/train-ledger binding and poison sensitivity control (synthetic)
# ----------------------------------------------------------------------------
def test_prep_pins_refuse_any_drift():
    prep = {"source_hashes": {"a": "1"}, "mean_sources": {"m": "2"},
            "mean_tests_inventory": {"n_tests": 3},
            "input_inventory": [{"path": "p", "bytes": 9, "sha256": "s",
                                 "mtime_utc": 1.0}],
            "versions": {"python": "3.12"},
            "direct_artifacts": {"dev": {"model_sha256": "x"}}}
    kw = {"source_hashes": {"a": "1"}, "mean_sources": {"m": "2"},
          "mean_tests_inventory": {"n_tests": 3},
          "input_inventory": [{"path": "p", "bytes": 9, "sha256": "s",
                               "mtime_utc": 2.0}],  # mtime is informational
          "versions": {"python": "3.12"},
          "direct_artifacts": {"dev": {"model_sha256": "x"}}}
    assert mean_driver.assert_prep_pins_current(prep, **kw) is None
    for key, bad in [("source_hashes", {"a": "9"}),
                     ("mean_sources", {"m": "9"}),
                     ("mean_tests_inventory", {"n_tests": 99}),
                     ("versions", {"python": "9"}),
                     ("direct_artifacts", {"dev": {"model_sha256": "y"}})]:
        drifted = dict(kw)
        drifted[key] = bad
        with pytest.raises(NotReadyError):
            mean_driver.assert_prep_pins_current(prep, **drifted)
    drifted = dict(kw)
    drifted["input_inventory"] = [{"path": "p", "bytes": 9, "sha256": "other",
                                   "mtime_utc": 2.0}]
    with pytest.raises(NotReadyError):
        mean_driver.assert_prep_pins_current(prep, **drifted)


def test_train_ledger_must_match_prepare_bytes():
    a = {"requested_digest": "r", "eligible_digest": "e", "label_digest": "l",
         "n_requested": 10, "n_eligible": 7, "n_excluded": 3}
    assert mean_driver.assert_train_ledger_matches_prep(dict(a), dict(a)) is None
    for key, bad in [("label_digest", "x"), ("eligible_digest", "x"),
                     ("n_eligible", 6), ("n_excluded", 4)]:
        b = dict(a)
        b[key] = bad
        with pytest.raises(ValueError):
            mean_driver.assert_train_ledger_matches_prep(b, a)


def test_positive_control_flags_insensitive_comparison():
    idx = pd.date_range("2025-01-01", periods=4, freq="3h", tz="UTC")
    clean = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    moved = pd.DataFrame({"f": [1.0, 2.5, 3.0, 4.0]}, index=idx)
    ctl = mean_driver.control_sensitivity(clean, moved)
    assert ctl["sensitive"] and ctl["changed_features"] == ["f"]
    same = mean_driver.control_sensitivity(clean, clean.copy())
    assert not same["sensitive"] and same["n_changed_features"] == 0
