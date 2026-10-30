"""Synthetic CPU-only tests for the scope-expansion reference harness.

No real data, no model fits, no network, no uploads. Every case exercises the
actual harness APIs (no source-substring assertions).
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PKG_PARENT = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(PKG_PARENT))
REPO_ROOT = Path(__file__).resolve().parents[2]

from mse20_scope_expansion import evaluate, protocol, reference  # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _synthetic_inputs(t0="2025-07-20", days=20, gap_hours=()):
    idx = pd.date_range(t0, periods=days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(7)
    vals = 400 + 30 * np.sin(np.arange(len(idx)) / 24 * 2 * np.pi) + rng.normal(0, 5, len(idx))
    truth = pd.Series(vals, index=idx)
    for g in gap_hours:
        truth.iloc[g] = np.nan
    origins = pd.date_range("2025-07-25", periods=8 * 12, freq="3h", tz="UTC")
    feats = pd.DataFrame({"v_lag0": np.arange(len(origins), dtype=float)}, index=origins)
    return origins, truth, truth.copy(), feats


# ----------------------------------------------------------------------------
# 1. fixed config / schema / train purge
# ----------------------------------------------------------------------------
def test_fixed_reference_config_and_schema():
    assert protocol.PARAMS == {
        "n_estimators": 600, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 20,
        "reg_lambda": 5, "tree_method": "hist", "n_jobs": 8, "random_state": 0}
    assert reference.PARAMS == protocol.PARAMS
    assert protocol.HORIZONS == tuple(range(1, 73))
    assert protocol.GRID_FREQ == "3h" and protocol.PURGE_HOURS == 72
    assert protocol.TARGET_MAX_UTC == "2026-05-31T23:00:00Z"
    assert reference.SCALERS is None
    cols = (["h", "v_lag0", "rec26", "rec_mean", "ace_speed_kms_last", "speed_d24",
             "epam_p1_d6", "ch_x_area", "enlil_next_arr_h", "donki_q", "official_z"])
    sel = reference.existing_all_columns(cols)
    assert sel == ["h", "v_lag0", "rec26", "rec_mean", "ace_speed_kms_last",
                   "speed_d24", "epam_p1_d6", "ch_x_area"]


def test_train_purge_boundary_and_requested_vs_observed():
    origins, truth, ace_filled, feats = _synthetic_inputs()
    df = reference.assemble_long_table(origins, feats, truth, ace_filled)
    assert len(df) == len(origins) * 72  # full requested grid, labels attached
    a = protocol.ts_utc(protocol.PERIODS["dev"][0])
    cutoff = a - pd.Timedelta(hours=72)
    o = pd.date_range("2025-07-30", periods=40, freq="3h", tz="UTC")
    big_truth = pd.Series(400.0, index=pd.date_range("2025-06-01", periods=90 * 24, freq="h", tz="UTC"))
    big_feats = pd.DataFrame({"v": np.zeros(len(o))}, index=o)
    big = reference.assemble_long_table(o, big_feats, big_truth, big_truth)
    tr = big[big["target_t"] < cutoff]
    assert (tr["target_t"] < cutoff).all() and len(tr) > 0
    eq = big[big["target_t"] == cutoff]
    assert len(eq) > 0  # boundary case present in the requested grid
    assert not eq.index.isin(tr.index).any()  # strict purge excludes it
    # unobserved labels: requested keys unchanged, observed mask drops NaN
    origins2, truth2, ace2, feats2 = _synthetic_inputs(gap_hours=[200, 201, 400])
    df2 = reference.assemble_long_table(origins2, feats2, truth2, ace2)
    req = reference.requested_keys(df2)
    obs = df2.dropna(subset=["y"])
    assert len(req) == len(origins2) * 72
    assert len(obs) < len(req) and len(reference.requested_keys(df2)) == len(req)


def test_target_boundary_and_rec_causality_enforced():
    origins, truth, ace_filled, feats = _synthetic_inputs()
    df = reference.assemble_long_table(origins, feats, truth, ace_filled)
    tmax = protocol.ts_utc(protocol.TARGET_MAX_UTC)
    assert (df["target_t"] <= tmax).all()
    for d in (26, 27, 28):
        look = df["target_t"] - pd.Timedelta(days=d)
        assert (look < df["origin"]).all()
    late = origins + pd.Timedelta(days=400)  # targets past May-31 -> refused
    with pytest.raises(ValueError):
        reference.assemble_long_table(late, pd.DataFrame({"v": [0.0] * len(late)}, index=late),
                                      truth, ace_filled)


# ----------------------------------------------------------------------------
# 2. exact poison len/index gate
# ----------------------------------------------------------------------------
def test_exact_index_gate_rejects_extra_missing_duplicate():
    keys = [("2025-01-01T00:00:00+00:00", 1), ("2025-01-01T00:00:00+00:00", 2)]
    assert protocol.assert_exact_requested_index(keys, list(reversed(keys))) == sorted(keys)
    with pytest.raises(ValueError):
        protocol.assert_exact_requested_index(keys, keys + [("2025-01-01T00:00:00+00:00", 3)])
    with pytest.raises(ValueError):
        protocol.assert_exact_requested_index(keys, keys[:1])
    with pytest.raises(ValueError):
        protocol.assert_exact_requested_index(keys, keys + keys[:1])


def test_future_input_poison_invariance_with_positive_control():
    idx = pd.date_range("2025-01-01", periods=30 * 24, freq="h", tz="UTC")
    s = pd.Series(np.sin(np.arange(len(idx)) / 10.0), index=idx)
    origins = pd.date_range("2025-01-10", periods=100, freq="3h", tz="UTC")
    cutoff = pd.Timestamp("2025-01-20", tz="UTC")
    dirty_s = s.copy()
    dirty_s.loc[dirty_s.index > cutoff] = 999.0
    clean = reference.speed_features(origins, s)
    dirty = reference.speed_features(origins, dirty_s)
    keep = origins <= cutoff
    protocol.assert_exact_requested_index(list(clean.index[keep]), list(dirty.index[keep]))
    for col in clean.columns:  # NaN masks included
        a, b = clean.loc[keep, col].to_numpy(float), dirty.loc[keep, col].to_numpy(float)
        assert np.allclose(a, b, equal_nan=True), col
        assert (np.isnan(a) == np.isnan(b)).all(), col
    # positive control: a leaky builder IS caught on the same keys
    leaky = pd.DataFrame({"peek": s.shift(-1).reindex(origins),
                          "peek_dirty": dirty_s.shift(-1).reindex(origins)})
    assert not np.allclose(leaky["peek"].to_numpy(float),
                           leaky["peek_dirty"].to_numpy(float), equal_nan=True)


def test_no_donki_no_official_inputs():
    for p in protocol.SOURCE_CLOSURE:
        for bad in ("donki", "enlil", "official"):
            assert bad not in p.lower()
    sel = reference.existing_all_columns(["h", "v_lag0", "enlil_next_arr_h"])
    assert "enlil_next_arr_h" not in sel  # never selected, even if present
    origins, truth, ace, _ = _synthetic_inputs()
    bad_feats = pd.DataFrame({"official_z": np.zeros(len(origins))}, index=origins)
    with pytest.raises(ValueError):
        reference.assemble_long_table(origins, bad_feats, truth, ace)


# ----------------------------------------------------------------------------
# 3. claims: exclusive, failed consumes, global 36 cap
# ----------------------------------------------------------------------------
def test_claim_exclusivity_failure_consumes_and_global_cap(tmp_path):
    ledger = protocol.ClaimLedger(tmp_path)
    assert sum(protocol.CLAIM_ALLOCATION.values()) == protocol.CLAIM_BUDGET_TOTAL == 36
    with pytest.raises(protocol.NotReadyError):
        ledger.open_claim("x", "reference_replay", tests_passed=False)
    n = 0
    for cat, k in protocol.CLAIM_ALLOCATION.items():
        for i in range(k):
            ledger.open_claim(f"{cat}_{i}", cat, tests_passed=True)
            n += 1
    assert n == 36 and ledger.counts()["total"] == 36
    with pytest.raises(protocol.CapExceededError):
        ledger.open_claim("one_too_many", "final", tests_passed=True)
    with pytest.raises(protocol.DuplicateClaimError):
        ledger.open_claim("reference_replay_0", "reference_replay", tests_passed=True)
    ledger.close_claim("reference_replay_0", "failed", "interrupted")
    with pytest.raises(protocol.DuplicateClaimError):  # no retry even after failure
        ledger.open_claim("reference_replay_0", "reference_replay", tests_passed=True)
    assert ledger.counts()["total"] == 36  # failed slot still consumed


# ----------------------------------------------------------------------------
# 4. admission gate boundaries
# ----------------------------------------------------------------------------
def test_admission_two_periods_and_arithmetic_mean():
    assert protocol.admission_decision(
        {"dev": 100.0, "selection": 100.0, "audit": 100.0},
        {"dev": 96.0, "selection": 98.0, "audit": 100.0})["admitted"] is True
    r = protocol.admission_decision(
        {"dev": 100.0, "selection": 100.0, "audit": 100.0},
        {"dev": 94.0, "selection": 101.0, "audit": 101.0})
    assert r["admitted"] is False and r["reason"] == "fewer_than_two_periods"
    r = protocol.admission_decision(
        {"dev": 100.0, "selection": 100.0, "audit": 100.0},
        {"dev": 99.0, "selection": 99.0, "audit": 100.0})
    assert r["admitted"] is False and r["reason"] == "mean_above_minus_2pct"
    r = protocol.admission_decision(  # exact mean -0.02 passes
        {"dev": 100.0, "selection": 100.0, "audit": 100.0},
        {"dev": 97.0, "selection": 97.0, "audit": 100.0})
    assert r["admitted"] is True
    r = protocol.admission_decision(  # exact zero is not improvement
        {"dev": 100.0, "selection": 100.0, "audit": 100.0},
        {"dev": 100.0, "selection": 100.0, "audit": 100.0})
    assert r["admitted"] is False
    for bad_ref, bad_aug in [({"dev": 0.0, "selection": 100.0, "audit": 100.0},
                              {"dev": 90.0, "selection": 90.0, "audit": 90.0}),
                             ({"dev": float("nan"), "selection": 100.0, "audit": 100.0},
                              {"dev": 90.0, "selection": 90.0, "audit": 90.0})]:
        r = protocol.admission_decision(bad_ref, bad_aug)
        assert r["admitted"] is False and r["reason"] == "indeterminate_reference"
    r = protocol.admission_decision({"dev": 100.0}, {"dev": 90.0})
    assert r["admitted"] is False and r["reason"] == "key_mismatch"
    good_ref = {"dev": 100.0, "selection": 100.0, "audit": 100.0}
    good_aug = {"dev": 96.0, "selection": 98.0, "audit": 100.0}
    full = {p: 1.0 for p in protocol.PERIOD_ORDER}
    many = {p: 90 for p in protocol.PERIOD_ORDER}
    assert protocol.admission_decision(good_ref, good_aug, coverage=full,
                                       n_origins=many)["admitted"] is True
    r = protocol.admission_decision(good_ref, good_aug,
                                    coverage={**full, "audit": 0.79}, n_origins=many)
    assert r["admitted"] is False and r["reason"] == "coverage_below_80pct"
    r = protocol.admission_decision(good_ref, good_aug, coverage=full,
                                    n_origins={**many, "dev": 59})
    assert r["admitted"] is False and r["reason"] == "fewer_than_60_origins"


def test_publication_proof_gate():
    o = "2025-09-15T12:00:00Z"
    assert protocol.is_product_admissible("2025-09-15T12:00:00Z", o) is True  # equality allowed
    assert protocol.is_product_admissible("2025-09-14T00:00:00Z", o) is True
    # initialized before origin but published after: excluded
    assert protocol.is_product_admissible("2025-09-16T00:00:00Z", o) is False
    with pytest.raises(ValueError):
        protocol.is_product_admissible("2025-09-14", o)  # naive timestamps rejected


# ----------------------------------------------------------------------------
# 5. evaluator fail-closed
# ----------------------------------------------------------------------------
def test_final_defaults_fail_closed_without_proof(tmp_path):
    assert evaluate.DEFAULT_PHASE == "final"
    assert evaluate.evaluate_phase(tmp_path, "final") == 2
    assert evaluate.evaluate_phase(tmp_path, "reference") == 2
    code, report = evaluate.evaluate_final(tmp_path)
    assert code == 2 and report["gaps"] == list(evaluate.FINAL_GATES)
    assert evaluate.main(["--run-dir", str(tmp_path)]) == 2  # default phase is final


def test_reference_phase_requires_full_proof(tmp_path):
    code, _ = evaluate.evaluate_reference(tmp_path)
    assert code == 2
    (tmp_path / "registration.json").write_text(json.dumps({"code_tests_status": "passed"}))
    assert evaluate.evaluate_reference(tmp_path)[0] == 2  # fits still missing


# ----------------------------------------------------------------------------
# 6. registration hashes + caller interop
# ----------------------------------------------------------------------------
def test_registration_hashes_and_caller_interop(tmp_path):
    reg = protocol.build_registration(REPO_ROOT)
    assert reg["amendment_pin"] == protocol.AMENDMENT_SHA256
    assert reg["user_start_utc"] == "2026-10-03T06:26:29.494380Z"
    assert reg["immutable_existing_all_config"] == protocol.immutable_config_digest()
    assert reg["immutable_existing_all_config"] == protocol.build_registration(
        REPO_ROOT)["immutable_existing_all_config"]
    assert reg["user_deadline_utc"] is None
    assert reg["closeout_sha256"] == protocol.CLOSEOUT_SHA256
    for rel, expect in protocol.EXPECTED_DOC_SHAS.items():
        assert reg["plan_documents"][rel] == expect
    assert set(reg["source_hashes"]) == set(protocol.SOURCE_FILES)
    assert sum(reg["claim_allocation"].values()) == 36
    run = tmp_path / "run"
    assert protocol.write_registration(run, REPO_ROOT).is_file()
    with pytest.raises(protocol.DuplicateClaimError):
        protocol.write_registration(run, REPO_ROOT)
    # Forged weak log fails closed (old "12 passed" text is not sufficient).
    bad = tmp_path / "bad.log"
    bad.write_text("12 passed")
    with pytest.raises(protocol.NotReadyError):
        protocol.mark_tests_passed(run, bad)
    inv = protocol.tests_inventory(REPO_ROOT)
    log = tmp_path / "pytest.log"
    log.write_text(f"{inv['n_tests']} passed in 0.34s")
    protocol.mark_tests_passed(run, log)
    with pytest.raises(protocol.DuplicateClaimError):
        protocol.mark_tests_passed(run, log)
    assert protocol.require_tests_passed(run)["user_start_utc"] == "2026-10-03T06:26:29.494380Z"
    # Registration is never rewritten by readiness; tampering is detected.
    stored = json.loads((run / "registration.json").read_text())
    stored["user_start_utc"] = "2026-10-04T15:21:00Z"
    (run / "registration.json").write_text(json.dumps(stored))
    with pytest.raises(protocol.NotReadyError):
        protocol.validate_registration(str(run), REPO_ROOT)
    assert reference.SUBMODULE_PROTOCOL_VERSION == protocol.PROTOCOL_VERSION


def test_pilot_registration_helper_and_schemas(tmp_path):
    grid = protocol.pilot_origin_grid("dev")
    assert ((grid.hour == 12) & (grid.minute == 0)).all()
    a, b = (protocol.ts_utc(t) for t in protocol.PERIODS["dev"])
    assert (grid >= a.floor("D")).all() and (grid <= b).all()
    assert len(grid) >= protocol.PILOT_ORIGINS_MIN
    assert set(protocol.SOURCE_AUGMENTATION_SCHEMA) == {
        "src_pred_speed_at_target", "src_origin_speed", "src_pred_mean_72h",
        "src_age_h", "src_missing_flag"}
    assert (protocol.AVAIL_BYTES_MAX, protocol.PILOT_BYTES_MAX,
            protocol.STALENESS_MAX_HOURS) == (64 * 1024**2, 512 * 1024**2, 24)
    assert protocol.build_registration(REPO_ROOT)["readiness"]["noaa_wsa_enlil"] == "UNVERIFIED"
    p = protocol.register_pilot(tmp_path, "NOAA-WSA-ENLIL", REPO_ROOT)
    assert p.is_file()
    with pytest.raises(protocol.DuplicateClaimError):
        protocol.register_pilot(tmp_path, "NOAA-WSA-ENLIL", REPO_ROOT)


# ----------------------------------------------------------------------------
# 7. vendored builders match the read-only originals (behavioral parity)
# ----------------------------------------------------------------------------
def _load_originals(monkeypatch):
    stub = types.ModuleType("xgboost")
    stub.XGBRegressor = type("XGBRegressor", (), {})
    stub.Booster = type("Booster", (), {})
    stub.DMatrix = type("DMatrix", (), {})
    monkeypatch.setitem(sys.modules, "xgboost", stub)
    hd = Path("/home/t-lab01/sunrun/leaderboard/experiments/headroom_audit")
    monkeypatch.syspath_prepend(str(hd))
    import headroom
    import info_ablation
    import importlib
    return importlib.reload(headroom), importlib.reload(info_ablation)


def test_vendored_speed_and_arms_match_originals(monkeypatch):
    headroom, ia = _load_originals(monkeypatch)
    idx = pd.date_range("2025-02-01", periods=600, freq="h", tz="UTC")
    s = pd.Series(np.sin(np.arange(len(idx)) / 8.0), index=idx)
    origins = pd.date_range("2025-02-10", periods=50, freq="3h", tz="UTC")
    mine = reference.speed_features(origins, s)
    theirs = ia.speed_features(origins, s)
    pd.testing.assert_frame_equal(mine, theirs)
    cols = (["h", "v_lag3", "v_m6", "rec27", "rec_mean", "ace_bt_nt_m6", "speed_d24",
             "epam_p1_d6", "ch_foo_area", "ch_lon00", "enlil_next_arr_h", "official_x"])
    assert ia.arms(cols)["existing_all"] == reference.existing_all_columns(cols)


def test_vendored_ace_ch_match_originals_on_synthetic_io(monkeypatch):
    headroom, _ = _load_originals(monkeypatch)
    hours = pd.date_range("2025-03-01", periods=400, freq="h", tz="UTC")
    raw = pd.DataFrame({
        "timestamp_utc": hours,
        "ace_speed_kms": 400.0, "ace_density_cm3": 5.0, "ace_temperature_k": 1e5,
        "ace_bt_nt": 5.0, "ace_bz_gsm_nt": -1.0, "ace_by_gsm_nt": 2.0,
        "ace_epam_e_channel1": 10.0, "ace_epam_e_channel2": 10.0,
        "ace_epam_p_channel1": 10.0, "ace_epam_p_channel3": 10.0,
        "ace_epam_p_channel5": 10.0, "ace_sis_gt10_flux": 1.0})
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: raw)
    import glob as _glob
    monkeypatch.setattr(_glob, "glob", lambda *a, **k: ["fake.parquet"])
    origins = pd.date_range("2025-03-10", periods=30, freq="3h", tz="UTC")
    pd.testing.assert_frame_equal(reference.ace_features(origins, raw_frames=[raw]),
                                  headroom.ace_features(origins))
    ch = pd.DataFrame({"timestamp_utc": hours, "ch_t035_area": 3.0,
                       "ch_t035_meridian30_equatorial_area": 1.0,
                       "ch_t035_lon00_lat0_area": 9.0})
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: ch)
    pd.testing.assert_frame_equal(reference.ch_features(origins, ch_frame=ch),
                                  headroom.ch_features(origins))


# ----------------------------------------------------------------------------
# 8. guard repairs: forged/stale/skip/concurrency/forged-final/grid/strict-mean
# ----------------------------------------------------------------------------
def _fresh_run_with_proof(tmp_path):
    run = tmp_path / "run"
    protocol.write_registration(run, REPO_ROOT)
    inv = protocol.tests_inventory(REPO_ROOT)
    log = tmp_path / "pytest.log"
    log.write_text(f"{inv['n_tests']} passed in 0.34s")
    protocol.mark_tests_passed(run, log)
    return run, log


def test_user_start_preserved_and_deadline_null():
    assert protocol.USER_START_UTC == "2026-10-03T06:26:29.494380Z"
    assert protocol.USER_DEADLINE_UTC is None
    reg = protocol.build_registration(REPO_ROOT)
    assert reg["user_start_utc"] == "2026-10-03T06:26:29.494380Z"
    assert reg["user_deadline_utc"] is None


def test_closeout_pins_match_review_closeout():
    closeout = json.loads((REPO_ROOT / protocol.CLOSEOUT_PATH).read_text())
    assert closeout["plan"]["sha256"] == protocol.EXPECTED_DOC_SHAS[
        ".omx/plans/addendum-mse20-scope-expansion.md"]
    assert closeout["test_spec"]["sha256"] == protocol.EXPECTED_DOC_SHAS[
        ".omx/plans/test-spec-mse20-scope-expansion.md"]
    assert closeout["architect"]["sha256"] == protocol.EXPECTED_DOC_SHAS[
        ".omx/context/scope-expansion-architect-r2-20261005.md"]
    assert closeout["critic"]["sha256"] == protocol.EXPECTED_DOC_SHAS[
        ".omx/context/scope-expansion-critic-r2-20261005.md"]
    reg = protocol.build_registration(REPO_ROOT)
    for rel, expect in protocol.EXPECTED_DOC_SHAS.items():
        assert reg["plan_documents"][rel] == expect
    tampered = dict(reg["plan_documents"])
    tampered[".omx/plans/addendum-mse20-scope-expansion.md"] = "0" * 64
    reg["plan_documents"] = tampered
    assert reg["plan_documents"] != protocol.build_registration(REPO_ROOT)["plan_documents"]


def test_forged_logs_rejected(tmp_path):
    run = tmp_path / "run"
    protocol.write_registration(run, REPO_ROOT)
    inv = protocol.tests_inventory(REPO_ROOT)
    for text, exit_code, command in [
        ("", 0, protocol.ROOT_READY_COMMAND),
        ("1 failed, 3 passed", 0, protocol.ROOT_READY_COMMAND),
        (f"{inv['n_tests']} passed", 1, protocol.ROOT_READY_COMMAND),
        (f"{inv['n_tests']} passed", 0, "make test"),
    ]:
        bad = tmp_path / f"bad_{abs(hash(text)) % 10000}.log"
        bad.write_text(text or "empty")
        with pytest.raises(protocol.NotReadyError):
            protocol.mark_tests_passed(run, bad, exit_code=exit_code, command=command)
    assert not protocol.tests_proof_path(run).exists()


def test_stale_readiness_rejected(tmp_path):
    import os
    import time
    run = tmp_path / "run"
    protocol.write_registration(run, REPO_ROOT)
    inv = protocol.tests_inventory(REPO_ROOT)
    log = tmp_path / "pytest.log"
    log.write_text(f"{inv['n_tests']} passed in 0.34s")
    old = (run / "registration.json").stat().st_mtime - 100
    os.utime(log, (old, old))
    with pytest.raises(protocol.NotReadyError):
        protocol.mark_tests_passed(run, log)
    time.sleep(0.01)


def test_changed_source_and_test_rejected(tmp_path):
    run, log = _fresh_run_with_proof(tmp_path)
    proof = json.loads((run / "tests_proof.json").read_text())
    proof["source_hashes"]["leaderboard/tests/test_scope_expansion_reference.py"] = "0" * 64
    (run / "tests_proof.json").write_text(json.dumps(proof))
    with pytest.raises(protocol.NotReadyError):
        protocol.require_tests_passed(run)
    assert protocol.tests_inventory(REPO_ROOT)["sha256"] == protocol.sha256_file(
        REPO_ROOT / "leaderboard/tests/test_scope_expansion_reference.py")


def test_poison_gate_blocks_fit_before_claim(tmp_path):
    run = tmp_path / "run"
    protocol.write_registration(run, REPO_ROOT)
    inv = protocol.tests_inventory(REPO_ROOT)
    log = tmp_path / "pytest.log"
    log.write_text(f"{inv['n_tests']} passed in 0.34s")
    protocol.mark_tests_passed(run, log)
    with pytest.raises(protocol.NotReadyError):
        protocol.require_poison_ready(str(run))
    ledger = protocol.ClaimLedger(str(run))
    assert ledger.counts()["total"] == 0  # no slot consumed on missing readiness
    (run / "poison_diagnostic.json").write_text(json.dumps({"cutoffs": {}}))
    with pytest.raises(protocol.NotReadyError):
        protocol.require_poison_ready(str(run))
    assert ledger.counts()["total"] == 0


def test_ledger_37th_and_concurrent(tmp_path):
    import threading
    run = tmp_path / "run"
    ledger = protocol.ClaimLedger(str(run))
    for cat, k in protocol.CLAIM_ALLOCATION.items():
        for i in range(k):
            ledger.open_claim(f"{cat}_{i}", cat, tests_passed=True)
    assert ledger.counts()["total"] == 36
    with pytest.raises(protocol.CapExceededError):
        ledger.open_claim("one_too_many", "final", tests_passed=True)
    errors = []

    def _race(i):
        try:
            ledger.open_claim(f"race_{i}", "final", tests_passed=True)
        except (protocol.CapExceededError, protocol.DuplicateClaimError):
            errors.append(i)

    threads = [threading.Thread(target=_race, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(errors) == 8 and ledger.counts()["total"] == 36


def test_final_forged_allpass_still_invalid(tmp_path):
    forged = {g: {"pass": True} for g in evaluate.FINAL_GATES}
    (tmp_path / "final_proof.json").write_text(json.dumps(forged))
    code, report = evaluate.evaluate_final(tmp_path)
    assert code == 2
    assert report["gaps"] == list(evaluate.FINAL_GATES)
    assert report.get("forged_proof_ignored") is True


def test_full_grid_missing_key_rejected():
    origins, truth, ace, feats = _synthetic_inputs(gap_hours=[200, 201, 400])
    df = reference.assemble_long_table(origins, feats, truth, ace)
    full = reference.requested_keys(df)
    assert len(full) == len(origins) * 72
    with pytest.raises(ValueError):
        protocol.assert_exact_requested_index(full, full[1:])
    with pytest.raises(ValueError):
        protocol.assert_exact_requested_index(full, full + [("2099-01-01T00:00:00+00:00", 1)])
    obs = df.dropna(subset=["y"])
    assert len(reference.requested_keys(obs)) < len(full)


def test_strict_mean_threshold_no_loosening():
    base = {"dev": 100.0, "selection": 100.0, "audit": 100.0}
    exact = {"dev": 97.0, "selection": 97.0, "audit": 100.0}  # mean exactly -0.02
    assert protocol.admission_decision(base, exact)["admitted"] is True
    above = {"dev": 97.00001, "selection": 97.0, "audit": 100.0}  # just above -0.02
    r = protocol.admission_decision(base, above)
    assert r["admitted"] is False and r["reason"] == "mean_above_minus_2pct"
    r = protocol.admission_decision(base, {"dev": 96.0, "selection": 98.0, "audit": 100.0},
                                    coverage={p: 1.0 for p in protocol.PERIOD_ORDER},
                                    n_origins={p: 90 for p in protocol.PERIOD_ORDER})
    assert r["admitted"] is True


def test_input_inventory_pins_sha256():
    inv = reference.input_inventory()
    assert len(inv) >= 3
    for e in inv:
        assert "sha256" in e and len(e["sha256"]) == 64
        assert e["bytes"] > 0
        assert e["sha256"] == protocol.sha256_file(Path(e["path"]))


def test_reference_evaluator_rejects_forged_verify(tmp_path):
    run, log = _fresh_run_with_proof(tmp_path)
    (run / "prepare.json").write_text(json.dumps({
        "counts_match_original": True, "n_rows": protocol.EXPECTED_N_ROWS,
        "periods": {p: {"n_train": protocol.EXPECTED_N_TRAIN[p],
                        "n_test": protocol.EXPECTED_N_TEST[p]} for p in protocol.PERIOD_ORDER}}))
    (run / "poison_diagnostic.json").write_text(json.dumps({
        "cutoffs": {protocol.PERIODS[p][0]: {"exact_index_validated": True,
                                             "changed_features": []} for p in protocol.PERIOD_ORDER}}))
    for p in protocol.PERIOD_ORDER:
        (run / f"fit_{p}.json").write_text(json.dumps({"bogus": True}))
    (run / "verify.json").write_text(json.dumps({"ok": True, "periods": {}}))
    assert evaluate.evaluate_reference(str(run))[0] == 2
