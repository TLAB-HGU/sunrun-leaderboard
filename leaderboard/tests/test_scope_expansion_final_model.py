"""Synthetic-only tests for the final prospective producer (final_model.py).

No real fits, claims, registration, scoring, or official-truth reads. Small
stubs only; every stage-level check here exercises refusal paths or pure
helpers (receipt pre-exists -> DuplicateClaimError before any I/O).
"""

import json

import numpy as np
import pandas as pd
import pytest

from experiments.mse20_scope_expansion import final_contract, final_model, mean_level, protocol, router_v2

FIRST = "2026-05-31T23:00:00Z"
CUTOFF = "2026-05-28T23:00:00Z"
POISON_START = "2026-05-28T12:00:00Z"  # 8x3h origins straddle the 28T23Z cutoff


def test_direct_grid_matches_frozen_reference_features_without_labels():
    from experiments.mse20_scope_expansion import reference

    origins = pd.date_range("2025-05-01", periods=3, freq="h", tz="UTC")
    hours = pd.date_range("2025-04-01", "2025-05-05", freq="h", tz="UTC")
    speed = pd.Series(np.arange(len(hours), dtype=float) + 300, index=hours)
    feats = pd.DataFrame({"v_lag0": 400., "ace_speed_kms_last": 401.,
                          "ch_x_area": 2.}, index=origins)
    old = reference.assemble_long_table(origins, feats, speed, speed)
    cols = reference.existing_all_columns(old.columns)
    grid = final_model.official_direct_feature_grid(origins, feats, speed)
    pd.testing.assert_frame_equal(grid, old[cols])
    assert len(grid) == 3 * 72 and "y" not in grid and "target_t" not in grid


def test_direct_grid_official_horizons_never_need_future_truth():
    origins = pd.date_range(protocol.ts_utc(FIRST), periods=3, freq="h")
    hours = pd.date_range(protocol.ts_utc("2026-05-01"), origins.max(), freq="h")
    speed = pd.Series(400., index=hours)
    feats = pd.DataFrame({"v_lag0": 400., "ace_speed_kms_last": 401.}, index=origins)
    grid = final_model.official_direct_feature_grid(origins, feats, speed)
    assert len(grid) == 216 and np.isfinite(grid.to_numpy()).all()
    assert grid["h"].tolist() == list(range(1, 73)) * 3
    with pytest.raises(ValueError):
        final_model.official_direct_feature_grid(origins, feats.iloc[:-1], speed)


def test_router_training_medians_ignore_non_oof_rows(monkeypatch, tmp_path):
    times = pd.DatetimeIndex(pd.to_datetime([
        "2024-10-01", "2025-01-01", "2025-04-01", "2026-05-31"], utc=True))
    features = pd.DataFrame({"v_lag0": [2., 4., 6., 1e9]}, index=times)
    lookup = dict(zip(router_v2.OOF_ORDER, times[:3]))
    monkeypatch.setattr(final_model, "_final_oof_base_train_origins",
                        lambda man: {q: [] for q in router_v2.OOF_ORDER})
    monkeypatch.setattr(final_model, "_load_oof_pair", lambda run, q: (q, q))
    monkeypatch.setattr(router_v2, "filled_truth_and_observed_mask", lambda inp: (None, None))
    monkeypatch.setattr(router_v2, "build_router_labels",
                        lambda direct, *a, **kw: (pd.DataFrame({"origin": [lookup[direct]],
                                                               "choice": [0]}), {}))
    monkeypatch.setattr(mean_level, "issue_features",
                        lambda inp, origins: inp["features"].reindex(origins))
    X, y = final_model.final_router_training(tmp_path, {"inputs": {"features": features}})
    assert len(X) == len(y) == 3 and times[-1] not in X.index
    assert router_v2.fit_medians(X, list(X.columns))["v_lag0"] == 4.


def test_poison_future_nan_changes_reported_without_hiding_prefix_changes():
    origins = pd.date_range(protocol.ts_utc(FIRST), periods=2, freq="h")
    clean = pd.DataFrame({"v_lag0": [400., 410.]}, index=origins)
    dirty = clean.copy()
    dirty.iloc[1, 0] = np.nan
    report = final_model.poison_compare(clean, dirty, origins, origins[0])
    assert report["pre_cutoff_changed"] == []
    assert report["changed_features"] == ["v_lag0"]
    assert report["post_cutoff_nan_changes"] == 1
    dirty.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN-mask"):
        final_model.poison_compare(clean, dirty, origins, origins[0])


def _origins(n=8, start="2026-05-20T00:00:00Z", freq="3h"):
    return pd.date_range(protocol.ts_utc(start), periods=n, freq=freq, tz="UTC")


def _feats(origins, shift=0.0):
    return pd.DataFrame({"v_lag0": np.arange(len(origins), dtype=float) + shift,
                         "ace_speed_kms_last": 400.0,
                         "ch_x_area": 2.0}, index=origins)


# ----------------------------------------------------------------------------
# Frozen recipe / budget pins.
# ----------------------------------------------------------------------------

def test_params_single_sourced_frozen():
    assert final_model.FINAL_PARAMS == dict(protocol.PARAMS)
    assert final_model.FINAL_MEAN_PARAMS == dict(mean_level.MEAN_PARAMS)
    assert final_model.FINAL_ROUTER_PARAMS == dict(router_v2.ROUTER_PARAMS)
    assert final_model.FINAL_ROUTER_PARAMS == {"max_depth": 2, "random_state": 0}


def test_claim_ids_fixed_category_final():
    assert tuple(final_model.FINAL_CLAIM_IDS) == ("final_direct", "final_level", "final_router")
    assert final_model.FINAL_CLAIM_CATEGORY == "final"
    assert final_model.FINAL_ALLOWANCE == 3


def test_cutoff_rule_is_first_origin_minus_72h():
    got = final_model.cutoff_for_first_origin(FIRST)
    assert protocol.ts_utc(got) == protocol.ts_utc(CUTOFF)
    assert protocol.ts_utc(got) != protocol.ts_utc("2026-05-29T00:00:00Z")
    assert protocol.ts_utc(got) != protocol.ts_utc("2026-05-31T23:00:00Z")


def test_strict_cutoff_equality_fails():
    final_model.assert_strict_cutoff("2026-05-28T22:00:00Z", FIRST)
    with pytest.raises(ValueError):
        final_model.assert_strict_cutoff(CUTOFF, FIRST)
    with pytest.raises(ValueError):
        final_model.assert_strict_cutoff("2026-05-29T00:00:00Z", FIRST)


def test_complete_filled72_gate():
    final_model.assert_complete_filled72(np.full(72, 400.0))
    with pytest.raises(ValueError):
        final_model.assert_complete_filled72(np.full(71, 400.0))
    bad = np.full(72, 400.0)
    bad[3] = np.nan
    with pytest.raises(ValueError):
        final_model.assert_complete_filled72(bad)


# ----------------------------------------------------------------------------
# Exact requested grid (no drop / intersect / duplicate / bad horizon).
# ----------------------------------------------------------------------------

def test_exact_grid_roundtrip():
    req = final_model.canonical_final_keys(["2026-06-01T00:00:00Z", "2026-06-01T03:00:00Z"])
    assert len(req) == 144
    digest = final_model.assert_exact_final_grid(list(req), list(req))
    assert digest == protocol.index_digest(list(req))


def test_exact_grid_missing_key_fails():
    req = final_model.canonical_final_keys(["2026-06-01T00:00:00Z"])
    with pytest.raises(ValueError):
        final_model.assert_exact_final_grid(req[:-1], req)


def test_exact_grid_extra_key_fails():
    req = final_model.canonical_final_keys(["2026-06-01T00:00:00Z"])
    with pytest.raises(ValueError):
        final_model.assert_exact_final_grid(req + [("2026-06-02T00:00:00Z", 1)], req)


def test_exact_grid_duplicate_fails():
    req = final_model.canonical_final_keys(["2026-06-01T00:00:00Z"])
    with pytest.raises(ValueError):
        final_model.assert_exact_final_grid(req + [req[0]], req)


def test_exact_grid_noninteger_horizon_fails():
    req = final_model.canonical_final_keys(["2026-06-01T00:00:00Z"])
    bad = [(o, float(h)) for o, h in req]
    with pytest.raises(ValueError):
        final_model.assert_exact_final_grid(bad, req)


def test_horizon_support_per_origin():
    req = final_model.canonical_final_keys(["2026-06-01T00:00:00Z"])
    origins = ["2026-06-01T00:00:00Z"]
    final_contract.assert_horizon_support_per_origin(req, origins)
    with pytest.raises(ValueError):
        final_contract.assert_horizon_support_per_origin(req[:-72], origins)


# ----------------------------------------------------------------------------
# Poison compare: exact index + NaN mask first, pre-cutoff identical.
# ----------------------------------------------------------------------------

def test_poison_clean_passes_and_vacuous_detected_elsewhere():
    origins = _origins()
    clean, dirty = _feats(origins), _feats(origins)
    rep = final_model.poison_compare(clean, dirty, origins, "2026-05-28T23:00:00Z")
    assert rep["exact_index_validated"] and rep["nan_mask_validated"]
    assert rep["changed_features"] == [] and rep["pre_cutoff_changed"] == []


def test_poison_missing_origin_fails_before_compare():
    origins = _origins()
    clean = _feats(origins)
    dirty = _feats(origins[1:])
    with pytest.raises(ValueError):
        final_model.poison_compare(clean, dirty, origins, "2026-05-28T23:00:00Z")


def test_poison_nan_mask_mismatch_fails():
    origins = _origins()
    clean, dirty = _feats(origins), _feats(origins)
    dirty.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        final_model.poison_compare(clean, dirty, origins, "2026-05-28T23:00:00Z")


def test_poison_pre_cutoff_change_flagged():
    origins = _origins(start=POISON_START)
    clean, dirty = _feats(origins), _feats(origins)
    dirty.iloc[0, 0] += 999.0  # pre-cutoff origin
    rep = final_model.poison_compare(clean, dirty, origins, "2026-05-28T23:00:00Z")
    assert rep["pre_cutoff_changed"] == ["v_lag0"]
    assert rep["changed_features"] == []


def test_poison_post_cutoff_change_reported():
    origins = _origins(start=POISON_START)
    clean, dirty = _feats(origins), _feats(origins)
    dirty.iloc[-1, 0] += 999.0  # post-cutoff origin
    rep = final_model.poison_compare(clean, dirty, origins, "2026-05-28T23:00:00Z")
    assert rep["changed_features"] == ["v_lag0"]
    assert rep["pre_cutoff_changed"] == []


def test_inference_schema_order_strict():
    origins = _origins()
    feats = _feats(origins)
    final_model.assert_inference_schema(feats, list(feats.columns))
    with pytest.raises(ValueError):
        final_model.assert_inference_schema(feats, list(feats.columns)[::-1])
    with pytest.raises(ValueError):
        final_model.assert_inference_schema(feats, list(feats.columns)[:-1])


# ----------------------------------------------------------------------------
# OOF profiles: exactly 3 valid partitions, never fewer.
# ----------------------------------------------------------------------------

def test_require_three_oof_profiles():
    good = {q: {"router_labels_ok": True} for q in router_v2.OOF_ORDER}
    final_model.require_three_oof_profiles(good)
    with pytest.raises(ValueError):
        final_model.require_three_oof_profiles({q: {"router_labels_ok": True}
                                                for q in router_v2.OOF_ORDER[:2]})
    bad = {q: {"router_labels_ok": True} for q in router_v2.OOF_ORDER}
    bad["2025Q1"] = {"router_labels_ok": False}
    with pytest.raises(ValueError):
        final_model.require_three_oof_profiles(bad)


# ----------------------------------------------------------------------------
# Exactly-3 fit budget, no hidden verifier fits.
# ----------------------------------------------------------------------------

def test_run_fit_plan_exactly_three():
    calls = []
    out = final_model.run_fit_plan([lambda i=i: calls.append(i) or i for i in range(3)])
    assert out == [0, 1, 2] and len(calls) == 3
    with pytest.raises(ValueError):
        final_model.run_fit_plan([lambda: 1, lambda: 2])
    with pytest.raises(ValueError):
        final_model.run_fit_plan([lambda: 1] * 4)


def test_counted_fit_increments_once():
    final_model.FIT_EXECUTIONS["n"] = 0
    final_model.counted_fit(lambda x: x, 1)
    final_model.counted_fit(lambda x: x, 2)
    assert final_model.FIT_EXECUTIONS["n"] == 2
    final_model.assert_exactly_three_fit_calls(3)
    with pytest.raises(ValueError):
        final_model.assert_exactly_three_fit_calls(2)
    with pytest.raises(ValueError):
        final_model.assert_exactly_three_fit_calls(4)
    final_model.FIT_EXECUTIONS["n"] = 0


def test_native_parity_bounds():
    assert final_model.check_native_parity(1e-10, 0.0)["ok"]
    assert not final_model.check_native_parity(1e-6, 0.0)["ok"]
    assert not final_model.check_native_parity(0.0, 1e-3)["ok"]


# ----------------------------------------------------------------------------
# Readiness proof: actual log, exact command, exit 0, no self-hash, no rewrite.
# ----------------------------------------------------------------------------

def _write_log(path, n, failures=0):
    lines = f"tests/{final_model.FINAL_TEST_FILE.split('/')[-1]} ....\n{n} passed"
    if failures:
        lines += f", {failures} failed"
    path.write_text(lines + "\n")


def test_mark_and_require_proof(tmp_path):
    inv = final_model.final_tests_inventory("/home/t-lab01/sunrun")
    log = tmp_path / "pytest.log"
    _write_log(log, inv["n_tests"])
    dest = final_model.mark_final_tests_passed(str(tmp_path), str(log), repo_root="/home/t-lab01/sunrun")
    proof = json.loads(dest.read_text())
    assert proof["exit_code"] == 0 and proof["passed_count"] == inv["n_tests"]
    assert "pytest" in proof["command"] and "test_scope_expansion_final_model.py" in proof["command"]
    assert final_model.require_final_tests_passed(str(tmp_path), "/home/t-lab01/sunrun")["log_sha256"] == proof["log_sha256"]


def test_mark_rejects_wrong_command(tmp_path):
    inv = final_model.final_tests_inventory("/home/t-lab01/sunrun")
    log = tmp_path / "pytest.log"
    _write_log(log, inv["n_tests"])
    with pytest.raises(protocol.NotReadyError):
        final_model.mark_final_tests_passed(str(tmp_path), str(log), command="pytest other.py -q",
                                            repo_root="/home/t-lab01/sunrun")


def test_mark_rejects_bad_count_and_failures(tmp_path):
    log = tmp_path / "pytest.log"
    _write_log(log, 1)
    with pytest.raises(protocol.NotReadyError):
        final_model.mark_final_tests_passed(str(tmp_path), str(log), repo_root="/home/t-lab01/sunrun")
    log2 = tmp_path / "pytest2.log"
    inv = final_model.final_tests_inventory("/home/t-lab01/sunrun")
    _write_log(log2, inv["n_tests"], failures=1)
    with pytest.raises(protocol.NotReadyError):
        final_model.mark_final_tests_passed(str(tmp_path), str(log2), repo_root="/home/t-lab01/sunrun")


def test_mark_rejects_nonzero_exit_and_rewrite(tmp_path):
    inv = final_model.final_tests_inventory("/home/t-lab01/sunrun")
    log = tmp_path / "pytest.log"
    _write_log(log, inv["n_tests"])
    with pytest.raises(protocol.NotReadyError):
        final_model.mark_final_tests_passed(str(tmp_path), str(log), exit_code=1,
                                            repo_root="/home/t-lab01/sunrun")
    final_model.mark_final_tests_passed(str(tmp_path), str(log), repo_root="/home/t-lab01/sunrun")
    with pytest.raises(protocol.DuplicateClaimError):
        final_model.mark_final_tests_passed(str(tmp_path), str(log), repo_root="/home/t-lab01/sunrun")


def test_require_proof_missing_fails(tmp_path):
    with pytest.raises(protocol.NotReadyError):
        final_model.require_final_tests_passed(str(tmp_path), "/home/t-lab01/sunrun")


# ----------------------------------------------------------------------------
# Append-only: pre-existing receipts refuse before any I/O.
# ----------------------------------------------------------------------------

def test_stages_refuse_overwrite(tmp_path):
    fdir = tmp_path / "final"
    fdir.mkdir()
    (fdir / final_model.FINAL_PREPARE_FILENAME).write_text("{}")
    with pytest.raises(protocol.DuplicateClaimError):
        final_model.stage_prepare(str(tmp_path), "/home/t-lab01/sunrun")
    (fdir / final_model.FINAL_POISON_FILENAME).write_text("{}")
    with pytest.raises(protocol.DuplicateClaimError):
        final_model.stage_poison_check(str(tmp_path))
    (fdir / final_model.FINAL_REGISTRATION_FILENAME).write_text("{}")
    with pytest.raises(protocol.DuplicateClaimError):
        final_model.stage_register(str(tmp_path), "/home/t-lab01/sunrun")
    (fdir / final_model.FINAL_FIT_FILENAME).write_text("{}")
    with pytest.raises(protocol.DuplicateClaimError):
        final_model.stage_fit(str(tmp_path), "/home/t-lab01/sunrun")
    (fdir / final_model.FINAL_PREDICTIONS_FILENAME).write_text("x")
    with pytest.raises(protocol.DuplicateClaimError):
        final_model.stage_predict(str(tmp_path))
    (fdir / final_model.FINAL_VERIFY_FILENAME).write_text("{}")
    with pytest.raises(protocol.DuplicateClaimError):
        final_model.stage_verify(str(tmp_path))


def test_official_grid_missing_config_fails_closed(tmp_path):
    import os

    os.environ.pop(final_model.FOLDS_ENV_VAR, None)
    with pytest.raises(protocol.NotReadyError):
        final_model.read_official_origin_grid()


def test_official_grid_wrong_size_fails(tmp_path):
    import pyarrow.parquet as pq  # noqa: F401  (parquet engine for the stub grid)

    origins = _origins(n=4)
    frame = pd.DataFrame({final_model.FOLDS_ORIGIN_COLUMN: origins.astype(str)})
    p = tmp_path / "folds.parquet"
    frame.to_parquet(p)
    with pytest.raises(ValueError):
        final_model.read_official_origin_grid(p)


def test_official_grid_hash_stable():
    origins = _origins(n=4)
    assert final_model.official_grid_hash(origins) == final_model.official_grid_hash(list(origins))


# ----------------------------------------------------------------------------
# Provenance closure pins every frozen builder.
# ----------------------------------------------------------------------------

def test_transitive_closure_covers_frozen_builders():
    rels = final_model.TRANSITIVE_CLOSURE_RELS
    for must in ("protocol.py", "reference.py", "mean_level.py", "router_v2.py",
                 "selection.py", "final_contract.py", "evaluate.py", "scoring.py"):
        assert any(r.endswith(must) for r in rels), must
    hashes = final_model.transitive_closure_hashes("/home/t-lab01/sunrun")
    assert set(hashes) == set(rels) and all(len(v) == 64 for v in hashes.values())


# ----------------------------------------------------------------------------
# Guards repair: OOF arms/canonical/FINAL-cutoff/in-sample (file-shaped).
# ----------------------------------------------------------------------------

def test_oof_arms_level_maps_to_mean_no_keyerror(tmp_path):
    import pyarrow.parquet as pq  # noqa: F401

    assert tuple(router_v2.OOF_ARMS) == ("direct", "level")
    for q in router_v2.OOF_ORDER:
        for arm in router_v2.OOF_ARMS:
            frame = pd.DataFrame({"origin": ["2024-10-05T00:00:00Z"],
                                  "h": [1], "y_pred": [400.0]})
            frame.to_parquet(tmp_path / f"predictions_oof_{q}_{arm}.parquet", index=False)
    for q in router_v2.OOF_ORDER:
        d, m = final_model._load_oof_pair(tmp_path, q)
        assert list(d.columns) == [mean_level.CANON_ORIGIN, mean_level.CANON_H, mean_level.CANON_PRED]
        assert list(m.columns) == [mean_level.CANON_ORIGIN, mean_level.CANON_H, mean_level.CANON_PRED]


def test_canonicalize_accepts_alias_and_canonical():
    alias = pd.DataFrame({"origin": ["2024-10-05T00:00:00Z"], "h": [1], "y_pred": [1.0]})
    canon = pd.DataFrame({mean_level.CANON_ORIGIN: ["2024-10-05T00:00:00Z"],
                          mean_level.CANON_H: [1], mean_level.CANON_PRED: [1.0]})
    assert list(final_model._canonicalize_oof_frame(alias).columns) == [
        mean_level.CANON_ORIGIN, mean_level.CANON_H, mean_level.CANON_PRED]
    assert list(final_model._canonicalize_oof_frame(canon).columns) == [
        mean_level.CANON_ORIGIN, mean_level.CANON_H, mean_level.CANON_PRED]
    with pytest.raises(ValueError):
        final_model._canonicalize_oof_frame(pd.DataFrame({"a": [1]}))


def test_oof_profiles_pin_final_cutoff_source():
    import inspect

    src = inspect.getsource(final_model._final_oof_label_profiles)
    assert "PINNED_TRAIN_CUTOFF_UTC" in src
    assert "_final_oof_base_train_origins" in src and "_load_oof_pair" in src
    assert 'pair["mean"]' not in src
    assert "oof_bounds(q)[1]" not in src
    fit_src = inspect.getsource(final_model.stage_fit)
    assert "final_router_training(run, man)" in fit_src
    training_src = inspect.getsource(final_model.final_router_training)
    assert "_load_oof_pair" in training_src and "base_train_origins=base_map[q]" in training_src
    assert "PINNED_TRAIN_CUTOFF_UTC" in training_src
    assert 'pair["mean"]' not in fit_src


def test_fit_router_uses_in_sample_exclusion_source():
    import inspect

    assert "base_train_origins" in inspect.getsource(final_model._final_oof_base_train_origins)
    assert "build_mean_training" in inspect.getsource(final_model._final_oof_base_train_origins)
    assert "base_cutoff_for_partition" in inspect.getsource(final_model._final_oof_base_train_origins)


# ----------------------------------------------------------------------------
# Guards repair: poison semantics (allow post, forbid pre; earlier control).
# ----------------------------------------------------------------------------

def test_poison_semantics_allow_post_forbid_pre():
    origins = _origins(start=POISON_START)
    clean = _feats(origins)
    post_only = _feats(origins)
    post_only.iloc[-1, 0] += 5.0
    rep = final_model.poison_compare(clean, post_only, origins, CUTOFF)
    assert rep["changed_features"] == ["v_lag0"] and rep["pre_cutoff_changed"] == []
    pre_bad = _feats(origins)
    pre_bad.iloc[0, 0] += 5.0
    rep2 = final_model.poison_compare(clean, pre_bad, origins, CUTOFF)
    assert rep2["pre_cutoff_changed"] == ["v_lag0"]


def test_live_pins_reject_pre_not_any_post_source():
    import inspect

    src = inspect.getsource(final_model._assert_live_pins)
    assert "pre_cutoff_changed" in src
    assert "positive_control_changed" in src
    assert "nan_mask_validated" in src


def test_poison_control_not_may31_source():
    import inspect

    src = inspect.getsource(final_model.stage_poison_check)
    assert "control_cutoff = PINNED_FIRST_OFFICIAL_ORIGIN_UTC" not in src
    assert "official_causality" in src and "future_invariant" in src


def test_official_rep_origins_earliest_middle_latest():
    idx = pd.date_range("2026-05-31T23:00:00Z", periods=7, freq="h", tz="UTC")
    reps = final_model._official_rep_origins(idx)
    assert len(reps) == 3 and reps[0] == idx[0] and reps[-1] == idx[-1]


def test_pinned_folds_path_rejects_lookalike(monkeypatch):
    import os

    monkeypatch.setenv(final_model.FOLDS_ENV_VAR, "/tmp/folds.parquet")
    with pytest.raises(protocol.NotReadyError):
        final_model._assert_pinned_folds_path()
    monkeypatch.setenv(final_model.FOLDS_ENV_VAR,
                       f"/home/t-lab01/.cache/huggingface/hub/datasets--tlabtlab--sunrun-lb-store/snapshots/{final_model.PINNED_SNAPSHOT_SHA256}/folds.parquet")
    final_model._assert_pinned_folds_path()
    monkeypatch.delenv(final_model.FOLDS_ENV_VAR, raising=False)


# ----------------------------------------------------------------------------
# Guards repair: exact 7-category budget + full verify reconstruction.
# ----------------------------------------------------------------------------

def test_claim_budget_requires_all7_and_base15_plus3():
    full = {k: 0 for k in protocol.CLAIM_ALLOCATION}
    full.update({"reference_replay": 3, "mean_level": 3, "oof_base": 6,
                 "router_eval": 3, "final": 3})
    assert final_contract.check_claim_budget(full, 18)["ok"]
    assert not final_contract.check_claim_budget({"final": 3}, 18)["ok"]
    bad_source = dict(full)
    bad_source["source_pilot"] = 1
    assert not final_contract.check_claim_budget(bad_source, 19)["ok"]


def test_verify_reconstructs_full_not_sample_source():
    import inspect

    src = inspect.getsource(final_model.stage_verify)
    assert "head(256)" not in src
    assert "EXPECTED_OFFICIAL_TOTAL_KEYS" in src
    assert "replay_routed" in src and "route_predictions" in src
    assert "decompose_predictions" in src and "scalar_booster" in src
    assert "max_depth" in src and "direct_trees" in src
    assert "ledger_counts_by_category" in src
    assert "ZERO" in src or "fits_before" in src
    assert '{"final": 3}' not in src and "{FINAL_CLAIM_CATEGORY: 3}" not in src


def test_predict_dmatrix_preserves_feature_names_source():
    import inspect

    src = inspect.getsource(final_model.stage_predict)
    assert "xgb.DMatrix(feat_ordered)" in src
    assert "feature_names" in src


def test_scalar_uses_native_booster_only_source():
    import inspect

    fit_src = inspect.getsource(final_model.stage_fit)
    assert "fit_mean_level" in fit_src
    assert "save_model(" not in fit_src
    recipe_src = inspect.getsource(mean_level.fit_mean_level)
    assert "get_booster().save_model" in recipe_src
    assert recipe_src.count("model.fit(") == 1


def test_endtoend_stub_three_fits_and_tamper_detected():
    calls = []
    final_model.run_fit_plan([lambda i=i: calls.append(i) or i for i in range(3)])
    assert len(calls) == 3
    origins = pd.DatetimeIndex(["2026-06-01T00:00:00Z", "2026-06-01T01:00:00Z"], tz="UTC")
    req = final_model.canonical_final_keys([str(o) for o in origins])
    assert len(req) == 144
    direct = pd.DataFrame({mean_level.CANON_ORIGIN: np.repeat(np.asarray(origins), 72),
                           mean_level.CANON_H: np.tile(np.arange(1, 73), len(origins)),
                           mean_level.CANON_PRED: np.full(144, 400.0)})
    mean = direct.copy()
    mean[mean_level.CANON_PRED] = 500.0
    choices = pd.Series([0, 1], index=protocol.idx_utc(origins))
    routed = router_v2.route_predictions(choices, direct, mean)
    assert len(routed) == 144
    tampered = routed.copy()
    tampered.loc[tampered[mean_level.CANON_H] == 1, mean_level.CANON_PRED] += 10.0
    diff = float(np.max(np.abs(routed[mean_level.CANON_PRED].to_numpy(float)
                                   - tampered[mean_level.CANON_PRED].to_numpy(float))))
    assert diff > 1e-9
    assert not final_model.check_native_parity(diff, 0.0)["ok"]
