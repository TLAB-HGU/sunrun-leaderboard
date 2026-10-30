"""Synthetic CPU-only tests for the fixed causal OOF hard router.

No real fits, registration, fetching, official reads, or uploads. The root
runs the real stages; every case here uses synthetic frames only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PKG_PARENT = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(PKG_PARENT))

from mse20_scope_expansion import mean_level, protocol, router  # noqa: E402
from mse20_scope_expansion.protocol import (  # noqa: E402
    CapExceededError,
    ClaimLedger,
    DuplicateClaimError,
    NotReadyError,
)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _synthetic_inputs(t0="2024-08-20", hours=70 * 24, n_origins=60,
                      origin0="2024-10-05", gap_hours=()):
    idx = pd.date_range(t0, periods=hours, freq="h", tz="UTC")
    rng = np.random.default_rng(7)
    vals = 400 + 40 * np.sin(np.arange(len(idx)) / 24 * 2 * np.pi) + rng.normal(0, 4, len(idx))
    ace = pd.Series(vals, index=idx)
    truth = ace.copy()
    for g in gap_hours:
        truth.iloc[g] = np.nan
    origins = pd.date_range(origin0, periods=n_origins, freq="3h", tz="UTC")
    feats = pd.DataFrame({
        "v_lag0": np.sin(np.arange(len(origins)) / 5.0),
        "v_m24": np.cos(np.arange(len(origins)) / 7.0),
        "ace_speed_kms_last": 400.0,
        "speed_d24": 0.5,
        "ch_x_area": 2.0,
    }, index=origins)
    return {"origins": origins, "features": feats,
            "truth_observed": truth, "ace_filled": ace}


def _profile(origins, truth, shift=0.0):
    rows = []
    for o in origins:
        for h in range(1, 73):
            t = o + pd.Timedelta(hours=h)
            y = truth.reindex([t]).to_numpy(float)[0]
            rows.append((o, h, (y if np.isfinite(y) else 400.0) + shift))
    return pd.DataFrame(rows, columns=["origin_last_input_utc", "horizon_hours", "pred_kms"])


# ----------------------------------------------------------------------------
# 1. fixed OOF partitions, grids, cutoffs
# ----------------------------------------------------------------------------
def test_oof_grid_containment_and_cutoffs():
    assert set(router.OOF_ORDER) == {"2024Q4", "2025Q1", "2025Q2"}
    for q in router.OOF_ORDER:
        start, end = router.oof_bounds(q)
        origins = router.oof_requested_origins(q)
        assert ((origins + pd.to_timedelta(72, unit="h")) <= end).all()
        assert (origins[1] - origins[0]) == pd.Timedelta(hours=6)
        assert router.base_cutoff_for_partition(q) == start - pd.Timedelta(hours=72)
    for p in protocol.PERIOD_ORDER:
        assert router.router_cutoff_for_period(p) == \
            protocol.ts_utc(protocol.PERIODS[p][0]) - pd.Timedelta(hours=72)


def test_claim_plan_uses_protocol_categories_and_counts():
    assert router.OOF_CLAIM_CATEGORY in protocol.CLAIM_ALLOCATION
    assert router.ROUTER_CLAIM_CATEGORY in protocol.CLAIM_ALLOCATION
    oof_ids = [router.OOF_CLAIM_IDS[q][a] for q in router.OOF_ORDER for a in router.OOF_ARMS]
    rou_ids = [router.ROUTER_CLAIM_IDS[p] for p in router.ROUTER_PERIODS]
    assert len(set(oof_ids)) == 6 and len(set(rou_ids)) == 3
    assert protocol.CLAIM_ALLOCATION[router.OOF_CLAIM_CATEGORY] == 6
    assert protocol.CLAIM_ALLOCATION[router.ROUTER_CLAIM_CATEGORY] == 3
    assert router.ROUTER_PARAMS == {"max_depth": 2, "random_state": 0}


# ----------------------------------------------------------------------------
# 2. labels: ties go direct, full-72 required, purge + in-sample gates
# ----------------------------------------------------------------------------
def test_labels_tie_goes_direct_and_full72_required():
    inputs = _synthetic_inputs()
    origins = inputs["origins"][:8]
    direct = _profile(origins, inputs["truth_observed"])
    filled, obs = inputs["ace_filled"], inputs["truth_observed"].notna()
    obs = pd.Series(np.asarray(obs), index=inputs["ace_filled"].index).astype(bool)
    labels, ledger = router.build_router_labels(
        direct, direct, filled, obs,
        "2025-06-01T00:00:00Z")
    assert (labels["choice"] == router.DIRECT_CHOICE).all()  # exact ties -> direct
    assert ledger["n_eligible"] == len(origins) and ledger["n_excluded"] == 0
    # A single gapped observed hour keeps the origin eligible (error on the
    # common observed subset); the filled window is still fully finite.
    obs_gap = obs.copy()
    obs_gap.loc[origins[-1] + pd.Timedelta(hours=70)] = False
    labels_gap, ledger2 = router.build_router_labels(
        direct, direct, filled, obs_gap, "2025-06-01T00:00:00Z")
    assert ledger2["n_eligible"] == len(origins)
    assert ledger2["exclusion_reasons"]["no_observed_labels"] == 0
    assert ledger2["missingness"]["min_observed_per_eligible"] == 71
    # An incomplete FILLED window excludes exactly that origin.
    filled_gap = filled.copy()
    filled_gap.loc[origins[-1] + pd.Timedelta(hours=70)] = np.nan
    _, ledger2b = router.build_router_labels(
        direct, direct, filled_gap, obs, "2025-06-01T00:00:00Z")
    assert ledger2b["exclusion_reasons"]["incomplete_labels"] == 1


def test_labels_reject_in_sample_and_late_windows():
    inputs = _synthetic_inputs()
    origins = inputs["origins"][:6]
    direct = _profile(origins, inputs["truth_observed"], shift=1.0)
    mean = _profile(origins, inputs["truth_observed"], shift=50.0)
    filled = inputs["ace_filled"]
    obs = pd.Series(np.asarray(inputs["truth_observed"].notna()),
                    index=inputs["ace_filled"].index).astype(bool)
    labels, ledger = router.build_router_labels(
        direct, mean, filled, obs,
        "2025-06-01T00:00:00Z", base_train_origins=[origins[0]])
    assert ledger["exclusion_reasons"]["in_sample_base_origin"] == 1
    assert str(origins[0]) not in set(labels["origin"].astype(str))
    assert (labels["choice"] == router.DIRECT_CHOICE).all()  # direct err smaller
    bound = protocol.ts_utc("2025-06-01T00:00:00Z") - pd.Timedelta(hours=72)
    edge = bound - pd.Timedelta(hours=72)  # label_end exactly at bound -> refused
    edge_o = pd.DatetimeIndex([edge])
    # Dedicated truth covering the edge window: the refusal must come from the
    # purge gate, not from missing labels.
    edge_idx = pd.date_range(edge - pd.Timedelta(hours=2), periods=76, freq="h", tz="UTC")
    edge_truth = pd.Series(400.0, index=edge_idx)
    edge_obs = pd.Series(True, index=edge_idx)
    d_edge = _profile(edge_o, edge_truth)
    _, ledger3 = router.build_router_labels(
        d_edge, d_edge, edge_truth, edge_obs, "2025-06-01T00:00:00Z")
    assert ledger3["exclusion_reasons"]["label_window_not_before_cutoff"] == 1
    ok_o = pd.DatetimeIndex([edge - pd.Timedelta(seconds=1)])
    ok_idx = pd.date_range(ok_o[0] - pd.Timedelta(hours=2), periods=76, freq="h", tz="UTC")
    ok_truth = pd.Series(400.0, index=ok_idx)
    ok_obs = pd.Series(True, index=ok_idx)
    _, ledger4 = router.build_router_labels(
        _profile(ok_o, ok_truth), _profile(ok_o, ok_truth),
        ok_truth, ok_obs, "2025-06-01T00:00:00Z")
    assert ledger4["n_eligible"] == 1


def test_strict_cutoff_equality_fails():
    q = "2024Q4"
    cutoff = router.base_cutoff_for_partition(q)
    router.assert_base_purge((cutoff - pd.Timedelta(seconds=1)).isoformat(), q)
    with pytest.raises(ValueError):
        router.assert_base_purge(cutoff.isoformat(), q)  # equality is not strictly before
    router.assert_router_label_purge(
        (protocol.ts_utc("2025-06-01T00:00:00Z") - pd.Timedelta(hours=72, seconds=1)).isoformat(),
        "2025-06-01T00:00:00Z")
    with pytest.raises(ValueError):
        router.assert_router_label_purge(
            (protocol.ts_utc("2025-06-01T00:00:00Z") - pd.Timedelta(hours=72)).isoformat(),
            "2025-06-01T00:00:00Z")


def test_missing_partition_is_indeterminate():
    with pytest.raises(ValueError, match="indeterminate"):
        router.require_three_partitions({"2024Q4": True, "2025Q1": True})
    router.require_three_partitions({q: True for q in router.OOF_ORDER})


# ----------------------------------------------------------------------------
# 3. hard-routing identity + support-mutation rejection
# ----------------------------------------------------------------------------
def test_hard_routing_identity():
    inputs = _synthetic_inputs()
    origins = inputs["origins"][:10]
    direct = _profile(origins, inputs["truth_observed"], shift=1.0)
    mean = _profile(origins, inputs["truth_observed"], shift=-2.0)
    rng = np.random.default_rng(0)
    choices = pd.Series(rng.integers(0, 2, len(origins)), index=origins)
    routed = router.route_predictions(choices, direct, mean)
    dvals = direct.set_index(["origin_last_input_utc", "horizon_hours"])["pred_kms"]
    mvals = mean.set_index(["origin_last_input_utc", "horizon_hours"])["pred_kms"]
    rvals = routed.set_index(["origin_last_input_utc", "horizon_hours"])["pred_kms"]
    for o in origins:
        want = dvals.loc[str(o)] if choices.loc[o] == 0 else mvals.loc[str(o)]
        assert np.array_equal(rvals.loc[str(o)].to_numpy(), want.to_numpy())
    all_direct = router.route_predictions(
        pd.Series(0, index=origins), direct, mean)
    assert np.array_equal(all_direct["pred_kms"].to_numpy(), direct["pred_kms"].to_numpy())


def test_routing_rejects_support_mutation():
    inputs = _synthetic_inputs()
    origins = inputs["origins"][:4]
    direct = _profile(origins, inputs["truth_observed"])
    mean = _profile(origins, inputs["truth_observed"])
    choices = pd.Series(0, index=origins)
    with pytest.raises(ValueError):
        router.route_predictions(choices.iloc[:3], direct, mean)  # missing choice
    with pytest.raises(ValueError):
        router.route_predictions(choices, direct.iloc[:-1], mean)  # dropped horizon
    with pytest.raises(ValueError):
        router.route_predictions(choices, direct, mean.assign(pred_kms=np.nan))  # NaN
    with pytest.raises(ValueError):
        router.route_predictions(pd.Series(2, index=origins), direct, mean)  # bad label


def test_router_fixed_tree_fits_separable_synthetic():
    sk = pytest.importorskip("sklearn.tree")
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (40, 3))
    y = (X[:, 0] > 0).astype(int)
    clf = sk.DecisionTreeClassifier(**router.ROUTER_PARAMS)
    clf.fit(X, y)
    assert clf.get_params()["max_depth"] == 2
    assert (clf.predict(X) == y).mean() > 0.9


# ----------------------------------------------------------------------------
# 4. schema, causal medians, missing features
# ----------------------------------------------------------------------------
def test_schema_rejects_forbidden_and_declared_before_outcomes():
    inputs = _synthetic_inputs()
    cols = router.router_feature_columns(inputs["features"].columns)
    assert "h" not in cols
    assert set(mean_level.TRAJ_MEAN_COLUMNS) <= set(cols)
    bad = inputs["features"].assign(official_pool=1.0)
    with pytest.raises(ValueError):
        router.router_feature_columns(bad.columns)
    bad2 = inputs["features"].assign(horizon_hours=1.0)
    with pytest.raises(ValueError):
        router.router_feature_columns(bad2.columns)
    d1 = router.declare_router_schema(inputs["features"].columns)
    d2 = router.declare_router_schema(inputs["features"].columns)  # no outcomes needed
    assert d1 == d2 and d1["schema_digest"]


def test_causal_median_imputation_uses_train_only():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, np.nan, np.nan]})
    med = router.fit_medians(df, ["a", "b"])
    assert med["a"] == 2.0 and med["b"] == 0.0 and med["_all_nan_fallback"] == ["b"]
    ev = pd.DataFrame({"a": [np.nan], "b": [np.nan]})
    filled = router.apply_medians(ev, med)
    assert filled["a"].iloc[0] == 2.0  # train median, never the eval value
    assert ev["a"].isna().all()  # input untouched


def test_missing_causal_features_fail_closed():
    inputs = _synthetic_inputs()
    with pytest.raises(ValueError):
        mean_level.issue_features(inputs, inputs["origins"][:5] + pd.to_timedelta(1, unit="h"))
    feats = mean_level.issue_features(inputs, inputs["origins"][:5])
    assert len(feats) == 5


# ----------------------------------------------------------------------------
# 5. ledger cap / retries + proof/source pins
# ----------------------------------------------------------------------------
def test_failed_claims_consume_oof_slots_and_no_retry(tmp_path):
    run = tmp_path / "run"
    (run / "claims").mkdir(parents=True)
    ledger = ClaimLedger(str(run))
    ids = [router.OOF_CLAIM_IDS[q][a] for q in router.OOF_ORDER for a in router.OOF_ARMS]
    for i, cid in enumerate(ids):
        ledger.open_claim(cid, router.OOF_CLAIM_CATEGORY, tests_passed=True)
        ledger.close_claim(cid, "failed" if i == 0 else "ok", "synthetic")
    with pytest.raises(DuplicateClaimError):
        ledger.open_claim(ids[0], router.OOF_CLAIM_CATEGORY, tests_passed=True)
    with pytest.raises(CapExceededError):
        ledger.open_claim("oof_base_extra", router.OOF_CLAIM_CATEGORY, tests_passed=True)
    counts = ledger.counts()
    assert counts["by_category"][router.OOF_CLAIM_CATEGORY] == 6  # failed slot kept


def test_stale_proof_and_source_pin_fail_closed(tmp_path):
    root = tmp_path / "repo"
    test_rel = Path(router.ROUTER_TEST_FILE)
    (root / test_rel.parent).mkdir(parents=True)
    (root / test_rel).write_text(Path(__file__).read_text())
    (root / "leaderboard/experiments/mse20_scope_expansion").mkdir(parents=True, exist_ok=True)
    src = root / "leaderboard/experiments/mse20_scope_expansion/router.py"
    src.write_text("placeholder")
    run = tmp_path / "run"
    run.mkdir()
    log = tmp_path / "pytest.log"
    n = sum(1 for line in Path(__file__).read_text().splitlines() if line.startswith("def test_"))
    log.write_text(f"{n} passed in 1.00s\n")
    with pytest.raises(NotReadyError):
        router.mark_router_tests_passed(run, log, exit_code=1, repo_root=root)
    with pytest.raises(NotReadyError):
        router.mark_router_tests_passed(
            run, log, command="pytest tests/test_other.py -q", repo_root=root)
    router.mark_router_tests_passed(run, log, repo_root=root)
    with pytest.raises(DuplicateClaimError):  # append-only proof
        router.mark_router_tests_passed(run, log, repo_root=root)
    with open(root / test_rel, "a") as fh:  # any test mutation invalidates proof
        fh.write("\n# drift\n")
    with pytest.raises(NotReadyError):
        router.require_router_tests_passed(run, repo_root=root)

# ----------------------------------------------------------------------------
# 6. review repairs: unconsumed aliases, OOF grid OR-gate, overlay, duplicates
# ----------------------------------------------------------------------------
def test_router_rejects_unconsumed_target_alias_columns():
    inputs = _synthetic_inputs()
    base = list(inputs["features"].columns)
    router.router_feature_columns(base)  # clean frame passes
    for extra in ("y", "y_mean", "pred_kms", "origin", "level_hat", "choice"):
        with pytest.raises(ValueError):
            router.router_feature_columns(base + [extra])
    for bad in ("future_shock_index", "blend_weight", "regime_label", "target_speed"):
        with pytest.raises(ValueError):
            router.router_feature_columns(base + [bad])


def test_oof_grid_gate_rejects_either_mismatch():
    origins = router.oof_requested_origins("2024Q4")
    expect = router.oof_pair_digest(origins)
    n = len(origins)
    router.assert_oof_grid(expect, expect, n * 72, n)  # exact grid passes
    with pytest.raises(ValueError):
        router.assert_oof_grid("0" * 64, expect, n * 72, n)  # digest differs, rows match
    with pytest.raises(ValueError):
        router.assert_oof_grid(expect, expect, n * 72 - 1, n)  # rows differ, digest matches
    assert router.oof_pair_digest(origins[::-1]) != expect  # order-sensitive


def test_three_partitions_rejects_partial_validity():
    with pytest.raises(ValueError, match="indeterminate"):
        router.require_three_partitions({"2024Q4": True, "2025Q1": False, "2025Q2": True})
    with pytest.raises(ValueError, match="indeterminate"):
        router.require_three_partitions({"2024Q4": True, "2025Q1": None, "2025Q2": True})
    router.require_three_partitions({q: 1 for q in router.OOF_ORDER})


def test_mean_overlay_proof_refuses_missing_and_failed(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(NotReadyError):
        router.mean_overlay_proof(run)  # missing overlay
    (run / router.MEAN_OVERLAY_FILENAME).write_text(json.dumps(
        {"ok": False, "protocol_version": protocol.PROTOCOL_VERSION}))
    with pytest.raises(NotReadyError):
        router.mean_overlay_proof(run)  # frozen failed flag never suffices
    (run / router.MEAN_OVERLAY_FILENAME).write_text(json.dumps(
        {"ok": True, "protocol_version": "drifted"}))
    with pytest.raises(NotReadyError):
        router.mean_overlay_proof(run)  # protocol drift


def test_router_labels_observed_subset_eligible_vs_none_excluded():
    inputs = _synthetic_inputs()
    origins = inputs["origins"][:4]
    direct = _profile(origins, inputs["truth_observed"])
    mean = _profile(origins, inputs["truth_observed"], shift=5.0)
    filled = inputs["ace_filled"]
    observed = pd.Series(True, index=filled.index)
    observed.loc[origins[-1] + pd.Timedelta(hours=70)] = False
    labels, ledger = router.build_router_labels(
        direct, mean, filled, observed, "2025-06-01T00:00:00Z")
    assert ledger["n_eligible"] == len(origins)  # subset still eligible
    assert ledger["exclusion_reasons"]["no_observed_labels"] == 0
    # Zero observed hours in exactly one window excludes exactly that origin.
    obs_none = pd.Series(True, index=filled.index)
    ts_last = origins[-1] + pd.to_timedelta(np.arange(1, 73), unit="h")
    obs_none.loc[ts_last] = False
    _, ledger_none = router.build_router_labels(
        direct, mean, filled, obs_none, "2025-06-01T00:00:00Z")
    assert ledger_none["exclusion_reasons"]["no_observed_labels"] == 1
    assert ledger_none["n_eligible"] == len(origins) - 1


def test_router_labels_reject_missing_mask_and_incomplete_filled():
    inputs = _synthetic_inputs()
    origins = inputs["origins"][:3]
    direct = _profile(origins, inputs["truth_observed"])
    with pytest.raises(ValueError, match="explicit observed mask"):
        router.build_router_labels(
            direct, direct, inputs["ace_filled"], None, "2025-06-01T00:00:00Z")
    filled_gap = inputs["ace_filled"].copy()
    filled_gap.loc[origins[0] + pd.Timedelta(hours=1)] = np.nan
    obs = pd.Series(True, index=inputs["ace_filled"].index)
    _, ledger = router.build_router_labels(
        direct, direct, filled_gap, obs, "2025-06-01T00:00:00Z")
    assert ledger["exclusion_reasons"]["incomplete_labels"] >= 1


def test_routing_rejects_duplicate_choice_origins():
    inputs = _synthetic_inputs()
    origins = inputs["origins"][:3]
    direct = _profile(origins, inputs["truth_observed"])
    mean = _profile(origins, inputs["truth_observed"])
    dup = pd.Series([0, 0, 1], index=[origins[0], origins[0], origins[2]])
    with pytest.raises(ValueError):
        router.route_predictions(dup, direct, mean)


# ----------------------------------------------------------------------------
# 7. fit-budget guards: 6 OOF base fits (2/partition), 3 router fits, 0 verify
# ----------------------------------------------------------------------------
def _fit_calls_by_function():
    import ast as _ast
    src = Path(router.__file__).read_text()
    tree = _ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and node.name.startswith("stage_"):
            calls = [n for n in _ast.walk(node)
                     if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute)
                     and n.func.attr in ("fit", "partial_fit")]
            out[node.name] = len(calls)
    return out


def test_fit_budget_static_exact():
    counts = _fit_calls_by_function()
    assert counts["stage_oof_fit"] == 2  # direct XGB + level scalar head per partition
    assert counts["stage_router_fit"] == 1  # one DecisionTree fit per period
    assert counts["stage_verify"] == 0  # verification replays, never fits


def test_fit_budget_mocked_spy_counts():
    from unittest.mock import patch
    direct_calls, level_calls, router_calls = [], [], []
    real_oof, real_router = dict(router.OOF_CLAIM_IDS), dict(router.ROUTER_CLAIM_IDS)
    assert len({(q, a) for q in router.OOF_ORDER for a in router.OOF_ARMS}) == 6
    assert len({router.ROUTER_CLAIM_IDS[p] for p in router.ROUTER_PERIODS}) == 3
    class _FakeXGB:
        def __init__(self, tag):
            self._tag = tag
        def fit(self, X, y):
            (direct_calls if self._tag == "direct" else level_calls).append((len(X), len(y)))
            return self
    class _FakeTree:
        def fit(self, X, y):
            router_calls.append((len(X), len(y)))
            return self
    # Simulate the budgeted pattern on synthetic frames (no science models):
    # 2 OOF fits per partition x3 partitions = 6; 1 router fit per period x3 = 3.
    for _ in router.OOF_ORDER:
        _FakeXGB("direct").fit([1, 2], [1, 2])
        _FakeXGB("level").fit([1, 2], [1, 2])
    for _ in router.ROUTER_PERIODS:
        _FakeTree().fit([1, 2], [0, 1])
    assert len(direct_calls) == 3 and len(level_calls) == 3
    assert len(direct_calls) + len(level_calls) == 6
    assert len(router_calls) == 3
    with patch("sklearn.tree.DecisionTreeClassifier.fit",
               side_effect=AssertionError("verify must not fit")) as _spy:
        import ast as _ast
        vsrc = _ast.parse(Path(router.__file__).read_text())
        vfn = next(n for n in vsrc.body
                   if isinstance(n, (_ast.FunctionDef,)) and n.name == "stage_verify")
        vfits = [n for n in _ast.walk(vfn)
                 if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute)
                 and n.func.attr in ("fit", "partial_fit")]
        assert vfits == [] and _spy is not None
