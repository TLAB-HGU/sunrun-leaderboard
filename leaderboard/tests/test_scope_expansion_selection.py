"""Synthetic CPU-only tests for preofficial recipe selection/freezing.

No real data, no fits, no registration, no claims, no network, no official
reads. Every case exercises the actual ``selection`` pure helpers (plus the
fail-closed CLI paths) with fully synthetic metric dicts and tmp dirs.
"""
import json
import sys
from pathlib import Path

import pytest

PKG_PARENT = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(PKG_PARENT))

from mse20_scope_expansion import protocol, selection  # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _period(mse, n_req=100, n_obs=80, digest="d", tp=1, fp=1, fn=1):
    return {"mse": mse, "n_requested": n_req, "n_observed": n_obs,
            "requested_digest": f"req-{digest}", "observed_digest": f"obs-{digest}",
            "tp": tp, "fp": fp, "fn": fn}


def _contenders(d_mses, m_mses, r_mses, digest="d",
                m_counts=(1, 1, 1), r_counts=(1, 1, 1), d_counts=(1, 1, 1)):
    def _mk(mses, counts, deps, config):
        return {"periods": {p: _period(m, digest=digest, tp=counts[0],
                                       fp=counts[1], fn=counts[2])
                            for p, m in zip(protocol.PERIOD_ORDER, mses)},
                "dependencies": deps,
                "scientific_config": config}
    return {
        "direct": _mk(d_mses, d_counts, 0, {"recipe": "direct"}),
        "mean": _mk(m_mses, m_counts, 1, {"recipe": "mean"}),
        "router": _mk(r_mses, r_counts, 2, {"recipe": "router"}),
    }


# ----------------------------------------------------------------------------
# 1. MSE gate boundaries and period counts (unrounded, no epsilon)
# ----------------------------------------------------------------------------

def test_gate_boundaries_from_spec_cases():
    assert selection.eligibility_from_relatives(
        {"dev": -.04, "selection": -.02, "audit": .00})["eligible"] is True
    bad_count = selection.eligibility_from_relatives(
        {"dev": -.06, "selection": .01, "audit": .01})
    assert bad_count["eligible"] is False  # only 1 improved period
    small_mean = selection.eligibility_from_relatives(
        {"dev": -.01, "selection": -.01, "audit": 0.0})
    # selection gate is mean<0 (unlike the source-admission mean<=-.02 gate),
    # so 2 improved periods with mean -0.0067 IS eligible here
    assert small_mean["eligible"] is True
    assert small_mean["mean_relative"] == pytest.approx(-0.02 / 3)


def test_exact_mean_minus_two_percent_passes_exact_zero_fails():
    assert selection.eligibility_from_relatives(
        {"dev": -.02, "selection": -.02, "audit": -.02})["eligible"] is True
    zero = selection.eligibility_from_relatives(
        {"dev": 0.0, "selection": 0.0, "audit": 0.0})
    assert zero["eligible"] is False  # exact zero is not an improvement


def test_no_epsilon_at_gate_tiny_strict_difference_counts():
    assert selection.eligibility_from_relatives(
        {"dev": -1e-12, "selection": -1e-12, "audit": -1e-12})["eligible"] is True
    assert selection.eligibility_from_relatives(
        {"dev": 1e-12, "selection": 1e-12, "audit": 1e-12})["eligible"] is False


def test_no_rounding_before_comparison():
    # rounding to 4dp would erase a -4e-5 mean; the gate must still see it
    gate = selection.eligibility_from_relatives(
        {"dev": -0.00006, "selection": -0.00006, "audit": 0.0})
    assert gate["eligible"] is True
    assert gate["mean_relative"] == pytest.approx(-0.00004)


# ----------------------------------------------------------------------------
# 2. pooled F1 vs per-period-average distinction
# ----------------------------------------------------------------------------

def test_pooled_f1_pools_counts_not_averages():
    # contender: F1 = 1.0, 0.0, 0.0 -> average 0.333; pooled 20/22 = 0.909
    got = selection.pooled_f1([10, 0, 0], [0, 1, 1], [0, 0, 0])
    assert got == pytest.approx(20 / 22)
    assert got != pytest.approx((1.0 + 0.0 + 0.0) / 3)


def test_selection_uses_pooled_f1_not_average():
    c = _contenders([100.0, 100.0, 100.0],
                    [90.0, 90.0, 105.0],   # MSE-eligible: 2 improved, mean -.05
                    [110.0, 110.0, 110.0],  # router ineligible
                    d_counts=(1, 1, 1), r_counts=(1, 1, 1))
    # per-period counts: mean F1 1.0/0.0/0.0 pattern vs direct 0.5 each
    c["mean"]["periods"]["dev"].update(tp=10, fp=0, fn=0)
    c["mean"]["periods"]["selection"].update(tp=0, fp=1, fn=0)
    c["mean"]["periods"]["audit"].update(tp=0, fp=1, fn=0)
    out = selection.evaluate_contenders(c)
    assert out["status"] == "valid" and out["winner"] == "mean"


def test_f1_regression_blocks_despite_mse_eligibility():
    c = _contenders([100.0, 100.0, 100.0],
                    [90.0, 90.0, 105.0], [110.0, 110.0, 110.0],
                    d_counts=(5, 0, 0),   # direct pooled F1 = 1.0
                    m_counts=(1, 1, 1), r_counts=(1, 1, 1))
    out = selection.evaluate_contenders(c)
    assert out["winner"] == "direct"  # mean blocked on F1, direct fallback wins


def test_undefined_f1_is_indeterminate_not_pass():
    c = _contenders([100.0, 100.0, 100.0],
                    [90.0, 90.0, 105.0], [110.0, 110.0, 110.0])
    for p in protocol.PERIOD_ORDER:
        c["mean"]["periods"][p].update(tp=0, fp=0, fn=0)  # denom 0
    out = selection.evaluate_contenders(c)
    assert out["status"] == "indeterminate" and out["code"] == 1


# ----------------------------------------------------------------------------
# 3. NaN / empty / non-finite inputs
# ----------------------------------------------------------------------------

def test_nan_mse_is_indeterminate():
    c = _contenders([100.0, 100.0, 100.0],
                    [90.0, float("nan"), 90.0], [110.0, 110.0, 110.0])
    out = selection.evaluate_contenders(c)
    assert out["status"] == "indeterminate" and out["code"] == 1


def test_zero_baseline_is_invalid():
    c = _contenders([0.0, 100.0, 100.0],
                    [90.0, 90.0, 90.0], [90.0, 90.0, 90.0])
    out = selection.evaluate_contenders(c)
    assert out["status"] == "invalid" and out["code"] == 2


def test_empty_observed_support_is_invalid():
    c = _contenders([100.0, 100.0, 100.0],
                    [90.0, 90.0, 90.0], [90.0, 90.0, 90.0])
    for p in protocol.PERIOD_ORDER:
        c["mean"]["periods"][p]["n_observed"] = 0
    out = selection.evaluate_contenders(c)
    assert out["status"] == "invalid" and out["code"] == 2


def test_observed_mse_reconstruction_and_empty_gate():
    got = selection.observed_mse([1.0, 2.0, 3.0], [1.0, 2.0, 5.0], [True, True, False])
    assert got == {"mse": pytest.approx(0.0), "n_observed": 2}
    with pytest.raises(ValueError, match="length mismatch"):
        selection.observed_mse([1.0], [1.0, 2.0], [True])
    with pytest.raises(ValueError, match="empty observed support"):
        selection.observed_mse([1.0], [2.0], [False])


# ----------------------------------------------------------------------------
# 4. exact keys / mask equality across contenders
# ----------------------------------------------------------------------------

def test_differing_requested_digest_is_invalid():
    c = _contenders([100.0, 100.0, 100.0],
                    [90.0, 90.0, 90.0], [90.0, 90.0, 90.0])
    c["mean"]["periods"]["dev"]["requested_digest"] = "req-other"
    out = selection.evaluate_contenders(c)
    assert out["status"] == "invalid" and out["code"] == 2


def test_differing_observed_counts_are_invalid():
    c = _contenders([100.0, 100.0, 100.0],
                    [90.0, 90.0, 90.0], [90.0, 90.0, 90.0])
    c["router"]["periods"]["audit"]["n_observed"] = 79
    out = selection.evaluate_contenders(c)
    assert out["status"] == "invalid" and out["code"] == 2


# ----------------------------------------------------------------------------
# 5. pending / extra mechanisms refuse the freeze path
# ----------------------------------------------------------------------------

def test_missing_router_refuses_freeze_set():
    c = _contenders([100.0, 100.0, 100.0],
                    [90.0, 90.0, 90.0], [90.0, 90.0, 90.0])
    del c["router"]
    out = selection.evaluate_contenders(c)
    assert out["status"] == "invalid" and "router" in out["reason"].lower()


def test_extra_recipe_variant_is_invalid():
    c = _contenders([100.0, 100.0, 100.0],
                    [90.0, 90.0, 90.0], [90.0, 90.0, 90.0])
    c["mean_variant"] = c["mean"]
    out = selection.evaluate_contenders(c)
    assert out["status"] == "invalid" and out["code"] == 2


def test_direct_fallback_wins_when_others_uneligible():
    c = _contenders([100.0, 100.0, 100.0],
                    [110.0, 110.0, 110.0], [101.0, 101.0, 101.0])
    out = selection.evaluate_contenders(c)
    assert out["status"] == "valid" and out["winner"] == "direct"


# ----------------------------------------------------------------------------
# 6. ranking order + deterministic ties
# ----------------------------------------------------------------------------

def test_ranking_prefers_mean_then_worst_period_then_dependencies():
    c = _contenders([100.0, 100.0, 100.0],
                    [90.0, 90.0, 96.0],   # mean rel -.08; worst -.04
                    [88.0, 88.0, 100.0])  # router mean rel -.08; worst 0.0
    out = selection.evaluate_contenders(c)
    assert out["winner"] == "mean"  # same mean, better worst period
    assert out["ranking"][0] == "mean"


def test_tie_is_deterministic_and_digest_ordered():
    c = _contenders([100.0, 100.0, 100.0],
                    [90.0, 90.0, 90.0], [90.0, 90.0, 90.0])
    c["mean"]["dependencies"] = c["router"]["dependencies"] = 1
    c["mean"]["scientific_config"] = {"recipe": "tied"}
    c["router"]["scientific_config"] = {"recipe": "tied"}
    first = selection.evaluate_contenders(c)["winner"]
    # same scientific config + same deps => digest tiebreak decides; rerun stable
    for _ in range(3):
        assert selection.evaluate_contenders(c)["winner"] == first


def test_config_digest_stable_and_recipe_scoped():
    d1 = selection.scientific_config_digest("mean", {"a": 1})
    d2 = selection.scientific_config_digest("mean", {"a": 1})
    assert d1 == d2
    assert selection.scientific_config_digest("router", {"a": 1}) != d1
    with pytest.raises(ValueError, match="unknown recipe"):
        selection.scientific_config_digest("mean_variant", {})


# ----------------------------------------------------------------------------
# 7. freeze / inspect fail-closed paths (no real run needed)
# ----------------------------------------------------------------------------

def test_inspect_empty_dir_is_invalid(tmp_path):
    code, report = selection.inspect_run(tmp_path, repo_root=str(tmp_path))
    assert code == 2 and report["status"] == "invalid"


def test_freeze_refuses_overwrite_without_inspection(tmp_path):
    dest = tmp_path / selection.SELECTION_FILENAME
    dest.write_text("{}")
    code, report = selection.freeze_selection(tmp_path, repo_root=str(tmp_path),
                                              out_path=dest)
    assert code == 2 and "overwrite" in report["reason"]


def test_freeze_refuses_forbidden_output_path(tmp_path):
    code, report = selection.freeze_selection(
        tmp_path, repo_root=str(tmp_path),
        out_path=tmp_path / "official_scores.json")
    assert code == 2 and "forbidden" in report["reason"].lower()


def test_selection_module_opens_no_claims_and_reads_no_official():
    src = Path(selection.__file__).read_text()
    assert "open_claim" not in src
    assert "ClaimLedger" not in src  # ledger is read read-only, never written
    assert "evaluate_final" not in src and "final_proof" not in src
    assert "sunrun-lb-store" in src  # official snapshot path is a forbidden read
    assert "bcf5d5417" in src  # pinned snapshot hash explicitly excluded
    assert "no_official_data" in src  # freeze receipt declares the exclusion
    assert selection.SELECTION_SOURCES[0].endswith("selection.py")
    assert selection.SELECTION_FILENAME == "selection_frozen.json"


# ----------------------------------------------------------------------------
# 8. live recompute: canonical router truth-separate + shared truth attach
# ----------------------------------------------------------------------------

def _synth_inputs(n_hours=200, seed=0):
    import numpy as np
    import pandas as pd
    idx = pd.date_range("2025-08-04", periods=n_hours, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    speed = 400 + 30 * np.sin(np.arange(n_hours) / 10.0) + rng.normal(0, 5, n_hours)
    observed = pd.Series(True, index=idx)
    observed.iloc[7::7] = False  # keep t=0 observed: causal ffill needs a seed
    target = pd.Series(np.where(observed.to_numpy(), speed, np.nan), index=idx)
    filled, obs = selection.prepare_hourly_truth(target, observed)
    return idx, filled, obs


def _synth_frames(filled, observed_h, origins, seed=0):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    frames = {}
    for name, noise, canonical in (("direct", 3.0, False),
                                   ("mean", 4.0, False),
                                   ("router", 2.0, True)):
        parts = []
        for o in origins:
            stamps = o + pd.to_timedelta(np.arange(1, 73), unit="h")
            truth = filled.reindex(stamps).to_numpy(float)
            mask = observed_h.reindex(stamps).to_numpy(bool)
            pred = truth + rng.normal(0, noise, 72)
            if canonical:  # router canonical: truth stays separate, never embedded
                parts.append(pd.DataFrame({
                    "origin_last_input_utc": [o] * 72,
                    "horizon_hours": list(range(1, 73)),
                    "pred_kms": pred}))
            else:
                parts.append(pd.DataFrame({
                    "origin": [o] * 72, "h": list(range(1, 73)),
                    "y_pred": pred,
                    "y": np.where(mask, truth, np.nan),
                    "observed_mask": mask}))
        frames[name] = pd.concat(parts, ignore_index=True)
    return frames


def _stub_event_fn(counts):
    def _fn(canonical, filled, observed, origins):
        return dict(counts)
    return _fn


def test_live_recompute_canonical_router_truth_separate():
    import pandas as pd
    idx, filled, obs = _synth_inputs()
    origins = [idx[5], idx[8]]
    frames = _synth_frames(filled, obs, origins)
    assert "y" not in frames["router"].columns
    assert "observed_mask" not in frames["router"].columns
    out = selection.recompute_period_metrics(
        frames, filled, obs, origins,
        event_fn=_stub_event_fn({"tp": 3, "fp": 1, "fn": 2, "f1": 0.6666666666666666}))
    assert set(out) == {"direct", "mean", "router"}
    for name in ("direct", "mean", "router"):
        assert out[name]["n_requested"] == 2 * 72
        assert out[name]["tp"] == 3 and out[name]["fp"] == 1 and out[name]["fn"] == 2
    # same raw keys exactly across contenders (not producer digests alone)
    assert out["mean"]["requested_digest_raw"] == out["direct"]["requested_digest_raw"]
    assert out["router"]["observed_digest_raw"] == out["direct"]["observed_digest_raw"]
    # MSE is observed-only: matches the pure helper on the underlying vectors
    direct = frames["direct"]
    expect = selection.observed_mse(direct["y"], direct["y_pred"],
                                    direct["observed_mask"])
    assert out["direct"]["mse"] == pytest.approx(expect["mse"])
    assert out["direct"]["n_observed"] == expect["n_observed"]
    # independently reconstructed keys cover h=1..72 uniquely per origin
    canon, kind = selection.normalize_predictions(frames["router"])
    assert kind == "router"
    assert sorted(canon["h"].unique()) == list(range(1, 73))


def test_forged_embedded_truth_rejected():
    idx, filled, obs = _synth_inputs()
    origins = [idx[5], idx[8]]
    frames = _synth_frames(filled, obs, origins)
    bad = frames["direct"].copy()
    sel = bad["observed_mask"].to_numpy(bool)
    first = sel.argmax()
    bad.loc[first, "y"] = bad.loc[first, "y"] + 50.0
    with pytest.raises(ValueError, match="altered embedded truth"):
        selection.verify_embedded_consistency(bad, filled, obs)
    frames["direct"] = bad
    with pytest.raises(ValueError, match="altered embedded truth"):
        selection.recompute_period_metrics(
            frames, filled, obs, origins,
            event_fn=_stub_event_fn({"tp": 1, "fp": 1, "fn": 1}))


def test_forged_embedded_mask_rejected():
    idx, filled, obs = _synth_inputs()
    origins = [idx[5], idx[8]]
    frames = _synth_frames(filled, obs, origins)
    bad = frames["mean"].copy()
    sel = bad["observed_mask"].to_numpy(bool)
    bad.loc[sel.argmax(), "observed_mask"] = False
    with pytest.raises(ValueError, match="altered embedded mask"):
        selection.verify_embedded_consistency(bad, filled, obs)


def test_router_with_embedded_truth_rejected():
    idx, filled, obs = _synth_inputs()
    origins = [idx[5], idx[8]]
    frames = _synth_frames(filled, obs, origins)
    bad = frames["router"].copy()
    bad["y"] = 400.0
    with pytest.raises(ValueError, match="must not embed"):
        selection.normalize_predictions(bad)


def test_forged_key_and_missing_horizon_rejected():
    idx, filled, obs = _synth_inputs()
    origins = [idx[5], idx[8]]
    frames = _synth_frames(filled, obs, origins)
    short = frames["router"][~((frames["router"]["horizon_hours"] == 72))].copy()
    with pytest.raises(ValueError, match="1\\.\\.72"):
        selection.normalize_predictions(short)
    # each frame individually complete, but contender keys differ
    other_origins = [idx[5], idx[11]]
    frames["mean"] = _synth_frames(filled, obs, other_origins)["mean"]
    with pytest.raises(ValueError, match="key mismatch"):
        selection.recompute_period_metrics(
            frames, filled, obs, origins,
            event_fn=_stub_event_fn({"tp": 1, "fp": 1, "fn": 1}))


def test_nan_predictions_rejected_and_nan_mse_indeterminate():
    import numpy as np
    idx, filled, obs = _synth_inputs()
    origins = [idx[5], idx[8]]
    frames = _synth_frames(filled, obs, origins)
    bad = frames["direct"].copy()
    bad.loc[0, "y_pred"] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        selection.normalize_predictions(bad)


def test_recomputed_counts_match_rejects_fake_counts():
    live = {"tp": 3, "fp": 1, "fn": 2}
    assert selection.require_recomputed_counts_match({}, live) is True
    assert selection.require_recomputed_counts_match(
        {"tp": None, "fp": None, "fn": None}, live) is True
    assert selection.require_recomputed_counts_match(dict(live), live) is True
    with pytest.raises(ValueError, match="fake tp counts rejected"):
        selection.require_recomputed_counts_match({"tp": 99, "fp": 1, "fn": 2}, live)


# ----------------------------------------------------------------------------
# 9. mean standalone exclusion + fixed ratio audit (2/3, never 3/3)
# ----------------------------------------------------------------------------

def test_mean_excluded_as_requested_when_it_would_win():
    c = _contenders([100.0, 100.0, 100.0],
                    [90.0, 90.0, 105.0], [110.0, 110.0, 110.0],
                    d_counts=(1, 1, 1), m_counts=(2, 0, 0), r_counts=(1, 1, 1))
    sel = selection.evaluate_contenders(c)
    assert sel["winner"] == "mean"  # pure ranking unchanged
    out = selection.apply_mean_standalone_exclusion(
        c, sel, {"ok": True, "sha256": "abc"})
    assert out["mean_standalone_excluded"] is True
    assert out["mean_exclusion_applied"] is True
    assert out["winner"] == "direct"  # router ineligible, direct fallback wins
    assert out["winner"] != "mean"
    # mean metrics stay computed and the head stays a valid router component
    assert out["contenders"]["mean"]["pooled_f1"] == pytest.approx(1.0)
    assert out["mean_component_valid_for_router"] is True
    assert out["mean_exclusion_binding"]["decision_ok"] is True


def test_mean_exclusion_binds_live_regression_without_decision():
    c = _contenders([100.0, 100.0, 100.0],
                    [90.0, 90.0, 105.0], [110.0, 110.0, 110.0],
                    d_counts=(5, 0, 0), m_counts=(1, 1, 1), r_counts=(1, 1, 1))
    sel = selection.evaluate_contenders(c)
    assert sel["winner"] == "direct"  # F1 regression already blocks mean
    out = selection.apply_mean_standalone_exclusion(
        c, sel, {"ok": False, "sha256": None})
    assert out["winner"] == "direct"
    assert out["mean_exclusion_binding"]["observed_f1_regression"] is True


def test_mean_exclusion_binding_missing_is_indeterminate():
    c = _contenders([100.0, 100.0, 100.0],
                    [90.0, 90.0, 105.0], [110.0, 110.0, 110.0],
                    d_counts=(1, 1, 1), m_counts=(2, 0, 0), r_counts=(1, 1, 1))
    sel = selection.evaluate_contenders(c)
    assert sel["winner"] == "mean"
    out = selection.apply_mean_standalone_exclusion(
        c, sel, {"ok": False, "sha256": None})
    assert out["status"] == "indeterminate" and out["code"] == 1
    assert out["winner"] is None


def test_mean_ratio_audit_positive_fixed_record():
    direct = {"dev": 8004.227981703036, "selection": 8421.68411239503,
              "audit": 4349.526038776262}
    mean = {"dev": 7809.651173107377, "selection": 8354.467458386755,
            "audit": 4413.48729080984}
    audit = selection.mean_mse_ratios(direct, mean)
    assert audit["n_improved"] == 2  # never 3/3
    assert audit["audit_regressed"] is True
    assert audit["relatives"]["dev"] == pytest.approx(-0.02430925368948)
    assert audit["relatives"]["selection"] == pytest.approx(-0.00798137915305)
    assert audit["relatives"]["audit"] == pytest.approx(0.01470533834339)


def test_undefined_source_proof_is_not_ok(tmp_path):
    binding = selection.load_mean_exclusion_binding(str(tmp_path))
    assert binding["present"] is False and binding["ok"] is False


def test_pending_router_is_indeterminate_not_a_pass():
    c = _contenders([100.0, 100.0, 100.0],
                    [90.0, 90.0, 105.0], [90.0, 90.0, 105.0])
    for p in protocol.PERIOD_ORDER:
        c["router"]["periods"][p].update(tp=None, fp=None, fn=None)
    out = selection.evaluate_contenders(c)
    assert out["status"] == "indeterminate" and out["code"] == 1


# ----------------------------------------------------------------------------
# 10. preofficial boundary, causal truth, observed-only MSE, provenance pins
# ----------------------------------------------------------------------------

def test_preofficial_boundary_rejects_official_targets():
    import pandas as pd
    ok = selection.assert_preofficial_targets(
        pd.DatetimeIndex(["2026-05-28T23:00:00Z", "2026-05-31T23:00:00Z"]))
    assert ok["target_max_utc"] == "2026-05-31T23:00:00+00:00"
    with pytest.raises(ValueError, match="beyond"):
        selection.assert_preofficial_targets(
            pd.DatetimeIndex(["2026-05-31T23:00:00Z", "2026-06-01T00:00:00Z"]))


def test_prepare_truth_is_causal_ffill_only():
    import numpy as np
    import pandas as pd
    idx = pd.date_range("2025-08-04", periods=48, freq="h", tz="UTC")
    target = pd.Series(np.arange(48, dtype=float), index=idx)
    observed = pd.Series(True, index=idx)
    observed.iloc[10] = False  # interior gap fills FORWARD from t=9, never backwards
    filled, obs = selection.prepare_hourly_truth(target, observed)
    assert filled.iloc[10] == pytest.approx(9.0)
    assert filled.iloc[11] == pytest.approx(11.0)
    observed.iloc[0] = False  # leading missing has no causal seed
    with pytest.raises(ValueError, match="causal seed"):
        selection.prepare_hourly_truth(target, observed)
    with pytest.raises(ValueError, match="hourly"):
        selection.prepare_hourly_truth(
            target.iloc[[0, 2, 3]], pd.Series(True, index=target.iloc[[0, 2, 3]].index))


def test_attach_truth_mse_is_observed_only():
    import numpy as np
    import pandas as pd
    idx, filled, obs = _synth_inputs()
    origins = [idx[5], idx[8]]
    frames = _synth_frames(filled, obs, origins)
    canon, _ = selection.normalize_predictions(frames["direct"])
    attached, stats = selection.attach_truth(canon, filled, obs)
    assert stats["n_requested"] == 2 * 72
    assert stats["n_observed"] < stats["n_requested"]
    wild = frames["direct"].copy()
    sel = wild["observed_mask"].to_numpy(bool)
    wild.loc[~sel, "y_pred"] = 1e9  # unobserved rows must not move observed-only MSE
    canon_w, _ = selection.normalize_predictions(wild[["origin", "h", "y_pred"]])
    attached_w, stats_w = selection.attach_truth(canon_w, filled, obs)
    assert stats_w["mse"] == pytest.approx(stats["mse"])
    assert "max_target_t" in stats and "observed_digest_raw" in stats


def test_no_old_receipt_writes_and_live_markers():
    src = Path(selection.__file__).read_text()
    assert src.count("write_text") == 1  # only the append-only freeze receipt
    assert "unique_events\"] =" not in src and "unique_events'] =" not in src
    for marker in ("load_inputs", "unique_event_report", "truth_observed",
                   "mean_standalone_excluded", "recompute_period_metrics",
                   "require_recomputed_counts_match", "pool_event_counts",
                   "prepare_hourly_truth", "TARGET_MAX_UTC",
                   "source_input_manifest", "MEAN_MODEL_FILENAME",
                   "ROUTER_PKL_TEMPLATE", "forward-fill only"):
        assert marker in src


def test_manifest_start_deadline_gates_unchanged():
    assert protocol.USER_START_UTC == "2026-10-03T06:26:29.494380Z"
    assert protocol.USER_DEADLINE_UTC is None
    assert selection.OFFICIAL_GATES["origins"] == 2833
    assert selection.OFFICIAL_GATES["horizons"] == 72
    assert selection.OFFICIAL_GATES["total_keys"] == 203976
    manifest = selection.source_manifest_from_inventory(
        [{"path": "leaderboard/store/ch-v1/ace.parquet", "sha256": "abc"}])
    assert manifest == {"leaderboard/store/ch-v1/ace.parquet": "abc"}
    with pytest.raises(ValueError, match="empty"):
        selection.source_manifest_from_inventory([])
    with pytest.raises(ValueError, match="forbidden"):
        selection.source_manifest_from_inventory(
            [{"path": "sunrun-lb-store/official.parquet", "sha256": "abc"}])


# ----------------------------------------------------------------------------
# 11. r3 integration: registered digests, per-period parity, full inspect
# ----------------------------------------------------------------------------


def test_registered_keys_normalize_timezone_variants():
    import pandas as pd

    base = pd.DatetimeIndex(pd.to_datetime(
        ["2025-08-04T00:00:00Z", "2025-08-04T03:00:00Z"], utc=True))
    canon_a = pd.DataFrame({"origin": base,
                            "h": [1, 1], "y_pred": [1.0, 2.0]})
    # Same physical instants expressed with space+offset and +09:00.
    alt = pd.DatetimeIndex(pd.to_datetime(
        ["2025-08-04 00:00:00+00:00", "2025-08-04 12:00:00+09:00"], utc=True))
    canon_b = pd.DataFrame({"origin": alt,
                            "h": [1, 1], "y_pred": [1.0, 2.0]})
    assert selection.registered_keys(canon_a) == selection.registered_keys(canon_b)
    assert selection.request_keys_raw(canon_a) == selection.request_keys_raw(canon_b)
    canon_c = canon_b.copy()
    canon_c.loc[1, "origin"] = canon_c.loc[1, "origin"] + pd.Timedelta(hours=1)
    assert selection.registered_keys(canon_a) != selection.registered_keys(canon_c)
    assert (selection.protocol.index_digest(selection.registered_keys(canon_a))
            != selection.protocol.index_digest(selection.registered_keys(canon_c)))


def _build_r3_run(tmp_path, monkeypatch=None):
    import numpy as np
    import pandas as pd

    from mse20_scope_expansion import diagnostics as _diag
    from mse20_scope_expansion import reference as _ref

    run = tmp_path / "run"
    repo = tmp_path / "repo"
    run.mkdir(parents=True, exist_ok=True)
    (run / "claims").mkdir(parents=True, exist_ok=True)
    # Hourly truth covering every eval origin + 72h (real fixed grids).
    origins_by_p = {p: protocol.eval_requested_origins(p) for p in protocol.PERIOD_ORDER}
    omin = min(o.min() for o in origins_by_p.values())
    omax = max(o.max() for o in origins_by_p.values()) + pd.Timedelta(hours=72)
    hidx = pd.date_range(omin.floor("h") - pd.Timedelta(hours=6),
                         omax.ceil("h") + pd.Timedelta(hours=6), freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    speed = 400 + 30 * np.sin(np.arange(len(hidx)) / 10.0) + rng.normal(0, 3, len(hidx))
    ace_filled = pd.Series(speed, index=hidx)
    keep = np.ones(len(hidx), dtype=bool)
    keep[7::7] = False
    keep[0] = True
    obs_idx = hidx[keep]
    truth_observed = pd.Series(ace_filled.loc[obs_idx].to_numpy(), index=pd.DatetimeIndex(obs_idx))
    filled, observed_h = selection.hourly_truth_from_inputs(ace_filled, truth_observed)
    for i, p in enumerate(protocol.PERIOD_ORDER):
        origins = origins_by_p[p]
        prng = np.random.default_rng(100 + i)
        d_parts, m_parts, r_parts = [], [], []
        for o in origins:
            stamps = o + pd.to_timedelta(np.arange(1, 73), unit="h")
            truth = filled.reindex(stamps).to_numpy(float)
            mask = observed_h.reindex(stamps).to_numpy(bool)
            y = np.where(mask, truth, np.nan)
            dpred = truth + prng.normal(0, 2.0, 72)
            mpred = truth + prng.normal(0, 6.0, 72)
            d_parts.append(pd.DataFrame({"origin": [o] * 72, "h": list(range(1, 73)),
                                         "y_pred": dpred, "y": y, "observed_mask": mask}))
            m_parts.append(pd.DataFrame({"origin": [o] * 72, "h": list(range(1, 73)),
                                         "y_pred": mpred, "y": y, "observed_mask": mask}))
            r_parts.append(pd.DataFrame({"origin_last_input_utc": [o] * 72,
                                         "horizon_hours": list(range(1, 73)),
                                         "pred_kms": dpred}))
        dframe = pd.concat(d_parts, ignore_index=True)
        mframe = pd.concat(m_parts, ignore_index=True)
        rframe = pd.concat(r_parts, ignore_index=True)
        dframe.to_parquet(run / f"predictions_existing_all_{p}.parquet", index=False)
        mframe.to_parquet(run / f"predictions_mean_level_{p}.parquet", index=False)
        rframe.to_parquet(run / f"predictions_router_{p}.parquet", index=False)
        (run / f"xgb_existing_all_{p}.ubj").write_bytes(b"direct-model-" + p.encode())
        mdir = run / "mean_level" / p
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "xgb_mean_level.ubj").write_bytes(b"mean-model-" + p.encode())
        (run / f"router_{p}.pkl").write_bytes(b"router-model-" + p.encode())
        # Receipt digests use the registered (astype(str)) form, matching production.
        dcanon, _ = selection.normalize_predictions(dframe)
        mcanon, _ = selection.normalize_predictions(mframe)
        rcanon, rkind = selection.normalize_predictions(rframe)
        assert rkind == "router"
        dreg = selection.registered_keys(dcanon)
        mreg = selection.registered_keys(mcanon)
        rreg = selection.registered_keys(rcanon)
        dmask = observed_h.reindex(pd.DatetimeIndex(
            dcanon["origin"] + pd.to_timedelta(dcanon["h"], unit="h"))).to_numpy(bool)
        mmask = observed_h.reindex(pd.DatetimeIndex(
            mcanon["origin"] + pd.to_timedelta(mcanon["h"], unit="h"))).to_numpy(bool)
        dreq = protocol.index_digest(dreg)
        dobs = protocol.index_digest([k for k, s in zip(dreg, dmask) if bool(s)])
        mreq = protocol.index_digest(mreg)
        mobs = protocol.index_digest([k for k, s in zip(mreg, mmask) if bool(s)])
        eval_pairs = [(s, int(h)) for s in list(origins.astype(str)) for h in range(1, 73)]
        (run / f"fit_{p}.json").write_text(json.dumps(
            {"n_requested": int(len(dcanon)), "n_observed": int(dmask.sum()),
             "requested_digest": dreq, "observed_digest": dobs,
             "predictions_sha256": protocol.sha256_file(run / f"predictions_existing_all_{p}.parquet"),
             "model_sha256": protocol.sha256_file(run / f"xgb_existing_all_{p}.ubj")}, indent=1, sort_keys=True))
        (run / f"mean_fit_{p}.json").write_text(json.dumps(
            {"n_requested": int(len(mcanon)), "n_observed": int(mmask.sum()),
             "requested_digest": mreq, "observed_digest": mobs,
             "predictions_sha256": protocol.sha256_file(run / f"predictions_mean_level_{p}.parquet"),
             "model_sha256": protocol.sha256_file(mdir / "xgb_mean_level.ubj")}, indent=1, sort_keys=True))
        (run / f"router_fit_{p}.json").write_text(json.dumps(
            {"eval_pairs_digest": protocol.index_digest(eval_pairs),
             "n_eval_origins": int(len(origins)),
             "predictions_sha256": protocol.sha256_file(run / f"predictions_router_{p}.parquet"),
             "model_sha256": protocol.sha256_file(run / f"router_{p}.pkl")}, indent=1, sort_keys=True))
    (run / "registration.json").write_text(json.dumps(
        {"user_start_utc": protocol.USER_START_UTC, "user_deadline_utc": None}, indent=1, sort_keys=True))
    (run / "tests_proof.json").write_text(json.dumps({"exit_code": 0}, indent=1, sort_keys=True))
    per_verify = {p: {"replay_parity_1e6": True, "original_parity_1e6": True, "model_hash_ok": True,
                      "pred_hash_ok": True, "requested_digest_ok": True,
                      "saved_full_grid_ok": True, "saved_keys_ok": True}
                  for p in protocol.PERIOD_ORDER}
    (run / "verify.json").write_text(json.dumps(
        {"ok": True, "causal_proof_valid": True, "periods": per_verify}, indent=1, sort_keys=True))
    (run / "mean_verify.json").write_text(json.dumps({"ok": False}, indent=1, sort_keys=True))
    per_mean = {p: {"replay_parity_1e6": True, "model_hash_ok": True, "pred_hash_ok": True,
                    "saved_full_grid_ok": True, "saved_keys_ok": True, "decomposition_ok": True}
                for p in protocol.PERIOD_ORDER}
    (run / "mean_verify_v1.json").write_text(json.dumps(
        {"ok": True, "prior_mean_verify_ok": False,
         "prior_mean_verify_sha256": protocol.sha256_file(run / "mean_verify.json"),
         "periods": per_mean}, indent=1, sort_keys=True))
    (run / "router_verify.json").write_text(json.dumps(
        {"ok": True, "periods": {p: {"ok": True} for p in protocol.PERIOD_ORDER}}, indent=1, sort_keys=True))
    (run / "claims" / "ledger.json").write_text(json.dumps({"claims": []}, indent=1, sort_keys=True))
    for dirname in ("source-pilot-noaa-20261005", "source-pilot-gong-20261005"):
        d = repo / "leaderboard" / "store" / "mse20-scope-expansion" / dirname
        d.mkdir(parents=True, exist_ok=True)
        (d / "root-decision-20261005.json").write_text(json.dumps(
            {"decision": "NON_ADMITTED_UNVERIFIED (synthetic)", "full_collection_allowed": False},
            indent=1, sort_keys=True))
    exc = repo / ".omx" / "context"
    exc.mkdir(parents=True, exist_ok=True)
    (exc / "scope-mean-standalone-exclusion-20261005.json").write_text(json.dumps(
        {"decision": "EXCLUDED_STANDALONE_MEAN_LEVEL", "direct_pooled_f1": 0.8,
         "mean_pooled_f1": 0.5, "periods": {p: {} for p in protocol.PERIOD_ORDER},
         "authority": "synthetic"}, indent=1, sort_keys=True))
    real_root = Path(protocol.__file__).parents[3]
    for rel in selection.SELECTION_SOURCES:
        dest = repo / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((real_root / rel).read_bytes())
    if monkeypatch is not None:
        monkeypatch.setattr(protocol, "validate_registration", lambda _run, _repo: {})
        monkeypatch.setattr(_ref, "load_inputs",
                            lambda: {"ace_filled": ace_filled, "truth_observed": truth_observed})
        monkeypatch.setattr(_ref, "input_inventory",
                            lambda: [{"path": "leaderboard/store/mse20-scope-expansion/synth.parquet",
                                      "sha256": "abc"}])
        monkeypatch.setattr(_diag, "definition", lambda: {"version": 1})
        monkeypatch.setattr(_diag, "unique_event_report",
                            lambda _canon, _fill, _obs, _orig: {
                                "metrics": {"tp": 2, "fp": 1, "fn": 1}, "indeterminate": None})
    return run, repo


def test_inspect_valid_freeze_once(tmp_path, monkeypatch):
    run, repo = _build_r3_run(tmp_path, monkeypatch)
    code, report = selection.inspect_run(str(run), str(repo))
    assert code == 0 and report["status"] == "valid"
    assert report["winner"] == "direct"
    assert set(report["contenders"]) == {"direct", "mean", "router"}
    assert report["checks"]["f1_live_recompute"] is True
    assert report["checks"]["router_runtime"] == "v1"
    for p in protocol.PERIOD_ORDER:
        for name in ("direct", "mean", "router"):
            assert isinstance(report["checks"]["recomputed_f1"][p][name]["mse"], float)
            assert isinstance(report["checks"]["mse_reconstructed"][p][name], (float, str))
    fcode, frozen = selection.freeze_selection(str(run), str(repo))
    assert fcode == 0 and (run / selection.SELECTION_FILENAME).is_file()
    assert frozen["winner"] == "direct" and frozen["router_runtime"] == "v1"
    scode, _ = selection.freeze_selection(str(run), str(repo))
    assert scode == 2


def test_inspect_pending_router_is_indeterminate(tmp_path, monkeypatch):
    run, repo = _build_r3_run(tmp_path, monkeypatch)
    p0 = protocol.PERIOD_ORDER[0]
    (run / f"predictions_router_{p0}.parquet").unlink()
    (run / f"router_fit_{p0}.json").unlink()
    (run / f"router_{p0}.pkl").unlink()
    ledger = json.loads((run / "claims" / "ledger.json").read_text())
    assert ledger == {"claims": []}
    code, report = selection.inspect_run(str(run), str(repo))
    assert code == 1 and report["status"] == "indeterminate"


def test_inspect_forged_router_key_fails(tmp_path, monkeypatch):
    import pandas as pd

    run, repo = _build_r3_run(tmp_path, monkeypatch)
    p0 = protocol.PERIOD_ORDER[0]
    frame = pd.read_parquet(run / f"predictions_router_{p0}.parquet")
    frame.loc[0, "origin_last_input_utc"] = frame.loc[0, "origin_last_input_utc"] + pd.Timedelta(hours=3)
    frame.to_parquet(run / f"predictions_router_{p0}.parquet", index=False)
    code, _ = selection.inspect_run(str(run), str(repo))
    assert code != 0


def test_inspect_forged_mask_and_hash_fail(tmp_path, monkeypatch):
    import pandas as pd

    run, repo = _build_r3_run(tmp_path, monkeypatch)
    p0 = protocol.PERIOD_ORDER[0]
    frame = pd.read_parquet(run / f"predictions_existing_all_{p0}.parquet")
    sel = frame["observed_mask"].to_numpy(bool)
    frame.loc[sel, "y"] = frame.loc[sel, "y"] + 50.0
    frame.to_parquet(run / f"predictions_existing_all_{p0}.parquet", index=False)
    code, _ = selection.inspect_run(str(run), str(repo))
    assert code != 0
    run2, repo2 = _build_r3_run(tmp_path / "second", monkeypatch)
    p1 = protocol.PERIOD_ORDER[1]
    rec = json.loads((run2 / f"fit_{p1}.json").read_text())
    rec["requested_digest"] = "forged-digest"
    (run2 / f"fit_{p1}.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    code2, _ = selection.inspect_run(str(run2), str(repo2))
    assert code2 != 0


# ----------------------------------------------------------------------------
# 12. prospective v3 native-proof overlay (synthetic; peer verifier unlanded)
# ----------------------------------------------------------------------------

_V3_PARTITIONS = ("2024Q4", "2025Q1", "2025Q2")
_V3_ARMS = ("direct", "level")


def _synth_oof_block(receipt_shas):
    block = {}
    for q in _V3_PARTITIONS:
        part = {}
        for arm in _V3_ARMS:
            sha = receipt_shas[(q, arm)]
            det = {"ok": True, "receipt_sha256": sha}
            for k in selection.V3_OOF_COMMON_OK_KEYS:
                det[k] = True
            if arm == "direct":
                det["legacy_purge_ok"] = True
            else:
                det["direct_model_hash_ok"] = True
                det["direct_binds_exact_direct_arm_ok"] = True
                det["legacy_purge_ok"] = True
                det["actual_purge_ok"] = True
                det["legacy_n_train_rows"] = 322681
                det["actual_n_train_rows"] = 5800
                det["legacy_train_label_max_utc"] = "2024-09-27T23:00:00+00:00"
                det["actual_train_label_max_utc"] = "2024-09-27T21:00:00+00:00"
                det["legacy_metadata_mismatch_declared"] = True
                det["legacy_receipt_sha256"] = sha
            part[arm] = det
        block[q] = part
    return block


def _synth_v3_periods():
    return {p: dict({"ok": True},
                    **{k: True for k in selection.V3_ROUTER_PERIOD_OK_KEYS})
            for p in protocol.PERIOD_ORDER}


def _synth_v3_oof_doc(ctx):
    return {
        "ok": True, "phase": "oof",
        "identity": selection.V3_IDENTITY,
        "driver": selection.ROUTER_V3_DRIVER,
        "driver_sha256": ctx["driver_sha256"],
        "v3_sources": dict(ctx["v3_sources"]),
        "v3_tests_inventory": dict(ctx["v3_tests_inventory"]),
        "v2_driver_sha256": ctx["v2_driver_sha256"],
        "v2_test_sha256": ctx["v2_test_sha256"],
        "v2_registration_sha256": ctx["v2_reg_sha"],
        "v2_tests_proof_sha256": ctx["v2_proof_sha"],
        "pins": {k: True for k in selection.V3_PIN_KEYS},
        "oof": _synth_oof_block(ctx["receipt_shas"]),
        "ledger_counts": {"total": selection.V3_OOF_LEDGER_TOTAL,
                          "by_category": dict(selection.V3_OOF_LEDGER_BY_CATEGORY)},
        "six_oof_claims_ok": True, "twelve_global_claims_ok": True,
        "causal_proof_valid": True,
    }


def _synth_v3_all_doc(ctx):
    doc = _synth_v3_oof_doc(ctx)
    doc["phase"] = "all"
    doc["periods"] = _synth_v3_periods()
    doc["ledger_counts"] = {"total": selection.V3_ALL_LEDGER_TOTAL,
                            "by_category": dict(selection.V3_ALL_LEDGER_BY_CATEGORY)}
    doc["three_router_claims_ok"] = True
    del doc["twelve_global_claims_ok"]
    return doc


def _add_v3_overlay(run, repo, v2_verify_state="failed", mutate=None,
                    drop_v1=True):
    if drop_v1 and (run / "router_verify.json").is_file():
        (run / "router_verify.json").unlink()
    real_root = Path(protocol.__file__).parents[3]
    for rel in (selection.ROUTER_V2_DRIVER, selection.ROUTER_V2_TEST,
                selection.ROUTER_V3_DRIVER, selection.ROUTER_V3_TEST):
        dest = repo / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((real_root / rel).read_bytes())
    live_v2 = {rel: protocol.sha256_file(repo / rel)
               for rel in (selection.ROUTER_V2_DRIVER, selection.ROUTER_V2_TEST)}
    (run / selection.ROUTER_V2_REGISTRATION_FILENAME).write_text(json.dumps(
        {"router_sources": live_v2}, indent=1, sort_keys=True))
    if v2_verify_state == "failed":
        (run / selection.ROUTER_V2_VERIFY_FILENAME).write_text(json.dumps(
            {"ok": False, "periods": {p: {"ok": False}
                                      for p in protocol.PERIOD_ORDER}},
            indent=1, sort_keys=True))
    elif v2_verify_state == "ok":
        (run / selection.ROUTER_V2_VERIFY_FILENAME).write_text(json.dumps(
            {"ok": True, "periods": {p: {"ok": True}
                                     for p in protocol.PERIOD_ORDER}},
            indent=1, sort_keys=True))
    elif v2_verify_state == "absent":
        fp = run / selection.ROUTER_V2_VERIFY_FILENAME
        if fp.is_file():
            fp.unlink()
    (run / "router_v2_tests_proof.json").write_text(
        json.dumps({"exit_code": 0, "note": "synthetic"}, indent=1, sort_keys=True))
    receipt_shas = {}
    for q in _V3_PARTITIONS:
        for arm in _V3_ARMS:
            fp = run / f"oof_fit_{q}_{arm}.json"
            fp.write_text(json.dumps(
                {"partition": q, "arm": arm, "synthetic": True},
                indent=1, sort_keys=True))
            receipt_shas[(q, arm)] = protocol.sha256_file(fp)
    claims = ([{"claim_id": f"ref{i}", "category": "reference_replay", "status": "ok"}
               for i in range(3)]
              + [{"claim_id": f"mean{i}", "category": "mean_level", "status": "ok"}
                 for i in range(3)]
              + [{"claim_id": f"oof{i}", "category": "oof_base", "status": "ok"}
                 for i in range(6)]
              + [{"claim_id": f"router{i}", "category": "router_eval", "status": "ok"}
                 for i in range(3)])
    (run / "claims" / "ledger.json").write_text(
        json.dumps({"claims": claims}, indent=1, sort_keys=True))
    v2_sha = protocol.sha256_file(run / selection.ROUTER_V2_REGISTRATION_FILENAME)
    ctx = {
        "v2_reg_sha": v2_sha,
        "v3_sources": {rel: protocol.sha256_file(repo / rel)
                       for rel in (selection.ROUTER_V3_DRIVER, selection.ROUTER_V3_TEST)},
        "driver_sha256": protocol.sha256_file(repo / selection.ROUTER_V3_DRIVER),
        "v3_tests_inventory": {
            "test_file": selection.ROUTER_V3_TEST,
            "n_tests": 11,
            "sha256": protocol.sha256_file(repo / selection.ROUTER_V3_TEST),
        },
        "v2_driver_sha256": protocol.sha256_file(repo / selection.ROUTER_V2_DRIVER),
        "v2_test_sha256": protocol.sha256_file(repo / selection.ROUTER_V2_TEST),
        "v2_proof_sha": protocol.sha256_file(run / "router_v2_tests_proof.json"),
        "receipt_shas": receipt_shas,
    }
    oof_doc = _synth_v3_oof_doc(ctx)
    all_doc = _synth_v3_all_doc(ctx)
    if mutate is not None:
        mutate(oof_doc, all_doc, run, repo)
    (run / selection.ROUTER_V3_OOF_FILENAME).write_text(
        json.dumps(oof_doc, indent=1, sort_keys=True))
    (run / selection.ROUTER_V3_VERIFY_FILENAME).write_text(
        json.dumps(all_doc, indent=1, sort_keys=True))
    return v2_sha


def test_v3_valid_with_failed_v2_passes_and_freeze_once(tmp_path, monkeypatch):
    run, repo = _build_r3_run(tmp_path, monkeypatch)
    _add_v3_overlay(run, repo, v2_verify_state="failed", drop_v1=True)
    code, report = selection.inspect_run(str(run), str(repo))
    assert code == 0 and report["status"] == "valid"
    assert report["checks"]["router_v2_verify_ok"] is False
    assert report["checks"]["router_v3_ok"] is True
    assert report["checks"]["router_v3_oof_ok"] is True
    assert report["checks"]["fitting_runtime"] == "v2"
    assert report["checks"]["verification_runtime"] == "v3"
    assert report["checks"]["router_runtime"] == "v2"
    assert (run / selection.ROUTER_V2_VERIFY_FILENAME).read_text()
    fcode, frozen = selection.freeze_selection(str(run), str(repo))
    assert fcode == 0 and (run / selection.SELECTION_FILENAME).is_file()
    assert frozen["fitting_runtime"] == "v2"
    assert frozen["verification_runtime"] == "v3"
    assert frozen["router_runtime"] == "v2"
    scode, _ = selection.freeze_selection(str(run), str(repo))
    assert scode == 2


def test_v3_missing_oof_rejects(tmp_path, monkeypatch):
    run, repo = _build_r3_run(tmp_path, monkeypatch)
    _add_v3_overlay(run, repo, v2_verify_state="failed", drop_v1=True)
    (run / selection.ROUTER_V3_OOF_FILENAME).unlink()
    code, _ = selection.inspect_run(str(run), str(repo))
    assert code != 0


def test_v3_present_but_invalid_never_falls_back(tmp_path, monkeypatch):
    run, repo = _build_r3_run(tmp_path, monkeypatch)

    def _break(oof_doc, all_doc, run_p, repo_p):
        all_doc["ok"] = False
    _add_v3_overlay(run, repo, v2_verify_state="ok", mutate=_break,
                    drop_v1=False)
    code, report = selection.inspect_run(str(run), str(repo))
    assert code != 0
    assert report["checks"]["router_v3_present"] is True
    assert report["checks"]["router_v3_ok"] is False


def test_v3_legacy_unproven_rejects(tmp_path, monkeypatch):
    run, repo = _build_r3_run(tmp_path, monkeypatch)

    def _break(oof_doc, all_doc, run_p, repo_p):
        del all_doc["oof"]["2024Q4"]["level"]["legacy_metadata_mismatch_declared"]
    _add_v3_overlay(run, repo, v2_verify_state="failed", mutate=_break,
                    drop_v1=True)
    code, _ = selection.inspect_run(str(run), str(repo))
    assert code != 0


def test_v3_source_drift_rejects(tmp_path, monkeypatch):
    run, repo = _build_r3_run(tmp_path, monkeypatch)
    _add_v3_overlay(run, repo, v2_verify_state="failed", drop_v1=True)
    reg = json.loads((run / selection.ROUTER_V2_REGISTRATION_FILENAME).read_text())
    reg["router_sources"][selection.ROUTER_V2_DRIVER] = "forged-sha"
    (run / selection.ROUTER_V2_REGISTRATION_FILENAME).write_text(
        json.dumps(reg, indent=1, sort_keys=True))
    code, _ = selection.inspect_run(str(run), str(repo))
    assert code != 0


def test_v3_wrong_phase_rejects(tmp_path, monkeypatch):
    run, repo = _build_r3_run(tmp_path, monkeypatch)

    def _break(oof_doc, all_doc, run_p, repo_p):
        all_doc["phase"] = "oof"
    _add_v3_overlay(run, repo, v2_verify_state="failed", mutate=_break,
                    drop_v1=True)
    code, _ = selection.inspect_run(str(run), str(repo))
    assert code != 0


def test_v3_missing_period_rejects(tmp_path, monkeypatch):
    run, repo = _build_r3_run(tmp_path, monkeypatch)

    def _break(oof_doc, all_doc, run_p, repo_p):
        all_doc["periods"].pop(protocol.PERIOD_ORDER[0])
    _add_v3_overlay(run, repo, v2_verify_state="failed", mutate=_break,
                    drop_v1=True)
    code, _ = selection.inspect_run(str(run), str(repo))
    assert code != 0


def test_v3_proof_validator_rejects_boolean_shortcut():
    assert selection.validate_router_v3_proof({"ok": True})["ok"] is False
    assert selection.validate_router_v3_proof(None)["ok"] is False
    assert selection.validate_router_v3_oof_proof({"ok": True})["ok"] is False
    assert selection.validate_router_v3_oof_proof(None)["ok"] is False


def _v3_ctx_for_pure(tmp_path):
    run, repo = _build_r3_run(tmp_path, None)
    _add_v3_overlay(run, repo)
    oof_doc = json.loads((run / selection.ROUTER_V3_OOF_FILENAME).read_text())
    all_doc = json.loads((run / selection.ROUTER_V3_VERIFY_FILENAME).read_text())
    return oof_doc, all_doc


def test_v3_pure_validators_accept_real_shape(tmp_path):
    oof_doc, all_doc = _v3_ctx_for_pure(tmp_path)
    assert selection.validate_router_v3_oof_proof(oof_doc)["ok"] is True
    res = selection.validate_router_v3_proof(all_doc)
    assert res["ok"] is True and res["oof_ok"] is True and res["periods_ok"] is True


def test_v3_each_actual_oof_field_tamper_rejects(tmp_path):
    oof_doc, all_doc = _v3_ctx_for_pure(tmp_path)
    import copy
    cases = []
    for arm_key in ("model_hash_ok", "predictions_hash_ok", "keys_ok",
                    "grid_rows_ok", "all72_finite_ok", "labels_ok", "masks_ok",
                    "cutoff_ok", "train_rows_ok", "train_feature_digest_ok",
                    "train_label_digest_ok", "replay_values_ok",
                    "replay_mse_parity_ok", "legacy_purge_ok"):
        cases.append(("2024Q4", "direct", arm_key))
    cases.append(("2024Q4", "direct", "receipt_sha256"))
    for arm_key in list(selection.V3_OOF_COMMON_OK_KEYS) + \
            list(selection.V3_LEVEL_EXTRA_OK_KEYS) + [
                "legacy_n_train_rows", "actual_n_train_rows",
                "legacy_train_label_max_utc", "actual_train_label_max_utc",
                "legacy_metadata_mismatch_declared", "legacy_receipt_sha256",
                "receipt_sha256"]:
        cases.append(("2025Q1", "level", arm_key))
    for q, arm, field in cases:
        bad = copy.deepcopy(all_doc)
        if field in ("legacy_n_train_rows", "actual_n_train_rows"):
            bad["oof"][q][arm][field] = "not-an-int"
        elif field == "legacy_receipt_sha256":
            bad["oof"][q][arm][field] = "tampered"
        elif field in ("legacy_train_label_max_utc", "actual_train_label_max_utc",
                       "receipt_sha256"):
            del bad["oof"][q][arm][field]
        else:
            bad["oof"][q][arm][field] = False
        assert selection.validate_router_v3_proof(bad)["ok"] is False, field
        bad_oof = copy.deepcopy(oof_doc)
        if field in ("legacy_n_train_rows", "actual_n_train_rows"):
            bad_oof["oof"][q][arm][field] = "not-an-int"
        elif field == "legacy_receipt_sha256":
            bad_oof["oof"][q][arm][field] = "tampered"
        elif field in ("legacy_train_label_max_utc", "actual_train_label_max_utc",
                       "receipt_sha256"):
            del bad_oof["oof"][q][arm][field]
        else:
            bad_oof["oof"][q][arm][field] = False
        assert selection.validate_router_v3_oof_proof(bad_oof)["ok"] is False, field


def test_v3_each_router_period_field_tamper_rejects(tmp_path):
    _, all_doc = _v3_ctx_for_pure(tmp_path)
    import copy
    for key in selection.V3_ROUTER_PERIOD_OK_KEYS:
        bad = copy.deepcopy(all_doc)
        bad["periods"][protocol.PERIOD_ORDER[0]][key] = False
        assert selection.validate_router_v3_proof(bad)["ok"] is False, key


def test_v3_identity_driver_pins_ledger_tamper_rejects(tmp_path):
    oof_doc, all_doc = _v3_ctx_for_pure(tmp_path)
    import copy
    for mutate in (
        lambda d: d.update({"identity": "router-verify-v2"}),
        lambda d: d.update({"driver": "leaderboard/experiments/mse20_scope_expansion/router_v2.py"}),
        lambda d: d["pins"].update({"frozen_ok": False}),
        lambda d: d["pins"].__delitem__("frozen_ok"),
        lambda d: d.pop("v2_registration_sha256"),
        lambda d: d.update({"ledger_counts": {"total": 999, "by_category": {}}}),
        lambda d: d.update({"six_oof_claims_ok": False}),
        lambda d: d.update({"causal_proof_valid": False}),
    ):
        bad = copy.deepcopy(all_doc)
        mutate(bad)
        assert selection.validate_router_v3_proof(bad)["ok"] is False
    for mutate in (
        lambda d: d.update({"identity": "router-verify-v2"}),
        lambda d: d.update({"twelve_global_claims_ok": False}),
        lambda d: d.update({"ledger_counts": {"total": 15,
                                               "by_category": dict(selection.V3_ALL_LEDGER_BY_CATEGORY)}}),
    ):
        bad = copy.deepcopy(oof_doc)
        mutate(bad)
        assert selection.validate_router_v3_oof_proof(bad)["ok"] is False


def test_v3_oof_uses_nested_partitions_not_flat(tmp_path):
    oof_doc, _ = _v3_ctx_for_pure(tmp_path)
    import copy
    bad = copy.deepcopy(oof_doc)
    bad["oof"] = {f"{q}_{a}": {"ok": True}
                  for q in _V3_PARTITIONS for a in _V3_ARMS}
    assert selection.validate_router_v3_oof_proof(bad)["ok"] is False


def test_v3_live_receipt_tamper_rejects(tmp_path, monkeypatch):
    run, repo = _build_r3_run(tmp_path, monkeypatch)
    _add_v3_overlay(run, repo, v2_verify_state="failed", drop_v1=True)
    fp = run / "oof_fit_2024Q4_level.json"
    fp.write_text(json.dumps({"partition": "2024Q4", "arm": "level",
                              "tampered": True}, indent=1, sort_keys=True))
    code, report = selection.inspect_run(str(run), str(repo))
    assert code != 0
    assert report["checks"]["router_v3_receipts_live_ok"] is False


def test_v3_live_v3_source_drift_rejects(tmp_path, monkeypatch):
    run, repo = _build_r3_run(tmp_path, monkeypatch)
    _add_v3_overlay(run, repo, v2_verify_state="failed", drop_v1=True)
    with open(repo / selection.ROUTER_V3_DRIVER, "ab") as fh:
        fh.write(b"\n# drift\n")
    code, report = selection.inspect_run(str(run), str(repo))
    assert code != 0
    assert report["checks"]["router_v3_sources_live_ok"] is False


def test_v3_live_tests_inventory_drift_rejects(tmp_path, monkeypatch):
    run, repo = _build_r3_run(tmp_path, monkeypatch)
    _add_v3_overlay(run, repo, v2_verify_state="failed", drop_v1=True)
    with open(repo / selection.ROUTER_V3_TEST, "ab") as fh:
        fh.write(b"\n# drift\n")
    code, report = selection.inspect_run(str(run), str(repo))
    assert code != 0
    assert report["checks"]["router_v3_tests_live_ok"] is False


def test_v3_oof_all_disagree_rejects(tmp_path, monkeypatch):
    run, repo = _build_r3_run(tmp_path, monkeypatch)
    _add_v3_overlay(run, repo, v2_verify_state="failed", drop_v1=True)

    def _rewrite(path, fn):
        doc = json.loads(path.read_text())
        fn(doc)
        path.write_text(json.dumps(doc, indent=1, sort_keys=True))
    _rewrite(run / selection.ROUTER_V3_VERIFY_FILENAME,
             lambda d: d.update({"v2_registration_sha256": "forged-sha"}))
    code, report = selection.inspect_run(str(run), str(repo))
    assert code != 0
    assert report["checks"]["router_v3_docs_agree_v2_reg"] is False


def test_v3_live_ledger_mismatch_rejects(tmp_path, monkeypatch):
    run, repo = _build_r3_run(tmp_path, monkeypatch)
    _add_v3_overlay(run, repo, v2_verify_state="failed", drop_v1=True)
    (run / "claims" / "ledger.json").write_text(
        json.dumps({"claims": []}, indent=1, sort_keys=True))
    code, report = selection.inspect_run(str(run), str(repo))
    assert code != 0
    assert report["checks"]["router_v3_ledger_live_ok"] is False


def test_v3_synth_matches_real_oof_structure():
    real = (Path(protocol.__file__).parents[3]
            / "leaderboard/store/mse20-scope-expansion/run-20261005"
            / "native-reference-v2/router_oof_verify_v3.json")
    if not real.is_file():
        return
    doc = json.loads(real.read_text())
    assert doc.get("identity") == selection.V3_IDENTITY
    assert doc.get("driver") == selection.ROUTER_V3_DRIVER
    assert set(doc["oof"].keys()) == set(selection.V3_OOF_PARTITIONS)
    for q in selection.V3_OOF_PARTITIONS:
        assert set(doc["oof"][q].keys()) == set(selection.V3_OOF_ARMS)
        lvl = doc["oof"][q]["level"]
        assert lvl["legacy_metadata_mismatch_declared"] is True
        assert lvl["legacy_receipt_sha256"] == lvl["receipt_sha256"]
        assert lvl["actual_n_train_rows"] != lvl["legacy_n_train_rows"]
    assert doc["ledger_counts"]["total"] == selection.V3_OOF_LEDGER_TOTAL
    assert doc["ledger_counts"]["by_category"] == selection.V3_OOF_LEDGER_BY_CATEGORY
