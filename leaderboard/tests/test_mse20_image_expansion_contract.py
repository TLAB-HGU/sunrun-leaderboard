"""Targeted adversarial tests for mse20_image_expansion guards.

Pure/CPU-only: no fits, no data loading, no training, no scoring, no network,
no official reads. Every case uses synthetic numerics against the actual
protocol/geometry/evaluate helpers.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

PKG_PARENT = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(PKG_PARENT))

from mse20_image_expansion import evaluate, geometry, protocol  # noqa: E402
from mse20_scope_expansion import final_contract as fc  # noqa: E402
from mse20_scope_expansion import protocol as scope  # noqa: E402
from mse20_scope_expansion import selection  # noqa: E402


def _cand(fid, obs, avail):
    return {"file_id": fid, "obs_end_utc": obs, "avail_utc": avail}


# --- inherited config: read-only imports, never duplicated ------------------

def test_inherited_periods_params_horizons_match_scope():
    assert protocol.PERIODS is scope.PERIODS
    assert protocol.PARAMS is scope.PARAMS
    assert protocol.HORIZONS is scope.HORIZONS
    assert tuple(protocol.HORIZONS) == tuple(range(1, 73))
    assert protocol.PLAN_SHA256 == "dd57b8273cc43a6367be03293b04272b18f4475121233df56fff7b8403eb8aff"


def test_image_cap_is_own_eleven_not_old_budget():
    assert protocol.IMAGE_FIT_CAP == 11
    assert sum(protocol.IMAGE_FIT_BREAKDOWN.values()) == 11
    assert "IMAGE_FIT_CAP" not in dir(scope)


def test_f1_is_same_unique_event_function():
    assert protocol.pooled_f1 is selection.pooled_f1
    # pooled UNIQUE counts, never the mean of per-period ratios
    per = {"dev": (2, 0, 1), "selection": (0, 2, 1), "audit": (1, 1, 1)}
    tps = [per[p][0] for p in ("dev", "selection", "audit")]
    fps = [per[p][1] for p in ("dev", "selection", "audit")]
    fns = [per[p][2] for p in ("dev", "selection", "audit")]
    pooled = protocol.pooled_f1(tps, fps, fns)
    assert pooled == pytest.approx(2 * 3 / (2 * 3 + 3 + 3))
    assert pooled is not None


# --- strict/inclusive gates --------------------------------------------------

def test_cell_gate_strict_3800():
    bad = {k: 100.0 for k in fc.CELL_NAMES}
    bad["high_high_high"] = 3800.0
    assert protocol.check_cells_gate(bad)["ok"] is False
    good = {k: 3799.999 for k in fc.CELL_NAMES}
    assert protocol.check_cells_gate(good)["ok"] is True


def test_total_gate_inclusive_boundary():
    import math
    assert protocol.check_total_gate(3289.0890348547073)["ok"] is True
    assert protocol.check_total_gate(math.nextafter(3289.0890348547073, math.inf))["ok"] is False


def test_total_cells_f1_finite_undefined_fail_closed():
    for bad in (None, float("nan"), float("inf"), -1.0, "100"):
        assert protocol.check_total_gate(bad)["ok"] is False
    assert protocol.check_cells_gate({})["ok"] is False
    assert protocol.check_f1_gate(None, fc.CHAMPION_F1)["ok"] is False
    assert protocol.check_f1_gate(0.1, None)["ok"] is False


def test_a_gate_mean_minus002_inclusive():
    rel = {"dev": -0.02, "selection": -0.02, "audit": -0.02}
    out = protocol.a_eligibility(rel, 0.10, fc.CHAMPION_F1)
    assert out["eligible"] is True and out["mean_relative"] == pytest.approx(-0.02)
    rel2 = {"dev": -0.03, "selection": -0.03, "audit": 0.0001}
    out2 = protocol.a_eligibility(rel2, 0.10, fc.CHAMPION_F1)
    assert out2["eligible"] is False  # mean > -0.02
    one_period = {"dev": -0.5, "selection": 0.01, "audit": 0.01}
    assert protocol.a_eligibility(one_period, 0.10, fc.CHAMPION_F1)["eligible"] is False


def test_a_gate_undefined_fails_closed():
    rel = {"dev": -0.5, "selection": -0.5, "audit": None}
    out = protocol.a_eligibility(rel, 0.10, fc.CHAMPION_F1)
    assert out["eligible"] is False and "undefined" in out["reason"]
    rel2 = {"dev": -0.5, "selection": -0.5, "audit": -0.5}
    out2 = protocol.a_eligibility(rel2, None, fc.CHAMPION_F1)
    assert out2["eligible"] is False
    out3 = protocol.a_eligibility(rel2, 0.05, fc.CHAMPION_F1)  # F1 regression
    assert out3["eligible"] is False


def test_b_never_runs_unless_a_positive():
    rel_d = {"dev": -0.1, "selection": -0.1, "audit": -0.1}
    rel_a = {"dev": -0.05, "selection": -0.05, "audit": -0.05}
    out = protocol.b_eligibility(rel_d, rel_a, 0.10, 0.09, a_positive=False)
    assert out["eligible"] is False and "NEVER" in out["reason"]
    out2 = protocol.b_eligibility(rel_d, rel_a, 0.10, 0.09, a_positive=True)
    assert out2["eligible"] is True
    rel_a_weak = {"dev": -0.01, "selection": 0.02, "audit": 0.03}
    out3 = protocol.b_eligibility(rel_d, rel_a_weak, 0.10, 0.09, a_positive=True)
    assert out3["eligible"] is False


def test_b_rejects_champion_level_f1_below_direct():
    # Old bug passed F1=0.2 vs CHAMPION 0.0833; plan requires vs direct 0.4.
    rel_d = {"dev": -0.1, "selection": -0.1, "audit": -0.1}
    rel_a = {"dev": -0.05, "selection": -0.05, "audit": -0.05}
    out = protocol.b_eligibility(rel_d, rel_a, 0.20, 0.40, a_positive=True)
    assert out["eligible"] is False


def test_b_passes_below_a_without_f1_ge_a_gate():
    # Candidate 0.5 >= direct 0.4 but below A 0.6: plan imposes direct
    # non-regression only, so relative gates decide -> PASS.
    rel_d = {"dev": -0.1, "selection": -0.1, "audit": -0.1}
    rel_a = {"dev": -0.05, "selection": -0.05, "audit": -0.05}
    out = protocol.b_eligibility(rel_d, rel_a, 0.50, 0.40, a_positive=True)
    assert out["eligible"] is True


def test_b_undefined_direct_f1_rejects_despite_a_positive():
    rel_d = {"dev": -0.1, "selection": -0.1, "audit": -0.1}
    rel_a = {"dev": -0.05, "selection": -0.05, "audit": -0.05}
    out = protocol.b_eligibility(rel_d, rel_a, 0.50, None, a_positive=True)
    assert out["eligible"] is False
    rel_d_bad = {"dev": -0.1, "selection": -0.1, "audit": None}
    out2 = protocol.b_eligibility(rel_d_bad, rel_a, 0.50, 0.40, a_positive=True)
    assert out2["eligible"] is False
    rel_a_bad = {"dev": -0.05, "selection": None, "audit": -0.05}
    out3 = protocol.b_eligibility(rel_d, rel_a_bad, 0.50, 0.40, a_positive=True)
    assert out3["eligible"] is False


# --- image schema / causality -------------------------------------------------

def test_image_schema_22_exact_names_order():
    assert len(protocol.IMAGE_FEATURE_SCHEMA) == 22
    for v in protocol.SPEEDS:
        for s in ("mean_logI", "ch_frac", "coverage", "avail"):
            assert f"IMG_v{v}_{s}" in protocol.IMAGE_FEATURE_SCHEMA
    assert protocol.IMAGE_FEATURE_SCHEMA[-2:] == ("img_age_h", "img_stale")


def test_newest_eligible_requires_both_timestamps():
    o = "2026-01-01T00:00:00Z"
    cands = [_cand("a", "2025-12-31T20:00:00Z", "2026-01-01T01:00:00Z"),  # avail late
             _cand("b", "2026-01-01T01:00:00Z", "2025-12-31T20:00:00Z"),  # obs future
             _cand("c", "2025-12-31T20:00:00Z", "2025-12-31T21:00:00Z")]
    rec = protocol.select_newest_eligible(cands, o)
    assert rec is not None and rec["file_id"] == "c"
    assert protocol.select_newest_eligible(cands[:2], o) is None


def test_shifted_timestamp_flips_status_positive_control():
    o = "2026-01-01T00:00:00Z"
    ok = [_cand("a", "2025-12-31T22:00:00Z", "2025-12-31T23:00:00Z")]
    assert protocol.select_image_for_origin(ok, o)["status"] == "ok"
    poisoned = [_cand("a", "2026-01-01T01:00:00Z", "2025-12-31T23:00:00Z")]
    assert protocol.select_image_for_origin(poisoned, o)["status"] == "missing"
    with pytest.raises(ValueError):
        protocol.assert_feature_asof("2026-01-01T01:00:00Z", "2025-12-31T23:00:00Z", o)


def test_stale_boundary_6h():
    o = "2026-01-01T06:00:00Z"
    fresh = [_cand("a", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")]
    assert protocol.select_image_for_origin(fresh, o)["status"] == "ok"  # ==6h fresh
    stale = [_cand("a", "2025-12-31T23:59:00Z", "2025-12-31T23:59:00Z")]
    assert protocol.select_image_for_origin(stale, o)["status"] == "stale"


def test_tie_deterministic_identity():
    o = "2026-01-01T00:00:00Z"
    cands = [_cand("b", "2025-12-31T20:00:00Z", "2025-12-31T21:00:00Z"),
             _cand("a", "2025-12-31T20:00:00Z", "2025-12-31T21:00:00Z")]
    r1 = protocol.select_newest_eligible(cands, o)
    r2 = protocol.select_newest_eligible(list(reversed(cands)), o)
    assert r1 == r2 and r1["file_id"] == "a"


def test_malformed_timestamp_fails_closed_not_substituted():
    with pytest.raises(ValueError):
        protocol.select_newest_eligible(
            [_cand("a", "not-a-time", "2025-12-31T21:00:00Z")], "2026-01-01T00:00:00Z")
    with pytest.raises(ValueError):
        protocol.parse_utc("2026-01-01T00:00:00")  # naive rejected, no mtime use


def test_parse_utc_fractional_real_header():
    # Real SUVI obs_end carries fractional seconds (e.g. 00:59:11.7Z).
    dt = protocol.parse_utc("2025-12-31T00:59:11.7Z")
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second) == (2025, 12, 31, 0, 59, 11)
    assert dt.microsecond == 700000 and dt.tzinfo is not None
    assert protocol.parse_utc("2026-01-01T00:00:00Z").microsecond == 0
    # Fractional header participates in causality/stale exactly like whole-second.
    o = "2025-12-31T01:00:00Z"
    ok = [_cand("a", "2025-12-31T00:59:11.7Z", "2025-12-31T00:59:20.25Z")]
    assert protocol.select_image_for_origin(ok, o)["status"] == "ok"
    with pytest.raises(ValueError):
        protocol.parse_utc("2026-01-01T00:00:00+09:00")  # non-UTC rejected
    with pytest.raises(ValueError):
        protocol.parse_utc("2026-01-01 00:00:00Z")  # malformed rejected


# --- geometry -----------------------------------------------------------------

def test_horizons_grid_and_h0_rejected():
    assert tuple(protocol.HORIZONS) == tuple(range(1, 73))
    with pytest.raises(ValueError):
        geometry.strip_center_lambda(0, 350, 1.0)
    with pytest.raises(ValueError):
        geometry.strip_center_lambda(73, 350, 1.0)


def test_h72_vs_h1_shift_minus3905_and_older_image_slope():
    assert geometry.shift_h72_vs_h1() == pytest.approx(-39.05)
    for v in protocol.SPEEDS:
        d = geometry.strip_center_lambda(72, v, 2.0) - geometry.strip_center_lambda(1, v, 2.0)
        assert d == pytest.approx(-39.05)
    assert geometry.shift_for_older_image(10.0) == pytest.approx(-5.5)
    for v in protocol.SPEEDS:
        d = geometry.strip_center_lambda(10, v, 12.0) - geometry.strip_center_lambda(10, v, 2.0)
        assert d == pytest.approx(-0.55 * 10.0)


def test_tau_monotonic_and_parker_control():
    geometry.assert_tau_monotonic()
    taus = geometry.tau_table()
    assert set(taus) == set(protocol.SPEEDS)
    for v in protocol.SPEEDS:
        assert geometry.parker_control_west(v) == pytest.approx(0.55 * taus[v])


def test_longitude_sign_prograde():
    # prograde hole: later release target maps WEST-positive of CM at image time
    west = geometry.strip_center_lambda(1, 350, 1.0)
    assert west > 0


# --- CNN frozen spec ------------------------------------------------------------

def test_cnn_shape_and_param_bound():
    assert protocol.CNN_SPEC["input_shape"] == (2, 64, 128)
    assert protocol.CNN_SPEC["output_horizons"] == 72
    assert protocol.CNN_SPEC["flatten_dim"] + protocol.CNN_SPEC["condition_dim"] == 1096
    assert protocol.cnn_param_count() == 83928
    assert protocol.cnn_param_count() <= protocol.CNN_SPEC["param_max"] <= 200000


def test_cnn_spec_immutable():
    with pytest.raises(TypeError):
        protocol.CNN_SPEC["lr"] = 0.01


def test_stale_absent_residual_zero_origin_kept():
    z = protocol.zero_residual()
    assert len(z) == 72 and all(v == 0.0 for v in z)
    assert protocol.select_image_for_origin([], "2026-01-01T00:00:00Z")["status"] == "missing"


# --- evaluator fails closed ----------------------------------------------------

def test_evaluator_exit2_absent_evidence_appends(tmp_path):
    run = tmp_path / "mse20-image-expansion" / "run-20261005"
    r1 = evaluate.evaluate_run(str(run))
    assert r1["exit"] == 2 and r1["verdict"] == "NOT_PRODUCTION_READY"
    assert "missing-required-verifier" in r1["reason"]
    p = evaluate.append_evidence(str(run), r1)
    p2 = evaluate.append_evidence(str(run), evaluate.evaluate_run(str(run)))
    assert p == p2
    lines = p.read_text().strip().split("\n")
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["plan_sha256"] == protocol.PLAN_SHA256 and rec["exit"] == 2


def test_evaluator_cli_exit2_never_zero(tmp_path):
    run = tmp_path / "mse20-image-expansion" / "run-20261005"
    cmd = [sys.executable, "-m", "mse20_image_expansion.evaluate",
           "--run-dir", str(run)]
    env = {"PYTHONPATH": str(PKG_PARENT), "PATH": "/usr/bin:/bin"}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 2
    assert "NOT_PRODUCTION_READY" in r.stdout
