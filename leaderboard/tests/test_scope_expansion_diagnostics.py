"""Synthetic CPU-only tests for scope-expansion contender diagnostics.

No real data, no fits, no network. Every case exercises the actual
diagnostics APIs against hand-computable synthetic frames.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PKG_PARENT = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(PKG_PARENT))
sys.path.insert(0, str(PKG_PARENT.parent))

from mse20_scope_expansion import diagnostics  # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _frame(origins, values, canonical=False):
    rows = []
    for origin, series in zip(origins, values):
        for h, pred in enumerate(series, start=1):
            rows.append((origin, h, float(pred)))
    frame = pd.DataFrame(rows, columns=["origin", "h", "y_pred"])
    if canonical:
        return frame.rename(columns={"origin": "origin_last_input_utc",
                                     "h": "horizon_hours",
                                     "y_pred": "pred_kms"})
    return frame


def _flat_truth(t0="2025-02-01", hours=24 * 12, speed=400.0, gaps=()):
    idx = pd.date_range(t0, periods=hours, freq="h", tz="UTC")
    target = pd.Series(np.full(len(idx), speed), index=idx)
    observed = pd.Series(True, index=idx)
    for start, stop in gaps:
        observed.loc[start:stop] = False
    return diagnostics.prepare_target(target, observed)


def _origins(t0="2025-02-03", n=4, freq="3h"):
    return pd.date_range(t0, periods=n, freq=freq, tz="UTC")


# ----------------------------------------------------------------------------
# 1. alias normalization + strict grid
# ----------------------------------------------------------------------------
def test_alias_and_canonical_normalize_identically():
    origins = _origins()
    rng = np.random.default_rng(3)
    values = [400 + rng.normal(0, 5, 72) for _ in origins]
    alias = _frame(origins, values)
    canon = _frame(origins, values, canonical=True)
    pd.testing.assert_frame_equal(diagnostics.canonicalize_predictions(alias),
                                  diagnostics.canonicalize_predictions(canon))


def test_strict_grid_rejects_missing_duplicate_nonfinite():
    origins = _origins(n=2)
    good = _frame(origins, [np.full(72, 400.0), np.full(72, 410.0)])
    diagnostics.canonicalize_predictions(good)
    with pytest.raises(ValueError):
        diagnostics.canonicalize_predictions(good.iloc[:-1])  # missing key
    with pytest.raises(ValueError):
        diagnostics.canonicalize_predictions(
            pd.concat([good, good.iloc[[0]]], ignore_index=True))  # duplicate
    bad = good.copy()
    bad.loc[0, "y_pred"] = np.nan
    with pytest.raises(ValueError):
        diagnostics.canonicalize_predictions(bad)


def test_exact_keys_between_contenders():
    origins = _origins(n=2)
    base = _frame(origins, [np.full(72, 400.0), np.full(72, 400.0)])
    shifted = base.copy()
    shifted.loc[0, "h"] = 72  # collides/duplicates -> key set differs
    with pytest.raises(ValueError):
        diagnostics.require_exact_keys(
            {"a": diagnostics.canonicalize_predictions(base),
             "b": diagnostics.canonicalize_predictions(shifted)})
    short = base.iloc[:-1]
    with pytest.raises(ValueError):
        diagnostics.compare(
            {"a": base, "b": short}, *_flat_truth())


# ----------------------------------------------------------------------------
# 2. observed-only scoring, bands, per-origin stats
# ----------------------------------------------------------------------------
def test_observed_only_total_and_bands():
    origins = _origins(n=2)
    filled, observed = _flat_truth()
    pred = _frame(origins, [np.full(72, 410.0), np.full(72, 400.0)])
    report = diagnostics.compare({"a": pred}, filled, observed)["contenders"]["a"]
    assert report["total"]["mse"] == pytest.approx(50.0)  # (100*72+0)/144
    assert report["total"]["bias_kms"] == pytest.approx(5.0)
    assert report["horizon_bands"]["1-6"]["mse"] == pytest.approx(50.0)
    assert report["horizon_bands"]["49-72"]["n_observed"] == 48
    per = report["per_origin"]
    assert per["origins"][0]["bias_kms"] == pytest.approx(10.0)
    assert per["origins"][0]["rmse_kms"] == pytest.approx(10.0)
    assert per["origins"][1]["bias_kms"] == pytest.approx(0.0)
    assert report["selection_claim"] is None


def test_unobserved_rows_excluded_from_mse():
    origins = _origins(n=1)
    filled, observed = _flat_truth(
        gaps=[(origins[0] + pd.Timedelta(hours=1), origins[0] + pd.Timedelta(hours=6))])
    pred = _frame(origins, [np.full(72, 500.0)])
    report = diagnostics.compare({"a": pred}, filled, observed)["contenders"]["a"]
    assert report["total"]["n_observed"] == 66
    assert report["total"]["mse"] == pytest.approx(100.0 ** 2)
    assert report["horizon_bands"]["1-6"]["mse"] is None


# ----------------------------------------------------------------------------
# 3. causal targets: ffill + origin speeds never use the future
# ----------------------------------------------------------------------------
def test_causal_ffill_and_leading_missing_rejected():
    idx = pd.date_range("2025-02-01", periods=48, freq="h", tz="UTC")
    target = pd.Series(400.0, index=idx)
    observed = pd.Series(True, index=idx)
    observed.iloc[10:14] = False
    filled, _ = diagnostics.prepare_target(target, observed)
    assert (filled.iloc[10:14] == 400.0).all()
    observed.iloc[0] = False
    with pytest.raises(ValueError):
        diagnostics.prepare_target(target, observed)


def test_origin_speed_ignores_future_truth():
    origins = _origins(n=2)
    filled, observed = _flat_truth(speed=400.0)
    before = diagnostics.origin_speeds(filled, origins)
    poisoned = filled.copy()
    poisoned.loc[origins[0] + pd.Timedelta(hours=1):
                 origins[1] - pd.Timedelta(hours=1)] += 250.0
    after = diagnostics.origin_speeds(poisoned, origins)
    pd.testing.assert_series_equal(before, after)
    late = filled.copy()
    late.loc[origins.max() + pd.Timedelta(hours=1):] += 250.0
    pd.testing.assert_series_equal(
        before, diagnostics.origin_speeds(late, origins))


def test_jump_buckets_relative_to_origin_speed():
    origins = _origins(n=1)
    idx = pd.date_range("2025-01-30", periods=24 * 10, freq="h", tz="UTC")
    target = pd.Series(400.0, index=idx)
    target.loc[origins[0] + pd.Timedelta(hours=1):
               origins[0] + pd.Timedelta(hours=72)] = 600.0  # jump +200
    filled, observed = diagnostics.prepare_target(
        target, pd.Series(True, index=idx))
    pred = _frame(origins, [np.full(72, 600.0)])
    buckets = diagnostics.compare(
        {"a": pred}, filled, observed)["contenders"]["a"]["jump_buckets"]
    assert buckets[">=150"]["n_observed"] == 72
    assert buckets[">=150"]["mse"] == pytest.approx(0.0)
    assert buckets["[-50,50)"]["n_observed"] == 0


def test_persistence_naive_skill_zero_when_equal():
    origins = pd.date_range("2025-02-07T18:00:00Z", periods=2, freq="3h", tz="UTC")
    filled, observed = _bumpy_truth()
    speeds = diagnostics.origin_speeds(filled, origins)
    pred = _frame(origins, [np.full(72, float(speeds.loc[o])) for o in origins])
    naive = diagnostics.compare(
        {"a": pred}, filled, observed)["contenders"]["a"]["persistence_1_6h"]
    assert naive["naive_mse"] > 0
    assert naive["model_mse"] == pytest.approx(naive["naive_mse"])
    assert naive["skill_vs_naive"] == pytest.approx(0.0)


# ----------------------------------------------------------------------------
# 4. version-1 events/HSS reuse + indeterminate + oracle flags
# ----------------------------------------------------------------------------
def _bumpy_truth():
    idx = pd.date_range("2025-02-01", periods=24 * 14, freq="h", tz="UTC")
    vals = np.full(len(idx), 380.0)
    center = len(idx) // 2
    width = np.arange(len(idx)) - center
    vals += 340.0 * np.exp(-0.5 * (width / 4.0) ** 2)  # peak ~720, interior
    return diagnostics.prepare_target(
        pd.Series(vals, index=idx), pd.Series(True, index=idx))


def test_unique_f1_matches_arrival_exact_definition():
    from experiments.mse20_hss_arrival import diagnostics as arrival
    assert arrival.definition()["version"] == 1
    filled, observed = _bumpy_truth()
    origins = pd.date_range("2025-02-05", periods=8, freq="3h", tz="UTC")
    stamps = [o + pd.Timedelta(hours=h) for o in origins for h in range(1, 73)]
    truth_vals = filled.reindex(pd.DatetimeIndex(stamps)).to_numpy(float)
    pred = _frame(origins, [truth_vals[i * 72:(i + 1) * 72] for i in range(len(origins))])
    canonical = diagnostics.canonicalize_predictions(pred)
    arrival_frame = pd.DataFrame({
        "origin_last_input_utc": pd.DatetimeIndex(canonical["origin"]),
        "horizon_hours": canonical["h"].to_numpy(int),
        "pred_kms": canonical["y_pred"].to_numpy(float)})
    expected = arrival.unique_event_metrics(arrival_frame, filled, observed, origins)
    got = diagnostics.unique_event_report(canonical, filled, observed, origins)
    assert got["f1"] == pytest.approx(expected["f1"])
    assert got["gate"] is None  # descriptive only; never a pass claim


def test_flat_series_is_indeterminate_not_pass():
    origins = _origins()
    filled, observed = _flat_truth()
    pred = _frame(origins, [np.full(72, 400.0) for _ in origins])
    unique = diagnostics.compare(
        {"a": pred}, filled, observed)["contenders"]["a"]["unique_events"]
    assert unique["f1"] is None
    assert unique["gate"] is None
    assert "indeterminate" in unique


def test_oracle_is_diagnostic_only_and_hss_present():
    origins = _origins()
    filled, observed = _flat_truth()
    pred = _frame(origins, [np.full(72, 400.0) for _ in origins])
    contender = diagnostics.compare(
        {"a": pred}, filled, observed)["contenders"]["a"]
    oracle = contender["oracle_shift"]
    assert oracle["diagnostic_only"] is True
    assert oracle["uses_truth"] is True
    assert oracle["not_for_selection_or_predictions"] is True
    assert "warning" in oracle
    assert contender["hss_onsets"]["by_lead_hours"].keys() == {"24", "48"}


# ----------------------------------------------------------------------------
# 5. CLI: new separate dir, append-only, frozen run untouched
# ----------------------------------------------------------------------------
def _embedded_frame(origins, pred_value):
    idx0 = (origins.min() - pd.Timedelta(days=2)).floor("h")
    idx1 = (origins.max() + pd.Timedelta(hours=73)).ceil("h")
    full = pd.date_range(idx0, idx1, freq="h", tz="UTC")
    rows = []
    for origin in origins:
        for h in range(1, 73):
            stamp = origin + pd.Timedelta(hours=h)
            rows.append((origin, h, float(pred_value), 400.0, True, stamp))
    return pd.DataFrame(rows, columns=["origin", "h", "y_pred", "y",
                                       "observed_mask", "target_t"])


def test_cli_writes_new_dir_and_refuses_overwrite(tmp_path):
    origins = _origins(n=2)
    ref_path = tmp_path / "ref.parquet"
    _embedded_frame(origins, 405.0).to_parquet(ref_path)
    alt_path = tmp_path / "alt.parquet"
    frame = _embedded_frame(origins, 400.0)
    frame.to_parquet(alt_path)
    out_dir = tmp_path / "diag-new"
    code = diagnostics.main(["--ref", str(ref_path),
                             "--alt", f"alt={alt_path}",
                             "--output-dir", str(out_dir)])
    assert code == 0
    reports = list(out_dir.glob("scope-diagnostics__*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["selection_claim"] is None
    assert "cannot affect predictions" in report["oracle_warning"].lower() \
        or "MUST NOT" in report["oracle_warning"]
    assert set(report["contenders"]) == {"reference", "alt"}
    with pytest.raises(FileExistsError):
        diagnostics.main(["--ref", str(ref_path),
                          "--alt", f"alt={alt_path}",
                          "--output-dir", str(out_dir)])
    with pytest.raises(ValueError):
        diagnostics.main(["--ref", str(ref_path),
                          "--output-dir", str(ref_path.parent)])
