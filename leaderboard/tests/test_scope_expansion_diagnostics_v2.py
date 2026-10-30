"""Synthetic CPU-only tests for scope-expansion diagnostics v2.

No real data, no fits, no network. Covers only the v2 additions over frozen v1:
forward-filled-72h oracle on gappy windows (causal), key/support preservation,
diagnostic-only markers, exact-72h level stats vs observed-only bias, leading-
missing fail-closed, and fresh-dir CLI behavior.
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

from mse20_scope_expansion import diagnostics as v1  # noqa: E402
from mse20_scope_expansion import diagnostics_v2 as v2  # noqa: E402


def _frame(origins, values):
    rows = []
    for origin, series in zip(origins, values):
        for h, pred in enumerate(series, start=1):
            rows.append((origin, h, float(pred)))
    return pd.DataFrame(rows, columns=["origin", "h", "y_pred"])


def _flat_truth(t0="2025-02-01", hours=24 * 12, speed=400.0, gaps=()):
    idx = pd.date_range(t0, periods=hours, freq="h", tz="UTC")
    target = pd.Series(np.full(len(idx), speed), index=idx)
    observed = pd.Series(True, index=idx)
    for start, stop in gaps:
        observed.loc[start:stop] = False
    return v1.prepare_target(target, observed)


def _origins(t0="2025-02-03", n=4, freq="3h"):
    return pd.date_range(t0, periods=n, freq=freq, tz="UTC")


def test_ffill_oracle_permits_gappy_windows_and_ignores_future():
    origins = _origins(n=2)
    gaps = [(origins[0] + pd.Timedelta(hours=1), origins[0] + pd.Timedelta(hours=6)),
            (origins[1] + pd.Timedelta(hours=10), origins[1] + pd.Timedelta(hours=20))]
    filled, observed = _flat_truth(gaps=gaps)
    pred = _frame(origins, [np.full(72, 400.0), np.full(72, 400.0)])
    old = v1.compare({"a": pred}, filled, observed)["contenders"]["a"]
    assert "indeterminate" in old["oracle_shift"]  # frozen v1 mismatch preserved
    rep = v2.compare_v2({"a": pred}, filled, observed)["contenders"]["a"]
    oracle = rep["oracle_shift_ffill72h"]
    assert "indeterminate" not in oracle
    assert oracle["method"] == "forwardfilled72h"
    assert oracle["n_ffill_origins"] == 2
    expected_missing = 0
    for origin in origins:
        stamps = pd.DatetimeIndex(
            [pd.Timestamp(origin) + pd.Timedelta(hours=h) for h in range(1, 73)],
            tz="UTC")
        expected_missing += int((~observed.reindex(stamps).to_numpy(dtype=bool)).sum())
    assert oracle["n_missing_observed_rows"] == expected_missing
    assert expected_missing > 0
    # Causal: perturbing truth strictly after both windows changes nothing.
    late = filled.copy()
    late.loc[origins[1] + pd.Timedelta(hours=73):] += 250.0
    before = v2.exact72_level_stats(
        v1.canonicalize_predictions(pred), filled, observed, origins)
    after = v2.exact72_level_stats(
        v1.canonicalize_predictions(pred), late, observed, origins)
    assert [r["filled_truth72_mean_kms"] for r in before["origins"]] == \
        [r["filled_truth72_mean_kms"] for r in after["origins"]]


def test_keys_preserved_and_observed_mse_unchanged():
    origins = _origins(n=2)
    filled, observed = _flat_truth()
    pred = _frame(origins, [np.full(72, 410.0), np.full(72, 400.0)])
    report = v2.compare_v2({"a": pred}, filled, observed)
    assert report["key_equality"] == {"n_keys": 144, "contenders": ["a"], "exact": True}
    old = v1.compare({"a": pred}, filled, observed)["contenders"]["a"]
    new = report["contenders"]["a"]
    assert new["total"] == old["total"]
    assert new["per_origin"] == old["per_origin"]
    assert new["selection_claim"] is None


def test_oracle_markers_and_observed_oracle_retained():
    origins = _origins()
    filled, observed = _flat_truth()
    pred = _frame(origins, [np.full(72, 400.0) for _ in origins])
    rep = v2.compare_v2({"a": pred}, filled, observed)["contenders"]["a"]
    oracle = rep["oracle_shift_ffill72h"]
    assert oracle["diagnostic_only"] is True
    assert oracle["uses_truth"] is True
    assert oracle["not_for_selection_or_predictions"] is True
    assert "warning" in oracle and oracle["causal"].startswith("forward-fill")
    assert rep["oracle_shift"] == rep["oracle_shift_observed_only"]
    assert rep["oracle_shift"]["n_oracle_origins"] == len(origins)
    assert rep["diagnostics_v2"] is True


def test_exact72_differs_from_observed_only_with_hand_values():
    origin = pd.date_range("2025-02-03", periods=1, freq="3h", tz="UTC")
    idx = pd.date_range("2025-02-01", periods=24 * 12, freq="h", tz="UTC")
    target = pd.Series(np.where(idx <= origin[0] + pd.Timedelta(hours=10),
                                400.0, 600.0), index=idx)
    observed = pd.Series(True, index=idx)
    observed.loc[origin[0] + pd.Timedelta(hours=11):
                 origin[0] + pd.Timedelta(hours=20)] = False  # filled stale 400
    filled, mask = v1.prepare_target(target, observed)
    pred = _frame(origin, [np.full(72, 600.0)])
    rep = v2.compare_v2({"a": pred}, filled, mask)["contenders"]["a"]
    assert rep["per_origin"]["origins"][0]["bias_kms"] == pytest.approx(2000.0 / 62)
    exact = rep["per_origin_exact72"]["origins"][0]
    assert exact["n_missing_observed"] == 10
    assert exact["bias72_kms"] == pytest.approx(4000.0 / 72)
    assert exact["rmse72_kms"] == pytest.approx(np.sqrt(20 * 200.0 ** 2 / 72))
    assert exact["bias72_kms"] != pytest.approx(rep["per_origin"]["origins"][0]["bias_kms"])


def test_leading_missing_without_seed_is_fail_closed():
    idx = pd.date_range("2025-02-01", periods=48, freq="h", tz="UTC")
    target = pd.Series(400.0, index=idx)
    observed = pd.Series(True, index=idx)
    observed.iloc[0] = False
    with pytest.raises(ValueError):
        v1.prepare_target(target, observed)
    origins = pd.date_range("2025-02-02", periods=1, freq="3h", tz="UTC")
    pred = _frame(origins, [np.full(72, 400.0)])
    with pytest.raises(ValueError):
        v2.compare_v2({"a": pred}, target, observed)


def _embedded_frame(origins, pred_value):
    rows = []
    for origin in origins:
        for h in range(1, 73):
            stamp = origin + pd.Timedelta(hours=h)
            rows.append((origin, h, float(pred_value), 400.0, True, stamp))
    return pd.DataFrame(rows, columns=["origin", "h", "y_pred", "y",
                                       "observed_mask", "target_t"])


def test_cli_writes_fresh_dir_and_refuses_overwrite(tmp_path):
    origins = _origins(n=2)
    ref_path = tmp_path / "ref.parquet"
    _embedded_frame(origins, 405.0).to_parquet(ref_path)
    alt_path = tmp_path / "alt.parquet"
    _embedded_frame(origins, 400.0).to_parquet(alt_path)
    out_dir = tmp_path / "diag-v2-new"
    assert v2.main(["--ref", str(ref_path), "--alt", f"alt={alt_path}",
                    "--output-dir", str(out_dir)]) == 0
    reports = list(out_dir.glob("scope-diagnostics-v2__*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["selection_claim"] is None
    contender = report["contenders"]["reference"]
    assert contender["oracle_shift_ffill72h"]["n_ffill_origins"] == 2
    assert contender["per_origin_exact72"]["n_origins"] == 2
    with pytest.raises(FileExistsError):
        v2.main(["--ref", str(ref_path), "--alt", f"alt={alt_path}",
                 "--output-dir", str(out_dir)])
    with pytest.raises(ValueError):
        v2.main(["--ref", str(ref_path), "--output-dir", str(ref_path.parent)])
