"""Synthetic CPU-only tests for the 72h mean-level head.

No real data, no scientific fits, no network, no uploads. Every case
exercises the actual mean_level APIs. Persistence uses a fake xgboost
adapter in sys.modules (repo-established stub pattern); persisted metadata
from the fake is marked UNVERIFIED and real native replay stays UNVERIFIED.
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

from mse20_scope_expansion import mean_level, protocol  # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _synthetic_inputs(t0="2024-12-01", hours=60 * 24, n_origins=120,
                      origin0="2025-01-10", gap_hours=()):
    # History starts 40d before the first origin so the fixed 26/27/28-day
    # trajectory lookups are fully covered (causal and finite).
    idx = pd.date_range(t0, periods=hours, freq="h", tz="UTC")
    rng = np.random.default_rng(11)
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


def _fake_xgb(monkeypatch):
    mod = types.ModuleType("xgboost")
    mod.__version__ = "fake-0 (native replay UNVERIFIED)"

    class Regressor:
        def __init__(self, **params):
            self._params = params
            self._mean = 0.0

        def fit(self, X, y):
            self._mean = float(np.asarray(y, float).mean())

        def get_booster(self):
            # Mirrors the real call chain: fit uses get_booster().save_model.
            return self

        def save_model(self, path):
            Path(path).write_bytes(json.dumps(
                {"params": self._params, "mean": self._mean}).encode())

        def predict(self, X):
            return np.full(len(X), self._mean)

    class Booster:
        def __init__(self):
            self._mean = 0.0

        def load_model(self, path):
            self._mean = float(json.loads(Path(path).read_bytes())["mean"])

        def predict(self, dm):
            return np.full(len(dm._arr), self._mean)

    class DMatrix:
        def __init__(self, data):
            self._arr = np.asarray(data)

    mod.XGBRegressor, mod.Booster, mod.DMatrix = Regressor, Booster, DMatrix
    monkeypatch.setitem(sys.modules, "xgboost", mod)
    return mod


# ----------------------------------------------------------------------------
# 1. fixed recipe: params, schema, no horizon/future columns
# ----------------------------------------------------------------------------
def test_fixed_params_closure_normalizer_and_schema():
    assert mean_level.MEAN_PARAMS == protocol.PARAMS
    assert mean_level.MEAN_PARAMS["random_state"] == 0
    assert mean_level.NORMALIZER is None
    assert list(protocol.SOURCE_CLOSURE) == [
        "leaderboard/store/ch-v1/ace.parquet",
        "leaderboard/store/ch-breakthrough-v2/upload/raw-ace-snapshot/",
        "leaderboard/store/ch-v1/ch-hourly.parquet"]
    assert mean_level.SUBMODULE_PROTOCOL_VERSION == protocol.PROTOCOL_VERSION
    cols = mean_level.mean_feature_columns(
        ["v_lag0", "ace_speed_kms_last", "speed_d24", "ch_x_area"])
    assert cols[-4:] == list(mean_level.TRAJ_MEAN_COLUMNS)
    for bad in ("h", "horizon", "horizon_hours", "y_pred", "official",
                "enlil_", "donki_", "simplex", "oof_blend"):
        assert bad not in cols
    d1 = mean_level.declare_schema(["v_lag0", "ch_x_area"])
    assert mean_level.declare_schema(["v_lag0", "ch_x_area"]) == d1  # stable pre-outcome
    dropped = mean_level.mean_feature_columns(["v_lag0", "donki_q", "official_z"])
    for bad in ("donki_", "official", "enlil_", "simplex", "oof_blend"):
        assert not any(bad in c for c in dropped)
    assert "v_lag0" in dropped


# ----------------------------------------------------------------------------
# 2. exact-72 labels, strict cutoff, eligibility ledger
# ----------------------------------------------------------------------------
def test_mean_labels_exact72_strict_cutoff_and_ledger():
    inputs = _synthetic_inputs()
    cutoff = inputs["origins"][-1] + pd.Timedelta(hours=72)  # last origin exactly at bound
    train, ledger = mean_level.build_mean_training(inputs, cutoff)
    assert ledger["n_requested"] == len(inputs["origins"])
    assert ledger["n_eligible"] + ledger["n_excluded"] == ledger["n_requested"]
    assert ledger["exclusion_reasons"]["label_window_not_before_cutoff"] >= 1
    last = inputs["origins"][-1]  # origin+72h == cutoff: strictly excluded
    assert last not in train.index
    assert protocol.ts_utc(ledger["train_label_max_utc"]) < protocol.ts_utc(cutoff)
    assert ledger["target_missingness"]["n_fully_observed_origins"] == ledger["n_requested"]
    for o in train.index:
        want = inputs["ace_filled"].reindex(
            o + pd.to_timedelta(np.arange(1, 73), unit="h")).to_numpy(float)
        assert np.isfinite(want).all()
        assert train.loc[o, "y_mean"] == pytest.approx(want.mean())
    # incomplete filled labels excluded without interpolation, reasons logged
    gap_origin = inputs["origins"][5]
    ace2 = inputs["ace_filled"].copy()
    ace2.loc[gap_origin + pd.Timedelta(hours=9)] = np.nan
    inputs2 = dict(inputs, ace_filled=ace2)
    train2, ledger2 = mean_level.build_mean_training(inputs2, cutoff)
    assert gap_origin not in train2.index
    assert ledger2["exclusion_reasons"]["incomplete_labels"] >= 1
    # training exclusions never drop evaluation origins
    assert (mean_level.issue_features(inputs2, inputs2["origins"]).index
            == inputs2["origins"]).all()


def test_missingness_distribution_logged():
    inputs = _synthetic_inputs()
    base = int((inputs["origins"][10] - inputs["truth_observed"].index[0]).total_seconds() // 3600)
    inputs["truth_observed"].iloc[[base + 10, base + 11, base + 12]] = np.nan
    cutoff = inputs["origins"][-1] + pd.Timedelta(hours=200)
    _, ledger = mean_level.build_mean_training(inputs, cutoff)
    miss = ledger["target_missingness"]
    assert miss["observed_fraction_min"] < 1.0
    assert miss["n_fully_observed_origins"] < ledger["n_requested"]


# ----------------------------------------------------------------------------
# 3. causal features: future poison/drop invariance + earlier positive control
# ----------------------------------------------------------------------------
def test_future_poison_and_drop_invariance_with_positive_control():
    inputs = _synthetic_inputs()
    cutoff = inputs["origins"][40]
    keep = inputs["origins"] <= cutoff
    clean = mean_level.issue_features(inputs, inputs["origins"][keep])
    dirty_ace = inputs["ace_filled"].copy()
    dirty_ace.loc[dirty_ace.index > cutoff] = 9999.0
    dirty = dict(inputs, ace_filled=dirty_ace)
    got = mean_level.issue_features(dirty, inputs["origins"][keep])
    protocol.assert_exact_requested_index(list(clean.index), list(got.index))
    assert list(clean.columns) == list(got.columns)
    for col in clean.columns:  # values and NaN masks identical
        a, b = clean[col].to_numpy(float), got[col].to_numpy(float)
        assert a.shape == b.shape
        assert (np.isnan(a) == np.isnan(b)).all()
        assert np.allclose(a[~np.isnan(a)], b[~np.isnan(a)], equal_nan=True)
    dropped = dict(inputs, ace_filled=inputs["ace_filled"].loc[:cutoff])
    got_drop = mean_level.issue_features(dropped, inputs["origins"][keep])
    for col in clean.columns:
        a, b = clean[col].to_numpy(float), got_drop[col].to_numpy(float)
        assert (np.isnan(a) == np.isnan(b)).all()
        assert np.allclose(a[~np.isnan(a)], b[~np.isnan(a)], equal_nan=True)
    # earlier positive control: poisoning inputs before the origin must change output
    early_ace = inputs["ace_filled"].copy()
    early_ace.loc[early_ace.index <= cutoff] = -777.0
    early = dict(inputs, ace_filled=early_ace)
    got_early = mean_level.issue_features(early, inputs["origins"][keep])
    assert any(not np.allclose(clean[c].to_numpy(float), got_early[c].to_numpy(float),
                               equal_nan=True) for c in clean.columns)


def test_recurrence_lookups_never_after_origin():
    inputs = _synthetic_inputs()
    origins = inputs["origins"]
    for d in (26, 27, 28):
        look_max = origins + pd.to_timedelta(72, unit="h") - pd.Timedelta(days=d)
        assert (look_max <= origins).all()
    feats = mean_level.issue_features(inputs, origins)
    assert feats[["traj_mean_26d", "traj_mean_27d", "traj_mean_28d"]].notna().all().all()
    assert np.allclose(feats["traj_mean"].to_numpy(float),
                       feats[["traj_mean_26d", "traj_mean_27d",
                              "traj_mean_28d"]].to_numpy(float).mean(axis=1))


# ----------------------------------------------------------------------------
# 4. decomposition identity + rejection paths + alias interop
# ----------------------------------------------------------------------------
def _direct_frame(origins, shift=0.0, canonical=True):
    rows = []
    for i, o in enumerate(origins):
        prof = 10 * np.sin(np.arange(1, 73) / 72 * 2 * np.pi) + i + shift
        for h, p in zip(range(1, 73), prof):
            rows.append((o, h, float(p)))
    df = pd.DataFrame(rows, columns=["o", "h", "p"])
    if canonical:
        return df.rename(columns={"o": mean_level.CANON_ORIGIN,
                                   "h": mean_level.CANON_H, "p": mean_level.CANON_PRED})
    return df.rename(columns={"o": "origin", "h": "h", "p": "y_pred"})


def test_decomposition_mean_identity_and_shape_invariance():
    origins = pd.date_range("2025-02-01", periods=6, freq="3h", tz="UTC")
    level = pd.Series(np.linspace(380, 460, len(origins)), index=origins)
    for direct in (_direct_frame(origins, canonical=True),
                   _direct_frame(origins, canonical=False)):
        out = mean_level.decompose_predictions(level, direct)
        assert list(out.columns) == [mean_level.CANON_ORIGIN, mean_level.CANON_H,
                                     mean_level.CANON_PRED]
        assert len(out) == len(origins) * 72
        assert np.allclose(out.groupby(mean_level.CANON_ORIGIN)[mean_level.CANON_PRED]
                           .mean().to_numpy(float), level.to_numpy(float))
        raw = _direct_frame(origins, canonical=True)
        merged = out.merge(raw, on=[mean_level.CANON_ORIGIN, mean_level.CANON_H],
                           suffixes=("", "_raw"))
        diff = merged[mean_level.CANON_PRED] - merged[f"{mean_level.CANON_PRED}_raw"]
        assert (diff.groupby(merged[mean_level.CANON_ORIGIN]).std() == 0.0).all()
        for o in origins:  # profile differences exactly preserved
            g = merged[merged[mean_level.CANON_ORIGIN] == o].sort_values(
                mean_level.CANON_H)
            assert np.allclose(g[mean_level.CANON_PRED].diff().iloc[1:].to_numpy(float),
                               g[f"{mean_level.CANON_PRED}_raw"].diff().iloc[1:].to_numpy(float))


def test_decomposition_rejects_bad_keys_and_nonfinite():
    origins = pd.date_range("2025-02-01", periods=3, freq="3h", tz="UTC")
    level = pd.Series(400.0, index=origins)
    good = _direct_frame(origins, canonical=False)
    with pytest.raises(ValueError):
        mean_level.decompose_predictions(level, good[good["h"] != 72])  # missing
    with pytest.raises(ValueError):
        mean_level.decompose_predictions(
            level, pd.concat([good, good.iloc[:1]]))  # duplicate
    bad = good.copy()
    bad.loc[bad.index[0], "y_pred"] = np.nan
    with pytest.raises(ValueError):
        mean_level.decompose_predictions(level, bad)  # nonfinite direct
    bad_inf = good.copy()
    bad_inf.loc[bad_inf.index[0], "y_pred"] = np.inf
    with pytest.raises(ValueError):
        mean_level.decompose_predictions(level, bad_inf)
    bad_level = level.copy()
    bad_level.iloc[0] = np.nan
    with pytest.raises(ValueError):
        mean_level.decompose_predictions(bad_level, good)
    with pytest.raises(ValueError):
        mean_level.decompose_predictions(level.iloc[:2], good)  # origin mismatch
    with pytest.raises(ValueError):
        mean_level.decompose_predictions(
            level, good.rename(columns={"y_pred": "nope"}))


# ----------------------------------------------------------------------------
# 5. persistence provenance (fake adapter, replay UNVERIFIED) + validation
# ----------------------------------------------------------------------------
def test_fit_predict_provenance_and_tamper_rejection(monkeypatch, tmp_path):
    _fake_xgb(monkeypatch)
    inputs = _synthetic_inputs()
    cutoff = inputs["origins"][60]
    train, _ = mean_level.build_mean_training(inputs, cutoff)
    cols = mean_level.mean_feature_columns(inputs["features"].columns)
    meta = mean_level.fit_mean_level(train, cols, tmp_path, cutoff=cutoff,
                                     registration={"ref": "root-claim-1"})
    assert (tmp_path / mean_level.MODEL_FILENAME).is_file()
    assert meta["model_sha256"] == protocol.sha256_file(tmp_path / mean_level.MODEL_FILENAME)
    assert meta["registration"] == {"ref": "root-claim-1"}
    assert meta["normalizer"] is None
    assert "UNVERIFIED" in meta["versions"]["xgboost"]
    assert meta["frozen_refs"]["reference_params"] == protocol.PARAMS
    assert protocol.ts_utc(meta["train_label_max_utc"]) < protocol.ts_utc(cutoff)
    got = mean_level.predict_saved(tmp_path, inputs, inputs["origins"][:10])
    assert len(got) == 10 and got.name == "level_hat" and np.isfinite(got.to_numpy()).all()
    tampered = json.loads((tmp_path / mean_level.METADATA_FILENAME).read_text())
    tampered["params"] = dict(tampered["params"], n_estimators=1)
    (tmp_path / mean_level.METADATA_FILENAME).write_text(json.dumps(tampered))
    with pytest.raises(ValueError):
        mean_level.predict_saved(tmp_path, inputs, inputs["origins"][:10])


def test_fit_fails_closed_without_xgboost(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "xgboost", None)
    inputs = _synthetic_inputs()
    cutoff = inputs["origins"][60]
    train, _ = mean_level.build_mean_training(inputs, cutoff)
    cols = mean_level.mean_feature_columns(inputs["features"].columns)
    with pytest.raises(protocol.NotReadyError):
        mean_level.fit_mean_level(train, cols, tmp_path, cutoff=cutoff)
