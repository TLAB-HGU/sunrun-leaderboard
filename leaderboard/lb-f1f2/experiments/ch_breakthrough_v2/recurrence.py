"""Solar-rotation recurrence lane (extension BEYOND the old trailing-5-day policy).

README
-----
Extension label: ``beyond-trailing-5d:synodic-recurrence-26/27/28d-v1``.

The frozen foundation (``common.build_features``) only sees the trailing ~72h
of ACE plus compact coronal-hole geometry. This lane extends it BEYOND the
old trailing-5-day policy with synodic-rotation recurrence over 26/27/28
days (fixed lags 624/648/672h), in two strictly separated forms:

- ``orec*`` (origin-relative control): for each origin ``o`` the ACE speed
  seen one synodic rotation earlier (``o - 26/27/28d``) plus nearby
  fixed-lag summaries (+/-12h and +/-24h rolling mean/std/min/max and
  genuine missing counts around each return) and the current-minus-return
  difference. Every source timestamp is ``<= o`` (return windows end at
  ``o-600h`` at the latest), so the control is causal at issuance time.
  These columns live per-origin in :func:`build_recurrence_features` and
  are labelled ``origin-recurrence`` — never called target-relative.

- ``trec*`` (target-relative, horizon-specific): for each ``(origin o,
  horizon h)`` pair the ACE speed at the synodic return of the *target*
  time, i.e. ``s = o + h - 26/27/28d``, plus the genuine missing flag at
  ``s``, the +/-12h rolling mean around ``s``, the current-minus-return
  difference, and the source age ``26/27/28d - h``. Because ``h <= 72``,
  every source satisfies ``s <= o - 552h <= o``, so the features are known
  at issuance time for all horizons 1..72. These columns vary with ``h``
  (see :func:`horizon_target_features`); the pooled forecaster does NOT
  repeat origin-lag values 72x. They are built inside
  :class:`RecurrenceResidualForecaster`, never from post-origin data.

Genuine missingness: recurrent missing flags (``orec*_missing_pm24h``,
``trec*D_missing``) derive from the genuine ``was_missing`` observation
validity (plus non-finite/non-positive speeds), never from finiteness of
the forward-filled speed. All-NaN/out-of-range windows stay NaN (native
XGBoost missingness).

Compact CH columns come from ``common.build_features`` unchanged (no
reimplementation).

The 26/27/28-day triplet and uniform (unweighted) treatment are fixed a
priori from the synodic rotation period; no lag, horizon, or weight was
picked because it scored better on the official diagnostic. Horizon
selection/weighting uses dev/selection only (here: none -- uniform).

Four matched-params arms (identical XGBoost params, identical forecaster
class, identical origins per split) are compared:

- ``recent-ACE``            : compact ``common.build_features`` ACE part
  only (no ``orec``, no ``trec``, no CH; baseline control).
- ``origin-recurrence``     : compact ACE columns + ``orec`` control
  columns (CH dropped; origin-only recurrence control).
- ``target-recurrence``     : compact ACE columns + horizon-specific
  ``trec`` columns (CH and ``orec`` dropped).
- ``target-recurrence+CH``  : compact ACE + CH columns + horizon-specific
  ``trec`` columns (``orec`` dropped; CH add-on to target-recurrence).

Chronological OOF per ``store/ch-breakthrough-v2/protocol.json`` blocks with
``common.purged_split`` (>=72h purge); bounded dev+selection+holdout with
6-hourly screening subsamples (recorded in every manifest), then one
official eval (all 2833 pinned origins, refit through 2026-05-31T23:00:00Z,
labelled exploratory, 2833 x 72 = 203976 keys). Every arm outputs horizons
1..72. Official pinned truth (``truth.parquet``) is scored read-only as a
diagnostic and never used for tuning or selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.ch_breakthrough_v2"

from . import common as _common
from .common import (
    HORIZON_HOURS,
    HORIZONS,
    N_THREADS,
    OUTPUT_RANGE_KMS,
    SEED,
    PooledResidualForecaster,
    acquire_gpu_lease,
    as_speed_series,
    build_features,
    check_cutoff,
    ensure_device,
    file_identity,
    fit_xgb_regressor,
    load_ace_hourly,
    load_ch_hourly,
    load_raw_ace,
    mean_reversion_predict,
    observed_mask,
    purged_split,
    score_forecast,
    subsample_origins,
    timed_predict,
    utc,
    write_manifest,
)

# Keep the frozen pooled control reachable for provenance checks.
_POOLED_BASE = PooledResidualForecaster

EXTENSION_LABEL = "beyond-trailing-5d:synodic-recurrence-26/27/28d-v1"
README = __doc__

RECURRENCE_DAYS = (26, 27, 28)
RECURRENCE_LAGS = tuple(day * 24 for day in RECURRENCE_DAYS)
REC_WINDOWS = (12, 24)

ARMS = ("recent-ACE", "origin-recurrence", "target-recurrence",
        "target-recurrence+CH")
ARM_USE_TARGET = {
    "recent-ACE": False,
    "origin-recurrence": False,
    "target-recurrence": True,
    "target-recurrence+CH": True,
}

HORIZON_DESIGN_NOTE = (
    "horizon-specific target-relative recurrence: trec columns vary by "
    "horizon (source o+h-26/27/28d); origin-lag values are NOT repeated 72x"
)

XGB_PARAMS = {
    "n_estimators": 150,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_lambda": 5.0,
}

SPLITS = (
    {"name": "dev-A", "train_through": "2024-12-31T23:00:00Z",
     "eval_start": "2025-01-01T00:00:00Z", "eval_end": "2025-03-31T23:00:00Z"},
    {"name": "dev-B", "train_through": "2025-04-30T23:00:00Z",
     "eval_start": "2025-05-01T00:00:00Z", "eval_end": "2025-07-31T23:00:00Z"},
    {"name": "dev-C", "train_through": "2025-08-31T23:00:00Z",
     "eval_start": "2025-09-01T00:00:00Z", "eval_end": "2025-11-30T23:00:00Z"},
    {"name": "selection", "train_through": "2025-11-30T23:00:00Z",
     "eval_start": "2025-12-01T00:00:00Z", "eval_end": "2026-02-28T23:00:00Z"},
    {"name": "holdout", "train_through": "2026-02-28T23:00:00Z",
     "eval_start": "2026-03-01T00:00:00Z", "eval_end": "2026-05-31T23:00:00Z"},
)
OFFICIAL = {"name": "official", "train_through": "2026-05-31T23:00:00Z",
            "n_origins": 2833,
            "label": "exploratory: evaluation period was previously "
                     "inspected during ch-v1; single pass only"}

PINNED_SNAPSHOT = Path(
    "/home/t-lab01/.cache/huggingface/hub/datasets--tlabtlab"
    "--sunrun-lb-store/snapshots"
    "/bcf5d5417d99eaa331ddca40581d48953e09a58e")

CH_PREFIX = "ch_"
OREC_PREFIX = "orec"
TREC_PREFIX = "trec"

TREC_COLUMNS: tuple = tuple(
    f"{TREC_PREFIX}{day}d{suffix}"
    for day in RECURRENCE_DAYS
    for suffix in ("", "_missing", "_mean_pm12h", "_diff", "_src_age_h")
)


def _speed_grid(ace_hourly) -> pd.Series:
    frame = ace_hourly.set_index("timestamp_utc").sort_index() \
        if "timestamp_utc" in ace_hourly.columns else ace_hourly.sort_index()
    speed = pd.to_numeric(frame["filled_speed_kms"], errors="coerce")
    speed.index = pd.DatetimeIndex(utc(speed.index))
    grid = pd.date_range(speed.index.min(), speed.index.max(), freq="h")
    return speed.reindex(grid)


def _genuine_missing_grid(ace_hourly) -> pd.Series:
    """Genuine (0/1) missingness on the hourly ACE grid.

    ``1`` where the hour was genuinely unobserved (``was_missing != 0`` or
    non-finite/non-positive speed); ``0`` otherwise. Forward-filled finite
    speeds with ``was_missing == 1`` stay missing here.
    """
    frame = ace_hourly.set_index("timestamp_utc").sort_index() \
        if "timestamp_utc" in ace_hourly.columns else ace_hourly.sort_index()
    index = pd.DatetimeIndex(utc(frame.index))
    vals = pd.to_numeric(frame["filled_speed_kms"], errors="coerce").to_numpy(
        dtype=float)
    if "was_missing" in frame.columns:
        raw = frame["was_missing"].to_numpy()
        miss = np.zeros(len(frame), dtype=bool)
        for i, value in enumerate(raw):
            try:
                miss[i] = True if pd.isna(value) else float(value) != 0.0
            except (TypeError, ValueError):
                miss[i] = True
    else:
        miss = np.zeros(len(frame), dtype=bool)
    invalid = (~np.isfinite(vals)) | ~(vals > 0)
    genuine = (miss | invalid).astype(float)
    grid = pd.date_range(index.min(), index.max(), freq="h")
    return pd.Series(genuine, index=index).reindex(grid).fillna(1.0)


def build_recurrence_features(ace_hourly, ch_hourly, raw_ace=None,
                              include_ch=True) -> pd.DataFrame:
    """Compact base features plus causal origin-relative ``orec`` control.

    ``include_ch=False`` drops the ``ch_*`` columns while keeping the
    identical origin grid (so arms stay matched). All source timestamps are
    ``<= origin``: return windows end at ``origin-600h`` at the latest.
    Missing counts use genuine ``was_missing`` validity, never finiteness
    of the forward-filled speed.
    """
    base = build_features(ace_hourly, ch_hourly, raw_ace)
    origins = base.index
    speed = _speed_grid(ace_hourly)
    genuine = _genuine_missing_grid(ace_hourly)
    rec = {}
    for day, lag in zip(RECURRENCE_DAYS, RECURRENCE_LAGS):
        lagged = speed.shift(lag)
        rec[f"{OREC_PREFIX}{day}d_lag0"] = lagged.reindex(
            origins).to_numpy(dtype=float)
        current = speed.reindex(origins).to_numpy(dtype=float)
        rec[f"{OREC_PREFIX}{day}d_diff"] = current - rec[
            f"{OREC_PREFIX}{day}d_lag0"]
        for width in REC_WINDOWS:
            window = 2 * width + 1
            center = speed.rolling(window, center=True, min_periods=1)
            rec[f"{OREC_PREFIX}{day}d_mean_pm{width}h"] = center.mean().shift(
                lag).reindex(origins).to_numpy(dtype=float)
            rec[f"{OREC_PREFIX}{day}d_std_pm{width}h"] = center.std().shift(
                lag).reindex(origins).to_numpy(dtype=float)
            rec[f"{OREC_PREFIX}{day}d_min_pm{width}h"] = center.min().shift(
                lag).reindex(origins).to_numpy(dtype=float)
            rec[f"{OREC_PREFIX}{day}d_max_pm{width}h"] = center.max().shift(
                lag).reindex(origins).to_numpy(dtype=float)
        missing = genuine.rolling(2 * REC_WINDOWS[1] + 1, center=True,
                                  min_periods=1).sum().shift(lag)
        rec[f"{OREC_PREFIX}{day}d_missing_pm{REC_WINDOWS[1]}h"] = missing.reindex(
            origins).to_numpy(dtype=float)
    rec_frame = pd.DataFrame(rec, index=origins).astype(np.float32)
    out = pd.concat([base, rec_frame], axis=1)
    # Lane-side guard: the frozen base can emit +/-inf (0/0 area ratios on
    # real data); XGBoost GPU hist rejects inf, so map it to native NaN.
    # common.py itself is untouched (owned by the foundation lane).
    out = out.replace([np.inf, -np.inf], np.nan)
    if not include_ch:
        out = out[[c for c in out.columns if not c.startswith(CH_PREFIX)]]
    out.index.name = "origin_last_input_utc"
    return out


def target_source_time(origin, horizon_hours: int, day: int) -> pd.Timestamp:
    """Issuance-known source timestamp for one target-relative lookup."""
    return pd.Timestamp(utc(origin)) + pd.Timedelta(hours=int(horizon_hours)) \
        - pd.Timedelta(hours=int(day) * 24)


def horizon_target_features(speed, origins) -> dict:
    """Per-(origin, horizon) target-relative recurrence lookups.

    For each ``(o, h)`` and each rotation ``D`` in 26/27/28 days the source
    ``s = o + h - D*24`` is looked up. Because ``h <= 72``, every source
    satisfies ``s <= o - 552h <= o``: known at issuance time. Returns
    float arrays of shape ``(n_origins, 72)`` keyed by ``TREC_COLUMNS``:

    - ``trec{D}d``: filled speed at ``s`` (NaN when ``s`` is out of range).
    - ``trec{D}d_missing``: genuine missing flag at ``s`` (1.0 when
      unobserved or out of range, else 0.0).
    - ``trec{D}d_mean_pm12h``: +/-12h rolling mean around ``s`` (window
      ends at ``s+12h <= o-540h``; causal).
    - ``trec{D}d_diff``: filled speed at ``o`` minus ``trec{D}d``.
    - ``trec{D}d_src_age_h``: source age at issuance (``D*24 - h``).
    """
    speed = speed.sort_index()
    origins = pd.DatetimeIndex(utc(origins)).sort_values()
    stored = getattr(speed, "attrs", {}).get("observed", None)
    grid = pd.date_range(speed.index.min(), speed.index.max(), freq="h")
    grid = pd.DatetimeIndex(utc(grid))
    filled = speed.reindex(grid).to_numpy(dtype=float)
    if isinstance(stored, pd.Series):
        observed = stored.reindex(grid).fillna(False).to_numpy(dtype=bool)
    else:
        observed = np.isfinite(filled) & (filled > 0)
    missing_grid = (~observed).astype(float)
    mean12 = pd.Series(filled, index=grid).rolling(
        25, center=True, min_periods=1).mean().to_numpy(dtype=float)
    current = pd.Series(filled, index=grid).reindex(origins).to_numpy(
        dtype=float)
    horizons = np.arange(1, HORIZON_HOURS + 1, dtype=np.int64)
    base_ns = pd.DatetimeIndex(utc(origins)).asi8  # ns since epoch, UTC
    out: dict[str, np.ndarray] = {}
    for day in RECURRENCE_DAYS:
        lag = int(day) * 24
        stamps = pd.DatetimeIndex(
            (base_ns[:, None]
             + (horizons[None, :] - lag).astype(np.int64)
             * 3_600_000_000_000).ravel(), tz="UTC")
        pos = grid.get_indexer(stamps)
        ok = pos >= 0
        value = np.where(ok, filled[np.clip(pos, 0, len(grid) - 1)], np.nan)
        miss = np.where(
            ok, missing_grid[np.clip(pos, 0, len(grid) - 1)], 1.0)
        mean = np.where(
            ok, mean12[np.clip(pos, 0, len(grid) - 1)], np.nan)
        shape = (len(origins), HORIZON_HOURS)
        value = value.reshape(shape)
        miss = miss.reshape(shape)
        mean = mean.reshape(shape)
        out[f"{TREC_PREFIX}{day}d"] = value
        out[f"{TREC_PREFIX}{day}d_missing"] = miss
        out[f"{TREC_PREFIX}{day}d_mean_pm12h"] = mean
        out[f"{TREC_PREFIX}{day}d_diff"] = current[:, None] - value
        out[f"{TREC_PREFIX}{day}d_src_age_h"] = np.tile(
            (lag - horizons).astype(float), (len(origins), 1))
    return out


def arm_columns(full_features, arm: str) -> list:
    """Per-origin column selection for one matched-params arm."""
    if arm == "recent-ACE":
        return [c for c in full_features.columns
                if not c.startswith(CH_PREFIX)
                and not c.startswith(OREC_PREFIX)
                and not c.startswith(TREC_PREFIX)]
    if arm == "origin-recurrence":
        return [c for c in full_features.columns
                if not c.startswith(CH_PREFIX)
                and not c.startswith(TREC_PREFIX)]
    if arm == "target-recurrence":
        return [c for c in full_features.columns
                if not c.startswith(CH_PREFIX)
                and not c.startswith(OREC_PREFIX)
                and not c.startswith(TREC_PREFIX)]
    if arm == "target-recurrence+CH":
        return [c for c in full_features.columns
                if not c.startswith(OREC_PREFIX)
                and not c.startswith(TREC_PREFIX)]
    raise ValueError(f"unknown arm: {arm}")


def arm_use_target(arm: str) -> bool:
    """Whether an arm adds horizon-specific ``trec`` columns in the pooled design."""
    try:
        return bool(ARM_USE_TARGET[arm])
    except KeyError:
        raise ValueError(f"unknown arm: {arm}") from None


class RecurrenceResidualForecaster:
    """Pooled residual XGBoost with horizon-specific recurrence interactions.

    Target is ``truth(origin+h) - mean_reversion(origin, h)``; the horizon
    enters as an explicit numeric feature and one estimator covers all 72
    horizons. Target arms additionally carry per-``(origin, horizon)``
    ``trec`` columns (see :func:`horizon_target_features`) so the design
    varies by horizon instead of repeating origin-lag values 72x.

    Fit/predict separation: :meth:`fit` builds labels from future truth
    (origins + 1..72h, all ``<= cutoff``); :meth:`predict` builds only the
    feature matrix plus the causal mean-reversion anchor and never touches
    future truth, so perturbing or deleting post-origin speeds leaves
    predictions bit-identical.
    """

    def __init__(self, params=None, device="cpu", seed=SEED,
                 use_target=True):
        self.params = dict(params or {})
        self.device = device
        self.seed = seed
        self.use_target = bool(use_target)
        self.model = None
        self.base_columns: list = []
        self.columns: list = []
        self.cutoff = None
        self.actual_device = None
        self.n_short_history_skipped = 0

    def _build_predict_matrix(self, features, origins, speed):
        """Feature matrix + causal anchor for prediction (no future truth)."""
        from experiments.ch_breakthrough_v2.common import MEAN_WINDOW_HOURS  # noqa: F401
        origins = pd.DatetimeIndex(utc(origins)).sort_values()
        base = mean_reversion_predict(speed, origins)
        mr = base["pred_kms"].to_numpy(dtype=float)
        X = features.reindex(origins, columns=self.base_columns) \
            if self.base_columns else features.reindex(origins)
        missing_origins = X.index[X.isna().all(axis=1)]
        if len(missing_origins):
            raise ValueError(
                f"features missing origins: {missing_origins[0]}")
        big_base = np.repeat(X.to_numpy(dtype=float), HORIZON_HOURS, axis=0)
        names = list(X.columns) + ["horizon_hours", "horizon_scaled"]
        parts = [big_base,
                 np.tile(np.arange(1, HORIZON_HOURS + 1, dtype=float),
                         len(origins))[:, None],
                 np.tile(np.arange(1, HORIZON_HOURS + 1, dtype=float)
                         / HORIZON_HOURS, len(origins))[:, None]]
        if self.use_target:
            inter = horizon_target_features(speed, origins)
            for name in TREC_COLUMNS:
                parts.append(inter[name].ravel()[:, None])
                names.append(name)
        big = pd.DataFrame(np.column_stack(parts), columns=names)
        big = big.replace([np.inf, -np.inf], np.nan)
        return origins, big, mr

    def _design(self, features, origins, speed):
        """Fit-time design: predict matrix plus pooled future-truth labels."""
        origins, big, mr = self._build_predict_matrix(features, origins,
                                                      speed)
        base_horizons = np.tile(np.arange(1, HORIZON_HOURS + 1),
                                len(origins))
        truth = speed.reindex(
            origins.repeat(HORIZON_HOURS)
            + pd.to_timedelta(base_horizons, unit="h")
        ).to_numpy(dtype=float)
        return origins, big, truth, mr

    def fit(self, features, speed, cutoff, max_origins=None):
        from experiments.ch_breakthrough_v2.common import MEAN_WINDOW_HOURS
        cutoff = utc(cutoff)
        self.base_columns = list(features.columns)
        origins = pd.DatetimeIndex(utc(features.index))
        origins = origins[origins <= cutoff]
        origins = origins[origins + pd.Timedelta(hours=HORIZON_HOURS) <= cutoff]
        check_cutoff(origins, origins.max() + pd.to_timedelta(
            np.arange(1, HORIZON_HOURS + 1), unit="h"), cutoff)
        # Mean-reversion anchor needs a full 648h window; skip short-history
        # origins explicitly instead of failing or leaking future data.
        depth = speed.sort_index().rolling(
            MEAN_WINDOW_HOURS, min_periods=MEAN_WINDOW_HOURS).mean()
        self.n_short_history_skipped = int(
            depth.reindex(origins).isna().sum())
        origins = origins[depth.reindex(origins).notna().to_numpy()]
        if max_origins is not None and len(origins) > max_origins:
            keep = np.linspace(0, len(origins) - 1, max_origins, dtype=int)
            origins = origins[keep]
        if len(origins) == 0:
            raise ValueError(
                "no training origins with complete labels before cutoff")
        origins, big, truth, mr = self._design(features, origins, speed)
        ok = np.isfinite(truth)
        if not ok.any():
            raise ValueError("no finite pooled training labels")
        # Unweighted control: every (origin, horizon) pair counts equally.
        self.model = fit_xgb_regressor(big.loc[ok], (truth - mr)[ok],
                                       self.params, self.device, self.seed)
        self.columns = list(big.columns)
        self.cutoff = cutoff
        try:
            self.actual_device = _common._booster_device(self.model,
                                                         self.device)
        except Exception:
            self.actual_device = str(self.device)
        return self

    def predict(self, features, speed, origins) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("fit the forecaster before predicting")
        origins, big, mr = self._build_predict_matrix(features, origins,
                                                      speed)
        resid = self.model.predict(big[self.columns])
        pred = np.clip(mr + resid, *OUTPUT_RANGE_KMS)
        return pd.DataFrame({
            "origin_last_input_utc": np.repeat(origins, HORIZON_HOURS),
            "horizon_hours": np.tile(np.arange(1, HORIZON_HOURS + 1),
                                     len(origins)),
            "pred_kms": pred,
        })


class _LeaseTimeout(Exception):
    pass


def fit_with_lease_or_cpu(fit_fn, lease_wait_s=30.0):
    """Run ``fit_fn(device)`` holding ``common.acquire_gpu_lease``.

    Waits at most ``lease_wait_s`` for a GPU lease, then falls back to CPU.
    The device actually used is fail-closed probed via
    ``common.ensure_device`` and returned alongside the fitted object.
    """
    def _handler(signum, frame):
        raise _LeaseTimeout()

    if hasattr(signal, "SIGALRM"):
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(int(lease_wait_s))
        try:
            with acquire_gpu_lease() as gpu:
                signal.alarm(0)
                device = f"cuda:{gpu}"
                ensure_device(device)
                return fit_fn(device), device, {"lease": "gpu", "gpu": gpu}
        except _LeaseTimeout:
            pass
        finally:
            try:
                signal.alarm(0)
            except Exception:
                pass
            signal.signal(signal.SIGALRM, old)
    else:
        try:
            with acquire_gpu_lease() as gpu:
                device = f"cuda:{gpu}"
                ensure_device(device)
                return fit_fn(device), device, {"lease": "gpu", "gpu": gpu}
        except Exception:
            pass
    ensure_device("cpu")
    return fit_fn("cpu"), "cpu", {"lease": "cpu-fallback",
                                  "note": "gpu leases busy or unavailable"}


def train_forecaster(train_features, speed, cutoff, params=None, device="cpu",
                     seed=SEED, max_train_origins=None,
                     use_target=True) -> RecurrenceResidualForecaster:
    """Fit one recurrence residual XGBoost forecaster (cutoff enforced)."""
    cutoff = utc(cutoff)
    origins = pd.DatetimeIndex(utc(train_features.index))
    origins = origins[origins <= cutoff]
    check_cutoff(origins, origins, cutoff)
    forecaster = RecurrenceResidualForecaster(
        params=dict(params if params is not None else XGB_PARAMS),
        device=device, seed=seed, use_target=use_target)
    forecaster.fit(train_features, speed, cutoff,
                   max_origins=max_train_origins)
    return forecaster


def train(*args, **kwargs) -> RecurrenceResidualForecaster:
    """Alias of :func:`train_forecaster` (required train entry point)."""
    return train_forecaster(*args, **kwargs)


def evaluate_forecaster(forecaster, features, speed, origins) -> pd.DataFrame:
    """Predict horizons 1..72 for ``origins`` (causal, no refit)."""
    return forecaster.predict(features, speed, origins)


def evaluate(*args, **kwargs) -> pd.DataFrame:
    """Alias of :func:`evaluate_forecaster` (required evaluate entry point)."""
    return evaluate_forecaster(*args, **kwargs)


def eval_origins_for_split(feature_index, cutoff, eval_start, eval_end,
                           step="6h", max_origins=None, seed=SEED):
    """Purged (>=72h) eval origins for one split, subsampled + recorded."""
    cutoff, eval_start, eval_end = utc(cutoff), utc(eval_start), utc(eval_end)
    grid = pd.date_range(eval_start, eval_end, freq="h")
    candidates = pd.DatetimeIndex(sorted(set(grid).intersection(
        set(pd.DatetimeIndex(utc(feature_index))))))
    _, test = purged_split(candidates, cutoff, end=eval_end,
                           horizon=HORIZON_HOURS, purge=72)
    if len(test) == 0:
        raise ValueError("no eval origins after purge")
    kept, sampling = subsample_origins(test, step=step,
                                       max_origins=max_origins, seed=seed)
    sampling = {"split_cutoff": cutoff.isoformat(), **sampling}
    return kept, sampling


def official_origins(feature_index, cutoff, n_origins=OFFICIAL["n_origins"]):
    """All pinned official origins starting at the refit cutoff (no sampling)."""
    cutoff = utc(cutoff)
    pinned = pd.date_range(cutoff, periods=n_origins, freq="h")
    kept = pd.DatetimeIndex(sorted(set(pinned).intersection(
        set(pd.DatetimeIndex(utc(feature_index))))))
    if len(kept) == 0:
        raise ValueError("no official origins on the feature grid")
    return kept, {"step": "hourly", "requested": len(pinned),
                  "kept": len(kept), "seed": SEED,
                  "note": "official: all supplied origins, no subsampling"}


def horizon_band_mse(predictions, speed) -> dict:
    """Filled-truth MSE split into 1-24 / 25-48 / 49-72h bands."""
    frame = predictions.copy()
    frame["truth"] = speed.reindex(
        pd.DatetimeIndex(utc(frame["origin_last_input_utc"]))
        + pd.to_timedelta(frame["horizon_hours"].to_numpy(), unit="h")
    ).to_numpy(dtype=float)
    frame = frame[np.isfinite(frame["truth"].to_numpy())]
    bands = {(1, 24): "h1_24", (25, 48): "h25_48", (49, 72): "h49_72"}
    out = {}
    for (lo, hi), name in bands.items():
        block = frame[(frame["horizon_hours"] >= lo)
                      & (frame["horizon_hours"] <= hi)]
        err = block["pred_kms"].to_numpy(dtype=float) - block[
            "truth"].to_numpy(dtype=float)
        out[name] = float(np.mean(err ** 2)) if len(block) else float("nan")
    out["n_scored"] = int(len(frame))
    return out


def compare_arms(full_features, speed, cutoff, eval_origins, params=None,
                 device="cpu", seed=SEED, max_train_origins=None) -> dict:
    """Train + predict the 4 arms with matched params on shared origins."""
    params = dict(params if params is not None else XGB_PARAMS)
    eval_origins = pd.DatetimeIndex(utc(eval_origins)).sort_values()
    result = {"params": params, "device": device,
              "origins": eval_origins, "arms": {}}
    for arm in ARMS:
        cols = arm_columns(full_features, arm)
        use_target = arm_use_target(arm)
        started = time.perf_counter()
        forecaster = train_forecaster(full_features[cols], speed, cutoff,
                                      params=params, device=device, seed=seed,
                                      max_train_origins=max_train_origins,
                                      use_target=use_target)
        fit_seconds = time.perf_counter() - started
        preds, secs = timed_predict(
            lambda _: evaluate_forecaster(forecaster, full_features[cols],
                                          speed, eval_origins), eval_origins)
        assert set(preds["horizon_hours"]) == set(HORIZONS)
        result["arms"][arm] = {"forecaster": forecaster,
                               "predictions": preds,
                               "columns": cols,
                               "use_target": use_target,
                               "fit_seconds": fit_seconds,
                               "inference_seconds_per_origin": secs}
    return result


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_pinned_truth():
    """Read-only pinned official truth (diagnostic; never used for tuning)."""
    try:
        truth_path = PINNED_SNAPSHOT / "truth.parquet"
        folds_path = PINNED_SNAPSHOT / "folds.parquet"
        if not truth_path.is_file():
            return None, None
        truth = pd.read_parquet(truth_path)
        folds = pd.read_parquet(folds_path) if folds_path.is_file() else None
        return truth, folds
    except Exception:
        return None, None


def _pinned_scores_for_split(outdir, split_name, arms):
    """Read-only pinned-truth MSE per arm (diagnostic; never tuning input)."""
    truth, _ = _read_pinned_truth()
    if truth is None:
        return None
    truth = truth.copy()
    truth["origin_last_input_utc"] = utc(truth["origin_last_input_utc"])
    keyed = truth.set_index(["origin_last_input_utc", "horizon_hours"])
    out = {}
    for arm in arms:
        path = Path(outdir) / f"{arm}__{split_name}.parquet"
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        idx = pd.MultiIndex.from_arrays(
            [utc(frame["origin_last_input_utc"]), frame["horizon_hours"]])
        aligned = keyed.reindex(idx)
        ok = np.isfinite(aligned["target_kms"].to_numpy())
        err = frame["pred_kms"].to_numpy()[ok] \
            - aligned["target_kms"].to_numpy()[ok]
        out[arm] = {"pairs": int(ok.sum()),
                    "mse_pinned": float(np.mean(err ** 2)) if ok.any() else None}
    if len(truth):
        out["imputed_truth_fraction"] = float(
            np.mean(truth["was_missing"].to_numpy() == 1))
        out["n_keys"] = int(len(truth))
    return out


def finish_arm_split(*, arm, split_name, full_features, speed, forecaster,
                     cutoff, fit_seconds, eval_origins, sampling, params,
                     device, device_info, seed, max_train_origins, outdir,
                     input_identities, extra_label=None) -> dict:
    """Score + write parquet/manifest/metrics for one fitted arm (no refit)."""
    cols = arm_columns(full_features, arm)
    preds, secs = timed_predict(
        lambda _: evaluate_forecaster(forecaster, full_features[cols],
                                      speed, eval_origins), eval_origins)
    assert set(preds["horizon_hours"]) == set(HORIZONS)
    keys = pd.MultiIndex.from_arrays(
        [utc(preds["origin_last_input_utc"]), preds["horizon_hours"]])
    assert keys.is_unique and len(keys) == len(eval_origins) * HORIZON_HOURS
    observed = observed_mask(speed, speed.index)
    metrics = score_forecast(preds, speed, observed, origins=eval_origins)
    bands = horizon_band_mse(preds, speed)
    metrics = {"arm": arm, "split": split_name, "horizons": [1, 72],
               "horizon_band_mse_filled": bands, "sampling": sampling,
               **metrics}
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    parquet_path = outdir / f"{arm}__{split_name}.parquet"
    preds.to_parquet(parquet_path, index=False)
    metrics_path = outdir / f"{arm}__{split_name}.metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True,
                                       default=str) + "\n")
    use_target = bool(getattr(forecaster, "use_target", arm_use_target(arm)))
    extra = {"extension": EXTENSION_LABEL, "arm": arm, "split": split_name,
             "use_target": use_target,
             "horizon_design": HORIZON_DESIGN_NOTE if use_target
             else "origin-only control: pooled residual with horizon as a "
                  "numeric feature (origin-lag values repeated across "
                  "horizons by construction; labelled control, not "
                  "target-relative)",
             "origins": len(eval_origins),
             "eval_start": pd.Timestamp(eval_origins.min()).isoformat(),
             "eval_end": pd.Timestamp(eval_origins.max()).isoformat(),
             "mse": metrics["mse"], "mse_observed": metrics["mse_observed"],
             "horizon_band_mse_filled": bands,
             "imputed_fraction": metrics.get("imputed_fraction"),
             "unique_observed_peak_timestamps": metrics.get(
                 "unique_observed_peak_timestamps"),
             "overlap_block_ci95_7day": metrics.get("overlap_block_ci95_7day"),
             "device_requested": device, "device_used": device,
             "device_info": device_info, "n_threads": N_THREADS,
             "fit_seconds": fit_seconds,
             "max_train_origins": max_train_origins,
             "prediction_sha256": _sha256_file(parquet_path),
             "horizon_note": "horizons 1..72 complete per origin; "
                             "26/27/28d triplet fixed a priori, not picked "
                             "on official truth (uniform weights)"}
    if extra_label:
        extra["label"] = extra_label
    manifest = write_manifest(
        outdir / f"{arm}__{split_name}.manifest.json",
        mode=f"recurrence-{arm}", train_cutoff=cutoff, params=params,
        tree_counts={"n_estimators": params.get("n_estimators")},
        columns=list(forecaster.columns),
        input_identities=input_identities, seed=seed,
        inference_seconds_per_origin=secs, sampling=sampling, extra=extra,
        actual_device=device, prediction_path=parquet_path)
    return {"arm": arm, "split": split_name, "mse": metrics["mse"],
            "mse_observed": metrics["mse_observed"], "bands": bands,
            "origins": len(eval_origins), "fit_seconds": fit_seconds,
            "secs_per_origin": secs, "device": device,
            "use_target": use_target,
            "parquet": str(parquet_path), "manifest": manifest}


def run_arm_split(*, arm, split_name, full_features, speed, cutoff,
                  eval_origins, sampling, params, device, device_info, seed,
                  max_train_origins, outdir, input_identities,
                  extra_label=None) -> dict:
    """Fit one arm on one split (given device); write parquet+manifest+metrics."""
    cols = arm_columns(full_features, arm)
    use_target = arm_use_target(arm)
    started = time.perf_counter()
    forecaster = train_forecaster(full_features[cols], speed, cutoff,
                                  params=params, device=device, seed=seed,
                                  max_train_origins=max_train_origins,
                                  use_target=use_target)
    fit_seconds = time.perf_counter() - started
    return finish_arm_split(
        arm=arm, split_name=split_name, full_features=full_features,
        speed=speed, forecaster=forecaster, cutoff=cutoff,
        fit_seconds=fit_seconds,
        eval_origins=eval_origins, sampling=sampling, params=params,
        device=device, device_info=device_info, seed=seed,
        max_train_origins=max_train_origins, outdir=outdir,
        input_identities=input_identities, extra_label=extra_label)


def run_experiment(*, ace_paths, ch_path, raw_ace_path=None, outdir,
                   origin_step="6h", max_train_origins=2000,
                   official_max_train_origins=3000, seed=SEED,
                   splits=None, params=None, lease_wait_s=30.0,
                   include_official=True) -> dict:
    """Bounded dev+selection+holdout sweep plus single official eval."""
    params = dict(params if params is not None else XGB_PARAMS)
    ace = load_ace_hourly(ace_paths)
    ch = load_ch_hourly(ch_path)
    raw = load_raw_ace(raw_ace_path)
    speed = as_speed_series(ace)
    full = build_recurrence_features(ace, ch, raw, include_ch=True)
    identities = {"ace": file_identity(ace_paths[0]),
                  "ch_hourly": file_identity(ch_path),
                  "protocol": file_identity("store/ch-breakthrough-v2/protocol.json")}
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    wanted = set(splits) if splits else None
    summary = {"extension": EXTENSION_LABEL, "params": params, "seed": seed,
               "feature_columns": len(full.columns),
               "orec_columns": sum(c.startswith(OREC_PREFIX)
                                   for c in full.columns),
               "trec_columns": len(TREC_COLUMNS),
               "horizon_design": HORIZON_DESIGN_NOTE,
               "splits": {}}

    def _fit_factory(arm, train_df, cutoff, cap):
        use_target = arm_use_target(arm)

        def _fit(device):
            return train_forecaster(train_df, speed, cutoff, params=params,
                                    device=device, seed=seed,
                                    max_train_origins=cap,
                                    use_target=use_target)
        return _fit

    for split in SPLITS:
        name = split["name"]
        if wanted and name not in wanted:
            continue
        cutoff = utc(split["train_through"])
        eval_origins, sampling = eval_origins_for_split(
            full.index, cutoff, split["eval_start"], split["eval_end"],
            step=origin_step, seed=seed)
        split_res = {"cutoff": cutoff.isoformat(), "sampling": sampling,
                     "arms": {}}
        for arm in ARMS:
            cols = arm_columns(full, arm)
            started = time.perf_counter()
            forecaster, device, device_info = fit_with_lease_or_cpu(
                _fit_factory(arm, full[cols], cutoff, max_train_origins),
                lease_wait_s=lease_wait_s)
            res = finish_arm_split(
                arm=arm, split_name=name, full_features=full, speed=speed,
                forecaster=forecaster, cutoff=cutoff,
                fit_seconds=time.perf_counter() - started,
                eval_origins=eval_origins, sampling=sampling, params=params,
                device=device, device_info=device_info, seed=seed,
                max_train_origins=max_train_origins, outdir=outdir,
                input_identities=identities)
            split_res["arms"][arm] = res
        summary["splits"][name] = split_res

    if include_official and (wanted is None or "official" in wanted):
        cutoff = utc(OFFICIAL["train_through"])
        eval_origins, sampling = official_origins(full.index, cutoff)
        split_res = {"cutoff": cutoff.isoformat(), "sampling": sampling,
                     "label": OFFICIAL["label"], "arms": {}}
        for arm in ARMS:
            cols = arm_columns(full, arm)
            started = time.perf_counter()
            forecaster, device, device_info = fit_with_lease_or_cpu(
                _fit_factory(arm, full[cols], cutoff,
                             official_max_train_origins),
                lease_wait_s=lease_wait_s)
            res = finish_arm_split(
                arm=arm, split_name="official", full_features=full,
                speed=speed, forecaster=forecaster, cutoff=cutoff,
                fit_seconds=time.perf_counter() - started,
                eval_origins=eval_origins,
                sampling=sampling, params=params, device=device,
                device_info=device_info, seed=seed,
                max_train_origins=official_max_train_origins, outdir=outdir,
                input_identities=identities, extra_label=OFFICIAL["label"])
            split_res["arms"][arm] = res
        pinned = _pinned_scores_for_split(outdir, "official", ARMS)
        if pinned is not None:
            split_res["pinned_truth"] = pinned
        summary["splits"]["official"] = split_res

    summary["status"] = "complete"
    summary["limitations"] = [
        "screening splits use 6-hourly eval subsamples (recorded); official "
        "uses all 2833 pinned origins (2833 x 72 = 203976 keys)",
        "target arms use horizon-specific trec features (source o+h-26/27/28d, "
        "all <= origin); origin-only arms repeat per-origin lags across "
        "horizons by pooled-residual construction and are labelled controls",
        "ACE raw point_in_time_vintage_verified is zero: operational-vintage "
        "limitation reported, never marked verified",
        "26/27/28d triplet fixed a priori; no lag/horizon/weight selection "
        "on official truth (uniform weights); official is exploratory "
        "(single pass) and never used for selection",
        "lane maps frozen-base +/-inf cells (0/0 area ratios, 5 cells in "
        "ch_agree_ratio_055_035) to native NaN so GPU hist accepts the "
        "design; common.py untouched",
    ]
    (outdir / "summary.json").write_text(json.dumps(
        {k: v for k, v in summary.items() if k != "splits"}, indent=2,
        sort_keys=True, default=str) + "\n")
    (outdir / "results.json").write_text(json.dumps(summary, indent=2,
                                                    sort_keys=True,
                                                    default=str) + "\n")
    return summary


def write_readme(outdir, summary=None) -> Path:
    """Artifact README documenting the extension label and outcomes."""
    lines = [
        "# Recurrence lane",
        "",
        f"Extension: `{EXTENSION_LABEL}` -- solar-rotation recurrence features",
        "BEYOND the old trailing-5-day policy: ACTUAL target-relative",
        "26/27/28-day ACE recurrence (source o+h-26/27/28d, causal for every",
        "horizon 1..72) with horizon-specific pooled design (trec varies by",
        "horizon; origin lags are NOT repeated 72x), plus a labelled",
        "origin-only orec control (source o-26/27/28d).",
        "",
        "Arms (matched params, shared origins per split): recent-ACE /",
        "origin-recurrence / target-recurrence / target-recurrence+CH.",
        "Horizons 1..72 everywhere. Official: 2833 pinned origins x 72 =",
        "203976 keys, exploratory single pass, pinned-truth diagnostic only.",
        "",
    ]
    if summary:
        lines.append("## MSE by split/arm")
        lines.append("")
        for name, split in summary.get("splits", {}).items():
            for arm, res in split.get("arms", {}).items():
                bands = res.get("bands", {})
                lines.append(
                    f"- {name}/{arm}: mse={res.get('mse')} "
                    f"mse_obs={res.get('mse_observed')} bands={bands}")
        lines.append("")
        pinned = (summary.get("splits", {}).get("official", {})
                  .get("pinned_truth"))
        if pinned:
            lines.append(f"Pinned official truth diagnostic: {pinned}")
            lines.append("")
        for item in summary.get("limitations", []):
            lines.append(f"- limitation: {item}")
        lines.append("")
    path = Path(outdir) / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ace", nargs="+", default=["store/ch-v1/ace.parquet"])
    p.add_argument("--ch-hourly", default="store/ch-v1/ch-hourly.parquet")
    p.add_argument("--raw-ace", default=None)
    p.add_argument("--output-dir",
                   default="store/ch-breakthrough-v2/validated/recurrence")
    p.add_argument("--origin-step", default="6h")
    p.add_argument("--max-train-origins", type=int, default=2000)
    p.add_argument("--official-max-train-origins", type=int, default=3000)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--splits", nargs="*", default=None)
    p.add_argument("--no-official", action="store_true")
    p.add_argument("--lease-wait-s", type=float, default=30.0)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    outdir = Path(args.output_dir)
    try:
        summary = run_experiment(
            ace_paths=args.ace, ch_path=args.ch_hourly,
            raw_ace_path=args.raw_ace, outdir=outdir,
            origin_step=args.origin_step,
            max_train_origins=args.max_train_origins,
            official_max_train_origins=args.official_max_train_origins,
            seed=args.seed, splits=args.splits, params=dict(XGB_PARAMS),
            lease_wait_s=args.lease_wait_s,
            include_official=not args.no_official)
    except Exception as error:  # budget failure: incomplete status, no fabrications
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "status.json").write_text(json.dumps(
            {"status": "incomplete", "extension": EXTENSION_LABEL,
             "error": str(error)}, indent=2, sort_keys=True,
            default=str) + "\n")
        raise
    write_readme(outdir, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "splits"},
                     indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
