"""Frozen shared foundation for ch_breakthrough_v2 lanes.

Causal-only compact features, baselines, and replay helpers. Lanes import
from this module and MUST NOT reimplement its protocol pieces.

Causality contract: every feature row at origin ``o`` uses source rows with
timestamps ``<= o`` only. Missing values stay NaN (native XGBoost support)
alongside explicit missingness indicators. No fitted global imputers live
here; any fitted transform a lane needs MUST be fit on training data only
and serialized with its manifest.
"""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = tuple(range(1, 73))
HORIZON_HOURS = 72
MEAN_WINDOW_HOURS = 648
TAU_HOURS = 48
OUTPUT_RANGE_KMS = (1.0, 3000.0)
N_THREADS = 4
SEED = 42

ACE_HISTORY_HOURS = 72
LEASE_DIR = Path("store/ch-breakthrough-v2/gpu-leases")

# Compact CH geometry: per-threshold base columns actually present in the
# ch-hourly cache. Deliberately NOT the full 5911-column expanded set.
CH_THRESHOLDS = (0.35, 0.45, 0.55)
CH_TAGS = ("t035", "t045", "t055")
CH_BASE_COLUMNS = ("area", "top1_lon", "top1_lat", "top1_lon_width",
                   "equatorial_area")
CH_LAG_FEATURES = (0, 24)
CH_CHANGE_LAGS = (24, 72)

ACE_LAGS = (0, 1, 2, 3, 6, 12, 24, 48, 72)
ACE_DIFF_LAGS = (1, 6, 24, 48, 72)


def utc(value):
    return pd.to_datetime(value, utc=True)


def _nanstat(func, values, axis=1):
    # All-NaN windows stay NaN by design (native XGBoost missingness);
    # silence the expected empty-slice warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return func(values, axis=axis)


def _utc_index(frame, column="timestamp_utc"):
    if isinstance(frame, (pd.Series, pd.DataFrame)) and isinstance(
            frame.index, pd.DatetimeIndex):
        return pd.DatetimeIndex(utc(frame.index))
    return pd.DatetimeIndex(utc(frame[column]))


# ---------------------------------------------------------------------------
# Input loading (mirrors experiments/xgboost_suvi_fusion/run.py patterns)
# ---------------------------------------------------------------------------

def read_table(path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir() or path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def file_identity(path) -> dict:
    """SHA-256 identity of a file or directory tree (mirrors run.py)."""
    path = Path(path)
    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    size = 0
    for item in files:
        if path.is_dir():
            digest.update(str(item.relative_to(path)).encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        size += item.stat().st_size
    return {"path": str(path.resolve()), "bytes": size,
            "files": len(files), "sha256": digest.hexdigest()}


def load_ace_hourly(paths) -> pd.DataFrame:
    """Combine hourly ACE inputs in priority order; past values only."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    parts = []
    for path in paths:
        table = read_table(path)
        index = pd.to_datetime(table["timestamp_utc"], utc=True)
        column = "filled_speed_kms" if "filled_speed_kms" in table else "ace_speed_kms"
        values = pd.to_numeric(table[column], errors="coerce")
        values = values.where(np.isfinite(values) & values.gt(0))
        missing = (table["was_missing"].astype(bool) if "was_missing" in table
                   else values.isna())
        parts.append(pd.DataFrame(
            {"filled_speed_kms": values.to_numpy(),
             "was_missing": missing.to_numpy()}, index=index))
    if not parts:
        raise ValueError("at least one ACE input is required")
    series = pd.concat(parts)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if len(series) == 0 or series["filled_speed_kms"].first_valid_index() is None:
        raise ValueError("ACE inputs contain no valid speed")
    if not series.index.equals(series.index.floor("h")):
        raise ValueError("ACE timestamps must be hourly")
    series = series.loc[series["filled_speed_kms"].first_valid_index():]
    grid = pd.date_range(series.index.min(), series.index.max(), freq="h")
    series = series.reindex(grid)
    series["was_missing"] = (series["was_missing"].astype("boolean").fillna(True)
                             | series["filled_speed_kms"].isna()).astype(int)
    series["filled_speed_kms"] = series["filled_speed_kms"].ffill()
    return series.rename_axis("timestamp_utc").reset_index()


def load_ch_hourly(path) -> pd.DataFrame:
    """Load the hourly CH geometry cache with a UTC slot index."""
    frame = read_table(path)
    if isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.DatetimeIndex(utc(frame.index))
        frame.index.name = "slot"
    elif "slot" in frame.columns:
        frame = frame.set_index(pd.DatetimeIndex(
            utc(frame["slot"]), name="slot")).drop(columns=["slot"])
    else:
        raise ValueError("CH hourly input needs a slot DatetimeIndex or column")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("CH slot index must be unique and sorted")
    if not frame.index.equals(frame.index.floor("h")):
        raise ValueError("CH slots must be hourly boundaries")
    return frame.sort_index()


def load_raw_ace(path=None):
    """Optional raw ACE table (status/density extras). Returns None when absent."""
    if path is None:
        return None
    table = read_table(path)
    if "timestamp_utc" in table.columns:
        table = table.set_index(pd.DatetimeIndex(
            utc(table["timestamp_utc"]), name="timestamp_utc")).drop(
            columns=["timestamp_utc"])
    table.index = pd.DatetimeIndex(utc(table.index))
    return table.sort_index()


# ---------------------------------------------------------------------------
# Frozen feature interface
# ---------------------------------------------------------------------------

def _slope(values):
    t = np.arange(values.shape[1]) - (values.shape[1] - 1) / 2
    denom = float(t @ t)
    out = np.full(values.shape[0], np.nan)
    ok = np.isfinite(values).all(axis=1)
    out[ok] = (values[ok] @ t) / denom
    return out


def _causal_frame(source, origins, width):
    """Causal (origin-width+1 .. origin) window matrix; NaN where unknown."""
    grid = pd.date_range(origins.min() - pd.Timedelta(hours=width - 1),
                         origins.max(), freq="h")
    pos = grid.get_indexer(origins) - (width - 1)
    base = source.reindex(grid).to_numpy(dtype=float)
    return np.lib.stride_tricks.sliding_window_view(base, width)[pos]


def build_features(ace_hourly, ch_hourly, raw_ace=None) -> pd.DataFrame:
    """Compact causal features indexed by origin (UTC, hourly).

    ``ace_hourly`` has ``timestamp_utc`` / ``filled_speed_kms`` / ``was_missing``.
    ``ch_hourly`` is slot-indexed CH geometry. ``raw_ace`` is optional extra
    status/density history (same index convention) or None.

    ACE missingness contract: ``ace_missing_72`` / ``ace_age_hours`` /
    ``ace_is_missing`` derive from genuine ``was_missing`` observation validity
    (plus non-finite/non-positive speeds), never from finiteness of the
    forward-filled speed. Every window uses source rows ``<= origin`` only.
    """
    ace = ace_hourly.set_index("timestamp_utc").sort_index() \
        if "timestamp_utc" in ace_hourly.columns else ace_hourly.sort_index()
    ace.index = pd.DatetimeIndex(utc(ace.index), name="origin_last_input_utc")
    ch = ch_hourly.sort_index()
    ch.index = pd.DatetimeIndex(utc(ch.index))
    origins = ace.index.intersection(ch.index).sort_values()
    origins = origins[origins == origins.floor("h")]
    if len(origins) == 0:
        raise ValueError("ACE and CH inputs share no hourly origins")
    width = ACE_HISTORY_HOURS + 1
    speed = _causal_frame(ace["filled_speed_kms"], origins, width)
    if "was_missing" in ace.columns:
        miss_series = pd.to_numeric(ace["was_missing"], errors="coerce").fillna(1)
        miss_flag = _causal_frame(miss_series, origins, width)
        miss_flag = np.where(np.isnan(miss_flag), 1.0, miss_flag)
    else:
        miss_flag = np.where(np.isfinite(speed), 0.0, 1.0)
    # Genuine missingness: explicit was_missing flag OR invalid observation
    # (non-finite / non-positive speed). Forward-filled finite speeds with
    # was_missing==1 stay missing here.
    invalid = (~np.isfinite(speed)) | ~(speed > 0)
    genuinely_missing = (miss_flag >= 0.5) | invalid
    out = {}
    for lag in ACE_LAGS:
        out[f"ace_lag_{lag}"] = speed[:, -1 - lag]
    for lag in ACE_DIFF_LAGS:
        out[f"ace_diff_{lag}"] = speed[:, -1] - speed[:, -1 - lag]
    for w in (6, 24, 73):
        v = speed[:, -w:]
        out[f"ace_mean_{w}"] = _nanstat(np.nanmean, v)
        out[f"ace_std_{w}"] = _nanstat(np.nanstd, v)
        out[f"ace_min_{w}"] = _nanstat(np.nanmin, v)
        out[f"ace_max_{w}"] = _nanstat(np.nanmax, v)
    out["ace_slope_24"] = _slope(speed[:, -24:])
    out["ace_slope_72"] = _slope(speed[:, -72:])
    out["ace_missing_72"] = genuinely_missing[:, -72:].sum(axis=1).astype(float)
    time = np.arange(width)[None, :]
    last = np.maximum.accumulate(
        np.where(~genuinely_missing, time, -1), axis=1)
    out["ace_age_hours"] = (width - 1 - last[:, -1]).astype(float)
    out["ace_age_hours"] = np.where(last[:, -1] < 0, np.nan, out["ace_age_hours"])
    out["ace_is_missing"] = genuinely_missing[:, -1].astype(float)
    for tag, threshold in zip(CH_TAGS, CH_THRESHOLDS):
        prefix = f"ch_{tag}_"
        cols = {name: (f"{prefix}{name}" if f"{prefix}{name}" in ch.columns else None)
                for name in CH_BASE_COLUMNS}
        area_col = f"{prefix}area"
        area = (_causal_frame(ch[area_col], origins, 73)
                if area_col in ch.columns else np.full((len(origins), 73), np.nan))
        for name, col in cols.items():
            vals = (_causal_frame(ch[col], origins, 73) if col is not None
                    else np.full((len(origins), 73), np.nan))
            for lag in CH_LAG_FEATURES:
                out[f"ch_{tag}_{name}_lag{lag}"] = vals[:, -1 - lag]
            for lag in CH_CHANGE_LAGS:
                out[f"ch_{tag}_{name}_change{lag}"] = vals[:, -1] - vals[:, -1 - lag]
            out[f"ch_{tag}_{name}_mean24"] = _nanstat(np.nanmean, vals[:, -24:])
        out[f"ch_{tag}_missing24"] = np.isnan(area[:, -24:]).sum(axis=1).astype(float)
        ok = np.isfinite(area)
        seen = np.maximum.accumulate(np.where(ok, time[:, -73:], -1), axis=1)
        age = (72 - seen[:, -1]).astype(float)
        out[f"ch_{tag}_age_hours"] = np.where(seen[:, -1] < 0, np.nan, age)
    with np.errstate(invalid="ignore", divide="ignore"):
        a035 = out["ch_t035_area_lag0"]
        a055 = out["ch_t055_area_lag0"]
        out["ch_agree_ratio_055_035"] = a055 / a035
    out["ch_agree_spread_035_055"] = a035 - a055
    if raw_ace is not None and len(raw_ace):
        raw = raw_ace.sort_index()
        raw.index = pd.DatetimeIndex(utc(raw.index))
        extras = [c for c in raw.columns
                  if c not in ("filled_speed_kms", "ace_speed_kms", "was_missing")
                  and pd.api.types.is_numeric_dtype(raw[c])][:8]
        for col in extras:
            vals = _causal_frame(raw[col], origins, 25)
            out[f"raw_{col}_lag0"] = vals[:, -1]
            out[f"raw_{col}_missing24"] = np.isnan(vals[:, -24:]).sum(axis=1).astype(float)
    frame = pd.DataFrame(out, index=origins)
    frame.index.name = "origin_last_input_utc"
    return frame.astype(np.float32, errors="ignore")


# ---------------------------------------------------------------------------
# Causal mean-reversion baseline (fixed 648h window, tau 48h)
# ---------------------------------------------------------------------------

def as_speed_series(ace_hourly) -> pd.Series:
    """Forward-filled speed indexed by UTC, preserving genuine missingness.

    The returned Series carries ``attrs["observed"]`` (bool, True where the
    hour was genuinely observed: ``was_missing==0`` and finite positive
    speed) plus ``attrs["was_missing"]`` so :func:`observed_mask` recovers the
    original missingness without changing existing lane call sites.
    """
    frame = ace_hourly.set_index("timestamp_utc").sort_index() \
        if "timestamp_utc" in ace_hourly.columns else ace_hourly.sort_index()
    idx = pd.DatetimeIndex(utc(frame.index))
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
    valid = np.isfinite(vals) & (vals > 0)
    observed_arr = (~miss) & valid
    speed = pd.Series(vals, index=idx).sort_index()
    observed = pd.Series(observed_arr, index=idx).sort_index()
    was_missing = pd.Series(miss.astype(int), index=idx).sort_index()
    # Align attrs to the sorted speed index.
    speed = speed.sort_index()
    observed = observed.reindex(speed.index).fillna(False).astype(bool)
    was_missing = was_missing.reindex(speed.index).fillna(1).astype(int)
    speed.attrs["observed"] = observed
    speed.attrs["was_missing"] = was_missing
    speed.attrs["observed_index"] = observed.index[observed.to_numpy(dtype=bool)]
    return speed


def mean_reversion_predict(speed, origins, window=MEAN_WINDOW_HOURS,
                           tau=TAU_HOURS) -> pd.DataFrame:
    """Fixed mean-reversion forecast; mirrors the published baseline formula."""
    speed = speed.sort_index()
    origins = pd.DatetimeIndex(utc(origins)).sort_values()
    missing = origins.difference(speed.index)
    if len(missing):
        raise ValueError(f"series does not contain origin: {missing[0]}")
    rolling_mean = speed.rolling(window, min_periods=window).mean().reindex(origins)
    if rolling_mean.isna().any():
        raise ValueError(f"at least {window} hourly values are required before every origin")
    last = speed.reindex(origins).to_numpy(dtype=float)
    mean = rolling_mean.to_numpy(dtype=float)
    horizons = np.arange(1, HORIZON_HOURS + 1)
    decay = np.exp(-horizons / tau)
    preds = mean[:, None] + (last - mean)[:, None] * decay[None, :]
    return pd.DataFrame({
        "origin_last_input_utc": np.repeat(origins, HORIZON_HOURS),
        "horizon_hours": np.tile(horizons, len(origins)),
        "pred_kms": preds.ravel(),
    })


# ---------------------------------------------------------------------------
# XGBoost helpers (numeric only, 4 CPU threads, fail-closed device probe)
# ---------------------------------------------------------------------------

def ensure_device(device):
    from experiments.xgboost_suvi_fusion.modeling import ensure_device as _probe
    _probe(device)


def _booster_device(model, requested) -> str:
    """Actual XGBoost device from the fitted booster; never silently CPU."""
    try:
        cfg = json.loads(model.get_booster().save_config())
        actual = cfg["learner"]["generic_param"]["device"]
        if isinstance(actual, str) and actual:
            return actual
    except Exception:
        pass
    return str(requested)


def runtime_versions() -> dict:
    """Pinned runtime versions for manifests."""
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": SEED,
    }
    for name in ("pandas", "numpy", "xgboost", "scipy", "sklearn"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = "missing"
    return versions


def fit_xgb_regressor(X, y, params=None, device="cpu", seed=SEED):
    """Single deterministic XGBoost regressor; NaN stays native."""
    from xgboost import XGBRegressor
    ensure_device(device)
    X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    y = np.asarray(y, dtype=float)
    if len(X) == 0:
        raise ValueError("no training rows")
    if not np.isfinite(y).all():
        raise ValueError("training targets must be finite")
    args = dict(params or {})
    args.update(objective="reg:squarederror", tree_method="hist",
                device=device, random_state=seed, n_jobs=N_THREADS)
    args.setdefault("n_estimators", 200)
    model = XGBRegressor(**args)
    model.fit(X, y, verbose=False)
    return model


class PooledResidualForecaster:
    """Unweighted pooled residual XGBoost control over horizon features.

    Target is ``truth(origin+h) - mean_reversion(origin, h)``; the horizon
    enters as an explicit numeric feature. One estimator covers all 72
    horizons (efficiency choice recorded in manifests).

    Fit/predict separation: :meth:`fit` builds labels from future truth
    (origins + 1..72h, all ``<= cutoff``); :meth:`predict` builds only the
    feature matrix plus the causal mean-reversion anchor and never touches
    future truth, so perturbing or deleting post-origin speeds leaves
    predictions bit-identical.
    """

    def __init__(self, params=None, device="cpu", seed=SEED):
        self.params = dict(params or {})
        self.device = device
        self.seed = seed
        self.model = None
        self.columns = []
        self.cutoff = None
        self.actual_device = None
        self.n_short_history_skipped = 0

    def _build_predict_matrix(self, features, origins, speed):
        """Feature matrix + causal anchor for prediction (no future truth)."""
        origins = pd.DatetimeIndex(utc(origins)).sort_values()
        base = mean_reversion_predict(speed, origins)
        mr = base["pred_kms"].to_numpy(dtype=float)
        X = features.reindex(origins)
        missing_origins = X.index[X.isna().all(axis=1)]
        if len(missing_origins):
            raise ValueError(f"features missing origins: {missing_origins[0]}")
        big = pd.DataFrame(np.repeat(X.to_numpy(dtype=float), HORIZON_HOURS, axis=0),
                           columns=list(X.columns))
        big["horizon_hours"] = np.tile(np.arange(1, HORIZON_HOURS + 1), len(origins))
        big["horizon_scaled"] = big["horizon_hours"] / HORIZON_HOURS
        return origins, big, mr

    def _design(self, features, origins, speed):
        """Fit-time design: predict matrix plus pooled future-truth labels."""
        origins, big, mr = self._build_predict_matrix(features, origins, speed)
        base_horizons = np.tile(np.arange(1, HORIZON_HOURS + 1), len(origins))
        truth = speed.reindex(
            origins.repeat(HORIZON_HOURS)
            + pd.to_timedelta(base_horizons, unit="h")
        ).to_numpy(dtype=float)
        return origins, big, truth, mr

    def fit(self, features, speed, cutoff, max_origins=None):
        cutoff = utc(cutoff)
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
            raise ValueError("no training origins with complete labels before cutoff")
        origins, big, truth, mr = self._design(features, origins, speed)
        ok = np.isfinite(truth)
        if not ok.any():
            raise ValueError("no finite pooled training labels")
        # Unweighted control: every (origin, horizon) pair counts equally.
        self.model = fit_xgb_regressor(big.loc[ok], (truth - mr)[ok],
                                       self.params, self.device, self.seed)
        self.columns = list(big.columns)
        self.cutoff = cutoff
        self.actual_device = _booster_device(self.model, self.device)
        return self

    def predict(self, features, speed, origins) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("fit the forecaster before predicting")
        origins, big, mr = self._build_predict_matrix(features, origins, speed)
        resid = self.model.predict(big[self.columns])
        pred = np.clip(mr + resid, *OUTPUT_RANGE_KMS)
        return pd.DataFrame({
            "origin_last_input_utc": np.repeat(origins, HORIZON_HOURS),
            "horizon_hours": np.tile(np.arange(1, HORIZON_HOURS + 1), len(origins)),
            "pred_kms": pred,
        })


# ---------------------------------------------------------------------------
# GPU leases (at most 2 concurrent training processes)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def acquire_gpu_lease(lock_dir=LEASE_DIR):
    """Yield 0 or 1; holds an exclusive fcntl lock on that GPU's lock file."""
    import fcntl
    lock_dir = Path(lock_dir)
    lock_dir.mkdir(parents=True, exist_ok=True)
    handles = {}
    for gpu in (0, 1):
        handle = open(lock_dir / f"gpu{gpu}.lock", "w")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handles[gpu] = handle
            break
        except OSError:
            handle.close()
    if not handles:
        handle = open(lock_dir / "gpu0.lock", "w")
        fcntl.flock(handle, fcntl.LOCK_EX)
        handles[0] = handle
    gpu = next(iter(handles))
    try:
        yield gpu
    finally:
        for handle in handles.values():
            try:
                fcntl.flock(handle, fcntl.LOCK_UN)
            finally:
                handle.close()


# ---------------------------------------------------------------------------
# Rolling splits with >=72h purge; cutoff enforcement
# ---------------------------------------------------------------------------

def check_cutoff(feature_index, target_index, cutoff):
    """Raise if any fitted observation or label target passes the cutoff."""
    cutoff = utc(cutoff)
    feature_index = pd.DatetimeIndex(utc(feature_index))
    target_index = pd.DatetimeIndex(utc(target_index))
    if (feature_index > cutoff).any():
        raise ValueError("feature observations pass the training cutoff")
    if (target_index > cutoff).any():
        raise ValueError("label targets pass the training cutoff")


def purged_split(origins, cutoff, end=None, horizon=HORIZON_HOURS, purge=72):
    """Train origins have full label windows at/before cutoff; test starts
    >=purge hours after cutoff (causal OOF only)."""
    origins = pd.DatetimeIndex(utc(origins)).sort_values().unique()
    cutoff, end = utc(cutoff), (utc(end) if end is not None else None)
    train = origins[origins + pd.Timedelta(hours=horizon) <= cutoff]
    test = origins[origins > cutoff + pd.Timedelta(hours=purge)]
    if end is not None:
        test = test[test + pd.Timedelta(hours=horizon) <= end]
    if len(train) and (train + pd.Timedelta(hours=horizon) > cutoff).any():
        raise ValueError("train labels pass the split cutoff")
    if len(train) and len(test) and (test.min() - cutoff) < pd.Timedelta(hours=purge):
        raise ValueError("purge gap violated at split boundary")
    return train, test


def subsample_origins(origins, step="6h", max_origins=None, seed=SEED):
    """Daily/6-hourly screening subsample; record the returned policy.

    The capped draw uses integer positions with ``index.take`` so the
    UTC-aware index is never passed through ``tz_localize`` (pandas 2.3
    raises ``TypeError: Already tz-aware`` there). Seeded and reproducible.
    """
    origins = pd.DatetimeIndex(utc(origins)).sort_values().unique()
    step = {"daily": "24h", "6hourly": "6h", "6-hourly": "6h",
            "hourly": "1h"}.get(step, step)
    kept = origins[origins == origins.floor(step)]
    if max_origins is not None and len(kept) > max_origins:
        rng = np.random.default_rng(seed)
        positions = np.sort(rng.choice(len(kept), size=max_origins,
                                       replace=False))
        kept = kept.take(positions).sort_values()
    return kept, {"step": step, "requested": len(origins),
                  "kept": len(kept), "seed": seed}


# ---------------------------------------------------------------------------
# Scoring (event definition + causal forward-fill truth policy preserved)
# ---------------------------------------------------------------------------

def observed_mask(speed, index, observed=None, observed_index=None) -> pd.Series:
    """Genuine observation mask aligned to ``index``.

    Priority: explicit ``observed`` argument (Series reindexed, or array with
    ``observed_index`` / positional ``index``) > ``speed.attrs["observed"]``
    preserved by :func:`as_speed_series` > finite-value fallback. The
    fallback exists for bare Series without metadata; lane callers passing
    ``as_speed_series`` output automatically get the genuine mask, so
    imputed forward-filled hours stay ``False``.
    """
    index = pd.DatetimeIndex(utc(index))
    if observed is not None:
        if isinstance(observed, pd.Series):
            return observed.reindex(index).fillna(False).astype(bool)
        arr = np.asarray(observed, dtype=bool)
        if observed_index is not None:
            ref = pd.DatetimeIndex(utc(observed_index))
            return pd.Series(arr, index=ref).reindex(index).fillna(False).astype(bool)
        if len(arr) == len(index):
            return pd.Series(arr, index=index, dtype=bool)
        raise ValueError("observed array length must match index or supply observed_index")
    stored = getattr(speed, "attrs", {}).get("observed", None)
    if isinstance(stored, pd.Series):
        return stored.reindex(index).fillna(False).astype(bool)
    stored_idx = getattr(speed, "attrs", {}).get("observed_index", None)
    if stored_idx is not None:
        hit = pd.DatetimeIndex(utc(stored_idx))
        return pd.Series(index.isin(hit), index=index, dtype=bool)
    speed = speed.sort_index()
    finite = speed.reindex(index).notna()
    return pd.Series(np.asarray(finite), index=index, dtype=bool)


def score_forecast(predictions, target, observed, origins=None) -> dict:
    """Wrap prediction/event diagnostics; report imputed support + uncertainty."""
    from experiments.xgboost_suvi_fusion import events as _events
    from experiments.xgboost_suvi_fusion.modeling import prediction_metrics as _pm
    target = target.sort_index()
    prediction = _pm(predictions, target, observed)
    event = _events.event_metrics(predictions, target, observed,
                                  origins=origins, missing_policy="forward_fill")
    diagnostics = _events.error_diagnostics(predictions, target, observed)
    return {
        "mse": prediction["mse"],
        "mse_observed": prediction["mse_observed"],
        "peak_macro_f1": prediction["peak_macro_f1"],
        "peak_bias": prediction["peak_bias"],
        "prediction": prediction,
        "events": event,
        "diagnostics": diagnostics,
        "imputed_fraction": event.get("imputed_fraction"),
        "unique_observed_peak_timestamps":
            event.get("unique_observed_peak_timestamps"),
        "overlap_block_ci95_7day": event.get("ci95_7day_blocks"),
        "definition": event.get("definition"),
        "missing_policy": event.get("missing_policy"),
    }


# ---------------------------------------------------------------------------
# Replay manifests (identity-checked cache reuse, never existence-only)
# ---------------------------------------------------------------------------

def code_hash(path=None) -> str:
    target = Path(path) if path is not None else Path(__file__)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _caller_identity(explicit=None) -> dict | None:
    """Hash of the lane/caller source file alongside common.py."""
    candidate = Path(explicit) if explicit is not None else None
    if candidate is None:
        here = Path(__file__).resolve()
        for frame in inspect.stack():
            try:
                path = Path(frame.filename).resolve()
            except Exception:
                continue
            if path != here and path.suffix == ".py" and path.is_file():
                candidate = path
                break
    if candidate is None or not candidate.is_file():
        return None
    return {"path": str(candidate), "sha256": code_hash(candidate),
            "bytes": candidate.stat().st_size}


def code_hashes(caller_file=None) -> dict:
    hashes = {"common": {"path": str(Path(__file__).resolve()),
                         "sha256": code_hash()}}
    caller = _caller_identity(caller_file)
    if caller is not None:
        hashes["caller"] = caller
    return hashes


def prediction_identity(path) -> dict:
    """SHA-256 identity of a prediction parquet for manifests."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def write_manifest(path, *, mode, train_cutoff, params, tree_counts,
                   columns, input_identities, seed, inference_seconds_per_origin,
                   sampling=None, extra=None, caller_file=None,
                   actual_device=None, prediction_path=None,
                   prediction_sha256=None) -> dict:
    hashes = code_hashes(caller_file)
    manifest = {
        "mode": mode,
        "train_cutoff": utc(train_cutoff).isoformat(),
        "params": dict(params),
        "tree_counts": tree_counts,
        "columns": list(columns),
        "n_columns": len(list(columns)),
        "input_identities": {k: dict(v) for k, v in input_identities.items()},
        "code_hash": hashes["common"]["sha256"],
        "code_hashes": hashes,
        "code_file": str(Path(__file__).resolve()),
        "seed": seed,
        "inference_seconds_per_origin": inference_seconds_per_origin,
        "sampling": sampling or {},
        "versions": runtime_versions(),
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    if actual_device is not None:
        manifest["actual_device"] = str(actual_device)
    if prediction_path is not None:
        manifest["prediction"] = prediction_identity(prediction_path)
    elif prediction_sha256 is not None:
        manifest["prediction"] = {"sha256": str(prediction_sha256)}
    if extra:
        manifest["extra"] = dict(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True,
                               allow_nan=False, default=str) + "\n")
    return manifest


def validate_manifest_identities(manifest, current_identities,
                                 current_code_hash=None,
                                 current_columns=None):
    """Cache reuse guard: identities must match, not merely exist.

    Source fixes invalidate previous fitted/predicted artifacts: a stale
    ``code_hash`` (common) or caller hash mismatch fails validation even when
    input hashes still match.
    """
    manifest = json.loads(Path(manifest).read_text()) \
        if not isinstance(manifest, dict) else manifest
    problems = []
    for name, ident in current_identities.items():
        saved = manifest.get("input_identities", {}).get(name)
        if saved is None:
            problems.append(f"{name}: not recorded in manifest")
        elif saved.get("sha256") != ident.get("sha256"):
            problems.append(f"{name}: sha256 mismatch")
    if manifest.get("columns") is None:
        problems.append("manifest has no recorded columns")
    elif current_columns is not None and list(manifest.get("columns")) != list(current_columns):
        problems.append("columns mismatch: refit required")
    saved_hash = manifest.get("code_hash")
    expected = current_code_hash if current_code_hash is not None else code_hash()
    if saved_hash is not None and saved_hash != expected:
        problems.append("code_hash mismatch: common.py changed, refit required")
    saved_hashes = manifest.get("code_hashes") or {}
    saved_common = saved_hashes.get("common", {}).get("sha256")
    if saved_common is not None and saved_common != code_hash():
        problems.append("code_hashes.common mismatch: refit required")
    saved_caller = saved_hashes.get("caller")
    if isinstance(saved_caller, dict) and saved_caller.get("path"):
        try:
            caller_path = Path(saved_caller["path"])
            if caller_path.is_file() and code_hash(caller_path) != saved_caller.get("sha256"):
                problems.append("code_hashes.caller mismatch: lane source changed, refit required")
        except Exception:
            pass
    if problems:
        raise ValueError("manifest identity check failed: " + "; ".join(problems))
    return True


def timed_predict(predict_fn, inputs, per_origin=True):
    started = time.perf_counter()
    out = predict_fn(inputs)
    elapsed = time.perf_counter() - started
    n = len(inputs) if per_origin else max(1, len(out))
    return out, elapsed / max(1, n)
