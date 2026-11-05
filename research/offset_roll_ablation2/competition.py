"""Lead-first competition metrics from onboarding.md and metric.ipynb.

The evaluation population is fixed independently of submitted predictions. Only
UTC 00/06/12/18 origins whose entire 72-hour forecast fits one period are scored.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "lead-timse-6h-v1"
LEADS = np.arange(1, 73)
KEY = ["origin_last_input_utc", "horizon_hours"]
PERIODS = {
    "F1": ("2026-01-01", "2026-04-01"),
    "F2": ("2026-04-01", "2026-07-01"),
    "F3": ("2026-07-01", "2026-09-30"),
}


def load_observations(path: str | Path) -> pd.DataFrame:
    """Read canonical observations or ACE raw year partitions, preserving gaps.

    Raw ACE status must be zero and speed must be finite and positive. Filled
    target-series files are deliberately not accepted as raw observations.
    """
    path = Path(path)
    files = sorted(path.glob("year=*/*.parquet")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"no observation parquet files: {path}")
    frames = []
    for file in files:
        frame = pd.read_parquet(file)
        if "speed_kms" in frame:
            frames.append(frame[["timestamp_utc", "speed_kms"]])
        else:
            speed = pd.to_numeric(frame["ace_speed_kms"], errors="raise")
            valid = frame["ace_swepam_status"].eq(0) & np.isfinite(speed) & speed.gt(0)
            frames.append(pd.DataFrame({"timestamp_utc": frame["timestamp_utc"],
                                        "speed_kms": speed.where(valid)}))
    return pd.concat(frames, ignore_index=True).sort_values("timestamp_utc").reset_index(drop=True)


def prepare_context(observations: pd.DataFrame, periods: dict | None = None) -> dict:
    """Freeze origins, unfilled truth, causal persistence and per-lead quantiles.

    ``periods`` defaults to the published F1/F2/F3 boundaries; an explicit mapping
    is useful for reproducible unit fixtures. Every period must have all 72 leads
    with at least two distinct valid truth values.
    """
    periods = PERIODS if periods is None else periods
    if set(periods) != set(PERIODS):
        raise ValueError("periods must define F1, F2 and F3")
    index = pd.DatetimeIndex(pd.to_datetime(observations["timestamp_utc"], utc=True, errors="raise"))
    if index.hasnans or not index.is_unique or not index.equals(index.floor("h")):
        raise ValueError("observations require unique hourly UTC timestamps")
    values = pd.to_numeric(observations["speed_kms"], errors="raise").to_numpy(dtype=float)
    if ((~np.isnan(values)) & (~np.isfinite(values) | (values <= 0))).any():
        raise ValueError("observations must be positive finite speeds or NaN")
    speed = pd.Series(values, index=index).sort_index()
    valid_speed = speed.dropna()
    if valid_speed.empty:
        raise ValueError("no valid observations")
    grids, labels, boundaries = [], [], []
    for name in PERIODS:
        start, end = (pd.Timestamp(t, tz="UTC") if pd.Timestamp(t).tzinfo is None
                      else pd.Timestamp(t).tz_convert("UTC") for t in periods[name])
        if start != start.floor("h") or end != end.floor("h") or end <= start:
            raise ValueError(f"{name}: invalid hourly period boundaries")
        first, last = start - pd.Timedelta(hours=1), end - pd.Timedelta(hours=73)
        origins = pd.date_range(first.ceil("6h"), last.floor("6h"), freq="6h")
        if origins.empty:
            raise ValueError(f"{name}: no complete 72-hour forecasts")
        grids.append(origins)
        labels.extend([name] * len(origins))
        boundaries.append((first, last))
    origins = grids[0].append(grids[1:])
    if not origins.is_unique or not origins.is_monotonic_increasing:
        raise ValueError("periods must be ordered and nonoverlapping")
    truth = np.column_stack([speed.reindex(origins + pd.Timedelta(hours=int(h))).to_numpy()
                             for h in LEADS])
    persistence = valid_speed.reindex(origins, method="ffill").to_numpy()
    if not (np.isfinite(persistence) & (persistence > 0)).all():
        raise ValueError("no valid observation at or before every forecast origin for persistence")
    period = np.asarray(labels)
    scopes = {}
    for name, selected in [("total", np.ones(len(origins), dtype=bool))] + [
            (name.lower(), period == name) for name in PERIODS]:
        population = truth[selected]
        leads = []
        for column, lead in enumerate(LEADS):
            valid = np.isfinite(population[:, column])
            observed = population[valid, column]
            if not len(observed):
                raise ValueError(f"{name} lead={lead}: no valid truth")
            if observed.max() == observed.min():
                raise ValueError(f"{name} lead={lead}: constant truth makes TIMSE undefined")
            quantiles = np.quantile(observed, np.arange(101) / 100, method="linear")
            order = np.argsort(observed, kind="stable")
            positions = np.searchsorted(observed[order], quantiles[:-1], side="left")
            leads.append({"valid": valid, "observed": observed, "order": order,
                          "positions": positions, "widths": np.diff(quantiles),
                          "range": float(quantiles[-1] - quantiles[0])})
        scopes[name] = {"selected": selected, "leads": leads}
    return {"origins": origins, "period": period, "truth": truth, "persistence": persistence,
            "scopes": scopes, "hourly_start": min(a for a, _ in boundaries),
            "hourly_end": max(b for _, b in boundaries)}


def align_predictions(pred: pd.DataFrame, context: dict) -> np.ndarray:
    """Validate keys globally, then select the common six-hour evaluation grid.

    Historical hourly submissions may contain boundary-crossing windows. Their
    extra hourly origins inside the evaluation envelope are ignored, while bad
    timestamps, lead values and duplicate keys are always rejected.
    """
    required = KEY + ["pred_kms"]
    if not set(required).issubset(pred.columns):
        raise ValueError(f"predictions require {required}")
    frame = pred[required].copy()
    origin = pd.to_datetime(frame[KEY[0]], utc=True, errors="raise")
    if (origin.isna().any() or not origin.equals(origin.dt.floor("h"))
            or not origin.between(context["hourly_start"], context["hourly_end"]).all()):
        raise ValueError("prediction origins must be hourly UTC timestamps inside the evaluation period")
    lead = pd.to_numeric(frame[KEY[1]], errors="raise")
    if not lead.isin(LEADS).all():
        raise ValueError("horizon_hours must be integers 1..72")
    frame[KEY[0]], frame[KEY[1]] = origin, lead.astype(int)
    if frame.duplicated(KEY).any():
        raise ValueError("duplicate (origin, horizon) prediction keys")
    frame["pred_kms"] = pd.to_numeric(frame["pred_kms"], errors="raise")
    frame = frame[frame[KEY[0]].isin(context["origins"])]
    return (frame.pivot(index=KEY[0], columns=KEY[1], values="pred_kms")
            .reindex(index=context["origins"], columns=LEADS).to_numpy(dtype=float))


@np.errstate(over="raise", invalid="raise", divide="raise")
def score_competition(pred: pd.DataFrame, context: dict) -> dict:
    """Score Total/F1/F2/F3 independently by lead, then average the 72 leads."""
    predicted = align_predictions(pred, context)
    missing = ~np.isfinite(predicted) | (predicted <= 0)
    filled = np.where(missing, context["persistence"][:, None], predicted)
    if not (np.isfinite(filled) & (filled > 0)).all():
        raise ValueError("missing predictions lack a valid causal persistence replacement")
    result = {"version": VERSION}
    for name, scope in context["scopes"].items():
        selected = scope["selected"]
        estimates, replacements = filled[selected], missing[selected]
        rows = []
        for column, population in enumerate(scope["leads"]):
            observed, valid = population["observed"], population["valid"]
            error = estimates[valid, column] - observed
            squared = error ** 2
            suffix = np.cumsum(squared[population["order"]][::-1])[::-1]
            positions = population["positions"]
            tail_mse = suffix[positions] / (len(observed) - positions)
            timse = float(np.dot(population["widths"], tail_mse) / population["range"])
            rmse, trmse = float(np.sqrt(squared.mean())), float(np.sqrt(timse))
            rows.append({"lead_hours": column + 1, "s": float(np.sqrt(rmse * trmse)),
                         "timse": timse, "rmse": rmse, "trmse": trmse,
                         "bias": float(error.mean()), "n_valid": len(observed),
                         "n_replaced": int(replacements[valid, column].sum())})
        if len(rows) != 72 or not np.isfinite(pd.DataFrame(rows).to_numpy()).all():
            raise ValueError(f"{name}: all 72 lead metrics must be finite")
        counts = [row["n_valid"] for row in rows]
        n_pairs = sum(counts)
        n_replaced = sum(row["n_replaced"] for row in rows)
        result[name] = {
            **{metric: float(np.mean([row[metric] for row in rows]))
               for metric in ("s", "timse", "rmse", "trmse")},
            "bias": float(np.average([row["bias"] for row in rows], weights=counts)),
            "n_pairs": n_pairs, "n_replaced": n_replaced,
            "replacement_fraction": n_replaced / n_pairs,
            "n_origins": int(selected.sum()), "by_lead": rows,
        }
    return result
