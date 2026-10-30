"""Forecast-window event matching with explicit coverage and boundary censoring."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

KEY = ["origin_last_input_utc", "horizon_hours"]


def find_events(values, threshold=600., gap=6):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("events require finite speed")
    active = np.flatnonzero(values >= threshold)
    if not len(active):
        return [], 0
    groups = np.split(active, np.flatnonzero(np.diff(active) > gap+1)+1)
    events, censored = [], 0
    for group in groups:
        start, end = int(group[0]), int(group[-1])
        if start == 0 or end == len(values)-1:
            censored += 1
            continue
        peak = start + int(np.argmax(values[start:end+1]))
        events.append({"peak": peak, "height": float(values[peak]), "start": start, "end": end})
    return events, censored


def match_events(actual, predicted, tolerance=12):
    n, m = len(actual), len(predicted)
    if not n or not m:
        return []
    # Dummy rows/columns permit every event to remain unmatched. A bonus bigger
    # than all timing costs enforces maximum cardinality before minimum time.
    costs = np.zeros((n+m, n+m))
    costs[:n, :m] = 1e9
    bonus = (n+m+1)*(tolerance+1)
    for i, a in enumerate(actual):
        for j, p in enumerate(predicted):
            distance = abs(a["peak"]-p["peak"])
            if distance <= tolerance:
                costs[i, j] = distance-bonus
    rows, cols = linear_sum_assignment(costs)
    return [(int(i), int(j)) for i, j in zip(rows, cols)
            if i < n and j < m and costs[i, j] < 0]


def _aggregate(records):
    tp = sum(r["tp"] for r in records)
    fp = sum(r["fp"] for r in records)
    fn = sum(r["fn"] for r in records)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp/(tp+fp) if tp+fp else None,
            "recall": tp/(tp+fn) if tp+fn else None,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            "height_mae_kms": sum(r["height_abs_sum"] for r in records)/tp if tp else None,
            "time_mae_hours": sum(r["time_abs_sum"] for r in records)/tp if tp else None,
            "height_bias_kms": sum(r["height_bias_sum"] for r in records)/tp if tp else None}


def event_gate(metrics):
    return bool(metrics["unique_observed_events"] >= 10
                and metrics.get("precision") is not None and metrics["precision"] >= .60
                and metrics.get("recall") is not None and metrics["recall"] >= .60
                and metrics.get("height_mae_kms") is not None and metrics["height_mae_kms"] <= 60)


def event_metrics(predictions, target, observed, origins=None, bootstrap=200, missing_policy="forward_fill"):
    if missing_policy not in ("strict", "forward_fill"):
        raise ValueError("unknown event missingness policy")
    observed = observed.reindex(target.index).fillna(False).astype(bool)
    if missing_policy == "forward_fill":
        # User-approved causal truth filling. No interpolation or future values.
        target = target.where(observed).ffill()
    p = predictions.copy()
    p[KEY[0]] = pd.to_datetime(p[KEY[0]], utc=True)
    if p.duplicated(KEY).any():
        raise ValueError("duplicate forecast keys")
    if origins is None:
        available = pd.DatetimeIndex(p[KEY[0]].unique()).sort_values()
        origins = available[(available.hour == 0) & (available.minute == 0)]
    else:
        origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
    if origins.has_duplicates:
        raise ValueError("duplicate event origins")
    records, unique, observed_peaks = [], set(), set()
    missing = 0
    imputed_pairs = 0
    censored_actual = censored_predicted = 0
    groups = {o: g for o, g in p.groupby(KEY[0], sort=False)}
    for origin in origins:
        times = pd.date_range(origin+pd.Timedelta(hours=1), periods=72, freq="h")
        y = target.reindex(times).to_numpy(dtype=float)
        obs = observed.reindex(times).fillna(False)
        if (missing_policy == "strict" and not obs.all()) or not np.isfinite(y).all():
            missing += 1
            continue
        imputed_pairs += int((~obs).sum())
        group = groups.get(origin)
        if group is None or set(group.horizon_hours) != set(range(1, 73)):
            raise ValueError("event evaluation needs complete 1–72h predictions")
        forecast = group.sort_values("horizon_hours").pred_kms.to_numpy(dtype=float)
        actual, ac = find_events(y)
        predicted, pc = find_events(forecast)
        censored_actual += ac
        censored_predicted += pc
        pairs = match_events(actual, predicted)
        unique.update(times[a["peak"]].isoformat() for a in actual)
        observed_peaks.update(times[a["peak"]].isoformat() for a in actual if obs.iloc[a["peak"]])
        heights = [predicted[j]["height"]-actual[i]["height"] for i, j in pairs]
        timing = [abs(predicted[j]["peak"]-actual[i]["peak"]) for i, j in pairs]
        records.append({"origin": origin.isoformat(), "tp": len(pairs),
                        "fp": len(predicted)-len(pairs), "fn": len(actual)-len(pairs),
                        "height_abs_sum": float(np.abs(heights).sum()),
                        "height_bias_sum": float(np.sum(heights)), "time_abs_sum": float(np.sum(timing))})
    result = {**_aggregate(records), "unique_observed_events": len(unique),
              "requested_windows": len(origins), "evaluated_windows": len(records),
              "missing_windows": missing, "excluded_fraction": missing/len(origins) if len(origins) else None,
              "censored_actual": censored_actual, "censored_predicted": censored_predicted,
              "missing_policy": missing_policy, "imputed_pairs": imputed_pairs,
              "imputed_fraction": imputed_pairs/(72*len(records)) if records else None,
              "unique_observed_peak_timestamps": len(observed_peaks),
              "definition": "600kms; quiet gap<=6h; boundary-censored; one-to-one peak matching<=12h",
              "windows": records}
    result["passed"] = event_gate(result)
    if bootstrap and records:
        blocks = {}
        anchor = pd.Timestamp(records[0]["origin"]).floor("D")
        for r in records:
            block = int((pd.Timestamp(r["origin"])-anchor).total_seconds()//(7*86400))
            blocks.setdefault(block, []).append(r)
        groups = list(blocks.values())
        rng = np.random.default_rng(42)
        metrics = [_aggregate([r for i in rng.integers(0, len(groups), len(groups)) for r in groups[i]])
                   for _ in range(bootstrap)]
        result["ci95_7day_blocks"] = {}
        for key in ("precision", "recall", "height_mae_kms", "time_mae_hours"):
            values = [m[key] for m in metrics if m[key] is not None]
            result["ci95_7day_blocks"][key] = np.percentile(values, [2.5, 97.5]).tolist() if values else None
    return result


def error_diagnostics(predictions, target, observed):
    p = predictions.copy()
    o = pd.DatetimeIndex(pd.to_datetime(p[KEY[0]], utc=True))
    times = o+pd.to_timedelta(p.horizon_hours, unit="h")
    y = target.reindex(times).to_numpy(dtype=float)
    last = target.reindex(o).to_numpy(dtype=float)
    se = (p.pred_kms.to_numpy()-y)**2
    valid = observed.reindex(times).fillna(False).to_numpy(dtype=bool)
    # Exclusive groups: high-speed first, then large increase/decrease, then quiet.
    regime = np.where(y >= 600, "high", np.where(y-last >= 100, "rising",
                      np.where(y-last <= -100, "falling", "quiet")))
    out = {}
    for name in ("quiet", "rising", "high", "falling"):
        mask = valid & (regime == name)
        out[name] = {"count": int(mask.sum()), "mse": float(se[mask].mean()) if mask.any() else None,
                     "squared_error_fraction": float(se[mask].sum()/se[valid].sum()) if se[valid].sum() else None}
    return out
