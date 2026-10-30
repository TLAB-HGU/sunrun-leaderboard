"""Absolute-time event diagnostics and baseline-retaining validation selection.

Exact peak timestamps are deduplicated, not clustered into physical episodes.
Matching maximizes cardinality, then minimizes timing error within 12 hours.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_PATH = Path('/home/t-lab01/sunrun/analysis/e2000fs_feat30_followups/evaluation.py')
_SPEC = importlib.util.spec_from_file_location('feat30_v2_frozen_evaluation', _PATH)
_OLD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_OLD)
frame = _OLD.frame
official_score = _OLD.official_score
json_safe = _OLD.json_safe
_OBJECTIVES = ('unique_peak_mae', 'unique_ramp6_mae', 'unique_timing_mae_h')


def _catalog(values, origins):
    observations = {}
    for row, origin in zip(values, origins):
        for peak in _OLD.load_scorer()._events(row):
            hour = int(origin.value // 3_600_000_000_000) + peak + 1
            ends = np.arange(max(6, peak - 12), peak + 1)
            ramp = float(max(0., np.max(row[ends] - row[ends - 6]))) if len(ends) else None
            observations.setdefault(hour, []).append((float(row[peak]), ramp))
    result = {}
    for hour, rows in observations.items():
        ramps = [r for _, r in rows if r is not None]
        result[hour] = dict(peak=float(np.median([p for p, _ in rows])),
                            ramp6=float(np.median(ramps)) if ramps else None,
                            windows=len(rows))
    return result


def metrics(y, p, origins):
    """Old all-window diagnostics plus deduplicated matched-event errors.

Event amplitude is the median of forecasts sharing an exact peak timestamp.
Ramp uses 6h increments ending in [peak-12h, peak], available within each
window; medians combine those observations. Missing ramp context stays null.
"""
    y, p = _OLD._arrays(y, p)
    origins = _OLD._origins(origins, len(y))
    result = _OLD.metrics(y, p, origins)
    true, pred = _catalog(y, origins), _catalog(p, origins)
    pairs = _OLD.load_scorer()._match(sorted(true), sorted(pred))
    peak_errors, ramp_errors, timing = [], [], []
    records = []
    for actual, forecast in pairs:
        a, b = true[actual], pred[forecast]
        peak_errors.append(abs(b['peak'] - a['peak']))
        ramp = abs(b['ramp6'] - a['ramp6']) if a['ramp6'] is not None and b['ramp6'] is not None else None
        if ramp is not None:
            ramp_errors.append(ramp)
        timing.append(abs(forecast - actual))
        records.append(dict(true_peak_utc=pd.Timestamp(actual, unit='h', tz='UTC').isoformat(),
                            pred_peak_utc=pd.Timestamp(forecast, unit='h', tz='UTC').isoformat(),
                            true=a, pred=b, ramp6_ae=ramp))
    tp, fp, fn = len(pairs), len(pred) - len(pairs), len(true) - len(pairs)
    result.update(unique_tp=tp, unique_fp=fp, unique_fn=fn,
                  unique_csi=tp / (tp + fp + fn) if tp + fp + fn else None,
                  unique_peak_mae=float(np.mean(peak_errors)) if tp else None,
                  unique_ramp6_mae=float(np.mean(ramp_errors)) if len(ramp_errors) == tp and tp else None,
                  unique_timing_mae_h=float(np.mean(timing)) if tp else None,
                  unique_events=dict(true_count=len(true), pred_count=len(pred), matches=records))
    return result


def aggregate_folds(folds):
    """Aggregate disjoint folds; event errors weighted by matched-event count."""
    if not folds:
        raise ValueError('At least one fold is required')
    n = sum(f['n_origins'] for f in folds)
    if n <= 0:
        raise ValueError('Folds must contain origins')
    result = dict(n_origins=n, mse=sum(f['mse'] * f['n_origins'] for f in folds) / n)
    for key in ('unique_tp', 'unique_fp', 'unique_fn'):
        result[key] = sum(f[key] for f in folds)
    total = sum(result[k] for k in ('unique_tp', 'unique_fp', 'unique_fn'))
    result['unique_csi'] = result['unique_tp'] / total if total else None
    matched = [f for f in folds if f['unique_tp']]
    for key in _OBJECTIVES:
        result[key] = (sum(f[key] * f['unique_tp'] for f in matched) / result['unique_tp']
                       if matched and all(f[key] is not None and np.isfinite(f[key]) for f in matched) else None)
    for key in ('peak_mae', 'ramp6_mae', 'peak_timing_mae_h', 'mae'):
        if all(key in f for f in folds):
            result[key] = sum(f[key] * f['n_origins'] for f in folds) / n
    return result


def selection_rank(candidate, parent):
    """Return comparable rank, or None for missing events or >3% MSE loss."""
    for entry in (candidate, parent):
        if not entry.get('unique_tp', 0):
            return None
        for key in (*_OBJECTIVES, 'mse', 'unique_csi'):
            value = entry.get(key)
            if value is None or not np.isfinite(value) or value < 0:
                return None
    if candidate['mse'] > parent['mse'] * 1.03:
        return None
    score = float(np.mean([candidate[k] / max(parent[k], 1e-12) for k in _OBJECTIVES]))
    return score, -candidate['unique_csi'], candidate['mse']


selection = selection_rank


def choose_candidate(candidates, parent_key='parent'):
    """Keep the baseline on invalid evidence or ties; never force an arm."""
    parent = candidates[parent_key]
    best, rank = parent_key, selection_rank(parent, parent)
    if rank is None:
        return best
    for name, candidate in candidates.items():
        candidate_rank = selection_rank(candidate, parent)
        if candidate_rank is not None and candidate_rank < rank:
            best, rank = name, candidate_rank
    return best
