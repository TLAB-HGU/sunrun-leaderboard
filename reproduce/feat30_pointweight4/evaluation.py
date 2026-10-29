"""Peak/ramp diagnostics and read-only evaluation against the frozen scorer."""
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402

SCORER = Path(__file__).resolve().parent / 'scorer/scoring.py'
HORIZON = 72
KEY = ['origin_last_input_utc', 'horizon_hours']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=1)
def load_scorer():
    expected = json.loads((SCORER.parent.parent / 'config/frozen.json').read_text())['scorer_sha256']
    if digest(SCORER) != expected:
        raise ValueError('Protected scorer hash differs from config/frozen.json')
    spec = importlib.util.spec_from_file_location('feat30_frozen_scoring', SCORER)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load frozen scorer: {SCORER}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arrays(y, p):
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != HORIZON or not len(y) or p.shape != y.shape:
        raise ValueError('truth and predictions must have matching nonempty (n, 72) shapes')
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('truth and predictions must be finite')
    return y, p


def _origins(origins, n):
    result = pd.DatetimeIndex(origins)
    if result.tz is None:
        raise ValueError('origins must be timezone-aware')
    result = result.tz_convert('UTC')
    if len(result) != n or result.hasnans or not result.is_unique or not result.is_monotonic_increasing:
        raise ValueError('origins must match predictions and be unique, finite and increasing')
    if (result != result.floor('h')).any():
        raise ValueError('origins must be aligned to UTC hours')
    return result


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def per_window(y, p, origins=None):
    """All-window diagnostics; argmax timing is supplemental, not event matching."""
    y, p = _arrays(y, p)
    true_peak, pred_peak = y.max(axis=1), p.max(axis=1)
    true_ramp = np.maximum(0, (y[:, 6:] - y[:, :-6]).max(axis=1))
    pred_ramp = np.maximum(0, (p[:, 6:] - p[:, :-6]).max(axis=1))
    result = pd.DataFrame({
        'mse': ((p - y) ** 2).mean(axis=1),
        'true_peak': true_peak, 'pred_peak': pred_peak,
        'peak_error': pred_peak - true_peak,
        'peak_ae': np.abs(pred_peak - true_peak),
        'true_ramp6': true_ramp, 'pred_ramp6': pred_ramp,
        'ramp6_ae': np.abs(pred_ramp - true_ramp),
        'peak_timing_error_h': p.argmax(axis=1) - y.argmax(axis=1),
    })
    if origins is not None:
        result.insert(0, 'origin_last_input_utc', _origins(origins, len(y)).strftime('%Y-%m-%dT%H:%M:%SZ'))
    return result


def metrics(y, p, origins=None):
    """Return JSON-safe diagnostics, including the protected scorer's event rules."""
    y, p = _arrays(y, p)
    rows = per_window(y, p, origins)
    scoring = load_scorer()
    tp = fp = fn = 0
    timing = []
    for actual, forecast in zip(y, p):
        true, pred = scoring._events(actual), scoring._events(forecast)
        pairs = scoring._match(true, pred)
        tp += len(pairs)
        fp += len(pred) - len(pairs)
        fn += len(true) - len(pairs)
        timing.extend(b - a for a, b in pairs)
    events = scoring._event_summary(tp, fp, fn, timing, len(y))
    error = p - y
    high = y >= scoring.HSS_KMS
    result = {
        'n_origins': len(y), 'n_pairs': y.size,
        'mse': float(np.mean(error ** 2)), 'mae': float(np.mean(np.abs(error))),
        'peak_mae': float(rows.peak_ae.mean()), 'peak_bias': float(rows.peak_error.mean()),
        'ramp6_mae': float(rows.ramp6_ae.mean()),
        'peak_timing_mae_h': float(rows.peak_timing_error_h.abs().mean()),
        'mse_ge550': float(np.mean(error[high] ** 2)) if high.any() else None,
        'bias_ge550': float(np.mean(error[high])) if high.any() else None,
        'n_pairs_ge550': int(high.sum()),
        'mse_by_band': {f'h{start + 1}-{start + 24}': float(np.mean(error[:, start:start + 24] ** 2))
                        for start in (0, 24, 48)},
        'events': events,
        **{key: events[key] for key in ('tp', 'fp', 'fn', 'csi', 'timing_mae_h', 'timing_bias_h')},
    }
    return json_safe(result)


def selection_score(candidate, parent):
    """Equal-weight normalized peak and ramp error; lower is better."""
    values = []
    for name in ('peak_mae', 'ramp6_mae'):
        c, b = float(candidate[name]), float(parent[name])
        if not np.isfinite([c, b]).all() or c < 0 or b < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
        values.append(c / max(b, 1e-12))
    return float(np.mean(values))


def selection_key(candidate, parent):
    csi = candidate.get('csi')
    csi = float(csi) if csi is not None and np.isfinite(csi) else -np.inf
    mse = float(candidate['mse'])
    if not np.isfinite(mse) or mse < 0:
        raise ValueError('selection MSE must be finite and nonnegative')
    return selection_score(candidate, parent), -csi, mse


def frame(origins, p):
    p = np.asarray(p, dtype=np.float64)
    _arrays(p, p)
    oo = _origins(origins, len(p))
    return pd.DataFrame({
        KEY[0]: np.repeat(oo.strftime('%Y-%m-%dT%H:%M:%SZ'), HORIZON),
        KEY[1]: np.tile(np.arange(1, HORIZON + 1), len(oo)),
        'pred_kms': p.ravel(),
    })


def official_score(predictions):
    """Score with the frozen scorer against truth/folds/naive from ``DATA_ROOT/lb-store`` (hash-pinned in data_manifest.json)."""
    tables = {name: pd.read_parquet(paths.data('lb-store', f'{name}.parquet')) for name in ('truth', 'folds', 'naive')}
    scoring = load_scorer()
    errors = scoring.validate(predictions, tables['folds'])
    if errors:
        raise ValueError(errors)
    joined = predictions.merge(tables['truth'], on=KEY, validate='one_to_one')
    if len(joined) != len(predictions) or not np.isfinite(joined.target_kms).all():
        raise ValueError('Pinned truth does not cover every prediction with a finite target')
    result = scoring.score(predictions, tables['truth'], tables['folds'], tables['naive'])
    expected_mse = float(np.mean((joined.pred_kms - joined.target_kms) ** 2))
    np.testing.assert_allclose(result['mse'], expected_mse, rtol=1e-12, atol=1e-10)
    return json_safe(result)
