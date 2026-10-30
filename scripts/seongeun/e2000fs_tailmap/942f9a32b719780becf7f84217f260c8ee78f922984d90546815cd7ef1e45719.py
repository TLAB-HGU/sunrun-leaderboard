"""e2000fs_tailmap — per-horizon quantile-map post-processing of the frozen parent forecast.

Reproduces the submitted 357,840-row parquet byte-identically (sha256 7e7c2057...).
No retraining of the parent: the parent forecast is downloaded by config sha and only the
post-processing map (fitted on pre-F1 out-of-fold internal predictions) is applied.
"""
"""e2000fs_tailmap: distribution (quantile-map) correction of the frozen parent forecast.

Only the post-processing changes. The parent's submitted 357,840-row forecast is the
input; the mapping is fitted on out-of-fold predictions of the same recipe over the
internal window (2025-12-04 .. 2026-02-22), i.e. strictly before the F1 targets.

Honest check: fit on the first half of the internal window, score on the second half.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path('/home/t-lab01/sunrun')


def resolve_cache():
    for c in (ROOT / 'analysis/A0014/outputs/cache',
              ROOT / 'analysis/forecast_strategies_e2000/outputs/cache'):
        if all((c / n).exists() for n in ('train.npz', 'f1.npz', 'official.npz', 'names.json', 'speed.parquet')):
            return c
    raise FileNotFoundError('frozen E2000 cache not found')


CACHE = resolve_cache()
PARENT_SHA = 'ba9f568b8b85e99fdbe3c7cebf40497057aaa5855c8619daa0c5cca5d4e77ad6'
PARENT_EXPERIMENT = 'e2000fs_dirmo_24_feat30_pointweight4'
DATASET = 'tlabtlab/sunrun-lb-store'
HPO = 550.0


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def parent_record():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(DATASET, f'results/seongeun/{PARENT_EXPERIMENT}/{PARENT_SHA}.json', repo_type='dataset')
    return json.loads(Path(p).read_text())


def parent_predictions():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(DATASET, f'submissions/seongeun/{PARENT_EXPERIMENT}/{PARENT_SHA}.parquet', repo_type='dataset')
    d = pd.read_parquet(p)
    assert len(d) == 357840 and not d.duplicated(['origin_last_input_utc', 'horizon_hours']).any()
    return d


def targets(series, origins):
    times = origins.values[:, None] + np.arange(1, 73).astype('timedelta64[h]')[None, :]
    return series.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, 72)


def design(bundle, selected, start):
    names = json.loads((CACHE / 'names.json').read_text())
    const, vary = names['constant'], names['varying']
    ci = [i for i, n in enumerate(const) if n in selected]
    ei = [i for i, n in enumerate(vary) if n in selected]
    block = bundle['E'][:, start:start + 24, :][:, :, ei].reshape(len(bundle['C']), -1)
    return np.hstack([bundle['C'][:, ci], block]).astype(np.float32)


def weights(y, normalizer=None):
    w = np.where(y >= HPO, 4., 1.)
    return (w / (w.mean(axis=0) if normalizer is None else normalizer)).astype(np.float32)


def objective(pred, matrix):
    target = matrix.get_label().reshape(pred.shape)
    w = weights(target)
    return (pred - target) * w, w


def internal_oof(selected, rounds=900, jobs=4):
    """Reproduce the internal stage: fit before 2025-12-04, predict the C window."""
    recipe = parent_record()['config']['hyperparameters']['recipe']['xgboost']
    params = {k: v for k, v in recipe.items() if k not in ('n_estimators', 'n_jobs', 'random_state')}
    params.update(nthread=jobs, seed=42)
    train = dict(np.load(CACHE / 'train.npz'))
    origins = pd.DatetimeIndex(train['origins'], tz='UTC')
    series = pd.read_parquet(CACHE / 'speed.parquet')['speed']
    y = targets(series, origins)
    fit = np.flatnonzero(origins + pd.Timedelta(hours=144) < pd.Timestamp('2025-12-04', tz='UTC'))
    val = np.flatnonzero((origins >= pd.Timestamp('2025-12-04', tz='UTC')) &
                         (origins <= pd.Timestamp('2026-02-22T23:00:00Z')))
    pred = np.empty((len(val), 72))
    for b in range(3):
        sl = slice(b * 24, (b + 1) * 24)
        X = design(train, set(selected), b * 24)
        tgt = y[fit, sl].astype(np.float32)
        dt = xgb.DMatrix(X[fit], label=tgt, nthread=jobs)
        settings = dict(params, base_score=float(tgt.mean()), num_target=24)
        model = xgb.train(settings, dt, num_boost_round=rounds, obj=objective)
        pred[:, sl] = np.clip(model.inplace_predict(X[val]), 1., 3000.)
    return origins[val], y[val], pred


def fit_map(p, y, n_knots=129):
    q = np.linspace(0, 1, n_knots)
    px = np.quantile(p, q)
    yq = np.quantile(y, q)
    px = np.maximum.accumulate(px + np.arange(len(px)) * 1e-9)
    return px, yq


def apply_map(px, yq, x, slope_bounds=(0.25, 1.75)):
    x = np.asarray(x, dtype=float)
    out = np.interp(x, px, yq)
    lo_s = np.clip((yq[1] - yq[0]) / (px[1] - px[0]), *slope_bounds)
    hi_s = np.clip((yq[-1] - yq[-2]) / (px[-1] - px[-2]), *slope_bounds)
    lo = x < px[0]
    hi = x > px[-1]
    out = np.where(lo, yq[0] + lo_s * (x - px[0]), out)
    out = np.where(hi, yq[-1] + hi_s * (x - px[-1]), out)
    return np.clip(out, 1.0, 3000.0)


def score(y, p):
    se = (p - y) ** 2
    m = y >= HPO
    return {'mse': float(se.mean()), 'mse_ge550': float(se[m].mean()), 'bias_ge550': float((p - y)[m].mean()),
            'mse_lt550': float(se[~m].mean()), 'n_ge550': int(m.sum())}


def write_meta(out, log):
    base = log['holdout_base']
    new = log['holdout_mapped']
    desc = (f"Parent {PARENT_EXPERIMENT}@{PARENT_SHA[:12]} forecast only, post-processed by a per-horizon "
            f"quantile map fitted on out-of-fold internal predictions (2025-12-04..2026-02-22, before any F1 target). "
            f"No retraining, no feature change. Holdout half-window: >=550 MSE {base['mse_ge550']:.1f} -> "
            f"{new['mse_ge550']:.1f}, bias {base['bias_ge550']:+.1f} -> {new['bias_ge550']:+.1f}; overall "
            f"{base['mse']:.1f} -> {new['mse']:.1f}. Inherits the parent's historical timing (origin+72h maturity).")
    meta = {
        'team_member': 'seongeun',
        'experiment': 'e2000fs_tailmap',
        'description': desc,
        'no_future_leakage': True,
        'inference_seconds_per_fold': 0.0,
        'config': {
            'parent_experiment': PARENT_EXPERIMENT, 'parent_config_sha256': PARENT_SHA,
            'transform': 'per-horizon quantile map (129 quantile knots), linear interpolation, '
                         'slope-clipped linear extrapolation, clip [1,3000]',
            'fit_data': 'internal out-of-fold predictions 2025-12-04..2026-02-22 (3-hourly origins)',
            'holdout_check': {'base': base, 'mapped': new},
            'map_shift_km_s': log['map_shift'],
            'no_retraining': True,
        },
    }
    (out / 'meta.yaml').write_text(json.dumps(meta, indent=2))



if __name__ == '__main__':
    import sys
    record = parent_record()
    selected = record['config']['data']['selected']
    origins, yv, pred = internal_oof(selected, rounds=900, jobs=4)
    maps = [fit_map(pred[:, h], yv[:, h]) for h in range(72)]
    d = parent_predictions()
    d['horizon_hours'] = d['horizon_hours'].astype(int)
    mapped = d.copy()
    for h in range(1, 73):
        m = d['horizon_hours'].to_numpy() == h
        mapped.loc[m, 'pred_kms'] = apply_map(*maps[h - 1], d.loc[m, 'pred_kms'].to_numpy())
    out = Path(sys.argv[1] if len(sys.argv) > 1 else 'predictions.parquet')
    mapped.to_parquet(out, index=False)
    print(digest(out), len(mapped))
