"""Reproducible replay for e2000fs_calib_pool_bodytail.

Walk-forward body/tail linear pooling of the archived e2000fs_dirmo_24_feat30 (body) and e2000fs_budget_spectral (tail) official forecasts, one weight per 24 h horizon band, fitted by least squares on past origins whose 72 targets have matured.

Inputs are archived, hash-pinned artefacts of the sunrun monorepo; nothing is retrained.
Run:  python e2000fs_calib_pool_bodytail.py --out /tmp/replay
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/home/t-lab01/sunrun')
SPEED = ROOT / 'data-v2/model-inputs/leaderboard-store/e2000fs-lag0/speed.parquet'
H = 72

BODY = ROOT / 'research/e2000fs_dirmo_24_feat30/results/forecast.npz'
TAIL = ROOT / 'research/e2000fs_budget_spectral/results/forecast.npz'
LAG = 72


def causal_weights(a, b, y, lag=LAG):
    d = b - a
    num = np.cumsum((d * (y - a)).mean(axis=1))
    den = np.cumsum((d * d).mean(axis=1))
    idx = np.clip(np.arange(len(a)) - lag, -1, len(a) - 1)
    w = np.array([num[k] / den[k] if k >= 0 and den[k] > 0 else 0.0 for k in idx])
    return np.clip(w, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    zb, zt = np.load(BODY), np.load(TAIL)
    origins = pd.DatetimeIndex(zb['origins_ns'], tz='UTC')
    assert np.array_equal(origins.asi8, zt['origins_ns'])
    body, tail = zb['predictions'].astype(np.float64), zt['predictions'].astype(np.float64)
    speed = pd.read_parquet(SPEED)['speed']
    times = origins.values[:, None] + np.arange(1, H + 1).astype('timedelta64[h]')[None, :]
    y = speed.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, H)
    pred = body.copy()
    for sl in (slice(0, 24), slice(24, 48), slice(48, 72)):
        w = causal_weights(body[:, sl], tail[:, sl], y[:, sl])[:, None]
        pred[:, sl] = (1 - w) * body[:, sl] + w * tail[:, sl]
    pred = np.clip(pred, 1., 3000.)
    frame = pd.DataFrame({'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), H),
                          'horizon_hours': np.tile(np.arange(1, H + 1), len(origins)),
                          'pred_kms': pred.ravel()})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / 'predictions.parquet', index=False)
    print(len(frame), 'rows ->', out / 'predictions.parquet')


if __name__ == '__main__':
    main()
