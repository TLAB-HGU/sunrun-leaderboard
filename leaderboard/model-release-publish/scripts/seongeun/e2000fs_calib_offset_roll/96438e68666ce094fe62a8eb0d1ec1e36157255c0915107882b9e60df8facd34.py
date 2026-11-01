"""Reproducible replay for e2000fs_calib_offset_roll.

Causal rolling per-horizon offset applied to the archived e2000fs_budget_spectral official forecast. Window 336 origins, alpha 0.25, both selected on the D1-D3/C internal stages with the preregistered three-condition rule. Each offset uses only residuals whose target time is at or before the origin, so the 72 h maturity delay is respected.

Inputs are archived, hash-pinned artefacts of the sunrun monorepo; nothing is retrained.
Run:  python e2000fs_calib_offset_roll.py --out /tmp/replay
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

SPECTRAL = ROOT / 'research/e2000fs_budget_spectral/results/forecast.npz'
WARM = ROOT / 'research/e2000fs_budget_spectral/results/runs/full_v1/C_spectral_predictions.npz'
WINDOW, ALPHA = 336, 0.25          # chosen on D1-D3/C internal stages, never on F1/F2


def causal_offsets(pred, y, window, alpha):
    """Offset for (i, h) uses only residuals whose target index s + h is at or before i."""
    resid = y - pred
    n = len(pred)
    idx = np.arange(n)
    off = np.zeros_like(pred)
    for h in range(H):
        r = resid[:, h]
        avail = np.concatenate([[0.], np.cumsum(r)])
        cum_k = np.clip(np.minimum(idx - h, n), 0, n)
        cum_lo = np.clip(cum_k - window, 0, n)
        total = avail[cum_k] - avail[cum_lo]
        count = np.maximum(cum_k - cum_lo, 0)
        mean = np.divide(total, count, out=np.zeros(n), where=count > 0)
        off[:, h] = alpha * mean
    return off


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    z = np.load(SPECTRAL)
    origins = pd.DatetimeIndex(z['origins_ns'], tz='UTC')
    base = z['predictions'].astype(np.float64)
    speed = pd.read_parquet(SPEED)['speed']
    times = origins.values[:, None] + np.arange(1, H + 1).astype('timedelta64[h]')[None, :]
    y = speed.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, H)
    # warm start: prepend the archived internal C-stage origins so the first F1 origins already
    # have a matured residual window, exactly as in the recorded run
    zc = np.load(WARM)
    o_c = pd.DatetimeIndex(zc['origins'], tz='UTC')
    keep = o_c < origins[0]
    warm_o = o_c[keep].append(origins)
    warm_p = np.vstack([zc['pred'][keep], base])
    wt = warm_o.values[:, None] + np.arange(1, H + 1).astype('timedelta64[h]')[None, :]
    warm_y = speed.reindex(pd.DatetimeIndex(wt.ravel(), tz='UTC')).to_numpy().reshape(-1, H)
    warm = warm_p + causal_offsets(warm_p, warm_y, WINDOW, ALPHA)
    pred = np.clip(warm[len(o_c[keep]):], 1., 3000.)
    frame = pd.DataFrame({'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), H),
                          'horizon_hours': np.tile(np.arange(1, H + 1), len(origins)),
                          'pred_kms': pred.ravel()})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / 'predictions.parquet', index=False)
    print(len(frame), 'rows ->', out / 'predictions.parquet')


if __name__ == '__main__':
    main()
