"""e2000fs calib5: attach every archived arm to e2000fs_calib4_detpool_gpd (no retraining).

Frozen design (preregistration.md):
  pred[i] = base[i] + sum_j beta_j * (arm_j[i] - base[i]),  beta >= 0, sum(beta) <= 1
  beta is refit every 168 origins from the trailing 2160 matured origins (target time <= refit
  origin, 72h maturity): standardised non-negative least squares with a 1e-3 ridge, then scaled
  down until sum(beta) <= 1 and until the trailing >=550 MSE no longer exceeds the base's.
  If the guard cannot be met, beta = 0 for that block. No target of a scored origin is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path('/home/t-lab01/sunrun')
sys.path.insert(0, str(ROOT / 'research/e2000fs_calib4-20261031'))
sys.path.insert(0, str(ROOT / 'leaderboard'))
from pool_lane import matured_counts, score, sha  # noqa: E402

H, U = 72, 550.0
F2_CUT = pd.Timestamp('2026-05-29', tz='UTC')
BASE = ROOT / 'research/e2000fs_calib4-20261031/runs/gpd_v1/forecast.npz'
SERIES = ROOT / 'data-v2/model-inputs/leaderboard-store/e2000fs-lag0/speed.parquet'
WINDOW = 2160
REFIT = 168
RIDGE = 1e-3
SHRINK = (1.0, 0.75, 0.5, 0.25)
EXCLUDE = {'e2000fs_calib4-20261031/runs/gpd_v1/forecast.npz'}
RULE = ('trailing-window non-negative least squares on base-relative arm increments (ridge 1e-3, '
        'sum(beta)<=1) with a trailing >=550 no-regression shrink; matured targets only, refit every 168 origins')


def discover(origins_ns: np.ndarray, base: np.ndarray) -> tuple[list[str], np.ndarray]:
    seen = {hashlib.sha256(base.tobytes()).hexdigest()[:12]}
    names, mats = [], []
    for p in sorted(ROOT.glob('research/**/forecast.npz')):
        if 'publication' in p.parts:
            continue
        rel = str(p.relative_to(ROOT)).replace('research/', '')
        if rel in EXCLUDE:
            continue
        d = np.load(p)
        if 'predictions' not in d.files or not np.array_equal(d.get('origins_ns', np.array([])), origins_ns):
            continue
        P = np.asarray(d['predictions'], dtype=np.float64)
        if P.shape != base.shape or not np.isfinite(P).all():
            continue
        h = hashlib.sha256(P.tobytes()).hexdigest()[:12]
        if h in seen:
            continue
        seen.add(h)
        names.append(rel)
        mats.append(P - base)
    return names, np.stack(mats)          # (p, n_origins, 72)


def fit_block(delta: np.ndarray, resid: np.ndarray, high: np.ndarray) -> np.ndarray:
    """NNLS on standardised base-relative increments, then the simplex cap and the tail guard."""
    rows, p = delta.shape[1], delta.shape[0]
    rms = np.sqrt((delta ** 2).mean(axis=1))
    alive = rms > 1e-6
    if not alive.any():
        return np.zeros(p)
    X = (delta[alive] / rms[alive, None]).T
    w = nnls(np.vstack([X, np.sqrt(RIDGE * rows) * np.eye(int(alive.sum()))]),
             np.concatenate([resid, np.zeros(int(alive.sum()))]), maxiter=3 * int(alive.sum()))[0]
    beta = np.zeros(p)
    beta[alive] = w / rms[alive]
    base_tail = (resid[high] ** 2).mean()
    for s in SHRINK:
        cand = beta * s
        if cand.sum() > 1.0:
            cand = cand / cand.sum()
        comb = (delta * cand[:, None]).sum(axis=0)
        if ((resid - comb)[high] ** 2).mean() <= base_tail:
            return cand
    return np.zeros(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment', default='e2000fs_calib5_detpool_all')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    z = np.load(BASE)
    origins = pd.DatetimeIndex(z['origins_ns'], tz='UTC')
    base = np.asarray(z['predictions'], dtype=np.float64)
    series = pd.read_parquet(SERIES)['speed'].sort_index()
    times = origins.values[:, None] + np.arange(1, H + 1).astype('timedelta64[h]')[None, :]
    truth = series.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, H)
    if np.isnan(truth).any():
        raise ValueError('official window has unobserved targets')

    names, delta = discover(np.asarray(z['origins_ns']), base)
    if args.smoke:
        origins, base, truth, delta = origins[:192], base[:192], truth[:192], delta[:, :192]
    n = len(origins)
    high = truth >= U
    mature = matured_counts(origins)
    resid = truth - base

    betas = np.zeros((n, len(names)))
    t0 = time.perf_counter()
    for start in range(0, n, REFIT):
        k = int(mature[start])
        j = max(0, k - WINDOW)
        if k - j < 48 or high[j:k].sum() < 50:
            continue
        b = fit_block(delta[:, j:k].reshape(len(names), -1),
                      resid[j:k].ravel(), high[j:k].ravel())
        betas[start:start + REFIT] = b
    pred = base + np.einsum('np,pnh->nh', betas, delta)
    seconds = (time.perf_counter() - t0) / n
    if not np.isfinite(pred).all() or (pred <= 0).any() or (pred > 3000).any():
        raise SystemExit('pool produced invalid predictions')

    n_f1 = int((origins < F2_CUT).sum())
    scores = score(pred, truth, origins, n_f1)
    tag = 'poolall_smoke' if args.smoke else 'poolall_v1'
    out = ROOT / 'research/e2000fs_calib5-20261031/runs' / tag
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'forecast.npz', origins_ns=origins.asi8, predictions=pred)
    np.savez_compressed(out / 'weights.npz', betas=betas, arms=np.array(names))
    share = betas.mean(axis=0)
    top = sorted(zip(names, share), key=lambda kv: -kv[1])[:10]
    results = {
        'lane': 'poolall', 'experiment': args.experiment, 'n_arms': len(names),
        'window': WINDOW, 'refit': REFIT, 'ridge': RIDGE, 'rule': RULE, 'arms': names,
        'discipline': 'refits use only residuals with target time <= refit origin (72h maturity); '
                      'arm set, window, refit cadence, ridge and shrink ladder are frozen in '
                      'preregistration.md and never tuned on F1/F2',
        'weights': {'mean_sum': float(betas.sum(axis=1).mean()),
                    'share_nonzero_blocks': float((betas.sum(axis=1) > 0).mean()),
                    'top': [{'arm': a, 'share': float(v)} for a, v in top]},
        'inference_seconds_per_fold': seconds,
        'official': scores,
        'inputs': {str(BASE): sha(BASE), str(SERIES): sha(SERIES)},
    }
    (out / 'results.json').write_text(json.dumps(results, indent=1, default=str) + '\n')
    print(json.dumps({'arms': len(names), 'out': str(out),
                      **{k: scores[k] for k in ('mse', 'mse_f1', 'mse_f2', 'mse_ge550',
                                                'mse_f1_ge550', 'mse_f2_ge550', 'csi')}}, indent=1))
    print('top arms by mean weight:')
    for a, v in top:
        print(f'  {v:+.4f}  {a}')


if __name__ == '__main__':
    main()
