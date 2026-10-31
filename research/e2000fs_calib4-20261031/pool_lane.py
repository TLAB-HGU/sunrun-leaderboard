"""e2000fs calib4: apply the tail-centred arms to e2000fs_calib2_detroute (no retraining).

Frozen design (preregistration.md): at every origin the pooling weight w on one arm is chosen from
matured residuals only -- targets whose time is at or before the origin (origin' + 72h <= origin) --
inside a trailing window of 2160 origins, minimising the trailing total MSE subject to the trailing
>=550 MSE not exceeding detroute's. The grid, the window and the guard are fixed in advance and are
never tuned on F1/F2. Every arm writes the full 357,840-row F1+F2 forecast; there is no internal gate.
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

ROOT = Path('/home/t-lab01/sunrun')
sys.path.insert(0, str(ROOT / 'leaderboard'))

H, U = 72, 550.0
F2_CUT = pd.Timestamp('2026-05-29', tz='UTC')
WINDOW = 2160
GRID = np.round(np.arange(0.0, 0.75 + 1e-9, 0.025), 4)
BASE = ROOT / 'research/e2000fs_calib2-20261031/runs/detroute_v1/forecast.npz'
ARMS = {
    'gpd': ROOT / 'research/e2000fs_calib3-20261031/runs/gpd_v1/forecast.npz',
    'era': ROOT / 'research/e2000fs_calib3-20261031/runs/era_v1/forecast.npz',
    'mixture': ROOT / 'research/e2000fs_calib2-20261031/runs/mixture_v1/forecast.npz',
    'spectral': ROOT / 'research/e2000fs_budget_spectral/results/forecast.npz',
}
SERIES = ROOT / 'data-v2/model-inputs/leaderboard-store/e2000fs-lag0/speed.parquet'
RULE = ('trailing-window total-MSE minimum subject to trailing >=550 MSE <= detroute, '
        'matured targets only (origin + 72h <= evaluation origin)')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs() -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    base = np.load(BASE)
    origins = pd.DatetimeIndex(base['origins_ns'], tz='UTC')
    series = pd.read_parquet(SERIES)['speed'].sort_index()
    times = origins.values[:, None] + np.arange(1, H + 1).astype('timedelta64[h]')[None, :]
    truth = series.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, H)
    if np.isnan(truth).any():
        raise ValueError('official window has unobserved targets')
    return origins, np.asarray(base['predictions'], dtype=np.float64), truth


def matured_counts(origins: pd.DatetimeIndex) -> np.ndarray:
    """Number of origins whose whole 72h target window is at or before each origin (72h maturity)."""
    cutoff = pd.DatetimeIndex(origins) - pd.Timedelta(hours=H)
    return np.searchsorted(origins.values, cutoff.values, side='right')


def choose_weights(se_base: np.ndarray, se_arm: np.ndarray, cross: np.ndarray,
                   high: np.ndarray, mature: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Frozen rule, one weight per origin, matured residuals only."""
    n = len(mature)
    c_b = np.vstack([np.zeros((1, H)), np.cumsum(se_base, axis=0)])
    c_a = np.vstack([np.zeros((1, H)), np.cumsum(se_arm, axis=0)])
    c_x = np.vstack([np.zeros((1, H)), np.cumsum(cross, axis=0)])
    c_bh = np.vstack([np.zeros((1, H)), np.cumsum(se_base * high, axis=0)])
    c_ah = np.vstack([np.zeros((1, H)), np.cumsum(se_arm * high, axis=0)])
    c_xh = np.vstack([np.zeros((1, H)), np.cumsum(cross * high, axis=0)])
    c_nh = np.concatenate([[0], np.cumsum(high.sum(axis=1))])
    weights = np.zeros(n)
    g = GRID[:, None, None]
    for i in range(n):
        k = int(mature[i])
        j = max(0, k - window)
        rows = k - j
        if rows < 48 or c_nh[k] - c_nh[j] < 50:      # not enough matured history or too few tail cells
            continue
        a0 = (c_b[k] - c_b[j]).sum() / (rows * H)
        b0 = (c_a[k] - c_a[j]).sum() / (rows * H)
        x0 = (c_x[k] - c_x[j]).sum() / (rows * H)
        nh = c_nh[k] - c_nh[j]
        a1 = (c_bh[k] - c_bh[j]).sum() / nh
        b1 = (c_ah[k] - c_ah[j]).sum() / nh
        x1 = (c_xh[k] - c_xh[j]).sum() / nh
        tot = (1 - g[:, 0, 0]) ** 2 * a0 + g[:, 0, 0] ** 2 * b0 + 2 * g[:, 0, 0] * (1 - g[:, 0, 0]) * x0
        tail = (1 - g[:, 0, 0]) ** 2 * a1 + g[:, 0, 0] ** 2 * b1 + 2 * g[:, 0, 0] * (1 - g[:, 0, 0]) * x1
        feasible = tail <= a1
        if feasible.any():
            weights[i] = GRID[int(np.argmin(np.where(feasible, tot, np.inf)))]
    return weights


def score(pred: np.ndarray, truth: np.ndarray, origins: pd.DatetimeIndex, n_f1: int) -> dict:
    from scorer.scoring import event_scores
    err = pred - truth
    se = err ** 2
    high = truth >= U
    f1, f2 = slice(0, n_f1), slice(n_f1, None)
    frame = pd.DataFrame({
        'origin_last_input_utc': np.repeat(origins.astype(str), H),
        'horizon_hours': np.tile(np.arange(1, H + 1), len(origins)),
        'target_kms': truth.ravel(), 'pred_kms': pred.ravel(),
        'period': np.repeat(np.where(np.arange(len(origins)) < n_f1, 'F1', 'F2'), H)})
    ev = event_scores(frame)['']
    out = {
        'mse': float(se.mean()), 'mse_f1': float(se[f1].mean()), 'mse_f2': float(se[f2].mean()),
        'mse_ge550': float(se[high].mean()),
        'mse_f1_ge550': float(se[f1][high[f1]].mean()),
        'mse_f2_ge550': float(se[f2][high[f2]].mean()),
        'bias': float(err.mean()), 'bias_lt350': float(err[truth < 350].mean()),
        'bias_ge550': float(err[high].mean()),
        'csi': ev['csi'], 'pod': ev['pod'], 'far': ev['far'],
        'timing_mae_h': ev['timing_mae_h'], 'timing_bias_h': ev['timing_bias_h'],
        'n_pairs': int(pred.size), 'n_origins': int(len(origins)), 'n_f1': int(n_f1),
    }
    out['meets_target'] = bool(out['mse'] < 3200.0 and
                              max(out['mse_ge550'], out['mse_f1_ge550'], out['mse_f2_ge550']) < 6000.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=sorted(ARMS))
    ap.add_argument('--experiment', default=None, help='submission experiment name (recorded in results)')
    ap.add_argument('--window', type=int, default=WINDOW, help='diagnostic override; the submitted design uses 2160')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    origins, base, truth = load_inputs()
    arm = np.load(ARMS[args.arm])
    if not np.array_equal(arm['origins_ns'], np.asarray(np.load(BASE)['origins_ns'])):
        raise SystemExit('arm origins differ from the base forecast')
    arm = np.asarray(arm['predictions'], dtype=np.float64)
    if args.smoke:
        keep = 96
        origins, base, truth, arm = origins[:keep], base[:keep], truth[:keep], arm[:keep]

    high = truth >= U
    mature = matured_counts(origins)
    se_base, se_arm = (base - truth) ** 2, (arm - truth) ** 2
    cross = (base - truth) * (arm - truth)

    t0 = time.perf_counter()
    w = choose_weights(se_base, se_arm, cross, high, mature, args.window)
    pred = (1 - w)[:, None] * base + w[:, None] * arm
    seconds = (time.perf_counter() - t0) / len(origins)
    if not np.isfinite(pred).all() or (pred <= 0).any() or (pred > 3000).any():
        raise SystemExit('blend produced invalid predictions')

    n_f1 = int((origins < F2_CUT).sum())
    scores = score(pred, truth, origins, n_f1)
    tag = f'{args.arm}_smoke' if args.smoke else f'{args.arm}_v1'
    out = ROOT / 'research/e2000fs_calib4-20261031/runs' / tag
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'forecast.npz', origins_ns=origins.asi8, predictions=pred)
    np.savez_compressed(out / 'weights.npz', weights=w, mature=mature)
    results = {
        'lane': 'detpool', 'arm': args.arm, 'experiment': args.experiment,
        'base': {'path': str(BASE), 'sha256': sha(BASE)},
        'arm_path': str(ARMS[args.arm]), 'arm_sha256': sha(ARMS[args.arm]),
        'window': args.window, 'grid': GRID.tolist(), 'rule': RULE,
        'discipline': 'weights use only residuals with target time <= origin (72h maturity); '
                      'rule/window/grid are fixed in preregistration.md and never tuned on F1/F2',
        'weights': {'mean': float(w.mean()), 'median': float(np.median(w)), 'p10': float(np.percentile(w, 10)),
                    'p90': float(np.percentile(w, 90)), 'share_zero': float((w == 0).mean()),
                    'first_nonzero_origin': str(origins[int(np.argmax(w > 0))]) if (w > 0).any() else None},
        'inference_seconds_per_fold': seconds,
        'official': scores,
        'inputs': {str(BASE): sha(BASE), str(ARMS[args.arm]): sha(ARMS[args.arm]), str(SERIES): sha(SERIES)},
    }
    (out / 'results.json').write_text(json.dumps(results, indent=1, default=str) + '\n')
    print(json.dumps({'arm': args.arm, 'out': str(out), **{k: scores[k] for k in
           ('mse', 'mse_f1', 'mse_f2', 'mse_ge550', 'mse_f1_ge550', 'mse_f2_ge550', 'csi')}}, indent=1))
    print(f'weights mean {w.mean():.3f} p10 {np.percentile(w, 10):.3f} p90 {np.percentile(w, 90):.3f} '
          f'zero-share {(w == 0).mean():.2f}  inference {seconds*1e6:.1f} us/origin')


if __name__ == '__main__':
    main()
