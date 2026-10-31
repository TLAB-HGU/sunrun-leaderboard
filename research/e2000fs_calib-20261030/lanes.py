"""Post-processing lanes selected from the 2026-10-30 bottleneck literature review.

Three lanes, each a separate tmux session / pane:

  offset  causal rolling horizon offset      (arXiv:2501.14805 adaptation)
  pool    body/tail linear pooling           (arXiv:2407.15900 adaptation)
  pbo     probability of backtest overfitting over archived arms (KBS 2024 / arXiv:2402.08672)

Rules enforced here:
  * selection uses ONLY internal stages (C / D1-D3, targets before F1). Official F1/F2 scores
    are reported after the fact and never used to pick a hyperparameter.
  * every offset uses residuals whose target time is <= the origin time (72 h maturity), so no
    future observation can enter the correction.
  * all inputs are archived, hash-pinned forecasts; nothing is retrained in these lanes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/t-lab01/sunrun')
SPEED = ROOT / 'data-v2/model-inputs/leaderboard-store/e2000fs-lag0/speed.parquet'
SPECTRAL = ROOT / 'research/e2000fs_budget_spectral/results'
ARMS = {
    'spectral': SPECTRAL / 'forecast.npz',
    'dirmo_24_feat30': ROOT / 'research/e2000fs_dirmo_24_feat30/results/forecast.npz',
    'pointweight4': ROOT / 'research/e2000fs_dirmo_24_feat30_pointweight4/results/forecast.npz',
    'lgbm_extreme4': ROOT / 'research/e2000fs_feat30_lgbm_extreme4/results/forecast.npz',
    'lgbm_HPO': ROOT / 'research/e2000fs_dirmo_24_feat30_lgbm_HPO/results/forecast.npz',
    'dirmo_24': ROOT / 'research/e2000fs_dirmo_24/results/forecast.npz',
    'mimo': ROOT / 'research/e2000fs_mimo/results/forecast.npz',
    'recursive': ROOT / 'research/e2000fs_recursive/results/forecast.npz',
    'rectify': ROOT / 'research/e2000fs_rectify/results/forecast.npz',
    'last24': ROOT / 'research/e2000fs_dirmo_24_feat30_pointweight4_last24/results/forecast.npz',
    'last48': ROOT / 'research/e2000fs_dirmo_24_feat30_pointweight4_last48/results/forecast.npz',
}
H = 72


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_forecast(path: Path) -> tuple[pd.DatetimeIndex, np.ndarray]:
    z = np.load(path)
    origins = pd.DatetimeIndex(z['origins_ns'], tz='UTC')
    pred = z['predictions'].astype(np.float64)
    assert pred.shape == (len(origins), H)
    return origins, pred


def truth_matrix(origins: pd.DatetimeIndex, speed: pd.Series) -> np.ndarray:
    times = origins.values[:, None] + np.arange(1, H + 1).astype('timedelta64[h]')[None, :]
    y = speed.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, H)
    if np.isnan(y).any():
        raise ValueError('official window has unobserved targets')
    return y


def load_speed() -> pd.Series:
    return pd.read_parquet(SPEED)['speed'].sort_index()


def score(pred: np.ndarray, y: np.ndarray, origins: pd.DatetimeIndex, n_f1: int) -> dict:
    err = pred - y
    se = err ** 2
    hi = y >= 550.
    f1 = slice(0, n_f1)
    f2 = slice(n_f1, len(y))
    out = {
        'mse': float(se.mean()), 'mse_f1': float(se[f1].mean()), 'mse_f2': float(se[f2].mean()),
        'mse_ge550': float(se[hi].mean()), 'mse_f1_ge550': float(se[f1][hi[f1]].mean()),
        'mse_f2_ge550': float(se[f2][hi[f2]].mean()),
        'bias': float(err.mean()), 'bias_lt350': float(err[y < 350].mean()),
        'bias_ge550': float(err[hi].mean()),
        'n_pairs': int(se.size), 'n_origins': int(len(origins)),
        'first_origin': str(origins[0]), 'n_f1': int(n_f1),
    }
    return out


def load_internal(arm: str = 'spectral') -> dict[str, tuple[pd.DatetimeIndex, np.ndarray]]:
    """Internal stages (targets before F1) used for hyperparameter selection."""
    out = {}
    for stage in ('D1', 'D2', 'D3', 'C'):
        p = SPECTRAL / 'runs/full_v1' / f'{stage}_{arm}_predictions.npz'
        if p.exists():
            z = np.load(p)
            out[stage] = (pd.DatetimeIndex(z['origins'], tz='UTC'), z['pred'].astype(np.float64))
    if not out:
        raise FileNotFoundError('no internal stage predictions found')
    return out


def causal_offsets(pred: np.ndarray, y: np.ndarray, window: int, alpha: float) -> np.ndarray:
    """Offset for (i, h) from residuals whose target index s+h <= i (72 h maturity)."""
    resid = y - pred
    n = len(pred)
    idx = np.arange(n)
    off = np.zeros_like(pred)
    # cumulative sums over origin index for the residual of each horizon on its own diagonal
    for h in range(H):
        r = resid[:, h]
        # residual for origin s becomes available at s + h + 1 (target 1-based hour h)
        avail = np.concatenate([[0.], np.cumsum(r)])       # avail[k] = sum r[:k]
        cum_k = np.minimum(idx - h, n)                      # usable origins s <= i - h
        cum_k = np.clip(cum_k, 0, n)
        cum_lo = np.clip(cum_k - window, 0, n)
        total = avail[cum_k] - avail[cum_lo]
        count = np.maximum(cum_k - cum_lo, 0)
        mean = np.divide(total, count, out=np.zeros(n), where=count > 0)
        off[:, h] = alpha * mean
    return off


def lane_offset(args) -> dict:
    speed = load_speed()
    base_path = ARMS[args.base]
    origins, pred = read_forecast(base_path)
    y = truth_matrix(origins, speed)
    n_f1 = int((origins < pd.Timestamp('2026-05-29', tz='UTC')).sum())

    # --- selection on internal stages only ---
    internal = load_internal(args.base)
    grid = []
    for window in (168, 336, 720):
        for alpha in (0.25, 0.5, 0.75, 1.0):
            rows, ok = {}, True
            for stage, (o, p) in internal.items():
                yi = truth_matrix(o, speed)
                c = p + causal_offsets(p, yi, window, alpha)
                hi = yi >= 550.
                lo = yi < 350.
                base = {'mse': float(((p - yi) ** 2).mean()),
                        'ge550': float((((p - yi) ** 2)[hi]).mean()),
                        'lt350_bias': float((p - yi)[lo].mean())}
                cand = {'mse': float(((c - yi) ** 2).mean()),
                        'ge550': float((((c - yi) ** 2)[hi]).mean()),
                        'lt350_bias': float((c - yi)[lo].mean())}
                # preregistered 3 conditions, evaluated per internal stage
                stage_ok = (cand['mse'] < base['mse']) and (cand['ge550'] <= base['ge550']) \
                    and (abs(cand['lt350_bias']) < abs(base['lt350_bias']))
                ok = ok and stage_ok
                rows[stage] = {'base': base, 'candidate': cand,
                               'relative_gain': 1 - cand['mse'] / base['mse'],
                               'stage_passes_3_conditions': bool(stage_ok)}
            grid.append({'window': window, 'alpha': alpha, 'stages': rows,
                         'passes_all_stages': bool(ok),
                         'mean_relative_gain': float(np.mean([r['relative_gain'] for r in rows.values()]))})
    passing = [g for g in grid if g['passes_all_stages']]
    best = max(passing, key=lambda g: g['mean_relative_gain']) if passing else \
        max(grid, key=lambda g: g['mean_relative_gain'])
    all_positive = bool(best['passes_all_stages'])

    # --- apply to official window (warm start from the C stage residuals) ---
    cand_origins, cand_pred = origins, pred.copy()
    cstage = internal.get('C')
    if cstage is not None:
        o_c, p_c = cstage
        keep = o_c < origins[0]
        warm_o = o_c[keep].append(origins)
        warm_p = np.vstack([p_c[keep], pred])
        warm_y = truth_matrix(warm_o, speed)
        warm = warm_p + causal_offsets(warm_p, warm_y, best['window'], best['alpha'])
        cand_pred = warm[len(o_c[keep]):]
    else:
        cand_pred = pred + causal_offsets(pred, y, best['window'], best['alpha'])
    cand_pred = np.clip(cand_pred, 1., 3000.)

    result = {
        'lane': 'offset', 'base': args.base, 'n_f1': n_f1,
        'selection': {'grid': grid, 'chosen': {'window': best['window'], 'alpha': best['alpha']},
                      'internal_all_stages_improve': bool(all_positive)},
        'base_scores': score(pred, y, origins, n_f1),
        'candidate_scores': score(cand_pred, y, origins, n_f1),
        'inputs': {str(base_path): sha(base_path), str(SPEED): sha(SPEED)},
        'discipline': 'offsets use only residuals with target time <= origin (72h maturity); '
                      'hyperparameters chosen on D1-D3/C internal stages, never on F1/F2',
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'forecast.npz', origins_ns=origins.asi8, predictions=cand_pred)
    (out / 'results.json').write_text(json.dumps(result, indent=1))
    return result


def causal_pool_weights(a: np.ndarray, b: np.ndarray, y: np.ndarray, lag: int = 72) -> np.ndarray:
    """Per-origin, per-band weight on the tail arm from strictly matured past origins."""
    n = len(a)
    d = b - a
    num = np.cumsum((d * (y - a)).mean(axis=1))       # sum over past origins
    den = np.cumsum((d * d).mean(axis=1))
    idx = np.arange(n) - lag
    idx = np.clip(idx, -1, n - 1)
    out = np.zeros(n)
    for i in range(n):
        k = idx[i]
        if k < 0:
            out[i] = 0.0
        else:
            out[i] = num[k] / den[k] if den[k] > 0 else 0.0
    return np.clip(out, 0.0, 1.0)


def lane_pool(args) -> dict:
    speed = load_speed()
    o_a, p_a = read_forecast(ARMS[args.body])
    o_b, p_b = read_forecast(ARMS[args.tail])
    if not np.array_equal(o_a.asi8, o_b.asi8):
        raise ValueError('origin grids differ')
    y = truth_matrix(o_a, speed)
    n_f1 = int((o_a < pd.Timestamp('2026-05-29', tz='UTC')).sum())
    bands = {'h1-24': slice(0, 24), 'h25-48': slice(24, 48), 'h49-72': slice(48, 72)}

    cand = p_a.copy()
    weight_summary = {}
    for name, sl in bands.items():
        w = causal_pool_weights(p_a[:, sl], p_b[:, sl], y[:, sl], lag=72)
        cand[:, sl] = (1 - w)[:, None] * p_a[:, sl] + w[:, None] * p_b[:, sl]
        active = w[len(w) // 2:]
        weight_summary[name] = {'mean_weight_second_half': float(active.mean()),
                                'min': float(w.min()), 'max': float(w.max())}
    cand = np.clip(cand, 1., 3000.)
    result = {
        'lane': 'pool', 'body': args.body, 'tail': args.tail, 'n_f1': n_f1,
        'selection': {'band_weights': weight_summary,
                      'rule': 'walk-forward least squares on matured past origins (lag 72 h), clipped [0,1]'},
        'base_scores': score(p_a, y, o_a, n_f1),
        'tail_scores': score(p_b, y, o_a, n_f1),
        'candidate_scores': score(cand, y, o_a, n_f1),
        'inputs': {str(ARMS[args.body]): sha(ARMS[args.body]), str(ARMS[args.tail]): sha(ARMS[args.tail])},
        'discipline': 'weights at origin i use only origins with all 72 targets matured before i; '
                      'no F1/F2 score enters the weight',
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'forecast.npz', origins_ns=o_a.asi8, predictions=cand)
    (out / 'results.json').write_text(json.dumps(result, indent=1))
    return result


def lane_pbo(args) -> dict:
    speed = load_speed()
    arms, per_origin = {}, {}
    for name in args.arms.split(','):
        path = ARMS[name.strip()]
        if not path.exists():
            continue
        o, p = read_forecast(path)
        y = truth_matrix(o, speed)
        arms[name] = str(path)
        per_origin[name] = ((p - y) ** 2).mean(axis=1)
    names = list(per_origin)
    if len(names) < 3:
        raise SystemExit(f'need >=3 archived arms, found {names}')
    M = np.vstack([per_origin[n] for n in names])          # (n_arms, n_origins)
    n = M.shape[1]
    blocks = np.array_split(np.arange(n), args.blocks)
    stats, ranks = [], []
    for is_idx in combinations(range(len(blocks)), len(blocks) // 2):
        is_orig = np.concatenate([blocks[i] for i in is_idx])
        oos_orig = np.concatenate([blocks[i] for i in range(len(blocks)) if i not in is_idx])
        is_mse = M[:, is_orig].mean(axis=1)
        oos_mse = M[:, oos_orig].mean(axis=1)
        best = int(np.argmin(is_mse))
        order = np.argsort(np.argsort(oos_mse))          # 0 = best OOS
        rank = int(order[best])
        stats.append({'is_best': names[best], 'oos_rank': rank, 'n_arms': len(names)})
        ranks.append(rank)
    ranks = np.array(ranks)
    pbo = float((ranks > (len(names) - 1) / 2).mean())
    result = {
        'lane': 'pbo', 'arms': arms, 'n_origins': int(n), 'n_blocks': args.blocks,
        'n_splits': len(stats), 'probability_of_backtest_overfitting': pbo,
        'median_oos_rank': float(np.median(ranks) + 1), 'mean_oos_rank': float(ranks.mean() + 1),
        'arm_mean_mse': {k: float(v.mean()) for k, v in per_origin.items()},
        'splits': stats[:200],
        'rule': 'PBO = P(IS-best arm ranks below OOS median); CPCV-style contiguous blocks, no purging needed for a scoring-only diagnostic',
        'caveat': 'diagnostic of the existing arm set, not a new model; no forecast is produced',
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='lane', required=True)
    o = sub.add_parser('offset')
    o.add_argument('--base', default='spectral')
    o.add_argument('--out', required=True)
    p = sub.add_parser('pool')
    p.add_argument('--body', default='dirmo_24_feat30')
    p.add_argument('--tail', default='spectral')
    p.add_argument('--out', required=True)
    b = sub.add_parser('pbo')
    b.add_argument('--arms', default=','.join(ARMS))
    b.add_argument('--blocks', type=int, default=8)
    b.add_argument('--out', required=True)
    args = ap.parse_args()
    fn = {'offset': lane_offset, 'pool': lane_pool, 'pbo': lane_pbo}[args.lane]
    result = fn(args)
    print(json.dumps({k: v for k, v in result.items() if k != 'selection'}, indent=1)[:2500])


if __name__ == '__main__':
    main()
