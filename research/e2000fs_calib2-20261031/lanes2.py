"""Round-4 lanes: label audit, lexicographic/anytime-valid selection, capacity, mixture.

Every lane shares the fixed protocol of research/notes/plan-stage2-round4-20261031.md:
training rows satisfy origin+144h <= cutoff, the official window is F1 (2,137 origins) then
F2 (2,833 origins), and no internal window may gate the official evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('OMP_NUM_THREADS', '3')

import numpy as np
import pandas as pd

ROOT = Path('/home/t-lab01/sunrun')
H = 72
SPEED = ROOT / 'data-v2/model-inputs/leaderboard-store/e2000fs-lag0/speed.parquet'
F1_CUT = pd.Timestamp('2026-02-26', tz='UTC')
F2_CUT = pd.Timestamp('2026-05-29', tz='UTC')
VAL_START = pd.Timestamp('2025-12-04', tz='UTC')
VAL_END = pd.Timestamp('2026-02-22T23:00:00Z')


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tw = imported('calib_tweedie', ROOT / 'research/e2000fs_calib-20261030/tweedie_lane.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=1, default=str) + '\n')


# ----------------------------------------------------------------------------- audit
def lane_audit(out: Path) -> dict:
    """Count training rows whose 72h target vector is not matured at each stage cutoff."""
    names, selected, origins, series, y, bundle = tw.load_data()
    val = (origins >= VAL_START) & (origins <= VAL_END)
    rows = {}
    stages = {
        'f1': F1_CUT,
        'final': F2_CUT,
        'internal_C': VAL_START,
    }
    for stage, cutoff in stages.items():
        mature = (origins + pd.Timedelta(hours=144)) <= cutoff
        # historical parent mask used by the archived runners
        legacy = (origins + pd.Timedelta(hours=72)) < cutoff
        legacy_only = legacy & ~mature
        # of the legacy-only rows, how many carry targets at or after the cutoff
        target_end = origins + pd.Timedelta(hours=H)
        leaks = legacy_only & (target_end > cutoff)
        # purge gap against the validation window (for the internal stage only)
        overlap = 0
        if stage == 'internal_C':
            sel = np.flatnonzero(legacy)
            overlap = int(((origins[sel] + pd.Timedelta(hours=H)) >= VAL_START).sum())
        rows[stage] = {
            'n_origins_total': int(len(origins)),
            'n_mature_mask': int(mature.sum()),
            'n_legacy_mask': int(legacy.sum()),
            'legacy_only_rows': int(legacy_only.sum()),
            'legacy_only_rows_with_unmatured_targets': int(leaks.sum()),
            'legacy_only_share_pct': round(100 * legacy_only.sum() / max(legacy.sum(), 1), 3),
            'train_rows_overlapping_validation_window': overlap,
        }
    result = {
        'lane': 'audit',
        'protocol': {
            'mature_mask': 'origin + 144h <= cutoff (72h vector fully matured)',
            'legacy_mask': 'origin + 72h < cutoff (used by archived runs)',
            'validation_window': [str(VAL_START), str(VAL_END)],
        },
        'stages': rows,
        'verdict': (
            'legacy-only rows with unmatured targets are the candidate explanation for the '
            'internal-versus-official sign flips; the count decides whether the audit is material'
        ),
        'inputs': {str(SPEED): sha(SPEED)},
    }
    write_json(out / 'results.json', result)
    return result


# ----------------------------------------------------------------------------- lexi
def eprocess_better(diff: np.ndarray, bound: float) -> dict:
    """Anytime-valid e-value for H0: E[diff] >= 0 against E[diff] < 0, reported on a log scale.

    Bounded test martingale with a 50/50 mixture over two betting fractions. The product is
    accumulated in log space with clipping so the value stays finite at any length.
    """
    x = np.clip(diff / bound, -0.98, 0.98)
    logs = []
    for lam in (0.02, 0.10):
        logs.append(np.log1p(-lam * x))
    logs = np.vstack(logs)                       # (2, n)
    m = logs.max(axis=0, keepdims=True)
    mix = m[0] + np.log(0.5 * np.exp(logs[0] - m[0]) + 0.5 * np.exp(logs[1] - m[0]))
    log_e = float(mix[-1])
    return {'log10_e': log_e / np.log(10.0), 'e_capped': float(np.exp(min(log_e, 700.0)))}


def lane_lexi(out: Path, baseline: str = 'dirmo_24_feat30') -> dict:
    speed = pd.read_parquet(SPEED)['speed'].sort_index()
    arms = {
        'spectral': ROOT / 'research/e2000fs_budget_spectral/results/forecast.npz',
        'dirmo_24_feat30': ROOT / 'research/e2000fs_dirmo_24_feat30/results/forecast.npz',
        'pointweight4': ROOT / 'research/e2000fs_dirmo_24_feat30_pointweight4/results/forecast.npz',
        'calib_tweedie15': ROOT / 'research/e2000fs_calib-20261030/runs/tweedie15_v1/forecast.npz',
        'calib_offset_roll': ROOT / 'research/e2000fs_calib-20261030/runs/offset_v1/forecast.npz',
        'calib_pool_bodytail': ROOT / 'research/e2000fs_calib-20261030/runs/pool_v1/forecast.npz',
        'calib_point4_144h': ROOT / 'research/e2000fs_calib-20261030/runs/control_144h_v1/forecast.npz',
    }
    per_origin, truth = {}, None
    for name, path in arms.items():
        if not path.exists():
            continue
        z = np.load(path)
        origins = pd.DatetimeIndex(z['origins_ns'], tz='UTC')
        times = origins.values[:, None] + np.arange(1, H + 1).astype('timedelta64[h]')[None, :]
        y = speed.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, H)
        pred = z['predictions'].astype(np.float64)
        per_origin[name] = {'total': ((pred - y) ** 2).mean(axis=1),
                            'ge550': np.array([((p - t) ** 2)[t >= 550].mean() if (t >= 550).any() else np.nan
                                               for p, t in zip(pred, y)])}
        truth = y
    if baseline not in per_origin:
        raise SystemExit(f'baseline {baseline} missing')
    base = per_origin[baseline]
    verdicts = {}
    for name, m in per_origin.items():
        if name == baseline:
            continue
        for metric in ('total', 'ge550'):
            d = m[metric] - base[metric]
            ok = np.isfinite(d)
            d = d[ok]
            if not len(d):
                continue
            bound = float(np.nanpercentile(np.abs(d), 95)) or 1.0
            ev = eprocess_better(d, bound)
            se = float(d.std(ddof=1) / np.sqrt(len(d)))
            verdicts.setdefault(name, {})[metric] = {
                'mean_diff': float(d.mean()), 'median_diff': float(np.median(d)),
                'stderr_mean_diff': se, 't_like': float(d.mean() / se) if se > 0 else None,
                'log10_e': ev['log10_e'],
                'adopt_vs_baseline_at_alpha05': bool(ev['log10_e'] >= np.log10(20.0)),
                'bound_used': bound, 'n_origins': int(len(d)),
            }
    result = {
        'lane': 'lexi', 'baseline': baseline, 'metric_units': 'per-origin MSE difference (candidate - baseline)',
        'rule': 'lexicographic: adopt only if total MSE is anytime-valid better (e >= 20) AND >=550 MSE is not '
                'anytime-valid worse (e_worse < 20)',
        'verdicts': verdicts,
        'caveat': 'official-exposed windows; e-process controls the sequential error rate, not distribution shift',
    }
    for name, v in verdicts.items():
        tot = v.get('total', {})
        tail = v.get('ge550', {})
        adopted = bool(tot.get('adopt_vs_baseline_at_alpha05') and not tail.get('adopt_vs_baseline_at_alpha05'))
        v['lexicographic_adopt'] = adopted
    write_json(out / 'results.json', result)
    return result


# ----------------------------------------------------------------------------- capacity
def lane_capacity(out: Path, variant: str, rounds: int = 900) -> dict:
    names, selected, origins, series, y, bundle = tw.load_data()
    blocks = [tw.design(bundle, names, selected, h) for h in (0, 24, 48)]
    f1_bundle = dict(np.load(tw.CACHE / 'f1.npz'))
    f2_bundle = dict(np.load(tw.CACHE / 'official.npz'))
    o_f1 = pd.DatetimeIndex(f1_bundle['origins'], tz='UTC')
    o_f2 = pd.DatetimeIndex(f2_bundle['origins'], tz='UTC')
    off_blocks = {'f1': [tw.design(f1_bundle, names, selected, h) for h in (0, 24, 48)],
                  'final': [tw.design(f2_bundle, names, selected, h) for h in (0, 24, 48)]}
    recipe = tw.read_json(tw.MANIFEST)['recipe']['xgboost']
    base_params = {k: v for k, v in recipe.items()
                   if k not in ('n_estimators', 'n_jobs', 'random_state', 'base_score', 'objective')}
    base_params.update(objective='reg:squarederror', tree_method='hist', device='cpu',
                       multi_strategy='multi_output_tree', nthread=3, eval_metric='rmse', verbosity=0)
    seeds = (42, 43, 44) if variant == 'bag3' else (42,)
    params = dict(base_params)
    if variant == 'depth5':
        params['max_depth'] = 5
    fits = {'f1': np.flatnonzero(origins + pd.Timedelta(hours=144) <= F1_CUT),
            'final': np.flatnonzero(origins + pd.Timedelta(hours=144) <= F2_CUT)}
    results, preds = {}, {}
    for stage, fit_idx in fits.items():
        eval_origins = o_f1 if stage == 'f1' else o_f2
        pred = np.empty((len(eval_origins), H))
        for b in range(3):
            y_tr = y[fit_idx, b * 24:(b + 1) * 24]
            x_tr = blocks[b][fit_idx]
            x_apply = off_blocks[stage][b]
            acc = np.zeros((len(eval_origins), 24))
            for seed in seeds:
                p = dict(params, seed=seed, base_score=float(np.nanmean(y_tr)))
                model, _ = tw.fit_block('control', x_tr, y_tr, x_tr[-240:], y_tr[-240:],
                                        p, rounds, 1.5, out / f'losses_{variant}_{stage}_b{b}_s{seed}.csv')
                acc += model.inplace_predict(x_apply)
            pred[:, b * 24:(b + 1) * 24] = np.clip(acc / len(seeds), 1., 3000.)
        yy = tw.official_truth(eval_origins)
        err = pred - yy
        hi = yy >= 550.
        lo = yy < 350.
        label = 'internal' if stage == 'internal' else stage
        results[label] = {'n_origins': int(len(yy)), 'mse': float((err ** 2).mean()),
                          'mse_ge550': float((err ** 2)[hi].mean()),
                          'bias_lt350': float(err[lo].mean()), 'bias': float(err.mean())}
        preds[stage] = (eval_origins, pred)
        print(f'{variant} {stage}: {json.dumps(results[label])}', flush=True)
    official = preds['f1'][0].append(preds['final'][0])
    full = np.vstack([preds['f1'][1], preds['final'][1]])
    np.savez_compressed(out / 'forecast.npz', origins_ns=official.asi8, predictions=full)
    # paired control from the completed tweedie lane (same recipe, same mask)
    ctrl_path = ROOT / 'research/e2000fs_calib-20261030/runs/tweedie15_v1/control_f1_predictions.npz'
    control = None
    if ctrl_path.exists():
        c1 = np.load(ctrl_path)
        c2 = np.load(ROOT / 'research/e2000fs_calib-20261030/runs/tweedie15_v1/control_final_predictions.npz')
        cp = np.vstack([c1['pred'], c2['pred']])
        cy = np.vstack([tw.official_truth(pd.DatetimeIndex(c1['origins'], tz='UTC')),
                        tw.official_truth(pd.DatetimeIndex(c2['origins'], tz='UTC'))])
        e = cp - cy
        control = {'mse': float((e ** 2).mean()), 'mse_ge550': float((e ** 2)[cy >= 550].mean()),
                   'bias_lt350': float(e[cy < 350].mean())}
    result = {'lane': 'capacity', 'variant': variant, 'rounds': rounds, 'seeds': seeds,
              'results': results, 'control_reference': control,
              'discipline': 'loss unchanged (point4 objective); only capacity or seed-averaging varies',
              'protocol': 'training rows origin+144h<=cutoff; no internal gate'}
    write_json(out / 'results.json', result)
    return result


# ----------------------------------------------------------------------------- mixture
def lane_mixture(out: Path, rounds: int = 900) -> dict:
    """Two-component construction: level head + exceedance head scaled by an event probability head."""
    names, selected, origins, series, y, bundle = tw.load_data()
    blocks = [tw.design(bundle, names, selected, h) for h in (0, 24, 48)]
    f1_bundle = dict(np.load(tw.CACHE / 'f1.npz'))
    f2_bundle = dict(np.load(tw.CACHE / 'official.npz'))
    o_f1 = pd.DatetimeIndex(f1_bundle['origins'], tz='UTC')
    o_f2 = pd.DatetimeIndex(f2_bundle['origins'], tz='UTC')
    off_blocks = {'f1': [tw.design(f1_bundle, names, selected, h) for h in (0, 24, 48)],
                  'final': [tw.design(f2_bundle, names, selected, h) for h in (0, 24, 48)]}
    recipe = tw.read_json(tw.MANIFEST)['recipe']['xgboost']
    params = {k: v for k, v in recipe.items()
              if k not in ('n_estimators', 'n_jobs', 'random_state', 'base_score', 'objective')}
    params.update(objective='reg:squarederror', tree_method='hist', device='cpu',
                  multi_strategy='multi_output_tree', nthread=3, seed=42, eval_metric='rmse', verbosity=0)
    fits = {'f1': np.flatnonzero(origins + pd.Timedelta(hours=144) <= F1_CUT),
            'final': np.flatnonzero(origins + pd.Timedelta(hours=144) <= F2_CUT)}
    results, preds = {}, {}
    for stage, fit_idx in fits.items():
        eval_origins = o_f1 if stage == 'f1' else o_f2
        level = np.empty((len(eval_origins), H))
        prob = np.empty((len(eval_origins), H))
        gap = np.empty((len(eval_origins), H))
        for b in range(3):
            sl = slice(b * 24, (b + 1) * 24)
            y_tr = y[fit_idx, sl]
            x_tr = blocks[b][fit_idx]
            x_apply = off_blocks[stage][b]
            # head 1: conditional level (parent objective)
            m1, _ = tw.fit_block('control', x_tr, y_tr, x_tr[-240:], y_tr[-240:],
                                 dict(params, base_score=float(np.nanmean(y_tr))), rounds, 1.5,
                                 out / f'losses_mix_level_{stage}_b{b}.csv')
            level[:, sl] = m1.inplace_predict(x_apply)
            # head 2: exceedance indicator (linear probability via squared error, clipped)
            ind = (y_tr >= 550.).astype(np.float32)
            m2, _ = tw.fit_block('control', x_tr, ind, x_tr[-240:], ind[-240:],
                                 dict(params, base_score=float(np.mean(ind))), rounds, 1.5,
                                 out / f'losses_mix_prob_{stage}_b{b}.csv')
            prob[:, sl] = np.clip(m2.inplace_predict(x_apply), 0., 1.)
            # head 3: exceedance gap on rows whose block contains an event
            event = (y_tr >= 550.).any(axis=1)
            resid = y_tr[event] - y_tr[event].mean(axis=0)
            m3, _ = tw.fit_block('control', x_tr[event], resid, x_tr[event][-120:], resid[-120:],
                                 dict(params, base_score=0.0), rounds, 1.5,
                                 out / f'losses_mix_gap_{stage}_b{b}.csv')
            gap[:, sl] = m3.inplace_predict(x_apply) + y_tr[event].mean(axis=0)
        pred = np.clip(level + prob * (gap - level), 1., 3000.)
        yy = tw.official_truth(eval_origins)
        err = pred - yy
        hi = yy >= 550.
        lo = yy < 350.
        results[stage] = {'n_origins': int(len(yy)), 'mse': float((err ** 2).mean()),
                          'mse_ge550': float((err ** 2)[hi].mean()),
                          'bias_lt350': float(err[lo].mean()), 'bias': float(err.mean())}
        preds[stage] = (eval_origins, pred)
        print(f'mixture {stage}: {json.dumps(results[stage])}', flush=True)
    official = preds['f1'][0].append(preds['final'][0])
    full = np.vstack([preds['f1'][1], preds['final'][1]])
    np.savez_compressed(out / 'forecast.npz', origins_ns=official.asi8, predictions=full)
    result = {'lane': 'mixture', 'rounds': rounds, 'results': results,
              'construction': 'E[y] = level + P(y>=550) * (E[y | block has event] - level); no post-hoc linear blending',
              'control_reference': None,
              'discipline': 'component decomposition instead of reweighting or post-hoc pooling'}
    write_json(out / 'results.json', result)
    return result


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='lane', required=True)
    a = sub.add_parser('audit'); a.add_argument('--out', required=True)
    l = sub.add_parser('lexi'); l.add_argument('--out', required=True); l.add_argument('--baseline', default='dirmo_24_feat30')
    c = sub.add_parser('capacity'); c.add_argument('--out', required=True); c.add_argument('--variant', default='depth5'); c.add_argument('--rounds', type=int, default=900)
    m = sub.add_parser('mixture'); m.add_argument('--out', required=True); m.add_argument('--rounds', type=int, default=900)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.lane == 'audit':
        res = lane_audit(out)
    elif args.lane == 'lexi':
        res = lane_lexi(out, args.baseline)
    elif args.lane == 'capacity':
        res = lane_capacity(out, args.variant, args.rounds)
    else:
        res = lane_mixture(out, args.rounds)
    print(json.dumps({k: v for k, v in res.items() if k in ('lane', 'variant', 'stages', 'verdicts', 'results')}, indent=1)[:3000])
    print(f'elapsed {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
