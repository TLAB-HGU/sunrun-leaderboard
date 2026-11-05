"""Collect every number the e2000fs_calib4_detpool_gpd explainer shows.

The model is cheap (no retraining: 7 ms per origin end to end), so this script simply re-runs
pool_lane.choose_weights on the real inputs and dumps the intermediate quantities for one worked
origin plus the weight trajectory of all 4,970 origins.

Output: docs/explainers/data/detpool_gpd.json
Run:    /home/t-lab01/sunrun/.venv/bin/python docs/explainers/code/extract_detpool_gpd.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/t-lab01/sunrun')
OUT = ROOT / 'docs/explainers/data'
H, U = 72, 550.0
WINDOW = 2160
GRID = np.round(np.arange(0.0, 0.75 + 1e-9, 0.025), 4)

_spec = importlib.util.spec_from_file_location(
    'pool_lane', ROOT / 'research/e2000fs_calib4-20261031/pool_lane.py')
pool = importlib.util.module_from_spec(_spec)
sys.modules['pool_lane'] = pool
_spec.loader.exec_module(pool)


def grid_scan(i: int, j: int, k: int) -> tuple[list[dict], dict]:
    """MSE(w) on the frozen grid, computed from the matured trailing window of origin i."""
    se_b, se_a, cr = se_base[j:k], se_arm[j:k], cross[j:k]
    hi = high[j:k]
    rb = base[i] - truth[i]
    ra = arm[i] - truth[i]
    a0, b0, x0 = se_b.mean(), se_a.mean(), cr.mean()
    nh = int(hi.sum())
    a1, b1, x1 = (se_b * hi).sum() / nh, (se_a * hi).sum() / nh, (cr * hi).sum() / nh
    out = []
    for g in GRID:
        tot = (1 - g) ** 2 * a0 + g ** 2 * b0 + 2 * g * (1 - g) * x0
        tail = (1 - g) ** 2 * a1 + g ** 2 * b1 + 2 * g * (1 - g) * x1
        out.append({'w': float(g), 'total': float(tot), 'tail': float(tail),
                    'feasible': bool(tail <= a1),
                    'direct_total': float((((1 - g) * rb + g * ra) ** 2).mean())})
    return out, {'a0': float(a0), 'b0': float(b0), 'x0': float(x0),
                 'a1': float(a1), 'b1': float(b1), 'x1': float(x1), 'nh': nh, 'rows': int(k - j)}


def sanitize(x):
    """JSON has no NaN; turn every non-finite float into null for the browser."""
    if isinstance(x, dict):
        return {k: sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sanitize(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


if __name__ == '__main__':
    origins, base, truth = pool.load_inputs()
    arm = np.asarray(np.load(pool.ARMS['gpd'])['predictions'], dtype=np.float64)
    weights_npz = np.load(ROOT / 'research/e2000fs_calib4-20261031/runs/gpd_v1/weights.npz')
    w_stored, mature = weights_npz['weights'], weights_npz['mature']

    # recompute with the frozen rule, then confirm it equals the submitted weights bit for bit
    high = truth >= U
    mature2 = pool.matured_counts(origins)
    se_base, se_arm = (base - truth) ** 2, (arm - truth) ** 2
    cross = (base - truth) * (arm - truth)
    w = pool.choose_weights(se_base, se_arm, cross, high, mature2, WINDOW)
    pred = (1 - w)[:, None] * base + w[:, None] * arm
    assert np.array_equal(mature, mature2), 'matured counts differ from the stored ones'
    assert np.allclose(w, w_stored, atol=1e-12), 'recomputed weights differ from the submitted ones'

    results = json.loads((ROOT / 'research/e2000fs_calib4-20261031/runs/gpd_v1/results.json')
                         .read_text())
    detroute = json.loads((ROOT / 'research/e2000fs_calib2-20261031/runs/detroute_v1/results.json')
                          .read_text())
    base_res = detroute['metrics']['routed']
    official = results['official']
    n_f1 = int(results['official']['n_f1'])
    hi_all = high
    mse_base_hi = float((se_base[hi_all]).mean())
    mse_pool_hi = float((((pred - truth) ** 2)[hi_all]).mean())
    band = lambda m: {'h1_24': float(m[:24].mean()), 'h25_48': float(m[24:48].mean()),
                      'h49_72': float(m[48:72].mean())}

    # ---- worked origin 1: a mature-history cold-start origin (w has to be 0)
    m_rows = (mature2 >= 48)
    nh_all = np.concatenate([[0], np.cumsum(high.sum(axis=1))])
    nh_win = np.array([nh_all[m] - nh_all[max(0, m - WINDOW)] for m in mature2])
    gate_rows = np.flatnonzero(mature2 < 48)
    gate_tail = np.flatnonzero(m_rows & (nh_win < 50))
    cold_rows = int(gate_rows[-1])
    cold_tail = int(gate_tail[-1])
    # ---- worked origin 2: the busiest tail origin after the warm-up (w > 0, >=550 inside 72h)
    tail_cells = high.sum(axis=1)
    se_pool = (pred - truth) ** 2
    gain = se_base.sum(axis=1) - se_pool.sum(axis=1)          # >0: the blend helped this origin
    tmax = np.nanmax(truth, axis=1)
    pick_pool = np.flatnonzero((w > 0) & (tail_cells >= 6) & (tail_cells <= 40) & (gain > 0)
                               & (tmax >= 550) & (tmax <= 660) & (w >= 0.2) & (w <= 0.35))
    main = int(pick_pool[0])
    hurt = int(np.flatnonzero((w > 0) & (tail_cells >= 6)).__getitem__(
        np.argmin(gain[np.flatnonzero((w > 0) & (tail_cells >= 6))])))

    def detail(i: int) -> dict:
        k = int(mature2[i]); j = max(0, k - WINDOW)
        rows = k - j
        nh = int(high[:k].sum()) - int(high[:j].sum())
        scan, stats = grid_scan(i, j, k)
        chosen = float(w[i])
        best = next(r for r in scan if abs(r['w'] - chosen) < 1e-12)
        per_h = [{'h': int(h + 1),
                  'base': float(base[i, h]), 'arm': float(arm[i, h]),
                  'w': chosen,
                  'blend': float(pred[i, h]),
                  'truth': float(truth[i, h]) if not np.isnan(truth[i, h]) else None,
                  'resid_base': float(base[i, h] - truth[i, h]),
                  'resid_blend': float(pred[i, h] - truth[i, h])} for h in range(H)]
        return {
            'origin': str(origins[i]),
            'origin_index': int(i),
            'k': k, 'j': j, 'rows': rows, 'tail_cells': nh,
            'window_start': str(origins[j]) if rows else None,
            'window_end': str(origins[k - 1]) if rows else None,
            'matured_ok': bool(rows >= 48 and nh >= 50),
            'gate': ('rows<48' if rows < 48 else ('tail_cells<50' if nh < 50 else 'pass')),
            'stats': stats, 'scan': scan, 'chosen': chosen,
            'chosen_scan': {'total': best['total'], 'tail': best['tail'],
                            'direct_total': best['direct_total']},
            'w0_scan': scan[0], 'per_horizon': per_h,
            'truth_max': float(np.nanmax(truth[i])),
            'window_mse_base': stats['a0'], 'window_mse_arm': stats['b0'],
            'origin_mse_base': float(se_base[i].mean()), 'origin_mse_blend': float(se_pool[i].mean()),
            'origin_gain_mse': float(se_base[i].mean() - se_pool[i].mean()),
            'origin_mse_arm': float(se_arm[i].mean()),
            'share_w_zero_all': float((w == 0).mean()),
        }

    kk = int(mature2[main])          # matured origins available at the worked origin
    sample_window = {
        'origins': [str(origins[i]) for i in range(kk - 8, kk)],
        'h': 1,
        'base': [round(float(base[i, 0]), 3) for i in range(kk - 8, kk)],
        'arm': [round(float(arm[i, 0]), 3) for i in range(kk - 8, kk)],
        'truth': [round(float(truth[i, 0]), 3) for i in range(kk - 8, kk)],
    }
    out = {
        'model': 'e2000fs_calib4_detpool_gpd',
        'rule': results['rule'], 'window': WINDOW, 'grid': GRID.tolist(),
        'inputs': sorted(results['inputs']),
        'base_path': str(pool.BASE), 'arm_path': str(pool.ARMS['gpd']),
        'origins': {
            'n': int(len(origins)), 'first': str(origins[0]), 'last': str(origins[-1]),
            'step_hours': 1, 'n_f1': n_f1, 'f2_cut': '2026-05-29T00:00:00Z',
        },
        'weights': {
            'mean': float(w.mean()), 'median': float(np.median(w)), 'p10': float(np.percentile(w, 10)),
            'p90': float(np.percentile(w, 90)), 'share_zero': float((w == 0).mean()),
            'first_nonzero_origin': str(origins[int(np.argmax(w > 0))]),
            'count_nonzero': int((w > 0).sum()),
            'histogram': [{'w': float(g), 'count': int((w == g).sum())} for g in GRID if (w == g).any()],
            'series': [{'origin': str(o), 'w': float(v)} for o, v in
                       zip(origins[::5], w[::5])],
        },
        'scores': {
            'total': float(official['mse']), 'f1': official['mse_f1'], 'f2': official['mse_f2'],
            'ge550': official['mse_ge550'], 'f1_ge550': official['mse_f1_ge550'],
            'f2_ge550': official['mse_f2_ge550'], 'csi': official['csi'],
            'bias': official['bias'], 'bias_lt350': official['bias_lt350'],
            'bias_ge550': official['bias_ge550'],
            'n_pairs': official['n_pairs'], 'n_origins': official['n_origins'],
            'inference_us_per_origin': results['inference_seconds_per_fold'] * 1e6,
        },
        'scan_gain': float(0.0),
        'lineage': {'detector': detroute['metrics']['detector'],
                    'tau': detroute['tau_preregistered'],
                    'body': detroute['metrics']['body_tweedie'],
                    'tail': detroute['metrics']['tail_mixture'],
                    'routed': detroute['metrics']['routed'],
                    'arm': {'total': float(se_arm.mean()),
                            'ge550': float(se_arm[hi_all].mean())}},
        'zero_origins': int((w == 0).sum()),
        'base_scores': {
            'total': base_res['total'], 'f1': base_res['f1'], 'f2': base_res['f2'],
            'ge550': base_res['ge550'], 'f1_ge550': base_res['f1_ge550'],
            'f2_ge550': base_res['f2_ge550'],
            'csi_note': 'board table in research/e2000fs_calib4-20261031/README.md lists 0.077',
            'csi': 0.077,
        },
        'compare': {
            'base_total': float(se_base.mean()), 'pool_total': float(((pred - truth) ** 2).mean()),
            'base_ge550': mse_base_hi, 'pool_ge550': mse_pool_hi,
            'arm_total': float(se_arm.mean()), 'arm_ge550': float(se_arm[hi_all].mean()),
            'delta_total': float(((pred - truth) ** 2).mean() - se_base.mean()),
            'delta_ge550': mse_pool_hi - mse_base_hi,
            'mse_by_band_base': band(se_base.mean(axis=0)),
            'mse_by_band_pool': band(se_pool.mean(axis=0)),
            'base_f1': float(se_base[:n_f1].mean()), 'pool_f1': float(se_pool[:n_f1].mean()),
            'base_f2': float(se_base[n_f1:].mean()), 'pool_f2': float(se_pool[n_f1:].mean()),
            'origins_improved_pct': float((gain > 0).mean() * 100),
            'origins_worse_pct': float((gain < 0).mean() * 100),
            'gain_total_mse': float(-gain.mean() / H),
            'share_origins_w_gt0_improved_pct': float((gain[w > 0] > 0).mean() * 100),
        },
        'worked_counter': detail(hurt),
        'window_sample': sample_window,
        'worked_gate_rows': detail(cold_rows),
        'worked_gate_tail': detail(cold_tail),
        'worked_main': detail(main),
    }
    kk = out['worked_main']['k']
    out['scan_gain'] = out['worked_main']['w0_scan']['total'] - out['worked_main']['chosen_scan']['total']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'detpool_gpd.json').write_text(json.dumps(sanitize(out), indent=1) + '\n')
    print('weights equal to submitted:', np.array_equal(w, w_stored))
    print('gate(rows)', out['worked_gate_rows']['origin'], 'gate(tail)', out['worked_gate_tail']['origin'],
          'main', out['worked_main']['origin'], 'w', out['worked_main']['chosen'])
    print(json.dumps(out['compare'], indent=1))
    print('->', OUT / 'detpool_gpd.json')
