"""Re-fit the three heads of e2000fs_calib2_mixture2comp so the internal decomposition
(level / P(y>=550) / gap) can be shown number by number in the explainer HTML.

The archived run (research/e2000fs_calib2-20261031/runs/mixture_v1) saved only the composed
forecast plus the per-round losses; the three component arrays were never written. This script
replays the *final* stage exactly as lanes2.lane_mixture does (same blocks, same objective, same
900 rounds, same seed, nthread=3) and then

  1. writes mixture2comp_components_f2.npz -> level / prob / gap / pred for the F2 evaluation origins,
  2. checks the recomposed prediction against the archived forecast.npz,
  3. writes refit_receipt.json  -> timings, base rates, event-row counts, the comparison.

Run:  /home/t-lab01/sunrun/.venv/bin/python docs/explainers/code/refit_mixture2comp.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
os.environ.setdefault('OMP_NUM_THREADS', '3')

import numpy as np
import pandas as pd

ROOT = Path('/home/t-lab01/sunrun')
OUT = ROOT / 'docs/explainers/data'
H = 72
F2_CUT = pd.Timestamp('2026-05-29', tz='UTC')
ROUNDS = 900

_spec = importlib.util.spec_from_file_location(
    'calib_tweedie', ROOT / 'research/e2000fs_calib-20261030/tweedie_lane.py')
tw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tw)

_S = None


def shared():
    """Everything lanes2.lane_mixture prepares before the stage loop (identical calls)."""
    names, selected, origins, series, y, bundle = tw.load_data()
    blocks = [tw.design(bundle, names, selected, h) for h in (0, 24, 48)]
    f2 = dict(np.load(tw.CACHE / 'official.npz'))
    o_f2 = pd.DatetimeIndex(f2['origins'], tz='UTC')
    off = [tw.design(f2, names, selected, h) for h in (0, 24, 48)]
    fit_idx = np.flatnonzero(origins + pd.Timedelta(hours=144) <= F2_CUT)
    recipe = tw.read_json(tw.MANIFEST)['recipe']['xgboost']
    params = {k: v for k, v in recipe.items()
              if k not in ('n_estimators', 'n_jobs', 'random_state', 'base_score', 'objective')}
    params.update(objective='reg:squarederror', tree_method='hist', device='cpu',
                  multi_strategy='multi_output_tree', nthread=3, seed=42,
                  eval_metric='rmse', verbosity=0)
    return dict(names=names, selected=selected, origins=origins, y=y, blocks=blocks,
                o_f2=o_f2, off_blocks=off, fit_idx=fit_idx, params=params)


def _init():
    global _S
    _S = shared()


def _fit(job):
    head, b = job
    g = _S
    sl = slice(b * 24, (b + 1) * 24)
    y_tr = g['y'][g['fit_idx'], sl]
    x_tr, x_ap = g['blocks'][b][g['fit_idx']], g['off_blocks'][b]
    block_mean = None
    if head == 'level':
        y_lab, rows, hold = y_tr, np.arange(len(x_tr)), slice(-240, None)
        base = float(np.nanmean(y_tr))
    elif head == 'prob':
        y_lab, rows, hold = (y_tr >= 550.).astype(np.float32), np.arange(len(x_tr)), slice(-240, None)
        base = float(y_lab.mean())
    else:                                        # gap head: event rows only, demeaned
        rows = np.flatnonzero((y_tr >= 550.).any(axis=1))
        block_mean = y_tr[rows].mean(axis=0)     # per-horizon mean over event rows
        y_lab, hold = (y_tr[rows] - block_mean).astype(np.float32), slice(-120, None)
        base = 0.0
    x_fit, y_fit = x_tr[rows], y_lab
    t0 = time.perf_counter()
    model, hist = tw.fit_block('control', x_fit, y_fit, x_fit[hold], y_fit[hold],
                               dict(g['params'], base_score=base), ROUNDS, 1.5, None)
    raw = model.inplace_predict(x_ap)
    seconds = time.perf_counter() - t0
    if head == 'prob':
        raw = np.clip(raw, 0., 1.)
    elif head == 'gap':
        raw = raw + block_mean
    print(f'  {head:5s} b{b}: {seconds:6.1f}s rows={len(x_fit):5d} '
          f'valid_mse {hist["valid"][0]:9.2f} -> {hist["valid"][-1]:8.2f}', flush=True)
    return dict(head=head, block=b, seconds=seconds, raw=raw, train_rows=int(len(x_fit)),
                valid_first=float(hist['valid'][0]), valid_last=float(hist['valid'][-1]),
                train_last=float(hist['train'][-1]), block_mean=block_mean,
                hist_round=hist['round'], hist_train=hist['train'], hist_valid=hist['valid'])


def main():
    t0 = time.perf_counter()
    g = shared()
    jobs = [(h, b) for h in ('level', 'prob', 'gap') for b in (0, 1, 2)]
    with ProcessPoolExecutor(max_workers=3, initializer=_init) as ex:
        out = list(ex.map(_fit, jobs))

    n = len(g['o_f2'])
    level, prob, gap = (np.empty((n, H)) for _ in range(3))
    block_mean = np.empty((3, 24))
    fits = {}
    for r in out:
        sl = slice(r['block'] * 24, (r['block'] + 1) * 24)
        {'level': level, 'prob': prob, 'gap': gap}[r['head']][:, sl] = r['raw']
        if r['block_mean'] is not None:
            block_mean[r['block']] = r['block_mean']
        fits[f"{r['head']}_b{r['block']}"] = {
            'seconds': round(r['seconds'], 1), 'train_rows': r['train_rows'],
            'valid_mse_first': r['valid_first'], 'valid_mse_last': r['valid_last'],
            'train_mse_last': r['train_last']}
    raw_pred = level + prob * (gap - level)
    pred = np.clip(raw_pred, 1., 3000.)

    arch = np.load(ROOT / 'research/e2000fs_calib2-20261031/runs/mixture_v1/forecast.npz')
    arch_o = pd.DatetimeIndex(arch['origins_ns'], tz='UTC')
    n_f1 = int((arch_o < F2_CUT).sum())
    arch_f2 = np.asarray(arch['predictions'], dtype=np.float64)[n_f1:]
    prob_arch = np.load(ROOT / 'research/e2000fs_calib2-20261031/runs/mixture_v1/refit_prob_head.npz')
    prob_arch = np.asarray(prob_arch['prob'], dtype=np.float64)[n_f1:]
    y2 = tw.official_truth(g['o_f2'])
    d = np.abs(pred - arch_f2)
    receipt = {
        'what': 're-fit of the final stage (F2 evaluation window) heads of e2000fs_calib2_mixture2comp',
        'rounds': ROUNDS, 'nthread': 3, 'seed': 42, 'workers': 3,
        'wall_seconds': round(time.perf_counter() - t0, 1),
        'fits': fits, 'n_origins_f2': int(n),
        'gap_block_mean': block_mean.round(3).tolist(),
        'reproduction': {
            'max_abs_diff_vs_archive': float(d.max()),
            'mean_abs_diff_vs_archive': float(d.mean()),
            'share_equal_within_0.01_pct': float((d < 0.01).mean() * 100),
            'mse_refit_f2': float(((pred - y2) ** 2).mean()),
            'mse_archive_f2': float(((arch_f2 - y2) ** 2).mean()),
            'prob_max_abs_diff_vs_archived_prob_head': float(np.abs(prob - prob_arch).max()),
            'prob_mean_refit': float(prob.mean()), 'prob_mean_archived': float(prob_arch.mean()),
        },
        'clip_share_pct': {'lo': float((raw_pred < 1.).mean() * 100),
                           'hi': float((raw_pred > 3000.).mean() * 100)},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'mixture2comp_components_f2.npz',
                        origins_ns=g['o_f2'].asi8, level=level.astype(np.float32),
                        prob=prob.astype(np.float32), gap=gap.astype(np.float32),
                        pred=pred.astype(np.float32), truth=y2.astype(np.float32),
                        block_mean=block_mean)
    (OUT / 'refit_receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps(receipt['reproduction'], indent=1))
    print(f"wall {receipt['wall_seconds']}s -> {OUT / 'mixture2comp_components_f2.npz'}")


if __name__ == '__main__':
    main()
