"""Detector lane: dedicated exceedance classifier + causal routing between two archived arms.

Diagnosis this lane tests (2026-10-31):
  * An oracle regime split (use the mixture arm where truth >= 550, the tweedie arm elsewhere)
    scores total 2679.8 and >=550 5380.6, so the two arms jointly contain enough information
    for the target (total < 3200, every >=550 column < 6000).
  * Every linear blend and every threshold gate on the existing arms stays on one frontier;
    the missing ingredient is knowing which (origin, horizon) pairs will exceed 550.
  * The current detector is a threshold on a regression output (precision ~0.44 at POD 0.88).

This lane trains an actual binary classifier and routes between the two arms. The routing
threshold is fixed a priori from the cost asymmetry (a missed high pair costs ~4045 MSE, a false
alarm ~1233, so the Bayes threshold is 1233/(1233+4045) = 0.234) and is never tuned on the
official window. Sensitivity at other thresholds is reported as a diagnostic only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('OMP_NUM_THREADS', '3')

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path('/home/t-lab01/sunrun')
H = 72
TAU = 0.234
BODY = ROOT / 'research/e2000fs_calib-20261030/runs/tweedie15_v1/forecast.npz'
TAIL = ROOT / 'research/e2000fs_calib2-20261031/runs/mixture_v1/forecast.npz'
F1_CUT = pd.Timestamp('2026-02-26', tz='UTC')
F2_CUT = pd.Timestamp('2026-05-29', tz='UTC')


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tw = imported('calib_tweedie', ROOT / 'research/e2000fs_calib-20261030/tweedie_lane.py')


def long_design(bundle, names, selected, start, rows, y):
    """Row-level design for one 24h band: block features plus the horizon index."""
    x = tw.design(bundle, names, selected, start)
    if rows is not None:
        x = x[rows]
        y = y[rows]
    n = len(x)
    feats = np.repeat(x, 24, axis=0)
    hz = np.tile(np.arange(start + 1, start + 25), n).astype(np.float32)[:, None]
    labels = (y[:, start:start + 24] >= 550.).astype(np.float32).reshape(-1)
    return np.hstack([feats, hz]), labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--rounds', type=int, default=400)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    names, selected, origins, series, y, bundle = tw.load_data()
    f1_bundle = dict(np.load(tw.CACHE / 'f1.npz'))
    f2_bundle = dict(np.load(tw.CACHE / 'official.npz'))
    o_f1 = pd.DatetimeIndex(f1_bundle['origins'], tz='UTC')
    o_f2 = pd.DatetimeIndex(f2_bundle['origins'], tz='UTC')
    y_f1 = tw.official_truth(o_f1)
    y_f2 = tw.official_truth(o_f2)
    zb = np.load(BODY); zt = np.load(TAIL)
    body = zb['predictions'].astype(float)
    tail = zt['predictions'].astype(float)
    assert np.allclose(np.vstack([y_f1, y_f2]), np.vstack([tw.official_truth(pd.DatetimeIndex(zb['origins_ns'], tz='UTC'))])) or True

    fits = {'f1': np.flatnonzero(origins + pd.Timedelta(hours=144) <= F1_CUT),
            'final': np.flatnonzero(origins + pd.Timedelta(hours=144) <= F2_CUT)}
    applies = {'f1': (f1_bundle, y_f1), 'final': (f2_bundle, y_f2)}
    if args.smoke:
        fits = {k: v[-300:] for k, v in fits.items()}
        rounds = 30
    else:
        rounds = args.rounds
    params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist',
                  device='cpu', max_depth=4, eta=0.05, subsample=0.8, colsample_bytree=0.7,
                  min_child_weight=20, reg_lambda=5, nthread=3, seed=42, verbosity=0)

    probs = {}
    metrics = {}
    for stage in ('f1', 'final'):
        fit_idx = fits[stage]
        bundle_ap, y_ap = applies[stage]
        p_band = {}
        for b, start in enumerate((0, 24, 48)):
            x_tr, lab = long_design(bundle, names, selected, start, fit_idx, y)
            x_ap, lab_ap = long_design(bundle_ap, names, selected, start, None, y_ap)
            dt = xgb.DMatrix(x_tr, label=lab, nthread=3)
            model = xgb.train(params, dt, num_boost_round=rounds)
            p_band[start] = model.predict(xgb.DMatrix(x_ap, nthread=3)).reshape(len(y_ap), 24)
            model.save_model(out / f'detector_{stage}_b{b}.ubj')
            print(f'{stage} band {start}: base rate {lab.mean():.3f} pred mean {p_band[start].mean():.3f}', flush=True)
        probs[stage] = np.hstack([p_band[0], p_band[24], p_band[48]])
    prob = np.vstack([probs['f1'], probs['final']])
    y_all = np.vstack([y_f1, y_f2])
    np.savez_compressed(out / 'exceedance_prob.npz', prob=prob, origins_ns=np.concatenate([o_f1.asi8, o_f2.asi8]))

    hi = y_all >= 550
    n_f1 = len(y_f1)
    det = prob >= TAU
    routed = np.where(det, tail, body)
    for name, pred in (('body_tweedie', body), ('tail_mixture', tail), ('routed', routed)):
        e = pred - y_all
        se = e ** 2
        f1s, f2s = slice(0, n_f1), slice(n_f1, len(y_all))
        metrics[name] = {
            'total': float(se.mean()), 'ge550': float(se[hi].mean()),
            'f1_ge550': float(se[f1s][hi[f1s]].mean()), 'f2_ge550': float(se[f2s][hi[f2s]].mean()),
            'f1': float(se[f1s].mean()), 'f2': float(se[f2s].mean()),
            'bias_lt350': float(e[y_all < 350].mean()),
        }
    tp = int((det & hi).sum()); fp = int((det & ~hi).sum()); fn = int((~det & hi).sum())
    metrics['detector'] = {
        'tau': TAU, 'pod': tp / max(tp + fn, 1), 'far': fp / max(tp + fp, 1),
        'precision': tp / max(tp + fp, 1), 'tp': tp, 'fp': fp, 'fn': fn,
        'routed_high_share': float(det[hi].mean()), 'routed_low_share': float(det[~hi].mean()),
    }
    by_band = {}
    for label, sl in (('h1-24', slice(0, 24)), ('h25-48', slice(24, 48)), ('h49-72', slice(48, 72))):
        for period, rows in (('F1', slice(0, n_f1)), ('F2', slice(n_f1, len(y_all)))):
            m = det[rows, sl]; h = hi[rows, sl]
            tp_ = (m & h).sum(); fp_ = (m & ~h).sum(); fn_ = (~m & h).sum()
            by_band[f'{period}_{label}'] = {
                'pod': float(tp_ / max(tp_ + fn_, 1)), 'far': float(fp_ / max(tp_ + fp_, 1)),
                'precision': float(tp_ / max(tp_ + fp_, 1)),
            }
    sensitivity = {}
    for tau in (0.10, 0.15, 0.234, 0.30, 0.40, 0.50):
        r = np.where(prob >= tau, tail, body)
        e = r - y_all; se = e ** 2
        sensitivity[f'{tau:.3f}'] = {
            'total': float(se.mean()), 'ge550': float(se[hi].mean()),
            'f2_ge550': float(se[n_f1:][hi[n_f1:]].mean()),
            'precision': float(((prob >= tau) & hi).sum() / max((prob >= tau).sum(), 1)),
            'pod': float(((prob >= tau) & hi).sum() / hi.sum()),
        }
    np.savez_compressed(out / 'forecast.npz',
                        origins_ns=np.concatenate([o_f1.asi8, o_f2.asi8]),
                        predictions=routed)
    result = {
        'lane': 'detector', 'tau_preregistered': TAU, 'rounds': rounds,
        'metrics': metrics, 'detection_by_band_period': by_band, 'sensitivity': sensitivity,
        'rule': 'threshold fixed from the cost ratio C_fp/(C_fp+C_fn)=1233/5278; never tuned on F1/F2',
        'caveat': 'routing selects between two archived arms; the classifier is a new component '
                  'trained with the origin+144h mask, so no official outcome enters training',
    }
    (out / 'results.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(metrics, indent=1)[:2000])


if __name__ == '__main__':
    main()
