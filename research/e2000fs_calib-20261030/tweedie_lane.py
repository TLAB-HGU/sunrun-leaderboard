"""Tweedie-deviance lane: replace the 4x high-speed weighted L2 with a Tweedie deviance.

Rationale (bottleneck from the 2026-10-30 review): the archived point4 objective is a *weighted*
squared error whose minimiser is the tilted mean E[wY|X]/E[w|X], not E[Y|X]. Solar-wind speed is
positive and right-skewed with a floor, so a Tweedie deviance (variance function mu^p) is a
likelihood-consistent alternative that penalises relative error at the low end. arXiv:2509.08369
reports that swapping RMSE for a Tweedie deviance improves both tails of a skewed positive target.

Design (fixed, preregistered):
  * direct 72 h, three 24-output vector trees, frozen Top30 inputs, same 900-round recipe as the
    direct parent e2000fs_dirmo_24_feat30_pointweight4.
  * arm `control` reproduces the parent: w=4 for y>=550, mean-normalised per training horizon.
  * arm `tweedie` uses the deviance with variance power P (default 1.5) and no sample weights.
  * the deviance is multiplied by S**P (S=500 km/s, a constant) so gradients stay O(1) and the
    parent regularisation strength keeps its meaning; a constant factor does not move the optimum.
  * selection never touches F1/F2: only the internal window (2025-12-04..2026-02-22) is scored, and
    every stage is reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('OMP_NUM_THREADS', '3')

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path('/home/t-lab01/sunrun')
CACHE = ROOT / 'analysis/A0014/outputs/cache'
MANIFEST = ROOT / 'research/e2000fs_dirmo_24_feat30_pointweight4/results/runs/full_v1/manifest.json'
OFFICIAL = ROOT / 'research/e2000fs_budget_spectral/results/forecast.npz'
H = 72
HPO = 550.
SCALE = 500.


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=1, default=str) + '\n')


def load_data():
    names = read_json(CACHE / 'names.json')
    selected = read_json(MANIFEST)['selected']
    train = dict(np.load(CACHE / 'train.npz'))
    origins = pd.DatetimeIndex(train['origins'], tz='UTC')
    series = pd.read_parquet(CACHE / 'speed.parquet')['speed'].sort_index()
    times = origins.values[:, None] + np.arange(1, H + 1).astype('timedelta64[h]')[None, :]
    y = series.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, H)
    return names, selected, origins, series, y, train


def official_truth(o: pd.DatetimeIndex) -> np.ndarray:
    series = pd.read_parquet(ROOT / 'data-v2/model-inputs/leaderboard-store/e2000fs-lag0/speed.parquet')['speed']
    times = o.values[:, None] + np.arange(1, H + 1).astype('timedelta64[h]')[None, :]
    y = series.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, H)
    if np.isnan(y).any():
        raise ValueError('official window has unobserved targets')
    return y


def design(bundle, names, selected, start):
    ci = [i for i, n in enumerate(names['constant']) if n in selected]
    ei = [i for i, n in enumerate(names['varying']) if n in selected]
    block = bundle['E'][:, start:start + 24, :][:, :, ei].reshape(len(bundle['C']), -1)
    return np.hstack([bundle['C'][:, ci], block]).astype(np.float32)


def point_weights(target, normalizer=None):
    w = np.where(target >= HPO, 4., 1.)
    norm = w.mean(axis=0) if normalizer is None else np.asarray(normalizer)
    return (w / norm).astype(np.float32)


def tweedie_objective(pred, matrix, power):
    target = matrix.get_label().reshape(pred.shape).astype(np.float64)
    mu = np.clip(pred.astype(np.float64), 1., 3000.)
    s = SCALE ** power
    grad = s * (mu ** (1. - power) - target * mu ** (-power))
    hess = s * ((1. - power) * mu ** (-power) + power * target * mu ** (-power - 1.))
    hess = np.clip(hess, 1e-6 * s, None)
    return grad.astype(np.float32), hess.astype(np.float32)


def fit_block(arm, x_tr, y_tr, x_va, y_va, params, rounds, power, log_path=None):
    dt = xgb.DMatrix(x_tr, label=y_tr.astype(np.float32), nthread=3)
    dv = xgb.DMatrix(x_va, label=y_va.astype(np.float32), nthread=3)
    if arm == 'control':
        def objective(pred, matrix):
            t = matrix.get_label().reshape(pred.shape)
            w = point_weights(t)
            return ((pred - t) * w).astype(np.float32), w.astype(np.float32)
    else:
        def objective(pred, matrix):
            return tweedie_objective(pred, matrix, power)
    history = {'round': [], 'train': [], 'valid': []}

    class Recorder(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            history['round'].append(epoch + 1)
            history['train'].append(evals_log['train']['rmse'][-1] ** 2)
            history['valid'].append(evals_log['validation']['rmse'][-1] ** 2)
            return False

    model = xgb.train(params, dt, num_boost_round=rounds, obj=objective,
                      evals=[(dt, 'train'), (dv, 'validation')],
                      custom_metric=None, callbacks=[Recorder()], verbose_eval=False)
    if log_path:
        pd.DataFrame(history).to_csv(log_path, index=False)
    return model, history


def execute(out: Path, smoke: bool = False, power: float = 1.5, rounds: int = 900) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    names, selected, origins, series, y, bundle = load_data()
    blocks = [design(bundle, names, selected, h) for h in (0, 24, 48)]
    assert all(x.shape[1] == 237 for x in blocks), [x.shape for x in blocks]

    off = np.load(OFFICIAL)
    official = pd.DatetimeIndex(off['origins_ns'], tz='UTC')
    n_f1 = int((official < pd.Timestamp('2026-05-29', tz='UTC')).sum())
    f1_bundle = dict(np.load(CACHE / 'f1.npz'))
    f2_bundle = dict(np.load(CACHE / 'official.npz'))
    o_f1 = pd.DatetimeIndex(f1_bundle['origins'], tz='UTC')
    o_f2 = pd.DatetimeIndex(f2_bundle['origins'], tz='UTC')
    assert np.array_equal(o_f1.asi8, official[:n_f1].asi8), 'F1 origin grid mismatch'
    assert np.array_equal(o_f2.asi8, official[n_f1:].asi8), 'F2 origin grid mismatch'
    off_blocks = {'f1': [design(f1_bundle, names, selected, h) for h in (0, 24, 48)],
                  'final': [design(f2_bundle, names, selected, h) for h in (0, 24, 48)]}

    recipe = read_json(MANIFEST)['recipe']['xgboost']
    params = {k: v for k, v in recipe.items()
              if k not in ('n_estimators', 'n_jobs', 'random_state', 'base_score', 'objective')}
    params.update(objective='reg:squarederror', tree_method='hist', device='cpu',
                  multi_strategy='multi_output_tree', nthread=3, seed=42,
                  eval_metric='rmse', verbosity=1)

    val_start = pd.Timestamp('2025-12-04', tz='UTC')
    val_end = pd.Timestamp('2026-02-22T23:00:00Z')
    internal_val = np.flatnonzero((origins >= val_start) & (origins <= val_end))
    f1_fit = np.flatnonzero(origins + pd.Timedelta(hours=144) <= pd.Timestamp('2026-02-26', tz='UTC'))
    final_fit = np.flatnonzero(origins + pd.Timedelta(hours=144) <= pd.Timestamp('2026-05-29', tz='UTC'))
    internal_fit = np.flatnonzero(origins < val_start - pd.Timedelta(hours=144))
    if smoke:
        internal_fit = internal_fit[-120:]
        internal_val = internal_val[:48]
        f1_fit = f1_fit[-120:]
        final_fit = final_fit[-120:]
        rounds = min(rounds, 40)

    targets = {'internal': (internal_fit, internal_val, None),
               'f1': (f1_fit, None, o_f1), 'final': (final_fit, None, o_f2)}
    results, hashes = {}, {}
    for arm in ('control', 'tweedie'):
        results[arm] = {}
        for stage, (fit_idx, val_idx, apply_origins) in targets.items():
            if stage == 'internal':
                x_va, y_va, oo, apply_idx = blocks[0][val_idx], y[val_idx], origins[val_idx], val_idx
                eval_origins = origins[val_idx]
            else:
                eval_origins = apply_origins
                pred = None
            n_eval = len(val_idx) if stage == 'internal' else len(eval_origins)
            pred = np.empty((n_eval, H))
            for b in range(3):
                if stage == 'internal':
                    x_tr, y_tr = blocks[b][fit_idx], y[fit_idx, b * 24:(b + 1) * 24]
                    x_va_b, y_va_b = blocks[b][val_idx], y[val_idx, b * 24:(b + 1) * 24]
                    x_apply = x_va_b
                else:
                    x_tr, y_tr = blocks[b][fit_idx], y[fit_idx, b * 24:(b + 1) * 24]
                    x_va_b = off_blocks[stage][b]
                    # validation set for the recorder: last 240 training origins held out
                    hold = fit_idx[-240:]
                    x_val_h = blocks[b][hold]
                    y_val_h = y[hold, b * 24:(b + 1) * 24]
                    x_apply = x_va_b
                p = dict(params, base_score=float(np.nanmean(y_tr)))
                if stage == 'internal':
                    vx, vy = x_va_b, y_va_b
                else:
                    vx, vy = x_val_h, y_val_h
                model, hist = fit_block(arm, x_tr, y_tr, vx, vy, p, rounds,
                                        power, out / f'losses_{arm}_{stage}_b{b}.csv')
                raw = model.inplace_predict(x_apply)
                assert raw.shape == (len(x_apply), 24), raw.shape
                pred[:, b * 24:(b + 1) * 24] = np.clip(raw, 1., 3000.)
                model.save_model(out / f'model_{arm}_{stage}_b{b}.ubj')
                hashes[f'model_{arm}_{stage}_b{b}.ubj'] = digest(out / f'model_{arm}_{stage}_b{b}.ubj')
                write_json(out / f'status_{arm}_{stage}.json',
                           {'status': 'running', 'arm': arm, 'stage': stage, 'block': b,
                            'rounds': rounds, 'power': power, 'smoke': smoke})
            yy = y[val_idx] if stage == 'internal' else official_truth(eval_origins)
            err = pred - yy
            hi = yy >= HPO
            lo = yy < 350
            results[arm][stage] = {
                'n_origins': int(len(yy)),
                'mse': float((err ** 2).mean()),
                'mse_ge550': float((err ** 2)[hi].mean()) if hi.any() else None,
                'bias': float(err.mean()), 'bias_lt350': float(err[lo].mean()) if lo.any() else None,
                'bias_ge550': float(err[hi].mean()) if hi.any() else None,
                'mse_by_band': {k: float((err[:, sl] ** 2).mean())
                                for k, sl in {'h1-24': slice(0, 24), 'h25-48': slice(24, 48),
                                              'h49-72': slice(48, 72)}.items()},
            }
            np.savez_compressed(out / f'{arm}_{stage}_predictions.npz',
                                pred=pred, origins=eval_origins.asi8)
            write_json(out / f'status_{arm}_{stage}.json',
                       {'status': 'completed_evaluated', **results[arm][stage]})
            print(f'{arm} {stage}: ' + json.dumps(results[arm][stage]), flush=True)

    # official F1+F2 prediction grid for the tweedie arm (submission artefact)
    f1 = np.load(out / 'tweedie_f1_predictions.npz')
    fin = np.load(out / 'tweedie_final_predictions.npz')
    official_pred = np.vstack([f1['pred'], fin['pred']])
    assert official_pred.shape == (len(official), H)
    np.savez_compressed(out / 'forecast.npz', origins_ns=official.asi8, predictions=official_pred)
    control_pred = np.vstack([np.load(out / 'control_f1_predictions.npz')['pred'],
                              np.load(out / 'control_final_predictions.npz')['pred']])
    np.savez_compressed(out / 'forecast_control.npz', origins_ns=official.asi8, predictions=control_pred)

    summary = {'lane': 'tweedie', 'power': power, 'rounds': rounds, 'smoke': smoke,
               'results': results, 'model_sha256': hashes,
               'inputs': {str(CACHE / n): digest(CACHE / n) for n in
                          ('names.json', 'train.npz', 'speed.parquet', 'f1.npz', 'official.npz')},
               'frozen': {'selected': selected, 'manifest': str(MANIFEST), 'manifest_sha256': digest(MANIFEST)},
               'discipline': 'internal window only for any selection; F1/F2 scored once after fitting'}
    write_json(out / 'results.json', summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--power', type=float, default=1.5)
    ap.add_argument('--rounds', type=int, default=900)
    args = ap.parse_args()
    t0 = time.time()
    summary = execute(Path(args.out), args.smoke, args.power, args.rounds)
    print(f'elapsed {time.time() - t0:.1f}s')
    print(json.dumps({a: {k: (round(v, 2) if isinstance(v, float) else v)
                          for k, v in s.items() if k != 'mse_by_band'}
                      for a, st in summary['results'].items() for k, s in st.items()},
                     indent=1)[:2000])


if __name__ == '__main__':
    main()
