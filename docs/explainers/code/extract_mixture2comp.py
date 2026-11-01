"""Collect every number the e2000fs_calib2_mixture2comp explainer shows.

Inputs
  data/mixture2comp_components_f2.npz  (written by code/refit_mixture2comp.py)
  research/e2000fs_calib2-20261031/runs/mixture_v1/{forecast.npz, losses_*.csv, *.json}
  research/e2000fs_calib2-20261031/runs/detroute_v1/results.json   (control reference)
  analysis/A0014/outputs/cache/{official.npz, names.json}          (design row of the example)

Output
  docs/explainers/data/mixture2comp.json

Run:  /home/t-lab01/sunrun/.venv/bin/python docs/explainers/code/extract_mixture2comp.py
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
RUN = ROOT / 'research/e2000fs_calib2-20261031/runs/mixture_v1'
H, U = 72, 550.0
N_F1 = 2137

_spec = importlib.util.spec_from_file_location(
    'calib_tweedie', ROOT / 'research/e2000fs_calib-20261030/tweedie_lane.py')
tw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tw)

MANIFEST = json.loads((ROOT / 'research/e2000fs_dirmo_24_feat30_pointweight4/results/runs/'
                       'full_v1/manifest.json').read_text())


def curve(path: Path, every: int = 10) -> dict:
    df = pd.read_csv(path)
    idx = list(range(0, len(df), every))
    if idx[-1] != len(df) - 1:
        idx.append(len(df) - 1)
    d = df.iloc[idx]
    return {'round': d['round'].tolist(), 'train': d['train'].round(3).tolist(),
            'valid': d['valid'].round(3).tolist(),
            'first': [float(df['train'].iloc[0]), float(df['valid'].iloc[0])],
            'last': [float(df['train'].iloc[-1]), float(df['valid'].iloc[-1])],
            'best_valid': float(df['valid'].min()), 'best_round': int(df['valid'].idxmin() + 1)}


def sanitize(x):
    if isinstance(x, dict):
        return {k: sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sanitize(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def main() -> None:
    z = np.load(OUT / 'mixture2comp_components_f2.npz')
    origins = pd.DatetimeIndex(z['origins_ns'], tz='UTC')
    level, prob, gap = (np.asarray(z[k], dtype=np.float64) for k in ('level', 'prob', 'gap'))
    pred, truth = np.asarray(z['pred'], np.float64), np.asarray(z['truth'], np.float64)
    block_mean = np.asarray(z['block_mean'], np.float64)          # (3, 24)
    receipt = json.loads((OUT / 'refit_receipt.json').read_text())

    # ---- design rows for the examples (official bundle == the applied matrix)
    names = json.loads((ROOT / 'analysis/A0014/outputs/cache/names.json').read_text())
    sel = MANIFEST['selected']
    ci = [i for i, n in enumerate(names['constant']) if n in sel]
    ei = [i for i, n in enumerate(names['varying']) if n in sel]
    off = dict(np.load(ROOT / 'analysis/A0014/outputs/cache/official.npz'))
    const_vals = np.asarray(off['C'], dtype=np.float64)           # (2833, 90)
    vary_vals = np.asarray(off['E'], dtype=np.float64)            # (2833, 72, 20)

    # ---- pick examples -------------------------------------------------------
    hi = truth >= U
    err_l = (level - truth) ** 2
    err_p = (pred - truth) ** 2
    has_event = hi.any(axis=1)
    gain = err_l.sum(axis=1) - err_p.sum(axis=1)
    pmax = prob.max(axis=1)
    tmax = truth.max(axis=1)
    cand = np.flatnonzero(has_event & (pmax >= 0.55) & (pmax <= 0.95) & (tmax >= 560) & (tmax <= 700)
                          & (gain > 0))
    main_i = int(cand[np.argmin(np.abs(pmax[cand] - 0.7))])
    quiet_pool = np.flatnonzero((~has_event) & (pmax < 0.05) & (tmax < 470))
    quiet_i = int(quiet_pool[len(quiet_pool) // 2])
    miss_pool = np.flatnonzero(has_event & (gain < 0) & (tmax >= 600))
    miss_i = int(miss_pool[np.argmin(gain[miss_pool])])

    feats = {
        'constant': [{'name': names['constant'][i], 'value': float(const_vals[main_i, i])} for i in ci],
        'varying': [{'name': names['varying'][i],
                     'values': [round(float(v), 4) for v in vary_vals[main_i, :, i]]} for i in ei],
        'n_cols_per_block': len(ci) + 24 * len(ei),
        'n_constant_selected': len(ci), 'n_varying_selected': len(ei),
    }

    def example(i: int) -> dict:
        rows = []
        for h in range(H):
            rows.append({'h': h + 1, 'level': float(level[i, h]), 'prob': float(prob[i, h]),
                         'gap': float(gap[i, h]), 'pred': float(pred[i, h]),
                         'truth': float(truth[i, h]), 'up': bool(truth[i, h] >= U),
                         'block': h // 24, 'block_mean': float(block_mean[h // 24, h % 24]),
                         'err_level': float(level[i, h] - truth[i, h]),
                         'err_pred': float(pred[i, h] - truth[i, h])})
        return {'origin': str(origins[i]), 'origin_index': int(i), 'rows': rows,
                'mse_level': float(err_l[i].mean()), 'mse_pred': float(err_p[i].mean()),
                'truth_max': float(truth[i].max()), 'pmax': float(pmax[i]),
                'pmean': float(prob[i].mean()), 'event_hours': int(hi[i].sum())}

    # ---- losses, base rates, event rows -------------------------------------
    curves = {h: {b: curve(RUN / f'losses_mix_{h}_final_b{b}.csv') for b in range(3)}
              for h in ('level', 'prob', 'gap')}

    g = tw.load_data()
    fit_idx = np.flatnonzero(g[2] + pd.Timedelta(hours=144) <= pd.Timestamp('2026-05-29', tz='UTC'))
    y = g[4]
    block_rows, base_rate = {}, {}
    for b in range(3):
        y_tr = y[fit_idx, b * 24:(b + 1) * 24]
        block_rows[str(b)] = int(((y_tr >= U).any(axis=1)).sum())
        base_rate[str(b)] = float((y_tr >= U).mean())

    diag = json.loads((RUN / 'diag_verdict.json').read_text())
    corr = json.loads((RUN / 'diag_correction_structure.json').read_text())
    prob_json = json.loads((RUN / 'refit_prob_head.json').read_text())
    arm = json.loads((RUN / 'diag_prob_and_arm.json').read_text())
    lane = json.loads((RUN / 'results.json').read_text())
    meta = (RUN / 'meta.yaml').read_text()

    # ---- control (the level head's own model) and mixture, both re-scored from the archives
    ctrl_npz = np.load(ROOT / 'research/e2000fs_calib-20261030/runs/tweedie15_v1/forecast_control.npz')
    mix_npz = np.load(RUN / 'forecast.npz')
    y_all = tw.official_truth(pd.DatetimeIndex(mix_npz['origins_ns'], tz='UTC'))
    hi_all = y_all >= U
    f1s, f2s = slice(0, N_F1), slice(N_F1, None)
    def metrics(pred):
        se = (pred - y_all) ** 2
        return {'total': float(se.mean()), 'f1': float(se[f1s].mean()), 'f2': float(se[f2s].mean()),
                'ge550': float(se[hi_all].mean()), 'f1_ge550': float(se[f1s][hi_all[f1s]].mean()),
                'f2_ge550': float(se[f2s][hi_all[f2s]].mean()),
                'bias_lt350': float((pred - y_all)[y_all < 350].mean())}
    mix_m = metrics(np.asarray(mix_npz['predictions'], dtype=np.float64))
    ctrl_m = metrics(np.asarray(ctrl_npz['predictions'], dtype=np.float64))
    assert abs(mix_m['total'] - diag['mse_mixture']) < 1e-6, mix_m['total']
    assert abs(ctrl_m['total'] - diag['mse_control']) < 1e-6, ctrl_m['total']

    base_score_level, base_score_prob = {}, {}
    for b in range(3):
        y_tr = y[fit_idx, b * 24:(b + 1) * 24]
        base_score_level[str(b)] = float(np.nanmean(y_tr))
        base_score_prob[str(b)] = float((y_tr >= U).mean())

    out = {
        'model': 'e2000fs_calib2_mixture2comp',
        'submission_script': 'leaderboard/model-release-publish/scripts/seongeun/'
                             'e2000fs_calib2_mixture2comp/88d14fe5…aacf7a.py',
        'trained_from': 'research/e2000fs_calib2-20261031/lanes2.py :: lane_mixture',
        'run_dir': str(RUN),
        'construction': lane['construction'],
        'discipline': lane['discipline'],
        'horizon': H, 'threshold': U, 'n_f1': N_F1, 'n_origins_f2': int(len(origins)),
        'rows_total': 357840,
        'hyper': MANIFEST['recipe'],
        'features': feats,
        'training': {
            'n_fit_rows_final': int(len(fit_idx)),
            'event_rows_final': block_rows,
            'base_rate_by_block': base_rate,
            'block_mean': block_mean.round(3).tolist(),
            'event_row_share': {k: v / len(fit_idx) for k, v in block_rows.items()},
            'base_score_level': base_score_level,
            'base_score_prob': base_score_prob,
        },
        'losses': curves,
        'examples': {'main': example(main_i), 'quiet': example(quiet_i), 'miss': example(miss_i)},
        'main_gain': float((err_l[main_i] - err_p[main_i]).mean()),
        'refit': receipt,
        'scores': {
            'total': mix_m['total'], 'f1': mix_m['f1'], 'f2': mix_m['f2'],
            'ge550': mix_m['ge550'], 'f1_ge550': mix_m['f1_ge550'], 'f2_ge550': mix_m['f2_ge550'],
            'bias_lt350': mix_m['bias_lt350'],
            'control_total': ctrl_m['total'], 'control_f1': ctrl_m['f1'], 'control_f2': ctrl_m['f2'],
            'control_ge550': ctrl_m['ge550'], 'control_f1_ge550': ctrl_m['f1_ge550'],
            'control_f2_ge550': ctrl_m['f2_ge550'],
            'delta_total': mix_m['total'] - ctrl_m['total'],
            'delta_ge550': mix_m['ge550'] - ctrl_m['ge550'],
            'counterfactuals': {
                'plus_mean': diag['counterfactuals']['control + constant_mean_k'],
                'demeaned': diag['counterfactuals']['control + demeaned_k'],
                'sign_flip': diag['counterfactuals']['control - 0.905*k (in-sample optimal global sign flip)'],
            },
            'clip_low_hits': diag['clip_low_hits_mixture'],
            'verified_against_diag': True,
        },
        'regions': diag['regions'],
        'body_by_block': diag['body_damage_by_hblock'],
        'share_delta_h25_72': diag['share_of_delta_from_h25_72_pct'],
        'correction': {
            'mean_k': corr['mean_variance_split']['mean_k'],
            'std_k': corr['mean_variance_split']['std_k'],
            'corr_k_r': corr['mean_variance_split']['corr_k_r'],
            'constant_cost': corr['mean_variance_split']['constant_uplift_cost_2a_ebar_plus_a2'],
            'fluct_cost': corr['mean_variance_split']['fluctuation_cost_var_k_minus_2cov'],
            'share_constant_pct': corr['mean_variance_split']['share_of_delta_from_constant_pct'],
        },
        'bands': corr['per_band_optimal_scale'],
        'prob_head': prob_json,
        'prob_calibration': arm['calibration_by_p_decile'],
        'prob_quantiles': arm['p_quantiles'],
        'prob_brier': arm['brier'],
        'prob_mean_pct': arm['p_mean_pct'], 'true_rate_pct': arm['true_rate_pct'],
        'meta_yaml_head': meta.split('config:')[0].strip(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'mixture2comp.json').write_text(json.dumps(sanitize(out), indent=1) + '\n')
    print('main', out['examples']['main']['origin'], 'pmax', round(out['examples']['main']['pmax'], 3),
          'quiet', out['examples']['quiet']['origin'], 'miss', out['examples']['miss']['origin'])
    print('reproduction', receipt['reproduction'])
    print('->', OUT / 'mixture2comp.json')


if __name__ == '__main__':
    main()
