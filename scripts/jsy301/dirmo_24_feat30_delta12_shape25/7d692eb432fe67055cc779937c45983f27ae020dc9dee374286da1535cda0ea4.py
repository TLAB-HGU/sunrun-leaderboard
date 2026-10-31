"""DirMO-24 feat30 with per-block methods: leaderboard entry script (self-contained).

  leads  1-12  delta target (y - v_lag0) with the team DirMO-24 feat30 recipe, block 1 model
  leads 13-24  team DirMO-24 feat30 (seongeun e2000fs_dirmo_24_feat30), unchanged
  leads 25-72  team DirMO-24 feat30 + 0.25 x stage-2 residual correction: a small LightGBM fit on out-of-fold DirMO
               forecasts of the training rows (leads 25-72 only), with level-free inputs only (forecast path shape,
               lead, recurrence/ACE/coronal-hole/ENLIL issue-time values, and level differences such as recurrence
               minus forecast; absolute levels and season are left out because their effect flips between periods)

Per period (F1: targets before 2026-02-26, F2: before 2026-05-29): fit DirMO on the period's training origins, build
out-of-fold DirMO forecasts on those origins (K contiguous folds, 72 h purge), fit the residual model on them (leads
25-72), and add kappa (fixed) x its prediction to the period's forecast.

    python <this file> --cache <dir with train.npz f1.npz official.npz names.json> \
        --speed <ACE speed parquet, column 'speed', hourly UTC index, ends before 2026-05-29> --out predictions.parquet
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

CONFIG = {'data': {'base': 'e2000fs_dirmo_24_feat30', 'feature_count': 30, 'block_input_columns': 237, 'f1_training_target_end_utc': '2026-02-26T00:00:00Z', 'f2_training_target_end_utc': '2026-05-29T00:00:00Z', 'cache_sha256': {'train.npz': '5f9c446b1beb7c8fb4290236bfbbde08f32153433cfafadc3c02fd189406a8a9', 'f1.npz': 'f126160c2a8d5bb12df376781a4d5712cdae0708385c717cc82419556629db1d', 'official.npz': '9878a560b0d81ddec35f65184afe6a446f425750cce2074290f320d6613ec6ae', 'names.json': '3b25e892c0093215c90f0f3e579a72a0c6cb314a38c694a4c38e5c58744b14ce', 'speed.parquet': '96f6e22e1643d79c5dd49f04825128dc25c3b8985a5415fa5828fd7db3012877'}}, 'hyperparameters': {'strategy': 'DirMO-24-Top30; leads 1-12 delta target; leads 25-72 + 0.25 x level-free stage-2 residual model (fit on out-of-fold leads 25-72)', 'delta_leads': 12, 'feature_groups': ['v_lag0', 'rec_mean', 'enlil_kp', 'ch_ch_t035_meridian30_equatorial_area', 'ace_bt_nt_last', 'ace_speed_kms_last', 'ch_ch_t035_equatorial_area', 'ch_ch_t035_meridian15_equatorial_area', 'ace_bt_nt_m6', 'rec26', 'strip_v450_logmed', 'strip_v350_logmed', 'ch_ch_t055_equatorial_area', 'rec27', 'ace_density_cm3_last', 'ch_ch_t055_meridian30_equatorial_area', 'ace_by_gsm_nt_m24', 'ace_density_cm3_m6', 'ace_temperature_k_last', 'enlil_n_pending', 'ace_temperature_k_m24', 'strip_v750_logmed', 'ch_ch_t055_meridian15_area', 'ch_ch_t035_top2_area', 'strip_v550_dark', 'strip_v650_dark', 'ace_sis_gt10_flux_last', 'ace_epam_p_channel3_last', 'img_age_h', 'rec28'], 'xgboost': {'n_estimators': 900, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 20, 'reg_lambda': 5, 'random_state': 42, 'objective': 'reg:squarederror', 'tree_method': 'hist', 'device': 'cpu', 'multi_strategy': 'multi_output_tree', 'n_jobs': 4}, 'stack': {'first_lead': 25, 'folds': 4, 'kappa': 0.25, 'keep_columns': [1, 4, 10, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 35], 'lightgbm': {'objective': 'regression', 'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 7, 'min_child_samples': 2000, 'subsample': 0.7, 'subsample_freq': 1, 'colsample_bytree': 0.8, 'reg_lambda': 10.0, 'n_jobs': 8, 'verbosity': -1, 'random_state': 0}}, 'seed': 42}}
CONFIG_SHA256 = "7d692eb432fe67055cc779937c45983f27ae020dc9dee374286da1535cda0ea4"
H = 72


def design(b, names, start, groups):
    ci = [i for i, g in enumerate(names['constant']) if g in groups]
    ei = [i for i, g in enumerate(names['varying']) if g in groups]
    n = len(b['C'])
    return np.hstack([b['C'][:, ci], b['E'][:, start:start + 24, :][:, :, ei].reshape(n, -1)]).astype(np.float32)


def targets(series, origins):
    times = origins.values[:, None] + np.arange(1, H + 1).astype('timedelta64[h]')[None, :]
    return series.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, H)


def take(b, idx):
    return {k: b[k][idx] for k in ('C', 'E', 'past', 'origins')}


def fit_blocks(tr, names, groups, params, y, fit, target_b, blocks=(0, 24, 48), offset=None):
    """DirMO blocks fit on rows `fit` of `tr`, predicted on `target_b`; offset=(train v0, target v0) for delta."""
    out = np.full((len(target_b['C']), H), np.nan)
    models = {}
    for s0 in blocks:
        X, Xt = design(tr, names, s0, groups), design(target_b, names, s0, groups)
        Y = y[fit, s0:s0 + 24]
        o_tr, o_ev = (offset[0][fit, None], offset[1][:, None]) if offset is not None else (0., 0.)
        m = xgb.XGBRegressor(**params).fit(X[fit], Y - o_tr, verbose=False)
        out[:, s0:s0 + 24] = np.clip(m.predict(Xt) + o_ev, 1., 3000.)
        models[s0] = (m, Xt)
    return out, models


def crossfit(tr, names, groups, params, y, fit, origins, k):
    """Out-of-fold DirMO forecasts on rows `fit`: K contiguous folds, each predicted by a model fit on the other
    rows of `fit` whose targets end before the fold or whose origin is more than 72 h after it."""
    o = origins[fit]
    edges = np.linspace(0, len(fit), k + 1).astype(int)
    oof = np.full((len(fit), H), np.nan)
    Xs = {s0: design(tr, names, s0, groups) for s0 in (0, 24, 48)}
    for f in range(k):
        lo, hi = edges[f], edges[f + 1]
        t_lo, t_hi = o[lo], o[hi - 1]
        sub = np.flatnonzero((o + pd.Timedelta(hours=H) < t_lo) | (o > t_hi + pd.Timedelta(hours=H)))
        for s0 in (0, 24, 48):
            m = xgb.XGBRegressor(**params).fit(Xs[s0][fit[sub]], y[fit[sub], s0:s0 + 24], verbose=False)
            oof[lo:hi, s0:s0 + 24] = np.clip(m.predict(Xs[s0][fit[lo:hi]]), 1., 3000.)
    return oof


ROW_CONSTANTS = ('v_lag0', 'v_m24', 'v_m72', 'v_sd24', 'speed_d24', 'ace_density_cm3_last', 'ace_temperature_k_last',
                 'ace_bt_nt_last', 'ch_ch_t035_equatorial_area', 'ch_ch_t035_meridian30_equatorial_area',
                 'ch_ch_t055_equatorial_area', 'enlil_kp', 'enlil_next_arr_h', 'enlil_n_pending')


def row_feats(p, b, names):
    ci = {n: i for i, n in enumerate(names['constant'])}
    rec = b['E'][:, :, names['varying'].index('rec_mean')]
    o = pd.DatetimeIndex(b['origins'], tz='UTC')
    past = b['past']
    cols = [p.max(1), p.argmax(1), p.min(1), p.mean(1), p.std(1), p[:, 0], p[:, 23], p[:, 47], p[:, 71],
            np.nanmax(rec, 1), np.nanargmax(np.nan_to_num(rec, nan=-1), 1), np.nanmax(rec, 1) - np.nanmin(rec, 1),
            past.max(1) - past.min(1), past[:, -1] - past[:, -7],
            np.sin(2 * np.pi * o.dayofyear / 365.25), np.cos(2 * np.pi * o.dayofyear / 365.25)]
    cols += [b['C'][:, ci[n]] for n in ROW_CONSTANTS]
    return np.column_stack(cols).astype(np.float32), rec


def long_feats(p, b, names):
    F, rec = row_feats(p, b, names)
    n = len(p)
    lead = np.tile(np.arange(1, H + 1), n)
    ph = p.ravel()
    prev = np.concatenate([p[:, :1].repeat(6, 1), p[:, :-6]], 1).ravel()
    return np.column_stack([np.repeat(F, H, 0), lead, ph, ph - prev, ph - np.repeat(p[:, 0], H), rec.ravel(),
                            ph - np.repeat(p.max(1), H)]).astype(np.float32)


def shape(X, keep):
    """level-free columns of long_feats plus level differences: recurrence - forecast, speed now - forecast,
    recurrence max - forecast max"""
    return np.column_stack([X[:, keep], X[:, 34] - X[:, 31], X[:, 16] - X[:, 31], X[:, 9] - X[:, 0]])


def stack(oof, b_fit, y_fit, pe, b_eval, names, cfg):
    """pe + kappa x LightGBM residual model fit on the out-of-fold rows' leads >= first_lead (level-free inputs)."""
    n = len(oof)
    Xl, R = shape(long_feats(oof, b_fit, names), cfg['keep_columns']), (y_fit - oof).ravel()
    rows = np.tile(np.arange(1, H + 1) >= cfg['first_lead'], n)
    m = lgb.LGBMRegressor(**cfg['lightgbm']).fit(Xl[rows], R[rows])
    Xe = shape(long_feats(pe, b_eval, names), cfg['keep_columns'])
    corr = m.predict(Xe).reshape(len(pe), H)
    kappa = cfg['kappa']
    return np.clip(pe + kappa * corr, 1., 3000.), kappa, (m, Xe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--speed', required=True)
    ap.add_argument('--out', default='predictions.parquet')
    a = ap.parse_args()
    t0 = time.time()
    hp, data = CONFIG['hyperparameters'], CONFIG['data']
    groups, k_delta, s_lead = hp['feature_groups'], hp['delta_leads'], hp['stack']['first_lead']
    params = dict(hp['xgboost'])
    cache = Path(a.cache)
    names = json.loads((cache / 'names.json').read_text())
    tr = dict(np.load(cache / 'train.npz'))
    origins = pd.DatetimeIndex(tr['origins'], tz='UTC')
    y = targets(pd.read_parquet(a.speed)['speed'], origins)
    v0 = names['constant'].index('v_lag0')
    oo, preds, info, timing = [], [], {}, []
    for stage, grid, cut in (('f1', 'f1', data['f1_training_target_end_utc']),
                             ('final', 'official', data['f2_training_target_end_utc'])):
        g = dict(np.load(cache / f'{grid}.npz'))
        fit = np.flatnonzero(origins + pd.Timedelta(hours=H) < pd.Timestamp(cut))
        assert np.isfinite(y[fit]).all()
        level, lm = fit_blocks(tr, names, groups, params, y, fit, g)
        delta, dm = fit_blocks(tr, names, groups, params, y, fit, g, blocks=(0,),
                               offset=(tr['C'][:, v0].astype(float), g['C'][:, v0].astype(float)))
        print(f'[{time.time() - t0:6.0f}s] {stage}: DirMO level + delta block 1 fit ({len(fit)} origins)', flush=True)
        oof = crossfit(tr, names, groups, params, y, fit, origins, hp['stack']['folds'])
        print(f'[{time.time() - t0:6.0f}s] {stage}: out-of-fold DirMO done', flush=True)
        stacked, kappa, (sm, Xe) = stack(oof, take(tr, fit), y[fit], level, g, names, hp['stack'])
        print(f'[{time.time() - t0:6.0f}s] {stage}: residual model, kappa={kappa}', flush=True)
        pred = np.concatenate([delta[:, :k_delta], level[:, k_delta:s_lead - 1], stacked[:, s_lead - 1:]], 1)
        assert np.isfinite(pred).all()
        preds.append(pred)
        oo.append(pd.DatetimeIndex(g['origins'], tz='UTC'))
        info[stage] = dict(n_fit=int(len(fit)), kappa=kappa)
        # post-fit predict time of one 72 h forecast: 3 level blocks + delta block + residual model (72 rows)
        for i in range(0, len(g['origins']), 29):
            ts = time.perf_counter()
            for s0, (m, Xt) in lm.items():
                m.predict(Xt[i:i + 1])
            dm[0][0].predict(dm[0][1][i:i + 1])
            sm.predict(Xe[i * H:(i + 1) * H])
            timing.append(time.perf_counter() - ts)
    o, pred = oo[0].append(oo[1]), np.vstack(preds)
    frame = pd.DataFrame({'origin_last_input_utc': np.repeat(o.strftime('%Y-%m-%dT%H:%M:%SZ'), H),
                          'horizon_hours': np.tile(np.arange(1, H + 1), len(o)), 'pred_kms': pred.ravel()})
    frame.to_parquet(a.out, index=False)
    print(json.dumps(dict(rows=len(frame), out=str(a.out), config_sha256=CONFIG_SHA256, **info,
                          inference_seconds_per_fold=float(np.mean(timing)), elapsed_s=round(time.time() - t0))))


if __name__ == '__main__':
    main()
