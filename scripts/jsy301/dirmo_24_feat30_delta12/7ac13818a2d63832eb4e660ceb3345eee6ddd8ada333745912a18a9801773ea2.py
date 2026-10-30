"""DirMO-24 feat30 + delta-target short leads: leaderboard entry script (self-contained).

Leads 1..K from the team DirMO-24 feat30 recipe trained on the delta target (y - v_lag0, ACE speed at origin);
leads K+1..72 from the same recipe on the speed itself (= seongeun e2000fs_dirmo_24_feat30). F1 models fit on
training origins whose 72 h targets end before 2026-02-26, F2 models before 2026-05-29.

    python <this file> --cache <dir with train.npz f1.npz official.npz names.json> \
        --speed <ACE speed parquet, column 'speed', hourly UTC index> --out predictions.parquet

Inputs: team cache reference/pointweight4_cache (tlabtlab/sunrun-repro-data); hashes in CONFIG.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

CONFIG = {'data': {'base': 'e2000fs_dirmo_24_feat30', 'feature_count': 30, 'block_input_columns': 237, 'f1_training_target_end_utc': '2026-02-26T00:00:00Z', 'f2_training_target_end_utc': '2026-05-29T00:00:00Z', 'cache_sha256': {'train.npz': '5f9c446b1beb7c8fb4290236bfbbde08f32153433cfafadc3c02fd189406a8a9', 'f1.npz': 'f126160c2a8d5bb12df376781a4d5712cdae0708385c717cc82419556629db1d', 'official.npz': '9878a560b0d81ddec35f65184afe6a446f425750cce2074290f320d6613ec6ae', 'names.json': '3b25e892c0093215c90f0f3e579a72a0c6cb314a38c694a4c38e5c58744b14ce', 'speed.parquet': '96f6e22e1643d79c5dd49f04825128dc25c3b8985a5415fa5828fd7db3012877'}}, 'hyperparameters': {'strategy': 'DirMO-24-Top30 + delta-target leads 1-12', 'delta_leads': 12, 'feature_groups': ['v_lag0', 'rec_mean', 'enlil_kp', 'ch_ch_t035_meridian30_equatorial_area', 'ace_bt_nt_last', 'ace_speed_kms_last', 'ch_ch_t035_equatorial_area', 'ch_ch_t035_meridian15_equatorial_area', 'ace_bt_nt_m6', 'rec26', 'strip_v450_logmed', 'strip_v350_logmed', 'ch_ch_t055_equatorial_area', 'rec27', 'ace_density_cm3_last', 'ch_ch_t055_meridian30_equatorial_area', 'ace_by_gsm_nt_m24', 'ace_density_cm3_m6', 'ace_temperature_k_last', 'enlil_n_pending', 'ace_temperature_k_m24', 'strip_v750_logmed', 'ch_ch_t055_meridian15_area', 'ch_ch_t035_top2_area', 'strip_v550_dark', 'strip_v650_dark', 'ace_sis_gt10_flux_last', 'ace_epam_p_channel3_last', 'img_age_h', 'rec28'], 'xgboost': {'n_estimators': 900, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 20, 'reg_lambda': 5, 'random_state': 42, 'objective': 'reg:squarederror', 'tree_method': 'hist', 'device': 'cpu', 'multi_strategy': 'multi_output_tree', 'n_jobs': 4}, 'seed': 42}}
CONFIG_SHA256 = "7ac13818a2d63832eb4e660ceb3345eee6ddd8ada333745912a18a9801773ea2"
H = 72


def design(b, names, start, groups):
    ci = [i for i, g in enumerate(names['constant']) if g in groups]
    ei = [i for i, g in enumerate(names['varying']) if g in groups]
    n = len(b['C'])
    return np.hstack([b['C'][:, ci], b['E'][:, start:start + 24, :][:, :, ei].reshape(n, -1)]).astype(np.float32)


def targets(series, origins):
    times = origins.values[:, None] + np.arange(1, H + 1).astype('timedelta64[h]')[None, :]
    return series.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--speed', required=True)
    ap.add_argument('--out', default='predictions.parquet')
    a = ap.parse_args()
    cfg = CONFIG['hyperparameters']
    groups, k = cfg['feature_groups'], cfg['delta_leads']
    params = dict(cfg['xgboost'])
    cache = Path(a.cache)
    names = json.loads((cache / 'names.json').read_text())
    tr = dict(np.load(cache / 'train.npz'))
    series = pd.read_parquet(a.speed)['speed']
    origins = pd.DatetimeIndex(tr['origins'], tz='UTC')
    y = targets(series, origins)
    v0 = names['constant'].index('v_lag0')
    oo, preds = [], []
    for grid, cut in (('f1', CONFIG['data']['f1_training_target_end_utc']),
                      ('official', CONFIG['data']['f2_training_target_end_utc'])):
        g = dict(np.load(cache / f'{grid}.npz'))
        fit = np.flatnonzero(origins + pd.Timedelta(hours=H) < pd.Timestamp(cut))
        assert np.isfinite(y[fit]).all()
        out = {}
        for target in ('level', 'delta'):
            off_tr = tr['C'][fit, v0].astype(float)[:, None] if target == 'delta' else 0.
            off_ev = g['C'][:, v0].astype(float)[:, None] if target == 'delta' else 0.
            p = np.empty((len(g['origins']), H))
            for s0 in (0, 24, 48):
                X, Xt = design(tr, names, s0, groups), design(g, names, s0, groups)
                m = xgb.XGBRegressor(**params).fit(X[fit], y[fit, s0:s0 + 24] - off_tr, verbose=False)
                p[:, s0:s0 + 24] = np.clip(m.predict(Xt) + off_ev, 1., 3000.)
            out[target] = p
        preds.append(np.concatenate([out['delta'][:, :k], out['level'][:, k:]], 1))
        oo.append(pd.DatetimeIndex(g['origins'], tz='UTC'))
    o, pred = oo[0].append(oo[1]), np.vstack(preds)
    frame = pd.DataFrame({'origin_last_input_utc': np.repeat(o.strftime('%Y-%m-%dT%H:%M:%SZ'), H),
                          'horizon_hours': np.tile(np.arange(1, H + 1), len(o)), 'pred_kms': pred.ravel()})
    frame.to_parquet(a.out, index=False)
    print(json.dumps(dict(rows=len(frame), out=a.out, config_sha256=CONFIG_SHA256)))


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f'elapsed {time.time() - t0:.0f}s')
