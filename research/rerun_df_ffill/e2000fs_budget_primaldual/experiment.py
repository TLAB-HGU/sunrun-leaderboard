"""Original structured and budget model reruns on the prepared causal-ffill data.

Features and training targets are causally forward-filled upstream per user request.
Query truth is never loaded. Any leading missing target coordinates remain excluded.
All fitted helpers and calibrations use matured training rows, never query truth.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time
for _name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(_name,'1')
import numpy as np
import pandas as pd
import xgboost as xgb


H = 72

CUTOFFS = {'F1': '2025-12-29', 'F2': '2026-03-29', 'F3': '2026-06-29'}

QUERY_COUNTS = {'F1': 2089, 'F2': 2113, 'F3': 2113}

BASE_NAMES = {'constant': ['v_lag0', 'v_lag1', 'v_lag3', 'v_lag6', 'v_lag12', 'v_lag24', 'v_lag48', 'v_lag72', 'v_lag120', 'v_m6', 'v_sd6', 'v_m24', 'v_sd24', 'v_m72', 'v_sd72', 'ace_speed_kms_last', 'ace_speed_kms_m6', 'ace_speed_kms_m24', 'ace_density_cm3_last', 'ace_density_cm3_m6', 'ace_density_cm3_m24', 'ace_temperature_k_last', 'ace_temperature_k_m6', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6', 'ace_bt_nt_m24', 'ace_bz_gsm_nt_last', 'ace_bz_gsm_nt_m6', 'ace_bz_gsm_nt_m24', 'ace_by_gsm_nt_last', 'ace_by_gsm_nt_m6', 'ace_by_gsm_nt_m24', 'ace_epam_e_channel1_last', 'ace_epam_e_channel1_m6', 'ace_epam_e_channel1_m24', 'ace_epam_e_channel2_last', 'ace_epam_e_channel2_m6', 'ace_epam_e_channel2_m24', 'ace_epam_p_channel1_last', 'ace_epam_p_channel1_m6', 'ace_epam_p_channel1_m24', 'ace_epam_p_channel3_last', 'ace_epam_p_channel3_m6', 'ace_epam_p_channel3_m24', 'ace_epam_p_channel5_last', 'ace_epam_p_channel5_m6', 'ace_epam_p_channel5_m24', 'ace_sis_gt10_flux_last', 'ace_sis_gt10_flux_m6', 'ace_sis_gt10_flux_m24', 'speed_d24', 'epam_p1_d6', 'ch_ch_t035_area', 'ch_ch_t035_max_area', 'ch_ch_t035_top1_area', 'ch_ch_t035_top2_area', 'ch_ch_t035_top3_area', 'ch_ch_t035_meridian15_area', 'ch_ch_t035_meridian15_equatorial_area', 'ch_ch_t035_meridian30_area', 'ch_ch_t035_meridian30_equatorial_area', 'ch_ch_t035_equatorial_area', 'ch_ch_t045_area', 'ch_ch_t045_max_area', 'ch_ch_t045_top1_area', 'ch_ch_t045_top2_area', 'ch_ch_t045_top3_area', 'ch_ch_t045_meridian15_area', 'ch_ch_t045_meridian15_equatorial_area', 'ch_ch_t045_meridian30_area', 'ch_ch_t045_meridian30_equatorial_area', 'ch_ch_t045_equatorial_area', 'ch_ch_t055_area', 'ch_ch_t055_max_area', 'ch_ch_t055_top1_area', 'ch_ch_t055_top2_area', 'ch_ch_t055_top3_area', 'ch_ch_t055_meridian15_area', 'ch_ch_t055_meridian15_equatorial_area', 'ch_ch_t055_meridian30_area', 'ch_ch_t055_meridian30_equatorial_area', 'ch_ch_t055_equatorial_area', 'img_age_h', 'img_stale', 'enlil_next_arr_h', 'enlil_since_arr_h', 'enlil_n_pending', 'enlil_kp', 'enlil_glancing'], 'varying': ['rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed', 'strip_v350_dark', 'strip_v350_cover', 'strip_v450_logmed', 'strip_v450_dark', 'strip_v450_cover', 'strip_v550_logmed', 'strip_v550_dark', 'strip_v550_cover', 'strip_v650_logmed', 'strip_v650_dark', 'strip_v650_cover', 'strip_v750_logmed', 'strip_v750_dark', 'strip_v750_cover', 'enlil_dt_target']}

RECIPES = {'e2000fs_budget_complexity': 'import os\n\nimport argparse\n\nimport json\n\nfrom pathlib import Path\n\nimport numpy as np\n\nimport pandas as pd\n\nimport xgboost as xgb\n\nFEATURES = [\'v_lag0\', \'rec_mean\', \'enlil_kp\', \'ch_ch_t035_meridian30_equatorial_area\', \'ace_bt_nt_last\', \'ace_speed_kms_last\',\n            \'ch_ch_t035_equatorial_area\', \'ch_ch_t035_meridian15_equatorial_area\', \'ace_bt_nt_m6\', \'rec26\',\n            \'strip_v450_logmed\', \'strip_v350_logmed\', \'ch_ch_t055_equatorial_area\', \'rec27\', \'ace_density_cm3_last\',\n            \'ch_ch_t055_meridian30_equatorial_area\', \'ace_by_gsm_nt_m24\', \'ace_density_cm3_m6\',\n            \'ace_temperature_k_last\', \'enlil_n_pending\', \'ace_temperature_k_m24\', \'strip_v750_logmed\',\n            \'ch_ch_t055_meridian15_area\', \'ch_ch_t035_top2_area\', \'strip_v550_dark\', \'strip_v650_dark\',\n            \'ace_sis_gt10_flux_last\', \'ace_epam_p_channel3_last\', \'img_age_h\', \'rec28\']\n\nPARAMS = dict(max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, min_child_weight=20, reg_lambda=5,\n              objective=\'reg:squarederror\', tree_method=\'hist\', device=\'cpu\', eval_metric=\'rmse\',\n              multi_strategy=\'multi_output_tree\', nthread=3, seed=42, num_target=24)\n\nROUNDS = 900\n\nBLOCKS = (0, 24, 48)\n\nOOF_ROUNDS = 100\n\nCONTEXT = (\'v_lag0\', \'ace_bt_nt_last\', \'ace_density_cm3_last\')\n\ndef design(grid, names, start):\n    """Selected constants, then the selected varying features of the block\'s 24 hours, flattened."""\n    constant = [i for i, n in enumerate(names[\'constant\']) if n in FEATURES]\n    varying = [i for i, n in enumerate(names[\'varying\']) if n in FEATURES]\n    block = grid[\'E\'][:, start:start + 24, :][:, :, varying].reshape(len(grid[\'C\']), -1)\n    return np.hstack([grid[\'C\'][:, constant], block]).astype(np.float32)\n\ndef point4_weights(y):\n    """Each target at or above 550 km/s counts four times; every horizon\'s weights have mean 1."""\n    w = np.where(y >= 550., 4., 1.)\n    return (w / w.mean(axis=0)).astype(np.float32)\n\ndef point4_l2(pred, matrix):\n    target = matrix.get_label().reshape(pred.shape)\n    w = point4_weights(target)\n    return (pred - target) * w, w\n\ndef weighted_l2(weight):\n    def objective(pred, matrix):\n        target = matrix.get_label().reshape(pred.shape)\n        return (pred - target) * weight, weight\n    return objective\n\ndef density(previous, query):\n    """Inverse-occupancy factor from octile bins of the three context columns, in [.5, 2]."""\n    previous, query = np.asarray(previous, float), np.asarray(query, float)\n    a, b = [], []\n    for k in range(previous.shape[1]):\n        finite = previous[:, k][np.isfinite(previous[:, k])]\n        median = np.median(finite) if len(finite) else 0.\n        historical = np.where(np.isfinite(previous[:, k]), previous[:, k], median)\n        future = np.where(np.isfinite(query[:, k]), query[:, k], median)\n        edges = np.unique(np.quantile(historical, np.arange(1, 8) / 8))\n        a.append(np.searchsorted(edges, historical))\n        b.append(np.searchsorted(edges, future))\n    counts = np.bincount(np.ravel_multi_index(tuple(a), (8, 8, 8)), minlength=512)\n    return np.clip(np.sqrt(np.median(counts[counts > 0]) / (counts[np.ravel_multi_index(tuple(b), (8, 8, 8))] + 1)), .5, 2)\n\ndef forward_complexity(X, y, origins, context, rounds):\n    """Expanding-window out-of-fold error times context rarity over the last three quarters of the training rows."""\n    raw = np.ones_like(y, dtype=float)\n    active = np.zeros(len(y), dtype=bool)\n    quarters = np.linspace(0, len(y), 5, dtype=int)\n    for fold in range(1, 4):\n        val = np.arange(quarters[fold], quarters[fold + 1])\n        if not len(val):\n            continue\n        fit = np.flatnonzero(origins + pd.Timedelta(hours=144) <= pd.Timestamp(origins[val[0]]))\n        if len(fit) < 32:\n            continue\n        d = density(context[fit], context[val])\n        prediction = np.empty((len(val), 72), dtype=np.float32)\n        for b, start in enumerate(BLOCKS):\n            target = y[fit, start:start + 24].astype(np.float32)\n            model = xgb.train(dict(PARAMS, base_score=float(target.mean())), xgb.DMatrix(X[b][fit], label=target, nthread=3),\n                              num_boost_round=rounds, obj=point4_l2)\n            prediction[:, start:start + 24] = np.clip(model.inplace_predict(X[b][val]), 1, 3000)\n        error = np.clip(np.abs(prediction - y[val]) / 100., 0, 3)\n        raw[val] = np.clip(1 + .25 * (error * d[:, None] - 1), .5, 2)\n        active[val] = True\n    return raw, active\n\ndef mass_weights(y, origins, raw, active):\n    """point4 weights times relative complexity, keeping point4 mass in each year x horizon x high/low group."""\n    base = point4_weights(y)\n    relative = np.ones(y.shape, dtype=np.float64)\n    for year in np.unique(origins.year):\n        for h in range(y.shape[1]):\n            for high in (False, True):\n                chosen = (origins.year == year) & ((y[:, h] >= 550) == high) & active\n                if chosen.sum() >= 32:\n                    relative[chosen, h] = raw[chosen, h] / raw[chosen, h].mean()\n    return (base.astype(float) * relative).astype(np.float32)', 'e2000fs_budget_covshift': 'import os\n\nimport argparse\n\nimport json\n\nfrom pathlib import Path\n\nimport numpy as np\n\nimport pandas as pd\n\nimport xgboost as xgb\n\nFEATURES = [\'v_lag0\', \'rec_mean\', \'enlil_kp\', \'ch_ch_t035_meridian30_equatorial_area\', \'ace_bt_nt_last\', \'ace_speed_kms_last\',\n            \'ch_ch_t035_equatorial_area\', \'ch_ch_t035_meridian15_equatorial_area\', \'ace_bt_nt_m6\', \'rec26\',\n            \'strip_v450_logmed\', \'strip_v350_logmed\', \'ch_ch_t055_equatorial_area\', \'rec27\', \'ace_density_cm3_last\',\n            \'ch_ch_t055_meridian30_equatorial_area\', \'ace_by_gsm_nt_m24\', \'ace_density_cm3_m6\',\n            \'ace_temperature_k_last\', \'enlil_n_pending\', \'ace_temperature_k_m24\', \'strip_v750_logmed\',\n            \'ch_ch_t055_meridian15_area\', \'ch_ch_t035_top2_area\', \'strip_v550_dark\', \'strip_v650_dark\',\n            \'ace_sis_gt10_flux_last\', \'ace_epam_p_channel3_last\', \'img_age_h\', \'rec28\']\n\nPARAMS = dict(max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, min_child_weight=20, reg_lambda=5,\n              objective=\'reg:squarederror\', tree_method=\'hist\', device=\'cpu\', eval_metric=\'rmse\',\n              multi_strategy=\'multi_output_tree\', nthread=3, seed=42, num_target=24)\n\nROUNDS = 900\n\nBLOCKS = (0, 24, 48)\n\nCLASSIFIER_ROUNDS = 80\n\ndef design(grid, names, start):\n    """Selected constants, then the selected varying features of the block\'s 24 hours, flattened."""\n    constant = [i for i, n in enumerate(names[\'constant\']) if n in FEATURES]\n    varying = [i for i, n in enumerate(names[\'varying\']) if n in FEATURES]\n    block = grid[\'E\'][:, start:start + 24, :][:, :, varying].reshape(len(grid[\'C\']), -1)\n    return np.hstack([grid[\'C\'][:, constant], block]).astype(np.float32)\n\ndef point4_weights(y):\n    """Each target at or above 550 km/s counts four times; every horizon\'s weights have mean 1."""\n    w = np.where(y >= 550., 4., 1.)\n    return (w / w.mean(axis=0)).astype(np.float32)\n\ndef weighted_l2(weight):\n    def objective(pred, matrix):\n        target = matrix.get_label().reshape(pred.shape)\n        return (pred - target) * weight, weight\n    return objective\n\ndef density_ratio(x, origins, cutoff, rounds):\n    """Past-only cross-fitted p_recent / p_all over 27 day blocks, with 72 h embargo; clipped to [.25, 4]."""\n    index = np.flatnonzero(origins < cutoff)\n    xx, oo = x[index], origins[index]\n    recent = np.asarray(oo >= cutoff - pd.Timedelta(days=81))\n    day27, day3 = pd.Timedelta(days=27).value, pd.Timedelta(hours=72).value\n    block = oo.asi8 // day27\n    folds = block % 3\n    raw = np.empty(len(index))\n    for fold in range(3):\n        held = folds == fold\n        embargo = np.zeros(len(index), dtype=bool)\n        for value in np.unique(block[held]):\n            embargo |= (oo.asi8 >= value * day27 - day3) & (oo.asi8 < (value + 1) * day27 + day3)\n        source, recent_rows = np.flatnonzero(~embargo), np.flatnonzero(~embargo & recent)\n        labels = np.r_[np.zeros(len(source)), np.ones(len(recent_rows))]\n        model = xgb.train(dict(objective=\'binary:logistic\', tree_method=\'hist\', device=\'cpu\', max_depth=2, eta=.05,\n                               min_child_weight=20, reg_lambda=5, subsample=1., colsample_bytree=1., seed=42, nthread=3,\n                               base_score=len(recent_rows) / len(labels)),\n                          xgb.DMatrix(np.vstack([xx[source], xx[recent_rows]]), label=labels, nthread=3), num_boost_round=rounds)\n        q = model.inplace_predict(xx[held]).astype(float)\n        raw[held] = q / np.maximum(1 - q, 1e-12) * len(source) / len(recent_rows)\n    return index, np.clip(raw, .25, 4.)\n\ndef shifted_weights(y, origins, ratio):\n    """point4 weights times the ratio, rescaled to keep point4 mass in each year x horizon x speed band."""\n    base = point4_weights(y)\n    result = base.astype(float) * np.asarray(ratio)[:, None]\n    bands = np.digitize(y, [400., 550., 700.])\n    eras = np.asarray(origins.year)\n    for era in np.unique(eras):\n        for h in range(y.shape[1]):\n            for band in range(4):\n                mask = (eras == era) & (bands[:, h] == band)\n                if mask.any():\n                    result[mask, h] *= float(base[mask, h].sum(dtype=float)) / result[mask, h].sum()\n    return result.astype(np.float32)', 'e2000fs_budget_primaldual': 'import os\n\nimport argparse\n\nimport json\n\nfrom pathlib import Path\n\nimport numpy as np\n\nimport pandas as pd\n\nimport xgboost as xgb\n\nFEATURES = [\'v_lag0\', \'rec_mean\', \'enlil_kp\', \'ch_ch_t035_meridian30_equatorial_area\', \'ace_bt_nt_last\', \'ace_speed_kms_last\',\n            \'ch_ch_t035_equatorial_area\', \'ch_ch_t035_meridian15_equatorial_area\', \'ace_bt_nt_m6\', \'rec26\',\n            \'strip_v450_logmed\', \'strip_v350_logmed\', \'ch_ch_t055_equatorial_area\', \'rec27\', \'ace_density_cm3_last\',\n            \'ch_ch_t055_meridian30_equatorial_area\', \'ace_by_gsm_nt_m24\', \'ace_density_cm3_m6\',\n            \'ace_temperature_k_last\', \'enlil_n_pending\', \'ace_temperature_k_m24\', \'strip_v750_logmed\',\n            \'ch_ch_t055_meridian15_area\', \'ch_ch_t035_top2_area\', \'strip_v550_dark\', \'strip_v650_dark\',\n            \'ace_sis_gt10_flux_last\', \'ace_epam_p_channel3_last\', \'img_age_h\', \'rec28\']\n\nPARAMS = dict(max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, min_child_weight=20, reg_lambda=5,\n              objective=\'reg:squarederror\', tree_method=\'hist\', device=\'cpu\', eval_metric=\'rmse\',\n              multi_strategy=\'multi_output_tree\', nthread=3, seed=42, num_target=24)\n\nROUNDS = 900\n\nBLOCKS = (0, 24, 48)\n\nDUAL_STEP = .02\n\ndef design(grid, names, start):\n    """Selected constants, then the selected varying features of the block\'s 24 hours, flattened."""\n    constant = [i for i, n in enumerate(names[\'constant\']) if n in FEATURES]\n    varying = [i for i, n in enumerate(names[\'varying\']) if n in FEATURES]\n    block = grid[\'E\'][:, start:start + 24, :][:, :, varying].reshape(len(grid[\'C\']), -1)\n    return np.hstack([grid[\'C\'][:, constant], block]).astype(np.float32)\n\ndef point4_weights(y):\n    """Each target at or above 550 km/s counts four times; every horizon\'s weights have mean 1."""\n    w = np.where(y >= 550., 4., 1.)\n    return (w / w.mean(axis=0)).astype(np.float32)\n\ndef point4_l2(pred, matrix):\n    target = matrix.get_label().reshape(pred.shape)\n    w = point4_weights(target)\n    return (pred - target) * w, w\n\nclass RiskObjective:\n    """point4-like high-speed weight whose dual grows while training high-speed MSE exceeds the budget."""\n\n    def __init__(self, target, budget):\n        self.high = np.asarray(target) >= 550.\n        self.p = float(self.high.mean())\n        self.budget = float(budget)\n        self.dual = 3. * self.p\n\n    def __call__(self, pred, matrix):\n        y = matrix.get_label().reshape(pred.shape)\n        if self.p:\n            a = ((1. + self.dual * self.high / self.p) / (1. + self.dual)).astype(np.float32)\n        else:\n            a = np.ones(self.high.shape, dtype=np.float32)\n        return (pred - y) * a, a\n\nclass DualUpdate(xgb.callback.TrainingCallback):\n    def __init__(self, objective):\n        self.objective = objective\n\n    def after_iteration(self, model, epoch, evals_log):\n        obj = self.objective\n        high_risk = evals_log[\'train\'][\'high_mse\'][-1]\n        obj.dual = float(np.clip(obj.dual + DUAL_STEP * (high_risk / obj.budget - 1.), 0., 10.)) if obj.p else 0.\n        return False\n\ndef high_mse(pred, matrix):\n    target = matrix.get_label().reshape(pred.shape)\n    error = (pred - target) ** 2\n    high = target >= 550.\n    return \'high_mse\', float(error[high].mean()) if high.any() else 0.'}

def selected_names(features):
    return {key: [n for n in values if n in features] for key, values in BASE_NAMES.items()}

def mapped(name):
    return name.replace('ch_ch_', 'suvi_', 1) if name.startswith('ch_ch_') else name

def load_grid(data, period, names, training):
    filters = [(f'fit_{period}', '==', True)] if training else [('set', '==', 'query'), ('period', '==', period)]
    constant = pd.read_parquet(data, columns=['origin', *[mapped(c) for c in names['constant']]], filters=[*filters, ('h', '==', 1)]).sort_values('origin', kind='stable')
    origins = pd.DatetimeIndex(pd.to_datetime(constant.origin, utc=True))
    if origins.has_duplicates:
        raise ValueError('duplicate input origins')
    columns = ['origin', 'h', *names['varying']] + (['y'] if training else [])
    rows = pd.read_parquet(data, columns=columns, filters=filters).sort_values(['origin', 'h'], kind='stable')
    if len(rows) != len(origins) * H or not np.array_equal(rows.h.to_numpy().reshape(-1, H), np.broadcast_to(np.arange(1, H + 1), (len(origins), H))):
        raise ValueError('incomplete origin/horizon grid')
    if not origins.equals(pd.DatetimeIndex(pd.to_datetime(rows.origin.iloc[::H], utc=True))):
        raise ValueError('constant and varying grids disagree')
    if training and (not (origins + pd.Timedelta(hours=144) <= pd.Timestamp(CUTOFFS[period], tz='UTC')).all()):
        raise ValueError('training cutoff violated')
    grid = {'origins': origins, 'C': constant[[mapped(c) for c in names['constant']]].to_numpy(np.float32), 'E': rows[names['varying']].to_numpy(np.float32).reshape(len(origins), H, len(names['varying']))}
    return (grid, rows.y.to_numpy(np.float64).reshape(len(origins), H) if training else None)

def design(grid, start):
    return np.hstack([grid['C'], grid['E'][:, start:start + 24].reshape(len(grid['C']), -1)]).astype(np.float32)

def merge(out):
    out = Path(out)
    frames = []
    for period in CUTOFFS:
        z = np.load(out / f'{period}_predictions.npz')
        origins = pd.DatetimeIndex(z['origins'], tz='UTC')
        pred = z['pred']
        if len(origins) != QUERY_COUNTS[period] or pred.shape != (len(origins), H) or (not np.isfinite(pred).all()):
            raise ValueError(f'incomplete full predictions for {period}')
        frames.append(pd.DataFrame({'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), H), 'horizon_hours': np.tile(np.arange(1, H + 1), len(origins)), 'pred_kms': pred.ravel()}))
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(['origin_last_input_utc', 'horizon_hours']).any():
        raise ValueError('duplicate prediction keys')
    result.to_parquet(out / 'predictions.parquet', index=False)
    return len(result)

RECIPES['e2000fs_feat30_probtraj'] = 'import os\n\nimport argparse\n\nimport json\n\nfrom pathlib import Path\n\nimport numpy as np\n\nimport pandas as pd\n\nfrom scipy import sparse\n\nimport xgboost as xgb\n\nFEATURES = [\'v_lag0\', \'rec_mean\', \'enlil_kp\', \'ch_ch_t035_meridian30_equatorial_area\', \'ace_bt_nt_last\', \'ace_speed_kms_last\',\n            \'ch_ch_t035_equatorial_area\', \'ch_ch_t035_meridian15_equatorial_area\', \'ace_bt_nt_m6\', \'rec26\',\n            \'strip_v450_logmed\', \'strip_v350_logmed\', \'ch_ch_t055_equatorial_area\', \'rec27\', \'ace_density_cm3_last\',\n            \'ch_ch_t055_meridian30_equatorial_area\', \'ace_by_gsm_nt_m24\', \'ace_density_cm3_m6\',\n            \'ace_temperature_k_last\', \'enlil_n_pending\', \'ace_temperature_k_m24\', \'strip_v750_logmed\',\n            \'ch_ch_t055_meridian15_area\', \'ch_ch_t035_top2_area\', \'strip_v550_dark\', \'strip_v650_dark\',\n            \'ace_sis_gt10_flux_last\', \'ace_epam_p_channel3_last\', \'img_age_h\', \'rec28\']\n\nPARAMS = dict(n_estimators=900, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,\n              min_child_weight=20, reg_lambda=5, n_jobs=3, random_state=42, objective=\'reg:squarederror\',\n              tree_method=\'hist\', device=\'cpu\', eval_metric=\'rmse\', multi_strategy=\'multi_output_tree\')\n\nBLOCKS = (0, 24, 48)\n\ndef design(grid, names, start):\n    """Selected constants, then the selected varying features of the block\'s 24 hours, flattened."""\n    constant = [i for i, n in enumerate(names[\'constant\']) if n in FEATURES]\n    varying = [i for i, n in enumerate(names[\'varying\']) if n in FEATURES]\n    block = grid[\'E\'][:, start:start + 24, :][:, :, varying].reshape(len(grid[\'C\']), -1)\n    return np.hstack([grid[\'C\'][:, constant], block]).astype(np.float32)\n\ndef leaf_codes(leaves):\n    """Per-tree leaf vocabularies, for sparse inverse-occupancy proximity."""\n    values, offsets, inverse_counts, codes, offset = [], [], [], np.empty(leaves.shape, dtype=np.int32), 0\n    for tree in range(leaves.shape[1]):\n        v, inverse, counts = np.unique(leaves[:, tree], return_inverse=True, return_counts=True)\n        values.append(v)\n        offsets.append(offset)\n        codes[:, tree] = inverse + offset\n        inverse_counts.append(1. / counts)\n        offset += len(v)\n    return values, offsets, np.concatenate(inverse_counts), codes, offset\n\ndef trajectory_means(fit_leaves, fit_targets, query_leaves):\n    """Each query averages the K training trajectories sharing the most (rare) leaves with it, or all if none do."""\n    n, trees = fit_leaves.shape\n    values, offsets, inverse_counts, codes, width = leaf_codes(fit_leaves)\n    membership = sparse.csr_matrix((inverse_counts[codes.ravel()], codes.ravel(),\n                                    np.arange(0, (n + 1) * trees, trees, dtype=np.int64)), shape=(n, width)).T\n    means = []\n    for first in range(0, len(query_leaves), BATCH):\n        batch = query_leaves[first:first + BATCH]\n        rows, cols = [], []\n        for tree, (v, offset) in enumerate(zip(values, offsets)):\n            at = np.searchsorted(v, batch[:, tree])\n            valid = at < len(v)\n            valid[valid] &= v[at[valid]] == batch[valid, tree]\n            rows.append(np.flatnonzero(valid))\n            cols.append(at[valid] + offset)\n        rows, cols = np.concatenate(rows), np.concatenate(cols)\n        query = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(batch), width))\n        for score in (query @ membership).toarray():\n            donors = np.flatnonzero(score > 0)\n            if not len(donors):\n                means.append(np.full(len(fit_targets), 1. / len(fit_targets)) @ fit_targets)\n                continue\n            donors = donors[np.lexsort((donors, -score[donors]))[:K]]\n            means.append(score[donors] / score[donors].sum() @ fit_targets[donors])\n    return np.asarray(means)'

RECIPES['e2000fs_feat30_probtraj'] += '\n\nK, BATCH = 128, 8'

RECIPES['e2000fs_feat30_agnostic_v2_context'] = 'import os\n\nimport argparse\n\nimport json\n\nfrom pathlib import Path\n\nimport numpy as np\n\nimport pandas as pd\n\nfrom scipy.special import expit\n\nimport xgboost as xgb\n\nFEATURES = [\'v_lag0\', \'rec_mean\', \'enlil_kp\', \'ch_ch_t035_meridian30_equatorial_area\', \'ace_bt_nt_last\', \'ace_speed_kms_last\',\n            \'ch_ch_t035_equatorial_area\', \'ch_ch_t035_meridian15_equatorial_area\', \'ace_bt_nt_m6\', \'rec26\',\n            \'strip_v450_logmed\', \'strip_v350_logmed\', \'ch_ch_t055_equatorial_area\', \'rec27\', \'ace_density_cm3_last\',\n            \'ch_ch_t055_meridian30_equatorial_area\', \'ace_by_gsm_nt_m24\', \'ace_density_cm3_m6\',\n            \'ace_temperature_k_last\', \'enlil_n_pending\', \'ace_temperature_k_m24\', \'strip_v750_logmed\',\n            \'ch_ch_t055_meridian15_area\', \'ch_ch_t035_top2_area\', \'strip_v550_dark\', \'strip_v650_dark\',\n            \'ace_sis_gt10_flux_last\', \'ace_epam_p_channel3_last\', \'img_age_h\', \'rec28\']\n\nPARAMS = dict(n_estimators=900, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,\n              min_child_weight=20, reg_lambda=5, n_jobs=3, random_state=42, objective=\'reg:squarederror\',\n              tree_method=\'hist\', device=\'cpu\', eval_metric=\'rmse\', multi_strategy=\'multi_output_tree\')\n\nBLOCKS = (0, 24, 48)\n\nSTRENGTH, MARGIN, SCALE = .25, 20., 10.\n\ndef design(grid, names, start):\n    """Selected constants, then the selected varying features of the block\'s 24 hours, flattened."""\n    constant = [i for i, n in enumerate(names[\'constant\']) if n in FEATURES]\n    varying = [i for i, n in enumerate(names[\'varying\']) if n in FEATURES]\n    block = grid[\'E\'][:, start:start + 24, :][:, :, varying].reshape(len(grid[\'C\']), -1)\n    return np.hstack([grid[\'C\'][:, constant], block]).astype(np.float32)\n\ndef purged_folds(origins):\n    """Five contiguous folds; origins within 72 h of the held fold leave its training set."""\n    origins = np.asarray(origins).astype(\'datetime64[ns]\')\n    gap = np.timedelta64(72, \'h\')\n    for held in np.array_split(np.arange(len(origins)), 5):\n        yield np.flatnonzero((origins < origins[held[0]] - gap) | (origins > origins[held[-1]] + gap)), held\n\ndef event_probabilities(X, block_y, origins, rounds):\n    """Out-of-fold probability that this 24 h block reaches 550 km/s or rises 100 km/s in 6 h."""\n    event = ((block_y.max(1) >= 550) | ((block_y[:, 6:] - block_y[:, :-6]).max(1) >= 100)).astype(np.int32)\n    params = {k: v for k, v in PARAMS.items() if k != \'multi_strategy\'}\n    params.update(n_estimators=rounds, objective=\'binary:logistic\', eval_metric=\'logloss\')\n    probabilities = np.full(len(event), np.nan)\n    for fit, held in purged_folds(origins):\n        if len(np.unique(event[fit])) < 2:\n            probabilities[held] = event[fit].mean()\n            continue\n        model = xgb.XGBClassifier(**params, base_score=float(np.clip(np.mean(event[fit]), 1e-4, 1 - 1e-4)))\n        model.fit(X[fit], np.asarray(event[fit], dtype=np.float32), verbose=False)\n        probabilities[held] = model.get_booster().predict(xgb.DMatrix(X[held], nthread=3))\n    return probabilities\n\ndef context_loss(q):\n    """Squared error plus a softplus penalty on under-forecasts beyond MARGIN, scaled by the event probability."""\n    q = np.asarray(q, dtype=float).reshape(-1, 1)\n\n    def objective(y_true, y_pred):\n        p = np.asarray(y_pred)\n        e = np.asarray(y_true).reshape(p.shape) - p\n        probability = expit((e - MARGIN) / SCALE)\n        return -e - STRENGTH * q * SCALE * probability, np.ones_like(p) + STRENGTH * q * probability * (1 - probability)\n\n    return objective'

RECIPES['e2000fs_feat30_agnostic_v2_event'] = 'import os\n\nimport argparse\n\nimport json\n\nfrom pathlib import Path\n\nfrom joblib import Parallel, delayed\n\nimport numpy as np\n\nimport pandas as pd\n\nfrom scipy.optimize import least_squares\n\nimport xgboost as xgb\n\nFEATURES = [\'v_lag0\', \'rec_mean\', \'enlil_kp\', \'ch_ch_t035_meridian30_equatorial_area\', \'ace_bt_nt_last\', \'ace_speed_kms_last\',\n            \'ch_ch_t035_equatorial_area\', \'ch_ch_t035_meridian15_equatorial_area\', \'ace_bt_nt_m6\', \'rec26\',\n            \'strip_v450_logmed\', \'strip_v350_logmed\', \'ch_ch_t055_equatorial_area\', \'rec27\', \'ace_density_cm3_last\',\n            \'ch_ch_t055_meridian30_equatorial_area\', \'ace_by_gsm_nt_m24\', \'ace_density_cm3_m6\',\n            \'ace_temperature_k_last\', \'enlil_n_pending\', \'ace_temperature_k_m24\', \'strip_v750_logmed\',\n            \'ch_ch_t055_meridian15_area\', \'ch_ch_t035_top2_area\', \'strip_v550_dark\', \'strip_v650_dark\',\n            \'ace_sis_gt10_flux_last\', \'ace_epam_p_channel3_last\', \'img_age_h\', \'rec28\']\n\nPARAMS = dict(n_estimators=900, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,\n              min_child_weight=20, reg_lambda=5, n_jobs=3, random_state=42, objective=\'reg:squarederror\',\n              tree_method=\'hist\', device=\'cpu\', eval_metric=\'rmse\', multi_strategy=\'multi_output_tree\')\n\nLOWER = np.array([1., 0., 0., 1., 0., 0., 1.])\n\nUPPER = np.array([3000., 1500., 72., 48., 1500., 72., 48.])\n\nHORIZONS = np.arange(1., 73.)\n\nWORKERS = 8\n\ndef design_all(grid, names):\n    """Selected constants, then the selected varying features of all 72 hours, flattened."""\n    constant = [i for i, n in enumerate(names[\'constant\']) if n in FEATURES]\n    varying = [i for i, n in enumerate(names[\'varying\']) if n in FEATURES]\n    return np.hstack([grid[\'C\'][:, constant], grid[\'E\'][:, :, varying].reshape(len(grid[\'C\']), -1)]).astype(np.float32)\n\ndef event_curve(params):\n    """Baseline, amplitude, onset, rise duration, decay, plateau gap, decay duration -> speeds at horizons 1..72."""\n    params = np.clip(np.asarray(params, dtype=np.float64), LOWER, UPPER)\n    b, a, tau, duration, c, gap, decay_duration = params.T[:, :, None]\n    rise = np.clip((HORIZONS - tau) / duration, 0., 1.)\n    decay = np.clip((HORIZONS - tau - duration - gap) / decay_duration, 0., 1.)\n    return np.clip(b + a * rise - c * decay, 1., 3000.)\n\ndef curve_jacobian(params):\n    b, a, tau, duration, c, gap, decay_duration = params\n    u = (HORIZONS - tau) / duration\n    v = (HORIZONS - tau - duration - gap) / decay_duration\n    rise, decay = np.clip(u, 0., 1.), np.clip(v, 0., 1.)\n    active_u, active_v = (u > 0.) & (u < 1.), (v > 0.) & (v < 1.)\n    raw = b + a * rise - c * decay\n    jac = np.column_stack([np.ones(72), rise, -a / duration * active_u + c / decay_duration * active_v,\n                           -a * u / duration * active_u + c / decay_duration * active_v, -decay,\n                           c / decay_duration * active_v, c * v / decay_duration * active_v])\n    jac[(raw < 1.) | (raw > 3000.)] = 0.\n    return np.clip(raw, 1., 3000.), jac\n\ndef fit_curve(target):\n    """Bounded least squares from several starts on one observed 72 h trajectory; lowest SSE wins."""\n    changes = target[6:] - target[:-6]\n    no_rise = changes.max() <= 0.\n    baseline = np.clip(target[:3].mean(), LOWER[0], UPPER[0])\n    amplitude = 0. if no_rise else np.clip(target.max() - baseline, 1., 1500.)\n    decay = np.clip(target.max() - target[-3:].mean(), 0., 1500.)\n    starts = [np.array([baseline, amplitude, tau, 6., decay, 12., 12.])\n              for tau in sorted({0., 24., 48., float(np.argmax(changes) + 1)})]\n    if no_rise:  # amplitude stays zero; an extra immediate-decay start\n        starts.append(np.array([np.clip(target[0], 1., 3000.), 0., 0., 1., np.clip(target[0] - target[-1], 0., 1500.), 0., 48.]))\n    free = np.array([0, 2, 3, 4, 5, 6] if no_rise else range(7))\n\n    def full(values):\n        result = np.zeros(7)\n        result[free] = values\n        return result\n\n    candidates = []\n    for start in starts:\n        start = np.clip(start, LOWER, UPPER)\n        result = least_squares(lambda v: curve_jacobian(full(v))[0] - target, start[free], jac=lambda v: curve_jacobian(full(v))[1][:, free],\n                               bounds=(LOWER[free], UPPER[free]), x_scale=\'jac\', max_nfev=80)\n        fitted = full(result.x)\n        error = event_curve(fitted[None, :])[0] - target\n        candidates.append((float(error @ error), float(np.linalg.norm((fitted - LOWER) / (UPPER - LOWER))), tuple(fitted)))\n    return np.array(min(candidates)[2])'

SUPPORTED = tuple(RECIPES)
UNSUPPORTED = {}


def kernel(experiment, threads):
    env={'__name__':'frozen_structured_recipe'}
    # These are resource controls only; objectives and statistical settings remain frozen.
    source=RECIPES[experiment].replace('nthread=3','nthread=THREADS')
    env['THREADS']=threads
    exec(compile(source,'<original:'+experiment+'>','exec'),env)
    env['PARAMS']=dict(env['PARAMS'],nthread=threads)
    if 'agnostic_v2_' in experiment:
        env['PARAMS']['n_jobs']=threads
        env['ROUNDS']=env['PARAMS']['n_estimators']
    if experiment.endswith('probtraj'):
        env['ROUNDS']=env['PARAMS'].pop('n_estimators')
        env['PARAMS'].pop('n_jobs')
        env['PARAMS']['seed']=env['PARAMS'].pop('random_state')
        env['PARAMS']['num_target']=24
    return env


def point_weights(target):
    observed=np.isfinite(target)
    count=observed.sum(0)
    if not count.all():
        raise ValueError('An output has no observed training targets')
    raw=np.where(target>=550.,4.,1.)*observed
    return (raw/(raw.sum(0)/count)).astype(np.float32)


def weighted_objective(weights):
    def objective(pred,matrix):
        target=matrix.get_label().reshape(pred.shape)
        return (pred-target)*weights, weights
    return objective


def train_model(env, x, y, rounds, weights=None, objective=None, **kwargs):
    # Zero is solely an XGBoost storage sentinel. Objective gradients and Hessians
    # are zero at every unobserved target and exclude sentinel values from statistics.
    matrix=xgb.DMatrix(x,label=np.nan_to_num(y,nan=0.).astype(np.float32),nthread=env['THREADS'])
    if objective is None:
        objective=weighted_objective(point_weights(y) if weights is None else weights)
    model=xgb.train(dict(env['PARAMS'],base_score=float(np.nanmean(y))),matrix,
                    num_boost_round=rounds,obj=objective,**kwargs)
    return model


def forward_complexity(env, X, y, origins, context, rounds):
    raw=np.ones_like(y,dtype=float)
    active=np.zeros(len(y),dtype=bool)
    quarters=np.linspace(0,len(y),5,dtype=int)
    for fold in range(1,4):
        val=np.arange(quarters[fold],quarters[fold+1])
        if not len(val):
            continue
        fit=np.flatnonzero(origins+pd.Timedelta(hours=144)<=origins[val[0]])
        if len(fit)<32:
            continue
        density=env['density'](context[fit],context[val])
        prediction=np.empty((len(val),72),dtype=np.float32)
        for b,start in enumerate((0,24,48)):
            model=train_model(env,X[b][fit],y[fit,start:start+24],rounds)
            prediction[:,start:start+24]=np.clip(model.inplace_predict(X[b][val]),1,3000)
        error=np.clip(np.abs(prediction-y[val])/100.,0,3)
        raw[val]=np.clip(1+.25*(error*density[:,None]-1),.5,2)
        active[val]=True
    return raw,active


def mass_weights(y,origins,raw,active):
    base=point_weights(y)
    relative=np.ones(y.shape,dtype=np.float64)
    for year in np.unique(origins.year):
        for h in range(y.shape[1]):
            for high in (False,True):
                chosen=(origins.year==year)&((y[:,h]>=550)==high)&active&np.isfinite(y[:,h])
                if chosen.sum()>=32:
                    relative[chosen,h]=raw[chosen,h]/raw[chosen,h].mean()
    return (base.astype(float)*relative).astype(np.float32)


def shifted_weights(y,origins,ratio):
    base=point_weights(y)
    result=base.astype(float)*np.asarray(ratio)[:,None]
    bands=np.digitize(y,[400.,550.,700.])
    for year in np.unique(origins.year):
        for h in range(y.shape[1]):
            for band in range(4):
                mask=(origins.year==year)&(bands[:,h]==band)&np.isfinite(y[:,h])
                if mask.any():
                    result[mask,h]*=float(base[mask,h].sum(dtype=float))/result[mask,h].sum()
    return result.astype(np.float32)


class MaskedRiskObjective:
    def __init__(self,target,budget):
        self.observed=np.isfinite(target)
        self.high=np.asarray(target)>=550.
        self.p=float(self.high.sum()/self.observed.sum())
        self.budget=float(budget)
        self.dual=3.*self.p

    def __call__(self,pred,matrix):
        y=matrix.get_label().reshape(pred.shape)
        if self.p:
            a=((1.+self.dual*self.high/self.p)/(1.+self.dual)).astype(np.float32)
        else:
            a=np.ones(self.high.shape,dtype=np.float32)
        a*=self.observed
        return (pred-y)*a,a


def load_classifier_grid(data,names,cutoff):
    # The original covariate classifier uses all pre-cutoff train contexts,
    # including rows whose target windows have not matured. It reads no labels.
    filters=[('set','==','train'),('origin','<',cutoff.to_pydatetime())]
    c=pd.read_parquet(data,columns=['origin',*[mapped(n) for n in names['constant']]],filters=[*filters,('h','==',1)]).sort_values('origin')
    e=pd.read_parquet(data,columns=['origin','h',*names['varying']],filters=filters).sort_values(['origin','h'])
    origins=pd.DatetimeIndex(pd.to_datetime(c.origin,utc=True))
    if origins.has_duplicates or len(e)!=72*len(c):
        raise ValueError('Classifier context grid is incomplete')
    if not np.array_equal(e.h.to_numpy().reshape(-1,72),np.broadcast_to(np.arange(1,73),(len(c),72))):
        raise ValueError('Classifier horizon ordering mismatch')
    if not origins.equals(pd.DatetimeIndex(pd.to_datetime(e.origin.iloc[::72],utc=True))):
        raise ValueError('Classifier origin ordering mismatch')
    return dict(origins=origins,C=c[[mapped(n) for n in names['constant']]].to_numpy(np.float32),
                E=e[names['varying']].to_numpy(np.float32).reshape(len(c),72,len(names['varying'])))


def fit_predict(experiment,env,train,query,y,names,cutoff,out,smoke=False,classifier=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if experiment.endswith('probtraj'):
        return fit_probtraj(env,train,query,y,out,smoke)
    if 'agnostic_v2_' in experiment:
        return fit_agnostic(experiment,env,train,query,y,names,out,smoke)
    X=[design(train,s) for s in (0,24,48)]
    Xq=[design(query,s) for s in (0,24,48)]
    origins=train['origins']; rounds=2 if smoke else env['ROUNDS']
    began=time.perf_counter()
    weights=None
    auxiliary={}
    if experiment.endswith('complexity'):
        context=train['C'][:,[names['constant'].index(n) for n in env['CONTEXT']]]
        raw,active=forward_complexity(env,X,y,origins,context,2 if smoke else env['OOF_ROUNDS'])
        weights=mass_weights(y,origins,raw,active)
        np.savez_compressed(out/'complexity_state.npz',raw=raw,active=active,weights=weights)
        auxiliary['active_complexity_origins']=int(active.sum())
    elif experiment.endswith('covshift'):
        if classifier is None:
            raise ValueError('Covshift requires the complete pre-cutoff classifier context grid')
        index,ratio=env['density_ratio'](design(classifier,0),classifier['origins'],cutoff,3 if smoke else env['CLASSIFIER_ROUNDS'])
        at=classifier['origins'][index].get_indexer(origins)
        if (at<0).any():
            raise ValueError('Training origin missing from covariate classifier grid')
        weights=shifted_weights(y,origins,ratio[at])
        np.savez_compressed(out/'covshift_state.npz',origins=classifier['origins'][index].asi8,ratio=ratio,weights=weights)
        auxiliary['classifier_origins']=len(index)
    auxiliary['fit_seconds']=time.perf_counter()-began
    predictions=np.empty((len(query['origins']),72),np.float64)
    reports=[]
    for b,start in enumerate((0,24,48)):
        target=y[:,start:start+24].astype(np.float32)
        began=time.perf_counter()
        details={}
        if experiment.endswith('primaldual'):
            start_calibration=cutoff-pd.Timedelta(days=90)
            reference=np.flatnonzero(origins+pd.Timedelta(hours=144)<=start_calibration)
            calibration=np.flatnonzero((origins>=start_calibration)&(origins+pd.Timedelta(hours=144)<=cutoff))
            if not len(reference) or not len(calibration):
                raise ValueError('No prior reference/calibration observations for primaldual budget')
            reference_y=target[reference];calibration_y=target[calibration]
            parent=train_model(env,X[b][reference],reference_y,rounds)
            parent.save_model(out/f'block{b}_reference.ubj')
            high=calibration_y>=550.
            if not high.any():
                raise ValueError('No observed high-speed targets in prior calibration span')
            budget=max(1.,float(np.mean((np.clip(parent.inplace_predict(X[b][calibration]),1.,3000.)-calibration_y)[high]**2)))
            objective=MaskedRiskObjective(target,budget)
            matrix=xgb.DMatrix(X[b],label=np.nan_to_num(target),nthread=env['THREADS'])
            model=xgb.train(dict(env['PARAMS'],base_score=float(np.nanmean(target))),matrix,num_boost_round=rounds,obj=objective,
                            evals=[(matrix,'train')],custom_metric=env['high_mse'],callbacks=[env['DualUpdate'](objective)],verbose_eval=False)
            details=dict(budget=budget,final_dual=objective.dual,reference_rows=len(reference),calibration_rows=len(calibration),calibration_high_targets=int(high.sum()))
        else:
            model=train_model(env,X[b],target,rounds,weights=weights[:,start:start+24])
        fit_seconds=time.perf_counter()-began
        model.save_model(out/f'block{b}.ubj')
        began=time.perf_counter()
        predictions[:,start:start+24]=np.clip(model.inplace_predict(Xq[b]),1.,3000.)
        inference_seconds=time.perf_counter()-began
        reports.append(dict(block=b,fit_seconds=fit_seconds,inference_seconds=inference_seconds,
                            observed_training_targets=int(np.isfinite(target).sum()),**details))
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite predictions')
    return predictions,reports,auxiliary


def run_period(experiment,data,out,period,threads=1,smoke=False):
    env=kernel(experiment,threads);names=selected_names(env['FEATURES'])
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    began=time.perf_counter()
    train,y=load_grid(data,period,names,True)
    query,_=load_grid(data,period,names,False)
    if len(query['origins'])!=QUERY_COUNTS[period]:
        raise ValueError('Unexpected query grid')
    cutoff=pd.Timestamp(CUTOFFS[period],tz='UTC')
    classifier=load_classifier_grid(data,names,cutoff) if experiment.endswith('covshift') else None
    if smoke:
        keep=np.linspace(0,len(y)-1,min(len(y),32 if experiment.endswith('agnostic_v2_event') else 512)).astype(int)
        train={k:v[keep] for k,v in train.items()};y=y[keep]
        query={k:v[:8] for k,v in query.items()}
    pred,reports,aux=fit_predict(experiment,env,train,query,y,names,cutoff,out/f'{period}_models',smoke,classifier)
    np.savez_compressed(out/f'{period}_predictions.npz',pred=pred,origins=query['origins'].asi8)
    revision=subprocess.run(['git','rev-parse','HEAD'],text=True,capture_output=True,check=False)
    metadata=dict(experiment=experiment,period=period,smoke=smoke,seed=42,data=str(Path(data).resolve()),
                  data_bytes=Path(data).stat().st_size,data_mtime_ns=Path(data).stat().st_mtime_ns,
                  git_commit=revision.stdout.strip(),features=names,parameters=env['PARAMS'],rounds=(3 if 'agnostic_v2_' in experiment else 2) if smoke else env['ROUNDS'],
                  training_cutoff=CUTOFFS[period],maturity_hours=144,train_origins=len(y),query_origins=len(pred),
                  training_start=str(train['origins'].min()),training_end=str(train['origins'].max()),
                  target_policy='Upstream causal ffill applies to features and training y only; query truth is never loaded. Residual missing objective coordinates are masked; agnostic curve/event fits require complete relevant windows.',
                  complete72_training_rows=int(np.isfinite(y).all(1).sum()),observed_training_targets=int(np.isfinite(y).sum()),
                  inference_seconds_per_fold=sum(r['inference_seconds'] for r in reports)/len(pred),fits=reports,auxiliary=aux,
                  end_to_end_seconds=time.perf_counter()-began,
                  versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,xgboost=xgb.__version__))
    (out/f'{period}_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def observed_donor_mean(weights,targets):
    observed=np.isfinite(targets)
    mass=np.asarray(weights,dtype=float)@observed
    if not (mass>0).all():
        raise ValueError('Selected donor set has no observed target for a forecast horizon')
    return (weights@np.nan_to_num(targets,nan=0.))/mass


def trajectory_means(env,fit_leaves,fit_targets,query_leaves):
    """Original leaf vocabulary, occupancy weighting and top-K donor selection.

    Only the final mean excludes missing donor coordinates and renormalizes each
    horizon; no replacement donors or filled targets are introduced.
    """
    from scipy import sparse
    n,trees=fit_leaves.shape
    values,offsets,inverse_counts,codes,width=env['leaf_codes'](fit_leaves)
    membership=sparse.csr_matrix((inverse_counts[codes.ravel()],codes.ravel(),
                                 np.arange(0,(n+1)*trees,trees,dtype=np.int64)),shape=(n,width)).T
    means=[]
    for first in range(0,len(query_leaves),env['BATCH']):
        batch=query_leaves[first:first+env['BATCH']]
        rows,cols=[],[]
        for tree,(v,offset) in enumerate(zip(values,offsets)):
            at=np.searchsorted(v,batch[:,tree]);valid=at<len(v)
            valid[valid]&=v[at[valid]]==batch[valid,tree]
            rows.append(np.flatnonzero(valid));cols.append(at[valid]+offset)
        rows,cols=np.concatenate(rows),np.concatenate(cols)
        query=sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(batch),width))
        for score in (query@membership).toarray():
            donors=np.flatnonzero(score>0)
            if not len(donors):
                means.append(observed_donor_mean(np.full(len(fit_targets),1./len(fit_targets)),fit_targets))
                continue
            donors=donors[np.lexsort((donors,-score[donors]))[:env['K']]]
            means.append(observed_donor_mean(score[donors]/score[donors].sum(),fit_targets[donors]))
    return np.asarray(means)


def fit_probtraj(env,train,query,y,out,smoke):
    reports=[];fit_leaves=[];query_leaves=[]
    for b,start in enumerate((0,24,48)):
        X,Xq=design(train,start),design(query,start)
        target=y[:,start:start+24].astype(np.float32)
        began=time.perf_counter()
        model=train_model(env,X,target,2 if smoke else env['ROUNDS'],weights=np.isfinite(target).astype(np.float32))
        model.save_model(out/f'block{b}.ubj')
        fit_leaves.append(model.predict(xgb.DMatrix(X,nthread=env['THREADS']),pred_leaf=True).astype(np.int32))
        fit_seconds=time.perf_counter()-began
        began=time.perf_counter()
        query_leaves.append(model.predict(xgb.DMatrix(Xq,nthread=env['THREADS']),pred_leaf=True).astype(np.int32))
        reports.append(dict(block=b,fit_seconds=fit_seconds,inference_seconds=time.perf_counter()-began,
                            observed_training_targets=int(np.isfinite(target).sum())))
    leaves=np.hstack(fit_leaves)
    np.savez_compressed(out/'donors.npz',leaves=leaves,targets=y,origins=train['origins'].asi8)
    began=time.perf_counter()
    predictions=np.clip(trajectory_means(env,leaves,y,np.hstack(query_leaves)),1.,3000.)
    reports[-1]['inference_seconds']+=time.perf_counter()-began
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite donor predictions')
    return predictions,reports,dict(donors=len(y),maximum_selected_donors=env['K'],donor_policy='Original top-K selection; normalize observed donor mass per horizon; fail if selected donors have no observed target.')


def fit_agnostic(experiment,env,train,query,y,names,out,smoke):
    rounds=3 if smoke else env['ROUNDS']
    reports=[]
    if experiment.endswith('_event'):
        keep=np.flatnonzero(np.isfinite(y).all(axis=1))
        if len(keep)<2:
            raise ValueError('Event curve model requires at least two complete training trajectories after upstream ffill')
        X=env['design_all'](train,names)[keep]
        Xq=env['design_all'](query,names)
        began=time.perf_counter()
        curves=np.stack(env['Parallel'](n_jobs=env['THREADS'])(env['delayed'](env['fit_curve'])(y[i]) for i in keep))
        mean,scale=curves.mean(0),np.maximum(curves.std(0),1.)
        coded=np.asarray((curves-mean)/scale,dtype=np.float32)
        model=xgb.XGBRegressor(**{**env['PARAMS'],'n_estimators':rounds,'base_score':float(coded.mean())})
        model.fit(X,coded,verbose=False)
        fit_seconds=time.perf_counter()-began
        model.get_booster().save_model(out/'event.ubj')
        np.savez_compressed(out/'event_transform.npz',mean=mean,scale=scale,curves=curves,origins=train['origins'][keep].asi8)
        began=time.perf_counter()
        raw=model.get_booster().predict(xgb.DMatrix(Xq,nthread=env['THREADS']))
        predictions=np.clip(env['event_curve'](raw*scale+mean),1.,3000.)
        reports.append(dict(model='event',fit_seconds=fit_seconds,inference_seconds=time.perf_counter()-began,
                            training_rows=len(keep),excluded_incomplete_rows=len(y)-len(keep),observed_training_targets=len(keep)*72))
    else:
        predictions=np.empty((len(query['origins']),72),np.float64)
        for b,start in enumerate((0,24,48)):
            target=y[:,start:start+24]
            keep=np.flatnonzero(np.isfinite(target).all(axis=1))
            if len(keep)<5:
                raise ValueError('Context model requires at least five complete block windows after upstream ffill')
            origins=train['origins'][keep].values
            if any(not len(fit) for fit,held in env['purged_folds'](origins)):
                raise ValueError('Context model has an empty purged classifier training fold')
            X,Xq=design(train,start)[keep],design(query,start)
            block_y=target[keep]
            began=time.perf_counter()
            q=env['event_probabilities'](X,block_y,origins,rounds)
            if not np.isfinite(q).all():
                raise ValueError('Nonfinite out-of-fold event probabilities')
            model=xgb.XGBRegressor(**{**env['PARAMS'],'n_estimators':rounds,'objective':env['context_loss'](q),
                                      'base_score':float(np.asarray(block_y,dtype=np.float32).mean())})
            model.fit(X,np.asarray(block_y,dtype=np.float32),verbose=False)
            fit_seconds=time.perf_counter()-began
            model.get_booster().save_model(out/f'block{b}.ubj')
            np.savez_compressed(out/f'block{b}_context.npz',probabilities=q,origins=train['origins'][keep].asi8)
            began=time.perf_counter()
            predictions[:,start:start+24]=model.get_booster().predict(xgb.DMatrix(Xq,nthread=env['THREADS']))
            reports.append(dict(block=b,fit_seconds=fit_seconds,inference_seconds=time.perf_counter()-began,
                                training_rows=len(keep),excluded_incomplete_rows=len(y)-len(keep),observed_training_targets=len(keep)*24))
        predictions=np.clip(predictions,1.,3000.)
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite agnostic forecasts')
    return predictions,reports,dict(training_policy='Original complete-window objectives, curves and purged folds on upstream training-y ffill; residual incomplete windows excluded.',rounds=rounds)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=SUPPORTED,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--period',choices=list(CUTOFFS))
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--merge',action='store_true')
    args=parser.parse_args()
    if args.threads<1:
        parser.error('threads must be positive')
    if args.merge:
        print(json.dumps(dict(prediction_rows=merge(args.out))));return
    for period in [args.period] if args.period else (['F1'] if args.smoke else list(CUTOFFS)):
        report=run_period(args.experiment,args.data,args.out,period,args.threads,args.smoke)
        print(json.dumps({k:report[k] for k in ['experiment','period','train_origins','query_origins','end_to_end_seconds']}),flush=True)
    if not args.period and not args.smoke:
        print(json.dumps(dict(prediction_rows=merge(args.out))))


if __name__ == "__main__":
    import sys
    if "--experiment" not in sys.argv:
        sys.argv.extend(["--experiment", 'e2000fs_budget_primaldual'])
    main()
