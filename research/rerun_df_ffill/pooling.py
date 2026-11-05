"""Training-only calibration; original eval-stream adaptation removed.

This is an explicit verification-policy repair, not exact evaluation-stream
reproduction or a structural SOTA ablation. Component models are newly trained on
the prepared feature/training-y ffill data. The latest prior training quarter is
predicted by models fitted before it with a 144-hour purge; calibration is frozen
before official query inference. No old prediction assets or query labels are read.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
import numpy as np
import pandas as pd
import xgboost as xgb


H = 72

CUTOFFS = {'F1': '2025-12-29', 'F2': '2026-03-29', 'F3': '2026-06-29'}

QUERY_COUNTS = {'F1': 2089, 'F2': 2113, 'F3': 2113}

BASE_NAMES = {'constant': ['v_lag0', 'v_lag1', 'v_lag3', 'v_lag6', 'v_lag12', 'v_lag24', 'v_lag48', 'v_lag72', 'v_lag120', 'v_m6', 'v_sd6', 'v_m24', 'v_sd24', 'v_m72', 'v_sd72', 'ace_speed_kms_last', 'ace_speed_kms_m6', 'ace_speed_kms_m24', 'ace_density_cm3_last', 'ace_density_cm3_m6', 'ace_density_cm3_m24', 'ace_temperature_k_last', 'ace_temperature_k_m6', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6', 'ace_bt_nt_m24', 'ace_bz_gsm_nt_last', 'ace_bz_gsm_nt_m6', 'ace_bz_gsm_nt_m24', 'ace_by_gsm_nt_last', 'ace_by_gsm_nt_m6', 'ace_by_gsm_nt_m24', 'ace_epam_e_channel1_last', 'ace_epam_e_channel1_m6', 'ace_epam_e_channel1_m24', 'ace_epam_e_channel2_last', 'ace_epam_e_channel2_m6', 'ace_epam_e_channel2_m24', 'ace_epam_p_channel1_last', 'ace_epam_p_channel1_m6', 'ace_epam_p_channel1_m24', 'ace_epam_p_channel3_last', 'ace_epam_p_channel3_m6', 'ace_epam_p_channel3_m24', 'ace_epam_p_channel5_last', 'ace_epam_p_channel5_m6', 'ace_epam_p_channel5_m24', 'ace_sis_gt10_flux_last', 'ace_sis_gt10_flux_m6', 'ace_sis_gt10_flux_m24', 'speed_d24', 'epam_p1_d6', 'ch_ch_t035_area', 'ch_ch_t035_max_area', 'ch_ch_t035_top1_area', 'ch_ch_t035_top2_area', 'ch_ch_t035_top3_area', 'ch_ch_t035_meridian15_area', 'ch_ch_t035_meridian15_equatorial_area', 'ch_ch_t035_meridian30_area', 'ch_ch_t035_meridian30_equatorial_area', 'ch_ch_t035_equatorial_area', 'ch_ch_t045_area', 'ch_ch_t045_max_area', 'ch_ch_t045_top1_area', 'ch_ch_t045_top2_area', 'ch_ch_t045_top3_area', 'ch_ch_t045_meridian15_area', 'ch_ch_t045_meridian15_equatorial_area', 'ch_ch_t045_meridian30_area', 'ch_ch_t045_meridian30_equatorial_area', 'ch_ch_t045_equatorial_area', 'ch_ch_t055_area', 'ch_ch_t055_max_area', 'ch_ch_t055_top1_area', 'ch_ch_t055_top2_area', 'ch_ch_t055_top3_area', 'ch_ch_t055_meridian15_area', 'ch_ch_t055_meridian15_equatorial_area', 'ch_ch_t055_meridian30_area', 'ch_ch_t055_meridian30_equatorial_area', 'ch_ch_t055_equatorial_area', 'img_age_h', 'img_stale', 'enlil_next_arr_h', 'enlil_since_arr_h', 'enlil_n_pending', 'enlil_kp', 'enlil_glancing'], 'varying': ['rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed', 'strip_v350_dark', 'strip_v350_cover', 'strip_v450_logmed', 'strip_v450_dark', 'strip_v450_cover', 'strip_v550_logmed', 'strip_v550_dark', 'strip_v550_cover', 'strip_v650_logmed', 'strip_v650_dark', 'strip_v650_cover', 'strip_v750_logmed', 'strip_v750_dark', 'strip_v750_cover', 'enlil_dt_target']}

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

def spectral_objective(target):
    observed = np.isfinite(target)
    raw = np.where(target >= 550.0, 4.0, 1.0) * observed
    weights = raw / (raw.sum(0) / observed.sum(0))

    def objective(prediction, matrix):
        prediction = np.asarray(prediction).reshape(target.shape)
        error = np.where(observed, prediction - np.nan_to_num(target), 0.0)
        z = np.fft.fft(error, axis=1, norm='ortho')
        rho = 1 / np.sqrt(1 + abs(z) ** 2 / 50.0 ** 2)
        gradient = (error * weights + 0.1 * np.fft.ifft(z * rho, axis=1, norm='ortho').real) * observed
        return (gradient.astype(np.float32), ((weights + 0.1) * observed).astype(np.float32))
    return objective

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

FEATURES = ['v_lag0', 'rec_mean', 'enlil_kp', 'ch_ch_t035_meridian30_equatorial_area', 'ace_bt_nt_last', 'ace_speed_kms_last', 'ch_ch_t035_equatorial_area', 'ch_ch_t035_meridian15_equatorial_area', 'ace_bt_nt_m6', 'rec26', 'strip_v450_logmed', 'strip_v350_logmed', 'ch_ch_t055_equatorial_area', 'rec27', 'ace_density_cm3_last', 'ch_ch_t055_meridian30_equatorial_area', 'ace_by_gsm_nt_m24', 'ace_density_cm3_m6', 'ace_temperature_k_last', 'enlil_n_pending', 'ace_temperature_k_m24', 'strip_v750_logmed', 'ch_ch_t055_meridian15_area', 'ch_ch_t035_top2_area', 'strip_v550_dark', 'strip_v650_dark', 'ace_sis_gt10_flux_last', 'ace_epam_p_channel3_last', 'img_age_h', 'rec28']

PARAMS = {'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 20, 'reg_lambda': 5, 'objective': 'reg:squarederror', 'tree_method': 'hist', 'device': 'cpu', 'multi_strategy': 'multi_output_tree', 'eval_metric': 'rmse', 'seed': 42, 'verbosity': 0}

def objective_for(y, kind="point4", row_weights=None):
    observed = np.isfinite(y)
    safe = np.where(observed, y, 0.)
    weights = np.where(safe >= 550., 4., 1.)
    if row_weights is None:
        weights /= np.sum(weights * observed, axis=0) / observed.sum(axis=0)
    else:
        weights *= np.asarray(row_weights)[:, None]

    def objective(pred, matrix):
        if kind == "tweedie":
            mu = np.clip(pred.astype(np.float64), 1., 3000.)
            scale = 500. ** 1.5
            grad = scale * (mu ** -.5 - safe * mu ** -1.5)
            hess = np.maximum(scale * (-.5 * mu ** -1.5 + 1.5 * safe * mu ** -2.5), 1e-6 * scale)
        else:
            grad, hess = (pred - safe) * weights, weights
        return np.where(observed, grad, 0.).astype(np.float32), np.where(observed, hess, 0.).astype(np.float32)
    return objective

def fit_head(x, y, rounds, threads, out, name, kind="point4", depth=3, base=None, row_weights=None):
    observed = np.isfinite(y)
    if not observed.any(axis=0).all():
        raise ValueError("Each output needs observed training targets")
    matrix = xgb.DMatrix(x, label=np.where(observed, y, 0.).astype(np.float32), nthread=threads)
    params = dict(PARAMS, nthread=threads, max_depth=depth, base_score=float(np.nanmean(y)) if base is None else base,
                  disable_default_eval_metric=True)
    def metric(pred, matrix):
        return "observed_mse", float(((pred-y)[observed] ** 2).mean())
    history = {}
    model = xgb.train(params, matrix, num_boost_round=rounds, obj=objective_for(y, kind, row_weights),
                      evals=[(matrix, "train")], custom_metric=metric, evals_result=history, verbose_eval=False)
    model.save_model(out / f"{name}.ubj")
    (out / f"{name}_loss.json").write_text(json.dumps(history, indent=2)+"\n")
    return model

def fit_recipe(experiment, x, xq, y, origins, block, rounds, threads, out, prefix):
    keep = np.isfinite(y).any(axis=1)
    x, y, origins = x[keep], y[keep], origins[keep]
    timing = 0.
    def predict(model, values):
        nonlocal timing
        start = time.perf_counter()
        result = model.inplace_predict(values)
        timing += time.perf_counter() - start
        return result
    def head(label, target=y, values=x, **kwargs):
        return fit_head(values, target, rounds, threads, out, prefix+"_"+label, **kwargs)
    def mixture():
        level = predict(head("level"), xq)
        indicator = np.where(np.isfinite(y), (y >= 550.).astype(float), np.nan)
        probability = np.clip(predict(head("probability", indicator), xq), 0., 1.)
        event = (y >= 550.).any(axis=1)
        if not event.any() or not np.isfinite(y[event]).any(axis=0).all():
            raise ValueError("Mixture event subset lacks observed support")
        center = np.nanmean(y[event], axis=0)
        gap = predict(head("gap", y[event]-center, x[event], base=0.), xq)+center
        (out / f"{prefix}_event_center.json").write_text(json.dumps(center.tolist())+"\n")
        return np.clip(level+probability*(gap-level), 1., 3000.)
    if experiment.endswith("capacity_depth5"):
        pred = predict(head("capacity", depth=5), xq)
    elif experiment.endswith("tweedie15"):
        pred = predict(head("tweedie", kind="tweedie"), xq)
    elif experiment.endswith("mixture2comp"):
        pred = mixture()
    elif experiment.endswith("detroute"):
        body = np.clip(predict(head("tweedie", kind="tweedie"), xq), 1., 3000.)
        tail = mixture()
        width = y.shape[1]
        def detector_design(values):
            return np.column_stack([np.repeat(values, width, axis=0), np.tile(np.arange(block*BLOCK+1, block*BLOCK+width+1), len(values))]).astype(np.float32)
        observed = np.isfinite(y).ravel()
        dx = detector_design(x)[observed]
        matrix = xgb.DMatrix(dx, label=(y.ravel()[observed] >= 550.).astype(np.float32), nthread=threads)
        params = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": "cpu",
                  "max_depth": 4, "eta": .05, "subsample": .8, "colsample_bytree": .7, "min_child_weight": 20,
                  "reg_lambda": 5., "seed": 42, "nthread": threads}
        history = {}
        model = xgb.train(params, matrix, num_boost_round=2 if rounds == 2 else 400,
                          evals=[(matrix,"train")], evals_result=history, verbose_eval=False)
        model.save_model(out / f"{prefix}_detector.ubj")
        (out / f"{prefix}_detector_loss.json").write_text(json.dumps(history)+"\n")
        probability = predict(model, detector_design(xq)).reshape(len(xq), width)
        pred = np.where(probability >= .234, tail, body)
    elif experiment.endswith("era"):
        months = origins.month.to_numpy()
        groups = np.unique(months)
        weights = np.ones(len(groups))
        for iteration in range(3):
            row_weights = weights[np.searchsorted(groups, months)]
            row_weights /= row_weights.mean()
            model = head(f"era{iteration}", row_weights=row_weights)
            inpred = model.inplace_predict(x)
            loss = np.array([np.nanmean((inpred[months == g]-y[months == g])**2) for g in groups])
            if loss.mean() > 0:
                weights *= np.exp(.5*(loss/loss.mean()-1.))
                weights /= weights.mean()
        pred = predict(model, xq)
        (out / f"{prefix}_era_weights.json").write_text(json.dumps(dict(zip(map(str,groups), weights.tolist())))+"\n")
    else:
        raise ValueError("Unsupported calibration recipe")
    return np.clip(pred, 1., 3000.), {}, {"predict_seconds": timing, "train_rows": len(y),
        "dropped_no_target": int((~keep).sum()), "observed_per_lead": np.isfinite(y).sum(axis=0).tolist()}

BLOCK = 24

WINDOW=2160
GRID=np.round(np.arange(0.,.75+1e-9,.025),4)

def choose_weights(se_base: np.ndarray, se_arm: np.ndarray, cross: np.ndarray,
                   high: np.ndarray, mature: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Frozen rule, one weight per origin, matured residuals only."""
    n = len(mature)
    c_b = np.vstack([np.zeros((1, H)), np.cumsum(se_base, axis=0)])
    c_a = np.vstack([np.zeros((1, H)), np.cumsum(se_arm, axis=0)])
    c_x = np.vstack([np.zeros((1, H)), np.cumsum(cross, axis=0)])
    c_bh = np.vstack([np.zeros((1, H)), np.cumsum(se_base * high, axis=0)])
    c_ah = np.vstack([np.zeros((1, H)), np.cumsum(se_arm * high, axis=0)])
    c_xh = np.vstack([np.zeros((1, H)), np.cumsum(cross * high, axis=0)])
    c_nh = np.concatenate([[0], np.cumsum(high.sum(axis=1))])
    weights = np.zeros(n)
    g = GRID[:, None, None]
    for i in range(n):
        k = int(mature[i])
        j = max(0, k - window)
        rows = k - j
        if rows < 48 or c_nh[k] - c_nh[j] < 50:      # not enough matured history or too few tail cells
            continue
        a0 = (c_b[k] - c_b[j]).sum() / (rows * H)
        b0 = (c_a[k] - c_a[j]).sum() / (rows * H)
        x0 = (c_x[k] - c_x[j]).sum() / (rows * H)
        nh = c_nh[k] - c_nh[j]
        a1 = (c_bh[k] - c_bh[j]).sum() / nh
        b1 = (c_ah[k] - c_ah[j]).sum() / nh
        x1 = (c_xh[k] - c_xh[j]).sum() / nh
        tot = (1 - g[:, 0, 0]) ** 2 * a0 + g[:, 0, 0] ** 2 * b0 + 2 * g[:, 0, 0] * (1 - g[:, 0, 0]) * x0
        tail = (1 - g[:, 0, 0]) ** 2 * a1 + g[:, 0, 0] ** 2 * b1 + 2 * g[:, 0, 0] * (1 - g[:, 0, 0]) * x1
        feasible = tail <= a1
        if feasible.any():
            weights[i] = GRID[int(np.argmin(np.where(feasible, tot, np.inf)))]
    return weights

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

from scipy.optimize import nnls
RIDGE=1e-3
SHRINK=(1.,.75,.5,.25)

def fit_block(delta: np.ndarray, resid: np.ndarray, high: np.ndarray) -> np.ndarray:
    """NNLS on standardised base-relative increments, then the simplex cap and the tail guard."""
    rows, p = delta.shape[1], delta.shape[0]
    rms = np.sqrt((delta ** 2).mean(axis=1))
    alive = rms > 1e-6
    if not alive.any():
        return np.zeros(p)
    X = (delta[alive] / rms[alive, None]).T
    w = nnls(np.vstack([X, np.sqrt(RIDGE * rows) * np.eye(int(alive.sum()))]),
             np.concatenate([resid, np.zeros(int(alive.sum()))]), maxiter=3 * int(alive.sum()))[0]
    beta = np.zeros(p)
    beta[alive] = w / rms[alive]
    base_tail = (resid[high] ** 2).mean()
    for s in SHRINK:
        cand = beta * s
        if cand.sum() > 1.0:
            cand = cand / cand.sum()
        comb = (delta * cand[:, None]).sum(axis=0)
        if ((resid - comb)[high] ** 2).mean() <= base_tail:
            return cand
    return np.zeros(p)

SUPPORTED=(
    'e2000fs_calib_offset_roll','e2000fs_calib_pool_bodytail',
    'e2000fs_calib4_detpool_era','e2000fs_calib4_detpool_gpd',
    'e2000fs_calib4_detpool_mixture','e2000fs_calib6_detpool_mix20','e2000fs_calib5_detpool_all',
)
POOLALL_ACTUAL_ARMS = ('pointweight4', 'point4_144h', 'tweedie15', 'mixture', 'era', 'gpd', 'spectral', 'dirmo24', 'detroute', 'offset_roll', 'pool_bodytail', 'detpool_era', 'detpool_mixture', 'lgbm_extreme4', 'lgbm_point4', 'lgbm_recent')
POOLALL_IDENTITY = {'submitted_utc': '20261031T172348000Z', 'team_member': 'seongeun', 'experiment': 'e2000fs_calib5_detpool_all', 'legacy_config_arms': 81, 'actual_published_arms': 16, 'receipt_commit': 'c482c10e2720e03157a512d347c559f9e319e82c', 'component_source_commit': '6b85acc0b00d731bf31b6885e0b847c18ee395d0', 'receipt_source': 'docs/visualizations/e2000fs_calib5_detpool_all/weights_receipt.json', 'submitted_record_source': 'docs/visualizations/e2000fs_calib5_detpool_all/leaderboard-result.json', 'recomputed_stack_equals_submission': True, 'max_abs_difference_float32': 0.0, 'matching_mse': {'mse': 4154.1526325545565, 'mse_f1': 6007.057035049542, 'mse_f2': 3600.168705922258, 'mse_f3': 2876.277923686607}}
POOLALL_ARMS = ('e2000fs_budget_complexity/results/forecast.npz', 'e2000fs_budget_covshift/results/forecast.npz', 'e2000fs_budget_primaldual/results/forecast.npz', 'e2000fs_budget_spectral/results/forecast.npz', 'e2000fs_calib-20261030/runs/control_144h_v1/forecast.npz', 'e2000fs_calib-20261030/runs/offset_v1/forecast.npz', 'e2000fs_calib-20261030/runs/pool_v1/forecast.npz', 'e2000fs_calib-20261030/runs/tweedie15_v1/forecast.npz', 'e2000fs_calib2-20261031/runs/capacity_depth5_v1/forecast.npz', 'e2000fs_calib2-20261031/runs/detroute_v1/forecast.npz', 'e2000fs_calib2-20261031/runs/mixture_v1/forecast.npz', 'e2000fs_calib3-20261031/runs/affine_v1/forecast.npz', 'e2000fs_calib3-20261031/runs/bag3_v1/forecast.npz', 'e2000fs_calib3-20261031/runs/era_v1/forecast.npz', 'e2000fs_calib3-20261031/runs/extrap_v1/forecast.npz', 'e2000fs_calib3-20261031/runs/gpd_v1/forecast.npz', 'e2000fs_calib3-20261031/runs/monotone_v1/forecast.npz', 'e2000fs_calib3-20261031/runs/restack_v1/forecast.npz', 'e2000fs_calib4-20261031/runs/era_v1/forecast.npz', 'e2000fs_calib4-20261031/runs/mixture_v1/forecast.npz', 'e2000fs_dirmo_12/results/forecast.npz', 'e2000fs_dirmo_24/results/forecast.npz', 'e2000fs_dirmo_24_feat30/results/forecast.npz', 'e2000fs_dirmo_24_feat30_lgbm/results/forecast.npz', 'e2000fs_dirmo_24_feat30_lgbm_HPO/results/forecast.npz', 'e2000fs_dirmo_24_feat30_pointweight4/results/forecast.npz', 'e2000fs_dirmo_24_feat30_pointweight4_last24/results/forecast.npz', 'e2000fs_dirmo_24_feat30_pointweight4_last48/results/forecast.npz', 'e2000fs_dirmo_6/results/forecast.npz', 'e2000fs_dirrec/results/forecast.npz', 'e2000fs_dirrecmo_12/results/forecast.npz', 'e2000fs_dirrecmo_24/results/forecast.npz', 'e2000fs_dirrecmo_6/results/forecast.npz', 'e2000fs_feat30_agnostic_v2_context/results/forecast.npz', 'e2000fs_feat30_agnostic_v2_event/results/forecast.npz', 'e2000fs_feat30_extreme/results/forecast.npz', 'e2000fs_feat30_extreme_histprior/results/forecast.npz', 'e2000fs_feat30_gaussian_pilot/results/forecast.npz', 'e2000fs_feat30_hyper_gain/results/forecast.npz', 'e2000fs_feat30_lgbm_extreme4/results/forecast.npz', 'e2000fs_feat30_lognormal_crps/results/forecast.npz', 'e2000fs_feat30_lognormal_nll/results/forecast.npz', 'e2000fs_feat30_mc_calibration/results/forecast.npz', 'e2000fs_feat30_pointweight4/results/forecast.npz', 'e2000fs_feat30_pointweight4_tcn_lb120/results/forecast.npz', 'e2000fs_feat30_pointweight4_tcn_lb120_regskip/results/forecast.npz', 'e2000fs_feat30_probtraj/results/forecast.npz', 'e2000fs_feat30_shape/results/forecast.npz', 'e2000fs_feat30_smooth_error_tail/results/forecast.npz', 'e2000fs_feat30_smooth_wcrps/results/forecast.npz', 'e2000fs_feat30_tailcal_crps/results/forecast.npz', 'e2000fs_feat30_transfer_center/results/forecast.npz', 'e2000fs_feat30_transfer_recency/results/forecast.npz', 'e2000fs_feat30_transfer_recent_window/results/forecast.npz', 'e2000fs_feat30_wavelet/results/forecast.npz', 'e2000fs_kumo_direct/results/forecast.npz', 'e2000fs_kumo_dirrec/results/forecast.npz', 'e2000fs_kumo_pooled/results/forecast.npz', 'e2000fs_kumo_recursive/results/forecast.npz', 'e2000fs_mimo/results/forecast.npz', 'e2000fs_point4_high_anchor/results/forecast.npz', 'e2000fs_point4_high_regret_guard/results/forecast.npz', 'e2000fs_point4_leaf_refit/results/forecast.npz', 'e2000fs_point4_linear_leaf/results/forecast.npz', 'e2000fs_point4_stable_retrain/results/forecast.npz', 'e2000fs_point4_teacher_trust/results/forecast.npz', 'e2000fs_realmlp_direct/results/forecast.npz', 'e2000fs_realmlp_dirrec/results/forecast.npz', 'e2000fs_realmlp_pooled/results/forecast.npz', 'e2000fs_realmlp_recursive/results/forecast.npz', 'e2000fs_recmo_12/results/forecast.npz', 'e2000fs_recmo_24/results/forecast.npz', 'e2000fs_recmo_6/results/forecast.npz', 'e2000fs_rectify/results/forecast.npz', 'e2000fs_rectifymo_24/results/forecast.npz', 'e2000fs_recursive/results/forecast.npz', 'e2000fs_stratify_dirrecmo_dirrecmo_24/results/forecast.npz', 'e2000fs_tabiclv2_direct/results/forecast.npz', 'e2000fs_tabiclv2_dirrec/results/forecast.npz', 'e2000fs_tabiclv2_pooled/results/forecast.npz', 'e2000fs_tabiclv2_recursive/results/forecast.npz')
POOLALL_BLOCKER = {'experiment': 'e2000fs_calib5_detpool_all', 'required_arms': 81, 'inventory_recovered': True, 'scope': 'obsolete_legacy81_recipe_only', 'inventory_source': 'docs/explainers/data/calib5.json:arms.names', 'reason': 'Exact 81-arm names recovered. Execution still requires a verified training recipe for calib3 monotone_v1 and fresh prior/full prediction artifacts for all 81 arms; no old forecasts or smaller substitute pool are accepted.', 'missing_verified_recipe': ['e2000fs_calib3-20261031/runs/monotone_v1/forecast.npz'], 'submission_variant_evidence': 'docs/visualizations/e2000fs_calib5_detpool_all/weights_receipt.json documents that the actual F1/F2/F3 submission used a distinct 16-arm refit stack; do not silently replace the requested legacy 81-arm recipe.'}
POLICY='training-only calibration; original eval-stream adaptation removed'


def prior_split(origins):
    origins=pd.DatetimeIndex(origins)
    if not origins.is_monotonic_increasing or origins.has_duplicates:
        raise ValueError('Training origins must be unique and sorted')
    boundary=origins.max()-pd.DateOffset(months=3)
    calibration=np.flatnonzero(origins>=boundary)
    reference=np.flatnonzero(origins+pd.Timedelta(hours=144)<=boundary)
    if len(reference)<2 or len(calibration)<48:
        raise ValueError('Insufficient strictly prior reference/calibration support')
    return reference,calibration,boundary


def required_components(experiment):
    if experiment=='e2000fs_calib5_detpool_all':return POOLALL_ACTUAL_ARMS
    if experiment=='e2000fs_calib_offset_roll':return ('spectral',)
    if experiment=='e2000fs_calib_pool_bodytail':return ('body','spectral')
    if experiment=='e2000fs_calib6_detpool_mix20':return ('detroute','gpd','mixture')
    if experiment.startswith('e2000fs_calib4_detpool_'):return ('detroute',experiment.rsplit('_',1)[1])
    raise ValueError(json.dumps(POOLALL_BLOCKER) if experiment==POOLALL_BLOCKER['experiment'] else 'Unknown pooling experiment')


def native_head(x,y,rounds,threads,out,label,objective=None):
    if not np.isfinite(y).all():
        raise ValueError('Component labels must be finite after upstream training-y ffill')
    matrix=xgb.DMatrix(x,label=y.astype(np.float32),nthread=threads)
    model=xgb.train(dict(PARAMS,nthread=threads,num_target=y.shape[1],base_score=float(y.mean())),matrix,
                    num_boost_round=rounds,obj=objective)
    model.save_model(out/f'{label}.ubj')
    return model


PROGRESS={'unit':0,'units':0}
COMPONENT_BLOCKS=3


def _log_block(out,component,block,began):
    """One train.log JSON line per fitted 24 h block so the sunrun TUI sees progress (period from <F*>_models path)."""
    out=Path(out)
    period=next((part[:-len('_models')] for part in reversed(out.parts) if part.endswith('_models')),None)
    stage=out.parent.name if period and out.parent.name!=f'{period}_models' else None
    record=dict(period=period,stage=stage,component=component,block=block,fit_seconds=round(time.perf_counter()-began,1))
    if PROGRESS['units']:
        PROGRESS['unit']+=1;record.update(unit=PROGRESS['unit'],units=PROGRESS['units'])
    print(json.dumps(record),flush=True)


def component_predictions(component,train,apply,y,threads,out,smoke):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if component.startswith('lgbm_'):
        return lgbm_component(component,train,apply,y,threads,out,smoke)
    if component=='dirmo24':component='body'
    pred=np.empty((len(apply['origins']),72),np.float64)
    seconds=0.;rounds=2 if smoke else 900
    for b,start in enumerate((0,24,48)):
        block_began=time.perf_counter()
        x,xq=design(train,start),design(apply,start)
        target=y[:,start:start+24]
        if component in ('pointweight4','point4_144h'):
            model=fit_head(x,target,rounds,threads,out,f'b{b}_point4')
            began=time.perf_counter();pred[:,start:start+24]=model.inplace_predict(xq);seconds+=time.perf_counter()-began
        elif component in ('detroute','mixture','era','tweedie15'):
            name={'detroute':'e2000fs_calib2_detroute','mixture':'e2000fs_calib2_mixture2comp','era':'e2000fs_calib3_era','tweedie15':'e2000fs_calib_tweedie15'}[component]
            pred[:,start:start+24],_,report=fit_recipe(name,x,xq,target,train['origins'],b,rounds,threads,out,f'b{b}')
            seconds+=report['predict_seconds']
        elif component in ('body','spectral'):
            obj=spectral_objective(target) if component=='spectral' else None
            model=native_head(x,target,rounds,threads,out,f'b{b}',obj)
            began=time.perf_counter();value=model.inplace_predict(xq);seconds+=time.perf_counter()-began
            pred[:,start:start+24]=value
        elif component=='gpd':
            level=fit_head(x,target,rounds,threads,out,f'b{b}_level')
            indicator=(target>=550.).astype(np.float32)
            probability=native_head(x,indicator,2 if smoke else 400,threads,out,f'b{b}_probability')
            event=(target>=550.).any(1)
            if not event.any():raise ValueError('GPD component has no training event rows')
            excess=np.log(np.maximum(target[event]-550.,1.))
            tail=native_head(x[event],excess,2 if smoke else 400,threads,out,f'b{b}_excess')
            began=time.perf_counter()
            L=level.inplace_predict(xq).astype(np.float64)
            P=np.clip(probability.inplace_predict(xq),0.,1.)
            Z=np.clip(tail.inplace_predict(xq),0.,np.log(2450.))
            pred[:,start:start+24]=np.where(P>=.5,550.+np.exp(Z),L)
            seconds+=time.perf_counter()-began
        else:raise ValueError('Unknown component '+component)
        _log_block(out,component,b,block_began)
    if not np.isfinite(pred).all():raise ValueError('Nonfinite component prediction')
    return np.clip(pred,1.,3000.),seconds


def detpool_weight(base,arm,y):
    # Original grid/tail guard at one final refit on the fully matured prior set.
    return float(choose_weights((base-y)**2,(arm-y)**2,(base-y)*(arm-y),y>=550.,np.array([len(y)]))[0])


def estimate_state(experiment,predictions,y):
    if not np.isfinite(y).all():raise ValueError('Calibration truth must contain finite training labels only')
    if any(p.shape!=y.shape or not np.isfinite(p).all() for p in predictions.values()):
        raise ValueError('Calibration component grids disagree')
    state=dict(policy=POLICY,components=list(required_components(experiment)),calibration_rows=len(y))
    if experiment=='e2000fs_calib_offset_roll':
        # Frozen original submitted settings: alpha=.25, window=336. Every
        # residual here is matured before the period cutoff; state never updates.
        state.update(window=336,alpha=.25,settings_source='Original submitted fixed alpha/window; residual state re-estimated on prior training only',offset=(.25*(y[-336:]-predictions['spectral'][-336:]).mean(0)).tolist())
    elif experiment=='e2000fs_calib_pool_bodytail':
        weights=[]
        for start in (0,24,48):
            a=predictions['body'][:,start:start+24];d=predictions['spectral'][:,start:start+24]-a
            denominator=float((d*d).sum())
            weights.append(float(np.clip((d*(y[:,start:start+24]-a)).sum()/denominator,0.,1.)) if denominator>0 else 0.)
        state['band_weights']=weights
    else:
        arm='gpd' if experiment=='e2000fs_calib6_detpool_mix20' else experiment.rsplit('_',1)[1]
        state.update(arm=arm,weight=detpool_weight(predictions['detroute'],predictions[arm],y),window=WINDOW,grid=GRID.tolist())
        if experiment=='e2000fs_calib6_detpool_mix20':state['fixed_mixture_weight']=.20
    return state


def apply_state(experiment,predictions,state):
    # This interface has no truth argument: only frozen state and new forecasts.
    if state['policy']!=POLICY:raise ValueError('Unexpected calibration policy')
    if experiment=='e2000fs_calib_offset_roll':
        result=predictions['spectral']+np.asarray(state['offset'])[None,:]
    elif experiment=='e2000fs_calib_pool_bodytail':
        result=predictions['body'].copy()
        for start,w in zip((0,24,48),state['band_weights']):
            result[:,start:start+24]=(1-w)*predictions['body'][:,start:start+24]+w*predictions['spectral'][:,start:start+24]
    else:
        w=state['weight'];result=(1-w)*predictions['detroute']+w*predictions[state['arm']]
        if experiment=='e2000fs_calib6_detpool_mix20':result=.80*result+.20*predictions['mixture']
    if not np.isfinite(result).all():raise ValueError('Nonfinite frozen pool output')
    return np.clip(result,1.,3000.)


def subset(grid,indices):
    return {k:v[indices] for k,v in grid.items()}


def fit_pool(experiment,train,query,y,threads,out,smoke=False):
    if experiment=='e2000fs_calib5_detpool_all':
        return fit_poolall16(train,query,y,threads,out,smoke)
    components=required_components(experiment)
    PROGRESS.update(unit=0,units=2*len(components)*COMPONENT_BLOCKS)  # prior + full fits, three 24 h blocks each
    reference,calibration,boundary=prior_split(train['origins'])
    if smoke:
        reference=reference[np.linspace(0,len(reference)-1,min(len(reference),128)).astype(int)]
        calibration=calibration[np.linspace(0,len(calibration)-1,min(len(calibration),64)).astype(int)]
        full=np.linspace(0,len(y)-1,min(len(y),192)).astype(int)
    else:full=np.arange(len(y))
    ref_grid,cal_grid=subset(train,reference),subset(train,calibration)
    prior={};future={};query_seconds=0.
    for component in components:
        prior[component],_=component_predictions(component,ref_grid,cal_grid,y[reference],threads,Path(out)/'prior'/component,smoke)
    state=estimate_state(experiment,prior,y[calibration])
    state.update(reference_rows=len(reference),reference_start=str(train['origins'][reference].min()),
                 reference_end=str(train['origins'][reference].max()),calibration_start=str(train['origins'][calibration].min()),
                 calibration_end=str(train['origins'][calibration].max()),split_boundary=str(boundary),purge_hours=144)
    Path(out).mkdir(parents=True,exist_ok=True)
    (Path(out)/'frozen_calibration.json').write_text(json.dumps(state,indent=2)+'\n')
    np.savez_compressed(Path(out)/'prior_predictions.npz',origins=train['origins'][calibration].asi8,**prior)
    for component in components:
        future[component],seconds=component_predictions(component,subset(train,full),query,y[full],threads,Path(out)/'full'/component,smoke)
        query_seconds+=seconds
    began=time.perf_counter();prediction=apply_state(experiment,future,state);query_seconds+=time.perf_counter()-began
    return prediction,state,query_seconds


def run_period(experiment,data,out,period,threads=1,smoke=False):
    required_components(experiment)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);began=time.perf_counter()
    names=selected_names(FEATURES)
    train,y=load_grid(data,period,names,True);query,_=load_grid(data,period,names,False)
    if len(query['origins'])!=QUERY_COUNTS[period]:raise ValueError('Unexpected query grid')
    if not np.isfinite(y).all():raise ValueError('Pooling requires the prepared training-y ffill snapshot')
    if smoke:query=subset(query,np.arange(min(8,len(query['origins']))))
    prediction,state,seconds=fit_pool(experiment,train,query,y,threads,out/f'{period}_models',smoke)
    np.savez_compressed(out/f'{period}_predictions.npz',pred=prediction,origins=query['origins'].asi8)
    revision=subprocess.run(['git','rev-parse','HEAD'],text=True,capture_output=True,check=False)
    report=dict(experiment=experiment,period=period,smoke=smoke,policy=POLICY,
                interpretation='Verification-policy repair, not exact original evaluation-stream reproduction; no structural SOTA ablation claim.',
                data=str(Path(data).resolve()),data_bytes=Path(data).stat().st_size,data_mtime_ns=Path(data).stat().st_mtime_ns,
                git_commit=revision.stdout.strip(),seed=42,features=names,parameters=PARAMS,rounds=2 if smoke else 900,
                component_settings=dict(spectral_coefficient=.10,spectral_scale=50.,gpd_threshold=550.,gpd_probability_gate=.5,
                    gpd_probability_excess_rounds=2 if smoke else 400,detroute_threshold=.234,detector_rounds=2 if smoke else 400,
                    era_iterations=3,era_eta=.5,mixture_event_threshold=550.),
                training_cutoff=CUTOFFS[period],maturity_hours=144,train_origins=len(y),query_origins=len(prediction),
                target_policy='Upstream causal ffill of feature and training-y columns only; no query truth read or evaluation-stream updates.',
                frozen_state=state,inference_seconds_per_fold=seconds/len(prediction),end_to_end_seconds=time.perf_counter()-began,
                versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,xgboost=xgb.__version__))
    (out/f'{period}_metadata.json').write_text(json.dumps(report,indent=2)+'\n');return report


LGB_PARAMS = dict(n_estimators=400, objective="regression", learning_rate=0.011665375590900416, max_depth=6, num_leaves=39,
                  subsample=0.8158360436957313, subsample_freq=1, colsample_bytree=0.796156879318864, min_child_samples=76,
                  min_child_weight=1.7495117760617724, reg_alpha=12.889193670860928, reg_lambda=9.785792859825623,
                  random_state=42, device_type="cpu", verbosity=-1)

def _event_weights(y_all, h, b):
    """feat30_lgbm_extreme4: origins whose 24 h block reaches 550 km/s or ramps 100 km/s in 6 h count four times; mean 1."""
    blk = y_all[:, b * 24:(b + 1) * 24]
    event = (blk.max(axis=1) >= 550) | ((blk[:, 6:] - blk[:, :-6]).max(axis=1) >= 100)
    w = np.where(event, 4.0, 1.0)
    return w / w.mean()

def _point_weights_h(y_all, h, b):
    """feat30_pointweight4 (LightGBM): only this horizon's own target >= 550 gets weight 4; mean 1."""
    w = np.where(y_all[:, h] >= 550.0, 4.0, 1.0)
    return w / w.mean()

def lgbm_component(component,train,apply,y,threads,out,smoke):
    import lightgbm as lgb
    import warnings
    keep=np.arange(len(y))
    if component=='lgbm_recent':
        age=np.asarray((train['origins'].max()-train['origins'])/pd.Timedelta(days=1))
        keep=keep[age<=365]
    pred=np.empty((len(apply['origins']),72),np.float64);seconds=0.
    for b,start in enumerate((0,24,48)):
        block_began=time.perf_counter()
        x,xq=design(train,start)[keep],design(apply,start)
        for h in range(start,start+24):
            weights=_event_weights(y[keep],h,b) if component=='lgbm_extreme4' else (_point_weights_h(y[keep],h,b) if component=='lgbm_point4' else None)
            model=lgb.LGBMRegressor(**dict(LGB_PARAMS,n_jobs=threads,n_estimators=2 if smoke else 400))
            model.fit(x,y[keep,h],sample_weight=weights)
            model.booster_.save_model(out/f'h{h+1:02}.txt')
            began=time.perf_counter()
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='X does not have valid feature names')
                pred[:,h]=np.clip(model.predict(xq),1.,3000.)
            seconds+=time.perf_counter()-began
        _log_block(out,component,b,block_began)
    return pred,seconds


POST_ARMS={
    'offset_roll':'e2000fs_calib_offset_roll',
    'pool_bodytail':'e2000fs_calib_pool_bodytail',
    'detpool_era':'e2000fs_calib4_detpool_era',
    'detpool_mixture':'e2000fs_calib4_detpool_mixture',
    'detpool_gpd':'e2000fs_calib4_detpool_gpd',
}


def post_arm_predictions(primitives,states):
    inputs={**primitives,'body':primitives['dirmo24']}
    result=dict(primitives)
    for arm,experiment in POST_ARMS.items():
        result[arm]=apply_state(experiment,inputs,states[arm])
    return result


def poolall_weights(base,arms,y):
    if tuple(arms)!=POOLALL_ACTUAL_ARMS:
        raise ValueError('Actual submitted pool requires exact documented 16-arm order')
    if not np.isfinite(y).all():raise ValueError('Prior pool truth must be finite')
    begin=max(0,len(y)-WINDOW)
    high=y[begin:]>=550.
    if len(y)-begin<48 or high.sum()<50:return np.zeros(16)
    delta=np.stack([arms[name][begin:]-base[begin:] for name in POOLALL_ACTUAL_ARMS])
    return fit_block(delta.reshape(16,-1),(y[begin:]-base[begin:]).ravel(),high.ravel())


def apply_poolall(base,arms,beta):
    if tuple(arms)!=POOLALL_ACTUAL_ARMS or np.shape(beta)!=(16,):
        raise ValueError('Actual submitted pool requires exact documented 16-arm order')
    delta=np.stack([arms[name]-base for name in POOLALL_ACTUAL_ARMS])
    result=base+np.einsum('p,pnh->nh',beta,delta)
    if not np.isfinite(result).all():raise ValueError('Nonfinite actual-submission pool forecast')
    return np.clip(result,1.,3000.)


def fit_poolall16(train,query,y,threads,out,smoke=False):
    """Honest nested prior predictions for the documented actual 16-arm submission.

    Inner postprocessor states use an earlier quarter, before the outer pooling
    calibration quarter. Thus outer arm predictions never adapt to outer truth.
    The final outer betas and inner states remain frozen for official inference.
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    outer_ref,outer_cal,outer_boundary=prior_split(train['origins'])
    inner_local_ref,inner_local_cal,inner_boundary=prior_split(train['origins'][outer_ref])
    inner_ref,inner_cal=outer_ref[inner_local_ref],outer_ref[inner_local_cal]
    full=np.arange(len(y))
    def sample(indices,n):
        return indices[np.linspace(0,len(indices)-1,min(len(indices),n)).astype(int)]
    if smoke:
        inner_ref=sample(inner_ref,128);inner_cal=sample(inner_cal,64)
        outer_ref=sample(outer_ref,160);outer_cal=sample(outer_cal,64);full=sample(full,192)
    inner_components=('detroute','gpd','mixture','era','spectral','dirmo24')
    primitive_count=sum(name not in POST_ARMS for name in POOLALL_ACTUAL_ARMS)
    PROGRESS.update(unit=0,units=(len(inner_components)+2*primitive_count)*COMPONENT_BLOCKS)  # inner + outer prior + full
    inner_pred={}
    for component in inner_components:
        inner_pred[component],_=component_predictions(component,subset(train,inner_ref),subset(train,inner_cal),y[inner_ref],threads,out/'inner'/component,smoke)
    inner_inputs={**inner_pred,'body':inner_pred['dirmo24']}
    inner_states={arm:estimate_state(exp,inner_inputs,y[inner_cal]) for arm,exp in POST_ARMS.items()}
    primitive_names=tuple(name for name in POOLALL_ACTUAL_ARMS if name not in POST_ARMS)
    outer_pred={}
    for component in primitive_names:
        outer_pred[component],_=component_predictions(component,subset(train,outer_ref),subset(train,outer_cal),y[outer_ref],threads,out/'outer_prior'/component,smoke)
    outer_all=post_arm_predictions(outer_pred,inner_states)
    outer_arms={name:outer_all[name] for name in POOLALL_ACTUAL_ARMS}
    beta=poolall_weights(outer_all['detpool_gpd'],outer_arms,y[outer_cal])
    state=dict(policy=POLICY,variant='documented_actual_submitted_16_arm_refit',identity=POOLALL_IDENTITY,
               arm_order=list(POOLALL_ACTUAL_ARMS),legacy_inventory_count=len(POOLALL_ARMS),beta=beta.tolist(),
               lgb_parameters=dict(LGB_PARAMS,n_jobs=threads,n_estimators=2 if smoke else 400),full_training_rows=len(full),
               inner_states=inner_states,window=WINDOW,ridge=RIDGE,shrink=list(SHRINK),purge_hours=144,
               inner_reference_end=str(train['origins'][inner_ref].max()),inner_calibration_start=str(train['origins'][inner_cal].min()),
               inner_calibration_end=str(train['origins'][inner_cal].max()),inner_boundary=str(inner_boundary),
               outer_reference_end=str(train['origins'][outer_ref].max()),outer_calibration_start=str(train['origins'][outer_cal].min()),
               outer_calibration_end=str(train['origins'][outer_cal].max()),outer_boundary=str(outer_boundary),
               inner_reference_rows=len(inner_ref),inner_calibration_rows=len(inner_cal),outer_reference_rows=len(outer_ref),outer_calibration_rows=len(outer_cal),
               maturity_policy='All new fits use current fit_F* 144h maturity; historical 72h/144h pointweight variants are both retained as named arms even if forecasts coincide.')
    (out/'frozen_calibration.json').write_text(json.dumps(state,indent=2)+'\n')
    np.savez_compressed(out/'prior_predictions.npz',origins=train['origins'][outer_cal].asi8,base=outer_all['detpool_gpd'],**outer_arms)
    query_pred={};seconds=0.
    for component in primitive_names:
        query_pred[component],elapsed=component_predictions(component,subset(train,full),query,y[full],threads,out/'full'/component,smoke)
        seconds+=elapsed
    began=time.perf_counter()
    query_all=post_arm_predictions(query_pred,inner_states)
    result=apply_poolall(query_all['detpool_gpd'],{name:query_all[name] for name in POOLALL_ACTUAL_ARMS},beta)
    seconds+=time.perf_counter()-began
    return result,state,seconds


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=SUPPORTED,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--threads',type=int,default=1);p.add_argument('--period',choices=list(CUTOFFS))
    p.add_argument('--smoke',action='store_true');p.add_argument('--merge',action='store_true')
    args=p.parse_args()
    if args.threads<1:p.error('threads must be positive')
    required_components(args.experiment)
    if args.merge:print(json.dumps(dict(prediction_rows=merge(args.out))));return
    for period in [args.period] if args.period else (['F1'] if args.smoke else list(CUTOFFS)):
        report=run_period(args.experiment,args.data,args.out,period,args.threads,args.smoke)
        print(json.dumps({k:report[k] for k in ['experiment','period','policy','train_origins','query_origins','end_to_end_seconds']}),flush=True)
    if not args.period and not args.smoke:print(json.dumps(dict(prediction_rows=merge(args.out))))


if __name__=='__main__':main()
