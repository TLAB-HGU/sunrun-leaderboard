"""q50_pinball_tail4: q50_pinball plus the baseline point4 tail weight on the smoothed-pinball objective.

Single-component follow-up of q50_pinball (research/offset_roll_quantile_tail/strategy.md): every
(row, horizon) loss term is multiplied by the point4 weight of e2000fs_calib_offset_roll
(4 where the training y >= 550 km/s, else 1, normalised to mean 1 per horizon column), the same
weight on all three levels. By Gneiting (2011, Thm 2.7) the fit targets the tau-quantiles of the
tail-tilted distribution w(y) f(y). Everything else is q50_pinball verbatim.
Replacement ablation of the offset_roll recipe (research/offset_roll_quantile_20261105/strategy.md).
Kept verbatim from the baseline: the 30 features, the 3 x 24-hour multi-output XGBoost blocks
(237 columns each), the tree parameters, 900 rounds, training cutoffs and 144-hour maturity.
Removed: the prior model, the frozen per-horizon offset and the spectral/point4 objective.
Added: a Gaussian convolution-smoothed pinball objective (QXGB, Yao & Franklin arXiv:2608.15290)
fitting tau = 0.1/0.5/0.9 jointly per block on block-standardised training y, monotone
rearrangement of the level axis (Chernozhukov et al. 2010), and q50 as the leaderboard pred_kms.
Only training rows are read for fitting; no query truth is read.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
import numpy as np
import pandas as pd
import scipy
from scipy.special import ndtr
import xgboost as xgb


EXPERIMENT = 'q50_pinball_tail4'

SEEDS = (42, 43, 44)

H = 72

CUTOFFS = {'F1': '2025-12-29', 'F2': '2026-03-29', 'F3': '2026-06-29'}

QUERY_COUNTS = {'F1': 2089, 'F2': 2113, 'F3': 2113}

BASE_NAMES = {'constant': ['v_lag0', 'v_lag1', 'v_lag3', 'v_lag6', 'v_lag12', 'v_lag24', 'v_lag48', 'v_lag72', 'v_lag120', 'v_m6', 'v_sd6', 'v_m24', 'v_sd24', 'v_m72', 'v_sd72', 'ace_speed_kms_last', 'ace_speed_kms_m6', 'ace_speed_kms_m24', 'ace_density_cm3_last', 'ace_density_cm3_m6', 'ace_density_cm3_m24', 'ace_temperature_k_last', 'ace_temperature_k_m6', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6', 'ace_bt_nt_m24', 'ace_bz_gsm_nt_last', 'ace_bz_gsm_nt_m6', 'ace_bz_gsm_nt_m24', 'ace_by_gsm_nt_last', 'ace_by_gsm_nt_m6', 'ace_by_gsm_nt_m24', 'ace_epam_e_channel1_last', 'ace_epam_e_channel1_m6', 'ace_epam_e_channel1_m24', 'ace_epam_e_channel2_last', 'ace_epam_e_channel2_m6', 'ace_epam_e_channel2_m24', 'ace_epam_p_channel1_last', 'ace_epam_p_channel1_m6', 'ace_epam_p_channel1_m24', 'ace_epam_p_channel3_last', 'ace_epam_p_channel3_m6', 'ace_epam_p_channel3_m24', 'ace_epam_p_channel5_last', 'ace_epam_p_channel5_m6', 'ace_epam_p_channel5_m24', 'ace_sis_gt10_flux_last', 'ace_sis_gt10_flux_m6', 'ace_sis_gt10_flux_m24', 'speed_d24', 'epam_p1_d6', 'ch_ch_t035_area', 'ch_ch_t035_max_area', 'ch_ch_t035_top1_area', 'ch_ch_t035_top2_area', 'ch_ch_t035_top3_area', 'ch_ch_t035_meridian15_area', 'ch_ch_t035_meridian15_equatorial_area', 'ch_ch_t035_meridian30_area', 'ch_ch_t035_meridian30_equatorial_area', 'ch_ch_t035_equatorial_area', 'ch_ch_t045_area', 'ch_ch_t045_max_area', 'ch_ch_t045_top1_area', 'ch_ch_t045_top2_area', 'ch_ch_t045_top3_area', 'ch_ch_t045_meridian15_area', 'ch_ch_t045_meridian15_equatorial_area', 'ch_ch_t045_meridian30_area', 'ch_ch_t045_meridian30_equatorial_area', 'ch_ch_t045_equatorial_area', 'ch_ch_t055_area', 'ch_ch_t055_max_area', 'ch_ch_t055_top1_area', 'ch_ch_t055_top2_area', 'ch_ch_t055_top3_area', 'ch_ch_t055_meridian15_area', 'ch_ch_t055_meridian15_equatorial_area', 'ch_ch_t055_meridian30_area', 'ch_ch_t055_meridian30_equatorial_area', 'ch_ch_t055_equatorial_area', 'img_age_h', 'img_stale', 'enlil_next_arr_h', 'enlil_since_arr_h', 'enlil_n_pending', 'enlil_kp', 'enlil_glancing'], 'varying': ['rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed', 'strip_v350_dark', 'strip_v350_cover', 'strip_v450_logmed', 'strip_v450_dark', 'strip_v450_cover', 'strip_v550_logmed', 'strip_v550_dark', 'strip_v550_cover', 'strip_v650_logmed', 'strip_v650_dark', 'strip_v650_cover', 'strip_v750_logmed', 'strip_v750_dark', 'strip_v750_cover', 'enlil_dt_target']}

FEATURES = ['v_lag0', 'rec_mean', 'enlil_kp', 'ch_ch_t035_meridian30_equatorial_area', 'ace_bt_nt_last', 'ace_speed_kms_last', 'ch_ch_t035_equatorial_area', 'ch_ch_t035_meridian15_equatorial_area', 'ace_bt_nt_m6', 'rec26', 'strip_v450_logmed', 'strip_v350_logmed', 'ch_ch_t055_equatorial_area', 'rec27', 'ace_density_cm3_last', 'ch_ch_t055_meridian30_equatorial_area', 'ace_by_gsm_nt_m24', 'ace_density_cm3_m6', 'ace_temperature_k_last', 'enlil_n_pending', 'ace_temperature_k_m24', 'strip_v750_logmed', 'ch_ch_t055_meridian15_area', 'ch_ch_t035_top2_area', 'strip_v550_dark', 'strip_v650_dark', 'ace_sis_gt10_flux_last', 'ace_epam_p_channel3_last', 'img_age_h', 'rec28']

PARAMS = {'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 20, 'reg_lambda': 5, 'objective': 'reg:squarederror', 'tree_method': 'hist', 'device': 'cpu', 'multi_strategy': 'multi_output_tree', 'eval_metric': 'rmse', 'seed': 42, 'verbosity': 0}

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


# Objective package (the single replacement): smoothed pinball needs a standardised target,
# a zero base score and a bounded leaf step because the Hessian vanishes in the tails.
OBJECTIVE_PARAMS = {'base_score': 0.0, 'max_delta_step': 1.0, 'disable_default_eval_metric': True}

LEVELS = np.array([0.1, 0.5, 0.9])

BANDWIDTH = 0.5

ROUNDS = 900

BLOCK = 24

STARTS = (0, 24, 48)

POLICY = 'no calibration; point4 tail-weighted smoothed-pinball q10/q50/q90; pred_kms = rearranged q50'


def experiment_name(seed):
    return f'{EXPERIMENT}_s{seed}'


def parse_experiment(name):
    match = re.fullmatch(rf'{EXPERIMENT}_s(\d+)', name or '')
    if not match or int(match.group(1)) not in SEEDS:
        raise ValueError(f'--experiment must be one of {[experiment_name(s) for s in SEEDS]}')
    return int(match.group(1))


def smoothed_pinball(u, tau, bandwidth=BANDWIDTH):
    """Gaussian convolution of the check loss, u = y - prediction."""
    z = u / bandwidth
    return bandwidth * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi) + u * (tau - ndtr(-z))


TAIL_THRESHOLD = 550.0

TAIL_WEIGHT = 4.0


def tail_weights(target_kms):
    """Baseline point4 weight, copied from e2000fs_calib_offset_roll (training y is fully observed after the upstream ffill)."""
    observed = np.isfinite(target_kms)
    raw = np.where(target_kms >= TAIL_THRESHOLD, TAIL_WEIGHT, 1.0) * observed
    return raw / (raw.sum(0) / observed.sum(0))


def pinball_objective(target, weights, levels=LEVELS, bandwidth=BANDWIDTH):
    """Weighted gradient/Hessian with respect to the prediction; outputs are level-major (level, horizon)."""
    if not np.isfinite(target).all():
        raise ValueError('Quantile labels must be finite after upstream training-y ffill')
    if weights.shape != target.shape or not (np.isfinite(weights).all() and (weights > 0).all()):
        raise ValueError('Tail weights must be positive and match the target grid')
    tau = np.repeat(levels, target.shape[1])[None, :]
    labels = np.tile(target, (1, len(levels)))
    tiled = np.tile(weights, (1, len(levels)))

    def objective(prediction, matrix):
        prediction = np.asarray(prediction, dtype=np.float64).reshape(labels.shape)
        z = (labels - prediction) / bandwidth
        gradient = tiled * (ndtr(-z) - tau)
        hessian = tiled * np.maximum(np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi) / bandwidth, 1e-6)
        return gradient.astype(np.float32), hessian.astype(np.float32)
    return objective, labels, tau, tiled


def block_features(names, start):
    return [*names['constant'], *[f'{f}@h{h}' for h in range(start + 1, start + BLOCK + 1) for f in names['varying']]]


def fit_block(x, xq, target, seed, rounds, threads, out, label):
    """One multi-output booster for 24 horizons x 3 levels; returns (n, 24, 3) km/s quantiles."""
    mu, sd = float(target.mean()), float(target.std())
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError('Quantile target scale must be positive')
    standard = (target - mu) / sd
    objective, labels, tau, tiled = pinball_objective(standard, tail_weights(target))
    matrix = xgb.DMatrix(x, label=labels.astype(np.float32), nthread=threads)
    params = dict(PARAMS, **OBJECTIVE_PARAMS, seed=seed, nthread=threads, num_target=labels.shape[1])

    def pinball(prediction, dmatrix):
        residual = labels - np.asarray(prediction).reshape(labels.shape)
        return 'weighted_pinball', float((tiled * np.maximum(tau * residual, (tau - 1) * residual)).mean())

    history = {}
    model = xgb.train(params, matrix, num_boost_round=rounds, obj=objective, evals=[(matrix, 'train')],
                      custom_metric=pinball, evals_result=history, verbose_eval=False)
    model.save_model(out / f'{label}.ubj')
    began = time.perf_counter()
    raw = model.inplace_predict(xq).astype(np.float64).reshape(len(xq), len(LEVELS), BLOCK).transpose(0, 2, 1)
    crossing = float((np.diff(raw, axis=2) < 0).any(axis=2).mean())
    quantiles = np.clip(np.sort(raw, axis=2) * sd + mu, 1., 3000.)
    seconds = time.perf_counter() - began
    if not np.isfinite(quantiles).all():
        raise ValueError('Nonfinite quantile prediction')
    curve = history['train']['weighted_pinball']
    return quantiles, dict(mu=mu, sd=sd, crossing_fraction_before_sort=crossing, predict_seconds=seconds,
                           train_pinball_first=curve[0], train_pinball_last=curve[-1],
                           train_pinball_every50=curve[::50] + [curve[-1]])


def run_period(data, out, period, seed, threads=1, smoke=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); began = time.perf_counter()
    names = selected_names(FEATURES)
    train, y = load_grid(data, period, names, True); query, _ = load_grid(data, period, names, False)
    if len(query['origins']) != QUERY_COUNTS[period]: raise ValueError('Unexpected query grid')
    if not np.isfinite(y).all(): raise ValueError('Requires the prepared training-y ffill snapshot')
    if smoke:
        rows = np.linspace(0, len(y) - 1, min(len(y), 192)).astype(int)
        train, y = {k: v[rows] for k, v in train.items()}, y[rows]
        query = {k: v[:min(8, len(query['origins']))] for k, v in query.items()}
    rounds = 2 if smoke else ROUNDS
    models = out / f'{period}_models'; models.mkdir(parents=True, exist_ok=True)
    quantiles = np.empty((len(query['origins']), H, len(LEVELS)), np.float64)
    blocks, seconds = [], 0.
    for b, start in enumerate(STARTS):
        fit_began = time.perf_counter()
        x, xq = design(train, start), design(query, start)
        if x.shape[1] != len(block_features(names, start)): raise ValueError('Unexpected design width')
        quantiles[:, start:start + BLOCK], info = fit_block(x, xq, y[:, start:start + BLOCK], seed, rounds,
                                                            threads, models, f'b{b}')
        seconds += info['predict_seconds']
        info.update(block=b, horizons=[start + 1, start + BLOCK], feature_count=x.shape[1],
                    fit_seconds=round(time.perf_counter() - fit_began, 3))
        blocks.append(info)
        print(json.dumps(dict(period=period, stage='full', component='pinball', block=b, horizon=start + BLOCK,
                              unit=b + 1, units=len(STARTS), fit_seconds=info['fit_seconds'],
                              crossing=round(info['crossing_fraction_before_sort'], 4))), flush=True)
    q50 = quantiles[:, :, 1]
    np.savez_compressed(out / f'{period}_predictions.npz', pred=q50, q10=quantiles[:, :, 0], q50=q50,
                        q90=quantiles[:, :, 2], origins=query['origins'].asi8)
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], text=True, capture_output=True, check=False,
                              cwd=Path(__file__).resolve().parent)
    report = dict(experiment=experiment_name(seed), period=period, smoke=smoke, policy=POLICY,
                  classification='single-component follow-up of q50_pinball (point4 tail weight); replacement ablation of e2000fs_calib_offset_roll',
                  data=str(Path(data).resolve()), data_bytes=Path(data).stat().st_size,
                  data_mtime_ns=Path(data).stat().st_mtime_ns, git_commit=revision.stdout.strip(), seed=seed,
                  features=names, feature_count=dict(constant=len(names['constant']), varying=len(names['varying']),
                                                     per_block=len(block_features(names, 0))),
                  block_features={f'b{b}': block_features(names, s) for b, s in enumerate(STARTS)},
                  parameters=dict(PARAMS, **OBJECTIVE_PARAMS, seed=seed, nthread=threads), rounds=rounds,
                  distribution=dict(levels=LEVELS.tolist(), loss='gaussian convolution-smoothed pinball',
                                    bandwidth_standardised=BANDWIDTH, standardisation='per block, training y',
                                    rearrangement='sort level axis', clip=[1., 3000.], point_readout='q50',
                                    tail_weight=dict(threshold_kms=TAIL_THRESHOLD, weight=TAIL_WEIGHT,
                                                     normalisation='mean 1 per horizon column (baseline point4)',
                                                     levels='same weight on q10/q50/q90')),
                  blocks=blocks, training_cutoff=CUTOFFS[period], maturity_hours=144,
                  train_origins=len(y), train_start=str(train['origins'].min()), train_end=str(train['origins'].max()),
                  query_origins=len(q50), query_start=str(query['origins'].min()), query_end=str(query['origins'].max()),
                  target_policy='Upstream causal ffill of feature and training-y columns only; no query truth read.',
                  inference_seconds_per_fold=seconds / len(q50), end_to_end_seconds=time.perf_counter() - began,
                  versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                scipy=scipy.__version__, xgboost=xgb.__version__))
    (out / f'{period}_metadata.json').write_text(json.dumps(report, indent=2) + '\n'); return report


def merge(out):
    out = Path(out)
    frames = []
    for period in CUTOFFS:
        z = np.load(out / f'{period}_predictions.npz')
        origins = pd.DatetimeIndex(z['origins'], tz='UTC')
        if len(origins) != QUERY_COUNTS[period] or z['pred'].shape != (len(origins), H):
            raise ValueError(f'incomplete full predictions for {period}')
        if not all(np.isfinite(z[k]).all() for k in ('q10', 'q50', 'q90')) or (z['q10'] > z['q50']).any() or (z['q50'] > z['q90']).any():
            raise ValueError(f'invalid quantile grid for {period}')
        frames.append(pd.DataFrame({'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), H),
                                    'horizon_hours': np.tile(np.arange(1, H + 1), len(origins)),
                                    'pred_kms': z['q50'].ravel(), 'q10_kms': z['q10'].ravel(),
                                    'q50_kms': z['q50'].ravel(), 'q90_kms': z['q90'].ravel()}))
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(['origin_last_input_utc', 'horizon_hours']).any():
        raise ValueError('duplicate prediction keys')
    result[['origin_last_input_utc', 'horizon_hours', 'pred_kms']].to_parquet(out / 'predictions.parquet', index=False)
    result.to_parquet(out / 'quantiles.parquet', index=False)
    return len(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['cpu'], default='cpu')
    p.add_argument('--threads', type=int, default=1); p.add_argument('--period', choices=list(CUTOFFS))
    p.add_argument('--smoke', action='store_true'); p.add_argument('--merge', action='store_true')
    args = p.parse_args()
    seed = parse_experiment(args.experiment)
    if args.threads < 1: p.error('threads must be positive')
    if args.merge: print(json.dumps(dict(prediction_rows=merge(args.out)))); return
    for period in [args.period] if args.period else (['F1'] if args.smoke else list(CUTOFFS)):
        report = run_period(args.data, args.out, period, seed, args.threads, args.smoke)
        print(json.dumps({k: report[k] for k in ['experiment', 'period', 'train_origins', 'query_origins', 'end_to_end_seconds']}), flush=True)
    if not args.period and not args.smoke: print(json.dumps(dict(prediction_rows=merge(args.out))))


if __name__ == "__main__":
    main()
