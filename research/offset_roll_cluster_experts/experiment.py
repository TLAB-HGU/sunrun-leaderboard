"""Cluster-expert A+B1 (round 9): calib_offset_roll trained separately per speed-pattern cluster.

Single architecture (CPU XGBoost, official truth enters scoring only):
  - A+B1 pipeline per expert: 30 source features, 3 x 24-lead multi_output_tree boosters, 900 rounds,
    point4 + spectral objective, frozen offset (alpha .25, last 336 calibration rows), B1 blend of leads 1-6
    with causal persistence v_lag0.  Numerics are those of the ablation3 runner.
  - The single change: training rows are split into K=3 speed-pattern clusters and one A+B1 expert is fitted per
    cluster.  The prior/calibration split is the global one (baseline.prior_split) intersected with the cluster,
    so offset and B1 weights are fitted per expert on that expert's own training-only holdout rows.
  - Pattern clusters: GaussianMixture(K=3, full) on 8 level-removed shape features of a 72 h speed window on a 6 h
    grid, rebuilt from df v_lag0 (ACE label origin-1h).  Fitted per period on windows of training origins only
    (24 h spaced) and ordered by normalised slope (falling, flat, rising).
  - label='causal'  : window [t0-72h, t0) for training rows and for query routing (submission eligible).
    label='forward' : window [t0, t0+72h) for TRAINING rows only (the user's clustering); no query routing is
                      possible, every expert predicts every query and the primary forecast is the expert mean.
    label='random'  : causal labels permuted in 72 h blocks (same proportions, no information) - control.
    clusters=1      : one expert on all rows; must reproduce A+B1 (equivalence gate only).
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import warnings

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import baseline

BLOCK, ROUNDS, WINDOW, ALPHA, LEADS = 24, 900, 336, .25, 6
SPECTRAL_COEFFICIENT, TAIL_WEIGHT = .1, 4.
DEFAULTS = dict(seed=42, label='causal', clusters=3)
CHOICES = dict(seed=(42, 43, 44), label=('causal', 'forward', 'random'), clusters=(1, 3))
POLICY = 'training-only clustering, frozen calibration; official labels used only for scoring'
SHAPE_FEATURES = ['slope_norm', 'delta_norm', 'curv', 'peak_pos', 'trough_pos', 'frac_inc', 'ac1', 'r2_lin']
WINDOW_HOURS, GRID_HOURS, MIN_BINS, MIN_SD = 72, 6, 9, 10.
GMM_SEED, BLOCK_PERMUTATION_HOURS = 42, 72
SMOKE_MINIMUM = 3
BLEND_GRID = np.round(np.arange(0., 1.0001, .05), 2)
STATUS0_TRUTH = Path(__file__).with_name('input') / 'train_status0_y_h1to6.parquet'


def file_identity(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def normalize_config(value):
    unknown = set(value) - set(DEFAULTS) - {'experiment'}
    if unknown:
        raise ValueError(f'Unknown experiment configuration keys: {sorted(unknown)}')
    config = dict(DEFAULTS, **value)
    for key, allowed in CHOICES.items():
        if config[key] not in allowed:
            raise ValueError(f'{key} is outside the preregistered choices')
    return config


def feature_names():
    return baseline.selected_names(list(baseline.FEATURES))


def expanded_names(names, start, size):
    return list(names['constant']) + [f'{name}@h{h + 1}'
                                    for h in range(start, start + size)
                                    for name in names['varying']]


def design(grid, start, size):
    return np.hstack([grid['C'], grid['E'][:, start:start + size].reshape(
        len(grid['C']), size * grid['E'].shape[2])]).astype(np.float32)


def sample_weights(target, observed):
    """Point4 tail weights normalised to mean one per lead over observed cells."""
    raw = np.where(target >= 550., TAIL_WEIGHT, 1.) * observed
    return raw / (raw.sum(0) / observed.sum(0))


def make_objective(target):
    if not np.isfinite(target).all():
        raise ValueError('Training targets must be finite')
    observed = np.isfinite(target)
    weights = sample_weights(target, observed)

    def objective(prediction, matrix):
        prediction = np.asarray(prediction).reshape(target.shape)
        error = np.where(observed, prediction - np.nan_to_num(target), 0.)
        z = np.fft.fft(error, axis=1, norm='ortho')
        rho = 1 / np.sqrt(1 + abs(z) ** 2 / 50. ** 2)
        gradient = (error * weights + SPECTRAL_COEFFICIENT * np.fft.ifft(
            z * rho, axis=1, norm='ortho').real) * observed
        hessian = (weights + SPECTRAL_COEFFICIENT) * observed
        return gradient.astype(np.float32), hessian.astype(np.float32)
    return objective


def fit_predict(train, query, target, seed, threads, out, period, stage, component, progress, smoke=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    predictions = np.empty((len(query['origins']), baseline.H), np.float64)
    predict_seconds = 0.
    for block, start in enumerate(range(0, baseline.H, BLOCK)):
        began = time.perf_counter()
        x, xq = design(train, start, BLOCK), design(query, start, BLOCK)
        y = target[:, start:start + BLOCK]
        matrix = xgb.DMatrix(x, label=y.astype(np.float32), nthread=threads)
        params = dict(baseline.PARAMS, seed=seed, device='cpu',
                      nthread=threads, num_target=BLOCK, base_score=float(y.mean()))
        model = xgb.train(params, matrix, num_boost_round=2 if smoke else ROUNDS, obj=make_objective(y))
        model.save_model(out / f'b{block}.ubj')
        inference = time.perf_counter()
        if len(xq):
            predictions[:, start:start + BLOCK] = model.inplace_predict(xq)
        predict_seconds += time.perf_counter() - inference
        progress['unit'] += 1
        print(json.dumps(dict(period=period, stage=stage, component=component, block=block,
                              horizon=start + BLOCK, fit_seconds=round(time.perf_counter() - began, 3),
                              **progress)), flush=True)
        del model, matrix, x, xq
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite component forecasts')
    return predictions, predict_seconds


def component_values(raw):
    return np.clip(raw, 1., 3000.)


def estimate_offset(prior, truth, *, smoke=False):
    if prior.shape != truth.shape or truth.ndim != 2 or truth.shape[1] != baseline.H:
        raise ValueError('Prior predictions and training truth require a shared 72-lead grid')
    if not np.isfinite(truth).all() or not np.isfinite(prior).all():
        raise ValueError('Nonfinite calibration values')
    rows = WINDOW
    if rows > len(truth) and not smoke:
        raise ValueError('Insufficient training calibration rows for frozen window')
    rows = min(rows, len(truth))
    if rows == 0:
        raise ValueError('Empty calibration window')
    residuals = truth[-rows:] - component_values(prior)[-rows:]
    return ALPHA * residuals.mean(0), rows


def apply_offset(raw, offset):
    parent = component_values(raw)
    corrected = np.clip(parent + offset[None, :], 1., 3000.)
    parent = np.clip(parent, 1., 3000.)
    if not np.isfinite(corrected).all() or not np.isfinite(parent).all():
        raise ValueError('Nonfinite final forecasts')
    return corrected, parent


def lead_s(prediction, observed):
    """Scorer per-lead S = sqrt(RMSE*TRMSE) (same quantile/tail-MSE construction as competition.py)."""
    order = np.argsort(observed, kind='stable')
    quantiles = np.quantile(observed, np.arange(101) / 100, method='linear')
    positions = np.searchsorted(observed[order], quantiles[:-1], side='left')
    widths, span = np.diff(quantiles), float(quantiles[-1] - quantiles[0])
    if not span > 0:
        raise ValueError('constant truth makes TIMSE undefined')
    squared = (prediction - observed) ** 2
    suffix = np.cumsum(squared[order][::-1])[::-1]
    timse = float(np.dot(widths, suffix[positions] / (len(observed) - positions)) / span)
    return float(np.sqrt(np.sqrt(squared.mean()) * np.sqrt(timse)))


def origin_values(data, origins, column, *, training, period):
    """One value per origin (h == 1 row) from the run's input df, reindexed to the given origins."""
    filters = [(f'fit_{period}', '==', True)] if training else [('set', '==', 'query'), ('period', '==', period)]
    frame = pd.read_parquet(data, columns=['origin', column], filters=[*filters, ('h', '==', 1)])
    series = pd.Series(frame[column].to_numpy(np.float64), index=pd.DatetimeIndex(pd.to_datetime(frame.origin, utc=True)))
    values = series.reindex(pd.DatetimeIndex(origins, tz='UTC')).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f'{period}: {column} missing for some {"training" if training else "query"} origins')
    return values


def status0_truth(path, origins, leads):
    """Unfilled status-0 training truth (h = 1..leads) for the calibration origins; NaN where unobserved."""
    path = Path(path)
    frame = pd.read_parquet(path, columns=['origin', 'h', 'y'], filters=[('h', '<=', leads)])
    frame['origin'] = pd.to_datetime(frame.origin, utc=True)
    table = frame.pivot(index='origin', columns='h', values='y').reindex(
        index=pd.DatetimeIndex(origins, tz='UTC'), columns=range(1, leads + 1))
    return table.to_numpy(np.float64), file_identity(path)


def fit_short_blend(model, persistence, truth, leads, minimum=100):
    """Per-lead model weight on the calibration holdout; returns weights and the holdout S table."""
    weights, table = [], []
    for h in range(leads):
        valid = np.isfinite(truth[:, h])
        if valid.sum() < minimum:
            raise ValueError(f'h{h + 1}: too few status-0 calibration pairs ({int(valid.sum())})')
        observed = truth[valid, h]
        scores = [lead_s(w * model[valid, h] + (1 - w) * persistence[valid], observed) for w in BLEND_GRID]
        best = min(scores)
        weight = float(max(w for w, s in zip(BLEND_GRID, scores) if s <= best + 1e-12))
        weights.append(weight)
        table.append(dict(lead=h + 1, pairs=int(valid.sum()), weight=weight, s_model=scores[-1],
                          s_persistence=scores[0], s_blend=best))
    return np.array(weights), table


def apply_short_blend(prediction, persistence, weights):
    blended = prediction.copy()
    leads = len(weights)
    blended[:, :leads] = weights[None, :] * prediction[:, :leads] + (1 - weights[None, :]) * persistence[:, None]
    return np.clip(blended, 1., 3000.)


# ---------------------------------------------------------------- speed-pattern clusters
def speed_series(data):
    """Hourly ACE speed from df v_lag0: the value at origin O is the hour labelled O-1h."""
    frame = pd.read_parquet(data, columns=['origin', 'v_lag0'], filters=[('h', '==', 1)])
    index = pd.DatetimeIndex(pd.to_datetime(frame.origin, utc=True)) - pd.Timedelta(hours=1)
    series = pd.Series(frame.v_lag0.to_numpy(np.float64), index=index).sort_index()
    repeated = series[series.index.duplicated(keep=False)].dropna()
    if len(repeated) and float(repeated.groupby(level=0).agg(np.ptp).max()) > 1e-6:
        raise ValueError('Duplicate origins disagree on v_lag0')
    series = series[~series.index.duplicated()]
    return series.reindex(pd.date_range(series.index.min(), series.index.max(), freq='1h'))


def window_bins(series, origins, mode, cutoff=None):
    """(n, 12) 6 h means (>= 2 hourly values) of the 72 h window before ('causal') or after ('forward') each origin."""
    values = series.to_numpy(np.float64).copy()
    if cutoff is not None:
        values[series.index >= pd.Timestamp(cutoff, tz='UTC')] = np.nan
    position = ((pd.DatetimeIndex(origins, tz='UTC') - series.index[0]) / pd.Timedelta(hours=1)).to_numpy().astype(int)
    start = position - WINDOW_HOURS if mode == 'causal' else position
    index = start[:, None] + np.arange(WINDOW_HOURS)[None, :]
    inside = (index >= 0) & (index < len(values))
    hourly = np.where(inside, values[np.clip(index, 0, len(values) - 1)], np.nan)
    hourly = hourly.reshape(len(origins), WINDOW_HOURS // GRID_HOURS, GRID_HOURS)
    finite = np.isfinite(hourly)
    count = finite.sum(2)
    mean = np.where(finite, hourly, 0.).sum(2) / np.maximum(count, 1)
    return np.where(count >= 2, mean, np.nan)


def shape_features(bins):
    """Level-removed shape features of one 12-point window; None when the window is unusable."""
    if int(np.isfinite(bins).sum()) < MIN_BINS:
        return None
    y = pd.Series(bins).interpolate(limit=2, limit_direction='both').to_numpy()
    if not np.isfinite(y).all():
        return None
    nb = len(y)
    mu, sd = float(y.mean()), float(y.std())
    if sd < MIN_SD:
        return None
    tt = np.arange(nb, dtype=float)
    tc = tt - tt.mean()
    slope_bin = float((tc * (y - mu)).sum() / (tc ** 2).sum())
    ss = float(((y - mu) ** 2).sum())
    resid = y - (mu + slope_bin * tc)
    features = [slope_bin / GRID_HOURS * WINDOW_HOURS / sd, float((y[-1] - y[0]) / sd),
                float(np.polyfit(tt, y, 2)[0]) * 1e3, float(tt[int(np.argmax(y))] / (nb - 1)),
                float(tt[int(np.argmin(y))] / (nb - 1)), float((np.diff(y) > 0).mean()),
                float(np.corrcoef(y[:-1], y[1:])[0, 1]), float(1 - (resid ** 2).sum() / ss)]
    return features if np.isfinite(features).all() else None


def feature_matrix(bins):
    features = [shape_features(row) for row in bins]
    usable = np.array([f is not None for f in features])
    matrix = np.full((len(bins), len(SHAPE_FEATURES)), np.nan)
    for i, f in enumerate(features):
        if f is not None:
            matrix[i] = f
    return matrix, usable


class Regimes:
    """GMM on shape features; clusters are numbered by ascending median normalised slope (falling..rising).
    Unusable windows (too few bins or sd < 10 km/s) join the middle cluster (flat)."""

    def __init__(self, clusters):
        self.clusters = clusters

    def fit(self, matrix, usable, fit_rows):
        rows = fit_rows & usable
        self.scaler = StandardScaler().fit(matrix[rows])
        self.gmm = GaussianMixture(self.clusters, covariance_type='full', n_init=5, random_state=GMM_SEED,
                                   reg_covar=1e-4).fit(self.scaler.transform(matrix[rows]))
        raw = self.gmm.predict(self.scaler.transform(matrix[rows]))
        median = {k: float(np.median(matrix[rows][raw == k, 0])) for k in range(self.clusters)}
        self.remap = {old: new for new, old in enumerate(sorted(median, key=median.get))}
        self.fit_count = int(rows.sum())
        return self

    def predict(self, matrix, usable):
        labels = np.full(len(matrix), self.clusters // 2, dtype=int)
        if usable.any():
            raw = self.gmm.predict(self.scaler.transform(matrix[usable]))
            labels[usable] = [self.remap[int(k)] for k in raw]
        return labels

    def describe(self):
        order = [old for old, _ in sorted(self.remap.items(), key=lambda kv: kv[1])]
        return dict(features=SHAPE_FEATURES, fit_windows=self.fit_count, random_state=GMM_SEED,
                    scaler_mean=self.scaler.mean_.tolist(), scaler_scale=self.scaler.scale_.tolist(),
                    weights=self.gmm.weights_[order].tolist(), means=self.gmm.means_[order].tolist(),
                    converged=bool(self.gmm.converged_))


def block_permute(labels, seed):
    """Permute consecutive 72-row blocks (same proportions and persistence, no information)."""
    blocks = [labels[i:i + BLOCK_PERMUTATION_HOURS] for i in range(0, len(labels), BLOCK_PERMUTATION_HOURS)]
    full = [b for b in blocks if len(b) == BLOCK_PERMUTATION_HOURS]
    tail = [b for b in blocks if len(b) != BLOCK_PERMUTATION_HOURS]
    order = np.random.default_rng(seed).permutation(len(full))
    return np.concatenate([full[i] for i in order] + tail) if full else labels.copy()


def cluster_labels(config, series, train_origins, query_origins, period):
    """Training labels (and query routing labels unless label='forward') for one period; fitted on training origins only."""
    clusters, mode, cutoff = config['clusters'], config['label'], baseline.CUTOFFS[period]
    if clusters == 1:
        return np.zeros(len(train_origins), int), np.zeros(len(query_origins), int), dict(kind='single_expert')
    window = 'forward' if mode == 'forward' else 'causal'
    train_matrix, train_usable = feature_matrix(window_bins(series, train_origins, window, cutoff))
    if (pd.DatetimeIndex(train_origins) > pd.Timestamp(cutoff, tz='UTC')).any():
        raise ValueError('Training origins pass the cutoff')
    fit_rows = np.asarray(pd.DatetimeIndex(train_origins).hour == 0)
    regimes = Regimes(clusters).fit(train_matrix, train_usable, fit_rows)
    train_labels = regimes.predict(train_matrix, train_usable)
    info = dict(kind=mode, window=window, gmm=regimes.describe(), train_unusable=int((~train_usable).sum()),
                fit_start=str(pd.DatetimeIndex(train_origins)[fit_rows & train_usable].min()),
                fit_end=str(pd.DatetimeIndex(train_origins)[fit_rows & train_usable].max()))
    if mode == 'forward':
        return train_labels, None, info
    query_matrix, query_usable = feature_matrix(window_bins(series, query_origins, 'causal'))
    query_labels = regimes.predict(query_matrix, query_usable)
    info['query_unusable'] = int((~query_usable).sum())
    if mode == 'random':
        base = 1000 * config['seed'] + int(period[1:])
        train_labels, query_labels = block_permute(train_labels, base), block_permute(query_labels, base + 500)
    return train_labels, query_labels, info


def check_support(labels_train, reference, calibration, cluster, minimum_offset=WINDOW):
    """Fail before any fit when an expert lacks reference/calibration support."""
    ref = int((labels_train[reference] == cluster).sum())
    cal = int((labels_train[calibration] == cluster).sum())
    if ref < 2 or cal < minimum_offset:
        raise ValueError(f'expert {cluster}: reference rows {ref}, calibration rows {cal} (need >= {minimum_offset})')
    return ref, cal


def prepare_period(config, data, period, series, smoke):
    names = feature_names()
    train, y = baseline.load_grid(data, period, names, True)
    query, query_truth = baseline.load_grid(data, period, names, False)
    if query_truth is not None:
        raise ValueError('Query loader must not return evaluation targets')
    if len(query['origins']) != baseline.QUERY_COUNTS[period]:
        raise ValueError('Unexpected official query grid')
    if not np.isfinite(y).all():
        raise ValueError('Prepared training targets must be finite')
    train_labels, query_labels, info = cluster_labels(config, series, train['origins'], query['origins'], period)
    reference, calibration, boundary = baseline.prior_split(train['origins'])
    full = np.arange(len(y))
    if smoke:
        reference = reference[np.linspace(0, len(reference) - 1, min(len(reference), 128)).astype(int)]
        calibration = calibration[np.linspace(0, len(calibration) - 1, min(len(calibration), 64)).astype(int)]
        full = np.linspace(0, len(y) - 1, min(len(y), 192)).astype(int)
        keep = np.arange(min(len(query['origins']), 8))
        query = baseline.subset(query, keep)
        if query_labels is not None:
            query_labels = query_labels[keep]
    return dict(names=names, train=train, y=y, query=query, train_labels=train_labels, query_labels=query_labels,
                info=info, reference=reference, calibration=calibration, full=full, boundary=boundary)


def run_period(config, data, out, period, series, threads=1, smoke=False, identity=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    seed, clusters, mode = config['seed'], config['clusters'], config['label']
    p = prepare_period(config, data, period, series, smoke)
    train, y, query, names = p['train'], p['y'], p['query'], p['names']
    train_labels, query_labels = p['train_labels'], p['query_labels']
    reference, calibration, full = p['reference'], p['calibration'], p['full']
    for k in range(clusters):
        check_support(train_labels, reference, calibration, k, 1 if smoke else WINDOW)
    cal_origins_all = train['origins'][calibration].asi8
    cal_persistence_all = origin_values(data, cal_origins_all, 'v_lag0', training=True, period=period)
    query_persistence_all = origin_values(data, query['origins'].asi8, 'v_lag0', training=False, period=period)
    cal_truth_all, truth_identity = status0_truth(STATUS0_TRUTH, cal_origins_all, LEADS)
    latest_target = pd.Timestamp(int(np.max(cal_origins_all)), tz='UTC') + pd.Timedelta(hours=LEADS)
    if latest_target > pd.Timestamp(baseline.CUTOFFS[period], tz='UTC') - pd.Timedelta(hours=72):
        raise ValueError('Blend calibration truth crosses the training cutoff maturity')
    for k in range(clusters):
        pairs = np.isfinite(cal_truth_all[train_labels[calibration] == k]).sum(0)
        if pairs.min() < (SMOKE_MINIMUM if smoke else 100):
            raise ValueError(f'expert {k}: status-0 calibration pairs per lead {pairs.tolist()} (need >= 100)')
    routed = query_labels is not None
    progress = dict(unit=0, units=2 * clusters * (baseline.H // BLOCK))
    pred = np.full((len(query['origins']), baseline.H), np.nan)
    no_offset = pred.copy()
    experts, seconds = {}, 0.
    for k in range(clusters):
        ref_k = reference[train_labels[reference] == k]
        cal_mask = train_labels[calibration] == k
        cal_k = calibration[cal_mask]
        full_k = full[train_labels[full] == k]
        rows_k = np.flatnonzero(query_labels == k) if routed else np.arange(len(query['origins']))
        component = f'expert{k}'
        model_dir = out / f'{period}_models' / component
        prior, _ = fit_predict(baseline.subset(train, ref_k), baseline.subset(train, cal_k), y[ref_k], seed, threads,
                               model_dir / 'prior', period, 'prior', component, progress, smoke)
        raw, spent = fit_predict(baseline.subset(train, full_k), baseline.subset(query, rows_k), y[full_k], seed,
                                 threads, model_dir / 'full', period, 'full', component, progress, smoke)
        seconds += spent
        truth = y[cal_k]
        offset, offset_rows = estimate_offset(prior, truth, smoke=smoke)
        forecast, forecast_no_offset = apply_offset(raw, offset)
        calibration_model = apply_offset(prior, offset)[0]
        weights, blend_table = fit_short_blend(calibration_model, cal_persistence_all[cal_mask],
                                               cal_truth_all[cal_mask], LEADS, minimum=SMOKE_MINIMUM if smoke else 100)
        unblended = forecast
        forecast = apply_short_blend(forecast, query_persistence_all[rows_k], weights)
        np.savez_compressed(out / f'{period}_expert{k}_predictions.npz', pred=forecast, unblended=unblended,
                            no_offset=forecast_no_offset, rows=rows_k, origins=query['origins'][rows_k].asi8,
                            prior=prior, prior_truth=truth, prior_origins=train['origins'][cal_k].asi8)
        experts[k] = dict(rows=rows_k, pred=forecast, no_offset=forecast_no_offset, state=dict(
            reference_rows=len(ref_k), calibration_rows=len(cal_k), full_rows=len(full_k), query_rows=len(rows_k),
            reference_end=str(train['origins'][ref_k].max()), calibration_start=str(train['origins'][cal_k].min()),
            calibration_end=str(train['origins'][cal_k].max()), offset=offset.tolist(), offset_rows=offset_rows,
            short_blend=dict(leads=LEADS, grid=BLEND_GRID.tolist(), weights=weights.tolist(), holdout=blend_table,
                             calibration_pairs=int(np.isfinite(cal_truth_all[cal_mask]).all(1).sum()))))
    if routed:
        for k, expert in experts.items():
            pred[expert['rows']], no_offset[expert['rows']] = expert['pred'], expert['no_offset']
        assembly = 'hard_route_by_query_label'
    else:
        pred = np.mean([e['pred'] for e in experts.values()], axis=0)
        no_offset = np.mean([e['no_offset'] for e in experts.values()], axis=0)
        assembly = 'uniform_expert_mean_diagnostic_no_routing'
    if not np.isfinite(pred).all():
        raise ValueError('Some queries were not forecast')
    origins = query['origins'].asi8
    np.savez_compressed(out / f'{period}_predictions.npz', pred=pred, origins=origins)
    np.savez_compressed(out / f'{period}_no_offset_predictions.npz', pred=no_offset, origins=origins)
    np.savez_compressed(out / f'{period}_labels.npz', train_origins=train['origins'].asi8, train_labels=train_labels,
                        query_origins=origins, query_labels=query_labels if routed else np.full(len(origins), -1))
    counts = lambda labels: np.bincount(labels, minlength=clusters).tolist() if labels is not None else None
    columns = {str(s): expanded_names(names, s, BLOCK) for s in range(0, baseline.H, BLOCK)}
    report = dict(experiment=config.get('experiment', out.name), period=period, smoke=smoke, config=config,
                  policy=POLICY, seed=seed, assembly=assembly, features=names,
                  feature_count=sum(map(len, names.values())), expanded_feature_count=len(next(iter(columns.values()))),
                  parameters=dict(baseline.PARAMS, seed=seed, nthread=threads),
                  rounds=2 if smoke else ROUNDS, training_cutoff=baseline.CUTOFFS[period], maturity_hours=144,
                  train_origins=len(full), query_origins=len(pred), train_start=str(train['origins'][full].min()),
                  train_end=str(train['origins'][full].max()),
                  query_start=str(pd.Timestamp(origins.min(), tz='UTC')), query_end=str(pd.Timestamp(origins.max(), tz='UTC')),
                  split_boundary=str(p['boundary']), purge_hours=144, labels=p['info'],
                  label_counts=dict(train=counts(train_labels), reference=counts(train_labels[reference]),
                                    calibration=counts(train_labels[calibration]), query=counts(query_labels)),
                  experts={str(k): e['state'] for k, e in experts.items()}, truth_identity=truth_identity,
                  inference_seconds_per_fold=seconds / len(pred), end_to_end_seconds=time.perf_counter() - began,
                  identity=identity or {},
                  versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                xgboost=xgb.__version__, sklearn=sklearn.__version__))
    write_json(out / f'{period}_metadata.json', report)
    return report


def labels_only(config, data, out, series):
    """Dry run: labels and support counts per period without any model fit."""
    report = {}
    for period in baseline.CUTOFFS:
        p = prepare_period(config, data, period, series, False)
        labels, ref, cal = p['train_labels'], p['reference'], p['calibration']
        clusters = config['clusters']
        count = lambda x: np.bincount(x, minlength=clusters).tolist()
        cal_truth = status0_truth(STATUS0_TRUTH, p['train']['origins'][cal].asi8, LEADS)[0]
        pairs = [np.isfinite(cal_truth[labels[cal] == k]).sum(0).tolist() for k in range(clusters)]
        report[period] = dict(train=count(labels), reference=count(labels[ref]), calibration=count(labels[cal]),
                              status0_pairs_per_expert_lead=pairs,
                              query=count(p['query_labels']) if p['query_labels'] is not None else None,
                              info=p['info'])
    write_json(Path(out) / 'labels_only.json', report)
    return report


def prediction_frame(out, suffix='predictions'):
    frames = []
    for period in baseline.CUTOFFS:
        with np.load(Path(out) / f'{period}_{suffix}.npz') as arrays:
            origins = pd.DatetimeIndex(arrays['origins'], tz='UTC')
            pred = arrays['pred']
        if len(origins) != baseline.QUERY_COUNTS[period] or pred.shape != (len(origins), baseline.H):
            raise ValueError(f'{period}: incomplete full prediction grid')
        if origins.has_duplicates or not origins.is_monotonic_increasing or not np.isfinite(pred).all():
            raise ValueError(f'{period}: invalid prediction grid')
        frames.append(pd.DataFrame(dict(origin_last_input_utc=np.repeat(
            origins.strftime('%Y-%m-%dT%H:%M:%SZ'), baseline.H),
            horizon_hours=np.tile(np.arange(1, baseline.H + 1), len(origins)), pred_kms=pred.ravel())))
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(['origin_last_input_utc', 'horizon_hours']).any():
        raise ValueError('Duplicate prediction keys')
    return frame


def score_outputs(out, observations):
    import competition
    out = Path(out)
    for period in baseline.CUTOFFS:
        if json.loads((out / f'{period}_metadata.json').read_text()).get('smoke'):
            raise ValueError('Smoke forecasts cannot be scored as official results')
    frames = {name: prediction_frame(out, suffix) for name, suffix in
              [('predictions', 'predictions'), ('no_offset_predictions', 'no_offset_predictions')]}
    for name, frame in frames.items():
        frame.to_parquet(out / f'{name}.parquet', index=False)
    observed = competition.load_observations(observations)
    context = competition.prepare_context(observed)
    identity = dict(observations=file_identity(observations), observation_rows=len(observed),
                    observation_start=str(observed.timestamp_utc.min()),
                    observation_end=str(observed.timestamp_utc.max()),
                    observation_columns=list(observed.columns),
                    scorer=file_identity(competition.__file__), scorer_version=competition.VERSION)
    scores = {}
    for name, frame in frames.items():
        score = competition.score_competition(frame, context)
        if score['total']['n_replaced']:
            raise ValueError('Scorer replaced forecasts; incomplete ablations are not accepted')
        result = dict(competition=score, **identity)
        filename = 'local_score.json' if name == 'predictions' else 'paired_no_offset_score.json'
        write_json(out / filename, result)
        if name == 'predictions':
            write_json(out / 'submission.json', dict(result, local_only=True, published=False))
        scores[name] = result
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--experiment')
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--period', choices=list(baseline.CUTOFFS))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--score-only', action='store_true')
    parser.add_argument('--labels-only', action='store_true')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU GBM runs require CUDA_VISIBLE_DEVICES="" before startup')
    if args.threads < 1:
        raise ValueError('threads must be positive')
    config = normalize_config(json.loads(args.config.read_text()))
    if args.experiment:
        config['experiment'] = args.experiment
    args.out.mkdir(parents=True, exist_ok=True)
    if args.score_only:
        score_outputs(args.out, args.observations)
        return
    warnings.filterwarnings('ignore', category=FutureWarning)
    series = speed_series(args.data)
    if args.labels_only:
        print(json.dumps(labels_only(config, args.data, args.out, series), indent=1))
        return
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
    identity = dict(data=file_identity(args.data), config=file_identity(args.config),
                    experiment=file_identity(__file__), baseline=file_identity(baseline.__file__),
                    git_commit=git.stdout.strip())
    provenance = Path(__file__).with_name('provenance.json')
    if provenance.exists():
        identity['campaign_provenance'] = json.loads(provenance.read_text())
    write_json(args.out / 'resolved_config.json', config)
    for period in ([args.period] if args.period else list(baseline.CUTOFFS)):
        run_period(config, args.data, args.out, period, series, args.threads, args.smoke, identity)
    if not args.smoke and not args.period:
        score_outputs(args.out, args.observations)


if __name__ == '__main__':
    main()
