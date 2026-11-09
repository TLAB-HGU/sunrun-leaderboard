"""Soft gating of the cluster experts (round 10): convex mixture of the three A+B1 experts of offset_roll_cluster_experts.

Single architecture (CPU XGBoost inference only, official truth enters scoring only):
  - The experts are the frozen round-9 experts (saved boosters, frozen offsets, frozen B1 weights).  Their final
    forecasts are rebuilt for EVERY query row and checked against the round-9 predictions before any mixing.
  - forecast_i = sum_k w_ik * expert_k(i), w_i on the simplex, no further calibration is fitted.
  - Gates (fixed in advance, no tuning, no temperature):
      gmm    experts c3 ; w = posterior of the causal-window GMM (near one-hot: mean entropy < 0.03)
      gmm_t  experts c3 ; w = tempered GMM posterior p^(1/T)/sum, T from a fixed grid chosen on the TRAINING calibration
             holdout (prior-model expert forecasts, mean per-lead scorer S) - never on leaderboard scores
      logit  experts f3 ; w = multinomial logistic regression (C=1) from causal window features
             (8 shape features + level + sd) to the FORWARD cluster, fitted on training origins only
      prior  experts c3 or f3 ; w = training label proportions (control: gate information removed)
      oracle experts f3 ; w = one-hot FORWARD query cluster (ceiling diagnostic, uses observed future speed:
             never eligible for adoption or submission)
  - Unusable windows (< 9 bins or sd < 10 km/s) get the training proportions (logit/prior) or the flat cluster (gmm).
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
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import baseline

H, BLOCK, K = 72, 24, 3
DEFAULTS = dict(seed=42, experts='c3', gate='gmm')
SEEDS = (42, 43, 44)
COMBINATIONS = (('c3', 'gmm'), ('c3', 'gmm_t'), ('f3', 'logit'), ('c3', 'prior'), ('f3', 'prior'), ('f3', 'oracle'))
ELIGIBLE = (('c3', 'gmm'), ('c3', 'gmm_t'), ('f3', 'logit'))
TEMPERATURES = (1., 2., 3., 4., 6., 8., 12., 16., 24.)
EXPERT_ROOT = Path(__file__).resolve().parent.parent / 'offset_roll_cluster_experts' / 'runs'
POLICY = 'frozen round-9 experts; gate fitted on training origins only; official labels used only for scoring'
SHAPE_FEATURES = ['slope_norm', 'delta_norm', 'curv', 'peak_pos', 'trough_pos', 'frac_inc', 'ac1', 'r2_lin']
WINDOW_HOURS, GRID_HOURS, MIN_BINS, MIN_SD, GMM_SEED = 72, 6, 9, 10., 42
LOGIT_C = 1.0


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
    if config['seed'] not in SEEDS or (config['experts'], config['gate']) not in COMBINATIONS:
        raise ValueError('Configuration is outside the preregistered arms')
    return config


def design(grid, start, size):
    return np.hstack([grid['C'], grid['E'][:, start:start + size].reshape(
        len(grid['C']), size * grid['E'].shape[2])]).astype(np.float32)


def apply_offset(raw, offset):
    parent = np.clip(raw, 1., 3000.)
    corrected = np.clip(parent + offset[None, :], 1., 3000.)
    if not np.isfinite(corrected).all():
        raise ValueError('Nonfinite final forecasts')
    return corrected


def apply_short_blend(prediction, persistence, weights):
    blended = prediction.copy()
    leads = len(weights)
    blended[:, :leads] = weights[None, :] * prediction[:, :leads] + (1 - weights[None, :]) * persistence[:, None]
    return np.clip(blended, 1., 3000.)


def origin_values(data, origins, column, *, period):
    frame = pd.read_parquet(data, columns=['origin', column],
                            filters=[('set', '==', 'query'), ('period', '==', period), ('h', '==', 1)])
    series = pd.Series(frame[column].to_numpy(np.float64), index=pd.DatetimeIndex(pd.to_datetime(frame.origin, utc=True)))
    values = series.reindex(pd.DatetimeIndex(origins, tz='UTC')).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f'{period}: {column} missing for some query origins')
    return values


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


# ---------------------------------------------------------------- frozen experts
def expert_forecasts(run, data, period, query, threads):
    """Final A+B1 forecast of each frozen expert for every query row: (3, m, 72)."""
    meta = json.loads((run / f'{period}_metadata.json').read_text())
    if meta['smoke'] or meta['config']['clusters'] != K:
        raise ValueError('Expert run must be a full K=3 run')
    persistence = origin_values(data, query['origins'].asi8, 'v_lag0', period=period)
    out = np.empty((K, len(query['origins']), H))
    for k in range(K):
        state = meta['experts'][str(k)]
        raw = np.empty((len(query['origins']), H))
        for block, start in enumerate(range(0, H, BLOCK)):
            model = xgb.Booster()
            model.load_model(run / f'{period}_models' / f'expert{k}' / 'full' / f'b{block}.ubj')
            model.set_param({'nthread': threads})
            raw[:, start:start + BLOCK] = model.inplace_predict(design(query, start, BLOCK))
        corrected = apply_offset(raw, np.array(state['offset']))
        out[k] = apply_short_blend(corrected, persistence, np.array(state['short_blend']['weights']))
    return out, meta


def check_reproduction(run, period, forecasts, origins):
    """Rebuilt expert forecasts must equal the round-9 saved predictions on the rows each expert predicted."""
    worst = 0.
    for k in range(K):
        with np.load(run / f'{period}_expert{k}_predictions.npz') as saved:
            if not np.array_equal(saved['origins'], origins[saved['rows']]):
                raise ValueError('Saved expert rows do not match the query grid')
            worst = max(worst, float(np.max(np.abs(saved['pred'] - forecasts[k][saved['rows']])))) if len(saved['rows']) else worst
    if worst > 1e-6:
        raise ValueError(f'{period}: rebuilt experts differ from round-9 predictions by {worst}')
    return worst


# ---------------------------------------------------------------- causal / forward windows
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
    return features + [mu, sd] if np.isfinite(features).all() else None


def feature_matrix(bins):
    """(n, 10): 8 shape features + window level + window sd; rows of unusable windows are NaN."""
    features = [shape_features(row) for row in bins]
    usable = np.array([f is not None for f in features])
    matrix = np.full((len(bins), len(SHAPE_FEATURES) + 2), np.nan)
    for i, f in enumerate(features):
        if f is not None:
            matrix[i] = f
    return matrix, usable


class Regimes:
    """GMM on the 8 shape features; clusters numbered by ascending median normalised slope (as in round 9)."""

    def fit(self, matrix, usable, fit_rows):
        rows = fit_rows & usable
        shape = matrix[:, :len(SHAPE_FEATURES)]
        self.scaler = StandardScaler().fit(shape[rows])
        self.gmm = GaussianMixture(K, covariance_type='full', n_init=5, random_state=GMM_SEED,
                                   reg_covar=1e-4).fit(self.scaler.transform(shape[rows]))
        raw = self.gmm.predict(self.scaler.transform(shape[rows]))
        median = {k: float(np.median(shape[rows][raw == k, 0])) for k in range(K)}
        self.remap = {old: new for new, old in enumerate(sorted(median, key=median.get))}
        self.fit_count = int(rows.sum())
        return self

    def proba(self, matrix, usable):
        out = np.zeros((len(matrix), K))
        out[:, K // 2] = 1.
        if usable.any():
            raw = self.gmm.predict_proba(self.scaler.transform(matrix[usable][:, :len(SHAPE_FEATURES)]))
            ordered = np.zeros_like(raw)
            for old, new in self.remap.items():
                ordered[:, new] = raw[:, old]
            out[usable] = ordered
        return out


# ---------------------------------------------------------------- gates
def tempered(proba, temperature):
    power = np.maximum(proba, 1e-300) ** (1. / temperature)
    return power / power.sum(1, keepdims=True)


def select_temperature(run, data, period, train_origins, proba_train, threads):
    """Temperature minimising the mean per-lead scorer S of the mixed PRIOR-model expert forecasts on the training
    calibration holdout (origins >= last-3-months boundary); plain clipped forecasts, no offset/B1 (those were fitted there)."""
    names = baseline.selected_names(list(baseline.FEATURES))
    train, y = baseline.load_grid(data, period, names, True)
    if not train['origins'].equals(train_origins):
        raise ValueError('Training grid differs from the round-9 training origins')
    calibration = baseline.prior_split(train['origins'])[1]
    sub = baseline.subset(train, calibration)
    forecasts = np.empty((K, len(calibration), H))
    for k in range(K):
        for block, start in enumerate(range(0, H, BLOCK)):
            model = xgb.Booster()
            model.load_model(run / f'{period}_models' / f'expert{k}' / 'prior' / f'b{block}.ubj')
            model.set_param({'nthread': threads})
            forecasts[k, :, start:start + BLOCK] = np.clip(model.inplace_predict(design(sub, start, BLOCK)), 1., 3000.)
    truth = y[calibration]
    table = []
    for temperature in TEMPERATURES:
        mixed = np.einsum('mk,kmh->mh', tempered(proba_train[calibration], temperature), forecasts)
        table.append(dict(temperature=temperature, s=float(np.mean([lead_s(mixed[:, h], truth[:, h]) for h in range(H)]))))
    best = min(row['s'] for row in table)
    chosen = min(row['temperature'] for row in table if row['s'] <= best + 1e-12)
    return chosen, table



def prior_weights(train_labels):
    return np.bincount(train_labels, minlength=K) / len(train_labels)


def gate_weights(config, series, period, train_origins, train_labels, query_origins, run=None, data=None, threads=1):
    """Query gate weights (m, 3) and diagnostics.  Everything is fitted on training origins only."""
    cutoff, gate = baseline.CUTOFFS[period], config['gate']
    prior = prior_weights(train_labels)
    info = dict(gate=gate, prior=prior.tolist())
    if gate == 'prior':
        return np.tile(prior, (len(query_origins), 1)), info
    train_matrix, train_usable = feature_matrix(window_bins(series, train_origins, 'causal', cutoff))
    query_matrix, query_usable = feature_matrix(window_bins(series, query_origins, 'causal'))
    if gate in ('gmm', 'gmm_t'):
        fit_rows = np.asarray(pd.DatetimeIndex(train_origins).hour == 0)
        regimes = Regimes().fit(train_matrix, train_usable, fit_rows)
        info.update(fit_windows=regimes.fit_count, query_unusable=int((~query_usable).sum()))
        proba = regimes.proba(query_matrix, query_usable)
        if gate == 'gmm':
            return proba, info
        chosen, table = select_temperature(run, data, period, pd.DatetimeIndex(train_origins),
                                           regimes.proba(train_matrix, train_usable), threads)
        info.update(temperature=chosen, temperature_grid=table, selection='training calibration holdout, prior-model forecasts')
        return tempered(proba, chosen), info
    scaler = StandardScaler().fit(train_matrix[train_usable])
    model = LogisticRegression(C=LOGIT_C, max_iter=2000).fit(scaler.transform(train_matrix[train_usable]),
                                                           train_labels[train_usable])
    if list(model.classes_) != list(range(K)):
        raise ValueError('Gate classifier is missing a class')
    train_proba = model.predict_proba(scaler.transform(train_matrix[train_usable]))
    info.update(train_rows=int(train_usable.sum()), train_accuracy=float((train_proba.argmax(1) == train_labels[train_usable]).mean()),
                train_log_loss=float(-np.log(np.maximum(train_proba[np.arange(len(train_proba)), train_labels[train_usable]], 1e-12)).mean()),
                query_unusable=int((~query_usable).sum()), C=LOGIT_C)
    weights = np.tile(prior, (len(query_origins), 1))
    if query_usable.any():
        weights[query_usable] = model.predict_proba(scaler.transform(query_matrix[query_usable]))
    if gate == 'logit':
        return weights, info
    # oracle: forward-window cluster of each query origin (observed future speed; diagnostic ceiling only)
    forward_train, forward_usable = feature_matrix(window_bins(series, train_origins, 'forward', cutoff))
    regimes = Regimes().fit(forward_train, forward_usable, np.asarray(pd.DatetimeIndex(train_origins).hour == 0))
    forward_query, forward_query_usable = feature_matrix(window_bins(series, query_origins, 'forward'))
    onehot = np.eye(K)[regimes.proba(forward_query, forward_query_usable).argmax(1)]
    weights[forward_query_usable] = onehot[forward_query_usable]
    info.update(oracle_rows=int(forward_query_usable.sum()), eligible=False)
    return weights, info


# ---------------------------------------------------------------- run
def run_period(config, data, out, period, series, threads, identity):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    run = EXPERT_ROOT / f"cluster_{config['experts']}_s{config['seed']}"
    names = baseline.selected_names(list(baseline.FEATURES))
    query, query_truth = baseline.load_grid(data, period, names, False)
    if query_truth is not None or len(query['origins']) != baseline.QUERY_COUNTS[period]:
        raise ValueError('Unexpected official query grid')
    forecasts, meta = expert_forecasts(run, data, period, query, threads)
    reproduction = check_reproduction(run, period, forecasts, query['origins'].asi8)
    with np.load(run / f'{period}_labels.npz') as labels:
        train_origins, train_labels = pd.DatetimeIndex(labels['train_origins'], tz='UTC'), labels['train_labels']
        saved_query_labels = labels['query_labels']
    weights, info = gate_weights(config, series, period, train_origins, train_labels, query['origins'], run, data, threads)
    if not (np.isfinite(weights).all() and (weights >= 0).all() and np.allclose(weights.sum(1), 1.)):
        raise ValueError('Gate weights must lie on the simplex')
    pred = np.einsum('mk,kmh->mh', weights, forecasts)
    if config['gate'] in ('gmm', 'gmm_t'):
        info['hard_label_agreement_with_round9_route'] = float((weights.argmax(1) == saved_query_labels).mean())
    entropy = float(-(weights * np.log(np.maximum(weights, 1e-12))).sum(1).mean())
    origins = query['origins'].asi8
    np.savez_compressed(out / f'{period}_predictions.npz', pred=pred, origins=origins)
    np.savez_compressed(out / f'{period}_gate.npz', weights=weights, experts=forecasts.astype(np.float32), origins=origins)
    report = dict(experiment=config.get('experiment', out.name), period=period, smoke=False, config=config, policy=POLICY,
                  expert_run=str(run), expert_train_cutoff=meta['training_cutoff'], expert_identity=meta.get('identity', {}).get('experiment'),
                  expert_reproduction_max_abs_diff=reproduction, gate=info, mean_weights=weights.mean(0).tolist(),
                  mean_gate_entropy=entropy, query_origins=len(pred), training_cutoff=baseline.CUTOFFS[period],
                  eligible_for_adoption=(config['experts'], config['gate']) in ELIGIBLE,
                  end_to_end_seconds=time.perf_counter() - began, identity=identity,
                  versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                xgboost=xgb.__version__, sklearn=sklearn.__version__))
    write_json(out / f'{period}_metadata.json', report)
    print(json.dumps(dict(period=period, stage='gate', component=config['gate'], unit=1, units=1,
                          mean_weights=report['mean_weights'])), flush=True)
    return report


def prediction_frame(out):
    frames = []
    for period in baseline.CUTOFFS:
        with np.load(Path(out) / f'{period}_predictions.npz') as arrays:
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
    frame = prediction_frame(out)
    frame.to_parquet(out / 'predictions.parquet', index=False)
    observed = competition.load_observations(observations)
    score = competition.score_competition(frame, competition.prepare_context(observed))
    if score['total']['n_replaced']:
        raise ValueError('Scorer replaced forecasts; incomplete runs are not accepted')
    result = dict(competition=score, observations=file_identity(observations), observation_rows=len(observed),
                  scorer=file_identity(competition.__file__), scorer_version=competition.VERSION)
    write_json(out / 'local_score.json', result)
    write_json(out / 'submission.json', dict(result, local_only=True, published=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--experiment')
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU GBM runs require CUDA_VISIBLE_DEVICES="" before startup')
    config = normalize_config(json.loads(args.config.read_text()))
    if args.experiment:
        config['experiment'] = args.experiment
    args.out.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings('ignore', category=FutureWarning)
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
    identity = dict(data=file_identity(args.data), config=file_identity(args.config), experiment=file_identity(__file__),
                    baseline=file_identity(baseline.__file__), git_commit=git.stdout.strip())
    provenance = Path(__file__).with_name('provenance.json')
    if provenance.exists():
        identity['campaign_provenance'] = json.loads(provenance.read_text())
    write_json(args.out / 'resolved_config.json', config)
    series = speed_series(args.data)
    for period in baseline.CUTOFFS:
        run_period(config, args.data, args.out, period, series, args.threads, identity)
    score_outputs(args.out, args.observations)


if __name__ == '__main__':
    main()
