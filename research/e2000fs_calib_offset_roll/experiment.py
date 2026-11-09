"""e2000fs_calib_offset_roll — board row (S 96.31): drop_t035_v650 + v_lag24/48/72, mean of seeds 42/43/44

The calib_offset_roll recipe without suvi_t035_equatorial_area and strip_v650_dark,
plus three constant speed lags (v_lag24, v_lag48, v_lag72). The board row of this name is
the fixed equal-weight mean of seeds 42/43/44 of this recipe; it was published under the
original name on user request (2026-11-09), replacing the 2026-11-04 rerun of the original
recipe (S 98.76). Source runs: research/lag24_48_72_confirm_20261109/runs/lag24_48_72_s4{2,3,4}.

This one file runs the whole experiment, from the raw table to the board forecast:
  1. data   data-v2/input/df.parquet -> causal per-lead feature forward fill over the
            train+query union -> training labels forward filled on target_time
            (query truth is never changed and never read before scoring)
  2. train  for each seed (42, 43, 44) and period (F1, F2, F3): a prior model on the
            reference window (origins + 144 h <= last 3 months) and a full model on all
            training origins; each is 3 lead blocks of 24 x 900 XGBoost rounds
  3. offset frozen per-lead offset = 0.25 x mean residual of the prior model over the
            last 336 calibration origins (training data only)
  4. infer  full-model query forecast, clip [1, 3000], + offset, clip [1, 3000]
  5. mean3  fixed equal-weight mean of the three seed forecasts (no fitted weights)
  6. score  official scorer leaderboard/scorer/competition.py; labels are used only here

Outputs (in this folder unless --out is given):
  weights/F{1,2,3}/s{seed}.npz   one checkpoint per seed x period: prior_b0..2 and
                                 full_b0..2 boosters (UBJSON bytes) + frozen calibration
  results/s{seed}/               predictions.parquet, score.json
  results/mean3/                 predictions.parquet, score.json (the board forecast)
  logs/*.jsonl                   every step, one JSON object per line

Usage (GBM: CPU only):
  CUDA_VISIBLE_DEVICES= python experiment.py --threads 4
  CUDA_VISIBLE_DEVICES= python experiment.py --predict-only --out DIR   # inference from weights/
  CUDA_VISIBLE_DEVICES= python experiment.py --smoke --out DIR          # 2-round wiring check
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import platform
import socket
import subprocess
import time

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = 'e2000fs_calib_offset_roll'
SEEDS = (42, 43, 44)
H = 72
CUTOFFS = {'F1': '2025-12-29', 'F2': '2026-03-29', 'F3': '2026-06-29'}
QUERY_COUNTS = {'F1': 2089, 'F2': 2113, 'F3': 2113}
METADATA = ('set', 'origin', 'h', 'target_time', 'period', 'fit_F1', 'fit_F2', 'fit_F3', 'y')
PARENT_CONSTANT = ['v_lag0', 'ace_speed_kms_last', 'ace_density_cm3_last', 'ace_density_cm3_m6',
                  'ace_temperature_k_last', 'ace_temperature_k_m24', 'ace_bt_nt_last',
                  'ace_bt_nt_m6', 'ace_by_gsm_nt_m24', 'ace_epam_p_channel3_last',
                  'ace_sis_gt10_flux_last', 'suvi_t035_top2_area',
                  'suvi_t035_meridian15_equatorial_area', 'suvi_t035_meridian30_equatorial_area',
                  'suvi_t055_meridian15_area', 'suvi_t055_meridian30_equatorial_area',
                  'suvi_t055_equatorial_area', 'img_age_h', 'enlil_n_pending', 'enlil_kp']
ADDED_CONSTANT = ['v_lag24', 'v_lag48', 'v_lag72']
CONSTANT = PARENT_CONSTANT + ADDED_CONSTANT
VARYING = ['rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed', 'strip_v450_logmed',
           'strip_v550_dark', 'strip_v750_logmed']
PARAMS = {'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7,
          'min_child_weight': 20, 'reg_lambda': 5, 'objective': 'reg:squarederror',
          'tree_method': 'hist', 'device': 'cpu', 'multi_strategy': 'multi_output_tree',
          'eval_metric': 'rmse', 'seed': 42, 'verbosity': 0}
ROUNDS = 900
BLOCK = 24
TAIL_THRESHOLD, TAIL_WEIGHT = 550., 4.
SPECTRAL_COEFFICIENT = .1
ALPHA = .25
OFFSET_WINDOW = 336
PURGE_HOURS = 144
POLICY = 'training-only frozen calibration; official labels used only for scoring'
KEYS = ['origin_last_input_utc', 'horizon_hours']


class Logs:
    """Append-only JSON-lines logs, one file per stream (run, data, s42, s43, s44, mean3)."""

    def __init__(self, folder, run_id):
        self.folder, self.run_id = Path(folder), run_id
        self.folder.mkdir(parents=True, exist_ok=True)

    def __call__(self, stream, event, **fields):
        record = dict(ts=datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                      run_id=self.run_id, experiment=EXPERIMENT, event=event, **fields)
        line = json.dumps(record, allow_nan=False, default=str)
        with (self.folder / f'{stream}.jsonl').open('a') as handle:
            handle.write(line + '\n')
        print(line, flush=True)


def file_identity(path):
    stat = Path(path).stat()
    return dict(path=str(Path(path).resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


# 1. data -----------------------------------------------------------------------------------

def timeline(origin, horizon):
    """Representative rows, original-row -> unique (origin, h) indices, and h boundaries."""
    if len(origin) == 0:
        raise ValueError('The source table is empty')
    if np.isnat(origin).any() or pd.isna(horizon).any():
        raise ValueError('origin and h must not contain missing values')
    order = np.lexsort((origin, horizon))
    sorted_origin, sorted_h = origin[order], horizon[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = (sorted_origin[1:] != sorted_origin[:-1]) | (sorted_h[1:] != sorted_h[:-1])
    representatives = order[first]
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.cumsum(first, dtype=np.int64) - 1
    unique_h = horizon[representatives]
    boundaries = np.r_[0, np.flatnonzero(unique_h[1:] != unique_h[:-1]) + 1, len(unique_h)]
    return representatives, inverse, boundaries


def fill_feature(values, representatives, inverse, boundaries, name):
    """Validate repeated inputs, then forward-fill independently along each h timeline."""
    unique = values[representatives]
    repeated = unique[inverse]
    equal = (values == repeated) | (np.isnan(values) & np.isnan(repeated))
    if not equal.all():
        bad = int(np.flatnonzero(~equal)[0])
        raise ValueError(f'Conflicting duplicate (origin, h) values in {name!r}, source row {bad}')
    del repeated, equal
    before = int(np.isnan(values).sum())
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        segment = unique[start:stop]
        valid = ~np.isnan(segment)
        previous = np.maximum.accumulate(np.where(valid, np.arange(len(segment)), -1))
        take = previous >= 0
        segment[take] = segment[previous[take]]
    filled = unique[inverse]
    after = int(np.isnan(filled).sum())
    return filled, dict(missing_before=before, missing_after=after, filled=before - after)


def fill_training_labels(metadata):
    """y with training rows filled from the last observed target timestamp; query y unchanged."""
    time_axis = pd.to_datetime(metadata.target_time, utc=True)
    if time_axis.isna().any():
        raise ValueError('target_time must be present')
    timeline_frame = pd.DataFrame({'time': time_axis, 'y': metadata.y.to_numpy()})
    if timeline_frame.groupby('time').y.nunique(dropna=False).gt(1).any():
        raise ValueError('The same target timestamp has inconsistent source labels')
    timeline_frame = timeline_frame.drop_duplicates('time').sort_values('time').set_index('time')
    filled = timeline_frame.y.ffill()
    labels = metadata.y.to_numpy(copy=True)
    training = metadata['set'].eq('train').to_numpy()
    labels[training] = filled.reindex(pd.DatetimeIndex(time_axis[training])).to_numpy()
    original = metadata.y.to_numpy()
    if not np.array_equal(labels[~training], original[~training], equal_nan=True):
        raise RuntimeError('Query truth was modified')
    if not np.array_equal(labels[np.isfinite(original)], original[np.isfinite(original)], equal_nan=True):
        raise RuntimeError('Observed labels were modified')
    return labels


def load_table(path, logs):
    """The model table: metadata plus this experiment's features, causally filled."""
    began = time.perf_counter()
    features = CONSTANT + VARYING
    table = pd.read_parquet(path, columns=[*METADATA, *features])
    origin = pd.DatetimeIndex(table['origin']).asi8.view('datetime64[ns]')
    representatives, inverse, boundaries = timeline(origin, table['h'].to_numpy())
    stats = {}
    for name in features:
        table[name], stats[name] = fill_feature(table[name].to_numpy(), representatives, inverse,
                                                boundaries, name)
    training = table['set'].eq('train').to_numpy()
    missing = int(table['y'][training].isna().sum())
    table['y'] = fill_training_labels(table)
    logs('data', 'table_ready', source=file_identity(path), rows=len(table),
         unique_origin_h_keys=len(representatives), features=features,
         feature_ffill=stats, training_y_missing_before=missing,
         training_y_missing_after=int(np.isnan(table['y'].to_numpy()[training]).sum()),
         policy='feature ffill by h over sorted origin (train+query union, leading NaNs kept); '
                'training y ffill on sorted target_time; query y unchanged',
         seconds=round(time.perf_counter() - began, 3))
    return table


def load_grid(table, period, training):
    """Constant (origin) and varying (origin x 72 leads) arrays for training or query origins."""
    if training:
        mask = table[f'fit_{period}'].to_numpy()
    else:
        mask = (table['set'].eq('query') & table['period'].eq(period)).to_numpy()
    selected = table[mask]
    constant = selected.loc[selected['h'].eq(1).to_numpy(), ['origin', *CONSTANT]].sort_values(
        'origin', kind='stable')
    origins = pd.DatetimeIndex(pd.to_datetime(constant.origin, utc=True))
    if origins.has_duplicates:
        raise ValueError('duplicate input origins')
    columns = ['origin', 'h', *VARYING] + (['y'] if training else [])
    rows = selected[columns].sort_values(['origin', 'h'], kind='stable')
    if len(rows) != len(origins) * H or not np.array_equal(
            rows.h.to_numpy().reshape(-1, H), np.broadcast_to(np.arange(1, H + 1), (len(origins), H))):
        raise ValueError('incomplete origin/horizon grid')
    if not origins.equals(pd.DatetimeIndex(pd.to_datetime(rows.origin.iloc[::H], utc=True))):
        raise ValueError('constant and varying grids disagree')
    if training and not (origins + pd.Timedelta(hours=PURGE_HOURS)
                         <= pd.Timestamp(CUTOFFS[period], tz='UTC')).all():
        raise ValueError('training cutoff violated')
    grid = {'origins': origins, 'C': constant[CONSTANT].to_numpy(np.float32),
            'E': rows[VARYING].to_numpy(np.float32).reshape(len(origins), H, len(VARYING))}
    if np.isinf(grid['C']).any() or np.isinf(grid['E']).any():
        raise ValueError('Infinite feature values are not supported')
    return grid, (rows.y.to_numpy(np.float64).reshape(len(origins), H) if training else None)


def prior_split(origins):
    """Reference rows end 144 h before the last three months; those months calibrate the offset."""
    origins = pd.DatetimeIndex(origins)
    if not origins.is_monotonic_increasing or origins.has_duplicates:
        raise ValueError('Training origins must be unique and sorted')
    boundary = origins.max() - pd.DateOffset(months=3)
    calibration = np.flatnonzero(origins >= boundary)
    reference = np.flatnonzero(origins + pd.Timedelta(hours=PURGE_HOURS) <= boundary)
    if len(reference) < 2 or len(calibration) < 48:
        raise ValueError('Insufficient strictly prior reference/calibration support')
    return reference, calibration, boundary


def subset(grid, indices):
    return {key: value[indices] for key, value in grid.items()}


# 2. model ----------------------------------------------------------------------------------

def design(grid, start, size):
    """Parent constant features, the parent varying features of each lead, then the added
    constant features (column order of the published run)."""
    base = len(PARENT_CONSTANT)
    return np.hstack([grid['C'][:, :base],
                      grid['E'][:, start:start + size].reshape(len(grid['C']), size * grid['E'].shape[2]),
                      grid['C'][:, base:]]).astype(np.float32)


def make_objective(target):
    """Tail-weighted squared error (targets >= 550 km/s weigh 4, normalised per lead) plus a Huber-like
    spectral term on the 24-lead error (Hessian = weight + coefficient)."""
    if not np.isfinite(target).all():
        raise ValueError('Training targets must be finite')
    observed = np.isfinite(target)
    raw = np.where(target >= TAIL_THRESHOLD, TAIL_WEIGHT, 1.) * observed
    weights = raw / (raw.sum(0) / observed.sum(0))

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


def load_booster(buffer, threads):
    booster = xgb.Booster(params={'nthread': threads, 'device': 'cpu'})
    booster.load_model(bytearray(np.asarray(buffer, np.uint8).tobytes()))
    return booster


def fit_predict(train, query, target, seed, threads, rounds, logs, stream, period, stage, progress):
    """Three 24-lead blocks; returns forecasts, inference seconds and the boosters' UBJSON bytes."""
    predictions = np.empty((len(query['origins']), H), np.float64)
    boosters, predict_seconds = [], 0.
    for block, start in enumerate(range(0, H, BLOCK)):
        began = time.perf_counter()
        x, xq = design(train, start, BLOCK), design(query, start, BLOCK)
        y = target[:, start:start + BLOCK]
        matrix = xgb.DMatrix(x, label=y.astype(np.float32), nthread=threads)
        params = dict(PARAMS, seed=seed, device='cpu', nthread=threads, num_target=BLOCK,
                      base_score=float(y.mean()))
        model = xgb.train(params, matrix, num_boost_round=rounds, obj=make_objective(y))
        buffer = np.frombuffer(bytes(model.save_raw('ubj')), np.uint8)
        inference = time.perf_counter()
        predictions[:, start:start + BLOCK] = model.inplace_predict(xq)
        predict_seconds += time.perf_counter() - inference
        replay = load_booster(buffer, threads)
        replay_error = float(np.max(np.abs(replay.inplace_predict(xq) - predictions[:, start:start + BLOCK])))
        if replay_error != 0. or replay.num_boosted_rounds() != rounds:
            raise ValueError('Saved booster did not exactly reproduce its forecasts/rounds')
        boosters.append(buffer)
        progress['unit'] += 1
        logs(stream, 'fit', period=period, stage=stage, component='spectral', block=block,
             horizon=start + BLOCK, rounds=rounds, train_rows=len(x), query_rows=len(xq),
             expanded_feature_count=x.shape[1], replay_max_abs_error=replay_error,
             fit_seconds=round(time.perf_counter() - began, 3), **progress)
        del model, replay, matrix, x, xq
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite component forecasts')
    return predictions, predict_seconds, boosters


def component_values(raw):
    return np.clip(raw, 1., 3000.)


def estimate_offset(prior, truth, smoke=False):
    if prior.shape != truth.shape or truth.ndim != 2 or truth.shape[1] != H:
        raise ValueError('Prior predictions and training truth require a shared 72-lead grid')
    if not np.isfinite(truth).all() or not np.isfinite(prior).all():
        raise ValueError('Nonfinite calibration values')
    rows = OFFSET_WINDOW
    if rows > len(truth) and not smoke:
        raise ValueError('Insufficient training calibration rows for frozen window')
    rows = min(rows, len(truth))
    residuals = truth[-rows:] - component_values(prior)[-rows:]
    return ALPHA * residuals.mean(0), rows


def apply_offset(raw, offset):
    corrected = np.clip(component_values(raw) + offset[None, :], 1., 3000.)
    if not np.isfinite(corrected).all():
        raise ValueError('Nonfinite final forecasts')
    return corrected


# 3. checkpoints ----------------------------------------------------------------------------

def save_checkpoint(path, prior, full, calibration):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f'prior_b{i}': b for i, b in enumerate(prior)}
    arrays.update({f'full_b{i}': b for i, b in enumerate(full)})
    arrays['calibration'] = np.frombuffer(json.dumps(calibration, allow_nan=False).encode(), np.uint8)
    temporary = path.with_name(path.stem + '.partial.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as arrays:
        blocks = H // BLOCK
        prior = [arrays[f'prior_b{i}'] for i in range(blocks)]
        full = [arrays[f'full_b{i}'] for i in range(blocks)]
        calibration = json.loads(arrays['calibration'].tobytes().decode())
    return prior, full, calibration


# 4. run ------------------------------------------------------------------------------------

def train_period(table, period, seed, args, logs):
    stream, began = f's{seed}', time.perf_counter()
    train, y = load_grid(table, period, True)
    query, _ = load_grid(table, period, False)
    if len(query['origins']) != QUERY_COUNTS[period]:
        raise ValueError('Unexpected official query grid')
    if not np.isfinite(y).all():
        raise ValueError('Prepared training targets must be finite')
    reference, calibration, boundary = prior_split(train['origins'])
    full = np.arange(len(y))
    if args.smoke:
        reference = reference[np.linspace(0, len(reference) - 1, min(len(reference), 128)).astype(int)]
        calibration = calibration[np.linspace(0, len(calibration) - 1, min(len(calibration), 64)).astype(int)]
        full = np.linspace(0, len(y) - 1, min(len(y), 192)).astype(int)
        query = subset(query, np.arange(min(len(query['origins']), 8)))
    rounds = 2 if args.smoke else ROUNDS
    progress = dict(unit=0, units=2 * (H // BLOCK))
    logs(stream, 'period_start', period=period, seed=seed, smoke=args.smoke,
         training_cutoff=CUTOFFS[period], maturity_hours=PURGE_HOURS,
         train_origins=len(full), train_start=str(train['origins'][full].min()),
         train_end=str(train['origins'][full].max()), query_origins=len(query['origins']),
         query_start=str(query['origins'].min()), query_end=str(query['origins'].max()), **progress)
    prior, _, prior_boosters = fit_predict(subset(train, reference), subset(train, calibration),
                                           y[reference], seed, args.threads, rounds, logs, stream,
                                           period, 'prior', progress)
    truth, cal_origins = y[calibration], train['origins'][calibration].asi8
    state = dict(calibration_rows=len(calibration), reference_rows=len(reference),
                 reference_start=str(train['origins'][reference].min()),
                 reference_end=str(train['origins'][reference].max()),
                 calibration_start=str(train['origins'][calibration].min()),
                 calibration_end=str(train['origins'][calibration].max()),
                 split_boundary=str(boundary), purge_hours=PURGE_HOURS)
    raw, seconds, full_boosters = fit_predict(subset(train, full), query, y[full], seed,
                                              args.threads, rounds, logs, stream, period, 'full', progress)
    offset, rows = estimate_offset(prior, truth, smoke=args.smoke)
    state.update(policy=POLICY, window=OFFSET_WINDOW, alpha=ALPHA, scope='horizon', applied=True,
                 offset=offset.tolist(), offset_rows=rows,
                 offset_start=str(pd.Timestamp(cal_origins[-rows], tz='UTC')),
                 offset_end=str(pd.Timestamp(cal_origins[-1], tz='UTC')))
    checkpoint = args.out / 'weights' / period / f's{seed}.npz'
    save_checkpoint(checkpoint, prior_boosters, full_boosters, state)
    prediction = apply_offset(raw, offset)
    component = component_values(raw)
    logs(stream, 'period_complete', period=period, seed=seed, checkpoint=str(checkpoint),
         frozen_calibration=state, inference_seconds_per_fold=seconds / len(prediction),
         clipping=dict(component_cells=int(np.count_nonzero(raw != component)),
                       final_cells=int(np.count_nonzero(component + offset[None, :] != prediction))),
         end_to_end_seconds=round(time.perf_counter() - began, 3))
    return query['origins'], prediction


def predict_period(table, period, seed, args, logs):
    """Inference only: full boosters and the frozen offset from weights/F{p}/s{seed}.npz."""
    stream, began = f's{seed}', time.perf_counter()
    query, _ = load_grid(table, period, False)
    checkpoint = args.weights / period / f's{seed}.npz'
    _, boosters, state = load_checkpoint(checkpoint)
    raw = np.empty((len(query['origins']), H), np.float64)
    for block, start in enumerate(range(0, H, BLOCK)):
        raw[:, start:start + BLOCK] = load_booster(boosters[block], args.threads).inplace_predict(
            design(query, start, BLOCK))
    prediction = apply_offset(raw, np.asarray(state['offset'], np.float64))
    logs(stream, 'predict_only', period=period, seed=seed, checkpoint=str(checkpoint),
         query_origins=len(prediction), offset=state['offset'],
         seconds=round(time.perf_counter() - began, 3))
    return query['origins'], prediction


def prediction_frame(parts):
    frames = []
    for period, (origins, pred) in parts.items():
        if origins.has_duplicates or not origins.is_monotonic_increasing or not np.isfinite(pred).all():
            raise ValueError(f'{period}: invalid prediction grid')
        frames.append(pd.DataFrame({
            KEYS[0]: np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), H),
            KEYS[1]: np.tile(np.arange(1, H + 1), len(origins)), 'pred_kms': pred.ravel()}))
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(KEYS).any():
        raise ValueError('Duplicate prediction keys')
    return frame


def mean3(frames):
    """Fixed equal-weight mean of the three seed forecasts on identical keys."""
    first = frames[0][KEYS]
    if any(not frame[KEYS].equals(first) for frame in frames):
        raise ValueError('All three seeds must have identical origin/horizon keys')
    matrix = np.stack([frame['pred_kms'].to_numpy(np.float64) for frame in frames])
    output = first.copy()
    output['pred_kms'] = matrix.mean(axis=0)
    return output


def score(frame, observations):
    spec = importlib.util.spec_from_file_location('competition', ROOT / 'leaderboard/scorer/competition.py')
    competition = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(competition)
    observed = competition.load_observations(observations)
    result = competition.score_competition(frame, competition.prepare_context(observed))
    if result['total']['n_replaced']:
        raise ValueError('Scorer replaced forecasts; the forecast grid is incomplete')
    return dict(competition=result, scorer_version=competition.VERSION,
                scorer=file_identity(spec.origin), observations=file_identity(observations))


def observations_path(path):
    if path is not None:
        return path
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download('tlabtlab/sunrun-lb-store', 'competition-observations.parquet',
                                repo_type='dataset'))


def write_results(folder, frame, scores):
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / 'predictions.parquet', index=False)
    if scores is not None:
        (folder / 'score.json').write_text(json.dumps(scores, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', choices=[EXPERIMENT], default=EXPERIMENT)
    parser.add_argument('--data', type=Path, default=ROOT / 'data-v2/input/df.parquet')
    parser.add_argument('--observations', type=Path, help='default: the leaderboard dataset copy')
    parser.add_argument('--out', type=Path, default=HERE)
    parser.add_argument('--weights', type=Path, help='checkpoint folder for --predict-only (default: OUT/weights)')
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--periods', nargs='+', choices=list(CUTOFFS), default=list(CUTOFFS))
    parser.add_argument('--predict-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('GBM runs are CPU only: start with CUDA_VISIBLE_DEVICES=""')
    if args.threads < 1:
        raise ValueError('threads must be positive')
    args.out = args.out.resolve()
    if args.smoke and (args.out == HERE or args.predict_only):
        raise ValueError('--smoke writes a throwaway run: give a separate --out and train')
    args.weights = (args.weights or args.out / 'weights').resolve()
    logs = Logs(args.out / 'logs', datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=False)
    logs('run', 'launch', mode='predict_only' if args.predict_only else ('smoke' if args.smoke else 'train'),
         host=socket.gethostname(), argv=vars(args), git_commit=git.stdout.strip(),
         versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                       xgboost=xgb.__version__),
         constant_features=CONSTANT, varying_features=VARYING,
         feature_count=len(CONSTANT) + len(VARYING), parameters=PARAMS, rounds=ROUNDS, block=BLOCK,
         tail=dict(threshold=TAIL_THRESHOLD, weight=TAIL_WEIGHT), spectral_coefficient=SPECTRAL_COEFFICIENT,
         alpha=ALPHA, offset_window=OFFSET_WINDOW, cutoffs=CUTOFFS, seeds=args.seeds)
    table = load_table(args.data, logs)
    observations = None if args.smoke else observations_path(args.observations)
    frames = {}
    for seed in args.seeds:
        parts = {}
        for period in args.periods:
            run = predict_period if args.predict_only else train_period
            parts[period] = run(table, period, seed, args, logs)
        frames[seed] = prediction_frame(parts)
        complete = not args.smoke and args.periods == list(CUTOFFS)
        scores = score(frames[seed], observations) if complete else None
        write_results(args.out / 'results' / f's{seed}', frames[seed], scores)
        logs(f's{seed}', 'seed_complete', seed=seed, rows=len(frames[seed]),
             total_s=scores['competition']['total']['s'] if scores else None)
    if sorted(frames) == list(SEEDS):
        ensemble = mean3([frames[seed] for seed in SEEDS])
        scores = score(ensemble, observations) if observations is not None and args.periods == list(CUTOFFS) else None
        write_results(args.out / 'results' / 'mean3', ensemble, scores)
        logs('mean3', 'mean3_complete', seeds=list(SEEDS), weights=[1 / 3] * 3, rows=len(ensemble),
             total_s=scores['competition']['total']['s'] if scores else None,
             periods={p: scores['competition'][p.lower()]['s'] for p in CUTOFFS} if scores else None)
    logs('run', 'complete', status='complete')


if __name__ == '__main__':
    main()
