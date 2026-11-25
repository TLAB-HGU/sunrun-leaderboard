"""fianl_v1 — plus7 with unused lead columns and two block-specific inputs removed, no ridge, no uplift; mean of
seeds 42/43/44.

End to end in this file: data-v2/input/df.parquet -> causal feature fill -> (origin x 72-lead) grids -> three
24-lead XGBoost blocks per model -> training-only frozen offset -> seam / phase post-processing -> mean of three seeds
-> official scorer.

Per seed (42, 43, 44) and period (F1, F2, F3; training origins + 144 h <= cutoff):
  inputs   38 origin constants and 13 lead-varying features of df.parquet (enlil_earth_speed from NOAA open-data S3
           runs only). Block designs keep 305 / 308 / 283 columns: the 96 lead columns unused by every split of the
           first fianl_v1 models, enlil_earth_speed in h1-24 and rec54_consensus in h49-72 are left out.
  model    per 24-lead block: 600 multi-output trees on a smooth-tail weighted squared error (A stage), then 6 stages
           x 30 direct-S proposal trees (S = sum_h (MSE_h * TIMSE_h)^(1/4) on the training rows), each stage
           accepted with an exact Armijo line search that keeps every lead's training MSE <= 1.01 x the A stage's.
  offset   a prior model on the origins up to 144 h before the last three training months; 0.25 x its mean
           residual over the last 336 calibration origins is added to the full model's forecast; clip [1, 3000].
  post     seam xf8 at h24/25 and h48/49 -> 3 h same-issue phase advance in h25-71 -> fixed equal-weight mean of
           the three seeds (the seed-agreement uplift of the parent rows is not applied).

CPU only:
  CUDA_VISIBLE_DEVICES='' python experiment.py --threads 6             # train all seeds and periods, score
  CUDA_VISIBLE_DEVICES='' python experiment.py --predict-only          # inference from weights/F{n}/s{seed}.npz
  CUDA_VISIBLE_DEVICES='' python experiment.py --smoke --out DIR       # 2-round wiring check, not scored
Without --out: weights/ in this folder, results/ and logs/ in research/result/fianl_v1/. With --out DIR: DIR/weights,
DIR/results, DIR/logs (and DIR/F{n}_metadata.json for the sunrun monitor).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['OMP_NUM_THREADS'] = sys.argv[sys.argv.index('--threads') + 1] if '--threads' in sys.argv[:-1] else '6'

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from scipy.special import expit  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = next(path for path in HERE.parents if (path / 'leaderboard' / 'scorer').is_dir())
EXPERIMENT = 'fianl_v1'
PARENT = 'e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam_union9_lbsel_plus7'
SEEDS = (42, 43, 44)
H = 72
CUTOFFS = {'F1': '2025-12-29', 'F2': '2026-03-29', 'F3': '2026-06-29'}
QUERY_COUNTS = {'F1': 2089, 'F2': 2113, 'F3': 2113}
METADATA = ('set', 'origin', 'h', 'target_time', 'period', 'fit_F1', 'fit_F2', 'fit_F3', 'y')
KEYS = ['origin_last_input_utc', 'horizon_hours']
MISSING = -999.  # "not available" marker of the event and union columns

# Design order of one block: BASE_CONSTANT, the 8 BASE_VARYING per lead, LAG_CONSTANT, DONKI, UNION_GLOBAL,
# UNION_LATE (h49-72 only), then per lead the 2 PLUS4_VARYING, SUVI_LATE (h49-72 only), per lead the 3 PLUS7_VARYING.
BASE_CONSTANT = ['v_lag0', 'ace_speed_kms_last', 'ace_density_cm3_last', 'ace_density_cm3_m6',
                 'ace_temperature_k_last', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6',
                 'ace_by_gsm_nt_m24', 'ace_epam_p_channel3_last', 'ace_sis_gt10_flux_last',
                 'suvi_t035_top2_area', 'suvi_t035_meridian15_equatorial_area',
                 'suvi_t035_meridian30_equatorial_area', 'suvi_t055_meridian15_area',
                 'suvi_t055_meridian30_equatorial_area', 'suvi_t055_equatorial_area', 'img_age_h',
                 'enlil_n_pending', 'enlil_kp']
LAG_CONSTANT = ['v_lag24', 'v_lag48', 'v_lag72']
DONKI = ['pub_enlil_next_arr_h', 'pub_enlil_since_arr_h', 'pub_enlil_glancing', 'pub_cme_next_speed_kms',
         'pub_cme_next_half_angle_deg', 'pub_cme_next_lon_deg', 'pub_cme_next_lat_deg', 'pub_cme_last_speed_kms']
UNION_GLOBAL = ['ace_log_proton_entropy', 'ace_nb_cocompression_6h']
UNION_LATE = ['swpc_hss_coverage49_72', 'ace_proton_ram_pressure_npa', 'srs_earthward_eruptive_area']
SUVI_LATE = ['suvi_t055_lowlat_top1_lon_width_cap67_deg', 'suvi_ballistic_dark_contrast']
BASE_VARYING = ['rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed', 'strip_v450_logmed', 'strip_v550_dark',
                'strip_v750_logmed']
PLUS4_VARYING = ['cme_phase_rec27', 'rec27_ramp']
PLUS7_VARYING = ['enlil_earth_speed', 'onset700_hazard', 'rec54_consensus']
CONSTANT = BASE_CONSTANT + LAG_CONSTANT + DONKI + UNION_GLOBAL + UNION_LATE + SUVI_LATE
VARYING = BASE_VARYING + PLUS4_VARYING + PLUS7_VARYING
LATE_START = 48

# Each column below was unused by all first-run fianl_v1 full models (F1-F3, seeds 42-44); continuation stages
# with zero weight do not count. BLOCK_REMOVALS drop a feature for all 24 leads of one block.
UNUSED_LEADS = {
    0: {'strip_v550_dark': (7, 8, 9, 10), 'strip_v750_logmed': (14, 15, 16, 17),
        'strip_v350_logmed': (22, 23, 24), 'strip_v450_logmed': (23, 24), 'onset700_hazard': (4, 5, 6)},
    24: {'strip_v550_dark': (35, 36, 37), 'strip_v750_logmed': (41, 42, 43, 44), 'strip_v350_logmed': (45, 47, 48),
         'cme_phase_rec27': (32, 33, 38, 39, 43, 44, 45), 'onset700_hazard': range(29, 49)},
    48: {'strip_v450_logmed': (53, 54), 'strip_v550_dark': (60, 62, 63, 64), 'strip_v750_logmed': range(68, 73),
         'strip_v350_logmed': (72,), 'cme_phase_rec27': (50, 51, 52, 55, 56, 57, 62),
         'onset700_hazard': range(49, 73)},
}
BLOCK_REMOVALS = {0: ('enlil_earth_speed',), 24: (), 48: ('rec54_consensus',)}
BLOCK_WIDTHS = (305, 308, 283)

PARAMS = {'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7,
          'min_child_weight': 20, 'reg_lambda': 5, 'objective': 'reg:squarederror',
          'tree_method': 'hist', 'device': 'cpu', 'multi_strategy': 'multi_output_tree',
          'eval_metric': 'rmse', 'seed': 42, 'verbosity': 0}
ROUNDS = 600            # A-stage trees per block
BLOCK = 24
PROPOSAL_ROUNDS = 180   # continuation trees per block, in stages of STAGE_ROUNDS
STAGE_ROUNDS = 30
STEP_SIZES = (1., .5, .25, .125, .0625, .03125, .015625)
ARMIJO = 1e-4
MSE_CAP = 1.01
OBJECTIVE = 'smooth_tail_scap_nospec'
ALPHA = .25
OFFSET_WINDOW = 336
PURGE_HOURS = 144
POLICY = 'training-only frozen calibration; official labels used only for scoring'
SEAMS, TAPER = (24, 48), 8
POSTPROCESS = ['seam xf8', 'phase advance 3 h']


class Logs:
    """Append-only JSON-lines logs, one file per stream (run, data, s42, s43, s44, mean3), echoed to stdout."""

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


def git_commit():
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
                          check=False).stdout.strip()


def versions():
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                xgboost=xgb.__version__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


# 1. data -------------------------------------------------------------------------------------------------------------

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
    """Validate repeated inputs, then forward-fill independently along each h timeline (NaN only; -999 is a value)."""
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
    """Metadata plus the 51 input columns of df.parquet, causally filled."""
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
    constant_rows = selected.loc[selected['h'].eq(1).to_numpy(), ['origin', *CONSTANT]].sort_values(
        'origin', kind='stable')
    origins = pd.DatetimeIndex(pd.to_datetime(constant_rows.origin, utc=True))
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
    grid = {'origins': origins, 'C': constant_rows[CONSTANT].to_numpy(np.float32),
            'E': rows[VARYING].to_numpy(np.float32).reshape(len(origins), H, len(VARYING))}
    if np.isinf(grid['C']).any() or np.isinf(grid['E']).any():
        raise ValueError('Infinite feature values are not supported')
    return grid, (rows.y.to_numpy(np.float64).reshape(len(origins), H) if training else None)


def subset(grid, indices):
    return {key: value[indices] for key, value in grid.items()}


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


# 2. block design -----------------------------------------------------------------------------------------------------

def lead_names(names, start, size):
    return [f'{name}_h{lead}' for lead in range(start + 1, start + size + 1) for name in names]


def full_design(grid, start, size):
    """All 345 / 345 / 350 candidate columns of one block, in design order (see full_names)."""
    n, constant, varying = len(grid['C']), grid['C'], grid['E'][:, start:start + size]
    base = len(BASE_CONSTANT)
    union_end = base + len(LAG_CONSTANT) + len(DONKI) + len(UNION_GLOBAL) + len(UNION_LATE)
    x = np.hstack([constant[:, :base], varying[:, :, :len(BASE_VARYING)].reshape(n, size * len(BASE_VARYING)),
                   constant[:, base:union_end]]).astype(np.float32)
    union = x[:, -(len(UNION_GLOBAL) + len(UNION_LATE)):]
    union[union == MISSING] = np.nan
    if start != LATE_START:
        x = np.ascontiguousarray(x[:, :-len(UNION_LATE)])
    first = len(BASE_VARYING)
    plus4 = varying[:, :, first:first + len(PLUS4_VARYING)].reshape(n, -1).copy()
    if start == LATE_START:
        plus4 = np.column_stack((plus4, constant[:, union_end:]))
    plus4[plus4 == MISSING] = np.nan
    x = np.column_stack((x, plus4)).astype(np.float32, copy=False)
    plus7 = varying[:, :, first + len(PLUS4_VARYING):].reshape(n, -1).copy()
    plus7[plus7 == MISSING] = np.nan
    return np.column_stack((x, plus7)).astype(np.float32, copy=False)


def full_names(start, size):
    late = UNION_LATE if start == LATE_START else []
    names = [*BASE_CONSTANT, *lead_names(BASE_VARYING, start, size), *LAG_CONSTANT, *DONKI, *UNION_GLOBAL, *late,
             *lead_names(PLUS4_VARYING, start, size), *(SUVI_LATE if start == LATE_START else []),
             *lead_names(PLUS7_VARYING, start, size)]
    return names


def removed_columns(start):
    return ({f'{name}_h{lead}' for name, leads in UNUSED_LEADS[start].items() for lead in leads}
            | {f'{name}_h{lead}' for name in BLOCK_REMOVALS[start] for lead in range(start + 1, start + 25)})


KEEP = {}
for _start, _width in zip(range(0, H, BLOCK), BLOCK_WIDTHS, strict=True):
    _names = full_names(_start, BLOCK)
    if not removed_columns(_start).issubset(_names):
        raise ValueError(f'Unknown removed columns for block starting at h{_start + 1}')
    KEEP[_start] = [i for i, name in enumerate(_names) if name not in removed_columns(_start)]
    if len(KEEP[_start]) != _width:
        raise ValueError(f'Unexpected input width for block starting at h{_start + 1}')


def design(grid, start, size):
    return np.ascontiguousarray(full_design(grid, start, size)[:, KEEP[start]])


def feature_names(start, size):
    names = full_names(start, size)
    return [names[i] for i in KEEP[start]]


# 3. model ------------------------------------------------------------------------------------------------------------

class SmoothTailMSE:
    """A-stage objective: squared error weighted 1 + 3 sigmoid((y - lead q80) / (0.2 max(std, 10))), weights mean 1
    per lead, gradient and Hessian scaled by the reciprocal global mean weight."""

    def __init__(self, target):
        self.target = np.array(target, dtype=np.float64, copy=True)
        if self.target.ndim != 2 or len(self.target) < 2 or not np.isfinite(self.target).all():
            raise ValueError('Expected finite two-dimensional fitted training targets')
        if np.any(self.target <= 0):
            raise ValueError('Speed targets must be positive')
        self.target.setflags(write=False)
        delta = max(float(self.target.std()), 10.)
        center = np.quantile(self.target, .8, axis=0)
        self.weights = 1. + 3. * expit((self.target - center) / (.2 * delta))
        self.weights /= self.weights.mean(0)
        self.fixed_scale = 1. / float(self.weights.mean())
        if not np.isfinite(self.fixed_scale) or self.fixed_scale <= 0:
            raise ValueError('Invalid initial curvature scale')

    def __call__(self, prediction, matrix):
        prediction = np.asarray(prediction, dtype=np.float64).reshape(self.target.shape)
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite predictions')
        gradient = self.weights * (prediction - self.target)
        gradient, hessian = self.fixed_scale * gradient, self.fixed_scale * self.weights
        if not (np.isfinite(gradient).all() and np.isfinite(hessian).all() and np.all(hessian >= 0)):
            raise FloatingPointError('Invalid exact derivatives')
        with np.errstate(over='raise', invalid='raise'):
            gradient, hessian = gradient.astype(np.float32), hessian.astype(np.float32)
        if not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
            raise FloatingPointError('Nonfinite float32 callback')
        return gradient, hessian


def timse_coefficients(target):
    """Official 100-quantile TIMSE coefficients of one training lead; they sum to one."""
    target = np.asarray(target, dtype=np.float64)
    if target.ndim != 1 or len(target) < 2 or not np.isfinite(target).all():
        raise ValueError('TIMSE requires at least two finite training targets')
    spread = np.ptp(target)
    if spread == 0:
        raise ValueError('Constant training targets make TIMSE undefined')
    quantiles = np.quantile(target, np.arange(101) / 100, method='linear')
    order = np.argsort(target, kind='stable')
    positions = np.searchsorted(target[order], quantiles[:-1], side='left')
    increments = np.zeros(len(target), dtype=np.float64)
    np.add.at(increments, positions, np.diff(quantiles) / spread / (len(target) - positions))
    coefficients = np.empty(len(target), dtype=np.float64)
    coefficients[order] = np.cumsum(increments)
    return coefficients


class TrainingLoss:
    """Fixed-scale training-sample S summed over the block's leads: scale * sum_h (MSE_h * TIMSE_h)^(1/4)."""

    def __init__(self, target):
        self.target = np.array(target, dtype=np.float64, copy=True)
        if self.target.ndim != 2 or min(self.target.shape) < 1 or not np.isfinite(self.target).all():
            raise ValueError('Expected a finite training-target matrix')
        self.weights = np.column_stack([timse_coefficients(t) for t in self.target.T])
        self.initial_prediction = float(np.float32(self.target.mean()))
        initial = np.full_like(self.target, self.initial_prediction)
        self.scale = float(1. / self.coefficient(initial).mean())

    def terms(self, prediction):
        prediction = np.asarray(prediction, dtype=np.float64)
        if prediction.shape != self.target.shape or not np.isfinite(prediction).all():
            raise ValueError('Expected finite predictions matching the training targets')
        error = prediction - self.target
        mse = np.mean(error ** 2, axis=0)
        timse = np.sum(self.weights * error ** 2, axis=0)
        if not (np.isfinite(mse).all() and np.isfinite(timse).all()):
            raise ValueError('Nonfinite objective terms')
        return error, mse, timse, (mse * timse) ** .25

    def coefficient(self, prediction):
        """Per-cell majorisation weight of S: d S_h / d e = coefficient * e."""
        _, mse, timse, score = self.terms(prediction)
        if not ((mse > 0) & (timse > 0)).all():
            raise ValueError('MM coefficient requires positive MSE and TIMSE')
        return .5 * score * (1. / (len(self.target) * mse) + self.weights / timse)

    def value(self, prediction):
        _, mse, timse, score = self.terms(prediction)
        if not ((mse > 0) & (timse > 0)).all():
            raise ValueError('S requires positive MSE and TIMSE')
        return float(self.scale * score.sum())

    def gradient(self, prediction):
        error, _, _, _ = self.terms(prediction)
        return self.scale * self.coefficient(prediction) * error

    def curvature(self, prediction):
        return self.scale * self.coefficient(prediction)

    def summary(self, prediction):
        _, mse, timse, score = self.terms(prediction)
        return dict(value=self.value(prediction), mean_s=float(score.mean()),
                    mse_by_lead=mse.tolist(), timse_by_lead=timse.tolist())


def feasible_upper(error, direction, budget):
    """Largest alpha in [0, 1] with mean_rows (error + alpha direction)^2 <= budget for every lead
    (exact quadratic roots; a current point above the budget beyond roundoff is an error)."""
    a = np.mean(direction ** 2, axis=0)
    b = 2 * np.mean(error * direction, axis=0)
    c = np.mean(error ** 2, axis=0) - budget
    if not all(np.isfinite(v).all() for v in (a, b, c)):
        raise ValueError('Nonfinite feasibility coefficients')
    if (c > 1e-8 * np.maximum(1., budget)).any():
        raise ValueError('Current predictor violates training MSE budget')
    c = np.minimum(c, 0.)
    upper = 1.
    for aa, bb, cc in zip(a, b, c):
        if aa == 0:
            bound = -cc / bb if bb > 0 else np.inf
        else:
            root = np.hypot(bb, 2 * np.sqrt(aa) * np.sqrt(-cc))
            # Stable positive root avoids cancellation when b is positive.
            bound = -2 * cc / (bb + root) if bb > 0 else (-bb + root) / (2 * aa)
        upper = min(upper, float(bound))
    return max(0., upper)


def choose_step(loss, current, direction, budget):
    """Exact Armijo line search on the training S over min(step, feasible upper); every accepted step keeps each
    lead's training MSE within the budget."""
    current, direction = np.asarray(current), np.asarray(direction)
    before = loss.value(current)
    if direction.shape != current.shape or not np.isfinite(direction).all():
        raise ValueError('Invalid or nonfinite proposal direction')
    upper = feasible_upper(current - loss.target, direction, budget)
    derivative = float(np.sum(loss.gradient(current) * direction))
    if not np.isfinite(derivative):
        raise ValueError('Nonfinite directional derivative')
    if derivative < 0:
        for alpha in dict.fromkeys(min(step, upper) for step in STEP_SIZES):
            if alpha <= 0:
                continue
            candidate = current + alpha * direction
            after = loss.value(candidate)
            feasible = bool((np.mean((candidate - loss.target) ** 2, axis=0)
                             <= budget + 1e-8 * np.maximum(1., budget)).all())
            if np.isfinite(after) and after < before and after <= before + ARMIJO * alpha * derivative and feasible:
                return alpha, dict(before=before, after=after, reason='accepted', feasible_upper=upper)
    reason = ('no_descent_direction' if derivative >= 0 else
              'no_positive_feasible_step' if upper == 0 else 'line_search_rejected')
    return 0., dict(before=before, after=before, reason=reason, feasible_upper=upper)


def load_booster(buffer, threads):
    booster = xgb.Booster(params={'nthread': threads, 'device': 'cpu'})
    booster.load_model(bytearray(np.asarray(buffer, np.uint8).tobytes()))
    return booster


def continue_block(reference, x, xq, y, seed, threads, proposal_rounds):
    """Continue the A-stage booster on its training rows. Returns the query forecast (A stage + sum of accepted
    alpha x stage forecast), the stage boosters' UBJSON bytes and the stage records."""
    loss = TrainingLoss(y)
    current = reference.inplace_predict(x).astype(np.float64).reshape(y.shape)
    query = reference.inplace_predict(xq).astype(np.float64).reshape(len(xq), y.shape[1])
    budget = MSE_CAP * np.mean((current - y) ** 2, axis=0)
    settings = dict(PARAMS, device='cpu', nthread=threads, num_target=y.shape[1], base_score=0.)
    matrix = xgb.DMatrix(x, label=y.astype(np.float32), nthread=threads)
    initial = loss.summary(current)
    buffers, stages, used = [], [], 0
    while used < proposal_rounds:
        stage, rounds = len(stages), min(STAGE_ROUNDS, proposal_rounds - used)
        base = current.copy()

        def objective(delta, _matrix, base=base):
            actual = base + np.asarray(delta).reshape(y.shape)
            return loss.gradient(actual).astype(np.float32), loss.curvature(actual).astype(np.float32)

        settings['seed'] = seed + stage
        model = xgb.train(settings, matrix, num_boost_round=rounds, obj=objective)
        direction = model.inplace_predict(x).astype(np.float64).reshape(y.shape)
        alpha, decision = choose_step(loss, current, direction, budget)
        if alpha:
            current += alpha * direction
            query += alpha * model.inplace_predict(xq).astype(np.float64).reshape(len(xq), y.shape[1])
        used += rounds
        buffers.append(np.frombuffer(bytes(model.save_raw('ubj')), np.uint8))
        stages.append(dict(stage=stage, rounds=rounds, seed=settings['seed'], alpha=alpha, **decision,
                           training_mean_s=loss.summary(current)['mean_s'],
                           max_relative_budget_violation=float(np.maximum(
                               (np.mean((current - y) ** 2, axis=0) - budget) / budget, 0.).max())))
    final = loss.summary(current)
    record = dict(fixed_block_scale=loss.scale, initial_prediction=loss.initial_prediction,
                  stages=stages, accepted_stages=sum(s['alpha'] > 0 for s in stages),
                  accepted_proposal_rounds=sum(s['rounds'] for s in stages if s['alpha'] > 0),
                  training_mean_s_initial=initial['mean_s'], training_mean_s_final=final['mean_s'],
                  training_mse_initial=initial['mse_by_lead'], training_mse_final=final['mse_by_lead'])
    return query, buffers, record


def block_forecast(reference, stages, alphas, xq, threads):
    """Inference from saved boosters: A-stage forecast + sum of accepted alpha x stage forecast."""
    prediction = load_booster(reference, threads).inplace_predict(xq).astype(np.float64).reshape(len(xq), -1)
    for buffer, alpha in zip(stages, alphas):
        if alpha:
            prediction += alpha * load_booster(buffer, threads).inplace_predict(xq).astype(
                np.float64).reshape(prediction.shape)
    return prediction


def fit_block(x, xq, y, seed, threads, rounds, smoke):
    """A-stage booster, then the continuation; returns the query forecast, the block artifact
    (A-stage bytes, stage bytes, accepted alphas), inference seconds and log fields."""
    proposal_rounds = 2 if smoke else PROPOSAL_ROUNDS
    matrix = xgb.DMatrix(x, label=y.astype(np.float32), nthread=threads)
    params = dict(PARAMS, seed=seed, device='cpu', nthread=threads, num_target=BLOCK, base_score=float(y.mean()))
    model = xgb.train(params, matrix, num_boost_round=rounds, obj=SmoothTailMSE(y))
    reference = np.frombuffer(bytes(model.save_raw('ubj')), np.uint8)
    del model, matrix
    forecast, stages, record = continue_block(load_booster(reference, threads), x, xq, y.astype(np.float64), seed,
                                              threads, proposal_rounds)
    alphas = [s['alpha'] for s in record['stages']]
    inference = time.perf_counter()
    replay = block_forecast(reference, stages, alphas, xq, threads)
    seconds = time.perf_counter() - inference
    replay_error = float(np.max(np.abs(replay - forecast)))
    if replay_error != 0. or load_booster(reference, threads).num_boosted_rounds() != rounds:
        raise ValueError('Saved boosters did not exactly reproduce their forecasts/rounds')
    return forecast, dict(reference=reference, stages=stages, alphas=alphas), seconds, dict(
        proposal_rounds=proposal_rounds, continuation=record, replay_max_abs_error=replay_error)


def fit_predict(train, query, target, seed, args, rounds, logs, stream, period, stage, progress):
    """Three 24-lead blocks; returns forecasts, inference seconds and the per-block artifacts."""
    predictions = np.empty((len(query['origins']), H), np.float64)
    blocks, predict_seconds = [], 0.
    for block, start in enumerate(range(0, H, BLOCK)):
        began = time.perf_counter()
        x, xq = design(train, start, BLOCK), design(query, start, BLOCK)
        y = target[:, start:start + BLOCK]
        forecast, artifact, seconds, extra = fit_block(x, xq, y, seed, args.threads, rounds, args.smoke)
        predictions[:, start:start + BLOCK] = forecast
        predict_seconds += seconds
        blocks.append(artifact)
        progress['unit'] += 1
        logs(stream, 'fit', period=period, stage=stage, component=OBJECTIVE, block=block,
             horizon=start + BLOCK, rounds=rounds, train_rows=len(x), query_rows=len(xq),
             expanded_feature_count=x.shape[1], **extra,
             fit_seconds=round(time.perf_counter() - began, 3), **progress)
        del x, xq
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite component forecasts')
    return predictions, predict_seconds, blocks


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


# 4. checkpoints ------------------------------------------------------------------------------------------------------

def save_checkpoint(path, stages, calibration):
    """{stage}_b{i} A-stage bytes, {stage}_b{i}_stage{k} continuation bytes, their alphas and the calibration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays, alphas = {}, {}
    for stage, blocks in stages.items():
        for i, block in enumerate(blocks):
            arrays[f'{stage}_b{i}'] = block['reference']
            for k, buffer in enumerate(block['stages']):
                arrays[f'{stage}_b{i}_stage{k}'] = buffer
            alphas[f'{stage}_b{i}'] = block['alphas']
    arrays['alphas'] = np.frombuffer(json.dumps(alphas, allow_nan=False).encode(), np.uint8)
    arrays['calibration'] = np.frombuffer(json.dumps(calibration, allow_nan=False).encode(), np.uint8)
    temporary = path.with_name(path.stem + '.partial.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_checkpoint(path):
    """Full-model block artifacts and the frozen calibration of one checkpoint."""
    with np.load(path, allow_pickle=False) as arrays:
        alphas = json.loads(arrays['alphas'].tobytes().decode())
        artifacts = []
        for i in range(H // BLOCK):
            name = f'full_b{i}'
            artifacts.append(dict(reference=arrays[name], alphas=alphas[name],
                                  stages=[arrays[f'{name}_stage{k}'] for k in range(len(alphas[name]))]))
        state = json.loads(arrays['calibration'].tobytes().decode())
    return artifacts, state


# 5. run --------------------------------------------------------------------------------------------------------------

def period_metadata(period, seed, args, rounds, train, full, query, seconds, rows, calibration):
    return dict(experiment=EXPERIMENT, period=period, seed=seed, smoke=args.smoke, rounds=rounds, block=BLOCK,
                objective=OBJECTIVE, parameters=dict(PARAMS, seed=seed),
                feature_count=len(CONSTANT) + len(VARYING), constant_features=CONSTANT, varying_features=VARYING,
                expanded_feature_count=len(CONSTANT) + BLOCK * len(VARYING),
                expanded_features_by_block={str(block): feature_names(start, BLOCK)
                                            for block, start in enumerate(range(0, H, BLOCK))},
                training_cutoff=CUTOFFS[period], maturity_hours=PURGE_HOURS,
                train_origins=len(full), train_start=str(train['origins'][full].min()),
                train_end=str(train['origins'][full].max()), query_origins=len(query['origins']),
                query_start=str(query['origins'].min()), query_end=str(query['origins'].max()),
                inference_seconds_per_fold=seconds / rows, versions=versions(), input_file=file_identity(args.data),
                policy=POLICY, git_commit=git_commit(), frozen_calibration=calibration)


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
    prior, _, prior_blocks = fit_predict(subset(train, reference), subset(train, calibration), y[reference], seed,
                                         args, rounds, logs, stream, period, 'prior', progress)
    truth, cal_origins = y[calibration], train['origins'][calibration].asi8
    state = dict(calibration_rows=len(calibration), reference_rows=len(reference),
                 reference_start=str(train['origins'][reference].min()),
                 reference_end=str(train['origins'][reference].max()),
                 calibration_start=str(train['origins'][calibration].min()),
                 calibration_end=str(train['origins'][calibration].max()),
                 split_boundary=str(boundary), purge_hours=PURGE_HOURS)
    raw, seconds, full_blocks = fit_predict(subset(train, full), query, y[full], seed, args, rounds, logs, stream,
                                            period, 'full', progress)
    checkpoint = args.weights / period / f's{seed}.npz'
    offset, rows = estimate_offset(prior, truth, smoke=args.smoke)
    state.update(policy=POLICY, window=OFFSET_WINDOW, alpha=ALPHA, scope='horizon', applied=True,
                 offset=offset.tolist(), offset_rows=rows,
                 offset_start=str(pd.Timestamp(cal_origins[-rows], tz='UTC')),
                 offset_end=str(pd.Timestamp(cal_origins[-1], tz='UTC')))
    save_checkpoint(checkpoint, dict(prior=prior_blocks, full=full_blocks), state)
    prediction = apply_offset(raw, offset)
    component = component_values(raw)
    clipping = dict(component_cells=int(np.count_nonzero(raw != component)),
                    final_cells=int(np.count_nonzero(component + offset[None, :] != prediction)))
    metadata = period_metadata(period, seed, args, rounds, train, full, query, seconds, len(prediction), state)
    write_json(args.out / 'logs' / f'{period}_s{seed}_metadata.json', metadata)
    if not args.board:
        write_json(args.out / f'{period}_metadata.json', metadata)  # period completion for the sunrun monitor
    logs(stream, 'period_complete', period=period, seed=seed, checkpoint=str(checkpoint),
         frozen_calibration=state, inference_seconds_per_fold=seconds / len(prediction), clipping=clipping,
         end_to_end_seconds=round(time.perf_counter() - began, 3))
    return query['origins'], prediction


def predict_period(table, period, seed, args, logs):
    """Inference only: full-model boosters and the frozen offset from weights/F{n}/s{seed}.npz."""
    stream, began = f's{seed}', time.perf_counter()
    query, _ = load_grid(table, period, False)
    checkpoint = args.weights / period / f's{seed}.npz'
    blocks, state = load_checkpoint(checkpoint)
    raw = np.empty((len(query['origins']), H), np.float64)
    inference_seconds = 0.
    for block, start in enumerate(range(0, H, BLOCK)):
        features = design(query, start, BLOCK)
        tick = time.perf_counter()
        artifact = blocks[block]
        raw[:, start:start + BLOCK] = block_forecast(artifact['reference'], artifact['stages'], artifact['alphas'],
                                                     features, args.threads)
        inference_seconds += time.perf_counter() - tick
    prediction = apply_offset(raw, np.asarray(state['offset'], np.float64))
    logs(stream, 'predict_only', period=period, seed=seed, checkpoint=str(checkpoint),
         query_origins=len(prediction), offset=state['offset'],
         inference_seconds_per_fold=inference_seconds / len(prediction),
         seconds=round(time.perf_counter() - began, 3))
    return query['origins'], prediction


# 6. post-processing --------------------------------------------------------------------------------------------------

def seam_smooth(pred, width=TAPER):
    """xf8 seam correction at the block joins h24/25 and h48/49, computed on the original curve; clip [1, 3000]."""
    p = np.asarray(pred, np.float64)
    q = p.copy()
    for b in SEAMS:
        slope = .5 * ((p[:, b - 1] - p[:, b - 2]) + (p[:, b + 1] - p[:, b]))
        jump = (p[:, b] - p[:, b - 1]) - slope
        for i in range(width):
            taper = .5 * jump * (width - i) / width
            q[:, b - 1 - i] += taper
            q[:, b + i] -= taper
    return np.clip(q, 1., 3000.)


def phase_advance(forecast):
    """Move h25-71 up to 3 h towards the same issue's forecast 6 h later (gated in and out over 6 leads)."""
    h = np.arange(1, H + 1)
    ahead = np.minimum(h + 6, H)
    gate = np.clip(np.minimum((h - 24) / 6, (72 - h) / 6), 0, 1)
    weight = np.where((h >= 25) & (h <= 71), 3 * gate / np.maximum(ahead - h, 1), 0.)
    return np.clip(forecast + weight[None, :] * (forecast[:, ahead - 1] - forecast), 1., 3000.)


def postprocess(raw):
    """Per seed: seam xf8 -> phase advance."""
    return [phase_advance(seam_smooth(profile)) for profile in raw]


# 7. results ----------------------------------------------------------------------------------------------------------

def forecast_frame(origins, pred):
    return pd.DataFrame({KEYS[0]: np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), H),
                         KEYS[1]: np.tile(np.arange(1, H + 1), len(origins)), 'pred_kms': pred.ravel()})


def prediction_frame(parts):
    frames = []
    for period, (origins, pred) in parts.items():
        if origins.has_duplicates or not origins.is_monotonic_increasing or not np.isfinite(pred).all():
            raise ValueError(f'{period}: invalid prediction grid')
        frames.append(forecast_frame(origins, pred))
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


def write_results(folder, tag, frame, scores):
    """results/{tag}_predictions.parquet and, when scored, results/{tag}_score.json."""
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / f'{tag}_predictions.parquet', index=False)
    if scores is not None:
        write_json(folder / f'{tag}_score.json', scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', choices=[EXPERIMENT], default=EXPERIMENT)
    parser.add_argument('--data', type=Path, default=ROOT / 'data-v2/input/df.parquet')
    parser.add_argument('--observations', type=Path, help='default: the leaderboard dataset copy')
    parser.add_argument('--out', type=Path, help='run folder (default: research/result/fianl_v1)')
    parser.add_argument('--weights', type=Path,
                        help='checkpoint folder (default: weights/ in this folder, or OUT/weights with --out)')
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--periods', nargs='+', choices=list(CUTOFFS), default=list(CUTOFFS))
    parser.add_argument('--predict-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('GBM runs are CPU only: start with CUDA_VISIBLE_DEVICES=""')
    if args.threads < 1:
        raise ValueError('threads must be positive')
    args.board = args.out is None
    args.out = (ROOT / 'research/result' / EXPERIMENT if args.board else args.out).resolve()
    args.weights = (args.weights or (HERE / 'weights' if args.board else args.out / 'weights')).resolve()
    if args.smoke and (args.board or args.predict_only):
        raise ValueError('--smoke writes a throwaway run: give a separate --out and train')
    logs = Logs(args.out / 'logs', datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    logs('run', 'launch', mode='predict_only' if args.predict_only else ('smoke' if args.smoke else 'train'),
         host=socket.gethostname(), argv=vars(args), git_commit=git_commit(), versions=versions(),
         constant_features=CONSTANT, varying_features=VARYING, feature_count=len(CONSTANT) + len(VARYING),
         parameters=PARAMS, rounds=ROUNDS, block=BLOCK, objective=OBJECTIVE, parent_experiment=PARENT,
         continuation=dict(proposal_rounds=PROPOSAL_ROUNDS, stage_rounds=STAGE_ROUNDS, step_sizes=STEP_SIZES,
                           armijo=ARMIJO, mse_cap=MSE_CAP),
         removed_features={start: sorted(removed_columns(start)) for start in KEEP},
         block_widths=list(BLOCK_WIDTHS), postprocess=POSTPROCESS, alpha=ALPHA, offset_window=OFFSET_WINDOW,
         cutoffs=CUTOFFS, seeds=args.seeds)
    table = load_table(args.data, logs)
    frames = {}
    for seed in args.seeds:
        parts = {}
        for period in args.periods:
            run = predict_period if args.predict_only else train_period
            parts[period] = run(table, period, seed, args, logs)
        frames[seed] = prediction_frame(parts)
    results = args.out / 'results'
    if args.smoke or sorted(frames) != list(SEEDS) or args.periods != list(CUTOFFS):
        # Partial runs keep the unscored seed forecasts before post-processing (it needs all three seeds).
        for seed, frame in frames.items():
            write_results(results, f's{seed}', frame, None)
            logs(f's{seed}', 'seed_complete', seed=seed, rows=len(frame), total_s=None)
        if sorted(frames) == list(SEEDS):
            write_results(results, 'mean3', mean3([frames[seed] for seed in SEEDS]), None)
        logs('run', 'complete', status='complete')
        return
    observations = observations_path(args.observations)
    final = postprocess([frames[seed].pred_kms.to_numpy(np.float64).reshape(-1, H) for seed in SEEDS])
    outputs = []
    for seed, matrix in zip(SEEDS, final, strict=True):
        frame = frames[seed][KEYS].copy()
        frame['pred_kms'] = matrix.ravel()
        scores = score(frame, observations)
        write_results(results, f's{seed}', frame, scores)
        logs(f's{seed}', 'seed_complete', seed=seed, rows=len(frame), total_s=scores['competition']['total']['s'])
        outputs.append(frame)
    ensemble = mean3(outputs)
    scores = score(ensemble, observations)
    write_results(results, 'mean3', ensemble, scores)
    logs('mean3', 'mean3_complete', seeds=list(SEEDS), weights=[1 / 3] * 3, rows=len(ensemble),
         total_s=scores['competition']['total']['s'],
         periods={p: scores['competition'][p.lower()]['s'] for p in CUTOFFS})
    logs('run', 'complete', status='complete', postprocess=POSTPROCESS)


if __name__ == '__main__':
    main()
