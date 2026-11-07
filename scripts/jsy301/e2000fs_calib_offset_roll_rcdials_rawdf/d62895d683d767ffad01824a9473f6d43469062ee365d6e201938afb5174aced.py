"""rank-1 e2000fs_calib_offset_roll + regime dials (tail weight, recurrence pull) chosen on the pre-cutoff calibration quarter, raw-only df, lab fill: leaderboard entry script for the 2026-10-31 F1/F2/F3 split on the team raw-only df (self-contained).

Base recipe: rank 1 `e2000fs_calib_offset_roll` (team research/e2000fs_calib_offset_roll/experiment.py). Its functions
design, spectral_objective, prior_split, native_head, estimate_state, apply_state, subset, selected_names and mapped,
and its BASE_NAMES / FEATURES / PARAMS / POLICY / OFFSET_* constants, are copied below unchanged. Per period P:
  * training rows = the df's fit_P flags (origin + 144 h <= cutoff); columns = the 30 feat30 groups in the team order
    (21 constant + 24 x 9 lead-varying = 237 per 24-lead block);
  * inputs: the DirMO lab fill (as our DS entry): causal last value along the hourly origin timeline, solar-wind
    observations without limit, image columns (suvi_*, strip_*, img_age_h) for at most 24 h with img_age_h grown by
    the hours carried, ENLIL columns not filled. The team fills every column without limit; this is the difference;
  * training y: the targets of the fit_P rows, forward-filled along target time (last observed hourly speed; team
    rule); origins whose filled targets still have a gap (start of the record only) are left out. The query rows are
    read without any y column;
  * prior model: 3 blocks x (spectral objective, PARAMS, 900 rounds, seed 42) on the reference rows (origin + 144 h
    <= boundary = last training origin - 3 months), predicting the calibration rows (origins >= boundary);
    offset = 0.25 x mean(filled y - prior) over the last 336 calibration rows per lead (estimate_state);
  * full model: the same recipe on all training rows -> query forecast -> apply_state (offset, clip 1-3000).
Additions of this entry, used only when CONFIG lists them:
  * dials (CONFIG hyperparameters.dials): the tail weight t (weight t instead of 4 on y >= 550 in the spectral
    objective; prior and full model) and a recurrence pull w on leads 25-72, pred <- pred + w (rec27 - pred) after
    the offset (rec27 = the df's lead-varying rec27 input of the same origin after the fill; NaN -> no pull), then
    clip 1-3000. (t*, w*) = argmin over the grid of S (official per-lead formula, all 72 leads) of
    prior_t + offset_t + pull w on the calibration quarter's 00/06/12/18 UTC origins against the raw (unfilled)
    training y, NaN cells excluded; ties -> smaller t, then smaller w. The full model is trained with t* and the
    query forecast gets offset_{t*} and pull w*. Only pre-cutoff training rows are read for the choice;
  * no_image_block0: block 1 (leads 1-24) of the prior and the full model without the suvi_*, strip_* and img_age_h
    columns; blocks 2-3 unchanged.

    python <this file> --df <team df.parquet | directory with train.parquet + query_inputs.parquet> \
        --out predictions.parquet [--work <artifact dir>] [--smoke | --smoke-rounds N]
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

CONFIG = {'data': {'base': 'e2000fs_calib_offset_roll', 'source': 'team data-v2/input/df.parquet (raw-only rebuild 2026-11-02)', 'df_sha256': '33e451e5718722dcd8737c1d0ef56a68f3707df55a3e7af39885efaccfb9f27c', 'split_files_sha256': {'train.parquet': '880c6fa6cfb146e3585c96430b8eb622f4fc6b60ccddee724912f97a63759254', 'query_inputs.parquet': '8f6e3c3c9b729b250049e34aed5e18b54997610a7bd6c38950a474061b5ce763'}, 'split': '2026-10-31 F1/F2/F3', 'training_rows': 'df fit_F1/fit_F2/fit_F3 (origin + 144 h <= cutoff), hourly origins; origins whose filled targets still have a gap (start of the record) are left out', 'gap_fill': 'inputs causal last value along the hourly origin timeline (observations unlimited, images <= 24 h with img_age_h grown by the hours carried, ENLIL none) - the DirMO lab fill as in our DS entry; the team fills every column without limit', 'training_y': "targets of the fit rows forward-filled along target time (last observed hourly speed, only the fit rows' own targets; team rule); query rows are read without y", 'time_convention': 'newest input of origin O is the hour labelled O-1h; lead h target is hour O+h'}, 'hyperparameters': {'strategy': 'rank-1 e2000fs_calib_offset_roll: DirMO-24 feat30, 3 blocks of 24 leads, multi_output_tree', 'constant_columns': ['v_lag0', 'ace_speed_kms_last', 'ace_density_cm3_last', 'ace_density_cm3_m6', 'ace_temperature_k_last', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6', 'ace_by_gsm_nt_m24', 'ace_epam_p_channel3_last', 'ace_sis_gt10_flux_last', 'suvi_t035_top2_area', 'suvi_t035_meridian15_equatorial_area', 'suvi_t035_meridian30_equatorial_area', 'suvi_t035_equatorial_area', 'suvi_t055_meridian15_area', 'suvi_t055_meridian30_equatorial_area', 'suvi_t055_equatorial_area', 'img_age_h', 'enlil_n_pending', 'enlil_kp'], 'varying_columns': ['rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed', 'strip_v450_logmed', 'strip_v550_dark', 'strip_v650_dark', 'strip_v750_logmed'], 'block_input_columns': 237, 'block0_input_columns': 237, 'params': {'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 20, 'reg_lambda': 5, 'objective': 'reg:squarederror', 'tree_method': 'hist', 'device': 'cpu', 'multi_strategy': 'multi_output_tree', 'eval_metric': 'rmse', 'seed': 42, 'verbosity': 0}, 'rounds': 900, 'seed': 42, 'seeds': [42], 'threads': 6, 'base_score': 'block mean of the filled training targets (team native_head)', 'loss': 'team spectral_objective: weight t on y >= 550 else 1 (normalised to mean 1 per lead), + 0.1 x pseudo-Huber (scale 50) term on the FFT of the 24-lead error; hessian weights + 0.1; t = 4 is the team function', 'prior_model': 'same recipe on reference rows origin + 144 h <= boundary (boundary = last training origin - 3 months), predicting the calibration rows (origins >= boundary)', 'offset': 'team estimate_state: 0.25 x mean(filled y - prior) over the last 336 calibration rows per lead; team apply_state: forecast + offset, clip 1-3000', 'dials': {'tail_weights': [2, 4, 6], 'pull_weights': [0.0, 0.1, 0.2, 0.3], 'pull': "leads 25-72: pred + w (rec27 - pred) after the offset, rec27 = the df's lead-varying rec27 input of the same origin after the fill, NaN -> no pull; then clip 1-3000", 'selection': '(t*, w*) = argmin over the grid of S (official per-lead formula: mean over leads 1-72 of sqrt(RMSE_h sqrt(TIMSE_h)), 100 linear percentile intervals) of apply_state(prior_t, offset_t) then the pull w on leads 25-72, on the calibration rows with origin hour in {0, 6, 12, 18} UTC, truth = the raw (unfilled) training y of those rows, NaN cells excluded; ties -> smaller t, then smaller w. The full model is trained with t*, the query forecast gets offset_{t*} and pull w*. Only pre-cutoff training rows are read; no placebo or control arm is part of the entry'}, 'no_image_block0': False, 'no_image_columns': {}, 'clip': [1.0, 3000.0], 'placebo_free': True, 'seed_note': 'seed 42 only (PARAMS seed of rank 1)'}}
CONFIG_SHA256 = "d62895d683d767ffad01824a9473f6d43469062ee365d6e201938afb5174aced"
MODEL = "r1_rc_rawdf"
IMAGE_CAP_H = 24
PERIODS = {"F1": 2089, "F2": 2113, "F3": 2113}  # name -> query origins
CUTOFFS = {'F1': '2025-12-29', 'F2': '2026-03-29', 'F3': '2026-06-29'}

# ---------------------------------------------------------------- team rank-1 code (copied unchanged) ----------
H = 72

BASE_NAMES = {'constant': ['v_lag0', 'v_lag1', 'v_lag3', 'v_lag6', 'v_lag12', 'v_lag24', 'v_lag48', 'v_lag72', 'v_lag120', 'v_m6', 'v_sd6', 'v_m24', 'v_sd24', 'v_m72', 'v_sd72', 'ace_speed_kms_last', 'ace_speed_kms_m6', 'ace_speed_kms_m24', 'ace_density_cm3_last', 'ace_density_cm3_m6', 'ace_density_cm3_m24', 'ace_temperature_k_last', 'ace_temperature_k_m6', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6', 'ace_bt_nt_m24', 'ace_bz_gsm_nt_last', 'ace_bz_gsm_nt_m6', 'ace_bz_gsm_nt_m24', 'ace_by_gsm_nt_last', 'ace_by_gsm_nt_m6', 'ace_by_gsm_nt_m24', 'ace_epam_e_channel1_last', 'ace_epam_e_channel1_m6', 'ace_epam_e_channel1_m24', 'ace_epam_e_channel2_last', 'ace_epam_e_channel2_m6', 'ace_epam_e_channel2_m24', 'ace_epam_p_channel1_last', 'ace_epam_p_channel1_m6', 'ace_epam_p_channel1_m24', 'ace_epam_p_channel3_last', 'ace_epam_p_channel3_m6', 'ace_epam_p_channel3_m24', 'ace_epam_p_channel5_last', 'ace_epam_p_channel5_m6', 'ace_epam_p_channel5_m24', 'ace_sis_gt10_flux_last', 'ace_sis_gt10_flux_m6', 'ace_sis_gt10_flux_m24', 'speed_d24', 'epam_p1_d6', 'ch_ch_t035_area', 'ch_ch_t035_max_area', 'ch_ch_t035_top1_area', 'ch_ch_t035_top2_area', 'ch_ch_t035_top3_area', 'ch_ch_t035_meridian15_area', 'ch_ch_t035_meridian15_equatorial_area', 'ch_ch_t035_meridian30_area', 'ch_ch_t035_meridian30_equatorial_area', 'ch_ch_t035_equatorial_area', 'ch_ch_t045_area', 'ch_ch_t045_max_area', 'ch_ch_t045_top1_area', 'ch_ch_t045_top2_area', 'ch_ch_t045_top3_area', 'ch_ch_t045_meridian15_area', 'ch_ch_t045_meridian15_equatorial_area', 'ch_ch_t045_meridian30_area', 'ch_ch_t045_meridian30_equatorial_area', 'ch_ch_t045_equatorial_area', 'ch_ch_t055_area', 'ch_ch_t055_max_area', 'ch_ch_t055_top1_area', 'ch_ch_t055_top2_area', 'ch_ch_t055_top3_area', 'ch_ch_t055_meridian15_area', 'ch_ch_t055_meridian15_equatorial_area', 'ch_ch_t055_meridian30_area', 'ch_ch_t055_meridian30_equatorial_area', 'ch_ch_t055_equatorial_area', 'img_age_h', 'img_stale', 'enlil_next_arr_h', 'enlil_since_arr_h', 'enlil_n_pending', 'enlil_kp', 'enlil_glancing'], 'varying': ['rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed', 'strip_v350_dark', 'strip_v350_cover', 'strip_v450_logmed', 'strip_v450_dark', 'strip_v450_cover', 'strip_v550_logmed', 'strip_v550_dark', 'strip_v550_cover', 'strip_v650_logmed', 'strip_v650_dark', 'strip_v650_cover', 'strip_v750_logmed', 'strip_v750_dark', 'strip_v750_cover', 'enlil_dt_target']}

FEATURES = ['v_lag0', 'rec_mean', 'enlil_kp', 'ch_ch_t035_meridian30_equatorial_area', 'ace_bt_nt_last', 'ace_speed_kms_last', 'ch_ch_t035_equatorial_area', 'ch_ch_t035_meridian15_equatorial_area', 'ace_bt_nt_m6', 'rec26', 'strip_v450_logmed', 'strip_v350_logmed', 'ch_ch_t055_equatorial_area', 'rec27', 'ace_density_cm3_last', 'ch_ch_t055_meridian30_equatorial_area', 'ace_by_gsm_nt_m24', 'ace_density_cm3_m6', 'ace_temperature_k_last', 'enlil_n_pending', 'ace_temperature_k_m24', 'strip_v750_logmed', 'ch_ch_t055_meridian15_area', 'ch_ch_t035_top2_area', 'strip_v550_dark', 'strip_v650_dark', 'ace_sis_gt10_flux_last', 'ace_epam_p_channel3_last', 'img_age_h', 'rec28']

PARAMS = {'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 20, 'reg_lambda': 5, 'objective': 'reg:squarederror', 'tree_method': 'hist', 'device': 'cpu', 'multi_strategy': 'multi_output_tree', 'eval_metric': 'rmse', 'seed': 42, 'verbosity': 0}

POLICY = 'training-only calibration; original eval-stream adaptation removed'

# Frozen original submitted settings for the offset: alpha=.25 over the last 336 calibration origins.
OFFSET_ALPHA = .25
OFFSET_WINDOW = 336


def selected_names(features):
    return {key: [n for n in values if n in features] for key, values in BASE_NAMES.items()}

def mapped(name):
    return name.replace('ch_ch_', 'suvi_', 1) if name.startswith('ch_ch_') else name

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


def native_head(x,y,rounds,threads,out,label,objective=None):
    if not np.isfinite(y).all():
        raise ValueError('Component labels must be finite after upstream training-y ffill')
    matrix=xgb.DMatrix(x,label=y.astype(np.float32),nthread=threads)
    model=xgb.train(dict(PARAMS,nthread=threads,num_target=y.shape[1],base_score=float(y.mean())),matrix,
                    num_boost_round=rounds,obj=objective)
    model.save_model(out/f'{label}.ubj')
    return model


def estimate_state(prediction,y):
    if not np.isfinite(y).all():raise ValueError('Calibration truth must contain finite training labels only')
    if prediction.shape!=y.shape or not np.isfinite(prediction).all():
        raise ValueError('Calibration component grids disagree')
    # Every residual here is matured before the period cutoff; state never updates.
    offset=OFFSET_ALPHA*(y[-OFFSET_WINDOW:]-prediction[-OFFSET_WINDOW:]).mean(0)
    return dict(policy=POLICY,components=['spectral'],calibration_rows=len(y),window=OFFSET_WINDOW,alpha=OFFSET_ALPHA,
                settings_source='Original submitted fixed alpha/window; residual state re-estimated on prior training only',
                offset=offset.tolist())


def apply_state(prediction,state):
    # This interface has no truth argument: only frozen state and new forecasts.
    if state['policy']!=POLICY:raise ValueError('Unexpected calibration policy')
    result=prediction+np.asarray(state['offset'])[None,:]
    if not np.isfinite(result).all():raise ValueError('Nonfinite frozen pool output')
    return np.clip(result,1.,3000.)


def subset(grid,indices):
    return {k:v[indices] for k,v in grid.items()}

# ------------------------------------------------------------------- end of the copied team code ----------------


def spectral_objective_t(target, t):
    """spectral_objective with weight t (instead of 4) on y >= 550; t = 4 equals the team function bit for bit."""
    observed = np.isfinite(target)
    raw = np.where(target >= 550.0, float(t), 1.0) * observed
    weights = raw / (raw.sum(0) / observed.sum(0))

    def objective(prediction, matrix):
        prediction = np.asarray(prediction).reshape(target.shape)
        error = np.where(observed, prediction - np.nan_to_num(target), 0.0)
        z = np.fft.fft(error, axis=1, norm='ortho')
        rho = 1 / np.sqrt(1 + abs(z) ** 2 / 50.0 ** 2)
        gradient = (error * weights + 0.1 * np.fft.ifft(z * rho, axis=1, norm='ortho').real) * observed
        return (gradient.astype(np.float32), ((weights + 0.1) * observed).astype(np.float32))
    return objective


def objective_for(target, t):
    return spectral_objective(target) if t == 4 else spectral_objective_t(target, t)


# ---------------------------------------------- official per-lead S (DirMO lab lab/score_mw.py, copied) ----------
def lead_parts(preds, y):
    """For each prediction (n, 72): per-lead (rmse, timse) arrays, scorer formulas, NaN truth excluded."""
    out = [(np.empty(H), np.empty(H)) for _ in preds]
    for t in range(H):
        v = np.isfinite(y[:, t])
        obs = y[v, t]
        q = np.quantile(obs, np.arange(101) / 100, method='linear')
        order = np.argsort(obs, kind='stable')
        pos = np.searchsorted(obs[order], q[:-1], side='left')
        w, rng = np.diff(q), q[-1] - q[0]
        for k, p in enumerate(preds):
            e2 = (p[v, t] - obs) ** 2
            suf = np.cumsum(e2[order][::-1])[::-1]
            out[k][0][t] = np.sqrt(e2.mean())
            out[k][1][t] = np.dot(w, suf[pos] / (len(obs) - pos)) / rng
    return out


def s_of(parts):
    rmse, timse = parts
    return float(np.mean(np.sqrt(rmse * np.sqrt(timse))))


# --------------------------- official metric (proswin scripts/official_scoring/swp_scoring/core/metric.py, copied) --
def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def timse_percentile(
    observed: np.ndarray,
    predicted: np.ndarray,
    intervals: int = 100,
    method: str = "linear",
) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.shape != predicted.shape or observed.ndim != 1:
        raise ValueError(
            f"expected two equal-length 1-D arrays, got {observed.shape} and {predicted.shape}"
        )
    n = observed.size
    if n == 0:
        raise ValueError("timse_percentile() requires at least one pair")

    span = observed.max() - observed.min()
    if span == 0.0:
        return 0.0

    q = np.quantile(
        observed, np.arange(intervals + 1) / intervals, method=method  # type: ignore[call-overload]
    )
    widths = np.diff(q)

    order = np.argsort(observed, kind="stable")
    v_sorted = observed[order]
    sq_err_sorted = ((observed - predicted) ** 2)[order]
    suffix_sum = np.concatenate([np.cumsum(sq_err_sorted[::-1])[::-1], [0.0]])

    first = np.searchsorted(v_sorted, q[:-1], side="left")
    n_k = n - first  # >= 1, since every q_k <= v_max
    m_k = suffix_sum[first] / n_k

    return float(np.sum(np.where(widths > 0, widths * m_k, 0.0)) / span)


def official_s(y, p):
    """Equal-weight mean over leads of sqrt(RMSE_tau * sqrt(TIMSE_tau)), NaN observations excluded."""
    s = []
    for t in range(y.shape[1]):
        v = np.isfinite(y[:, t])
        if v.any():
            s.append(np.sqrt(rmse(y[v, t], p[v, t]) * np.sqrt(timse_percentile(y[v, t], p[v, t]))))
    return float(np.mean(s))


# ------------------------------------------------ data: read, check hashes, lab fill (as our DS entry) ----------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fill_rule(name):
    if name.startswith("enlil_"):
        return "none"
    if name.startswith(("suvi_", "strip_")) or name == "img_age_h":
        return "image"
    return "obs"


def ffill(a, limit=None):
    """Causal fill along axis 0 of a 2-D array; returns (filled, hours carried)."""
    n = a.shape[0]
    last = np.maximum.accumulate(np.where(np.isfinite(a), np.arange(n)[:, None], -1), axis=0)
    carried = np.where(last >= 0, np.arange(n)[:, None] - last, np.inf)
    out = np.take_along_axis(a, np.maximum(last, 0), axis=0)
    keep = (last >= 0) & (carried <= (np.inf if limit is None else limit))
    return np.where(keep, out, np.nan), np.where(keep, carried, np.nan)


def read(df_path, constant, varying):
    """(train rows with y, query rows without y) as DataFrames, after checking the input hash."""
    p = Path(df_path)
    data = CONFIG["data"]
    keys = ["origin", "h"]
    if p.is_dir():
        for f, want in data["split_files_sha256"].items():
            assert sha256(p / f) == want, f"{f} sha256 mismatch"
        tr = pd.read_parquet(p / "train.parquet", columns=keys + ["fit_F1", "fit_F2", "fit_F3", "y"] + constant + varying)
        q = pd.read_parquet(p / "query_inputs.parquet", columns=keys + ["period"] + constant + varying)
    else:
        assert sha256(p) == data["df_sha256"], "df.parquet sha256 mismatch"
        tr = pd.read_parquet(p, columns=keys + ["fit_F1", "fit_F2", "fit_F3", "y"] + constant + varying,
                             filters=[("set", "==", "train")])
        q = pd.read_parquet(p, columns=keys + ["period"] + constant + varying, filters=[("set", "==", "query")])
    assert "y" not in q.columns
    return tr, q


def grid(df, cols, origins):
    df = df.sort_values(["origin", "h"], kind="stable")
    o = pd.DatetimeIndex(df["origin"].to_numpy()[::H])
    assert len(df) == len(o) * H and (df["h"].to_numpy().reshape(-1, H) == np.arange(1, H + 1)).all()
    pos = origins.get_indexer(o)
    assert (pos >= 0).all()
    out = np.full((len(origins), H, len(cols)), np.nan, np.float64)
    out[pos] = df[list(cols)].to_numpy(np.float64).reshape(len(o), H, len(cols))
    return out, pos


def prepare(df_path, constant, varying):
    """Read, check and fill the df; returns the filled inputs of every origin of the hourly timeline."""
    tr, q = read(df_path, constant, varying)
    origins = pd.DatetimeIndex(sorted(set(tr["origin"]) | set(q["origin"])))
    assert (np.diff(origins.asi8) == 3600 * 10 ** 9).all(), "origin timeline must be hourly without gaps"
    cols = constant + varying
    Xt, pt = grid(tr, cols + ["y"], origins)
    Xq, pq_ = grid(q, cols, origins)
    both = np.intersect1d(pt, pq_)
    same = (Xt[both, :, :-1] == Xq[both]) | (np.isnan(Xt[both, :, :-1]) & np.isnan(Xq[both]))
    assert same.all(), "train and query inputs differ on shared origins"
    X = np.where(np.isnan(Xt[..., :-1]), Xq, Xt[..., :-1])
    y = Xt[..., -1]
    is_train = np.zeros(len(origins), bool)
    is_train[pt] = True
    Cm, Em = X[:, 0, :len(constant)].copy(), X[:, :, len(constant):].copy()
    for j, c in enumerate(constant):
        if fill_rule(c) != "none":
            f, carried = ffill(Cm[:, j:j + 1], IMAGE_CAP_H if fill_rule(c) == "image" else None)
            Cm[:, j] = (f + carried if c == "img_age_h" else f)[:, 0]
    for j, c in enumerate(varying):
        if fill_rule(c) != "none":
            Em[:, :, j] = ffill(Em[:, :, j], IMAGE_CAP_H if fill_rule(c) == "image" else None)[0]
    fit_flags = {p: tr.loc[tr["h"] == 1].set_index("origin")[f"fit_{p}"].astype(bool)
                 .reindex(origins, fill_value=False).to_numpy() for p in PERIODS}
    period = q.loc[q["h"] == 1].set_index("origin")["period"].astype(str).reindex(origins).to_numpy()
    return dict(origins=origins, C=Cm, E=Em, y=y, is_train=is_train, fit_flags=fit_flags, period=period)


def filled_targets(origins, y):
    """Targets of the given training rows with missing hours set to the last observed hourly speed along target
    time; only these rows' own targets are read (team training-y ffill)."""
    hours = (origins.asi8 // (3600 * 10 ** 9))[:, None] + np.arange(1, H + 1)[None, :]
    k = hours - hours.min()
    s = np.full(int(k.max()) + 1, np.nan)
    ok = np.isfinite(y)
    s[k[ok]] = y[ok]
    assert np.array_equal(s[k[ok]], y[ok]), "one target hour carries two different speeds"
    return ffill(s[:, None])[0][:, 0][k]


# ------------------------------------------------------------------------------ recipe --------------------------
def no_image(g):
    """The grid without the image columns (suvi_*, img_age_h constant; strip_* varying)."""
    hp = CONFIG["hyperparameters"]
    kc = [j for j, c in enumerate(hp["constant_columns"]) if not (c.startswith("suvi_") or c == "img_age_h")]
    ke = [j for j, c in enumerate(hp["varying_columns"]) if not c.startswith("strip_")]
    return dict(g, C=g["C"][:, kc], E=g["E"][:, :, ke])


def block_predictions(train, apply, y, t, rounds, threads, out, timing, label):
    """rank-1 spectral_predictions with tail weight t (and block 1 without image columns if CONFIG says so);
    returns the clipped forecast and the fitted models (one per block)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ni = CONFIG["hyperparameters"]["no_image_block0"]
    pred = np.empty((len(apply['origins']), 72), np.float64)
    models = []
    for b, start in enumerate((0, 24, 48)):
        tr, ap = (no_image(train), no_image(apply)) if (ni and b == 0) else (train, apply)
        x, xq = design(tr, start), design(ap, start)
        target = y[:, start:start + 24]
        began = time.perf_counter()
        model = native_head(x, target, rounds, threads, out, f'b{b}', objective_for(target, t))
        timing[f'{label}_b{b}_fit_s'] = round(time.perf_counter() - began, 3)
        timing[f'{label}_b{b}_columns'] = int(x.shape[1])
        pred[:, start:start + 24] = model.inplace_predict(xq)
        models.append((model, start, (ni and b == 0)))
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite component prediction')
    return np.clip(pred, 1., 3000.), models


def pull(prediction, rec, w):
    """Recurrence pull on leads 25-72: pred + w (rec27 - pred) where rec27 is finite, then clip 1-3000."""
    if w == 0:
        return prediction
    out = prediction.copy()
    p, r = prediction[:, 24:], rec[:, 24:]
    out[:, 24:] = np.where(np.isfinite(r), p + w * (np.nan_to_num(r) - p), p)
    return np.clip(out, 1., 3000.)


def self_checks():
    """spectral_objective_t(., 4) == spectral_objective bit for bit; lab S == official S on random data."""
    rng = np.random.default_rng(0)
    target = rng.normal(450, 120, (300, 24))
    target[rng.random(target.shape) < 0.1] = np.nan
    pr = rng.normal(450, 100, target.size)
    a, b = spectral_objective(target)(pr, None), spectral_objective_t(target, 4)(pr, None)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), "tail-weight objective differs at t = 4"
    y = rng.normal(450, 120, (200, H))
    y[rng.random(y.shape) < 0.2] = np.nan
    p = rng.normal(450, 100, (200, H))
    s1, s2 = s_of(lead_parts([p], y)[0]), official_s(y, p)
    assert abs(s1 - s2) <= 1e-9 * s2, (s1, s2)


def run_period(d, p, rounds, threads, work, smoke):
    hp = CONFIG["hyperparameters"]
    dials = hp["dials"]
    origins, is_train = d["origins"], d["is_train"]
    ev = np.flatnonzero(d["period"] == p)
    fit = np.flatnonzero(d["fit_flags"][p] & is_train)
    assert len(ev) == PERIODS[p] and is_train[fit].all(), (p, len(ev))
    assert (origins[fit] + pd.Timedelta(hours=144) <= pd.Timestamp(CUTOFFS[p], tz='UTC')).all(), 'training cutoff'
    yf = filled_targets(origins[fit], d["y"][fit])
    keep = np.isfinite(yf).all(1)
    dropped = int((~keep).sum())
    assert not keep[:np.argmax(keep)].any() and keep[np.argmax(keep):].all(), "gaps after the record start"
    fit, yf = fit[keep], yf[keep]
    y_raw = d["y"][fit]
    last_target = origins[fit].max() + pd.Timedelta(hours=H)
    assert last_target < origins[ev].min(), (p, last_target)
    f32 = lambda idx: {'origins': origins[idx], 'C': d["C"][idx].astype(np.float32),  # noqa: E731
                       'E': d["E"][idx].astype(np.float32)}
    train, query = f32(fit), f32(ev)
    ri = hp["varying_columns"].index("rec27")
    reference, calibration, boundary = prior_split(train['origins'])
    full = np.arange(len(fit))
    if smoke:  # team-style subsampling (calibration: evenly spaced 6-hourly origins)
        reference = reference[np.linspace(0, len(reference) - 1, min(len(reference), 128)).astype(int)]
        six = calibration[train['origins'][calibration].hour % 6 == 0]
        calibration = six[np.linspace(0, len(six) - 1, min(len(six), 64)).astype(int)]
        full = np.linspace(0, len(fit) - 1, min(len(fit), 192)).astype(int)
    out = Path(work) / p
    out.mkdir(parents=True, exist_ok=True)
    timing, states, cand = {}, {}, {}
    cal_o = train['origins'][calibration]
    six_cal = np.flatnonzero(cal_o.hour % 6 == 0)
    y_sel = y_raw[calibration][six_cal]
    rec_cal = train['E'][calibration][:, :, ri].astype(np.float64)
    for t in dials["tail_weights"]:
        prior, _ = block_predictions(subset(train, reference), subset(train, calibration), yf[reference], t, rounds,
                                     threads, out / f'prior_t{t}', timing, f'prior_t{t}')
        st = estimate_state(prior, yf[calibration])
        st.update(tail_weight=t, reference_rows=len(reference), reference_start=str(train['origins'][reference].min()),
                  reference_end=str(train['origins'][reference].max()), calibration_start=str(cal_o.min()),
                  calibration_end=str(cal_o.max()), split_boundary=str(boundary), purge_hours=144)
        (out / f'frozen_calibration_t{t}.json').write_text(json.dumps(st, indent=2) + '\n')
        np.savez_compressed(out / f'prior_predictions_t{t}.npz', origins=cal_o.asi8, spectral=prior)
        states[t] = st
        adj = apply_state(prior, st)
        for w in dials["pull_weights"]:
            cand[t, w] = pull(adj, rec_cal, w)[six_cal]
    keys = list(cand)  # grid order: t ascending, then w ascending
    assert keys == sorted(keys)
    S = dict(zip(keys, (s_of(pp) for pp in lead_parts([cand[k] for k in keys], y_sel))))
    best = keys[0]
    for k in keys[1:]:  # strict improvement only: ties keep the smaller t, then the smaller w
        if S[k] < S[best]:
            best = k
    t_star, w_star = best
    s_check = official_s(y_sel, cand[best])
    assert abs(s_check - S[best]) <= 1e-9 * S[best], (s_check, S[best])
    sel = dict(rule=dials["selection"], tail_weights=dials["tail_weights"], pull_weights=dials["pull_weights"],
               S={f't{t}': {f'w{w}': S[t, w] for w in dials["pull_weights"]} for t in dials["tail_weights"]},
               choice=dict(t=t_star, w=w_star, S=S[best], S_official_metric_code=s_check),
               population=dict(origins=int(len(six_cal)), first=str(cal_o[six_cal].min()),
                               last=str(cal_o[six_cal].max()), observed_cells=int(np.isfinite(y_sel).sum()),
                               truth='raw training y of these origins (NaN excluded)',
                               rec27_finite_fraction_leads_25_72=float(np.isfinite(rec_cal[six_cal][:, 24:]).mean())))
    (out / 'selection.json').write_text(json.dumps(sel, indent=2) + '\n')
    future, models = block_predictions(subset(train, full), query, yf[full], t_star, rounds, threads,
                                       out / f'full_t{t_star}', timing, f'full_t{t_star}')
    rec_q = query['E'][:, :, ri].astype(np.float64)
    pred = pull(apply_state(future, states[t_star]), rec_q, w_star)
    assert np.isfinite(pred).all() and pred.min() >= 1. and pred.max() <= 3000.
    lat = []  # post-fit time of one 72 h forecast: three block models + offset + pull
    for i in range(0, len(ev), 29):
        ts = time.perf_counter()
        f = np.empty((1, H))
        for m, start, ni in models:
            g = subset(query, slice(i, i + 1))
            f[:, start:start + 24] = m.inplace_predict(design(no_image(g) if ni else g, start))
        pull(apply_state(np.clip(f, 1., 3000.), states[t_star]), rec_q[i:i + 1], w_star)
        lat.append(time.perf_counter() - ts)
    timing['inference_seconds_per_forecast_mean'] = float(np.mean(lat))
    (out / 'timing.json').write_text(json.dumps(timing, indent=2) + '\n')
    info = dict(n_fit_origins=int(len(fit)), dropped_start_origins_unfilled_y=dropped,
                reference_rows=int(len(reference)), calibration_rows=int(len(calibration)),
                selection_origins=int(len(six_cal)), split_boundary=str(boundary),
                last_training_target_utc=str(last_target), t_star=t_star, w_star=w_star, S_calibration=S[best],
                offset_mean=float(np.mean(states[t_star]['offset'])))
    return pred, origins[ev], info, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--df", required=True)
    ap.add_argument("--out", default="predictions.parquet")
    ap.add_argument("--work", help="artifact directory (default: <out dir>/artifacts)")
    ap.add_argument("--smoke", action="store_true", help="testing only: 2 rounds, team-style row subsampling")
    ap.add_argument("--smoke-rounds", type=int, help="testing only: all rows, fewer boosting rounds")
    a = ap.parse_args()
    t0 = time.time()
    hp = CONFIG["hyperparameters"]
    assert hp["params"] == PARAMS and hp["seed"] == PARAMS["seed"] == 42
    assert hp["constant_columns"] == [mapped(c) for c in selected_names(FEATURES)["constant"]]
    assert hp["varying_columns"] == selected_names(FEATURES)["varying"]
    self_checks()
    rounds = 2 if a.smoke else (a.smoke_rounds or hp["rounds"])
    smoke = bool(a.smoke or a.smoke_rounds)
    threads = hp["threads"]
    work = Path(a.work) if a.work else Path(a.out).resolve().parent / "artifacts"
    d = prepare(a.df, hp["constant_columns"], hp["varying_columns"])
    print(f"[{time.time() - t0:6.0f}s] data ready: {len(d['origins'])} origins", flush=True)
    oo, preds, info, lat = [], [], {}, []
    for p in PERIODS:
        pred, o, info[p], lt = run_period(d, p, rounds, threads, work, a.smoke)
        lat += lt
        print(f"[{time.time() - t0:6.0f}s] {p}: {len(o)} origins, {json.dumps(info[p])}", flush=True)
        preds.append(pred)
        oo.append(o)
    o = oo[0].append(oo[1]).append(oo[2])
    pred = np.vstack(preds)
    res = pd.DataFrame({"origin_last_input_utc": np.repeat(o.strftime("%Y-%m-%dT%H:%M:%SZ"), H),
                        "horizon_hours": np.tile(np.arange(1, H + 1), len(o)), "pred_kms": pred.ravel()})
    assert not res.duplicated(["origin_last_input_utc", "horizon_hours"]).any()
    res.to_parquet(a.out, index=False)
    print(json.dumps(dict(rows=len(o) * H, out=a.out, model=MODEL, config_sha256=CONFIG_SHA256, smoke=smoke,
                          rounds=rounds, **info, inference_seconds_per_fold=float(np.mean(lat)),
                          elapsed_s=round(time.time() - t0), xgboost=xgb.__version__)))


if __name__ == "__main__":
    main()
