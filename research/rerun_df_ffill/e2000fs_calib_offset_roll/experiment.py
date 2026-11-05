"""e2000fs_calib_offset_roll: training-only calibration; original eval-stream adaptation removed.

This is an explicit verification-policy repair, not exact evaluation-stream
reproduction or a structural SOTA ablation. The spectral component model is newly
trained on the prepared feature/training-y ffill data. The latest prior training
quarter is predicted by models fitted before it with a 144-hour purge; the
per-horizon offset is frozen before official query inference. No old prediction
assets or query labels are read.
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


EXPERIMENT = 'e2000fs_calib_offset_roll'

H = 72

CUTOFFS = {'F1': '2025-12-29', 'F2': '2026-03-29', 'F3': '2026-06-29'}

QUERY_COUNTS = {'F1': 2089, 'F2': 2113, 'F3': 2113}

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


def spectral_predictions(train,apply,y,threads,out,smoke):
    """Three 24-hour multi-output XGBoost blocks with the spectral objective."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    pred=np.empty((len(apply['origins']),72),np.float64)
    seconds=0.;rounds=2 if smoke else 900
    for b,start in enumerate((0,24,48)):
        x,xq=design(train,start),design(apply,start)
        target=y[:,start:start+24]
        model=native_head(x,target,rounds,threads,out,f'b{b}',spectral_objective(target))
        began=time.perf_counter();pred[:,start:start+24]=model.inplace_predict(xq);seconds+=time.perf_counter()-began
    if not np.isfinite(pred).all():raise ValueError('Nonfinite component prediction')
    return np.clip(pred,1.,3000.),seconds


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


def fit_offset(train,query,y,threads,out,smoke=False):
    reference,calibration,boundary=prior_split(train['origins'])
    if smoke:
        reference=reference[np.linspace(0,len(reference)-1,min(len(reference),128)).astype(int)]
        calibration=calibration[np.linspace(0,len(calibration)-1,min(len(calibration),64)).astype(int)]
        full=np.linspace(0,len(y)-1,min(len(y),192)).astype(int)
    else:full=np.arange(len(y))
    prior,_=spectral_predictions(subset(train,reference),subset(train,calibration),y[reference],threads,Path(out)/'prior'/'spectral',smoke)
    state=estimate_state(prior,y[calibration])
    state.update(reference_rows=len(reference),reference_start=str(train['origins'][reference].min()),
                 reference_end=str(train['origins'][reference].max()),calibration_start=str(train['origins'][calibration].min()),
                 calibration_end=str(train['origins'][calibration].max()),split_boundary=str(boundary),purge_hours=144)
    Path(out).mkdir(parents=True,exist_ok=True)
    (Path(out)/'frozen_calibration.json').write_text(json.dumps(state,indent=2)+'\n')
    np.savez_compressed(Path(out)/'prior_predictions.npz',origins=train['origins'][calibration].asi8,spectral=prior)
    future,query_seconds=spectral_predictions(subset(train,full),query,y[full],threads,Path(out)/'full'/'spectral',smoke)
    began=time.perf_counter();prediction=apply_state(future,state);query_seconds+=time.perf_counter()-began
    return prediction,state,query_seconds


def run_period(data,out,period,threads=1,smoke=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);began=time.perf_counter()
    names=selected_names(FEATURES)
    train,y=load_grid(data,period,names,True);query,_=load_grid(data,period,names,False)
    if len(query['origins'])!=QUERY_COUNTS[period]:raise ValueError('Unexpected query grid')
    if not np.isfinite(y).all():raise ValueError('Pooling requires the prepared training-y ffill snapshot')
    if smoke:query=subset(query,np.arange(min(8,len(query['origins']))))
    prediction,state,seconds=fit_offset(train,query,y,threads,out/f'{period}_models',smoke)
    np.savez_compressed(out/f'{period}_predictions.npz',pred=prediction,origins=query['origins'].asi8)
    revision=subprocess.run(['git','rev-parse','HEAD'],text=True,capture_output=True,check=False)
    report=dict(experiment=EXPERIMENT,period=period,smoke=smoke,policy=POLICY,
                interpretation='Verification-policy repair, not exact original evaluation-stream reproduction; no structural SOTA ablation claim.',
                data=str(Path(data).resolve()),data_bytes=Path(data).stat().st_size,data_mtime_ns=Path(data).stat().st_mtime_ns,
                git_commit=revision.stdout.strip(),seed=42,features=names,parameters=PARAMS,rounds=2 if smoke else 900,
                component_settings=dict(spectral_coefficient=.10,spectral_scale=50.),
                training_cutoff=CUTOFFS[period],maturity_hours=144,train_origins=len(y),query_origins=len(prediction),
                target_policy='Upstream causal ffill of feature and training-y columns only; no query truth read or evaluation-stream updates.',
                frozen_state=state,inference_seconds_per_fold=seconds/len(prediction),end_to_end_seconds=time.perf_counter()-began,
                versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,xgboost=xgb.__version__))
    (out/f'{period}_metadata.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=(EXPERIMENT,),default=EXPERIMENT)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--threads',type=int,default=1);p.add_argument('--period',choices=list(CUTOFFS))
    p.add_argument('--smoke',action='store_true');p.add_argument('--merge',action='store_true')
    args=p.parse_args()
    if args.threads<1:p.error('threads must be positive')
    if args.merge:print(json.dumps(dict(prediction_rows=merge(args.out))));return
    for period in [args.period] if args.period else (['F1'] if args.smoke else list(CUTOFFS)):
        report=run_period(args.data,args.out,period,args.threads,args.smoke)
        print(json.dumps({k:report[k] for k in ['experiment','period','policy','train_origins','query_origins','end_to_end_seconds']}),flush=True)
    if not args.period and not args.smoke:print(json.dumps(dict(prediction_rows=merge(args.out))))


if __name__ == "__main__":
    main()
