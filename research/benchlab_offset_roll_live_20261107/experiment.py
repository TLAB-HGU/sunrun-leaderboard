"""benchlab_offset_roll_live: the e2000fs_calib_offset_roll architecture fitted once for the BenchLab live tournament.

Identical model to research/rerun_df_ffill_20261103/e2000fs_calib_offset_roll (30 features, three 24-hour
multi-output XGBoost blocks, spectral objective, point4 tail weight, frozen per-horizon offset estimated by models
fitted before the latest quarter with a 144-hour purge). Only the training window differs: every origin whose 144-hour
target window ends on or before --cutoff, instead of the leaderboard period cutoffs. There is no query set and no
evaluation here; the outputs are the booster files, the frozen calibration and the feature specification that the
submission container reads. Deployment model: leaderboard-period truth is part of its training data.
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


EXPERIMENT = 'benchlab_offset_roll_live'

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

def load_grid(data, names, cutoff, fit_column=None):
    """Training rows: `set == train` (every origin the table built for the live cutoff), or `fit_column == True`."""
    filters = [(fit_column, '==', True)] if fit_column else [('set', '==', 'train')]
    constant = pd.read_parquet(data, columns=['origin', *[mapped(c) for c in names['constant']]], filters=[*filters, ('h', '==', 1)]).sort_values('origin', kind='stable')
    origins = pd.DatetimeIndex(pd.to_datetime(constant.origin, utc=True))
    if origins.has_duplicates:
        raise ValueError('duplicate input origins')
    columns = ['origin', 'h', *names['varying'], 'y']
    rows = pd.read_parquet(data, columns=columns, filters=filters).sort_values(['origin', 'h'], kind='stable')
    if len(rows) != len(origins) * H or not np.array_equal(rows.h.to_numpy().reshape(-1, H), np.broadcast_to(np.arange(1, H + 1), (len(origins), H))):
        raise ValueError('incomplete origin/horizon grid')
    if not origins.equals(pd.DatetimeIndex(pd.to_datetime(rows.origin.iloc[::H], utc=True))):
        raise ValueError('constant and varying grids disagree')
    if not (origins + pd.Timedelta(hours=144) <= pd.Timestamp(cutoff, tz='UTC')).all():
        raise ValueError('training cutoff violated')
    grid = {'origins': origins, 'C': constant[[mapped(c) for c in names['constant']]].to_numpy(np.float32), 'E': rows[names['varying']].to_numpy(np.float32).reshape(len(origins), H, len(names['varying']))}
    return grid, rows.y.to_numpy(np.float64).reshape(len(origins), H)

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


def spectral_predictions(train,apply,y,threads,out,smoke):
    """Three 24-hour multi-output XGBoost blocks with the spectral objective."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    pred=np.empty((len(apply['origins']),72),np.float64)
    rounds=2 if smoke else 900
    for b,start in enumerate((0,24,48)):
        x,xq=design(train,start),design(apply,start)
        target=y[:,start:start+24]
        model=native_head(x,target,rounds,threads,out,f'b{b}',spectral_objective(target))
        pred[:,start:start+24]=model.inplace_predict(xq)
    if not np.isfinite(pred).all():raise ValueError('Nonfinite component prediction')
    return np.clip(pred,1.,3000.)


def estimate_state(prediction,y):
    if not np.isfinite(y).all():raise ValueError('Calibration truth must contain finite training labels only')
    if prediction.shape!=y.shape or not np.isfinite(prediction).all():
        raise ValueError('Calibration component grids disagree')
    # Every residual here is matured before the cutoff; state never updates.
    offset=OFFSET_ALPHA*(y[-OFFSET_WINDOW:]-prediction[-OFFSET_WINDOW:]).mean(0)
    return dict(policy=POLICY,components=['spectral'],calibration_rows=len(y),window=OFFSET_WINDOW,alpha=OFFSET_ALPHA,
                settings_source='Original submitted fixed alpha/window; residual state re-estimated on prior training only',
                offset=offset.tolist())


def subset(grid,indices):
    return {k:v[indices] for k,v in grid.items()}


def fit(train,y,threads,out,smoke=False):
    """Prior models -> frozen offset, then the full models that the container loads."""
    reference,calibration,boundary=prior_split(train['origins'])
    if smoke:
        reference=reference[np.linspace(0,len(reference)-1,min(len(reference),128)).astype(int)]
        calibration=calibration[np.linspace(0,len(calibration)-1,min(len(calibration),64)).astype(int)]
        full=np.linspace(0,len(y)-1,min(len(y),192)).astype(int)
    else:full=np.arange(len(y))
    prior=spectral_predictions(subset(train,reference),subset(train,calibration),y[reference],threads,Path(out)/'prior'/'spectral',smoke)
    state=estimate_state(prior,y[calibration])
    state.update(reference_rows=len(reference),reference_start=str(train['origins'][reference].min()),
                 reference_end=str(train['origins'][reference].max()),calibration_start=str(train['origins'][calibration].min()),
                 calibration_end=str(train['origins'][calibration].max()),split_boundary=str(boundary),purge_hours=144)
    Path(out).mkdir(parents=True,exist_ok=True)
    (Path(out)/'frozen_calibration.json').write_text(json.dumps(state,indent=2)+'\n')
    spectral_predictions(subset(train,full),subset(train,calibration[:1]),y[full],threads,Path(out)/'full'/'spectral',smoke)
    return state


def run(data,out,cutoff,threads=1,smoke=False,fit_column=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);began=time.perf_counter()
    names=selected_names(FEATURES)
    train,y=load_grid(data,names,cutoff,fit_column)
    if not np.isfinite(y).all():raise ValueError('Fitting requires the prepared training-y ffill snapshot')
    state=fit(train,y,threads,out/'models',smoke)
    (out/'feature_spec.json').write_text(json.dumps(dict(model=EXPERIMENT,features=names,blocks=[0,24,48]),indent=1)+'\n')
    revision=subprocess.run(['git','rev-parse','HEAD'],text=True,capture_output=True,check=False)
    report=dict(experiment=EXPERIMENT,smoke=smoke,policy=POLICY,
                data=str(Path(data).resolve()),data_bytes=Path(data).stat().st_size,data_mtime_ns=Path(data).stat().st_mtime_ns,
                git_commit=revision.stdout.strip(),seed=42,features=names,parameters=PARAMS,rounds=2 if smoke else 900,
                component_settings=dict(spectral_coefficient=.10,spectral_scale=50.),
                training_cutoff=cutoff,fit_column=fit_column,maturity_hours=144,train_origins=len(y),
                train_first_origin=str(train['origins'].min()),train_last_origin=str(train['origins'].max()),
                target_policy='Upstream causal ffill of feature and training-y columns only; no evaluation truth is used.',
                frozen_state=state,end_to_end_seconds=time.perf_counter()-began,
                versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,xgboost=xgb.__version__))
    (out/'metadata.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=(EXPERIMENT,),default=EXPERIMENT)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cutoff',required=True,help='UTC date; every training origin must satisfy origin + 144 h <= cutoff')
    p.add_argument('--fit-column',help='train on rows of this boolean column instead of set == train (equivalence check)')
    p.add_argument('--threads',type=int,default=1);p.add_argument('--smoke',action='store_true')
    p.add_argument('--device',choices=('cpu',),default='cpu',help='GBM runs on CPU only')
    args=p.parse_args()
    if args.threads<1:p.error('threads must be positive')
    report=run(args.data,args.out,args.cutoff,args.threads,args.smoke,args.fit_column)
    print(json.dumps({k:report[k] for k in ['experiment','training_cutoff','train_origins','train_first_origin','train_last_origin','end_to_end_seconds']}),flush=True)


if __name__ == "__main__":
    main()
