"""Frozen 28-feature parent input contract for additive feature experiments."""
import numpy as np
import pandas as pd

H = 72
CUTOFFS = {'F1': '2025-12-29', 'F2': '2026-03-29', 'F3': '2026-06-29'}
QUERY_COUNTS = {'F1': 2089, 'F2': 2113, 'F3': 2113}
PARENT_NAMES = {'constant': ['v_lag0', 'ace_speed_kms_last', 'ace_density_cm3_last', 'ace_density_cm3_m6', 'ace_temperature_k_last', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6', 'ace_by_gsm_nt_m24', 'ace_epam_p_channel3_last', 'ace_sis_gt10_flux_last', 'suvi_t035_top2_area', 'suvi_t035_meridian15_equatorial_area', 'suvi_t035_meridian30_equatorial_area', 'suvi_t055_meridian15_area', 'suvi_t055_meridian30_equatorial_area', 'suvi_t055_equatorial_area', 'img_age_h', 'enlil_n_pending', 'enlil_kp'], 'varying': ['rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed', 'strip_v450_logmed', 'strip_v550_dark', 'strip_v750_logmed']}
LOCKED_REMOVALS = ("suvi_t035_equatorial_area", "strip_v650_dark")
PARAMS = {'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 20, 'reg_lambda': 5, 'objective': 'reg:squarederror', 'tree_method': 'hist', 'device': 'cpu', 'multi_strategy': 'multi_output_tree', 'eval_metric': 'rmse', 'seed': 42, 'verbosity': 0}

def load_grid(data, period, names, training):
    for kind, retained in PARENT_NAMES.items():
        if names[kind][:len(retained)] != retained:
            raise ValueError('Parent source features/order must be retained')
    filters = [(f'fit_{period}', '==', True)] if training else [('set', '==', 'query'), ('period', '==', period)]
    constant = pd.read_parquet(data, columns=['origin', *names['constant']], filters=[*filters, ('h', '==', 1)]).sort_values('origin', kind='stable')
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
    grid = {'origins': origins, 'C': constant[names['constant']].to_numpy(np.float32), 'E': rows[names['varying']].to_numpy(np.float32).reshape(len(origins), H, len(names['varying']))}
    if np.isinf(grid['C']).any() or np.isinf(grid['E']).any():
        raise ValueError('Infinite feature values are not supported')
    return (grid, rows.y.to_numpy(np.float64).reshape(len(origins), H) if training else None)


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

def subset(grid,indices):
    return {k:v[indices] for k,v in grid.items()}
