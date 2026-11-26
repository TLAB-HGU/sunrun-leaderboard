"""e2000fs_calib_offset_roll_relevance_fs: point4 + fixed selected additional information. See README for frozen study design."""
from pathlib import Path
import os
import sys
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _module import gbm, objectives
EXPERIMENT = 'e2000fs_calib_offset_roll_relevance_fs'
CONSTANT = ['v_lag0', 'ace_speed_kms_last', 'ace_density_cm3_last', 'ace_density_cm3_m6', 'ace_temperature_k_last', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6', 'ace_by_gsm_nt_m24', 'ace_epam_p_channel3_last', 'ace_sis_gt10_flux_last', 'suvi_t035_top2_area', 'suvi_t035_meridian15_equatorial_area', 'suvi_t035_meridian30_equatorial_area', 'suvi_t055_meridian15_area', 'suvi_t055_meridian30_equatorial_area', 'suvi_t055_equatorial_area', 'img_age_h', 'enlil_n_pending', 'enlil_kp', 'v_lag24', 'v_lag48', 'v_lag72', 'v_lag1', 'v_m6', 'v_lag3', 'v_lag6']
VARYING = ['rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed', 'strip_v450_logmed', 'strip_v550_dark', 'strip_v750_logmed']
PARENT_CONSTANT = ['v_lag0', 'ace_speed_kms_last', 'ace_density_cm3_last', 'ace_density_cm3_m6', 'ace_temperature_k_last', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6', 'ace_by_gsm_nt_m24', 'ace_epam_p_channel3_last', 'ace_sis_gt10_flux_last', 'suvi_t035_top2_area', 'suvi_t035_meridian15_equatorial_area', 'suvi_t035_meridian30_equatorial_area', 'suvi_t055_meridian15_area', 'suvi_t055_meridian30_equatorial_area', 'suvi_t055_equatorial_area', 'img_age_h', 'enlil_n_pending', 'enlil_kp']

SPEC = gbm.GBMSpec(
    experiment=EXPERIMENT, constant=CONSTANT, varying=VARYING,
    parent_constant=PARENT_CONSTANT, objective='point4 + fixed selected additional information', component='E-FS',
    make_objective=lambda target, context: objectives.point4(target),
    params={'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 20, 'reg_lambda': 5, 'objective': 'reg:squarederror', 'tree_method': 'hist', 'device': 'cpu', 'multi_strategy': 'multi_output_tree', 'eval_metric': 'rmse', 'seed': 42, 'verbosity': 0}, rounds=900,
    launch=dict(preregistration='continuous-relevance-pareto-xgb-20261113', arm='E-FS'))
if __name__ == '__main__':
    gbm.main(SPEC, __doc__, HERE)
