"""current_offset_roll_cpc4: PureDirectS900 initialization; mgda4 single-tree proposals, raw-risk gate, <=180 added trees. See README and frozen external specification."""
from pathlib import Path
import os
import sys
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _module import gbm, pareto
EXPERIMENT = 'current_offset_roll_cpc4'
CONSTANT = ['v_lag0', 'ace_speed_kms_last', 'ace_density_cm3_last', 'ace_density_cm3_m6', 'ace_temperature_k_last', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6', 'ace_by_gsm_nt_m24', 'ace_epam_p_channel3_last', 'ace_sis_gt10_flux_last', 'suvi_t035_top2_area', 'suvi_t035_meridian15_equatorial_area', 'suvi_t035_meridian30_equatorial_area', 'suvi_t055_meridian15_area', 'suvi_t055_meridian30_equatorial_area', 'suvi_t055_equatorial_area', 'img_age_h', 'enlil_n_pending', 'enlil_kp', 'v_lag24', 'v_lag48', 'v_lag72']
VARYING = ['rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed', 'strip_v450_logmed', 'strip_v550_dark', 'strip_v750_logmed']
PARENT_CONSTANT = ['v_lag0', 'ace_speed_kms_last', 'ace_density_cm3_last', 'ace_density_cm3_m6', 'ace_temperature_k_last', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6', 'ace_by_gsm_nt_m24', 'ace_epam_p_channel3_last', 'ace_sis_gt10_flux_last', 'suvi_t035_top2_area', 'suvi_t035_meridian15_equatorial_area', 'suvi_t035_meridian30_equatorial_area', 'suvi_t055_meridian15_area', 'suvi_t055_meridian30_equatorial_area', 'suvi_t055_equatorial_area', 'img_age_h', 'enlil_n_pending', 'enlil_kp']
FITTER = pareto.ParetoContinuation(tuple(CONSTANT), tuple(VARYING), mode='mgda4')

SPEC = gbm.GBMSpec(
    experiment=EXPERIMENT, constant=CONSTANT, varying=VARYING,
    parent_constant=PARENT_CONSTANT, objective='PureDirectS900 initialization; mgda4 single-tree proposals, raw-risk gate, <=180 added trees', component='CPC-MGDA4',
    prepare_period=FITTER.prepare_period, make_objective=None, fit_block=FITTER.fit_block, predict_block=FITTER.predict_block,
    keep_observed=True, context_name=None, rounds=180, params={'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 20, 'reg_lambda': 5, 'objective': 'reg:squarederror', 'tree_method': 'hist', 'device': 'cpu', 'multi_strategy': 'multi_output_tree', 'eval_metric': 'rmse', 'seed': 42, 'verbosity': 0},
    launch=dict(arm='CPC-MGDA4', initial_recipe='current_offset_roll_direct_s',
                preregistration='sep-inhar-cpc-followups-20261113', model_structure='3x24lead',
                reference_rounds=900, proposal_rounds=180))
if __name__ == '__main__':
    gbm.main(SPEC, __doc__, HERE)
