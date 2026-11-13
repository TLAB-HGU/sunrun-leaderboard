"""e2000fs_calib_offset_roll_CDF_1p3F — replacement ablation: target weight 1 + 3 * lognormal CDF instead of the 550 step

The e2000fs_calib_offset_roll recipe (same 31 features, XGBoost parameters, 900 rounds, prior model + frozen per-lead
offset, clip [1, 3000], seeds 42/43/44) with one change: the squared-error weight of a target v is
1 + 3 * F(v), F(v) = Phi(ln(v / scale) / sigma), instead of the step (4 if v >= 550 else 1). It keeps the step's
range [1, 4] (slow-wind floor 1) and removes only the discontinuity. Weights are normalised to mean 1 per lead;
gradient = w * error, hessian = w. sigma and scale are the loc = 0 maximum-likelihood fit to the observed
(unfilled) targets of each period's own training rows, so every period has its own pre-cutoff CDF
(function form from analysis/solar_wind_cdf_approx_2025_2026.ipynb; its 2025-2026 constants are not used).

Usage (GBM: CPU only; the shared runner is research/_module/gbm.py):
  CUDA_VISIBLE_DEVICES= python experiment.py --threads 4             # all seeds, F1/F2/F3, mean3, score
  CUDA_VISIBLE_DEVICES= python experiment.py --predict-only          # inference from weights/
  CUDA_VISIBLE_DEVICES= python experiment.py --smoke --out DIR       # 2-round wiring check
Outputs: weights/F{1,2,3}/s{seed}.npz in this folder; results/{s42,s43,s44,mean3}_{predictions.parquet,score.json}
and logs/ (*.jsonl, F{n}_s{seed}_metadata.json) in research/result/e2000fs_calib_offset_roll_CDF_1p3F/ (or DIR/ with --out DIR).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _module import gbm, objectives  # noqa: E402

EXPERIMENT = 'e2000fs_calib_offset_roll_CDF_1p3F'
PARENT_CONSTANT = ['v_lag0', 'ace_speed_kms_last', 'ace_density_cm3_last', 'ace_density_cm3_m6',
                   'ace_temperature_k_last', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6',
                   'ace_by_gsm_nt_m24', 'ace_epam_p_channel3_last', 'ace_sis_gt10_flux_last',
                   'suvi_t035_top2_area', 'suvi_t035_meridian15_equatorial_area',
                   'suvi_t035_meridian30_equatorial_area', 'suvi_t055_meridian15_area',
                   'suvi_t055_meridian30_equatorial_area', 'suvi_t055_equatorial_area', 'img_age_h',
                   'enlil_n_pending', 'enlil_kp']
ADDED_CONSTANT = ['v_lag24', 'v_lag48', 'v_lag72']
VARYING = ['rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed', 'strip_v450_logmed', 'strip_v550_dark',
           'strip_v750_logmed']
OBJECTIVE = 'lognormal-CDF weighted L2 (weight = 1 + 3 F(target), normalised per lead)'
WEIGHT_FLOOR, WEIGHT_SPAN = 1., 3.


SPEC = gbm.GBMSpec(
    experiment=EXPERIMENT, constant=PARENT_CONSTANT + ADDED_CONSTANT, varying=VARYING, parent_constant=PARENT_CONSTANT,
    objective=OBJECTIVE, component='cdf_weight',
    make_objective=lambda target, cdf: objectives.cdf_weighted(target, cdf, WEIGHT_FLOOR, WEIGHT_SPAN),
    keep_observed=True, prepare_period=objectives.fit_speed_cdf, context_name='target_weight_cdf',
    launch=dict(weight=dict(floor=WEIGHT_FLOOR, span=WEIGHT_SPAN)))


if __name__ == '__main__':
    gbm.main(SPEC, __doc__, HERE)
