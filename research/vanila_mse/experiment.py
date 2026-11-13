"""vanila_mse — unweighted squared error, mean of seeds 42/43/44.

Fresh 900-round XGBoost models use gradient prediction - target and unit
Hessian. The 31-source / 215-expanded-feature offset-roll inputs, prior/full
split, training-only frozen per-lead offset and clipping are retained.

Usage (GBM: CPU only; the shared runner is research/module/gbm.py):
  CUDA_VISIBLE_DEVICES= python experiment.py --threads 4             # all seeds, F1/F2/F3, mean3, score
  CUDA_VISIBLE_DEVICES= python experiment.py --predict-only          # inference from weights/
  CUDA_VISIBLE_DEVICES= python experiment.py --smoke --out DIR       # 2-round wiring check
Outputs: weights/F{1,2,3}_s{seed}.npz in this folder; results/{s42,s43,s44,mean3}_{predictions.parquet,score.json}
and logs/ (*.jsonl, F{n}_s{seed}_metadata.json) in research/result/vanila_mse/ (or DIR/ with --out DIR).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from module import gbm  # noqa: E402

EXPERIMENT = 'vanila_mse'
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


import numpy as np  # noqa: E402


def make_objective(target):
    """Unweighted squared error in XGBoost's half-squared-error convention."""
    target = np.asarray(target, dtype=np.float32)
    if target.ndim != 2 or min(target.shape) == 0 or not np.isfinite(target).all():
        raise ValueError('Training targets require a nonempty finite matrix')

    def objective(prediction, matrix):
        prediction = np.asarray(prediction, dtype=np.float32).reshape(target.shape)
        if not np.isfinite(prediction).all():
            raise ValueError('Training predictions must be finite')
        return prediction - target, np.ones_like(target)

    return objective


SPEC = gbm.GBMSpec(
    experiment=EXPERIMENT, constant=PARENT_CONSTANT + ADDED_CONSTANT, varying=VARYING, parent_constant=PARENT_CONSTANT,
    objective='unweighted_mse (gradient=prediction-target, hessian=1)', component='mse',
    make_objective=lambda target, context: make_objective(target))


if __name__ == '__main__':
    gbm.main(SPEC, __doc__, HERE)
