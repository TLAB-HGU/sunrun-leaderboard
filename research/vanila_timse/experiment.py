"""vanila_timse — training-distribution TIMSE, mean of seeds 42/43/44.

Fresh 900-round XGBoost models use training-only TIMSE coefficients with
per-lead mean-one Hessian scale. The 31-source / 215-expanded-feature offset-roll inputs, prior/full
split, training-only frozen per-lead offset and clipping are retained.

Usage (GBM: CPU only; the shared runner is research/_module/gbm.py):
  CUDA_VISIBLE_DEVICES= python experiment.py --threads 4             # all seeds, F1/F2/F3, mean3, score
  CUDA_VISIBLE_DEVICES= python experiment.py --predict-only          # inference from weights/
  CUDA_VISIBLE_DEVICES= python experiment.py --smoke --out DIR       # 2-round wiring check
Outputs: weights/F{1,2,3}/s{seed}.npz in this folder; results/{s42,s43,s44,mean3}_{predictions.parquet,score.json}
and logs/ (*.jsonl, F{n}_s{seed}_metadata.json) in research/result/vanila_timse/ (or DIR/ with --out DIR).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _module import gbm  # noqa: E402

EXPERIMENT = 'vanila_timse'
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

from _module.objectives import timse_coefficients  # noqa: E402


def make_objective(target):
    """Half TIMSE with mean-one per-lead curvature: w = N * official coefficients.

    Coefficients use only this fit's training labels, before float32 DMatrix
    conversion. Mean curvature one preserves the MSE regularization scale;
    local curvature varies with the training target distribution.
    """
    target = np.array(target, dtype=np.float64, copy=True)
    if target.ndim != 2 or min(target.shape) == 0 or not np.isfinite(target).all():
        raise ValueError('Training targets require a nonempty finite matrix')
    weights = len(target) * np.column_stack([timse_coefficients(column) for column in target.T])
    if not np.allclose(weights.mean(axis=0), 1., rtol=1e-12, atol=1e-12):
        raise ValueError('TIMSE coefficients must have mean one per lead')

    def objective(prediction, matrix):
        prediction = np.asarray(prediction, dtype=np.float64).reshape(target.shape)
        if not np.isfinite(prediction).all():
            raise ValueError('Training predictions must be finite')
        return (weights * (prediction - target)).astype(np.float32), weights.astype(np.float32)

    return objective


SPEC = gbm.GBMSpec(
    experiment=EXPERIMENT, constant=PARENT_CONSTANT + ADDED_CONSTANT, varying=VARYING, parent_constant=PARENT_CONSTANT,
    objective='pure_timse (gradient=w*(prediction-target), hessian=w; w=N*c, sum(c)=1 per lead)', component='timse',
    make_objective=lambda target, context: make_objective(target))


if __name__ == '__main__':
    gbm.main(SPEC, __doc__, HERE)
