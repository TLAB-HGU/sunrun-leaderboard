"""vanila_mse_l27 — vanila_mse with only the current speed and its 27-day recurrence as inputs.

Replacement ablation of vanila_mse: the same unweighted squared error objective (gradient prediction - target,
unit Hessian), 900-round 24-lead XGBoost blocks, prior/full split, training-only frozen per-lead offset and
clipping, but the inputs are only v_lag0 (speed in the origin bin) and rec27 (speed at target_time - 27 d,
per lead): 2 source / 25 expanded features per block.

Usage (GBM: CPU only; the shared runner is research/_module/gbm.py):
  CUDA_VISIBLE_DEVICES= python experiment.py --threads 4             # all seeds, F1/F2/F3, mean3, score
  CUDA_VISIBLE_DEVICES= python experiment.py --predict-only          # inference from weights/
  CUDA_VISIBLE_DEVICES= python experiment.py --smoke --out DIR       # 2-round wiring check
Outputs: weights/F{1,2,3}/s{seed}.npz in this folder; results/{s42,s43,s44,mean3}_{predictions.parquet,score.json}
and logs/ (*.jsonl, F{n}_s{seed}_metadata.json) in research/result/vanila_mse_l27/ (or DIR/ with --out DIR).
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

EXPERIMENT = 'vanila_mse_l27'
CONSTANT = ['v_lag0']
VARYING = ['rec27']


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
    experiment=EXPERIMENT, constant=CONSTANT, varying=VARYING,
    objective='unweighted_mse (gradient=prediction-target, hessian=1)', component='mse',
    make_objective=lambda target, context: make_objective(target))


if __name__ == '__main__':
    gbm.main(SPEC, __doc__, HERE)
