"""current_offset_roll_direct_s — direct training-sample S only, mean of seeds 42/43/44.

Fresh 900-round XGBoost models minimize sum_h (MSE_h * TIMSE_h)**0.25 with
one fixed initial block scale and a positive diagonal curvature proxy.
No point4 tail threshold/weight and no Fourier penalty are used. Features,
prior/full split, training-only frozen offset and clipping follow the current
31-source / 215-expanded-feature offset-roll recipe. Source runs are
research/current_offset_roll_direct_s_20261109/runs/B_s{42,43,44}.

Usage (GBM: CPU only; the shared runner is research/_module/gbm.py):
  CUDA_VISIBLE_DEVICES= python experiment.py --threads 4             # all seeds, F1/F2/F3, mean3, score
  CUDA_VISIBLE_DEVICES= python experiment.py --predict-only          # inference from weights/
  CUDA_VISIBLE_DEVICES= python experiment.py --smoke --out DIR       # 2-round wiring check
Outputs: weights/F{1,2,3}/s{seed}.npz in this folder; results/{s42,s43,s44,mean3}_{predictions.parquet,score.json}
and logs/ (*.jsonl, F{n}_s{seed}_metadata.json) in research/result/current_offset_roll_direct_s/ (or DIR/ with --out DIR).
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

EXPERIMENT = 'current_offset_roll_direct_s'
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


class DirectSObjective:
    """Callable XGBoost objective; all internal arithmetic uses float64."""

    def __init__(self, target):
        self.target = np.array(target, dtype=np.float64, copy=True)
        if (self.target.ndim != 2 or min(self.target.shape) == 0
                or not np.isfinite(self.target).all()):
            raise ValueError('Training targets require a nonempty finite matrix')
        self.weights = np.column_stack([
            timse_coefficients(column) for column in self.target.T])
        # XGBoost 2.1 stores the explicitly supplied scalar base_score as float32.
        self.initial_prediction = float(np.float32(self.target.mean()))
        _, _, _, initial_coefficient = self._terms(
            np.full_like(self.target, self.initial_prediction) - self.target)
        self.scale = float(1. / initial_coefficient.mean())
        self.metadata = {
            'objective': 'direct_s',
            'initial_prediction': self.initial_prediction,
            'fixed_block_scale': self.scale,
            'initial_mean_s_curvature': 1.,
            'curvature': 'positive_residual_coefficient_diagonal_proxy',
            'rows': self.target.shape[0], 'leads': self.target.shape[1],
        }
        self.trace = []
        self.calls = 0

    def _error(self, prediction):
        prediction = np.asarray(prediction, dtype=np.float64)
        if prediction.shape != self.target.shape or not np.isfinite(prediction).all():
            raise ValueError('Predictions require the finite training-target shape')
        return prediction - self.target

    def _terms(self, error):
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            squared = error ** 2
            mse = squared.mean(axis=0)
            timse = (self.weights * squared).sum(axis=0)
            if not ((mse > 0) & (timse > 0)).all():
                raise ValueError('Direct S requires positive MSE and TIMSE at every lead')
            score = (mse * timse) ** .25
            coefficient = .5 * score * (
                1. / (len(error) * mse) + self.weights / timse)
        return mse, timse, score, coefficient

    def value(self, prediction):
        """Scalar training objective, useful for independent gradient checks."""
        error = self._error(prediction)
        _, _, score, _ = self._terms(error)
        value = self.scale * score.sum()
        return float(value)

    def derivatives(self, prediction):
        """Return float64 gradient and curvature proxy without advancing traces."""
        error = self._error(prediction)
        _, _, _, coefficient = self._terms(error)
        hessian = self.scale * coefficient
        gradient = hessian * error
        return gradient, hessian

    def __call__(self, prediction, matrix):
        prediction = np.asarray(prediction).reshape(self.target.shape)
        gradient, hessian = self.derivatives(prediction)
        if self.calls % 50 == 0:
            mse, timse, score, _ = self._terms(self._error(prediction))
            self.trace.append({
                'round': self.calls, 'fixed_block_scale': self.scale,
                'mse_by_lead': mse.tolist(), 'timse_by_lead': timse.tolist(),
                's_by_lead': score.tolist(),
                'mean_curvature': float(hessian.mean()),
                'max_curvature': float(hessian.max()),
            })
        self.calls += 1
        with np.errstate(over='raise', invalid='raise'):
            return gradient.astype(np.float32), hessian.astype(np.float32)


SPEC = gbm.GBMSpec(
    experiment=EXPERIMENT, constant=PARENT_CONSTANT + ADDED_CONSTANT, varying=VARYING, parent_constant=PARENT_CONSTANT,
    objective='direct_s (fixed initial block scale, diagonal curvature proxy)', component='direct_s',
    make_objective=lambda target, context: DirectSObjective(target))


if __name__ == '__main__':
    gbm.main(SPEC, __doc__, HERE)
