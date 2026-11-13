"""e2000fs_calib_offset_roll_splus — A900 board recipe (point4 + spectral) + 180 direct-S continuation trees, mean of seeds 42/43/44

The point4 + spectral e2000fs_calib_offset_roll recipe (git e6dde5e1a; drop_t035_v650 + v_lag24/48/72,
prior model + frozen per-lead offset, 900 rounds per 24-lead block) is trained first. Each of its
boosters is then continued with the training-sample competition score S = mean_h (MSE_h * TIMSE_h)^(1/4)
plus the same 0.1 x robust Fourier term: 6 stages x 30 proposal trees (stage seed = seed + stage),
S coefficients refreshed every round, and each stage accepted with an exact Armijo line search
(c = 1e-4, alpha in 1, 1/2, ..., 1/64, else 0). Source runs:
research/loss_mechanism_ablation_20261112/runs/splus_s4{2,3,4}.

Usage (GBM: CPU only; the shared runner is research/_module/gbm.py):
  CUDA_VISIBLE_DEVICES= python experiment.py --threads 5             # all seeds, F1/F2/F3, mean3, score
  CUDA_VISIBLE_DEVICES= python experiment.py --predict-only          # inference from weights/
  CUDA_VISIBLE_DEVICES= python experiment.py --smoke --out DIR       # 2-round wiring check
Outputs: weights/F{1,2,3}_s{seed}.npz in this folder; results/{s42,s43,s44,mean3}_{predictions.parquet,score.json}
and logs/ (*.jsonl, F{n}_s{seed}_metadata.json) in research/result/e2000fs_calib_offset_roll_splus/ (or DIR/ with --out DIR).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _module import continuation, gbm, objectives  # noqa: E402

EXPERIMENT = 'e2000fs_calib_offset_roll_splus'
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
SPECTRAL_COEFFICIENT = .1
OBJECTIVE = 'direct_s_spectral_continuation'
MSE_CAP = None
CONTINUATION = continuation.Continuation(OBJECTIVE, spectral_coefficient=SPECTRAL_COEFFICIENT, mse_cap=MSE_CAP)


SPEC = gbm.GBMSpec(
    experiment=EXPERIMENT, constant=PARENT_CONSTANT + ADDED_CONSTANT, varying=VARYING, parent_constant=PARENT_CONSTANT,
    objective=OBJECTIVE, component=OBJECTIVE, threads=5,
    make_objective=lambda target, context: objectives.point4_spectral(target, SPECTRAL_COEFFICIENT),
    fit_block=CONTINUATION.fit_block, predict_block=CONTINUATION.predict_block,
    launch=dict(tail=dict(threshold=objectives.TAIL_THRESHOLD, weight=objectives.TAIL_WEIGHT), spectral_coefficient=SPECTRAL_COEFFICIENT,
                continuation=CONTINUATION.launch_record()))


if __name__ == '__main__':
    gbm.main(SPEC, __doc__, HERE)
