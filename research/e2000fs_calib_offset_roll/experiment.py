"""e2000fs_calib_offset_roll — board row: point4 tail-weighted L2 only (spectral term removed), mean of seeds 42/43/44

The calib_offset_roll recipe (suvi_t035_equatorial_area and strip_v650_dark dropped, v_lag24/48/72 added) trained with
the tail-weighted squared error only: targets >= 550 km/s weigh 4, normalised per lead; gradient = w * error,
hessian = w. The 0.1 x FFT pseudo-Huber spectral term of the earlier row (S 96.311, 2026-11-09) was removed on user
request (2026-11-10). Prior model + frozen per-lead offset (0.25 x mean residual of the last 336 calibration origins,
training data only), clip [1, 3000], three 24-lead XGBoost blocks x 900 rounds. Source runs:
research/e2000fs_calib_offset_roll_no_spectral_20261110/runs/no_spectral_s4{2,3,4}.

Usage (GBM: CPU only; the shared runner is research/module/gbm.py):
  CUDA_VISIBLE_DEVICES= python experiment.py --threads 4             # all seeds, F1/F2/F3, mean3, score
  CUDA_VISIBLE_DEVICES= python experiment.py --predict-only          # inference from weights/
  CUDA_VISIBLE_DEVICES= python experiment.py --smoke --out DIR       # 2-round wiring check
Outputs: weights/F{1,2,3}_s{seed}.npz in this folder; results/{s42,s43,s44,mean3}_{predictions.parquet,score.json}
and logs/ (*.jsonl, F{n}_s{seed}_metadata.json) in research/result/e2000fs_calib_offset_roll/ (or DIR/ with --out DIR).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from module import gbm, objectives  # noqa: E402

EXPERIMENT = 'e2000fs_calib_offset_roll'
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


SPEC = gbm.GBMSpec(
    experiment=EXPERIMENT, constant=PARENT_CONSTANT + ADDED_CONSTANT, varying=VARYING, parent_constant=PARENT_CONSTANT,
    objective='point4 L2 only (spectral term removed)', component='point4',
    make_objective=lambda target, context: objectives.point4(target),
    launch=dict(tail=dict(threshold=objectives.TAIL_THRESHOLD, weight=objectives.TAIL_WEIGHT)))


if __name__ == '__main__':
    gbm.main(SPEC, __doc__, HERE)
