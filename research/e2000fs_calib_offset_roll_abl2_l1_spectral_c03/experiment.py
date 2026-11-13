"""e2000fs_calib_offset_roll_abl2_l1_spectral_c03_mean3 — spectral coefficient 0.3, mean of seeds 42/43/44

The calib_offset_roll recipe (30 features) with the spectral loss coefficient raised from
0.1 to 0.3. Source runs: golab2 sunrun-offset-roll-ablation2-20261105/runs/l1_spectral_c03_s4{2,3,4}.

Usage (GBM: CPU only; the shared runner is research/_module/gbm.py):
  CUDA_VISIBLE_DEVICES= python experiment.py --threads 4             # all seeds, F1/F2/F3, mean3, score
  CUDA_VISIBLE_DEVICES= python experiment.py --predict-only          # inference from weights/
  CUDA_VISIBLE_DEVICES= python experiment.py --smoke --out DIR       # 2-round wiring check
Outputs: weights/F{1,2,3}_s{seed}.npz in this folder; results/{s42,s43,s44,mean3}_{predictions.parquet,score.json}
and logs/ (*.jsonl, F{n}_s{seed}_metadata.json) in research/result/e2000fs_calib_offset_roll_abl2_l1_spectral_c03/ (or DIR/ with --out DIR).
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

EXPERIMENT = 'e2000fs_calib_offset_roll_abl2_l1_spectral_c03_mean3'
CONSTANT = ['v_lag0', 'ace_speed_kms_last', 'ace_density_cm3_last', 'ace_density_cm3_m6',
            'ace_temperature_k_last', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6',
            'ace_by_gsm_nt_m24', 'ace_epam_p_channel3_last', 'ace_sis_gt10_flux_last', 'suvi_t035_top2_area',
            'suvi_t035_meridian15_equatorial_area', 'suvi_t035_meridian30_equatorial_area',
            'suvi_t035_equatorial_area', 'suvi_t055_meridian15_area', 'suvi_t055_meridian30_equatorial_area',
            'suvi_t055_equatorial_area', 'img_age_h', 'enlil_n_pending', 'enlil_kp']
VARYING = ['rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed', 'strip_v450_logmed', 'strip_v550_dark',
           'strip_v650_dark', 'strip_v750_logmed']
SPECTRAL_COEFFICIENT = 0.3


SPEC = gbm.GBMSpec(
    experiment=EXPERIMENT, constant=CONSTANT, varying=VARYING,
    objective=f'point4 + {SPECTRAL_COEFFICIENT} x FFT pseudo-Huber spectral term', component='spectral',
    make_objective=lambda target, context: objectives.point4_spectral(target, SPECTRAL_COEFFICIENT),
    launch=dict(tail=dict(threshold=objectives.TAIL_THRESHOLD, weight=objectives.TAIL_WEIGHT), spectral_coefficient=SPECTRAL_COEFFICIENT))


if __name__ == '__main__':
    gbm.main(SPEC, __doc__, HERE)
