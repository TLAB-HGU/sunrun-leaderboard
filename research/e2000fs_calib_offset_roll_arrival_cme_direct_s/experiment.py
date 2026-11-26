"""e2000fs_calib_offset_roll_arrival_cme_direct_s — arrival_cme features + stagewise dynamic Direct-S objective, mean of seeds 42/43/44

Replacement ablation of e2000fs_calib_offset_roll_scap_arrival_cme: the 223 design columns (S-cap's 215 + the 8
DONKI T/K origin constants, same names and order) and calibration (prior model + frozen per-lead offset) are kept;
the objective is replaced. The A900 point4 + 0.1 x spectral fit and its 180 S-cap continuation trees are removed and
each 24-lead block is fitted with the e2000fs_calib_offset_roll_relevance_direct_s objective instead: training-sample
S = sum_h (MSE_h * TIMSE_h)^(1/4) from a scalar mean start, 900 trees in 30-tree stages, coefficients refreshed every
round (mode 'dynamic'), each stage accepted only if its stage-start quadratic majorizer and exact training S do not
increase (research/_module/mm.py StagewiseS).

Usage (GBM: CPU only; the shared runner is research/_module/gbm.py):
  CUDA_VISIBLE_DEVICES= python experiment.py --threads 6 [--donki notifications.json]   # all seeds, F1/F2/F3, mean3, score
  CUDA_VISIBLE_DEVICES= python experiment.py --predict-only [--donki notifications.json] # inference from weights/
  CUDA_VISIBLE_DEVICES= python experiment.py --smoke --out DIR                           # 2-round wiring check
Without --donki the notifications are downloaded from the DONKI API (2022-01-01 to today) into a temporary folder.
--data is the df.parquet the DONKI columns are appended to (temporary copy, removed after the run).
Outputs: weights/F{1,2,3}/s{seed}.npz in this folder; results/{s42,s43,s44,mean3}_{predictions.parquet,score.json}
and logs/ in research/result/e2000fs_calib_offset_roll_arrival_cme_direct_s/ (or DIR/ with --out DIR).
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _module import common, gbm, mm, public_events  # noqa: E402

EXPERIMENT = 'e2000fs_calib_offset_roll_arrival_cme_direct_s'
PARENT_CONSTANT = ['v_lag0', 'ace_speed_kms_last', 'ace_density_cm3_last', 'ace_density_cm3_m6',
                   'ace_temperature_k_last', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6',
                   'ace_by_gsm_nt_m24', 'ace_epam_p_channel3_last', 'ace_sis_gt10_flux_last',
                   'suvi_t035_top2_area', 'suvi_t035_meridian15_equatorial_area',
                   'suvi_t035_meridian30_equatorial_area', 'suvi_t055_meridian15_area',
                   'suvi_t055_meridian30_equatorial_area', 'suvi_t055_equatorial_area', 'img_age_h',
                   'enlil_n_pending', 'enlil_kp']
ADDED_CONSTANT = ['v_lag24', 'v_lag48', 'v_lag72', *public_events.FEATURES]
VARYING = ['rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed', 'strip_v450_logmed', 'strip_v550_dark',
           'strip_v750_logmed']
OBJECTIVE = 'matched stagewise dynamic Direct-S with fixed stage-start Q acceptance'
FITTER = mm.StagewiseS(mode='dynamic', stage_rounds=30)

SPEC = gbm.GBMSpec(
    experiment=EXPERIMENT, constant=PARENT_CONSTANT + ADDED_CONSTANT, varying=VARYING,
    parent_constant=PARENT_CONSTANT, objective=OBJECTIVE, component='E3-R', threads=6,
    make_objective=None, fit_block=FITTER.fit_block, predict_block=FITTER.predict_block,
    launch=dict(feature_parent='e2000fs_calib_offset_roll_scap_arrival_cme',
                objective_parent='e2000fs_calib_offset_roll_relevance_direct_s',
                ablation='replacement: point4 + spectral A900 + S-cap continuation -> stagewise dynamic Direct-S',
                stagewise=FITTER.launch_record(),
                publication_gate='only DONKI notifications issued at or before the origin'))


def main():
    if {'-h', '--help'}.intersection(sys.argv[1:]):
        gbm.main(SPEC, __doc__, HERE)
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--donki', type=Path, help='saved DONKI CME notifications JSON (default: download)')
    parser.add_argument('--data', type=Path, default=common.DATA)
    known, rest = parser.parse_known_args()
    with tempfile.TemporaryDirectory(prefix=f'{EXPERIMENT}-') as work:
        corpus = known.donki or public_events.fetch_notifications(Path(work) / 'donki_notifications.json')
        table = Path(work) / 'table.parquet'
        SPEC.launch['donki'] = {**public_events.augment(known.data, corpus, table),
                                'df_source': common.file_identity(known.data)}
        sys.argv = [sys.argv[0], *rest, '--data', str(table)]
        gbm.main(SPEC, __doc__, HERE)


if __name__ == '__main__':
    main()
