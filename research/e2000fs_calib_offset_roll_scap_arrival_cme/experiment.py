"""e2000fs_calib_offset_roll_scap_arrival_cme — S-cap + public DONKI Earth-arrival state + CME kinematics, mean of seeds 42/43/44

The e2000fs_calib_offset_roll_scap recipe (A900 point4 + spectral, then 180 direct-S continuation trees under a
1.01 x training-MSE cap) unchanged, with 8 origin constants appended after its 215 design columns:
  T  pub_enlil_next_arr_h, pub_enlil_since_arr_h, pub_enlil_glancing
  K  pub_cme_next_speed_kms, pub_cme_next_half_angle_deg, pub_cme_next_lon_deg, pub_cme_next_lat_deg,
     pub_cme_last_speed_kms
Both groups are parsed from CCMC DONKI CME notifications issued at or before each origin
(research/_module/public_events.py; -999 = not available). Source runs: ablation arm precursor_tk (N2) in
~/sunrun-event-precursor-ablation/runs/precursor_tk_s4{2,3,4}.

Usage (GBM: CPU only; the shared runner is research/_module/gbm.py):
  CUDA_VISIBLE_DEVICES= python experiment.py --threads 6 [--donki notifications.json]   # all seeds, F1/F2/F3, mean3, score
  CUDA_VISIBLE_DEVICES= python experiment.py --predict-only [--donki notifications.json] # inference from weights/
  CUDA_VISIBLE_DEVICES= python experiment.py --smoke --out DIR                           # 2-round wiring check
Without --donki the notifications are downloaded from the DONKI API (2022-01-01 to today) into a temporary folder.
--data is the df.parquet the DONKI columns are appended to (temporary copy, removed after the run).
Outputs: weights/F{1,2,3}/s{seed}.npz in this folder; results/{s42,s43,s44,mean3}_{predictions.parquet,score.json}
and logs/ in research/result/e2000fs_calib_offset_roll_scap_arrival_cme/ (or DIR/ with --out DIR).
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _module import common, gbm, public_events  # noqa: E402

EXPERIMENT = 'e2000fs_calib_offset_roll_scap_arrival_cme'
PARENT = 'e2000fs_calib_offset_roll_scap'
_loader = importlib.util.spec_from_file_location(f'{PARENT}_experiment', HERE.parent / PARENT / 'experiment.py')
ANCHOR = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(ANCHOR)

SPEC = dataclasses.replace(
    ANCHOR.SPEC, experiment=EXPERIMENT, constant=[*ANCHOR.SPEC.constant, *public_events.FEATURES],
    launch={**ANCHOR.SPEC.launch, 'parent_experiment': PARENT, 'added_constant': public_events.FEATURES,
            'added_varying': [], 'publication_gate': 'only DONKI notifications issued at or before the origin',
            'source_ablation': '~/sunrun-event-precursor-ablation precursor_tk (N2)'})


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
