"""Reproduce e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam_union9_lbsel:
three seeds end to end (public corpora, GBM, ridge, fixed post-hoc steps). Run from this repository; CPU only, all
fitting uses the official training cutoffs. The components themselves were selected on leaderboard scores, so the row
does not meet the leaderboard's tuned-only-before-cutoff rule.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--observations', required=True)
    parser.add_argument('--out', type=Path, required=True)
    for name in ('donki', 'srs', 'discussion'):
        parser.add_argument('--' + name, type=Path, help='saved corpus; otherwise downloaded from the public archive')
    parser.add_argument('--threads', type=int, default=6)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / 'research/e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam_union9_lbsel/experiment.py'), '--data', args.data,
               '--observations', args.observations, '--out', str(args.out / 'run'), '--device', 'cpu',
               '--threads', str(args.threads)]
    for name in ('donki', 'srs', 'discussion'):
        if getattr(args, name):
            command += ['--' + name, str(getattr(args, name))]
    subprocess.run(command, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), cwd=ROOT)
    shutil.copy2(args.out / 'run/results/mean3_predictions.parquet', args.out / 'predictions.parquet')
