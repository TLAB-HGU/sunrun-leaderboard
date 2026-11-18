"""Reproduce e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam: retrain the base row, then apply the calibration-selected block-seam smoothing.
Run from this repository. All fitting is CPU-only and uses the official training cutoffs.
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
    parser.add_argument('--donki', type=Path, help='Saved DONKI notifications; otherwise download from CCMC')
    parser.add_argument('--threads', type=int, default=6)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    donki = args.donki
    if donki is None:
        sys.path.insert(0, str(ROOT / 'research'))
        from _module import public_events
        donki = public_events.fetch_notifications(args.out / 'donki_notifications.json')
    base, work = args.out / 'base', args.out / 'run'
    subprocess.run([sys.executable, str(ROOT / 'research/e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec/experiment.py'),
                    '--data', args.data, '--observations', args.observations, '--out', str(base),
                    '--device', 'cpu', '--threads', str(args.threads), '--donki', str(donki)],
                   check=True, env=env, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / 'research/e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam/experiment.py'), '--base', str(base),
                    '--data', args.data, '--observations', args.observations, '--out', str(work),
                    '--threads', str(args.threads), '--donki', str(donki)],
                   check=True, env=env, cwd=ROOT)
    shutil.copy2(work / 'results/mean3_predictions.parquet', args.out / 'predictions.parquet')
