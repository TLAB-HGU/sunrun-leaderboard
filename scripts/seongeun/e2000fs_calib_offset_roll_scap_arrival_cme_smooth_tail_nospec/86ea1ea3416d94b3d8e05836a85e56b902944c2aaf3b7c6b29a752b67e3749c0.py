"""Reproduce fixed seeds42/43/44 mean3: Arrival-CME + smooth-tail A900 + spectral-free S-cap180.
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
    parser.add_argument('--donki', help='Saved DONKI notifications; otherwise download from CCMC')
    parser.add_argument('--threads', type=int, default=6)
    args = parser.parse_args()
    work = args.out / 'run'
    subprocess.run([sys.executable, str(ROOT / 'research/e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec/experiment.py'),
                    '--data', args.data, '--observations', args.observations, '--out', str(work),
                    '--device', 'cpu', '--threads', str(args.threads),
                    *(['--donki', args.donki] if args.donki else [])],
                   check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), cwd=ROOT)
    shutil.copy2(work / 'results/mean3_predictions.parquet', args.out / 'predictions.parquet')
