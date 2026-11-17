"""Reproduce vanila_mse_l27: fresh pure MSE models on v_lag0 + rec27 only, fixed seeds 42/43/44 mean."""
import argparse, os, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--observations', required=True)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--threads', type=int, default=6)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    work = a.out / 'run'
    subprocess.run([sys.executable, str(ROOT / 'research/vanila_mse_l27/experiment.py'), '--data', a.data,
                    '--observations', a.observations, '--out', str(work), '--threads', str(a.threads)],
                   check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), cwd=ROOT)
    shutil.copy(work / 'results/mean3_predictions.parquet', a.out / 'predictions.parquet')
