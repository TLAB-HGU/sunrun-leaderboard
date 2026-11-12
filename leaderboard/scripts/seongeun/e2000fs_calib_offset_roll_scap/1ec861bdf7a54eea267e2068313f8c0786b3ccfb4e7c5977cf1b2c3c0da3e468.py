"""Reproduce e2000fs_calib_offset_roll_scap: train seeds 42/43/44 (A900 point4 + spectral + 180 direct-S continuation trees under a 1.01 x training-MSE cap) and write their fixed mean; CPU only.

Run from a checkout of this repository: the model is research/e2000fs_calib_offset_roll_scap/experiment.py.
"""
import argparse, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    work = args.out / "run"
    subprocess.run([sys.executable, str(ROOT / "research/e2000fs_calib_offset_roll_scap/experiment.py"),
                    "--data", args.data, "--observations", args.observations, "--out", str(work),
                    "--threads", str(args.threads)], check=True, env=env, cwd=ROOT)
    shutil.copy(work / "results/mean3/predictions.parquet", args.out / "predictions.parquet")
