"""Reproduce e2000fs_calib_offset_roll_scap_arrival_cme: the unchanged S-cap recipe plus 8 DONKI arrival / CME-kinematics
origin constants, seeds 42/43/44, and their fixed mean; CPU only.

Run from a checkout of this repository: the model is research/e2000fs_calib_offset_roll_scap_arrival_cme/experiment.py.
Without --donki the CME notifications are downloaded from the CCMC DONKI API.
"""
import argparse, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--donki", help="saved DONKI CME notifications JSON (default: download)")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    work = args.out / "run"
    subprocess.run([sys.executable, str(ROOT / "research/e2000fs_calib_offset_roll_scap_arrival_cme/experiment.py"),
                    "--data", args.data, "--observations", args.observations, "--out", str(work),
                    "--threads", str(args.threads), *(["--donki", args.donki] if args.donki else [])],
                   check=True, env=env, cwd=ROOT)
    shutil.copy(work / "results/mean3_predictions.parquet", args.out / "predictions.parquet")
