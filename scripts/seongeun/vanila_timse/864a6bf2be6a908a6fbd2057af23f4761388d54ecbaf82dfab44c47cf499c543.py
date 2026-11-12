"""Reproduce vanila_timse: fresh pure-TIMSE models, fixed seeds 42/43/44 mean; CPU only.

Run from a checkout of this repository: the model is research/vanila_timse/experiment.py.
"""
import argparse, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--observations", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--threads", type=int, default=6)
    a = p.parse_args()
    work = a.out / "run"
    subprocess.run([sys.executable, str(ROOT / "research/vanila_timse/experiment.py"), "--data", a.data,
                    "--observations", a.observations, "--out", str(work), "--device", "cpu",
                    "--threads", str(a.threads)], check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), cwd=ROOT)
    shutil.copy(work / "results/mean3_predictions.parquet", a.out / "predictions.parquet")
