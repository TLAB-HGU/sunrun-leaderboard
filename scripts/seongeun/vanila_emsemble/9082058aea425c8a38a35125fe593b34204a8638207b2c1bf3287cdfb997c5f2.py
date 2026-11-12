"""Reproduce vanila_emsemble: train vanila_mse and vanila_timse (seeds 42/43/44), then the fixed
50:50 per-seed average and equal seed mean; CPU only.

Run from a checkout of this repository: the averaging is research/vanila_emsemble/experiment.py.
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
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for name in ("vanila_mse", "vanila_timse"):
        subprocess.run([sys.executable, str(ROOT / f"research/{name}/experiment.py"), "--data", a.data,
                        "--observations", a.observations, "--out", str(a.out / name),
                        "--threads", str(a.threads)], check=True, env=env, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "research/vanila_emsemble/experiment.py"),
                    "--mse-root", str(a.out / "vanila_mse"), "--timse-root", str(a.out / "vanila_timse"),
                    "--out", str(a.out / "vanila_emsemble"), "--observations", a.observations],
                   check=True, env=env, cwd=ROOT)
    shutil.copy(a.out / "vanila_emsemble/results/mean3_predictions.parquet", a.out / "predictions.parquet")
