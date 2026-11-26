"""Reproduce the fianl_v2_hpo_t14 forecast from data-v2/input/df.parquet.

usage: python <this file> --data df.parquet --out DIR [--threads 8]
  1. research/fianl_v2_hpo/experiment.py trains seeds 42/43/44 for F1-F3 and writes unscored raw seed forecasts
     to DIR/run/results/s{seed}_predictions.parquet;
  2. per seed seam xf8 -> 3 h phase advance (experiment.seam_smooth / phase_advance), then the equal-weight mean of
     the three seeds -> DIR/predictions.parquet.
"""
import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

experiment = Path(__file__).resolve().parents[3] / "research/fianl_v2_hpo/experiment.py"
ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
subprocess.run([sys.executable, experiment, "--threads", str(a.threads), "--data", a.data, "--out", a.out / "run"],
               check=True, env=env)
spec = importlib.util.spec_from_file_location("fianl_v2_hpo", experiment)
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
frames = [pd.read_parquet(a.out / "run/results" / f"s{seed}_predictions.parquet").sort_values(E.KEYS, kind="stable")
          .reset_index(drop=True) for seed in E.SEEDS]
keys = frames[0][E.KEYS]
if any(not f[E.KEYS].equals(keys) for f in frames):
    raise ValueError("seed forecasts must share origin/horizon keys")
final = [E.phase_advance(E.seam_smooth(f.pred_kms.to_numpy(np.float64).reshape(-1, E.H))) for f in frames]
result = keys.copy()
result["pred_kms"] = np.mean(final, axis=0).ravel()
if not np.isfinite(result.pred_kms).all():
    raise ValueError("nonfinite forecast")
result.to_parquet(a.out / "predictions.parquet", index=False)
print(a.out / "predictions.parquet", len(result))
