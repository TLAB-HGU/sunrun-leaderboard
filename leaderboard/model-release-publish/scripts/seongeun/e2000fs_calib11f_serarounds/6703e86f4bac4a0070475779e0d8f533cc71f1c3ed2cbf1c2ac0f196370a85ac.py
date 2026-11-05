"""Reproducible replay for e2000fs_calib11f_serarounds (config_sha256 6703e86f4bac4a00).

Re-emits the archived official forecast of research/e2000fs_calib11f_serarounds/results/predictions.parquet.
The model itself is research/e2000fs_calib11f_serarounds/experiment.py (architecture link in meta.yaml).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

SRC = Path("/home/t-lab01/sunrun/research/e2000fs_calib11f_serarounds/results/predictions.parquet")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    frame = pd.read_parquet(SRC)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "predictions.parquet", index=False)
    print(len(frame), "rows ->", out / "predictions.parquet")


if __name__ == "__main__":
    main()
