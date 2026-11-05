"""Reproducible replay for e2000fs_calib10_xgboostlss.

Reads the archived official forecast of e2000fs_calib10_xgboostlss (latent-regime log-normal mixture fitted with
the XGBoostLSS scheme of Chevalier & Cote 2025, Algorithm 4) from
research/e2000fs_calib10_xgboostlss/results/predictions.parquet; config_sha256 1ef3ca59dc083219.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

SRC = Path("/home/t-lab01/sunrun/research/e2000fs_calib10_xgboostlss/results/predictions.parquet")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    frame = pd.read_parquet(SRC)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "predictions.parquet", index=False)
    print(len(frame), "rows ->", out / "predictions.parquet")


if __name__ == "__main__":
    main()
