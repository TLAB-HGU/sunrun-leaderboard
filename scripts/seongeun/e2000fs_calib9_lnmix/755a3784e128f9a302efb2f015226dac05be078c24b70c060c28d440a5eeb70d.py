"""Reproducible replay for e2000fs_calib9_lnmix.

Latent-regime log-normal mixture fitted with XGBoostLSS (Chevalier & Cote 2025, Eur. Actuar. J. 10.1007/s13385-025-00428-5,
Algorithm 4): the 3K-1 parameter groups are updated cyclically for two cycles of 450 Newton boosting rounds each. The
official forecast is the mixture expectation sum_k pi_k exp(mu_k + sigma_k^2/2) -- no threshold, no sample weight and no
clip anywhere in training or prediction. Architecture: research/e2000fs_calib9_lnmix/experiment.py at commit 565d53f.
Predictions are archived in research/e2000fs_calib9_lnmix/results/{F1,F2,F3}_predictions.npz and are replayed here.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

RUN = Path("/home/t-lab01/sunrun/research/e2000fs_calib9_lnmix/results")
H = 72


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    parts = []
    for period in ("F1", "F2", "F3"):
        z = np.load(RUN / f"{period}_predictions.npz")
        o = pd.DatetimeIndex(z["origins"], tz="UTC")
        pred = z["pred"].astype(np.float64)
        parts.append(pd.DataFrame({"origin_last_input_utc": np.repeat(o.strftime("%Y-%m-%dT%H:%M:%SZ"), H),
                                   "horizon_hours": np.tile(np.arange(1, H + 1), len(o)),
                                   "pred_kms": pred.ravel()}))
    frame = pd.concat(parts, ignore_index=True).sort_values(["origin_last_input_utc", "horizon_hours"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "predictions.parquet", index=False)
    print(len(frame), "rows ->", out / "predictions.parquet")


if __name__ == "__main__":
    main()
