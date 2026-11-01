"""Reproducible replay for e2000fs_calib8_qxgb.

QXGB (arXiv:2608.15290): convolution-smoothed quantile regression fitted jointly over a dense quantile grid with
XGBoost multi-output (vector-leaf) trees; the official forecast is the predictive mean read out of the estimated CDF.
Predictions are archived in research/e2000fs_f123-20261031/runs/qxgb_v1 ({F1,F2,F3}_predictions.npz) and are replayed
here. Training code: research/e2000fs_f123-20261031/{common.py,arms.py,qxgb.py}; config sha256 291c9fd5f23bc5a3.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

RUN = Path("/home/t-lab01/sunrun/research/e2000fs_f123-20261031/runs/qxgb_v1")
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
