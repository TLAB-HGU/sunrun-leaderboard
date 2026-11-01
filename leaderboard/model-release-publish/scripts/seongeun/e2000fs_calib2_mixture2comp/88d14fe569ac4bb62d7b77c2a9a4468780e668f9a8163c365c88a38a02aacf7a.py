"""Reproducible replay for e2000fs_calib2_mixture2comp.

Same direct 72h three-vector-tree recipe and point4 objective as the paired control, with
level + P(exceedance) + gap two-component construction. Training rows satisfy origin+144h<=cutoff. Training code, per-round losses
and predictions are archived in research/e2000fs_calib2-20261031/runs/mixture_v1
(lanes2.py lane_mixture, losses_mixture2comp_*.csv). This script replays the archived
official F1+F2 predictions.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/home/t-lab01/sunrun')
RUN = ROOT / 'research/e2000fs_calib2-20261031/runs/mixture_v1'
H = 72


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    z = np.load(RUN / 'forecast.npz')
    origins = pd.DatetimeIndex(z['origins_ns'], tz='UTC')
    pred = z['predictions'].astype(np.float64)
    frame = pd.DataFrame({'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), H),
                          'horizon_hours': np.tile(np.arange(1, H + 1), len(origins)),
                          'pred_kms': pred.ravel()})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / 'predictions.parquet', index=False)
    print(len(frame), 'rows ->', out / 'predictions.parquet')


if __name__ == '__main__':
    main()
