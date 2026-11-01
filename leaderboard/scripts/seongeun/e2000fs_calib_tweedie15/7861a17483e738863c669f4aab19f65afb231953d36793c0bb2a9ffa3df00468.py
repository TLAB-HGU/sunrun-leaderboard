"""Reproducible replay for e2000fs_calib_tweedie15.

Tweedie deviance (variance power 1.5) with no sample weights, replacing the 4x high-speed
weighted squared error of e2000fs_dirmo_24_feat30_pointweight4. Three 24-output vector trees,
frozen Top30 inputs (237 columns per block), 900 rounds, depth 3, eta 0.03, seed 42, and the
training mask origin+144h<=cutoff so every 72h training vector is matured at the cutoff.
The deviance is multiplied by (500 km/s)**1.5, a constant, so gradients keep O(1) scale and the
parent regularisation strength keeps its meaning; a constant factor does not move the optimum.
Training code, models and per-round losses are archived in
research/e2000fs_calib-20261030/runs/tweedie15_v1 (tweedie_lane.py, model_tweedie_*.ubj,
losses_tweedie_*.csv). This script replays the archived official F1+F2 predictions.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/home/t-lab01/sunrun')
RUN = ROOT / 'research/e2000fs_calib-20261030/runs/tweedie15_v1'
H = 72


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    f1 = np.load(RUN / 'tweedie_f1_predictions.npz')
    fin = np.load(RUN / 'tweedie_final_predictions.npz')
    origins = pd.DatetimeIndex(f1['origins'], tz='UTC').append(pd.DatetimeIndex(fin['origins'], tz='UTC'))
    pred = np.vstack([f1['pred'], fin['pred']]).astype(np.float64)
    assert pred.shape == (len(origins), H)
    frame = pd.DataFrame({'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), H),
                          'horizon_hours': np.tile(np.arange(1, H + 1), len(origins)),
                          'pred_kms': pred.ravel()})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / 'predictions.parquet', index=False)
    print(len(frame), 'rows ->', out / 'predictions.parquet')


if __name__ == '__main__':
    main()
