"""Reproducible replay for e2000fs_calib_point4_144h.

Paired control arm of the e2000fs_calib_tweedie15 lane: the point4 direct-24 recipe with training rows restricted to origin+144h<=cutoff so every 72 h training vector is fully matured. Training code, models and per-round losses are preserved in research/e2000fs_calib-20261030/runs/tweedie15_v1.

Inputs are archived, hash-pinned artefacts of the sunrun monorepo; nothing is retrained.
Run:  python e2000fs_calib_point4_144h.py --out /tmp/replay
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/home/t-lab01/sunrun')
SPEED = ROOT / 'data-v2/model-inputs/leaderboard-store/e2000fs-lag0/speed.parquet'
H = 72

RUN = ROOT / 'research/e2000fs_calib-20261030/runs/tweedie15_v1'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    f1 = np.load(RUN / 'control_f1_predictions.npz')
    fin = np.load(RUN / 'control_final_predictions.npz')
    origins = pd.DatetimeIndex(f1['origins'], tz='UTC').append(pd.DatetimeIndex(fin['origins'], tz='UTC'))
    pred = np.vstack([f1['pred'], fin['pred']]).astype(np.float64)
    frame = pd.DataFrame({'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), H),
                          'horizon_hours': np.tile(np.arange(1, H + 1), len(origins)),
                          'pred_kms': pred.ravel()})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / 'predictions.parquet', index=False)
    print(len(frame), 'rows ->', out / 'predictions.parquet')


if __name__ == '__main__':
    main()
