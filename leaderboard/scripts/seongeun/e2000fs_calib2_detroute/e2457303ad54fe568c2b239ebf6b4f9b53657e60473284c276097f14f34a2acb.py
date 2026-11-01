"""Reproducible replay for e2000fs_calib2_detroute.

Causal routing between two archived arms using a dedicated exceedance classifier.
At inference each (origin, horizon) pair is routed to the mixture2comp arm when the archived
detector probability is >= 0.234 (fixed a priori from the cost ratio) and to the tweedie15 arm
otherwise. The detector was trained only on origins satisfying origin+144h<=cutoff.
Archived artefacts: research/e2000fs_calib2-20261031/runs/detroute_v1 (detector_*.ubj,
exceedance_prob.npz, results.json) and the two arm forecasts.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/home/t-lab01/sunrun')
RUN = ROOT / 'research/e2000fs_calib2-20261031/runs/detroute_v1'
BODY = ROOT / 'research/e2000fs_calib-20261030/runs/tweedie15_v1/forecast.npz'
TAIL = ROOT / 'research/e2000fs_calib2-20261031/runs/mixture_v1/forecast.npz'
H, TAU = 72, 0.234


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    z = np.load(RUN / 'forecast.npz')
    origins = pd.DatetimeIndex(z['origins_ns'], tz='UTC')
    pred = z['predictions'].astype(np.float64)
    prob = np.load(RUN / 'exceedance_prob.npz')['prob']
    body = np.load(BODY)['predictions'].astype(np.float64)
    tail = np.load(TAIL)['predictions'].astype(np.float64)
    replay = np.where(prob >= TAU, tail, body)
    assert np.allclose(replay, pred, atol=1e-9), 'routed replay mismatch'
    frame = pd.DataFrame({'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), H),
                          'horizon_hours': np.tile(np.arange(1, H + 1), len(origins)),
                          'pred_kms': pred.ravel()})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / 'predictions.parquet', index=False)
    print(len(frame), 'rows ->', out / 'predictions.parquet')


if __name__ == '__main__':
    main()
