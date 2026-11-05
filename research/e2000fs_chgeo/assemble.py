"""Assemble the nine per-block predictions of one arm into the 454,680-row official forecast.

usage: python assemble.py --out DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
INPUT = HERE / 'inputs/input_df.parquet'
SPEED = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store/e2000fs-lag0/speed.parquet')
H = 72
STAGES = ('F1', 'F2', 'F3')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--arms', default='control,chgeo')
    a = ap.parse_args()
    series = pd.read_parquet(SPEED)['speed']
    keys = pd.read_parquet(INPUT, columns=['set', 'origin', 'period'])
    summary = {}
    for arm in a.arms.split(','):
        rows, scores, pairs = [], {}, []
        for stage in STAGES:
            parts = sorted(a.out.glob(f'part_{arm}_{stage}_b*.npy'))
            if len(parts) != 3:
                raise FileNotFoundError(f'{arm} {stage}: {len(parts)} parts')
            origins = pd.DatetimeIndex(sorted(keys[(keys['set'] == 'query') & (keys['period'] == stage)]['origin'].unique()))
            pred = np.hstack([np.load(p) for p in parts])
            if pred.shape != (len(origins), H):
                raise ValueError(f'{arm} {stage}: {pred.shape} vs {(len(origins), H)}')
            t = (origins.values[:, None] + np.arange(1, H + 1).astype('timedelta64[h]')[None, :]).ravel()
            y = series.reindex(pd.DatetimeIndex(t, tz='UTC')).to_numpy().reshape(len(origins), H)
            if np.isnan(y).any():
                raise ValueError(f'{arm} {stage}: unobserved truth')
            err, tail = (pred - y) ** 2, y >= 550
            scores[stage] = dict(mse=float(err.mean()), mse_ge550=float(err[tail].mean()),
                                 bias_ge550=float((pred - y)[tail].mean()), n_ge550=int(tail.sum()),
                                 origins=int(len(origins)))
            pairs.append((pred.ravel(), y.ravel()))
            rows.append(pd.DataFrame({'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), H),
                                      'horizon_hours': np.tile(np.arange(1, H + 1), len(origins)),
                                      'pred_kms': pred.ravel()}))
        frame = pd.concat(rows, ignore_index=True)
        if len(frame) != 454680:
            raise ValueError(f'{arm}: {len(frame)} rows')
        frame.to_parquet(a.out / f'predictions_{arm}.parquet', index=False)
        p, y = np.concatenate([q[0] for q in pairs]), np.concatenate([q[1] for q in pairs])
        m = y >= 550
        if len(p) != 454680:
            raise ValueError('pooled rows')
        scores['pooled'] = dict(mse=float(np.mean((p - y) ** 2)), mse_ge550=float(np.mean((p - y)[m] ** 2)),
                                bias_ge550=float(np.mean((p - y)[m])), n_ge550=int(m.sum()))
        summary[arm] = scores
        (a.out / f'scores_{arm}.json').write_text(json.dumps(scores, indent=1) + '\n')
        print(arm, json.dumps(scores['pooled']), flush=True)
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')


if __name__ == '__main__':
    main()
