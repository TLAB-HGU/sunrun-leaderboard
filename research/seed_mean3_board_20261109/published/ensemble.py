"""Fixed equal-weight mean of three seed forecasts (42/43/44); no label-fitted weights.

Used for every 3-seed leaderboard row: the board shows seed means, single-seed rows are hidden.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ['origin_last_input_utc', 'horizon_hours']
SEEDS = [42, 43, 44]


def average_predictions(frames):
    if len(frames) != 3:
        raise ValueError('Exactly three seed prediction frames are required')
    aligned = []
    for frame in frames:
        frame = frame[KEYS + ['pred_kms']].copy()
        frame[KEYS[0]] = pd.to_datetime(frame[KEYS[0]], utc=True, errors='raise')
        if frame[KEYS[0]].isna().any() or frame.duplicated(KEYS).any():
            raise ValueError('Missing origins or duplicate prediction keys')
        if not frame[KEYS[1]].isin(range(1, 73)).all():
            raise ValueError('Horizons must be integers 1..72')
        frame = frame.sort_values(KEYS).reset_index(drop=True)
        values = frame['pred_kms'].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not (values > 0).all():
            raise ValueError('Predictions must be finite and positive')
        if aligned and not frame[KEYS].equals(aligned[0][KEYS]):
            raise ValueError('All three seeds must have exactly identical origin/horizon keys')
        aligned.append(frame)
    matrix = np.stack([f['pred_kms'].to_numpy(dtype=np.float64) for f in aligned])
    started = time.perf_counter()
    prediction = matrix.mean(axis=0)
    seconds = time.perf_counter() - started
    output = aligned[0][KEYS].copy()
    output[KEYS[0]] = output[KEYS[0]].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    output['pred_kms'] = prediction
    return output, seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', nargs=3, required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    output, seconds = average_predictions([pd.read_parquet(p) for p in args.predictions])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.out, index=False)
    metadata = {'seeds': SEEDS, 'weights': [1 / 3] * 3, 'rows': len(output),
                    'origins': output[KEYS[0]].nunique(), 'keys_identical': True,
                    'prediction_mean_seconds': seconds,
                    'prediction_mean_seconds_per_fold': seconds / output[KEYS[0]].nunique(),
                    'policy': 'fixed equal weights; no labels or fitted weights'}
    args.out.with_suffix('.ensemble.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
