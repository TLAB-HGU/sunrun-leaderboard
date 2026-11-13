"""naive — the last observed hourly speed held for 72 hours

Leaderboard baseline (member baseline). Forecast for every lead = the forward-filled
speed at the origin hour.

This one file runs the whole baseline end to end:
  1. data   NASA ACE SWEPAM hourly bulk speed from data-v2/raw/ACE/monthly-txt
            (every record with a valid speed, any status), placed on an hourly grid and
            forward filled; the official query origins come from data-v2/input/df.parquet
  2. fit    nothing to fit: the forecast is the value at the origin
  3. infer  a 72-hour forecast for every query origin (F1, F2, F3)
  4. score  official scorer leaderboard/scorer/competition.py; labels are used only here

There are no trained weights: the forecast is a fixed formula of the input series.
Outputs (research/result/naive/ unless --out is given; runner research/module/baseline.py):
  results/predictions.parquet, results/score.json
  logs/run.jsonl   every step, one JSON object per line

Usage:
  python experiment.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from module import baseline  # noqa: E402

EXPERIMENT = 'naive'
HORIZON_HOURS = 72
HYPERPARAMETERS = {'horizon_hours': HORIZON_HOURS, 'strategy': 'last_value'}


def forecast(speed, origins, args):
    values = speed.to_numpy(np.float64)
    index = baseline.positions(speed, origins, 1)
    return np.repeat(values[index][:, None], HORIZON_HOURS, axis=1), {}


if __name__ == '__main__':
    baseline.main(EXPERIMENT, HYPERPARAMETERS, forecast, __doc__)
