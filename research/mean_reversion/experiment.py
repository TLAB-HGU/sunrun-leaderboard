"""mean_reversion — exponential decay from the last value to the 648-hour mean

Leaderboard baseline (member baseline). Forecast for lead h = mean + (last - mean) * exp(-h/48),
where mean is the 648-hour rolling mean ending at the origin.

This one file runs the whole baseline end to end:
  1. data   NASA ACE SWEPAM hourly bulk speed from data-v2/raw/ACE/monthly-txt
            (every record with a valid speed, any status), placed on an hourly grid and
            forward filled; the official query origins come from data-v2/input/df.parquet
  2. fit    the 648-hour rolling mean of the series ending at each origin
  3. infer  a 72-hour forecast for every query origin (F1, F2, F3)
  4. score  official scorer leaderboard/scorer/competition.py; labels are used only here

There are no trained weights: the forecast is a fixed formula of the input series.
Outputs (research/result/mean_reversion/ unless --out is given; runner research/_module/baseline.py):
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
from _module import baseline  # noqa: E402

EXPERIMENT = 'mean_reversion'
HORIZON_HOURS = 72
MEAN_WINDOW_HOURS = 648
TAU_HOURS = 48
HYPERPARAMETERS = {'horizon_hours': HORIZON_HOURS, 'mean_window_hours': MEAN_WINDOW_HOURS, 'tau_hours': TAU_HOURS}


def forecast(speed, origins, args):
    index = baseline.positions(speed, origins, MEAN_WINDOW_HOURS)
    mean = speed.rolling(MEAN_WINDOW_HOURS, min_periods=MEAN_WINDOW_HOURS).mean().loc[origins].to_numpy()
    last = speed.loc[origins].to_numpy()
    decay = np.exp(-np.arange(1, HORIZON_HOURS + 1) / TAU_HOURS)
    predictions = np.empty((len(origins), HORIZON_HOURS), dtype=float)
    for i in range(len(origins)):
        predictions[i] = mean[i] + (last[i] - mean[i]) * decay
    return predictions, {}


if __name__ == '__main__':
    baseline.main(EXPERIMENT, HYPERPARAMETERS, forecast, __doc__)
