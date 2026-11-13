"""seasonal_naive_648 — the speed 648 hours (27 days, one solar rotation) earlier

Leaderboard baseline (member baseline). Forecast for lead h = the forward-filled speed at
origin + h - 648 hours.

This one file runs the whole baseline end to end:
  1. data   NASA ACE SWEPAM hourly bulk speed from data-v2/raw/ACE/monthly-txt
            (every record with a valid speed, any status), placed on an hourly grid and
            forward filled; the official query origins come from data-v2/input/df.parquet
  2. fit    nothing to fit: each lead copies the value one 648-hour season earlier
  3. infer  a 72-hour forecast for every query origin (F1, F2, F3)
  4. score  official scorer leaderboard/scorer/competition.py; labels are used only here

There are no trained weights: the forecast is a fixed formula of the input series.
Outputs (research/result/seasonal_naive_648/ unless --out is given; runner research/module/baseline.py):
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

EXPERIMENT = 'seasonal_naive_648'
HORIZON_HOURS = 72
SEASON_LENGTH_HOURS = 648
HYPERPARAMETERS = {'horizon_hours': HORIZON_HOURS, 'season_length_hours': SEASON_LENGTH_HOURS, 'strategy': 'seasonal_naive'}


def forecast(speed, origins, args):
    values = speed.to_numpy(np.float64)
    index = baseline.positions(speed, origins, SEASON_LENGTH_HOURS)
    horizons = np.arange(1, HORIZON_HOURS + 1)
    return values[index[:, None] + horizons[None, :] - SEASON_LENGTH_HOURS], {}


if __name__ == '__main__':
    baseline.main(EXPERIMENT, HYPERPARAMETERS, forecast, __doc__)
