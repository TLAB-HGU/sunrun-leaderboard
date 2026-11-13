"""ses — optimized simple exponential smoothing refitted on the last 648 hours

Leaderboard baseline (member baseline). statsforecast SimpleExponentialSmoothingOptimized is
refitted for every origin on the 648 hours ending at it. The published forecast is exact
with leaderboard/requirements-baselines.txt (statsforecast 2.0.3, numba 0.67.0).

This one file runs the whole baseline end to end:
  1. data   NASA ACE SWEPAM hourly bulk speed from data-v2/raw/ACE/monthly-txt
            (every record with a valid speed, any status), placed on an hourly grid and
            forward filled; the official query origins come from data-v2/input/df.parquet
  2. fit    SimpleExponentialSmoothingOptimized on the 648 hours ending at each origin
  3. infer  a 72-hour forecast for every query origin (F1, F2, F3)
  4. score  official scorer leaderboard/scorer/competition.py; labels are used only here

There are no trained weights: the model is refitted at every origin from the input series.
Outputs (research/result/ses/ unless --out is given; runner research/module/baseline.py):
  results/predictions.parquet, results/score.json
  logs/run.jsonl   every step, one JSON object per line

Usage:
  python experiment.py --workers 8
"""
from __future__ import annotations

import concurrent.futures
import os
import sys
import time
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np

from statsforecast.models import SimpleExponentialSmoothingOptimized

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from module import baseline  # noqa: E402

EXPERIMENT = 'ses'
HORIZON_HOURS = 72
TRAINING_WINDOW_HOURS = 648
HYPERPARAMETERS = {'alpha_bounds': [0.01, 0.99], 'horizon_hours': HORIZON_HOURS,
                   'model': 'statsforecast.SimpleExponentialSmoothingOptimized',
                   'objective': 'training_one_step_squared_error',
                   'training_window_hours': TRAINING_WINDOW_HOURS}


def fit_predict(item):
    index, history = item
    fit_started = time.perf_counter()
    model = SimpleExponentialSmoothingOptimized().fit(history)
    fit_seconds = time.perf_counter() - fit_started
    inference_started = time.perf_counter()
    prediction = np.asarray(model.predict(h=HORIZON_HOURS)['mean'])
    return index, prediction, fit_seconds, time.perf_counter() - inference_started


def forecast(speed, origins, args):
    """Refit on the last TRAINING_WINDOW_HOURS hours ending at each origin, in worker processes."""
    values = speed.to_numpy(np.float64)
    index = baseline.positions(speed, origins, TRAINING_WINDOW_HOURS)
    jobs = [(i, values[p - TRAINING_WINDOW_HOURS + 1:p + 1].copy()) for i, p in enumerate(index)]
    fit_predict(jobs[0])  # warm JIT before worker processes are forked
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(fit_predict, jobs, chunksize=8))
    predictions = np.empty((len(origins), HORIZON_HOURS), dtype=float)
    fit_seconds = inference_seconds = 0.
    for index, prediction, fit_elapsed, inference_elapsed in results:
        predictions[index] = prediction
        fit_seconds += fit_elapsed
        inference_seconds += inference_elapsed
    return predictions, dict(workers=args.workers, fit_seconds_per_fold=fit_seconds / len(origins),
                             inference_seconds_per_fold=inference_seconds / len(origins))


if __name__ == '__main__':
    baseline.main(EXPERIMENT, HYPERPARAMETERS, forecast, __doc__, workers=8, statsforecast=__import__('statsforecast').__version__)
