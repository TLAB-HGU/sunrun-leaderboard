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
Outputs (in this folder unless --out is given):
  results/predictions.parquet, results/score.json
  logs/run.jsonl   every step, one JSON object per line

Usage:
  python experiment.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import platform
import socket
import subprocess
import time

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = 'naive'
HORIZON_HOURS = 72
HYPERPARAMETERS = {'horizon_hours': HORIZON_HOURS, 'strategy': 'last_value'}
KEYS = ['origin_last_input_utc', 'horizon_hours']


class Logs:
    """Append-only JSON-lines log (logs/run.jsonl)."""

    def __init__(self, folder, run_id):
        self.path, self.run_id = Path(folder) / 'run.jsonl', run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event, **fields):
        record = dict(ts=datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                      run_id=self.run_id, experiment=EXPERIMENT, event=event, **fields)
        line = json.dumps(record, allow_nan=False, default=str)
        with self.path.open('a') as handle:
            handle.write(line + '\n')
        print(line, flush=True)


def file_identity(path):
    stat = Path(path).stat()
    return dict(path=str(Path(path).resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


# 1. data -----------------------------------------------------------------------------------

def load_speed(folder, logs):
    """Hourly bulk speed: every SWEPAM record with speed > 0 (any status), forward filled."""
    files = sorted(Path(folder).glob('*_ace_swepam_1h.txt'))
    if not files:
        raise FileNotFoundError(f'no SWEPAM monthly files in {folder}')
    times, speeds = [], []
    for path in files:
        for line in path.read_text().splitlines():
            if not line.strip() or line[0] in ':#':
                continue
            parts = line.split()
            times.append(f'{parts[0]}-{parts[1]}-{parts[2]}T{parts[3][:2]}:{parts[3][2:]}:00Z')
            speeds.append(float(parts[8]))
    raw = pd.Series(speeds, index=pd.DatetimeIndex(pd.to_datetime(times, utc=True)))
    raw = raw[~raw.index.duplicated(keep='last')].sort_index()
    valid = raw.where(raw > 0)
    grid = pd.date_range(raw.index.min(), raw.index.max(), freq='h')
    speed = valid.reindex(grid).ffill()
    logs('data_ready', source=str(Path(folder).resolve()), files=len(files), records=len(raw),
         valid_records=int(valid.notna().sum()), hours=len(grid), start=str(grid.min()), end=str(grid.max()),
         leading_missing=int(speed.isna().sum()),
         policy='SWEPAM bulk speed, records with speed > 0 of any status, hourly grid, forward fill')
    return speed


def query_origins(path, logs):
    frame = pd.read_parquet(path, columns=['origin', 'period'], filters=[('set', '==', 'query'), ('h', '==', 1)])
    frame = frame[frame.period.isin(['F1', 'F2', 'F3'])]
    origins = pd.DatetimeIndex(pd.to_datetime(frame.origin, utc=True)).unique().sort_values()
    logs('origins_ready', source=file_identity(path), origins=len(origins), start=str(origins.min()),
         end=str(origins.max()), per_period=frame.groupby('period', observed=True).origin.nunique().to_dict())
    return origins


# 2-3. fit and forecast ---------------------------------------------------------------------

def positions(speed, origins, history):
    index = speed.index.get_indexer(origins)
    if (index < 0).any():
        raise ValueError('series does not contain every query origin')
    if (index + 1 - history < 0).any() or speed.iloc[index.min() - history + 1:].isna().any():
        raise ValueError(f'at least {history} filled hours are required before every origin')
    return index


def forecast(speed, origins, args):
    values = speed.to_numpy(np.float64)
    index = positions(speed, origins, 1)
    return np.repeat(values[index][:, None], HORIZON_HOURS, axis=1), {}


# 4. score ----------------------------------------------------------------------------------

def score(frame, observations):
    spec = importlib.util.spec_from_file_location('competition', ROOT / 'leaderboard/scorer/competition.py')
    competition = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(competition)
    observed = competition.load_observations(observations)
    result = competition.score_competition(frame, competition.prepare_context(observed))
    if result['total']['n_replaced']:
        raise ValueError('Scorer replaced forecasts; the forecast grid is incomplete')
    return dict(competition=result, scorer_version=competition.VERSION,
                scorer=file_identity(spec.origin), observations=file_identity(observations))


def observations_path(path):
    if path is not None:
        return path
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download('tlabtlab/sunrun-lb-store', 'competition-observations.parquet',
                                repo_type='dataset'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', choices=[EXPERIMENT], default=EXPERIMENT)
    parser.add_argument('--swepam', type=Path, default=ROOT / 'data-v2/raw/ACE/monthly-txt')
    parser.add_argument('--data', type=Path, default=ROOT / 'data-v2/input/df.parquet')
    parser.add_argument('--observations', type=Path, help='default: the leaderboard dataset copy')
    parser.add_argument('--out', type=Path, default=HERE)
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    args = parser.parse_args()
    args.out = args.out.resolve()
    logs = Logs(args.out / 'logs', datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=False)
    logs('launch', host=socket.gethostname(), argv=vars(args), git_commit=git.stdout.strip(),
         versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
         hyperparameters=HYPERPARAMETERS)
    speed = load_speed(args.swepam, logs)
    origins = query_origins(args.data, logs)
    began = time.perf_counter()
    predictions, details = forecast(speed, origins, args)
    if predictions.shape != (len(origins), HORIZON_HOURS) or not np.isfinite(predictions).all():
        raise ValueError('incomplete or non-finite forecasts')
    logs('forecast_complete', origins=len(origins), seconds=round(time.perf_counter() - began, 3),
         seconds_per_fold=(time.perf_counter() - began) / len(origins), **details)
    frame = pd.DataFrame({KEYS[0]: np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), HORIZON_HOURS),
                          KEYS[1]: np.tile(np.arange(1, HORIZON_HOURS + 1), len(origins)),
                          'pred_kms': predictions.ravel()})
    scores = score(frame, observations_path(args.observations))
    results = args.out / 'results'
    results.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(results / 'predictions.parquet', index=False)
    (results / 'score.json').write_text(json.dumps(scores, indent=2, allow_nan=False) + '\n')
    logs('complete', status='complete', rows=len(frame), total_s=scores['competition']['total']['s'],
         periods={p: scores['competition'][p]['s'] for p in ('f1', 'f2', 'f3')})


if __name__ == '__main__':
    main()
