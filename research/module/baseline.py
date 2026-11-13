"""ACE SWEPAM statistical baselines (member baseline): hourly speed series, query origins, runner.

An experiment.py defines forecast(speed, origins, args) -> (origins x 72 forecasts, log fields) and calls
main(EXPERIMENT, HYPERPARAMETERS, forecast, __doc__). Outputs: results/predictions.parquet,
results/score.json and logs/run.jsonl under research/result/<experiment> (or --out).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (DATA, H, ROOT, file_identity, forecast_frame, launch, observations_path, output_paths,
                     run_logs, score, versions, write_json)


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
    logs('run', 'data_ready', source=str(Path(folder).resolve()), files=len(files), records=len(raw),
         valid_records=int(valid.notna().sum()), hours=len(grid), start=str(grid.min()), end=str(grid.max()),
         leading_missing=int(speed.isna().sum()),
         policy='SWEPAM bulk speed, records with speed > 0 of any status, hourly grid, forward fill')
    return speed


def query_origins(path, logs):
    frame = pd.read_parquet(path, columns=['origin', 'period'], filters=[('set', '==', 'query'), ('h', '==', 1)])
    frame = frame[frame.period.isin(['F1', 'F2', 'F3'])]
    origins = pd.DatetimeIndex(pd.to_datetime(frame.origin, utc=True)).unique().sort_values()
    logs('run', 'origins_ready', source=file_identity(path), origins=len(origins), start=str(origins.min()),
         end=str(origins.max()), per_period=frame.groupby('period', observed=True).origin.nunique().to_dict())
    return origins


def positions(speed, origins, history):
    index = speed.index.get_indexer(origins)
    if (index < 0).any():
        raise ValueError('series does not contain every query origin')
    if (index + 1 - history < 0).any() or speed.iloc[index.min() - history + 1:].isna().any():
        raise ValueError(f'at least {history} filled hours are required before every origin')
    return index


def main(experiment, hyperparameters, forecast, doc, workers=None, **version_extra):
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', choices=[experiment], default=experiment)
    parser.add_argument('--swepam', type=Path, default=ROOT / 'data-v2/raw/ACE/monthly-txt')
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--observations', type=Path, help='default: the leaderboard dataset copy')
    parser.add_argument('--out', type=Path, help='run folder (default: research/result/<experiment>)')
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    if workers is not None:
        parser.add_argument('--workers', type=int, default=workers)
    args = parser.parse_args()
    output_paths(args, None, experiment)
    logs = run_logs(args, experiment)
    launch(logs, args, versions=versions(**version_extra), hyperparameters=hyperparameters)
    speed = load_speed(args.swepam, logs)
    origins = query_origins(args.data, logs)
    began = time.perf_counter()
    predictions, details = forecast(speed, origins, args)
    if predictions.shape != (len(origins), H) or not np.isfinite(predictions).all():
        raise ValueError('incomplete or non-finite forecasts')
    logs('run', 'forecast_complete', origins=len(origins), seconds=round(time.perf_counter() - began, 3),
         seconds_per_fold=(time.perf_counter() - began) / len(origins), **details)
    frame = forecast_frame(origins, predictions)
    scores = score(frame, observations_path(args.observations))
    results = args.out / 'results'
    results.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(results / 'predictions.parquet', index=False)
    write_json(results / 'score.json', scores)
    logs('run', 'complete', status='complete', rows=len(frame), total_s=scores['competition']['total']['s'],
         periods={p: scores['competition'][p]['s'] for p in ('f1', 'f2', 'f3')})
