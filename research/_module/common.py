"""Run plumbing shared by every board row: output paths, logs, official scoring and result files."""
from __future__ import annotations

import importlib.util
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH = Path(__file__).resolve().parents[1]
ROOT = RESEARCH.parent
RESULT = RESEARCH / 'result'
DATA = ROOT / 'data-v2/input/df.parquet'
H = 72
SEEDS = (42, 43, 44)
CUTOFFS = {'F1': '2025-12-29', 'F2': '2026-03-29', 'F3': '2026-06-29'}
QUERY_COUNTS = {'F1': 2089, 'F2': 2113, 'F3': 2113}
KEYS = ['origin_last_input_utc', 'horizon_hours']


def output_paths(args, here, experiment):
    """Board run (no --out): results/ and logs/ go to research/result/{experiment}, weights stay in the
    experiment folder. With --out DIR (campaign runs, replay scripts): DIR/results, DIR/logs, DIR/weights."""
    board = args.out is None
    args.out = (RESULT / experiment if board else args.out).resolve()
    if hasattr(args, 'weights'):
        args.weights = (args.weights or (here / 'weights' if board else args.out / 'weights')).resolve()
    return board


class Logs:
    """Append-only JSON-lines logs, one file per stream (run, data, s42, ...), echoed to stdout."""

    def __init__(self, folder, run_id, experiment):
        self.folder, self.run_id, self.experiment = Path(folder), run_id, experiment
        self.folder.mkdir(parents=True, exist_ok=True)

    def __call__(self, stream, event, **fields):
        record = dict(ts=datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                      run_id=self.run_id, experiment=self.experiment, event=event, **fields)
        line = json.dumps(record, allow_nan=False, default=str)
        with (self.folder / f'{stream}.jsonl').open('a') as handle:
            handle.write(line + '\n')
        print(line, flush=True)


def run_logs(args, experiment):
    return Logs(args.out / 'logs', datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'), experiment)


def git_commit():
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
                          check=False).stdout.strip()


def versions(**extra):
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, **extra)


def launch(logs, args, **fields):
    logs('run', 'launch', host=socket.gethostname(), argv=vars(args), git_commit=git_commit(), **fields)


def file_identity(path):
    stat = Path(path).stat()
    return dict(path=str(Path(path).resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


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


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_results(folder, tag, frame, scores):
    """results/{tag}_predictions.parquet and, when scored, results/{tag}_score.json."""
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / f'{tag}_predictions.parquet', index=False)
    if scores is not None:
        write_json(folder / f'{tag}_score.json', scores)


def forecast_frame(origins, pred):
    """Long competition frame (origin, horizon, pred_kms) of one (origins x 72) forecast matrix."""
    return pd.DataFrame({KEYS[0]: np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), H),
                         KEYS[1]: np.tile(np.arange(1, H + 1), len(origins)), 'pred_kms': pred.ravel()})


def prediction_frame(parts):
    frames = []
    for period, (origins, pred) in parts.items():
        if origins.has_duplicates or not origins.is_monotonic_increasing or not np.isfinite(pred).all():
            raise ValueError(f'{period}: invalid prediction grid')
        frames.append(forecast_frame(origins, pred))
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(KEYS).any():
        raise ValueError('Duplicate prediction keys')
    return frame


def mean3(frames):
    """Fixed equal-weight mean of the three seed forecasts on identical keys."""
    first = frames[0][KEYS]
    if any(not frame[KEYS].equals(first) for frame in frames):
        raise ValueError('All three seeds must have identical origin/horizon keys')
    matrix = np.stack([frame['pred_kms'].to_numpy(np.float64) for frame in frames])
    output = first.copy()
    output['pred_kms'] = matrix.mean(axis=0)
    return output
