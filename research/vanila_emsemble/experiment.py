"""Fixed 50:50 average of separately trained MSE and TIMSE cohorts, then mean3.

CPU only. Input roots contain flat results/ and logs/ artifacts for seeds 42/43/44.
Official observations are loaded only after all four prediction frames are assembled.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'leaderboard/scorer').is_dir())
EXPERIMENT = 'vanila_emsemble'
SEEDS = (42, 43, 44)
KEYS = ['origin_last_input_utc', 'horizon_hours']
CUTOFFS = {'F1': '2025-12-29', 'F2': '2026-03-29', 'F3': '2026-06-29'}
MATCH_FIELDS = ('rounds', 'block', 'feature_count', 'expanded_feature_count',
                'constant_features', 'varying_features', 'expanded_features_by_block',
                'training_cutoff', 'maturity_hours', 'train_origins', 'train_start',
                'train_end', 'query_origins', 'query_start', 'query_end', 'parameters',
                'input_file', 'policy')
CALIBRATION_FIELDS = ('policy', 'window', 'alpha', 'scope', 'applied', 'purge_hours',
                      'reference_rows', 'calibration_rows', 'reference_start', 'reference_end',
                      'calibration_start', 'calibration_end', 'offset_rows', 'offset_start', 'offset_end')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def identity(path):
    stat = path.stat()
    return {'path': str(path.resolve()), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def event(out, name, **fields):
    line = json.dumps(dict(ts=datetime.now(timezone.utc).isoformat(), experiment=EXPERIMENT,
                           event=name, **fields), allow_nan=False)
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    with (out / 'logs/run.jsonl').open('a') as stream:
        stream.write(line + '\n')
    with (out / 'train.log').open('a') as stream:
        stdout, target = os.fstat(1), os.fstat(stream.fileno())
        if (stdout.st_dev, stdout.st_ino) != (target.st_dev, target.st_ino):
            stream.write(line + '\n')
    print(line, flush=True)


def validate_frame(frame):
    if frame.empty or not set([*KEYS, 'pred_kms']).issubset(frame.columns):
        raise ValueError('Nonempty prediction frame with both keys and pred_kms required')
    frame = frame[[*KEYS, 'pred_kms']].copy()
    origin = pd.to_datetime(frame[KEYS[0]], utc=True, errors='raise')
    lead = pd.to_numeric(frame[KEYS[1]], errors='raise')
    if origin.isna().any() or not origin.eq(origin.dt.floor('h')).all():
        raise ValueError('Origins must be finite hourly UTC timestamps')
    if not (lead.between(1, 72) & lead.eq(np.floor(lead))).all():
        raise ValueError('Horizon must be an integer in 1..72')
    frame[KEYS[0]] = origin.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    frame[KEYS[1]] = lead.astype(int)
    if frame.duplicated(KEYS).any():
        raise ValueError('Duplicate origin/horizon keys')
    values = frame.pred_kms.to_numpy(dtype=np.float64)
    if not (np.isfinite(values) & (values > 0)).all():
        raise ValueError('Predictions must be positive and finite')
    frame['pred_kms'] = values
    return frame.sort_values(KEYS, kind='stable').reset_index(drop=True)


def average(frames):
    if not frames or any(not f[KEYS].equals(frames[0][KEYS]) for f in frames):
        raise ValueError('Prediction origin/horizon keys must match exactly')
    result = frames[0][KEYS].copy()
    result['pred_kms'] = np.stack([f.pred_kms.to_numpy(np.float64) for f in frames]).mean(0)
    return result


def validate_metadata(left, right, period, seed):
    for model, metadata in [('vanila_mse', left), ('vanila_timse', right)]:
        if (metadata.get('experiment') != model or metadata.get('period') != period
                or metadata.get('seed') != seed or metadata.get('smoke') is not False
                or metadata.get('rounds') != 900 or metadata.get('training_cutoff') != CUTOFFS[period]):
            raise ValueError(f'{model} {period} seed{seed}: source identity/protocol mismatch')
    for key in MATCH_FIELDS:
        if key not in left or key not in right or left[key] != right[key]:
            raise ValueError(f'{period} seed{seed}: source metadata differs: {key}')
    for key in CALIBRATION_FIELDS:
        a, b = left.get('frozen_calibration', {}), right.get('frozen_calibration', {})
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError(f'{period} seed{seed}: calibration protocol differs: {key}')


def assemble(mse_root, timse_root):
    """Read model artifacts only; no observations or scores can influence averaging."""
    paired, sources, metadata_by_period = [], [], {p: [] for p in CUTOFFS}
    for seed in SEEDS:
        frames = []
        for root in (mse_root, timse_root):
            path = root / 'results' / f's{seed}_predictions.parquet'
            frame = validate_frame(pd.read_parquet(path))
            frames.append(frame)
            sources.append(dict(identity(path), rows=len(frame), columns=list(frame.columns),
                                origin_start=frame[KEYS[0]].min(), origin_end=frame[KEYS[0]].max()))
        for period in CUTOFFS:
            paths = [root / 'logs' / f'{period}_s{seed}_metadata.json' for root in (mse_root, timse_root)]
            left, right = [json.loads(p.read_text()) for p in paths]
            validate_metadata(left, right, period, seed)
            sources.extend(identity(p) for p in paths)
            metadata_by_period[period].append({'seed': seed, 'mse': left, 'timse': right})
        paired.append(average(frames))
    mean = average(paired)
    return {**{f's{s}': f for s, f in zip(SEEDS, paired)}, 'mean3': mean}, sources, metadata_by_period


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=[EXPERIMENT], default=EXPERIMENT)
    parser.add_argument('--mse-root', type=Path, default=ROOT / 'research/vanila_mse')
    parser.add_argument('--timse-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or args.threads < 1:
        raise ValueError('Require CUDA_VISIBLE_DEVICES="" and positive --threads')
    out = args.out.resolve()
    if out in (args.mse_root.resolve(), args.timse_root.resolve()):
        raise ValueError('Output must not overwrite either source experiment')
    if (out / 'results').exists() and any((out / 'results').iterdir()):
        raise ValueError('Refusing to overwrite existing results')
    event(out, 'assembly_start', period='F1', unit=0, units=4, device='cpu', threads=args.threads)
    frames, sources, metadata = assemble(args.mse_root, args.timse_root)
    (out / 'results').mkdir(parents=True, exist_ok=True)
    for tag, frame in frames.items():
        frame.to_parquet(out / 'results' / f'{tag}_predictions.parquet', index=False)
    provenance = dict(experiment=EXPERIMENT, source_weights=[.5, .5], seeds=list(SEEDS),
                      seed_weights=[1/3]*3, source_artifacts=sources,
                      git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
                      implementation=identity(Path(__file__)))
    write_json(out / 'logs/provenance.json', provenance)
    # Official truth is first accessed only after all predictions are frozen on disk.
    spec = importlib.util.spec_from_file_location('competition', ROOT / 'leaderboard/scorer/competition.py')
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    context = scorer.prepare_context(scorer.load_observations(args.observations))
    scores = {}
    for unit, (tag, frame) in enumerate(frames.items(), 1):
        scores[tag] = scorer.score_competition(frame, context)
        if scores[tag]['total']['n_replaced']:
            raise ValueError('Official scorer replaced missing predictions')
        record = dict(competition=scores[tag], scorer_version=scorer.VERSION,
                      scorer=identity(Path(spec.origin)), observations=identity(args.observations))
        write_json(out / 'results' / f'{tag}_score.json', record)
        event(out, 'score_complete', period='F3', component=tag, unit=unit, units=4,
              total_s=scores[tag]['total']['s'])
    for period, source_metadata in metadata.items():
        write_json(out / f'{period}_metadata.json', dict(experiment=EXPERIMENT, period=period,
                   seeds=list(SEEDS), weights=[.5, .5], source_metadata=source_metadata,
                   competition=scores['mean3'][period.lower()]))
    seed_scores = np.array([scores[f's{s}']['total']['s'] for s in SEEDS])
    verification = dict(status='passed', fixed_equal_weights=True, prediction_rows=len(frames['mean3']),
                        seed_mean_s=float(seed_scores.mean()), seed_sd_s=float(seed_scores.std(ddof=1)),
                        seed_worst_s=float(seed_scores.max()), mean3_total_s=scores['mean3']['total']['s'],
                        source_provenance_recorded=True, official_replacements=0)
    write_json(out / 'logs/verification.json', verification)
    write_json(out / 'complete.json', dict(status='complete', experiment=EXPERIMENT,
               competition=scores['mean3'], publish='not_required',
               publish_reason='Execution and scoring requested; publication/adoption gate not yet evaluated.'))
    event(out, 'complete', period='F3', unit=4, units=4, **verification)


if __name__ == '__main__':
    main()
