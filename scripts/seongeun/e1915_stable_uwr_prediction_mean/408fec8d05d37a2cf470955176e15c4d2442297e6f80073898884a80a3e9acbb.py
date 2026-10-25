"""Reproduce the frozen E1915 official leaderboard forecast from hash-pinned seed archives.

This is archived forecast replay of a fixed grid, not new-origin inference and not
retraining. The leaderboard inference time is measured separately from the loaded
CPU models (see releases/e1915/inference-timing.json).
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

CONFIG_SHA256 = '408fec8d05d37a2cf470955176e15c4d2442297e6f80073898884a80a3e9acbb'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path,
                        default=Path(__file__).resolve().parents[3] / 'releases/e1915')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay-timing', type=Path)
    args = parser.parse_args()
    config = json.loads((args.release / 'config.json').read_text())
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if hashlib.sha256(canonical.encode()).hexdigest() != CONFIG_SHA256:
        raise ValueError('Configuration checksum differs')
    data = config['data']
    origins = pd.date_range(data['origins_start_utc'], data['origins_end_utc'], freq='h').as_unit('ns')
    if len(origins) != data['origin_count'] or data['horizons'] != 72:
        raise ValueError('Declared official grid is inconsistent')
    records = data['assets']
    if [x['seed'] for x in records] != [0, 1, 2]:
        raise ValueError('Exactly original seeds 0/1/2 required')
    seeds = []
    for record in records:
        name = record['filename']
        if Path(name).name != name:
            raise ValueError('Asset must be a local basename')
        path = args.release / name
        if sha(path) != record['sha256']:
            raise ValueError('Seed asset checksum differs')
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {'origins_ns', 'predictions'}:
                raise ValueError('Unexpected archive contents')
            if not np.array_equal(archive['origins_ns'], origins.asi8):
                raise ValueError('Official origin grid differs')
            values = archive['predictions'].copy()
        if values.shape != (data['origin_count'], 72) or values.dtype != np.float64:
            raise ValueError('Invalid seed predictions')
        if not np.isfinite(values).all():
            raise ValueError('Non-finite seed predictions')
        seeds.append(values)
    start = time.perf_counter()
    mean = np.mean(seeds, axis=0)
    seconds = time.perf_counter() - start
    frame = pd.DataFrame({'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), 72),
                          'horizon_hours': np.tile(np.arange(1, 73), data['origin_count']),
                          'pred_kms': mean.ravel()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    if args.replay_timing:
        args.replay_timing.write_text(json.dumps({'cached_averaging_seconds': seconds,
                                                  'not_model_inference': True,
                                                  'not_the_leaderboard_inference_timing': True}, indent=2) + '\n')
    print(json.dumps({'candidate': data['candidate'], 'rows': len(frame), 'config_sha256': CONFIG_SHA256,
                      'scope': 'Fixed-grid archived seed-forecast replay'}))


if __name__ == '__main__':
    main()
