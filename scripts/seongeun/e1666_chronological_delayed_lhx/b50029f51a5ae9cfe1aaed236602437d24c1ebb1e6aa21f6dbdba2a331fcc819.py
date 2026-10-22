"""Reproduce E1666's fixed-grid, frozen-feature submission (not a raw-data forecaster)."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

CONFIG_SHA256 = 'b50029f51a5ae9cfe1aaed236602437d24c1ebb1e6aa21f6dbdba2a331fcc819'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def predict(assets):
    return np.mean([a['control'] + np.einsum('nhk,hk->nh', a['design'], a['beta'])
                    for a in assets], axis=0)


def load_assets(release, config):
    assets = []
    expected = pd.date_range('2026-05-31T23:00:00Z', '2026-09-26T23:00:00Z', freq='h').asi8
    for record in config['data']['prepared_assets']:
        path = release / record['filename']
        if digest(path) != record['sha256']:
            raise ValueError(f'Asset hash mismatch: {path.name}')
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != {'control', 'design', 'beta', 'origins_ns'}:
                raise ValueError('Unexpected asset fields')
            a = {key: z[key] for key in z.files}
        for key, shape in [('control', (2833, 72)), ('design', (2833, 72, 5)), ('beta', (72, 5))]:
            if a[key].shape != shape or a[key].dtype != np.float64 or not np.isfinite(a[key]).all():
                raise ValueError(f'Invalid {key} shape/dtype/values')
        if a['origins_ns'].dtype != np.int64 or not np.array_equal(a['origins_ns'], expected):
            raise ValueError('Invalid hourly origin grid')
        assets.append(a)
    if len(assets) != 3:
        raise ValueError('Exactly three frozen seeds required')
    return assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path,
                        default=Path(__file__).resolve().parents[3] / 'releases' / 'e1666')
    parser.add_argument('--output', type=Path, default=Path('predictions_e1666.parquet'))
    parser.add_argument('--timing', type=Path, default=Path('timing_e1666.json'))
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('--repeats must be at least 2')
    config = json.loads((args.release / 'config.json').read_text())
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if hashlib.sha256(canonical.encode()).hexdigest() != CONFIG_SHA256:
        raise ValueError('Config identity mismatch')
    assets = load_assets(args.release, config)
    predict(assets)  # Warm initialization is excluded.
    elapsed = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        values = predict(assets)
        elapsed.append(time.perf_counter() - start)
    origins = pd.to_datetime(assets[0]['origins_ns'], utc=True)
    frame = pd.DataFrame({'origin_last_input_utc': np.repeat(origins, 72),
                          'horizon_hours': np.tile(np.arange(1, 73), len(origins)),
                          'pred_kms': values.ravel()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    timing = {'inference_seconds_per_fold': float(np.mean(elapsed) / len(origins)),
              'origins_per_repeat': len(origins), 'repeats': args.repeats,
              'batch_seconds': elapsed, 'clock': 'time.perf_counter',
              'scope': 'Three frozen residual heads plus equal seed mean over prepared features.',
              'excluded': ['I/O', 'initialization', 'feature preparation', 'training',
                           'upstream frozen base/HUXt/LHX prediction generation'],
              'not_end_to_end_or_new_origin_inference': True}
    args.timing.parent.mkdir(parents=True, exist_ok=True)
    args.timing.write_text(json.dumps(timing, indent=2) + '\n')
    print(json.dumps({'rows': len(frame), **timing}))


if __name__ == '__main__':
    main()
