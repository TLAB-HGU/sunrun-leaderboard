"""Reproduce a fixed official grid from frozen member forecasts; no scoring or training."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

CONFIG_SHA256 = 'abe4f2f3ea0a0ae198999e79382eef7f4d0cd2ba5dfdb6e940a1ca88812a9748'
CANDIDATE = 'E1813'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(config):
    return json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def origins():
    return pd.date_range('2026-05-31T23:00:00Z', '2026-09-26T23:00:00Z', freq='h').as_unit('ns')


def read_grid(path):
    frame = pd.read_parquet(path, columns=['origin_last_input_utc', 'horizon_hours', 'pred_kms'])
    frame['origin_last_input_utc'] = pd.to_datetime(frame.origin_last_input_utc, utc=True).dt.as_unit('ns')
    keys = ['origin_last_input_utc', 'horizon_hours']
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate origin/horizon keys')
    frame = frame.sort_values(keys).reset_index(drop=True)
    expected = origins()
    if len(frame) != 2833 * 72 or not np.array_equal(
            frame.origin_last_input_utc.array.asi8, np.repeat(expected.asi8, 72)):
        raise ValueError('Incorrect official origin grid')
    if not np.array_equal(frame.horizon_hours.to_numpy(), np.tile(np.arange(1, 73), 2833)):
        raise ValueError('Incorrect official horizon grid')
    values = frame.pred_kms.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite forecasts')
    return values.reshape(2833, 72)


def combine(members, control, alpha):
    if alpha == 1.0:
        return np.mean(members, axis=0)
    if control is None:
        raise ValueError('Damped combination requires control')
    return control + alpha * np.mean(members - control, axis=0)


def predict(assets, alpha):
    return np.mean([combine(a['members'], a.get('control'), alpha) for a in assets], axis=0)


def poison_test(assets, alpha):
    # Proves locality of the combination, not upstream model causality.
    base = predict(assets, alpha)
    changed = [{k: v.copy() for k, v in a.items()} for a in assets]
    for a in changed:
        a['members'] += 100.0
    power = bool(np.all(np.abs(predict(changed, alpha) - base) > 1.0))
    changed = [{k: v.copy() for k, v in a.items()} for a in assets]
    changed[0]['members'][0, 0, 0] += 100.0
    delta = predict(changed, alpha) - base
    locality = bool(np.count_nonzero(delta) == 1 and delta[0, 0] != 0)
    return {'passed': power and locality, 'power': power, 'locality': locality,
            'scope': 'Combination only; upstream causal provenance must be reviewed separately.'}


def load_assets(release, config):
    records = config['data']['assets']
    if [r['seed'] for r in records] != [0, 1, 2]:
        raise ValueError('Exactly seeds 0, 1, 2 required')
    count = len(config['hyperparameters']['sources'])
    alpha = config['hyperparameters']['alpha']
    assets = []
    for record in records:
        name = record['filename']
        if Path(name).name != name:
            raise ValueError('Asset must be a local basename')
        path = release / name
        if digest(path) != record['sha256']:
            raise ValueError('Asset hash mismatch')
        with np.load(path, allow_pickle=False) as archive:
            expected = {'members', 'origins_ns'} | ({'control'} if alpha != 1.0 else set())
            if set(archive.files) != expected:
                raise ValueError('Unexpected asset fields')
            a = {key: archive[key] for key in archive.files}
        if a['origins_ns'].dtype != np.int64 or not np.array_equal(a.pop('origins_ns'), origins().asi8):
            raise ValueError('Incorrect asset origins')
        for key, shape in [('members', (count, 2833, 72))] + ([('control', (2833, 72))] if alpha != 1.0 else []):
            if a[key].shape != shape or a[key].dtype != np.float64 or not np.isfinite(a[key]).all():
                raise ValueError('Invalid asset ' + key)
        assets.append(a)
    return assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=Path(__file__).resolve().parents[3] / 'releases' / CANDIDATE.lower())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timing', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('--repeats must be at least two')
    config = json.loads((args.release / 'config.json').read_text())
    if hashlib.sha256(canonical(config).encode()).hexdigest() != CONFIG_SHA256:
        raise ValueError('Configuration identity mismatch')
    assets = load_assets(args.release, config)
    alpha = config['hyperparameters']['alpha']
    poison = poison_test(assets, alpha)
    if not poison['passed']:
        raise ValueError('Combination poison test failed')
    predict(assets, alpha)
    elapsed = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        values = predict(assets, alpha)
        elapsed.append(time.perf_counter() - start)
    frame = pd.DataFrame({'origin_last_input_utc': np.repeat(origins().strftime('%Y-%m-%dT%H:%M:%SZ'), 72),
                          'horizon_hours': np.tile(np.arange(1, 73), 2833), 'pred_kms': values.ravel()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    timing = {'inference_seconds_per_fold': float(np.mean(elapsed) / 2833),
              'batch_seconds': elapsed, 'repeats': args.repeats, 'clock': 'time.perf_counter',
              'scope': 'Frozen member combination and equal mean of three seeds only.',
              'excluded': ['I/O', 'training', 'feature preparation', 'upstream member inference'],
              'not_end_to_end_or_new_origin_inference': True, 'poison': poison}
    args.timing.parent.mkdir(parents=True, exist_ok=True)
    args.timing.write_text(json.dumps(timing, indent=2) + '\n')
    print(json.dumps({'rows': len(frame), 'timing': timing}))


if __name__ == '__main__':
    main()
