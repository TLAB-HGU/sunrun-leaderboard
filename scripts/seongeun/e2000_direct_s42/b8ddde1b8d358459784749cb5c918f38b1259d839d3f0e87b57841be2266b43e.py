"""Reproduce one submitted E2000 forecast (F1 + F2) from a hash-pinned archive.

This is fixed-grid replay of archived forecasts, not retraining. The training and prediction code
that produced the archive is preserved under the release's source/ (hashes in config.json).
Leaderboard timing was measured separately on the model predict call only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG_SHA256 = 'b8ddde1b8d358459784749cb5c918f38b1259d839d3f0e87b57841be2266b43e'
SLUG = 'direct_multioutput'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path,
                        default=Path(__file__).resolve().parents[3] / 'releases/e2000/direct_multioutput')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.release / 'config.json').read_text())
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if hashlib.sha256(canonical.encode()).hexdigest() != CONFIG_SHA256:
        raise ValueError('configuration checksum differs')
    data = config['data']
    parts = []
    for period in ('F1', 'F2'):
        p = data['periods'][period]
        o = pd.date_range(p['origins_start_utc'], p['origins_end_utc'], freq='h', tz='UTC').as_unit('ns')
        if len(o) != p['origin_count']:
            raise ValueError(f'declared {period} grid is inconsistent')
        parts.append(o.asi8)
    origins = pd.DatetimeIndex(np.concatenate(parts), tz='UTC')
    asset = data['asset']
    path = args.release / asset['filename']
    if Path(asset['filename']).name != asset['filename'] or sha(path) != asset['sha256']:
        raise ValueError('forecast archive checksum differs')
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {'origins_ns', 'predictions'}:
            raise ValueError('unexpected archive contents')
        if not np.array_equal(archive['origins_ns'], origins.asi8):
            raise ValueError('origin grid differs')
        values = archive['predictions'].copy()
    if values.shape != (len(origins), 72) or values.dtype != np.float64 or not np.isfinite(values).all():
        raise ValueError('invalid predictions')
    frame = pd.DataFrame({'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), 72),
                          'horizon_hours': np.tile(np.arange(1, 73), len(origins)),
                          'pred_kms': values.ravel()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(json.dumps({'strategy': config['hyperparameters']['strategy'], 'rows': len(frame),
                      'config_sha256': CONFIG_SHA256, 'scope': 'fixed-grid archived forecast replay'}))


if __name__ == '__main__':
    main()
