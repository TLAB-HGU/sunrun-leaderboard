"""Fixed-grid archived e2000fs_feat30_smooth_wcrps forecast replay, not retraining or live inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

CONFIG_SHA256 = 'a2cd0af47e63312b3e4c502c7a952369c56b835ac0f1d91099ae0b644c783822'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', type=Path, default=Path(__file__).resolve().parents[3] / 'releases/e2000fs_feat30_smooth_wcrps')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    config = json.loads((a.release / 'config.json').read_text())
    text = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if hashlib.sha256(text.encode()).hexdigest() != CONFIG_SHA256:
        raise ValueError('configuration checksum differs')
    asset = config['data']['asset']
    if asset['filename'] != 'forecast.npz':
        raise ValueError('unexpected forecast archive path')
    path = a.release / asset['filename']
    if hashlib.sha256(path.read_bytes()).hexdigest() != asset['sha256']:
        raise ValueError('forecast archive checksum differs')
    origins = pd.DatetimeIndex([], tz='UTC')
    for period in ('F1', 'F2'):
        grid = config['data']['periods'][period]
        part = pd.date_range(grid['start'], grid['end'], freq='h', tz='UTC')
        if len(part) != grid['count']:
            raise ValueError('declared period grid differs')
        origins = origins.append(part)
    origins = origins.as_unit('ns')
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {'origins_ns', 'predictions'} or not np.array_equal(z['origins_ns'], origins.asi8):
            raise ValueError('forecast origin grid differs')
        pred = z['predictions'].copy()
    if pred.shape != (len(origins), 72) or not np.isfinite(pred).all() or not ((pred >= 1) & (pred <= 3000)).all():
        raise ValueError('invalid forecast values')
    for name, digest in config['data']['archived_source_sha256'].items():
        path = a.release / 'source' / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('archived source checksum differs: ' + name)
    frame = pd.DataFrame({'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), 72),
                          'horizon_hours': np.tile(np.arange(1, 73), len(origins)), 'pred_kms': pred.ravel()})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(a.output, index=False)
    print(json.dumps({'rows': len(frame), 'config_sha256': CONFIG_SHA256, 'scope': 'archived forecast replay'}))

if __name__ == '__main__':
    main()
