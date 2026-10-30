#!/usr/bin/env python3
"""Export central raw tables to local, regenerable year-partitioned Parquet."""
import argparse
import json
import os
from pathlib import Path
import uuid

from spaceweather_collector.rawdb import request

STAGE = Path.home() / '.local/state/sundb/stage'
TABLES = {'ace_solar': ('raw.solar', 'timestamp_utc'),
          'suvi_hourly': ('raw.suvi_hourly', 'slot')}


def export_stage(stage=STAGE, datasets=None, years=None):
    """Atomically refresh selected local partitions; never contact external storage."""
    stage = Path(stage).expanduser().resolve()
    receipts = []
    for name in datasets if datasets is not None else TABLES:
        table, timestamp = TABLES[name]
        available = request('query', sql=f'SELECT DISTINCT year({timestamp}) AS year FROM {table} WHERE {timestamp} IS NOT NULL ORDER BY year')['rows']
        for (year,) in available:
            if years is not None and year not in years:
                continue
            target = stage / name / f'year={year}' / 'part.parquet'
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f'.part-{uuid.uuid4().hex}.parquet')
            manifest = temporary.with_name(temporary.name + '.manifest.json')
            try:
                receipt = request('export', sql=f'SELECT * FROM {table} WHERE year({timestamp}) = ? ORDER BY {timestamp}',
                                  params=[year], path=str(temporary), format='parquet')
                receipt['path'] = str(target)
                receipt['manifest'] = str(target.with_name(target.name + '.manifest.json'))
                with manifest.open('w') as stream:
                    stream.write(json.dumps(receipt, indent=2) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
                os.replace(manifest, receipt['manifest'])
                descriptor = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                receipts.append(receipt)
            finally:
                temporary.unlink(missing_ok=True)
                manifest.unlink(missing_ok=True)
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=STAGE)
    parser.add_argument('--dataset', action='append', choices=TABLES)
    parser.add_argument('--year', type=int, action='append')
    args = parser.parse_args()
    print(json.dumps(export_stage(args.stage, args.dataset, args.year), indent=2))


if __name__ == '__main__':
    main()
