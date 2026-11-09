"""Build additions without labels; append them while preserving every parent column."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

import minute_features
import temporal_features

ROOT = Path(__file__).resolve().parent


def describe(axis):
    if axis in (1, 2):
        return minute_features.describe(axis)
    if axis in (3, 4):
        return temporal_features.describe(axis)
    raise ValueError('Axis must be 1..4')


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def unique_origins(data, axis):
    columns = ['origin'] + (list(describe(axis)['sources']) if axis in (3, 4) else [])
    columns = list(dict.fromkeys(columns))
    frame = pd.read_parquet(data, columns=columns, filters=[('h', '==', 1)])
    frame['origin'] = pd.to_datetime(frame.origin, utc=True)
    for name in columns[1:]:
        if frame.groupby('origin')[name].nunique(dropna=False).gt(1).any():
            raise ValueError('Conflicting input histories: ' + name)
    return frame.drop_duplicates('origin').sort_values('origin').reset_index(drop=True)


def validate_minute_sources(root=ROOT, start=None, end=None):
    root = Path(root)
    manifest_path = root / 'ace-source-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    files = manifest['files']
    if not files:
        raise ValueError('Empty ACE source manifest')
    for name, expected in files.items():
        path = Path(name)
        stat = path.stat()
        if dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns) != expected:
            raise ValueError('ACE minute source changed: ' + name)
    start = pd.Timestamp(manifest['required_start'] if start is None else start)
    end = pd.Timestamp(manifest['required_end'] if end is None else end)
    for day in pd.date_range(start.floor('D'), end.floor('D')):
        path = Path(manifest['cache_root']) / f'{day:%Y%m%d}.txt'
        if str(path) not in files or files[str(path)]['bytes'] <= 0:
            raise ValueError('Required ACE daily source missing: ' + str(path))
    return manifest


def build_cache(data, out, axis):
    """Preserve original row groups/keys so augmentation needs no fuzzy joins."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError('Refuse replacing frozen feature cache')
    spec = describe(axis)
    names = list(spec['constant']) + list(spec['varying'])
    origins = unique_origins(data, axis)
    if axis in (1, 2):
        cache_root = ROOT.parents[1] / 'data-v2/raw/ACE/daily-txt/swepam'
        start = origins.origin.min() - pd.Timedelta(hours=1)
        end = origins.origin.max()
        manifest = validate_minute_sources(ROOT, start, end)
        if Path(manifest['cache_root']) != cache_root:
            raise ValueError('ACE source manifest root differs')
        minutes = minute_features.load_minutes(cache_root, start, end)
        point = minute_features.build(pd.DatetimeIndex(origins.origin), minutes, axis)
    else:
        point = temporal_features.build(origins, axis)
    if list(point.columns) != names or not point.index.equals(pd.DatetimeIndex(origins.origin)):
        raise ValueError('Feature provider changed unique origin order or names')
    source = pq.ParquetFile(data)
    writer = None
    finite = {name: 0 for name in names}
    finite_training = {period: {name: 0 for name in names} for period in ('F1', 'F2', 'F3')}
    rows = 0
    try:
        for group in range(source.num_row_groups):
            columns = ['origin', 'h']
            columns += [name for name in ('set', 'fit_F1', 'fit_F2', 'fit_F3') if name in source.schema_arrow.names]
            frame = source.read_row_group(group, columns=columns, use_threads=False).to_pandas()
            values = pd.DataFrame(point.reindex(pd.DatetimeIndex(frame.origin)).to_numpy(), columns=names, index=frame.index)
            if list(values.columns) != names or not values.index.equals(frame.index):
                raise ValueError('Feature provider changed output order or keys')
            if np.isinf(values.to_numpy()).any():
                raise ValueError('Infinite derived inputs')
            cache = pa.Table.from_pandas(frame[['origin', 'h']], preserve_index=False)
            for name in names:
                array = values[name].to_numpy(np.float32)
                finite[name] += int(np.isfinite(array).sum())
                for period in finite_training:
                    flag = 'fit_' + period
                    if flag in frame:
                        mask = frame[flag].to_numpy(bool)
                        if 'set' in frame:
                            mask &= frame['set'].eq('train').to_numpy()
                        finite_training[period][name] += int((np.isfinite(array) & mask).sum())
                cache = cache.append_column(name, pa.array(array))
            if writer is None:
                writer = pq.ParquetWriter(out, cache.schema, compression='zstd')
            writer.write_table(cache, row_group_size=len(cache))
            rows += len(cache)
            if group % 16 == 0:
                print(json.dumps(dict(stage='feature_cache', axis=axis, unit=group + 1, units=source.num_row_groups)), flush=True)
    finally:
        if writer:
            writer.close()
    if rows != source.metadata.num_rows or any(count == 0 for count in finite.values()):
        raise ValueError('Feature cache incomplete or entirely missing')
    stat = Path(data).stat()
    proof = dict(axis=axis, source_identity=dict(path=str(Path(data).resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns),
                 rows=rows, row_groups=source.num_row_groups, constant=list(spec['constant']),
                 varying=list(spec['varying']), finite_counts=finite, finite_training_counts=finite_training,
                 target_columns_read=False,
                 ace_source_manifest='ace-source-manifest.json' if axis in (1, 2) else None)
    atomic(out.with_suffix('.json'), proof)
    return proof


def equal_original(left, right):
    if left.schema != right.schema or len(left) != len(right):
        return False
    for name in left.column_names:
        a, b = left[name], right[name]
        if not a.is_null().equals(b.is_null()):
            return False
        if pa.types.is_floating(a.type):
            same = pc.or_(pc.equal(a, b), pc.and_(pc.is_nan(a), pc.is_nan(b)))
            if pc.all(pc.fill_null(same, True)).as_py() is False:
                return False
        elif not a.equals(b):
            return False
    return True


def augment(data, cache, out):
    original, features = pq.ParquetFile(data), pq.ParquetFile(cache)
    out = Path(out)
    if out.exists():
        raise FileExistsError('Refuse replacing frozen training input')
    if original.num_row_groups != features.num_row_groups:
        raise ValueError('Feature cache row groups differ')
    additions = [n for n in features.schema_arrow.names if n not in ('origin', 'h')]
    if set(additions) & set(original.schema_arrow.names):
        raise ValueError('Feature replacement is forbidden')
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for i in range(original.num_row_groups):
            source, extra = original.read_row_group(i, use_threads=False), features.read_row_group(i, use_threads=False)
            if len(source) != len(extra) or any(not source[k].equals(extra[k]) for k in ('origin', 'h')):
                raise ValueError('Feature cache keys differ from immutable source')
            result = source
            for name in additions:
                result = result.append_column(name, extra[name])
            if writer is None:
                writer = pq.ParquetWriter(out, result.schema, compression='zstd')
            writer.write_table(result, row_group_size=len(result))
    finally:
        if writer:
            writer.close()
    restored = pq.ParquetFile(out)
    for i in range(original.num_row_groups):
        if not equal_original(original.read_row_group(i, use_threads=False), restored.read_row_group(i, columns=original.schema_arrow.names, use_threads=False)):
            raise ValueError('Original features, labels or metadata changed during append')
    proof = dict(passed=True, rows=original.metadata.num_rows, added_features=additions,
                 original_columns=original.schema_arrow.names, original_columns_exact=True,
                 source_bytes=Path(data).stat().st_size, output_bytes=out.stat().st_size)
    atomic(out.with_suffix('.validation.json'), proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--axis', type=int, choices=range(1, 5))
    args = parser.parse_args()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    if args.axis:
        build_cache(args.data, args.cache, args.axis)
    if args.out:
        print(json.dumps(augment(args.data, args.cache, args.out)))
