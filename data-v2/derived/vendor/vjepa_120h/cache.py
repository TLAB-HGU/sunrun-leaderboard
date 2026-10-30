"""Validate frozen inputs and prepare an independent, resumable hourly frame cache."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from experiments.vjepa_frozen_fusion.encoder import sha256
from experiments.vjepa_frozen_fusion.images import IntensityTransform, preprocess_frame
from .extract import write_json
from .temporal import plan_sources


def check_hash(path, expected):
    if sha256(Path(path)) != expected:
        raise ValueError(f'protected input hash mismatch: {path}')


def validate_inputs(run):
    """Check preregistered references and original preprocessing source identities."""
    protocol = json.loads((run/'protocol.json').read_text())
    reference = Path(protocol['reference_run']).resolve()
    if run.resolve() == reference or reference in run.resolve().parents:
        raise ValueError('new cache must be outside the frozen reference run')
    protected = json.loads((run/'protected-source-hashes.json').read_text())
    for path, expected in protected.items():
        check_hash(path, expected)
    for record in protocol['reference_identities'].values():
        check_hash(record['path'], record['sha256'])
    for name in ('inventory.parquet', 'intensity.json', 'planned_origins.npz'):
        check_hash(run/name, protocol['reference_identities'][name]['sha256'])
    check_hash(run/'checkpoints/vjepa2_1_vitb_dist_vitG_384.pt', protocol['checkpoint_sha256'])
    actual_sha = subprocess.check_output(
        ['git', '-C', str(run/'upstream'), 'rev-parse', 'HEAD'], text=True).strip()
    if actual_sha != protocol['upstream_sha']:
        raise ValueError('upstream revision mismatch')
    # The previous extraction bound the actual decoder and upstream Python files.
    binding = json.loads((reference/'execution-source-hashes.json').read_text())
    for path, expected in binding.items():
        check_hash(Path(path) if Path(path).is_absolute() else run/path, expected)
    return reference, binding


def transform_identity(transform_data, binding):
    transform_hash = hashlib.sha256(json.dumps(transform_data, sort_keys=True).encode()).hexdigest()
    sources = {k: v for k, v in binding.items()
               if k.endswith(('images.py', 'xgboost_suvi_fusion/features.py'))}
    preprocessing_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    return transform_hash, preprocessing_hash


def prepare_frame(row, frame_dir, reference_frames, old_record, transform, transform_hash,
                  preprocessing_hash):
    """Never write through a reused symlink; commit each new frame before metadata."""
    key = hashlib.sha256(
        f'{row.path}:{row.size_bytes}:{row.mtime_ns}:{transform_hash}:{preprocessing_hash}'.encode()
    ).hexdigest()
    output, metadata = frame_dir/(key+'.npy'), frame_dir/(key+'.json')
    raw = Path(row.path)
    before = raw.stat()
    if (before.st_size, before.st_mtime_ns) != (row.size_bytes, row.mtime_ns):
        raise ValueError('raw source stat changed')
    raw_hash = sha256(raw)
    expected = {'raw_path': row.path, 'raw_sha256': raw_hash, 'bytes': int(row.size_bytes),
                'mtime_ns': int(row.mtime_ns), 'source_slot': row.slot.isoformat(),
                'obs_end': row.obs_end.isoformat()}
    if old_record is not None:
        if old_record['key'] != key:
            raise ValueError('reference frame key mismatch')
        record = {k: v for k, v in old_record.items() if k != 'key'}
        source = reference_frames/(key+'.npy')
    elif metadata.exists():
        record = json.loads(metadata.read_text())
        source = output
    else:
        if output.is_symlink():
            raise ValueError('unrecognized frame symlink')
        image, _, geometry = preprocess_frame(row.path, transform)
        temp = output.with_suffix(f'.{os.getpid()}.tmp')
        with temp.open('wb') as handle:
            np.save(handle, image[0], allow_pickle=False)
        temp.replace(output)
        record = {**expected, 'preprocessed_sha256': sha256(output), 'geometry': geometry}
        source = output
    if any(record[k] != value for k, value in expected.items()):
        raise ValueError('frame raw source or timestamp identity mismatch')
    check_hash(source, record['preprocessed_sha256'])
    image = np.load(source, allow_pickle=False)
    if image.shape != (384, 384) or image.dtype != np.float32 or not np.isfinite(image).all():
        raise ValueError('invalid preprocessed frame')
    fraction = record['geometry']['valid_disk_fraction']
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('invalid frame geometry')
    after = raw.stat()
    if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise ValueError('raw source changed during validation')
    if old_record is not None:
        if output.is_symlink():
            if output.resolve() != source.resolve():
                raise ValueError('reference frame symlink target mismatch')
        elif output.exists():
            raise ValueError('reference frame destination is not a symlink')
        else:
            output.symlink_to(source.resolve())
    if metadata.exists():
        if json.loads(metadata.read_text()) != record:
            raise ValueError('local frame metadata mismatch')
    else:
        write_json(metadata, record)
    return {'key': key, **record}


def prepare_cache(run_dir, workers=6):
    run = Path(run_dir).resolve()
    if workers < 1:
        raise ValueError('workers must be positive')
    # The lock and all outputs belong to the new run. Reference files are read only.
    with (run/'cache-preparation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.perf_counter()
        reference, binding = validate_inputs(run)
        transform_data = json.loads((run/'intensity.json').read_text())
        transform = IntensityTransform(**transform_data)
        transform_hash, preprocessing_hash = transform_identity(transform_data, binding)
        old = json.loads((reference/'frame-cache-manifest.json').read_text())
        if old['transform_sha256'] != transform_hash:
            raise ValueError('reference transform identity mismatch')
        origins = np.load(run/'planned_origins.npz', allow_pickle=False)['origins']
        rows, ids, _, _, _ = plan_sources(pd.read_parquet(run/'inventory.parquet'), origins)
        required = np.unique(ids[ids >= 0])
        frame_dir = run/'cache/frames'
        if frame_dir.is_symlink() or not frame_dir.resolve().is_relative_to(run):
            raise ValueError('frame output directory must belong to the new run')
        frame_dir.mkdir(parents=True, exist_ok=True)
        def one(i):
            return str(i), prepare_frame(rows.iloc[i], frame_dir, reference/'cache/frames',
                old['frames'].get(str(i)), transform, transform_hash, preprocessing_hash)
        mapping = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for n, (i, record) in enumerate(pool.map(one, required), 1):
                mapping[i] = record
                if n % 256 == 0 or n == len(required):
                    write_json(run/'cache-preparation-progress.json', {'stage': 'validating',
                        'done': n, 'total': len(required), 'seconds': time.perf_counter()-started})
                    print(f'frame cache {n}/{len(required)}', flush=True)
        manifest = {'transform_sha256': transform_hash, 'preprocessing_sha256': preprocessing_hash,
                    'reference_manifest_sha256': sha256(reference/'frame-cache-manifest.json'),
                    'frames': mapping}
        destination = run/'frame-cache-manifest.json'
        if destination.exists():
            if json.loads(destination.read_text()) != manifest:
                raise ValueError('completed frame cache identity drift')
        else:
            write_json(destination, manifest)
        reused = sum(i in old['frames'] for i in mapping)
        report = {'stage': 'complete', 'required_frames': len(required), 'reused_frames': reused,
                  'new_frames': len(required)-reused, 'all_raw_hashes_verified': True,
                  'all_preprocessed_hashes_verified': True, 'requested_slots_per_origin': 120,
                  'manifest_sha256': sha256(destination), 'seconds': time.perf_counter()-started}
        write_json(run/'cache-preparation.json', report)
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(prepare_cache(args.run_dir, args.workers), indent=2))
