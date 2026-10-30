"""Fixed video-arm refit for the team leaderboard; never reads scoring truth.

The exploratory experiment's pre-June guards remain unchanged. This separate
adapter accepts only fold origins, refits through May, and records every input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

from experiments.ch_breakthrough_v2 import common, hpo
from experiments.ch_breakthrough_v2.recurrence import RecurrenceResidualForecaster
from experiments.mse20_four_regimes import protocol, tree
from . import images, numeric

CUTOFF = pd.Timestamp('2026-05-31T23:00:00Z')
UPSTREAM_SHA = '204698b45b3712590f06245fbfba32d3be539812'
CHECKPOINT = 'vjepa2_1_vitb_dist_vitG_384.pt'
CHECKPOINT_SHA = '848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d'


def fold_origins(path):
    # Read only the temporal input key, never fold regime/scoring columns.
    table = pd.read_parquet(path, columns=['origin_last_input_utc'])
    origins = tree._origins(table.origin_last_input_utc)
    if origins.min() < CUTOFF:
        raise ValueError('official fold origins must be at or after the training cutoff')
    return origins


def load_frames(end, root=None):
    root = Path(root or protocol.ROOT)
    frames = hpo.load_inputs(root / 'store/ch-v1/ace.parquet',
                            root / 'store/ch-v1/ch-hourly.parquet',
                            str(root / 'store/ch-breakthrough-v2/upload/raw-ace-snapshot/year=*/part.parquet'))
    # All builders are causal; cap returned history to the last requested origin.
    end = common.utc(end)
    return {key: value.loc[:end] if isinstance(value, (pd.DataFrame, pd.Series))
            and isinstance(value.index, pd.DatetimeIndex) else value
            for key, value in frames.items()}


def official_inventory(end, metadata_root=images.DEFAULT_METADATA, images_root=images.DEFAULT_IMAGES):
    """Same Fe195/G18, geometry and timestamp rules, extended to official inputs."""
    end = images.utc(end)
    parts = sorted(p for p in Path(metadata_root).glob('year=*/*.parquet')
                   if 2022 <= int(p.parent.name.split('=')[1]) <= end.year)
    if not parts:
        raise FileNotFoundError(f'no SUVI metadata partitions: {metadata_root}')
    frame = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    frame['slot'], frame['obs_end'] = images.utc(frame.slot), images.utc(frame.obs_end)
    frame = frame.loc[frame.slot.between(images.utc('2022-01-01'), end)
                      & frame.sat.str.upper().eq('G18') & frame.channel.eq('Fe195')].sort_values('slot').copy()
    if frame.slot.duplicated().any():
        raise ValueError('duplicate metadata slots')
    paths, sizes, mtimes = [], [], []
    for row in frame.itertuples():
        path = None
        if row.status == 'ok' and pd.notna(row.obs_end):
            if row.obs_end > row.slot:
                raise ValueError('observation end exceeds source slot')
            candidate = Path(images_root) / 'g18' / row.obs_end.strftime('%Y/%j') / str(row.filename).replace('.fits', '.nc')
            if candidate.is_file():
                path = candidate.resolve()
        stat = path.stat() if path else None
        paths.append(str(path) if path else None)
        sizes.append(stat.st_size if stat else 0)
        mtimes.append(stat.st_mtime_ns if stat else 0)
    frame['raw_path'], frame['size_bytes'], frame['mtime_ns'] = paths, sizes, mtimes
    frame['raw_available'] = frame.raw_path.notna()
    invalid = np.zeros(len(frame), dtype=bool)
    for column in ('rsun', 'crpix1', 'crpix2', 'crota'):
        if column in frame:
            values = pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype=float)
            invalid |= ~np.isfinite(values)
            if column == 'rsun':
                invalid |= values <= 0
    frame['invalid_geometry'] = invalid
    frame['path'] = frame.raw_path.where(~frame.invalid_geometry, None)
    frame['available'] = frame.path.notna()
    frame.attrs['arrival_limitation'] = ('Historical obs_end <= slot does not prove live '
        'arrival; cutoff_applied indicates collector live LastModified enforcement.')
    return frame.reset_index(drop=True)


def reference_identity(reference):
    """Verify the frozen numerical, image and upstream sources before reuse."""
    reference = Path(reference).resolve()
    hashes = {}
    for name in ('protected-source-hashes.json', 'comparison_code_identity.json', 'execution-source-hashes.json'):
        for source, expected in json.loads((reference / name).read_text()).items():
            if not source.endswith('.py'):
                continue  # Training config is pinned directly by comparison_protocol.json.
            relative = ('upstream/' + source.split('/upstream/', 1)[1]
                        if '/upstream/' in source else
                        'leaderboard/' + source.split('/leaderboard/', 1)[1]
                        if '/leaderboard/' in source else source)
            path = (reference / relative if relative.startswith('upstream/') else
                    protocol.ROOT / relative.removeprefix('leaderboard/'))
            if protocol.digest(path) != expected:
                raise ValueError(f'frozen source changed: {path}')
            hashes[relative] = expected
    checkpoint = reference / 'checkpoints' / CHECKPOINT
    if protocol.digest(checkpoint) != CHECKPOINT_SHA:
        raise ValueError('checkpoint identity mismatch')
    head = subprocess.check_output(['git', '-C', str(reference / 'upstream'), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(reference / 'upstream'), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if head != UPSTREAM_SHA or dirty:
        raise ValueError('upstream identity mismatch')
    for name in ('comparison_protocol.json', 'intensity.json'):
        hashes[name] = protocol.digest(reference / name)
    hashes['checkpoints/' + CHECKPOINT] = CHECKPOINT_SHA
    intensity = json.loads((reference / 'intensity.json').read_text())
    if common.utc(intensity['train_cutoff']) > CUTOFF:
        raise ValueError('intensity normalization uses future data')
    return hashes


def prepare(folds, run_dir, reference_run, root=None, metadata_root=images.DEFAULT_METADATA,
            images_root=images.DEFAULT_IMAGES):
    run, reference = Path(run_dir).resolve(), Path(reference_run).resolve()
    if run == reference or reference.is_relative_to(run):
        raise ValueError('official run must be isolated from reference artifacts')
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f'prepare requires an empty output directory: {run}')
    origins = fold_origins(folds)
    identity = reference_identity(reference)
    config = json.loads((reference / 'comparison_protocol.json').read_text())['config']
    if config['id'] != 'champion' or config['max_train_origins'] != 6000:
        raise ValueError('reference is not the fixed champion recipe')
    frames = load_frames(origins.max(), root)
    if len(origins.difference(frames['base'].index)):
        raise ValueError('numeric inputs do not cover every official origin')
    pool = numeric.train_origins(frames, CUTOFF, config['max_train_origins'])
    inventory = official_inventory(origins.max(), metadata_root, images_root)
    run.mkdir(parents=True, exist_ok=True)
    (run / 'checkpoints').mkdir()
    (run / 'upstream').symlink_to(reference / 'upstream', target_is_directory=True)
    (run / 'checkpoints' / CHECKPOINT).symlink_to(reference / 'checkpoints' / CHECKPOINT)
    shutil.copyfile(reference / 'intensity.json', run / 'intensity.json')
    # Share immutable entries, not the mutable frame directory or any manifests.
    cache = run / 'cache/frames'
    cache.mkdir(parents=True)
    for entry in sorted((reference / 'cache/frames').glob('*.npy')):
        if entry.suffix == '.npy' and entry.with_suffix('.json').is_file():
            (cache / entry.name).symlink_to(entry)
            (cache / entry.with_suffix('.json').name).symlink_to(entry.with_suffix('.json'))
    inventory.to_parquet(run / 'inventory.parquet', index=False)
    planned = pool.union(origins).sort_values().as_unit('ns')
    np.savez(run / 'planned_origins.npz', origins=planned.asi8,
             train=pool.as_unit('ns').asi8, eval=origins.as_unit('ns').asi8)
    manifest = dict(created_utc=protocol.utcnow(), arm='video', config=config,
                    cutoff=CUTOFF.isoformat(), n_train_origins=len(pool), n_folds=len(origins),
                    official_truth_access=False, purpose='team leaderboard only',
                    reference_run=str(reference), reference_identities=identity,
                    folds=protocol.file_ref(folds), input_identities=frames['identities'],
                    feature_end=origins.max().isoformat(), upstream_sha=UPSTREAM_SHA,
                    checkpoint_sha256=CHECKPOINT_SHA,
                    inventory_sha256=protocol.digest(run / 'inventory.parquet'),
                    planned_origins_sha256=protocol.digest(run / 'planned_origins.npz'),
                    intensity_sha256=protocol.digest(run / 'intensity.json'))
    protocol.write_json(run / 'official_protocol.json', manifest)
    return manifest


def saved_features(frames, origins, model_dir, embeddings, meta):
    features = tree._features(frames)
    visual = numeric.visual_features(embeddings, 'video', pd.DatetimeIndex([], tz='UTC'),
                                     origins, model_dir, fit=False)
    features = pd.concat([features, visual], axis=1)
    if list(features.columns) != meta['base_columns']:
        raise ValueError('saved feature schema mismatch')
    return features


def benchmark_saved(frames, origins, model_dir, embeddings, repeats=3):
    """Time restored CPU booster inference, excluding design/feature construction."""
    from xgboost import Booster
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / 'metadata.json').read_text())
    features = saved_features(frames, origins, model_dir, embeddings, meta)
    fore = RecurrenceResidualForecaster(params=meta['config']['params'], device='cpu',
                                        seed=meta['config']['seed'], use_target=True)
    fore.base_columns, fore.columns = meta['base_columns'], meta['columns']
    booster = Booster(model_file=model_dir / 'model.ubj')
    booster.set_param({'device': 'cpu', 'nthread': 4})
    samples = origins[np.unique(np.linspace(0, len(origins)-1, min(16, len(origins)), dtype=int))]
    elapsed = []
    for origin in samples:
        _, design, anchor = fore._build_predict_matrix(features, pd.DatetimeIndex([origin]), frames['speed'])
        design = design[fore.columns]
        booster.inplace_predict(design)
        for _ in range(repeats):
            begin = time.perf_counter()
            result = np.clip(anchor + booster.inplace_predict(design), 1, 3000).astype(float)
            elapsed.append(time.perf_counter() - begin)
            if result.shape != (72,) or not np.isfinite(result).all():
                raise ValueError('invalid benchmark prediction')
    return dict(inference_seconds_per_fold=float(np.mean(elapsed)),
                method='restored CPU booster.inplace_predict + anchor/clip/float vector; one 72h fold per call',
                excludes=['loading', 'training', 'model initialization', 'image encoding', 'PCA',
                          'numeric features', 'prediction matrix construction', 'serialization'],
                n_origins=len(samples), repeats=repeats, samples_seconds=elapsed, device='cpu', nthread=4)


def verify_saved(frames, origins, predictions, model_dir, embeddings):
    replay = numeric.predict_saved(frames, origins, model_dir, 'cpu', embeddings)
    np.testing.assert_allclose(predictions.pred_kms, replay.pred_kms, atol=1e-4, rtol=0)
    checks = []
    for origin in origins[np.unique(np.linspace(0, len(origins)-1, min(3, len(origins)), dtype=int))]:
        selected = pd.DatetimeIndex([origin])
        expected = replay.loc[replay.origin_last_input_utc.eq(origin), 'pred_kms'].to_numpy()
        for operation in ('perturb', 'delete'):
            altered = {}
            for key in ('base', 'physics', 'quality', 'speed', 'observed'):
                value = frames[key]
                if operation == 'delete':
                    value = value.loc[:origin].copy()
                else:
                    value = value.copy()
                    value.loc[value.index > origin] = True if key == 'observed' else 99999.
                altered[key] = value
            actual = numeric.predict_saved(altered, selected, model_dir, 'cpu', embeddings)
            np.testing.assert_array_equal(expected, actual.pred_kms.to_numpy())
            checks.append(dict(origin=origin.isoformat(), operation=operation, equal=True))
    return dict(saved_prediction_parity=True, future_numeric_checks=checks,
                scope='post-builder numeric features and ACE speed history; frozen builders covered by existing tests')


def predict(folds, run_dir, root=None, device='cpu'):
    run = Path(run_dir).resolve()
    manifest = json.loads((run / 'official_protocol.json').read_text())
    origins = fold_origins(folds)
    if protocol.digest(folds) != manifest['folds']['sha256']:
        raise ValueError('fold identity drift')
    for name, field in [('inventory.parquet', 'inventory_sha256'),
                        ('planned_origins.npz', 'planned_origins_sha256'), ('intensity.json', 'intensity_sha256')]:
        if protocol.digest(run / name) != manifest[field]:
            raise ValueError(f'prepared artifact changed: {name}')
    if reference_identity(manifest['reference_run']) != manifest['reference_identities']:
        raise ValueError('reference identity drift')
    frames = load_frames(origins.max(), root)
    if frames['identities'] != manifest['input_identities']:
        raise ValueError('numeric input identity drift')
    embeddings = run / 'embeddings.npz'
    extraction = json.loads((run / 'extraction.json').read_text())
    if protocol.digest(embeddings) != extraction['embeddings_sha256']:
        raise ValueError('embedding identity drift')
    with np.load(run / 'planned_origins.npz', allow_pickle=False) as planned:
        embedded, _, _ = numeric.read_embeddings(embeddings, 'video')
        np.testing.assert_array_equal(embedded.as_unit('ns').asi8, planned['origins'])
        pool = numeric.train_origins(frames, CUTOFF, manifest['config']['max_train_origins'])
        np.testing.assert_array_equal(pool.as_unit('ns').asi8, planned['train'])
        np.testing.assert_array_equal(origins.as_unit('ns').asi8, planned['eval'])
    model_dir = run / 'model'
    if (model_dir / 'metadata.json').exists():
        meta = json.loads((model_dir / 'metadata.json').read_text())
        if meta['config'] != manifest['config'] or common.utc(meta['cutoff']) != CUTOFF or meta['arm'] != 'video':
            raise ValueError('existing model differs from fixed recipe')
        if meta['embedding_sha256'] != protocol.digest(embeddings) or meta['input_identities'] != frames['identities']:
            raise ValueError('existing model input identity drift')
        predictions = numeric.predict_saved(frames, origins, model_dir, 'cpu', embeddings)
    else:
        if (model_dir / 'model.ubj').exists() or (model_dir / 'visual_transform.npz').exists():
            raise FileExistsError('partial model artifacts; use a fresh model directory')
        predictions, meta = numeric.fit_predict(frames, CUTOFF, origins, manifest['config'],
                                                device, model_dir, 'video', embeddings)
    if common.utc(meta['train_label_max']) > CUTOFF:
        raise ValueError('future training labels')
    validation = verify_saved(frames, origins, predictions, model_dir, embeddings)
    benchmark = benchmark_saved(frames, origins, model_dir, embeddings)
    tree._validate_predictions(predictions, origins)
    output = run / 'predictions.parquet'
    serialized = predictions.copy()
    serialized['origin_last_input_utc'] = pd.to_datetime(
        serialized.origin_last_input_utc, utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    serialized.to_parquet(output, index=False)
    report = dict(created_utc=protocol.utcnow(), arm='video', n_folds=len(origins), n_predictions=len(predictions),
                  train_cutoff=CUTOFF.isoformat(), train_label_max=meta['train_label_max'],
                  predictions=protocol.file_ref(output), model_metadata=protocol.file_ref(model_dir / 'metadata.json'),
                  embeddings=protocol.file_ref(embeddings), protocol=protocol.file_ref(run / 'official_protocol.json'),
                  inference_seconds_per_fold=benchmark['inference_seconds_per_fold'], benchmark=benchmark,
                  validation=validation, official_truth_access=False, selection_or_hpo=False)
    protocol.write_json(run / 'official_manifest.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--reference-run')
    parser.add_argument('--mode', choices=['prepare', 'predict'], required=True)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--metadata-root', type=Path, default=images.DEFAULT_METADATA)
    parser.add_argument('--images-root', type=Path, default=images.DEFAULT_IMAGES)
    args = parser.parse_args()
    if args.mode == 'prepare':
        if not args.reference_run:
            parser.error('--reference-run is required for prepare')
        result = prepare(args.folds, args.run_dir, args.reference_run, args.root, args.metadata_root, args.images_root)
    else:
        result = predict(args.folds, args.run_dir, args.root, args.device)
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
