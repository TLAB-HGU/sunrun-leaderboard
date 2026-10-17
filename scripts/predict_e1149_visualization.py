#!/usr/bin/env python3
"""Reconstruct E1149 seed 0 and its fully refitted ACE-zero-delay counterpart.

Research recipes are loaded read-only into private modules. Checkpoints contain
plain fitted estimator/state dictionaries, never the source/helper modules.
Outputs are reference visualizations, not experiment registration or scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import glob
import shutil
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import types

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HOUR = pd.Timedelta(hours=1)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Research recipe changed: expected one occurrence of {old!r}')
    return source.replace(old, new)


def load_recipes(root, latency, threads):
    """Keep the 2h recipe exact; only the four latency expressions change at 0h."""
    sys.path.insert(0, str(root))
    base_path = root / 'experiments/e0472_ablation_hpo/recipe.py'
    proxy_path = root / 'experiments/gjc_inc_l72cond_20261014/run_proxy.py'
    base_source = base_path.read_text()
    if latency == 0:
        for old, new in [('self.ace.index += 2 * HOUR', 'self.ace.index += 0 * HOUR'),
                         ('self.af.index += 2 * HOUR', 'self.af.index += 0 * HOUR'),
                         ('observation_cutoff = origin - 2 * HOUR', 'observation_cutoff = origin'),
                         ('err[14:] = past[14:] - cand_paths[:-14, :12]',
                          'err[12:] = past[12:] - cand_paths[:-12, :12]')]:
            base_source = replace_once(base_source, old, new)
    name = f'experiments.e0472_ablation_hpo._visualization_latency{latency}'
    base = types.ModuleType(name)
    base.__file__, base.__package__ = str(base_path), 'experiments.e0472_ablation_hpo'
    sys.modules[name] = base
    exec(compile(base_source, str(base_path), 'exec'), base.__dict__)
    base.__source__ = base_source
    source = proxy_path.read_text()
    source = replace_once(source, "os.environ[_key] = '4'", f"os.environ[_key] = '{threads}'")
    source = replace_once(source, 'from experiments.e0472_ablation_hpo.recipe import', f'from {name} import')
    if source.count('-14*HOUR') != 2:
        raise ValueError('Unexpected E1149 hindcast expression count')
    source = source.replace('-14*HOUR', f'-{12 + latency}*HOUR')
    proxy = types.ModuleType(f'_e1149_visualization_latency{latency}')
    proxy.__file__ = str(proxy_path)
    exec(compile(source, str(proxy_path), 'exec'), proxy.__dict__)
    return base, proxy, {str(p): sha256(p) for p in (base_path, proxy_path,
        root.parent / 'research/e0472-ablation-hpo/campaign/configs/baseline.json',
        root / 'experiments/e0472_ablation_hpo/configs.py')}


def save_checkpoint(path, state):
    temporary = path.with_suffix('.tmp')
    joblib.dump(state, temporary, compress=3)
    temporary.replace(path)


def cached_fit(base, features, cfg, cutoff, threads, path, resume):
    if resume and path.exists():
        state = joblib.load(path)
        model = base.Fit.__new__(base.Fit)
        model.__dict__.update(state['fit'])
        model.features = features
        features.__dict__.update(state['feature_receipt'])
        print(f'Restored fit {path.name}', flush=True)
        return model
    model = base.Fit(features, cfg, cutoff, 0, threads)
    save_checkpoint(path, dict(fit={k: v for k, v in model.__dict__.items() if k != 'features'},
                              feature_receipt={k: features.__dict__[k] for k in
                                               ('coverage', 'excluded_rows') if k in features.__dict__}))
    return model


def cached_replay(proxy, data, cutoff, factory, end, path, resume):
    if resume and path.exists():
        state = joblib.load(path)
        replay = proxy.Replay.__new__(proxy.Replay)
        replay.__dict__.update(state)
        replay.data = data
        replay.model, replay.cal = factory(replay.cutoff), factory(replay.cal_fit)
        print(f'Restored replay {path.name}', flush=True)
        return replay
    replay = proxy.Replay(data, cutoff, 0, factory, end=end)
    save_checkpoint(path, {k: v for k, v in replay.__dict__.items() if k not in ('data', 'model', 'cal')})
    return replay


def build_bank(base, proxy, data, out, threads, resume):
    class CheckpointBank(proxy.LevelBank):
        def raw(self, cutoff):
            key = str(cutoff)
            if key not in self.models:
                name = pd.Timestamp(cutoff).strftime('%Y%m%d%H')
                self.models[key] = cached_fit(base, data, proxy.CFG, cutoff, threads,
                                              out / f'raw_{name}.joblib', True)
            return self.models[key]

        def replay(self, cutoff):
            key = str(cutoff)
            if key not in self.replays:
                end = (proxy.FOLDS[list(proxy.FOLDS).index(cutoff) + 1] - HOUR
                       if cutoff in proxy.FOLDS[:-1] else proxy.END)
                name = pd.Timestamp(cutoff).strftime('%Y%m%d%H')
                self.replays[key] = cached_replay(proxy, data, cutoff, self.raw, end,
                                                  out / f'raw_replay_{name}.joblib', True)
            return self.replays[key]

    bank = CheckpointBank(data, 0, out)
    state_path = out / 'level_bank.joblib'
    if resume and state_path.exists():
        bank.__dict__.update(joblib.load(state_path))
    for start, end in zip(proxy.FOLDS[:-1], proxy.FOLDS[1:]):
        if any(r['start'] == str(start) for r in bank.records):
            continue
        origins = pd.date_range(start, end - HOUR, freq='h')
        x = proxy.summary(data, origins, bank.replay(start).paths(origins))
        head, valid = bank.head(start)
        level = np.full(len(origins), np.nan) if head is None else head.predict(x)
        bank.times = bank.times.append(origins)
        bank.x = np.vstack([bank.x, x])
        bank.level = pd.concat([bank.level, pd.Series(level, index=origins)])
        observed_count = np.column_stack([data.labels.reindex(origins + h * HOUR).notna().to_numpy()
                                          for h in range(1, 73)]).sum(1)
        bank.records.append(dict(start=str(start), end=str(end), n_origins=len(origins),
                                 head_train_n=int(valid.sum()), level_nonmissing=int(np.isfinite(level).sum()),
                                 observed_hours_min=int(observed_count.min()),
                                 observed_hours_mean=float(observed_count.mean()),
                                 label_proxy54_coverage=float((observed_count >= 54).mean()),
                                 complete72_count=int((observed_count == 72).sum())))
        save_checkpoint(state_path, {k: getattr(bank, k) for k in ('times', 'x', 'level', 'records')})
        proxy.dump(out / 'fold_progress_s0.json', bank.records)
        print('Level fold complete ' + str(start), flush=True)
        # Earlier raw trees and pools are reproducibly restored if poison probes need them.
        bank.models.clear()
        bank.replays.clear()
    y = proxy.labels72(data.labels, bank.times)
    finite = np.isfinite(y) & np.isfinite(bank.level.to_numpy()) & (bank.times + 144 * HOUR <= proxy.CUTOFF)
    bank.diagnostic = dict(folds=bank.records,
                          mature_proxy54_labels=int((np.isfinite(y) & (bank.times + 144 * HOUR <= proxy.CUTOFF)).sum()),
                          oof_correlation=float(np.corrcoef(y[finite], bank.level.to_numpy()[finite])[0, 1]),
                          oof_correlation_n=int(finite.sum()),
                          level_feature_coverage=float(np.isfinite(bank.level.to_numpy()).mean()),
                          cold_start='Initial missing chronological Ridge OOF rows excluded')
    pd.DataFrame({'origin': bank.times, 'level72_hat': bank.level.to_numpy(),
                  'level72_observed': y}).to_parquet(out / 'level_oof_s0.parquet', index=False)
    return bank


def prediction_frame(origins, predictions):
    if predictions.shape != (len(origins), 72) or not np.isfinite(predictions).all():
        raise ValueError('Incomplete or nonfinite prediction paths')
    return pd.DataFrame({'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), 72),
                         'horizon_hours': np.tile(np.arange(1, 73), len(origins)),
                         'pred_kms': predictions.ravel()})


def compare_original(frame, reference):
    keys = ['origin_last_input_utc', 'horizon_hours']
    expected = pd.read_parquet(reference)
    actual = frame.copy()
    for value in (expected, actual):
        value[keys[0]] = pd.to_datetime(value[keys[0]], utc=True)
        if value.duplicated(keys).any():
            raise ValueError('Duplicate prediction keys')
    expected, actual = [v.set_index(keys).sort_index() for v in (expected, actual)]
    if not expected.index.equals(actual.index):
        raise ValueError('Original E1149 prediction keys differ')
    delta = float(np.max(np.abs(expected.pred_kms.to_numpy() - actual.pred_kms.to_numpy())))
    if not np.isfinite(delta) or delta > 1e-5:
        raise ValueError(f'E1149 original reconstruction mismatch: maxabs={delta}')
    return dict(max_abs_delta=delta, tolerance=1e-5, reference_sha256=sha256(reference), passed=True)


def reference_inputs(snapshot):
    return dict(raw_ace_dir=str(snapshot / 'ace-snapshot'), speed_parquet=str(snapshot / 'ch/ace.parquet'),
                ch_parquet=str(snapshot / 'ch/ch-hourly.parquet'),
                suvi_parquet=str(snapshot / 'suvi-fusion/suvi-hourly-v2.parquet'),
                suvi_manifest=str(snapshot / 'suvi_availability.parquet'), donki_dir=str(snapshot / 'donki'))


def switch_to_reference(data, snapshot, latency):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model_release.features import FeatureInputs
    from model_release import numeric
    from model_release.strips import select_slots
    inputs = reference_inputs(snapshot)
    source = FeatureInputs(inputs, include_enlil=True)
    last = min(source.speed.index.max() - 72 * HOUR, pd.Timestamp('2026-10-31 22:00', tz='UTC'))
    origins = pd.date_range(pd.Timestamp('2026-09-30 23:00', tz='UTC'), last, freq='h')
    if not len(origins):
        raise ValueError('No complete October windows')
    required = pd.date_range(origins.min() - (167 + latency) * HOUR, origins.max() + 72 * HOUR, freq='h')
    if not np.isfinite(source.speed.reindex(required)).all():
        raise ValueError('Missing October observation/history values')
    selection = select_slots(origins, source.meta, source.block)
    valid = ~selection['missing']
    picked = source.meta.iloc[selection.loc[valid, 'slot_idx']]
    if not ((picked.s3_modified.to_numpy() <= origins[valid.to_numpy()].to_numpy()).all()
            and (picked.obs_end.to_numpy() <= (origins[valid.to_numpy()] - HOUR).to_numpy()).all()):
        raise ValueError('SUVI availability violation')
    grid = pd.date_range(pd.Timestamp('2021-01-01', tz='UTC'), last, freq='h')
    data.ace = source.speed.copy()
    data.ace.index += latency * HOUR
    data.af = numeric.ace_features(grid, source.raw)
    data.af.index += latency * HOUR
    data.cf = numeric.ch_features(grid, source.ch)
    metadata = pd.concat([pd.read_parquet(p) for p in sorted((snapshot / 'suvi-meta').glob('year=*/part.parquet'))], ignore_index=True)
    metadata['slot'] = pd.to_datetime(metadata['slot'], utc=True)
    paths = metadata.drop_duplicates('slot', keep='last').set_index('slot').nc_path
    source.meta['nc_path'] = source.meta.slot.map(paths)
    if (source.meta.available & source.meta.nc_path.isna()).any():
        raise ValueError('Available reference SUVI metadata missing nc_path')
    source.meta['nc_path'] = source.meta.nc_path.fillna('')
    data.meta, data.block, data.runs = source.meta, source.block, source.events
    data.enlil_cache = {}
    # Labels and all learned calibration state deliberately remain the training snapshot.
    return origins, dict(origins=len(origins), first_origin_utc=origins.min().isoformat(),
                         last_origin_utc=origins.max().isoformat(),
                         last_actual_utc=(origins.max() + 72 * HOUR).isoformat(),
                         suvi_stale_origins=int(selection.img_stale.sum()))


def verify_sources(sources):
    for path, receipt in sources.items():
        if sha256(path) != receipt['sha256']:
            raise ValueError(f'Source changed during reconstruction: {path}')


def pin_training_inputs(base, root, out):
    """Redirect recipe reads to immutable copies, including resumed runs."""
    import pyarrow.parquet as pq
    directory = out / 'training_snapshot'
    directory.mkdir(exist_ok=True)
    mapping_path = directory / 'mapping.json'
    sys.path.insert(0, str(root / 'experiments/d148_enlil_live'))
    import lean_enlil as E
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text())
    else:
        paths = [root / 'store/ch-v1/ace.parquet', root / 'store/ch-v1/ch-hourly.parquet',
                 root / 'store/suvi-fusion/suvi-hourly-v2.parquet']
        paths += list((root / 'store/ch-breakthrough-v2/upload/raw-ace-snapshot').glob('year=*/part.parquet'))
        paths += list(Path('/home/t-lab01/.local/state/sundb/stage/suvi_hourly').glob('year=*/part.parquet'))
        paths += list((E.L.ia.OUT / 'donki').glob('WSAEnlilSimulations_*.json'))
        paths += list(E.LIVE.glob('WSAEnlilSimulations_*.json'))
        mapping = {}
        for path in paths:
            path = path.resolve()
            target = directory / (hashlib.sha256(str(path).encode()).hexdigest()[:16] + '_' + path.name)
            before = sha256(path)
            shutil.copyfile(path, target)
            if sha256(target) != before or sha256(path) != before:
                raise ValueError(f'Training source changed while copied: {path}')
            mapping[str(path)] = dict(path=str(target), sha256=before)
        mapping_path.write_text(json.dumps(mapping, indent=2) + '\n')
    for entry in mapping.values():
        if sha256(entry['path']) != entry['sha256']:
            raise ValueError('Training snapshot identity mismatch')
    def snapshot_path(path):
        return Path(mapping[str(Path(path).resolve())]['path'])
    @contextmanager
    def source_guard(cap, access, allowed_paths=None):
        original = pd.read_parquet
        def read(path, *args, **kwargs):
            key = str(Path(path).resolve())
            if key not in mapping or not key.endswith('.parquet'):
                raise ValueError(f'Non-source parquet forbidden: {key}')
            entry = mapping[key]
            names = pq.ParquetFile(entry['path']).schema_arrow.names
            if 'timestamp_utc' in names:
                if kwargs.pop('filters', None):
                    raise ValueError('Source guard owns timestamp filters')
                kwargs['filters'] = [('timestamp_utc', '<=', cap.to_pydatetime())]
            access[entry['path']] = dict(sha256=entry['sha256'], cap=str(cap), original_path=key)
            frame = original(entry['path'], *args, **kwargs)
            if isinstance(frame.index, pd.DatetimeIndex):
                frame = frame.loc[pd.to_datetime(frame.index, utc=True) <= cap]
            return frame
        pd.read_parquet = read
        try:
            yield
        finally:
            pd.read_parquet = original
    # Preserve original DONKI ordering/filtering while replacing file contents.
    original_json = json.load
    original_glob = glob.glob
    def frozen_glob(pattern):
        import fnmatch
        if 'WSAEnlilSimulations_' in str(pattern):
            return [path for path in mapping if fnmatch.fnmatch(path, str(pattern))]
        if str(pattern).endswith('year=*/part.parquet'):
            return [path for path in mapping if fnmatch.fnmatch(path, str(pattern))]
        return original_glob(pattern)
    def frozen_json(stream, *args, **kwargs):
        if str(Path(stream.name).resolve()) in mapping:
            with snapshot_path(stream.name).open() as copied:
                return original_json(copied, *args, **kwargs)
        return original_json(stream, *args, **kwargs)
    glob.glob = frozen_glob
    json.load = frozen_json
    try:
        E.RUNS = E.enlil_runs()
    finally:
        json.load = original_json
    # Keep frozen glob inventory for lazy helper source reads and poison tests.
    # Only this dedicated process uses these imported helper globals.
    source = replace_once(base.__source__, "self.access[path] = {'sha256': sha256(path)}",
                          "self.access[str(snapshot_path(path))] = {'sha256': sha256(snapshot_path(path)), 'original_path': path}")
    exec(compile(source, base.__file__, 'exec'), base.__dict__)
    base.source_guard, base.snapshot_path = source_guard, snapshot_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latency-hours', type=int, choices=(0, 2), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--leaderboard-root', type=Path, required=True)
    parser.add_argument('--reference-input-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = str(args.threads)
    root, out, snapshot = args.leaderboard_root.resolve(), args.output_dir.resolve(), args.reference_input_dir.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    with threadpool_limits(limits=args.threads):
        base, proxy, code_hashes = load_recipes(root, args.latency_hours, args.threads)
        pin_training_inputs(base, root, out)
        proxy.Features, proxy.Fit, proxy.residual = base.Features, base.Fit, base.residual
        data = base.Features(proxy.END)
        code_hashes.update(data.code_hashes)
        identity = dict(ace_latency_h=args.latency_hours, seed=0, source_access=data.access,
                        code_hashes=code_hashes, script_sha256=sha256(__file__), config=proxy.CFG)
        identity_path = out / 'checkpoint_identity.json'
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError('Checkpoint identity differs from code/training sources/latency')
        else:
            proxy.dump(identity_path, identity)
        bank = build_bank(base, proxy, data, out, args.threads, args.resume)
        augmented = {}
        def factory(cutoff):
            key = str(cutoff)
            if key not in augmented:
                name = pd.Timestamp(cutoff).strftime('%Y%m%d%H')
                augmented[key] = cached_fit(base, proxy.Augmented(data, bank, cutoff), proxy.CFG, cutoff,
                                            args.threads, out / f'augmented_{name}.joblib', args.resume)
            return augmented[key]
        candidate = cached_replay(proxy, data, proxy.CUTOFF, factory, proxy.END,
                                   out / 'candidate_replay.joblib', args.resume)
        frame = prediction_frame(proxy.OFFICIAL, candidate.paths(proxy.OFFICIAL))
        comparison = (compare_original(frame, root.parent / 'research/expos/runs/E1149/predictions_official_s0.parquet')
                      if args.latency_hours == 2 else None)
        poison = proxy.poison(data, bank, candidate)
        proxy.dump(out / 'poison_s0.json', poison)
        if not poison['passed']:
            raise ValueError(f'Poison controls failed: {poison}')
        verify_sources(data.access)
        frame.to_parquet(out / 'predictions_official_s0.parquet', index=False)
        final_state = dict(
            identity=identity,
            bank={k: getattr(bank, k) for k in ('times', 'x', 'level', 'records', 'heads')},
            raw_models={k: {a: b for a, b in v.__dict__.items() if a != 'features'}
                        for k, v in bank.models.items()},
            raw_replays={k: {a: b for a, b in v.__dict__.items() if a not in ('data', 'model', 'cal')}
                         for k, v in bank.replays.items()},
            augmented_models={k: {a: b for a, b in v.__dict__.items() if a != 'features'}
                              for k, v in augmented.items()},
            candidate={k: v for k, v in candidate.__dict__.items() if k not in ('data', 'model', 'cal')})
        save_checkpoint(out / 'final_model_state.joblib', final_state)
        model_hash = sha256(out / 'final_model_state.joblib')
        del final_state
        reference_hashes = {str(p.relative_to(snapshot)): sha256(p) for p in sorted(snapshot.rglob('*')) if p.is_file()}
        origins, october = switch_to_reference(data, snapshot, args.latency_hours)
        for model in augmented.values():
            model.features.cache = {}
        # Raw replays for level heads retain their fitted pools and scales, but compute
        # current query paths from the newly installed source tables.
        prediction_frame(origins, candidate.paths(origins)).to_parquet(out / 'predictions_october_s0.parquet', index=False)
        for path, expected in reference_hashes.items():
            if sha256(snapshot / path) != expected:
                raise ValueError(f'Reference snapshot changed: {path}')
        verify_sources(identity['source_access'])
        provenance = dict(**identity, experiment='E1149' if args.latency_hours == 2 else 'E1149_NO_DELAY',
                          purpose='Seed-0 reference visualization; no registration or official scoring',
                          refit_from_training=True, model_sha256=model_hash, threads=args.threads, label_cutoff=str(proxy.CUTOFF),
                          label_maturity_hours=144, hindcast_lag_hours=12 + args.latency_hours,
                          oof=bank.diagnostic, replay=candidate.receipt, poison=poison,
                          original_comparison=comparison, official_origins=len(proxy.OFFICIAL),
                          october={**october, 'input_migration_status': 'pending_validation; reference use',
                                   'sources': reference_hashes},
                          train_feature_coverage=candidate.model.features.coverage,
                          excluded_initial_training_rows=candidate.model.features.excluded_rows,
                          predictions_sha256={p.name: sha256(p) for p in out.glob('predictions_*_s0.parquet')})
        proxy.dump(out / 'provenance.json', provenance)
        for period in ('official', 'october'):
            period_provenance = {**provenance, 'predictions_sha256': sha256(out / f'predictions_{period}_s0.parquet')}
            if period == 'october':
                period_provenance.update(october)
            proxy.dump(out / f'provenance_{period}.json', period_provenance)
        print(json.dumps({'completed': str(out), 'latency_hours': args.latency_hours,
                          'original_comparison': comparison, 'october': october}), flush=True)


if __name__ == '__main__':
    main()
