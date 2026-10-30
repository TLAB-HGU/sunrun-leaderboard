"""Package and publish the frozen HPO winner using the existing leaderboard client."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = 'e2000fs_dirmo_24_feat30_lgbm_HPO'
REMOTE = 'https://github.com/TLAB-HGU/sunrun-leaderboard.git'
CACHE = ROOT / 'analysis/forecast_strategies_e2000/outputs/cache'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


existing = load_module('hpo_existing_publisher', ROOT / 'analysis/forecast_strategies_kumo/publish.py')
read, write, sha = existing.read_json, existing.write_json, existing.sha


class Publisher(existing.Publisher):
    def git(self, *args, capture=False):
        if args and args[0] == 'commit':
            args = ('commit', '-m', 'E2000FS: publish LightGBM HPO winner (3 studies x 50 trials)')
        return super().git(*args, capture=capture)


def prepare(work):
    """Create an isolated checkout; never reset or clean an existing checkout."""
    if work.exists():
        remote = subprocess.check_output(['git', '-C', str(work), 'remote', 'get-url', 'origin'], text=True).strip()
        if remote != REMOTE or not (work / 'submit.py').is_file():
            raise ValueError('existing publication checkout has an unexpected remote or client')
        return
    work.parent.mkdir(parents=True, exist_ok=True)
    repository = ROOT / 'leaderboard'
    remote = subprocess.check_output(['git', '-C', str(repository), 'remote', 'get-url', 'origin'], text=True).strip()
    if remote != REMOTE:
        raise ValueError('unexpected parent publication repository remote')
    subprocess.run(['git', '-C', str(repository), 'fetch', 'origin', 'main'], check=True)
    subprocess.run(['git', '-C', str(repository), 'worktree', 'add', '--detach', str(work), 'origin/main'], check=True)


def frozen_inputs(campaign):
    final = campaign / 'final'
    if read(final / 'status.json', {}).get('status') != 'completed':
        raise ValueError('completed final training is required')
    manifest = read(final / 'manifest.json')
    best = read(campaign / 'best.json')
    if not isinstance(best, dict) or manifest.get('hpo_selection') != best:
        raise ValueError('final training selection differs from frozen HPO winner')
    if manifest.get('smoke') or manifest.get('experiment') != EXPERIMENT:
        raise ValueError('full HPO experiment manifest required')
    if len(set(manifest['selected'])) != 30 or not manifest.get('posthoc_official_exposed'):
        raise ValueError('frozen Top30 and official-exposed disclosure required')
    for key, value in best['params'].items():
        if manifest['params'].get(key) != value:
            raise ValueError(f'final model parameter differs from HPO winner: {key}')
    if not math.isfinite(float(best['value'])):
        raise ValueError('finite internal selection score required')
    winners = best.get('study_winners', [])
    if (len(winners) != 3 or {w['study'] for w in winners} != {1, 2, 3}
            or any(w['completed'] != 50 for w in winners)):
        raise ValueError('three completed studies with 50 successful trials each required')
    winner = min(winners, key=lambda w: (w['value'], w['study'], w['trial']))
    if any(best[k] != winner[k] for k in ('study', 'trial', 'value', 'params')):
        raise ValueError('selected winner differs from study minima')
    for filename, digest in manifest['hashes'].items():
        if sha(filename) != digest:
            raise ValueError(f'frozen input or source changed: {filename}')
    models = [final / 'artifacts' / f'{stage}_b{h // 24:02}_h{h + 1:02}.txt'
              for stage in ('f1', 'final') for h in range(72)]
    if not all(p.is_file() for p in models):
        raise ValueError('144 saved horizon models required')
    frame = pd.read_parquet(final / 'artifacts/predictions.parquet')
    existing.validate_frame(frame)
    scores = read(final / 'artifacts/scores.json')
    if not all(math.isfinite(float(scores[k])) for k in existing.METRICS.values()):
        raise ValueError('finite official scores required')
    return final, manifest, best, models, frame, scores


def measure_timing(final, manifest, frame):
    """Measure real CPU inference, separately from the public forecast replay."""
    import lightgbm as lgb
    baseline = load_module('hpo_frozen_design', ROOT / 'analysis/e2000fs_dirmo_24_feat30_lgbm/run.py')
    names = read(CACHE / 'names.json')
    expected = frame.pred_kms.to_numpy().reshape(4970, 72)
    receipt = {
        'scope': '32 evenly spaced origins per F1/F2; one warmup and three timed batches. '
                 'Includes block feature construction and 72-model CPU prediction/clipping with four threads; '
                 'excludes cache loading, model loading, training and output serialization. '
                 'Per-fold timing is median batch latency / 32, weighted by official period sizes.',
        'platform': platform.platform(), 'python': sys.version, 'lightgbm': lgb.__version__,
        'threads': 4, 'script_sha256': sha(__file__), 'stages': {},
    }
    offset = 0
    for stage, grid, count in (('f1', 'f1', 2137), ('final', 'official', 2833)):
        with np.load(CACHE / f'{grid}.npz', allow_pickle=False) as z:
            indices = np.linspace(0, len(z['origins']) - 1, 32).astype(int)
            bundle = {key: z[key][indices] for key in ('C', 'E', 'origins')}
            if len(z['origins']) != count:
                raise ValueError('timing cache period size differs')
            actual_origins = pd.DatetimeIndex(z['origins'], tz='UTC').as_unit('ns')
            np.testing.assert_array_equal(actual_origins.asi8, existing.expected_origins()[offset:offset + count].asi8)
        paths = [final / 'artifacts' / f'{stage}_b{h // 24:02}_h{h + 1:02}.txt' for h in range(72)]
        models = [lgb.Booster(model_file=str(p)) for p in paths]

        def infer():
            blocks = [baseline.design(bundle, names, manifest['selected'], start) for start in (0, 24, 48)]
            return np.clip(np.column_stack([m.predict(blocks[h // 24], num_threads=4)
                                           for h, m in enumerate(models)]), 1., 3000.)

        target = expected[offset:offset + count][indices]
        np.testing.assert_array_equal(infer(), target)
        elapsed = []
        for _ in range(3):
            start = time.perf_counter()
            prediction = infer()
            elapsed.append(time.perf_counter() - start)
            np.testing.assert_array_equal(prediction, target)
        receipt['stages'][stage] = {
            'sample_indices': indices.tolist(), 'batch_seconds': elapsed,
            'official_period_origins': count, 'inference_seconds_per_fold': float(np.median(elapsed) / 32),
            'prediction_parity': 'exact', 'model_sha256': {str(p): sha(p) for p in paths},
        }
        offset += count
    receipt['inference_seconds_per_fold'] = sum(v['inference_seconds_per_fold'] * v['official_period_origins']
                                               for v in receipt['stages'].values()) / 4970
    return receipt


def timing(campaign, publication):
    final, manifest, _, _, frame, _ = frozen_inputs(campaign)
    receipt = measure_timing(final, manifest, frame)
    receipt['manifest_sha256'] = sha(final / 'manifest.json')
    receipt['prediction_sha256'] = sha(final / 'artifacts/predictions.parquet')
    write(publication / 'timing.json', receipt)
    return receipt


def build(campaign, publication, work):
    final, manifest, best, models, frame, scores = frozen_inputs(campaign)
    publisher = Publisher(final, publication, work)
    identity = sha(final / 'manifest.json') + sha(final / 'artifacts/predictions.parquet')
    previous = publisher.load('build', {}).get('hpo')
    if previous:
        if previous['build_identity'] != identity:
            raise ValueError('publication already contains a different winner')
        publisher.check_item(previous)
        return
    if not (work / 'submit.py').is_file():
        raise ValueError('run prepare before build')
    relative = Path('releases') / EXPERIMENT
    release = work / relative
    if release.exists():
        raise ValueError('release already exists without a build receipt; preserve and inspect it')
    measured = read(publication / 'timing.json')
    if measured is None:
        measured = timing(campaign, publication)
    if (measured['manifest_sha256'] != sha(final / 'manifest.json')
            or measured['prediction_sha256'] != sha(final / 'artifacts/predictions.parquet')):
        raise ValueError('timing receipt belongs to different final results')
    for stage in measured.get('stages', {}).values():
        for filename, digest in stage['model_sha256'].items():
            if sha(filename) != digest:
                raise ValueError('timed model changed')
    release.mkdir(parents=True)
    archived = {}
    for filename, digest in manifest['hashes'].items():
        source = Path(filename)
        if source.suffix != '.py':
            continue
        name = str(source.relative_to(ROOT))
        target = release / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        archived[name] = digest
    for index in (1, 2, 3):
        for filename in ('identity.json', 'best.json', 'status.json', 'trials.csv'):
            source = campaign / f'study{index}' / filename
            target = release / 'hpo' / f'study{index}' / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    origins, values = existing.validate_frame(frame)
    np.savez(release / 'forecast.npz', origins_ns=origins.asi8, predictions=values)
    config = {
        'data': {
            'feature_names': manifest['selected'], 'ace_lag_hours': 0, 'prefill': 'causal forward fill',
            'target': 'absolute filled ACE speed km/s',
            'feature_selection': 'Inherited post-hoc official F1/F2 Top30 importance; official-exposed.',
            'f1_target_end_exclusive': '2026-02-26T00:00:00Z',
            'f2_target_end_exclusive': '2026-05-29T00:00:00Z',
            'periods': {p: {'start': a, 'end': b, 'count': n} for p, (a, b, n) in existing.GRID.items()},
            'input_and_source_sha256': manifest['hashes'], 'archived_source_sha256': archived,
            'trained_model_sha256': {str(p.relative_to(ROOT)): sha(p) for p in models},
            'manifest_sha256': sha(final / 'manifest.json'), 'evaluation_revision': manifest['evaluation_revision'],
            'asset': {'filename': 'forecast.npz', 'sha256': sha(release / 'forecast.npz')},
        },
        'hyperparameters': {
            'strategy': EXPERIMENT, 'base_model_parameters': manifest['params'], 'seed': 42, 'threads': 4,
            'selection': best, 'method': '72 independent LightGBM models sharing 24-hour block inputs',
            'hpo': 'Three parallel TPE studies, seeds 42/43/44, 50 successful trials each. '
                   'Minimum full 72-horizon internal validation MSE; ties by study then trial.',
            'sampler_seed_policy': 'Reinitialize TPE for each attempted trial with study_seed + 1009 * '
                                   'attempt_index, retaining study history for restart-stable suggestions.',
            'validation_origins': ['2025-12-04T00:00:00Z', '2026-02-22T23:00:00Z'],
            'hpo_training_target_end_exclusive': '2025-12-04T00:00:00Z',
        },
    }
    description = (f'{EXPERIMENT}. {config["hyperparameters"]["method"]}. '
                   'HPO selected on internal pre-F1 temporal validation with a 72-hour target maturity gap; '
                   'official scores were evaluated only after selecting the winner. '
                   'F1 training targets before 2026-02-26; F2 before 2026-05-29. '
                   'DISCLOSURE: Top30 features were inherited from already-scored official F1/F2 '
                   'permutation importance; E2000 recipe includes later-data research. '
                   'Post-hoc official-exposed research, not an untouched holdout. '
                   'Published script replays hash-pinned archived forecasts, not live model inference; '
                   'training source and HPO selection are archived. ' + measured['scope'])
    csha = existing.config_sha(config)
    meta = {'team_member': 'seongeun', 'experiment': EXPERIMENT, 'description': description,
            'no_future_leakage': True, 'f1_training_data_end_utc': '2026-02-26T00:00:00Z',
            'config': config, 'inference_seconds_per_fold': measured['inference_seconds_per_fold']}
    for filename, obj in [('config.json', config), ('meta.yaml', meta), ('run-manifest.json', manifest),
                          ('result.json', scores), ('timing.json', measured), ('hpo-selection.json', best)]:
        write(release / filename, obj)
    (release / 'README.md').write_text(description + '\n')
    script_relative = Path('scripts/seongeun') / EXPERIMENT / f'{csha}.py'
    script = work / script_relative
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(existing.REPLAY.replace('Kumo', EXPERIMENT).replace('__SHA__', repr(csha))
                      .replace('__RELEASE__', repr(str(relative))))
    replay = publication / 'replay/hpo.parquet'
    subprocess.run([str(existing.PYTHON), str(script), '--output', str(replay)], check=True)
    if not pd.read_parquet(replay).equals(frame):
        raise ValueError('full-grid archived replay differs')
    paths = [script] + sorted(p for p in release.rglob('*') if p.is_file())
    item = {'build_identity': identity, 'experiment': EXPERIMENT, 'config_sha256': csha,
            'script': str(script_relative), 'release': str(relative), 'predictions': str(replay),
            'prediction_sha256': sha(replay), 'meta': str(release / 'meta.yaml'),
            'scores': {local: scores[remote] for local, remote in existing.METRICS.items()},
            'release_sha256': {str(p.relative_to(work)): sha(p) for p in paths}}
    publisher.save('build', {'hpo': item})
    publisher.save('batch', ['hpo'])
    print('HPO winner built; exact replay and real CPU inference parity verified', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['prepare', 'timing', 'build', 'dryrun', 'push', 'github', 'submit', 'verify'])
    parser.add_argument('--campaign', type=Path, default=HERE / 'runs/main')
    parser.add_argument('--publication', type=Path, default=HERE / 'publication')
    args = parser.parse_args()
    work = args.publication / 'worktree'
    sys.path.insert(0, str(work))
    if args.step == 'prepare':
        prepare(work)
    elif args.step == 'timing':
        timing(args.campaign, args.publication)
    elif args.step == 'build':
        build(args.campaign, args.publication, work)
    else:
        getattr(Publisher(args.campaign / 'final', args.publication, work), args.step)()


if __name__ == '__main__':
    main()
