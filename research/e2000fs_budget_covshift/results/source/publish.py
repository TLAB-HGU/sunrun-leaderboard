"""Archive and publish a completed fixed DirMO-24 run; never retrain or select by scores.

--prepare-only validates and builds a local release without network access.
Normal execution uses a dedicated detached worktree, the unchanged leaderboard
client, and durable publication receipts. The published script is explicitly an
archived forecast replay, while training source and checkpoints are preserved.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = 'e2000fs_budget_covshift'
REPOSITORY = ROOT / 'leaderboard/model-release-publish'
HELPER = ROOT / 'analysis/forecast_strategies_kumo/publish.py'
spec = importlib.util.spec_from_file_location('dirmo_pointweight_publication_helpers', HELPER)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

DESCRIPTION = (
    'Single preregistered DA-CART covariate-shift adaptation of DirMO24 pointweight4. '
    'Historical source includes all pre-cutoff contexts, recent target is last81days; '
    'blockedOOF binary depth2 classifier80rounds estimates prior-corrected p_recent/p_all. '
    'Ratio clipped [.25,4], with calendar-year x horizon x speed-band mass normalization. '
    'Same frozen237columns, three vector24 XGBoost900rounds depth3 lr.03 seed42 CPU3. '
    'ACE lag0; all train targets mature72h, vector origin+144h<=cutoff for both arms. '
    'Paired maturity-corrected point4 controls D1/D2/D3/C/F1/F2; internal checks recorded '
    'without stopping official submission. No transduction or official-score tuning. '
    'Stable p(y|x) and support overlap required; no concept-drift guarantee. '
    'Historical parent3556.1984/5923.5239 used a different maturity boundary. '
    'Top30 and evaluation periods are official-exposed; archived forecast replay.'
)


def copy_immutable(source, target):
    """Resume matching artifacts without replacing an existing different release."""
    source, target = Path(source), Path(target)
    if target.exists():
        if P.sha(source) != P.sha(target):
            raise ValueError(f'existing publication artifact differs: {target}')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def prepare(out):
    """Build and replay locally; no Git, Hugging Face or other network writes."""
    out = Path(out).resolve()
    pub = out / 'publication'
    package = pub / 'package'
    original = out / 'predictions.parquet'
    manifest = P.read_json(out / 'manifest.json')
    state = P.read_json(out / 'status.json', {})
    if (manifest['experiment'] != EXPERIMENT or manifest.get('smoke', False)
            or state.get('status') != 'completed' or state.get('smoke', True)):
        raise ValueError('only the completed full DirMO-24 experiment may be published')
    frame = pd.read_parquet(original)
    origins, values = P.validate_frame(frame)
    results = P.read_json(out / 'results.json')
    official = results['combined']['official']
    timing = P.read_json(out / 'timing.json')
    seconds = float(timing['inference_seconds_per_fold'])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('positive finite inference timing is required')
    if not math.isfinite(float(official['mse'])):
        raise ValueError('finite official MSE is required')
    sources = dict(manifest['source_sha256'])
    sources.setdefault(str(Path(__file__).resolve()), P.sha(__file__))
    sources.setdefault(str(HELPER), P.sha(HELPER))
    for name, expected in sources.items():
        if P.sha(name) != expected:
            raise ValueError(f'run source changed: {name}')
    models = {f'models/{stage}_{arm}_b{block}.ubj': P.sha(out / 'models' / f'{stage}_{arm}_b{block}.ubj')
              for stage in ('D1', 'D2', 'D3', 'C', 'f1', 'final') for arm in ('parent', 'candidate') for block in range(3)}
    losses = {f'losses/{stage}_{arm}_b{block}.csv': P.sha(out / 'losses' / f'{stage}_{arm}_b{block}.csv')
              for stage in ('D1', 'D2', 'D3', 'C', 'f1', 'final') for arm in ('parent', 'candidate') for block in range(3)}
    identity = P.config_sha({
        'data': {'manifest': P.sha(out / 'manifest.json'), 'prediction': P.sha(original)},
        'hyperparameters': {'results': P.sha(out / 'results.json'), 'timing': P.sha(out / 'timing.json'),
                            'sources': sources, 'models': models, 'losses': losses},
    })
    old = P.read_json(pub / 'prepared.json')
    if old is not None:
        if old['build_identity'] != identity:
            raise ValueError('run changed since publication preparation')
        Publisher(out, pub, package).check_item(old)
        return old
    relative = Path('releases') / EXPERIMENT
    release = package / relative
    release.mkdir(parents=True, exist_ok=True)
    archived = {}
    for name, expected in sources.items():
        source = Path(name).resolve()
        if P.sha(source) != expected:
            raise ValueError(f'run source changed: {source}')
        key = str(source.relative_to(ROOT))
        copy_immutable(source, release / 'source' / key)
        archived[key] = expected
    for key in (*models, *losses):
        copy_immutable(out / key, release / key)
    for name in ('manifest.json', 'results.json', 'timing.json', 'predictions.parquet', 'parent_predictions.parquet', 'model_receipts.json'):
        copy_immutable(out / name, release / name)
    for evidence in sorted((out / 'diagnostics').glob('*')):
        copy_immutable(evidence, release / 'diagnostics' / evidence.name)
    np.savez_compressed(release / 'forecast.npz', origins_ns=origins.asi8, predictions=values)
    config = {
        'data': {
            'selected': manifest['selected'],
            'input_identity': manifest['input_identity'], 'block_hours': 24,
            'ace_lag_hours': 0, 'columns_per_block': 237,
            'periods': {k: dict(start=a, end=b, count=n) for k, (a, b, n) in P.GRID.items()},
            'asset': dict(filename='forecast.npz', sha256=P.sha(release / 'forecast.npz')),
            'archived_source_sha256': archived, 'model_sha256': models, 'loss_sha256': losses,
            'run_manifest_sha256': P.sha(out / 'manifest.json'),
            'original_prediction_sha256': P.sha(original),
        },
        'hyperparameters': dict(recipe=manifest['recipe'], parent='e2000fs_dirmo_24_feat30_pointweight4',
                                comparator='same-row maturity-corrected point4'),
    }
    csha = P.config_sha(config)
    meta = dict(team_member=P.MEMBER, experiment=EXPERIMENT, description=DESCRIPTION + ' Internal decision: ' + str(results['decision']),
                no_future_leakage=True, f1_training_data_end_utc='2026-02-26T00:00:00Z',
                config=config, inference_seconds_per_fold=seconds)
    P.write_json(release / 'config.json', config)
    P.write_json(release / 'meta.yaml', meta)
    (release / 'README.md').write_text(DESCRIPTION + '\n')
    script_relative = Path('scripts') / P.MEMBER / EXPERIMENT / f'{csha}.py'
    script = package / script_relative
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(P.REPLAY.replace('Kumo', 'DirMO-24 pointweight4').replace('__SHA__', repr(csha))
                      .replace('__RELEASE__', repr(str(relative))))
    replay = pub / 'replay.parquet'
    subprocess.run([str(P.PYTHON), str(script), '--output', str(replay)], check=True)
    if not pd.read_parquet(replay).equals(frame):
        raise ValueError('archived replay differs from original predictions')
    files = [script, *sorted(p for p in release.rglob('*') if p.is_file())]
    item = dict(build_identity=identity, experiment=EXPERIMENT, config_sha256=csha,
                script=str(script_relative), release=str(relative), predictions=str(replay),
                prediction_sha256=P.sha(replay), meta=str(release / 'meta.yaml'),
                scores=official,
                release_sha256={str(p.relative_to(package)): P.sha(p) for p in files})
    P.write_json(pub / 'prepared.json', item)
    return item


class Publisher(P.Publisher):
    @staticmethod
    def compare_scores(item, result):
        if result['config_sha256'] != item['config_sha256']:
            raise ValueError('client configuration hash differs')
        for key in ('mse', 'mse_f1', 'mse_f2', 'mse_ge550'):
            if not np.isclose(item['scores'][key], result[key], rtol=1e-9, atol=1e-9):
                raise ValueError(f'published {key} differs from frozen run score')

    def git(self, *args, capture=False):
        if args and args[0] == 'commit':
            args = ('commit', '-m', 'E2000FS: publish fixed covshift density-ratio experiment')
        return super().git(*args, capture=capture)


def install_worktree(out, item):
    pub = out / 'publication'
    work = pub / 'worktree'
    if not work.exists():
        subprocess.run(['git', '-C', str(REPOSITORY), 'fetch', 'origin', 'main'], check=True)
        subprocess.run(['git', '-C', str(REPOSITORY), 'worktree', 'add', '--detach',
                        str(work), 'origin/main'], check=True)
    actual = subprocess.check_output(['git', '-C', str(work), 'rev-parse', '--show-toplevel'], text=True)
    if Path(actual.strip()).resolve() != work.resolve() or not (work / '.git').is_file():
        raise ValueError('dedicated publication worktree identity differs')
    for relative in item['release_sha256']:
        copy_immutable(pub / 'package' / relative, work / relative)
    installed = dict(item, meta=str(work / item['release'] / 'meta.yaml'))
    P.write_json(pub / 'build.json', {EXPERIMENT: installed})
    P.write_json(pub / 'batch.json', [EXPERIMENT])
    return Publisher(out, pub, work)


def recover_submission(publisher):
    """Recover an interrupted successful upload without replacing its remote key."""
    if publisher.load('submission', {}).get(EXPERIMENT):
        return
    from huggingface_hub import HfApi, hf_hub_download
    item = publisher.batch()[EXPERIMENT]
    dataset = publisher.client().DATASET
    key = f'{P.MEMBER}/{EXPERIMENT}/{item["config_sha256"]}'
    remote = f'results/{key}.json'
    api = HfApi()
    revision = api.dataset_info(dataset).sha
    if not api.file_exists(dataset, remote, repo_type='dataset', revision=revision):
        return
    result = P.read_json(hf_hub_download(dataset, remote, repo_type='dataset', revision=revision))
    publisher.compare_scores(item, result)
    prediction = hf_hub_download(dataset, f'submissions/{key}.parquet', repo_type='dataset', revision=revision)
    if not pd.read_parquet(prediction).equals(pd.read_parquet(item['predictions'])):
        raise ValueError('existing remote submission differs from frozen predictions')
    publisher.save('submission', {EXPERIMENT: result})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    pub = out / 'publication'
    pub.mkdir(parents=True, exist_ok=True)
    with (pub / 'publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        item = prepare(out)
        if args.prepare_only:
            print(f'Local publication prepared: {item["config_sha256"]}', flush=True)
            return
        import os
        shared_lock = os.open(REPOSITORY / '.git', os.O_RDONLY)
        try:
            fcntl.flock(shared_lock, fcntl.LOCK_EX)
            publisher = install_worktree(out, item)
            publisher.dryrun()
            publisher.push()
            publisher.github()
            recover_submission(publisher)
            publisher.submit()
            publisher.verify()
        finally:
            os.close(shared_lock)


if __name__ == '__main__':
    main()
