"""Publish the frozen follow-up model ablations through the existing leaderboard client.

Each invocation performs one durable step: build, dryrun, push, github, submit,
or verify. No scoring or publication decision changes a model configuration.
Release scripts explicitly replay archived forecasts; training code is archived
alongside them. This module never publishes input data or model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON = ROOT / '.venv/bin/python'
STRATEGIES = ('Recursive', 'DirRec', 'Direct', 'pooled')
MEMBER = 'seongeun'
MODELS = ('tabiclv2', 'tabldm', 'realmlp', 'lightgbm', 'mitra')
KUMO = HERE.parent / 'forecast_strategies_kumo'
NO_PRETRAINED = frozenset(('realmlp', 'lightgbm'))
MODEL_NAMES = {'tabiclv2': 'TabICLv2', 'tabldm': 'Xiaomi-TabLDM', 'realmlp': 'RealMLP',
               'lightgbm': 'LightGBM', 'mitra': 'Mitra-v2'}
GRID = {'F1': ('2026-02-28T23:00:00Z', '2026-05-28T23:00:00Z', 2137),
        'F2': ('2026-05-31T23:00:00Z', '2026-09-26T23:00:00Z', 2833)}
KEYS = ['origin_last_input_utc', 'horizon_hours', 'pred_kms']
METRICS = {'mse': 'mse', 'mse_f1': 'mse_f1', 'mse_f2': 'mse_f2', 'mse550': 'mse_ge550'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def config_sha(config):
    if set(config) != {'data', 'hyperparameters'} or not all(isinstance(v, dict) and v for v in config.values()):
        raise ValueError('config requires nonempty data and hyperparameters mappings')
    return hashlib.sha256(canonical(config).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def slug(name):
    if name not in STRATEGIES:
        raise ValueError(f'unknown forecast strategy: {name}')
    return name.lower()


def experiment(model, name):
    if model not in MODELS:
        raise ValueError('unknown regression model: ' + str(model))
    return f'e2000fs_{model}_' + slug(name)


def expected_origins():
    parts = [pd.date_range(a, b, freq='h', tz='UTC') for a, b, _ in GRID.values()]
    return parts[0].append(parts[1]).as_unit('ns')


def validate_frame(frame):
    """Require the exact ordered complete F1+F2 grid, including missing-label rows."""
    if list(frame.columns) != KEYS:
        raise ValueError('prediction columns differ from the publication contract')
    origins = expected_origins()
    if len(frame) != len(origins) * 72:
        raise ValueError('prediction grid must contain 357840 rows')
    actual = pd.to_datetime(frame.origin_last_input_utc, utc=True).array.asi8
    if not np.array_equal(actual, np.repeat(origins.asi8, 72)):
        raise ValueError('prediction origin grid differs or is not ordered')
    if not np.array_equal(frame.horizon_hours.to_numpy(), np.tile(np.arange(1, 73), len(origins))):
        raise ValueError('prediction horizons differ or are duplicated')
    values = frame.pred_kms.to_numpy(np.float64)
    if not np.isfinite(values).all() or not ((values >= 1) & (values <= 3000)).all():
        raise ValueError('predictions must be finite and in [1,3000] km/s')
    return origins, values.reshape(-1, 72)


def require_hashes(manifest):
    for field in ('input_sha256', 'source_sha256', 'model_sha256'):
        hashes = manifest.get(field)
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError(f'manifest requires nonempty {field}')
        if any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v) for v in hashes.values()):
            raise ValueError(f'manifest has invalid {field}')
    if not isinstance(manifest.get('config'), dict) or not manifest['config']:
        raise ValueError('manifest requires frozen model config')


def description(model, name):
    structures = {'Recursive': 'one shared scalar model recursively rolled out for 72 hours',
                  'DirRec': '72 scalar models receiving earlier predicted horizons',
                  'Direct': '72 independent scalar models using horizon-specific features',
                  'pooled': 'one shared scalar model with a horizon feature'}
    recipes = {
        'tabiclv2': 'TabICLv2 regression through SDM 0.2.1, eight estimators; mean of 999 quantiles',
        'tabldm': 'Xiaomi-TabLDM plain ICL, eight estimators, enhancement disabled, KV cache enabled',
        'realmlp': 'RealMLP TD, up to 256 epochs; head-specific internal selection epoch fixed for F1/F2 refits',
        'lightgbm': 'LightGBM CPU L2 regression, 100 trees, learning rate 0.1, 31 leaves; no early stopping',
        'mitra': 'Mitra-v2 official 50-step fine-tuning and eight-fold bagging; persistent fitted artifacts',
    }
    return (f'E2000FS {MODEL_NAMES[model]} model ablation: {structures[name]}. {recipes[model]}. '
            'Seed 42, latest 5000 context rows per scalar table, outputs clipped to [1,3000] km/s. '
            'ACE at origin (lag 0), causal forward fill, 24-hour speed history. '
            'F1 context targets before 2026-02-26; F2 before 2026-05-29. '
            'Internal validation 2025-12-04..2026-02-22; no official-score selection. '
            'All completed predeclared arms are published regardless of score. '
            'Kumo and XGBoost comparisons require identical input and context hashes. '
            'Direct/pooled use horizon-specific features, unlike the historical 1530-column Direct experiment. '
            'Inherited E2000 features were developed using later data, so pre-F1 selection provenance is '
            'not proven. One seed; published periods are not fresh holdouts. '
            'The pinned script replays archived forecasts, not model inference or retraining. '
            'Model inference timing is measured separately. License and model identity are in the manifest.')


REPLAY = '''"""Fixed-grid archived model-ablation forecast replay, not retraining or live inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

CONFIG_SHA256 = __SHA__

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', type=Path, default=Path(__file__).resolve().parents[3] / __RELEASE__)
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
'''


class Publisher:
    def __init__(self, model, out, publication, worktree):
        if model not in MODELS:
            raise ValueError("unknown regression model: " + str(model))
        self.model = model
        self.out, self.pub, self.work = (Path(p).resolve() for p in (out, publication, worktree))
        self._client = None

    def require_model_identity(self, manifest):
        if manifest.get('model') != self.model:
            raise ValueError('manifest model differs from requested publication model')
        if self.model in NO_PRETRAINED:
            expected = hashlib.sha256(canonical(manifest['config']).encode()).hexdigest()
            if manifest['model_sha256'] != {'no_pretrained_config': expected}:
                raise ValueError('nonpretrained model requires explicit configuration identity')
        elif 'no_pretrained_config' in manifest['model_sha256']:
            raise ValueError('pretrained model requires actual weight hashes')

    def require_predecessor(self):
        state = read_json(KUMO / 'outputs/supervision.json', {})
        verified = read_json(KUMO / 'publication/verify.json', {})
        if state.get('status') != 'completed':
            raise RuntimeError('Kumo supervision has not completed')
        records = verified.get('records', {})
        if set(records) != set(STRATEGIES) or any(
                r.get('predictions_identical') is not True or r.get('scores_match') is not True
                for r in records.values()):
            raise RuntimeError('Kumo remote prediction and score verification is incomplete')

    def ensure_worktree(self):
        """Create only this model's isolated checkout after predecessor verification."""
        self.require_predecessor()
        if not (self.work / '.git').exists():
            if self.work.exists() and any(self.work.iterdir()):
                raise ValueError('publication worktree directory is not an empty owned checkout')
            self.work.parent.mkdir(parents=True, exist_ok=True)
            repository = ROOT / 'leaderboard'
            subprocess.run(['git', '-C', str(repository), 'fetch', 'origin', 'main'], check=True)
            subprocess.run(['git', '-C', str(repository), 'worktree', 'add', '--detach',
                            str(self.work), 'origin/main'], check=True)
        remote = self.git('remote', 'get-url', 'origin', capture=True)
        if remote not in ('https://github.com/TLAB-HGU/sunrun-leaderboard.git',
                          'git@github.com:TLAB-HGU/sunrun-leaderboard.git'):
            raise ValueError('unexpected publication Git remote')
        if Path(self.git('rev-parse', '--show-toplevel', capture=True)).resolve() != self.work:
            raise ValueError('publication path is not a worktree root')
        for relative in ('submit.py', 'scorer/scoring.py'):
            if not (self.work / relative).is_file():
                raise ValueError('publication checkout lacks the protected client/scorer')

    def load(self, name, default=None):
        return read_json(self.pub / f'{name}.json', default)

    def save(self, name, value):
        write_json(self.pub / f'{name}.json', value)

    def batch(self):
        built = self.load('build', {})
        names = self.load('batch', [])
        if not names:
            raise ValueError('build a publication batch first')
        return {name: built[name] for name in names}

    def client(self):
        if self._client is None:
            path = self.work / 'submit.py'
            spec = importlib.util.spec_from_file_location('followup_publication_submit_' + self.model, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._client = module
        return self._client

    def check_item(self, item):
        name = item.get('strategy')
        if item.get('model') != self.model or name not in STRATEGIES:
            raise ValueError('publication receipt model/strategy differs')
        if item.get('experiment') != experiment(self.model, name):
            raise ValueError('publication experiment differs')
        if sha(item['predictions']) != item['prediction_sha256']:
            raise ValueError('published prediction file changed after build')
        meta = read_json(item['meta'])
        if config_sha(meta['config']) != item['config_sha256']:
            raise ValueError('publication configuration changed after build')
        if meta['config']['hyperparameters']['model'] != self.model:
            raise ValueError('publication configuration model differs')
        if meta.get('experiment') != item['experiment'] or meta.get('team_member') != MEMBER:
            raise ValueError('publication submission identity changed')
        for relative, digest in item['release_sha256'].items():
            if sha(self.work / relative) != digest:
                raise ValueError(f'publication asset changed after build: {relative}')
        return meta

    def require_step(self, step, name, item):
        record = self.load(step, {}).get(name, {})
        if record.get('config_sha256') != item['config_sha256']:
            raise ValueError(f'{name}: matching {step} receipt required')
        return record

    def build(self, names=None):
        manifest = read_json(self.out / 'manifest.json')
        require_hashes(manifest)
        self.require_model_identity(manifest)
        report = read_json(self.out / 'report.json')
        if not isinstance(report, dict) or not isinstance(report.get('rows'), dict):
            raise ValueError('report.json requires rows keyed by <model>_<strategy>')
        built = self.load('build', {})
        names = list(STRATEGIES) if names is None else list(names)
        if not names or len(set(names)) != len(names):
            raise ValueError('publication strategies must be nonempty and unique')
        for name in names:
            slug(name)
            if not (self.out / 'predictions/combined' / f'{name}.parquet').is_file():
                raise ValueError(f'missing completed strategy: {name}')
        self.ensure_worktree()
        if not names:
            raise ValueError('no complete prediction files')
        for name in names:
            slug(name)
            original = self.out / 'predictions/combined' / f'{name}.parquet'
            frame = pd.read_parquet(original)
            origins, values = validate_frame(frame)
            row = report['rows'][f'{self.model}_{name}']
            timing = float(row['inference_seconds_per_fold'])
            if not math.isfinite(timing) or timing <= 0:
                raise ValueError('positive finite model inference timing required')
            scores = {k: float(row[k]) for k in METRICS}
            if not all(math.isfinite(v) for v in scores.values()):
                raise ValueError('finite overall/F1/F2/550 scores required before publication')
            identity = hashlib.sha256(canonical({'manifest': manifest, 'row': row, 'prediction': sha(original)}).encode()).hexdigest()
            if name in built:
                if built[name]['build_identity'] != identity:
                    raise ValueError(f'{name}: build identity changed; preserve the existing publication')
                self.check_item(built[name])
                continue
            release_relative = Path(f'releases/e2000fs_{self.model}') / slug(name)
            release = self.work / release_relative
            if release.exists():
                raise ValueError(f'owned release path already exists without a matching receipt: {release}')
            self.pub.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='build-', dir=self.pub) as temp:
                staged = Path(temp) / 'release'
                staged.mkdir()
                archived = {}
                for original_path, digest in manifest['source_sha256'].items():
                    path = Path(original_path)
                    path = path if path.is_absolute() else ROOT / path
                    relative = path.resolve().relative_to(ROOT)
                    if sha(path) != digest:
                        raise ValueError(f'registered source changed: {original_path}')
                    target = staged / 'source' / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
                    archived[str(relative)] = digest
                np.savez(staged / 'forecast.npz', origins_ns=origins.asi8, predictions=values)
                config = {'data': {'study': f'e2000fs {self.model} scalar-model ablation', 'ace_lag_hours': 0,
                                  'prefill': 'causal forward fill', 'target': 'absolute ACE filled speed km/s',
                                  'f1_target_end_exclusive': '2026-02-26T00:00:00Z',
                                  'f2_target_end_exclusive': '2026-05-29T00:00:00Z',
                                  'selection_fit_target_end_exclusive': '2025-12-01T00:00:00Z',
                                  'validation_origin_range': ['2025-12-04T00:00:00Z', '2026-02-22T23:00:00Z'],
                                  'periods': {p: {'start': a, 'end': b, 'count': n} for p, (a, b, n) in GRID.items()},
                                  'input_sha256': manifest['input_sha256'],
                                  'source_sha256': manifest['source_sha256'],
                                  'archived_source_sha256': archived,
                                  'model_sha256': manifest['model_sha256'],
                                  'model_identity_kind': ('training_configuration_no_pretrained_weights'
                                                          if self.model in NO_PRETRAINED else 'pretrained_weights'),
                                  'asset': {'filename': 'forecast.npz', 'sha256': sha(staged / 'forecast.npz')},
                                  'manifest_sha256': sha(self.out / 'manifest.json'),
                                  'evaluation_dataset_revision': report.get('dataset_revision')},
                          'hyperparameters': {'strategy': name, 'model': self.model,
                                              'model_identity': manifest.get('model_identity', {}),
                                              'point_prediction': ('arithmetic mean of q001..q999'
                                                                   if self.model == 'tabiclv2' else 'regression mean'),
                                              'lag_window_hours': 24, 'output_range_kms': [1, 3000],
                                              'frozen_config': manifest['config']}}
                csha = config_sha(config)
                write_json(staged / 'config.json', config)
                meta = {'team_member': MEMBER, 'experiment': experiment(self.model, name), 'description': description(self.model, name),
                        'no_future_leakage': True, 'f1_training_data_end_utc': '2026-02-26T00:00:00Z',
                        'config': config, 'inference_seconds_per_fold': timing}
                write_json(staged / 'meta.yaml', meta)
                write_json(staged / 'run-manifest.json', manifest)
                write_json(staged / 'result.json', row)
                (staged / 'README.md').write_text(description(self.model, name) + '\n')
                script_relative = Path('scripts') / MEMBER / experiment(self.model, name) / f'{csha}.py'
                script = self.work / script_relative
                if script.exists():
                    raise ValueError('publication script already exists without receipt')
                script.parent.mkdir(parents=True, exist_ok=True)
                script.write_text(REPLAY.replace('__SHA__', repr(csha)).replace('__RELEASE__', repr(str(release_relative))))
                release.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(staged), release)
            replay = self.pub / 'replay' / f'{slug(name)}.parquet'
            subprocess.run([str(PYTHON), str(script), '--output', str(replay)], check=True, cwd=self.work)
            if not pd.read_parquet(replay).equals(frame):
                raise ValueError(f'{name}: archived replay differs from the original prediction')
            paths = [script] + sorted(p for p in release.rglob('*') if p.is_file())
            built[name] = {'model': self.model, 'strategy': name, 'build_identity': identity,
                           'experiment': experiment(self.model, name), 'config_sha256': csha,
                           'script': str(script_relative), 'release': str(release_relative),
                           'predictions': str(replay), 'prediction_sha256': sha(replay),
                           'meta': str(release / 'meta.yaml'), 'scores': scores,
                           'release_sha256': {str(p.relative_to(self.work)): sha(p) for p in paths}}
            self.save('build', built)
            print(f'{name}: built and replay verified', flush=True)
        self.save('batch', names)

    def dryrun(self):
        done = self.load('dryrun', {})
        for name, item in self.batch().items():
            result = self.client().submit(item['predictions'], self.check_item(item), dry_run=True, replace=False)
            self.compare_scores(item, result)
            done[name] = result
            self.save('dryrun', done)
        print(f'{len(self.batch())} dry-run scores verified', flush=True)

    @staticmethod
    def compare_scores(item, result):
        if result['config_sha256'] != item['config_sha256']:
            raise ValueError('client configuration hash differs')
        for local, remote in METRICS.items():
            if not np.isclose(item['scores'][local], result[remote], rtol=1e-9, atol=1e-9):
                raise ValueError(f'leaderboard {remote} differs from the frozen report')

    def git(self, *args, capture=False):
        result = subprocess.run(['git', '-C', str(self.work), *args], check=True,
                                text=True, capture_output=capture)
        return result.stdout.strip() if capture else None

    def push(self):
        batch = self.batch()
        paths = []
        for name, item in batch.items():
            self.check_item(item)
            self.require_step('dryrun', name, item)
            paths.extend(item['release_sha256'])
        # The worktree belongs to this study; refuse to accidentally include other staged work.
        staged = self.git('diff', '--cached', '--name-only', capture=True).splitlines()
        changed = self.git('diff', '--name-only', capture=True).splitlines()
        if (set(staged) | set(changed)) - set(paths):
            raise ValueError('publication worktree contains unrelated tracked changes')
        remote = self.git('remote', 'get-url', 'origin', capture=True)
        if remote not in ('https://github.com/TLAB-HGU/sunrun-leaderboard.git',
                          'git@github.com:TLAB-HGU/sunrun-leaderboard.git'):
            raise ValueError('unexpected publication Git remote')
        for relative in paths:
            path = self.work / relative
            if path.suffix in ('.py', '.json', '.yaml', '.md', '.txt', '.toml'):
                if re.search(rb'(?:hf_|ghp_)[A-Za-z0-9]{20,}', path.read_bytes()):
                    raise ValueError('credential-like value in release source')
        self.git('add', '--', *paths)
        if self.git('diff', '--cached', '--name-only', capture=True):
            self.git('commit', '-m', f'E2000FS: publish frozen {self.model} scalar ablations (F1+F2)')
        self.git('fetch', 'origin', 'main')
        # This isolated worktree contains only this study's newly committed assets.
        # Integrate unrelated upstream progress without replacing remote history.
        ancestor = subprocess.run(['git', '-C', str(self.work), 'merge-base', '--is-ancestor',
                                   'origin/main', 'HEAD'], check=False)
        if ancestor.returncode == 1:
            self.git('rebase', 'origin/main')
        elif ancestor.returncode != 0:
            raise RuntimeError('cannot verify publication Git ancestry')
        self.git('push', 'origin', 'HEAD:main')
        self.save('push', {'commit': self.git('rev-parse', 'HEAD', capture=True), 'strategies': list(batch)})

    def github(self):
        done = self.load('github', {})
        for name, item in self.batch().items():
            self.check_item(item)
            _, raw, page = self.client().script_location(MEMBER, item['experiment'], item['config_sha256'])
            with urllib.request.urlopen(raw, timeout=30) as response:
                remote = response.read()
            if remote != (self.work / item['script']).read_bytes():
                raise ValueError(f'{name}: remote script differs')
            done[name] = {'config_sha256': item['config_sha256'], 'url': page, 'identical': True}
            self.save('github', done)

    def submit(self):
        done = self.load('submission', {})
        for name, item in self.batch().items():
            meta = self.check_item(item)
            self.require_step('dryrun', name, item)
            self.require_step('github', name, item)
            if name in done:
                if done[name]['config_sha256'] != item['config_sha256']:
                    raise ValueError('existing submission identity differs')
                continue
            result = self.client().submit(item['predictions'], meta, dry_run=False, replace=False)
            # Persist even if an unexpected score change needs investigation after the external write.
            done[name] = result
            self.save('submission', done)
            self.compare_scores(item, result)
            print(f'{name}: submitted MSE={result["mse"]:.6f}', flush=True)

    def verify(self):
        from huggingface_hub import HfApi, hf_hub_download
        dataset = self.client().DATASET
        revision = HfApi().dataset_info(dataset).sha
        done = self.load('submission', {})
        records = {}
        for name, item in self.batch().items():
            self.check_item(item)
            submitted = done[name]
            if submitted['config_sha256'] != item['config_sha256']:
                raise ValueError('submission receipt identity differs')
            key = f'{MEMBER}/{item["experiment"]}/{item["config_sha256"]}'
            fetch = lambda path: hf_hub_download(dataset, path, repo_type='dataset', revision=revision)
            stored = fetch(f'submissions/{key}.parquet')
            result = read_json(fetch(f'results/{key}.json'))
            self.compare_scores(item, result)
            if not pd.read_parquet(stored).equals(pd.read_parquet(item['predictions'])):
                raise ValueError(f'{name}: remote predictions differ')
            records[name] = {'key': key, 'config_sha256': item['config_sha256'], 'predictions_identical': True,
                             'scores_match': True, 'mse': result['mse']}
        self.save('verify', {'dataset_revision': revision, 'records': records})
        print(f'{len(records)} remote prediction files and scores verified at {revision}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=('build', 'dryrun', 'push', 'github', 'submit', 'verify'))
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--publication', type=Path)
    parser.add_argument('--worktree', type=Path)
    parser.add_argument('--names', nargs='+', choices=STRATEGIES)
    args = parser.parse_args()
    out = args.out or HERE / 'outputs' / args.model
    publication = args.publication or HERE / 'publication' / args.model
    worktree = args.worktree or publication / 'worktree'
    publisher = Publisher(args.model, out, publication, worktree)
    if args.step == 'build':
        publisher.build(args.names)
    else:
        if args.names:
            parser.error('--names is for build; subsequent steps use the frozen batch')
        getattr(publisher, args.step)()


if __name__ == '__main__':
    main()
