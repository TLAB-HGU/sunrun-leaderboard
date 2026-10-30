"""Frozen scalar-regression adapters; imports and weights are resolved at execution.

The caller owns GPU leasing, chronological contexts, stage/head checkpoints and
installation. No adapter installs dependencies or downloads observations.
"""
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

PINS = {'tabiclv2': {'structured-data-models': '0.2.1'},
        'tabldm': {'Xiaomi-TabLDM': '0.1.0'}, 'realmlp': {'pytabkit': '1.7.3'},
        'lightgbm': {'lightgbm': '4.7.0'}}
WEIGHTS = {
    'tabiclv2': {'repo': 'jingang/TabICL', 'revision': '4dcd344ece2c00be9e831fdd35bed57b5ad83e19',
                'filename': 'tabicl-regressor-v2-20260212.ckpt'},
    'tabldm': {'repo': 'occams/Xiaomi-TabLDM', 'revision': '712e5635be9fd8728084cd5ae105ddecb31dc03c',
              'filename': 'checkpoints/reg_default.ckpt'},
}
TABLDM_CODE_REVISION = '03cfbb9e7069b821dcef32676dc19b40d294e348'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    tmp.replace(path)


def require_versions(model):
    for package, expected in PINS[model].items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(f'{package} must be {expected}; found {actual}')
    if model == 'tabldm':
        dist = importlib.metadata.distribution('Xiaomi-TabLDM')
        direct = json.loads(dist.read_text('direct_url.json') or '{}')
        commit = direct.get('vcs_info', {}).get('commit_id')
        if commit != TABLDM_CODE_REVISION:
            raise RuntimeError('TabLDM must be installed from the pinned official Git revision')


def prepare_weights(model):
    """Explicit preparation step; returns pinned local checkpoint identity."""
    if model not in WEIGHTS:
        return {}
    from huggingface_hub import hf_hub_download
    ref = WEIGHTS[model]
    path = hf_hub_download(ref['repo'], ref['filename'], revision=ref['revision'])
    return dict(ref, path=str(Path(path).resolve()), sha256=sha(path))


def verified_weight(model, config):
    ref = config.get('weights')
    if ref is None:
        from huggingface_hub import hf_hub_download
        expected = WEIGHTS[model]
        path = hf_hub_download(expected['repo'], expected['filename'], revision=expected['revision'],
                               local_files_only=True)
        ref = dict(expected, path=str(Path(path).resolve()), sha256=sha(path))
        config['weights'] = ref
    if not isinstance(ref, dict):
        raise ValueError('invalid prepared weight identity')
    for key, expected in WEIGHTS[model].items():
        if ref.get(key) != expected:
            raise ValueError(f'{model} checkpoint {key} differs')
    path = Path(ref['path'])
    if sha(path) != ref['sha256']:
        raise ValueError('checkpoint bytes changed')
    return path


def gpu(config):
    if config.get('physical_gpu', 1) != 1 or os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('this campaign requires leased physical GPU1 (CUDA_VISIBLE_DEVICES=1)')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('physical GPU1 unavailable')
    if Path('/usr/bin/gcc-12').is_file():
        os.environ.setdefault('CC', '/usr/bin/gcc-12')
    torch.set_num_threads(config.get('n_jobs', 4))
    torch.cuda.reset_peak_memory_stats()
    return torch


def finite_vector(value, n):
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if value.shape != (n,) or not np.isfinite(value).all():
        raise ValueError('model produced invalid scalar predictions')
    return value


def quantile_mean(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 999 or not np.isfinite(values).all():
        raise ValueError('TabICLv2 must return 999 finite quantiles')
    return values.mean(axis=1)


class BatchedGPU:
    def predict(self, X):
        batch_size = self.config.get('batch', 256)
        parts, start = [], 0
        while start < len(X):
            end = min(start + batch_size, len(X))
            try:
                pred = finite_vector(self._predict(X[start:end]), end - start)
            except self.torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                self.torch.cuda.empty_cache()
                self.metadata.setdefault('oom_batch_sizes', []).append(batch_size)
                continue
            parts.append(pred)
            start = end
        self.torch.cuda.synchronize()
        self.metadata['peak_gpu_memory_bytes'] = int(self.torch.cuda.max_memory_allocated())
        self.metadata['query_batch_actual'] = batch_size
        return np.concatenate(parts)

    def clear(self):
        if getattr(self, 'model', None) is not None and hasattr(self.model, 'clear'):
            self.model.clear()
        self.model = None
        self.torch.cuda.empty_cache()


class TabICLModel(BatchedGPU):
    def __init__(self, config, stage, selection_path=None):
        require_versions('tabiclv2')
        self.config, self.torch = config, gpu(config)
        weight = verified_weight('tabiclv2', config)
        module = importlib.import_module('sdm.models.tabiclv2.model')

        def pinned(repo, filename, *args, **kwargs):
            if repo != WEIGHTS['tabiclv2']['repo'] or filename != WEIGHTS['tabiclv2']['filename']:
                raise ValueError('unexpected TabICLv2 checkpoint request')
            return str(weight)

        with patch.object(module, 'download_checkpoint', pinned):
            self.model = module.TabICLv2(task='regression', device='cuda')
        self.metadata = {'model': 'tabiclv2', 'weights': dict(config['weights']),
                         'point_prediction': 'mean(q001..q999)', 'loss_history': 'not applicable: ICL'}

    def table(self, X, target=False):
        import sdm
        X = np.asarray(X, dtype=np.float32)
        columns = ['target'] if target else [f'f{i:03}' for i in range(X.shape[1])]
        return sdm.TableTensor.from_pandas(pd.DataFrame(X, columns=columns),
                                          stypes={c: sdm.Stype.numerical for c in columns}, device='cuda')

    def fit(self, X, y, Xv, yv, directory):
        torch = self.torch
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
            self.model.fit(x=self.table(X), y=self.table(np.asarray(y)[:, None], True),
                           num_estimators=self.config.get('estimators', 8), estimator_batch_size=1,
                           generator=torch.Generator(device='cuda').manual_seed(self.config.get('seed', 42)))
        torch.cuda.synchronize()

    def _predict(self, X):
        with self.torch.inference_mode(), self.torch.amp.autocast('cuda', dtype=self.torch.float16):
            return quantile_mean(self.model.predict(self.table(X)).numerical.float().cpu().numpy())


class TabLDMModel(BatchedGPU):
    def __init__(self, config, stage, selection_path=None):
        require_versions('tabldm')
        self.config, self.torch = config, gpu(config)
        weight = verified_weight('tabldm', config)
        from tabldm import TabLDMRegressor
        self.model = TabLDMRegressor(n_estimators=config.get('estimators', 8),
                                     enhance_candidates=False, kv_cache=True,
                                     model_path=weight, allow_auto_download=False,
                                     device='cuda', random_state=config.get('seed', 42),
                                     n_jobs=config.get('n_jobs', 4))
        self.metadata = {'model': 'tabldm', 'weights': dict(config['weights']),
                         'point_prediction': 'official mean', 'recipe': 'plain ICL',
                         'loss_history': 'not applicable: ICL'}

    def fit(self, X, y, Xv, yv, directory):
        self.model.fit(X, y)
        self.torch.cuda.synchronize()

    def _predict(self, X):
        return self.model.predict(X, output_type='mean')


class LightGBMModel:
    def __init__(self, config, stage, selection_path=None):
        require_versions('lightgbm')
        from lightgbm import LGBMRegressor
        self.config = config
        self.model = LGBMRegressor(objective='regression', n_estimators=config.get('rounds', 100),
                                   learning_rate=.1, num_leaves=31, max_depth=-1,
                                   random_state=config.get('seed', 42), n_jobs=config.get('n_jobs', 4),
                                   device_type='cpu', verbosity=-1)
        self.metadata = {'model': 'lightgbm', 'loss_history': 'native per-round MSE',
                         'validation_role': 'held_out' if stage == 'select' else 'overlap_diagnostic_only'}

    def fit(self, X, y, Xv, yv, directory):
        from lightgbm import record_evaluation
        history = {}
        self.model.fit(X, y, eval_set=[(X, y), (Xv, yv)], eval_names=['train', 'validation'],
                       eval_metric='l2', callbacks=[record_evaluation(history)])
        frame = pd.DataFrame({'step': np.arange(1, len(history['train']['l2']) + 1),
                              'train_mse': history['train']['l2'],
                              'validation_mse': history['validation']['l2']})
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_csv(directory / 'loss.csv', index=False)
        self.model.booster_.save_model(str(directory / 'model.txt'))
        self.metadata['checkpoint_sha256'] = sha(directory / 'model.txt')

    def predict(self, X):
        return finite_vector(self.model.predict(X), len(X))

    def clear(self):
        self.model = None


def selected_epoch(path):
    path = Path(path)
    path = path / 'selected_epoch.json' if path.suffix != '.json' else path
    value = json.loads(path.read_text())
    epoch = value.get('selected_epoch')
    if value.get('selection_stage') != 'select' or not isinstance(epoch, int) or not 1 <= epoch <= 256:
        raise ValueError('invalid held-out selected epoch receipt')
    return epoch


def realmlp_params(config, stage, selection_path):
    params = {'random_state': config.get('seed', 42), 'device': 'cuda', 'n_threads': config.get('n_jobs', 4),
              'n_cv': 1, 'n_refit': 0, 'n_repeats': 1, 'n_ens': 1, 'val_fraction': 0.0,
              'n_epochs': config.get('epochs', 256), 'verbosity': 0,
              'train_metric_name': 'mse', 'val_metric_name': 'rmse'}
    if stage != 'select':
        if selection_path is None:
            raise ValueError('RealMLP refit requires held-out selected epoch receipt')
        params['stop_epoch'] = selected_epoch(selection_path)
    return params


@contextlib.contextmanager
def realmlp_history(estimator, X, y, Xv, yv, directory):
    """Owned subclass instruments real optimizer losses and epoch predictions.

    This process-local replacement is restored after fit. No installed source
    is edited. Diagnostics preserve RNG and model mode and never select weights.
    """
    import torch
    module = importlib.import_module('pytabkit.models.alg_interfaces.nn_interfaces')
    base = module.TabNNModule
    rows = []
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    class RecordedModule(base):
        def on_train_epoch_start(self):
            parent = getattr(super(), 'on_train_epoch_start', None)
            if parent:
                parent()
            self._native_loss_sum, self._native_n = 0., 0

        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            n = int(batch['y'].shape[-2])
            self._native_loss_sum += float(loss.detach().cpu()) * n
            self._native_n += n
            return loss

        def diagnostic_mse(self, features, labels):
            ds = estimator.x_converter_.transform(pd.DataFrame(features))
            ds = ds.to(next(self.model.parameters()).device)
            loader = self.get_predict_dataloader(ds)
            predictions = [self.predict_step(batch, i).detach().cpu().numpy()
                           for i, batch in enumerate(loader)]
            prediction = np.concatenate(predictions, axis=-2).reshape(-1)
            return float(np.mean((prediction - np.asarray(labels).reshape(-1)) ** 2))

        def on_train_epoch_end(self):
            parent = getattr(super(), 'on_train_epoch_end', None)
            if parent:
                parent()
            training = self.model.training
            with torch.random.fork_rng(), torch.no_grad():
                self.model.eval()
                train_mse = self.diagnostic_mse(X, y)
                validation_mse = self.diagnostic_mse(Xv, yv)
            self.model.train(training)
            if not self._native_n:
                raise RuntimeError('RealMLP produced no native training batches')
            rows.append({'step': int(self.current_epoch) + 1,
                         'native_train_loss': self._native_loss_sum / self._native_n,
                         'train_mse': train_mse, 'validation_mse': validation_mse})
            pd.DataFrame(rows).to_csv(directory / 'loss.csv', index=False)

    with patch.object(module, 'TabNNModule', RecordedModule):
        try:
            yield rows
        finally:
            fitted = getattr(getattr(estimator, 'alg_interface_', None), 'model', None)
            if isinstance(fitted, RecordedModule):
                fitted.__class__ = base  # Persist the official class, not the local instrumentation subclass.


class RealMLPModel(BatchedGPU):
    def __init__(self, config, stage, selection_path=None):
        require_versions('realmlp')
        self.config, self.torch = config, gpu(config)
        self.stage = stage
        self.selection_path = Path(selection_path) if selection_path is not None else None
        if self.selection_path is not None and self.selection_path.suffix != '.json':
            self.selection_path = self.selection_path / 'selected_epoch.json'
        from pytabkit import RealMLP_TD_Regressor
        self.model = RealMLP_TD_Regressor(**realmlp_params(config, stage, selection_path))
        self.metadata = {'model': 'realmlp', 'loss_history': 'native train loss plus epoch train/validation MSE',
                         'validation_role': 'held_out_selection' if stage == 'select' else 'overlap_diagnostic_only'}

    def fit(self, X, y, Xv, yv, directory):
        if not all(np.isfinite(a).all() for a in (X, y, Xv, yv)):
            raise ValueError('RealMLP requires finite prepared features/targets')
        directory = Path(directory)
        with realmlp_history(self.model, X, y, Xv, yv, directory) as history:
            if self.stage == 'select':
                self.model.fit(X, y, X_val=Xv, y_val=yv)
            else:
                self.model.fit(X, y)
        if not history:
            raise RuntimeError('RealMLP epoch instrumentation did not execute')
        if self.stage == 'select':
            stop = self.model.fit_params_['stop_epoch']
            epoch = int(stop['rmse'] if isinstance(stop, dict) else stop)
            if self.selection_path is None:
                raise ValueError('RealMLP select fit requires selection_path')
            write_json(self.selection_path, {'selection_stage': 'select', 'selected_epoch': epoch,
                                             'loss_sha256': sha(directory / 'loss.csv')})
        else:
            epoch = selected_epoch(self.selection_path)
            if len(history) != epoch:
                raise RuntimeError('RealMLP refit did not stop at its frozen epoch')
        self.metadata['selected_epoch'] = epoch
        self.metadata['selection_sha256'] = sha(self.selection_path)
        # Save the real fitted estimator for reproducible inference, not just predictions.
        import joblib
        joblib.dump(self.model, directory / 'model.joblib')
        self.metadata['checkpoint_sha256'] = sha(directory / 'model.joblib')

    def _predict(self, X):
        return self.model.predict(X)



def prepare_identity(model, config):
    """Freeze executable package versions and pinned checkpoint bytes before registration."""
    require_versions(model)
    packages = {name: importlib.metadata.version(name) for name in PINS[model]}
    for name in ('numpy', 'pandas', 'torch', 'scikit-learn'):
        packages[name] = importlib.metadata.version(name)
    identity = {'package_versions': packages, 'model_sha256': {}, 'weight_files': {}}
    if model in WEIGHTS:
        ref = prepare_weights(model)
        identity['weights'] = ref
        identity['weight_files'][ref['path']] = ref['sha256']
        identity['model_sha256'][ref['filename']] = ref['sha256']
    else:
        # These models are trained from scratch; identity binds the explicit recipe.
        identity['model_sha256']['no_pretrained_config'] = hashlib.sha256(
            json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                       allow_nan=False).encode()).hexdigest()
    return identity

def make_model(model, config, stage, selection_path=None):
    if stage not in ('select', 'f1', 'final'):
        raise ValueError('unknown stage')
    if model == 'mitra':
        from mitra_adapter import MitraAdapter
        return MitraAdapter(config, stage, selection_path)
    classes = {'tabiclv2': TabICLModel, 'tabldm': TabLDMModel,
               'realmlp': RealMLPModel, 'lightgbm': LightGBMModel}
    if model not in classes:
        raise ValueError(f'unsupported regression model: {model}')
    return classes[model](dict(config), stage, selection_path)
