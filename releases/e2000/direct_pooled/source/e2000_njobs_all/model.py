"""Independent horizon XGBoost fits with native losses and bound checkpoints.

Only trusted local checkpoints may be loaded. Targets are absolute km/s.
The pooled control receives the identical wide feature vector plus horizon.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from experiments.cumulative_20261024.contracts import (
    ContractModel, TargetSpec, checkpoint_identity, load_checkpoint, save_checkpoint,
)


@dataclass(frozen=True)
class ModelConfig:
    horizons: int = 72
    n_estimators: int = 900
    max_depth: int = 3
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.7
    min_child_weight: float = 20
    reg_lambda: float = 5
    n_jobs: int = -1

    def __post_init__(self):
        if (not 1 <= self.horizons <= 72 or self.n_estimators < 1
                or type(self.n_jobs) is not int or (self.n_jobs != -1 and self.n_jobs < 1)):
            raise ValueError('invalid horizons, rounds or CPU thread count')

    def params(self):
        return {k: v for k, v in asdict(self).items() if k != 'horizons'} | {
            'random_state': 42, 'objective': 'reg:squarederror',
            'tree_method': 'hist', 'device': 'cpu', 'eval_metric': 'rmse',
        }


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _array_hash(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def _atomic_text(path, text):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
            temporary = f.name
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _xy(X, Y, mask, horizons):
    X, Y = np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or not len(X) or not X.shape[1] or np.isinf(X).any():
        raise ValueError('features must be a nonempty matrix without infinities')
    if Y.shape != (len(X), horizons):
        raise ValueError('target shape must be (origins, horizons)')
    if mask is None:
        mask = np.isfinite(Y)
    else:
        mask = np.asarray(mask)
        if mask.shape != Y.shape or mask.dtype != np.bool_:
            raise ValueError('observed_mask must be a boolean target-shaped matrix')
        if not np.isfinite(Y[mask]).all():
            raise ValueError('observed targets must be finite')
    if not mask.any(axis=0).all():
        raise ValueError('every horizon requires observed targets')
    return X, Y, mask


def _prepare(X, Y, mask, validation, context, config, validation_role):
    X, Y, mask = _xy(X, Y, mask, config.horizons)
    names = tuple(context['feature_names'])
    if len(names) != X.shape[1] or len(set(names)) != len(names):
        raise ValueError('feature_names must match columns uniquely and in order')
    if context.get('stage') not in ('internal', 'official'):
        raise ValueError('stage must be internal or official')
    if validation_role not in ('held_out', 'overlapping_diagnostic', 'none'):
        raise ValueError('invalid validation_role')
    if validation is None:
        if context['stage'] != 'official' or validation_role != 'none':
            raise ValueError('validation may be absent only for official stage with role none')
        val = None
    else:
        if validation_role == 'none' or (context['stage'] == 'internal' and validation_role != 'held_out'):
            raise ValueError('invalid validation role for this stage')
        val = _xy(*validation, config.horizons)
        if val[0].shape[1] != X.shape[1]:
            raise ValueError('validation feature shape mismatch')
    # Bind actual arrays as well as the caller-provided source/grid/code identity.
    hashes = {'X': _array_hash(X), 'Y': _array_hash(Y), 'mask': _array_hash(mask)}
    if val is not None:
        hashes.update({f'validation_{k}': _array_hash(v) for k, v in zip(('X', 'Y', 'mask'), val)})
    return X, Y, mask, val, names, hashes


def _fit_one(X, y, validation, *, head, names, output_dir, context, config,
             array_hashes, validation_role):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f'model_s42_{head}.checkpoint'
    losses = output_dir / f'loss_history_s42_{head}.csv'
    receipt_path = output_dir / f'receipt_s42_{head}.json'
    params = config.params()
    identity = checkpoint_identity(
        code_hashes={'declared': context['code_hashes'], 'model.py': _hash(__file__)},
        data_hashes={'declared': context['data_hashes'], 'actual': array_hashes},
        target=TargetSpec(), seed=42,
        hyperparameters={'params': params, 'head': head, 'horizons': config.horizons,
                         'stage': context['stage'], 'validation_role': validation_role},
        feature_names=names, cutoff=context['cutoff'], grid=context['grid'],
        source_availability=context['source_availability'],
    )
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('identity') != identity or receipt.get('complete') is not True:
            raise ValueError(f'{head}: checkpoint identity mismatch or incomplete receipt')
        for path in (checkpoint, losses):
            if not path.is_file() or _hash(path) != receipt['sha256'].get(path.name):
                raise ValueError(f'{head}: checkpoint/loss integrity mismatch')
        model = load_checkpoint(checkpoint, identity)
        if model.estimator.get_params()['random_state'] != 42:
            raise ValueError('checkpoint seed mismatch')
        return model, receipt
    if checkpoint.exists() or losses.exists():
        quarantine_root = output_dir / 'quarantine'
        quarantine_root.mkdir(exist_ok=True)
        quarantine = Path(tempfile.mkdtemp(prefix=f's42_{head}_', dir=quarantine_root))
        # Receipt publication is the commit point. Preserve uncommitted files
        # from an interrupted fit, then refit only this head from scratch.
        for path in (checkpoint, losses):
            if path.exists():
                os.replace(path, quarantine / path.name)
        print(f'{head}: preserved incomplete artifacts in {quarantine}', flush=True)
    train_frame = pd.DataFrame(X, columns=names)
    target = TargetSpec()
    estimator = XGBRegressor(**params)
    eval_set = [(train_frame, target.encode(y))]
    if validation is not None:
        xv, yv = validation
        eval_set.append((pd.DataFrame(xv, columns=names), target.encode(yv)))
    # Use the shared absolute target codec and prediction contract; the native
    # estimator fit needs eval_set, which ContractModel.fit does not accept.
    model = ContractModel(estimator, target, names)
    estimator.fit(train_frame, target.encode(y), eval_set=eval_set, verbose=False)
    model._receipt = (model.target, model.feature_names)
    result = estimator.evals_result()
    train_loss = np.square(np.asarray(result['validation_0']['rmse'], dtype=float))
    valid_loss = (np.square(np.asarray(result['validation_1']['rmse'], dtype=float))
                  if validation is not None else np.full(len(train_loss), np.nan))
    if len(train_loss) != config.n_estimators or not np.isfinite(train_loss).all():
        raise ValueError('native loss history is incomplete or nonfinite')
    if validation is not None and (len(valid_loss) != len(train_loss) or not np.isfinite(valid_loss).all()):
        raise ValueError('native validation loss history is incomplete or nonfinite')
    loss_frame = pd.DataFrame({'step': np.arange(1, len(train_loss) + 1),
                              'train_mse': train_loss, 'validation_mse': valid_loss})
    save_checkpoint(checkpoint, model, identity)
    _atomic_text(losses, loss_frame.to_csv(index=False))
    receipt = {'identity': identity, 'complete': True, 'head': head, 'seed': 42,
               'target': 'absolute', 'stage': context['stage'], 'validation_role': validation_role,
               'train_rows': len(y), 'validation_rows': 0 if validation is None else len(validation[1]),
               'rounds': len(train_loss), 'sha256': {p.name: _hash(p) for p in (checkpoint, losses)}}
    _atomic_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True))
    return model, receipt


class LoggedMultiOutputRegressor(MultiOutputRegressor):
    """Sequential sklearn-style independent output fits, each with its own eval_set."""

    def __init__(self, config=ModelConfig()):
        self.config = config
        super().__init__(XGBRegressor(**config.params()), n_jobs=1)

    def fit(self, X, Y, *, observed_mask=None, validation=None, output_dir, context,
            validation_role='held_out'):
        X, Y, mask, val, names, hashes = _prepare(
            X, Y, observed_mask, validation, context, self.config, validation_role)
        self.models_, self.receipts_, self.estimators_ = [], [], []
        self.n_features_in_, self.feature_names_in_ = X.shape[1], np.asarray(names)
        for h in range(self.config.horizons):
            valid = None if val is None else (val[0][val[2][:, h]], val[1][val[2][:, h], h])
            model, receipt = _fit_one(
                X[mask[:, h]], Y[mask[:, h], h], valid,
                head=f'h{h + 1:02d}', names=names, output_dir=output_dir,
                context=context, config=self.config, array_hashes=hashes, validation_role=validation_role)
            self.models_.append(model)
            self.estimators_.append(model.estimator)
            self.receipts_.append(receipt)
            print(f'{context["stage"]} h{h + 1:02d}/{self.config.horizons} complete', flush=True)
        return self

    def predict(self, X):
        if len(getattr(self, 'models_', [])) != self.config.horizons:
            raise ValueError('all output models must be fitted before prediction')
        frame = pd.DataFrame(np.asarray(X, dtype=np.float32), columns=self.feature_names_in_)
        return np.column_stack([m.predict(frame) for m in self.models_])


class PooledRegressor:
    """One direct control model over wide origin features and horizon number."""

    def __init__(self, config=ModelConfig()):
        self.config = config

    @staticmethod
    def _expand(X, Y, mask):
        rows, horizons = np.nonzero(mask)
        # Keep expanded features float32: int64 horizons would otherwise upcast
        # a potentially large origin-by-horizon matrix to float64.
        flat = np.empty((len(rows), X.shape[1] + 1), dtype=np.float32)
        flat[:, :-1] = X[rows]
        flat[:, -1] = horizons + 1
        return flat, Y[rows, horizons]

    def fit(self, X, Y, *, observed_mask=None, validation=None, output_dir, context,
            validation_role='held_out'):
        X, Y, mask, val, names, hashes = _prepare(
            X, Y, observed_mask, validation, context, self.config, validation_role)
        if 'forecast_horizon' in names:
            raise ValueError('reserved pooled feature forecast_horizon')
        self.feature_names_in_, self.n_features_in_ = np.asarray(names), X.shape[1]
        flat_X, flat_y = self._expand(X, Y, mask)
        flat_val = None if val is None else self._expand(*val)
        self.model_, receipt = _fit_one(
            flat_X, flat_y, flat_val, head='pooled', names=names + ('forecast_horizon',),
            output_dir=output_dir, context=context, config=self.config,
            array_hashes=hashes, validation_role=validation_role)
        self.receipts_ = [receipt]
        return self

    def predict(self, X):
        if not hasattr(self, 'model_'):
            raise ValueError('pooled model must be fitted before prediction')
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError('feature shape mismatch')
        # Avoid expanding the entire forecast grid in memory at inference.
        names = tuple(self.feature_names_in_) + ('forecast_horizon',)
        return np.column_stack([
            self.model_.predict(pd.DataFrame(np.column_stack((X, np.full(len(X), h))), columns=names))
            for h in range(1, self.config.horizons + 1)
        ])
