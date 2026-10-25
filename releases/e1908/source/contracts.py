"""Fail-closed fit, cache and prediction contracts for direct forecast experiments.

Checkpoint pickle payloads are for trusted local artifacts only, never external files.
Feature NaNs are allowed; targets, current speed and predictions must be finite.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile

import numpy as np
import pandas as pd


def _vector(values, name, length=None):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite one-dimensional vector')
    if length is not None and len(result) != length:
        raise ValueError(f'{name} length mismatch')
    return result


@dataclass(frozen=True)
class TargetSpec:
    mode: str = 'absolute'
    factor: float = 1.0

    def __post_init__(self):
        if self.mode not in ('absolute', 'delta'):
            raise ValueError('unknown target mode')
        if not np.isfinite(self.factor) or not 0 < self.factor <= 1:
            raise ValueError('target factor must be in (0, 1]')
        if self.mode == 'absolute' and self.factor != 1:
            raise ValueError('absolute target does not take a fractional factor')

    def _offset(self, speed, length):
        if self.mode == 'absolute':
            return 0.
        if speed is None:
            raise ValueError('delta target requires current_speed')
        return self.factor * _vector(speed, 'current_speed', length)

    def encode(self, y, current_speed=None):
        y = _vector(y, 'target')
        return _vector(y - self._offset(current_speed, len(y)), 'encoded target')

    def decode(self, prediction, current_speed=None):
        prediction = _vector(prediction, 'prediction')
        return _vector(prediction + self._offset(current_speed, len(prediction)), 'decoded prediction')


class ContractModel:
    """One target contract governs both estimator training and prediction decoding."""
    def __init__(self, estimator, target, feature_names):
        if not isinstance(target, TargetSpec):
            raise ValueError('target must be TargetSpec')
        self.estimator = estimator
        self.target = target
        self.feature_names = tuple(feature_names)
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError('feature_names must be nonempty and unique')
        self._receipt = None

    def _features(self, X):
        if not isinstance(X, pd.DataFrame) or tuple(X.columns) != self.feature_names:
            raise ValueError('feature names/order mismatch; require a named DataFrame')
        if len(X) == 0 or np.isinf(X.to_numpy(dtype=float)).any():
            raise ValueError('empty or infinite features')
        return X

    def fit(self, X, y, current_speed=None):
        X = self._features(X)
        encoded = self.target.encode(_vector(y, 'target', len(X)), current_speed)
        self._receipt = None
        self.estimator.fit(X, encoded)
        self._receipt = (self.target, self.feature_names)
        return self

    def predict(self, X, current_speed=None, target=None):
        if self._receipt != (self.target, self.feature_names):
            raise ValueError('missing or changed fitted target/feature receipt')
        if target is not None and target != self.target:
            raise ValueError('target differs from fitted target')
        X = self._features(X)
        raw = _vector(self.estimator.predict(X), 'prediction', len(X))
        return self.target.decode(raw, current_speed)


def checkpoint_identity(*, code_hashes, data_hashes, target, seed, hyperparameters,
                        feature_names, cutoff, grid, source_availability):
    """Hash all caller-supplied recipe/input identities; feature order is significant."""
    if not isinstance(target, TargetSpec):
        raise ValueError('target must be TargetSpec')
    for name, value in [('code_hashes', code_hashes), ('data_hashes', data_hashes),
                        ('grid', grid), ('source_availability', source_availability)]:
        if not value:
            raise ValueError(f'missing {name}')
    names = list(feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError('invalid feature_names')
    timestamp = pd.Timestamp(cutoff)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError('cutoff must be timezone-aware')
    body = dict(version=1, code_hashes=code_hashes, data_hashes=data_hashes,
                target=asdict(target), seed=seed, hyperparameters=hyperparameters,
                feature_names=names, cutoff=timestamp.tz_convert('UTC').isoformat(),
                grid=grid, source_availability=source_availability)
    encoded = json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _identity(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('invalid checkpoint identity')


def save_checkpoint(path, model, identity):
    """Atomically publish a model and receipt in one checksummed file."""
    _identity(identity)
    if not isinstance(model, ContractModel) or model._receipt != (model.target, model.feature_names):
        raise ValueError('checkpoint requires a fitted ContractModel')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    header = json.dumps(dict(version=1, identity=identity, size=len(payload),
                             sha256=hashlib.sha256(payload).hexdigest()), sort_keys=True).encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', delete=False) as stream:
            temporary = stream.name
            stream.write(header + b'\n' + payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, expected_identity):
    _identity(expected_identity)
    try:
        with Path(path).open('rb') as stream:
            header = json.loads(stream.readline(4096))
            payload = stream.read()
        if (header.get('version') != 1 or header.get('identity') != expected_identity
                or header.get('size') != len(payload)
                or header.get('sha256') != hashlib.sha256(payload).hexdigest()):
            raise ValueError('checkpoint identity or integrity mismatch')
        model = pickle.loads(payload)
        if not isinstance(model, ContractModel) or model._receipt != (model.target, model.feature_names):
            raise ValueError('checkpoint has no valid fitted receipt')
        return model
    except (OSError, ValueError, TypeError, AttributeError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(f'invalid checkpoint: {exc}') from exc


def _utc_origins(values):
    dates = pd.DatetimeIndex(values)
    if dates.tz is None or str(dates.tz) != 'UTC' or dates.hasnans:
        raise ValueError('origins must be explicit finite UTC timestamps')
    return dates


def validate_prediction_chunk(frame):
    required = {'origin', 'horizon', 'prediction'}
    if not required.issubset(frame.columns) or frame.columns.duplicated().any() or frame.empty:
        raise ValueError('prediction columns missing/duplicated or empty chunk')
    _utc_origins(frame.origin)
    horizons = _vector(frame.horizon, 'horizon')
    if not np.isin(horizons, np.arange(1, 73)).all():
        raise ValueError('horizons must be integers 1..72')
    if frame.duplicated(['origin', 'horizon']).any():
        raise ValueError('duplicate prediction keys')
    _vector(frame.prediction, 'prediction', len(frame))


def validate_predictions(frame, expected_origins, horizons=range(1, 73)):
    validate_prediction_chunk(frame)
    horizons = tuple(horizons)
    if horizons != tuple(range(1, 73)):
        raise ValueError('full output requires horizons exactly 1..72')
    origins = _utc_origins(expected_origins)
    if origins.empty or origins.has_duplicates:
        raise ValueError('expected origins must be nonempty and unique')
    expected = pd.MultiIndex.from_product([origins, horizons], names=['origin', 'horizon'])
    actual = pd.MultiIndex.from_frame(frame[['origin', 'horizon']])
    if len(actual) != len(expected) or len(actual.difference(expected)) or len(expected.difference(actual)):
        raise ValueError('prediction grid does not match expected full grid')


def assemble_chunks(chunks, expected_origins):
    chunks = list(chunks)
    if not chunks:
        raise ValueError('no chunks')
    for chunk in chunks:
        validate_prediction_chunk(chunk)
    result = pd.concat(chunks, ignore_index=True)
    validate_predictions(result, expected_origins)
    return result.sort_values(['origin', 'horizon']).reset_index(drop=True)


def validate_feature_change(parent, candidate, feature_names, keys=('origin', 'horizon'),
                            mask_columns=('observed',)):
    """Reject confounded rows/masks and nominal ablations with no changed inputs.

    Include label values in mask_columns when available to prove paired labels.
    NaN is permitted in features and compared as a value, never converted to zero.
    """
    columns = list(keys) + list(mask_columns) + list(feature_names)
    if not feature_names or not keys or not mask_columns:
        raise ValueError('features, keys and masks are required')
    for frame in (parent, candidate):
        if frame.empty or frame.columns.duplicated().any() or not set(columns).issubset(frame.columns):
            raise ValueError('missing/duplicate columns or empty paired data')
        if frame[list(keys)].isna().any().any() or frame.duplicated(list(keys)).any():
            raise ValueError('invalid paired keys')
        if np.isinf(frame[list(feature_names)].to_numpy(dtype=float)).any():
            raise ValueError('infinite features')
    alignment = list(keys) + list(mask_columns)
    if not parent[alignment].reset_index(drop=True).equals(candidate[alignment].reset_index(drop=True)):
        raise ValueError('paired key/order/mask/label mismatch')
    a = parent[list(feature_names)].to_numpy(dtype=float)
    b = candidate[list(feature_names)].to_numpy(dtype=float)
    if np.array_equal(a, b, equal_nan=True):
        raise ValueError('no changed feature signal')
