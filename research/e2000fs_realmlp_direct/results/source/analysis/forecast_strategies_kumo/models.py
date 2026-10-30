"""Matched-context scalar Kumo/XGBoost forecasting, with bound head checkpoints.

Input objects expose ``ctx``, ``E`` (origin, horizon, feature), and ``past``.
Validation labels are used only for XGBoost's teacher-forced loss diagnostic;
query trajectories always begin empty and consume the model's own predictions.
"""
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

H = 72
STRATEGIES = ('Recursive', 'DirRec', 'Direct', 'pooled')


@dataclass(frozen=True)
class Config:
    size: str = 'large'
    n_estimators: int = 8
    context_rows: int = 5000
    seed: int = 42
    query_batch: int = 256
    xgb_rounds: int = 900
    n_jobs: int = 4
    physical_gpu: int = 1

    def __post_init__(self):
        if self.size not in ('small', 'medium', 'large'):
            raise ValueError('unknown Kumo size')
        if self.physical_gpu != 1:
            raise ValueError('this user-authorized campaign uses physical GPU1')
        if min(self.n_estimators, self.context_rows, self.query_batch,
               self.xgb_rounds, self.n_jobs) < 1:
            raise ValueError('counts must be positive')


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=6)
def _cached_weight_hash(path, size, modified_ns):
    return _file_hash(path)


def _weights_identity(config):
    from huggingface_hub import hf_hub_download
    weight = Path(hf_hub_download('nvidia/Kumo-Tabular', f'{config.size}/regressor.pt',
                                 revision='v1.0.1'))
    stat = weight.stat()
    return {'repo': 'nvidia/Kumo-Tabular', 'revision': 'v1.0.1',
            'weight_sha256': _cached_weight_hash(str(weight), stat.st_size, stat.st_mtime_ns)}


def _atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            name = f.name
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def atomic_json(path, value):
    _atomic(path, lambda f: f.write(json.dumps(value, indent=2, sort_keys=True).encode()))


def _check_inputs(inp):
    c, e, p = (np.asarray(getattr(inp, k)) for k in ('ctx', 'E', 'past'))
    if c.ndim != 2 or e.ndim != 3 or p.ndim != 2 or e.shape[1] != H:
        raise ValueError('invalid input dimensions')
    if not len(c) or len(c) != len(e) or len(c) != len(p) or not p.shape[1]:
        raise ValueError('empty or misaligned inputs')
    if any(np.isinf(a).any() for a in (c, e, p)) or not np.isfinite(p).all():
        raise ValueError('features contain infinities or past trajectory is missing')


def _input_hash(inp):
    return {k: array_hash(getattr(inp, k)) for k in ('ctx', 'E', 'past')}


def features(strategy, inp, previous, rows, horizons):
    """Construct selected zero-based origin/horizon cells without reading future Y."""
    rows, horizons = np.asarray(rows), np.asarray(horizons)
    c, e, past = inp.ctx[rows], inp.E[rows, horizons], inp.past[rows]
    if strategy == 'Recursive':
        trajectory = np.hstack([inp.past, previous])
        lag = trajectory[rows[:, None], horizons[:, None] + np.arange(past.shape[1])]
        parts = (c, lag, e, (horizons + 1)[:, None])
    elif strategy == 'DirRec':
        if not np.all(horizons == horizons[0]):
            raise ValueError('DirRec requires a single horizon per table')
        parts = (c, past, previous[rows, :int(horizons[0])], e)
    elif strategy == 'Direct':
        parts = (c, past, e)
    elif strategy == 'pooled':
        parts = (c, past, e, (horizons + 1)[:, None])
    else:
        raise ValueError('unknown strategy')
    result = np.hstack(parts).astype(np.float32)
    if result.shape[1] > 500:
        raise ValueError('feature count exceeds Kumo recipe cap; refuse implicit feature selection')
    return result


def context_cells(origins, strategy, head, limit):
    """Latest origins first selected, then returned in stable chronological order."""
    dates = pd.DatetimeIndex(origins)
    if dates.hasnans or dates.has_duplicates:
        raise ValueError('training origins must be finite and unique')
    order = np.argsort(dates.asi8, kind='stable')
    if strategy in ('Recursive', 'pooled'):
        # Bound materialization to the tail before expanding origin x horizon.
        order = order[-((limit + H - 1) // H):]
        rows = np.repeat(order, H)[-limit:]
        horizons = np.tile(np.arange(H), len(order))[-limit:]
    else:
        rows = order[-limit:]
        horizons = np.full(len(rows), head, dtype=int)
    return rows, horizons


def quantile_mean(values):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 999 or not np.isfinite(a).all():
        raise ValueError('Kumo regression must return 999 finite quantiles per row')
    return a.mean(axis=1)


def predict_batches(X, batch_size, predict, oom_type, recover):
    """Retry the same unconsumed rows on OOM; never alter model/context settings."""
    parts, start, reductions = [], 0, []
    while start < len(X):
        stop = min(start + batch_size, len(X))
        try:
            part = np.asarray(predict(X[start:stop]), dtype=np.float64)
        except oom_type:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            reductions.append(batch_size)
            recover()
            continue
        if part.shape != (stop - start,) or not np.isfinite(part).all():
            raise ValueError('invalid scalar predictions')
        parts.append(part)
        start = stop
    return np.concatenate(parts), batch_size, reductions


class XGBoostModel:
    def __init__(self, config):
        from xgboost import XGBRegressor
        self.config = config
        self.model = XGBRegressor(
            n_estimators=config.xgb_rounds, max_depth=3, learning_rate=.03,
            subsample=.8, colsample_bytree=.7, min_child_weight=20, reg_lambda=5,
            random_state=config.seed, n_jobs=config.n_jobs, objective='reg:squarederror',
            tree_method='hist', device='cpu', eval_metric='rmse')
        self.metadata = {'loss_history': 'native boosting rounds; validation is teacher-forced'}

    def fit(self, X, y, Xv, yv, directory):
        self.model.fit(X, y, eval_set=[(X, y), (Xv, yv)], verbose=False)
        ev = self.model.evals_result()
        frame = pd.DataFrame({'step': np.arange(1, self.config.xgb_rounds + 1),
                              'train_mse': np.square(ev['validation_0']['rmse']),
                              'validation_mse': np.square(ev['validation_1']['rmse'])})
        _atomic(directory / 'loss.csv', lambda f: f.write(frame.to_csv(index=False).encode()))

    def predict(self, X):
        return self.model.predict(X)

    def clear(self):
        self.model = None


class KumoModel:
    def __init__(self, config):
        import torch
        from sdm.models import KumoTabular
        # This host ships gcc-12 without the unversioned cc/gcc symlinks.
        if Path('/usr/bin/gcc-12').is_file():
            os.environ.setdefault('CC', '/usr/bin/gcc-12')
        if version('structured-data-models') != '0.2.1':
            raise RuntimeError('structured-data-models must be pinned to 0.2.1')
        if os.environ.get('CUDA_VISIBLE_DEVICES') != str(config.physical_gpu):
            raise RuntimeError('Kumo requires CUDA_VISIBLE_DEVICES=1 (physical GPU1)')
        if not torch.cuda.is_available():
            raise RuntimeError('Kumo GPU1 is unavailable')
        self.config, self.torch = config, torch
        self.batch_size = config.query_batch
        torch.set_num_threads(config.n_jobs)
        torch.cuda.reset_peak_memory_stats()
        self.model = KumoTabular(task='regression', size=config.size, device='cuda')
        self.metadata = {**_weights_identity(config),
                         'point_forecast': 'mean(q001..q999)', 'oom_batch_sizes': [],
                         'loss_history': 'not applicable: pretrained in-context inference, no epochs'}

    def _table(self, X, target=False):
        import sdm
        X = np.asarray(X, dtype=np.float32)
        cols = ['target'] if target else [f'f{i:03}' for i in range(X.shape[1])]
        frame = pd.DataFrame(X, columns=cols)
        return sdm.TableTensor.from_pandas(
            df=frame, stypes={c: sdm.Stype.numerical for c in cols}, device='cuda')

    def fit(self, X, y, Xv, yv, directory):
        torch = self.torch
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
            self.model.fit(x=self._table(X), y=self._table(y[:, None], target=True),
                           num_estimators=self.config.n_estimators,
                           estimator_batch_size=1,
                           generator=torch.Generator(device='cuda').manual_seed(self.config.seed))
        torch.cuda.synchronize()

    def predict(self, X):
        torch = self.torch

        def one(batch):
            with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
                out = self.model.predict(self._table(batch))
                return quantile_mean(out.numerical.detach().float().cpu().numpy())

        result, self.batch_size, reductions = predict_batches(
            X, self.batch_size, one, torch.cuda.OutOfMemoryError, torch.cuda.empty_cache)
        self.metadata['oom_batch_sizes'].extend(reductions)
        self.metadata['query_batch_actual'] = self.batch_size
        self.metadata['peak_gpu_memory_bytes'] = int(torch.cuda.max_memory_allocated())
        return result

    def clear(self):
        self.model.clear()
        self.model = None
        self.torch.cuda.empty_cache()


def _make_model(backend, config):
    return {'kumo': KumoModel, 'xgboost': XGBoostModel}[backend](config)


def _versions(backend):
    names = ['numpy', 'pandas', 'xgboost'] if backend == 'xgboost' else ['numpy', 'pandas', 'torch', 'structured-data-models']
    result = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = 'not installed'
    return result


def fit_predict(strategy, backend, fit_inputs, Y, valid_inputs, Yv, queries,
                origins, output_dir, stage, config=None):
    """Fit one strategy and return ``(query_predictions, metadata)``.

    Pass validation inputs in ``queries`` to obtain genuine recursive validation
    predictions. ``Yv`` is solely the native boosting loss diagnostic. Checkpoint
    scope is ``output_dir / stage / backend / strategy`` and refuses changed inputs.
    """
    config = config or Config()
    if strategy not in STRATEGIES or backend not in ('kumo', 'xgboost'):
        raise ValueError('unknown strategy or backend')
    if not queries:
        raise ValueError('queries cannot be empty')
    for inp in (fit_inputs, valid_inputs, *queries.values()):
        _check_inputs(inp)
        if any(getattr(inp, k).shape[1:] != getattr(fit_inputs, k).shape[1:]
               for k in ('ctx', 'E', 'past')):
            raise ValueError('feature layout differs across input sets')
    Y, Yv = np.asarray(Y, dtype=np.float64), np.asarray(Yv, dtype=np.float64)
    if Y.shape != (len(fit_inputs.ctx), H) or Yv.shape != (len(valid_inputs.ctx), H):
        raise ValueError('targets must be shaped (origins, 72)')
    if not np.isfinite(Y).all() or not np.isfinite(Yv).all() or len(origins) != len(Y):
        raise ValueError('targets must be finite and origins must align')
    dates = pd.DatetimeIndex(origins)
    root = Path(output_dir) / stage / backend / strategy
    root.mkdir(parents=True, exist_ok=True)
    identity = {'strategy': strategy, 'backend': backend, 'stage': stage,
                'config': asdict(config), 'versions': _versions(backend),
                'code_sha256': _file_hash(__file__), 'fit': _input_hash(fit_inputs),
                'valid': _input_hash(valid_inputs), 'Y': array_hash(Y), 'Yv': array_hash(Yv),
                'origins': array_hash(dates.asi8),
                'queries': {k: _input_hash(v) for k, v in queries.items()}}
    if backend == 'kumo':
        identity['weights'] = _weights_identity(config)
    manifest = root / 'identity.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError(f'checkpoint identity changed: {root}; use a new output directory')
    atomic_json(manifest, identity)
    preds = {k: np.full((len(v.ctx), H), np.nan) for k, v in queries.items()}
    totals = {'fit_seconds': 0., 'inference_seconds': {k: 0. for k in queries},
              'heads': [], 'identity_sha256': _file_hash(manifest)}
    model, shared_meta = None, None
    shared = strategy in ('Recursive', 'pooled')
    try:
        for head in range(H):
            directory = root / f'h{head + 1:02}'
            directory.mkdir(exist_ok=True)
            checkpoint, receipt = directory / 'predictions.npz', directory / 'receipt.json'
            if checkpoint.exists() and receipt.exists():
                meta = json.loads(receipt.read_text())
                if meta['prediction_sha256'] != _file_hash(checkpoint):
                    raise ValueError('checkpoint predictions hash mismatch')
                with np.load(checkpoint, allow_pickle=False) as z:
                    if set(z.files) != set(queries):
                        raise ValueError('checkpoint query names mismatch')
                    for key in queries:
                        a = z[key]
                        if a.shape != (len(preds[key]),) or not np.isfinite(a).all():
                            raise ValueError('invalid checkpoint shape or values')
                        preds[key][:, head] = a
                totals['heads'].append(meta)
                continue
            if model is None or not shared:
                rows, hs = context_cells(dates, strategy, head, config.context_rows)
                X = features(strategy, fit_inputs, Y, rows, hs)
                y = Y[rows, hs]
                vr = np.arange(len(Yv))
                if shared:
                    vr, vh = np.repeat(vr, H), np.tile(np.arange(H), len(vr))
                else:
                    vh = np.full(len(vr), head, dtype=int)
                Xv = features(strategy, valid_inputs, Yv, vr, vh)
                yv = Yv[vr, vh]
                keys = pd.DataFrame({'origin_utc': dates[rows].astype(str), 'horizon': hs + 1})
                context_hash = {'X': array_hash(X), 'y': array_hash(y),
                                'origin_ns': array_hash(dates.asi8[rows]), 'horizon': array_hash(hs + 1)}
                fit_dir = root / ('shared_fit' if shared else f'h{head + 1:02}')
                fit_dir.mkdir(exist_ok=True)
                _atomic(fit_dir / 'context.csv', lambda f: f.write(keys.to_csv(index=False).encode()))
                start = time.perf_counter()
                model = _make_model(backend, config)
                model.metadata['validation_role'] = (
                    'held_out_teacher_forced_diagnostic' if stage == 'select'
                    else 'final_fit_overlap_diagnostic_not_selection')
                model.fit(X, y, Xv, yv, fit_dir)
                elapsed = time.perf_counter() - start
                shared_meta = {'context_hashes': context_hash, 'n_context': len(rows),
                               'n_features': X.shape[1], 'fit_seconds': elapsed,
                               'model': dict(model.metadata)}
                atomic_json(fit_dir / 'fit.json', shared_meta)
                totals['fit_seconds'] += elapsed
                del X, Xv
            head_predictions, seconds = {}, {}
            for key, inp in queries.items():
                query_X = features(strategy, inp, preds[key], np.arange(len(inp.ctx)),
                                   np.full(len(inp.ctx), head, dtype=int))
                start = time.perf_counter()
                raw = np.asarray(model.predict(query_X), dtype=np.float64)
                seconds[key] = time.perf_counter() - start
                if raw.shape != (len(inp.ctx),) or not np.isfinite(raw).all():
                    raise ValueError('invalid model predictions')
                head_predictions[key] = np.clip(raw, 1., 3000.)
                preds[key][:, head] = head_predictions[key]
                totals['inference_seconds'][key] += seconds[key]
            _atomic(checkpoint, lambda f: np.savez(f, **head_predictions))
            meta = {'head': head + 1, 'context_hashes': shared_meta['context_hashes'],
                    'fit_seconds': shared_meta['fit_seconds'] if not shared or head == 0 else 0.,
                    'inference_seconds': seconds, 'model': dict(model.metadata),
                    'prediction_sha256': _file_hash(checkpoint)}
            atomic_json(receipt, meta)
            totals['heads'].append(meta)
            print(f'{stage} {backend} {strategy}: head {head + 1}/72', flush=True)
            if not shared:
                model.clear()
                model = None
    finally:
        if model is not None:
            model.clear()
    # Original measured times survive resume; wall time for this invocation is separate.
    totals['fit_seconds'] = sum(h['fit_seconds'] for h in totals['heads'])
    totals['inference_seconds'] = {
        key: sum(h['inference_seconds'][key] for h in totals['heads']) for key in queries}
    totals['inference_seconds_per_fold'] = {
        key: value / len(preds[key]) for key, value in totals['inference_seconds'].items()}
    atomic_json(root / 'summary.json', totals)
    return preds, totals
