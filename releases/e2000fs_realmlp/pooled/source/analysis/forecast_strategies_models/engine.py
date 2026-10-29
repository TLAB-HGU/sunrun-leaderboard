"""Independent scalar strategy engine sharing the frozen Kumo feature semantics."""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('frozen_kumo_features', HERE.parent / 'forecast_strategies_kumo/models.py')
K = importlib.util.module_from_spec(SPEC)
# Dataclasses require their defining module to be discoverable.
sys.modules[SPEC.name] = K
SPEC.loader.exec_module(K)
STRATEGIES = K.STRATEGIES


def fit_predict(strategy, model_name, fit_inputs, Y, valid_inputs, Yv, queries,
                origins, output_dir, stage, config, manifest_sha256):
    """Persist every scalar prediction and reject changed experimental identities."""
    from adapters import make_model
    if strategy not in STRATEGIES or stage not in ('select', 'f1', 'final'):
        raise ValueError('unknown strategy or stage')
    if not queries:
        raise ValueError('queries cannot be empty')
    for inp in (fit_inputs, valid_inputs, *queries.values()):
        K._check_inputs(inp)
        if any(getattr(inp, k).shape[1:] != getattr(fit_inputs, k).shape[1:]
               for k in ('ctx', 'E', 'past')):
            raise ValueError('feature layout differs')
    Y, Yv = np.asarray(Y, dtype=float), np.asarray(Yv, dtype=float)
    if Y.shape != (len(fit_inputs.ctx), 72) or Yv.shape != (len(valid_inputs.ctx), 72):
        raise ValueError('target shape differs')
    if not np.isfinite(Y).all() or not np.isfinite(Yv).all():
        raise ValueError('nonfinite target')
    dates = pd.DatetimeIndex(origins)
    if len(dates) != len(Y):
        raise ValueError('origin count differs')
    root = Path(output_dir) / strategy / stage
    root.mkdir(parents=True, exist_ok=True)
    identity = {'model': model_name, 'strategy': strategy, 'stage': stage,
                'manifest_sha256': manifest_sha256, 'config': config,
                'engine_sha256': K._file_hash(__file__), 'feature_engine_sha256': K._file_hash(K.__file__),
                'fit': K._input_hash(fit_inputs), 'valid': K._input_hash(valid_inputs),
                'Y': K.array_hash(Y), 'Yv': K.array_hash(Yv), 'origins': K.array_hash(dates.asi8),
                'queries': {key: K._input_hash(inp) for key, inp in queries.items()}}
    if model_name == 'realmlp' and stage != 'select':
        selected = Path(output_dir) / strategy / 'select'
        directories = [selected / 'shared_fit'] if strategy in ('Recursive', 'pooled') else [
            selected / f'h{head:02}' for head in range(1, 73)]
        identity['selected_epoch_sha256'] = {
            str(directory / 'selected_epoch.json'): K._file_hash(directory / 'selected_epoch.json')
            for directory in directories}
    path = root / 'identity.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError('checkpoint identity changed; use a new output directory')
    K.atomic_json(path, identity)
    predictions = {key: np.full((len(inp.ctx), 72), np.nan) for key, inp in queries.items()}
    shared = strategy in ('Recursive', 'pooled')
    model = None
    heads = []
    additional_refit_seconds = 0.
    try:
        for head in range(72):
            directory = root / f'h{head + 1:02}'
            directory.mkdir(exist_ok=True)
            checkpoint, receipt = directory / 'predictions.npz', directory / 'receipt.json'
            if checkpoint.exists() and receipt.exists():
                info = json.loads(receipt.read_text())
                if K._file_hash(checkpoint) != info['prediction_sha256']:
                    raise ValueError('checkpoint corrupted')
                with np.load(checkpoint, allow_pickle=False) as z:
                    if set(z.files) != set(queries):
                        raise ValueError('checkpoint query keys differ')
                    for key in queries:
                        a = z[key]
                        if a.shape != (len(predictions[key]),) or not np.isfinite(a).all():
                            raise ValueError('invalid checkpoint predictions')
                        predictions[key][:, head] = a
                heads.append(info)
                continue
            if model is None:
                rows, hs = K.context_cells(dates, strategy, head, config['context_rows'])
                X, y = K.features(strategy, fit_inputs, Y, rows, hs), Y[rows, hs]
                vr = np.arange(len(Yv))
                if shared:
                    vr, vh = np.repeat(vr, 72), np.tile(np.arange(72), len(vr))
                else:
                    vh = np.full(len(vr), head, dtype=int)
                Xv = K.features(strategy, valid_inputs, Yv, vr, vh)
                fit_dir = root / ('shared_fit' if shared else f'h{head + 1:02}')
                fit_dir.mkdir(exist_ok=True)
                select_dir = Path(output_dir) / strategy / 'select' / ('shared_fit' if shared else f'h{head + 1:02}')
                hashes = {'X': K.array_hash(X), 'y': K.array_hash(y),
                          'origin_ns': K.array_hash(dates.asi8[rows]), 'horizon': K.array_hash(hs + 1)}
                keys = pd.DataFrame({'origin_utc': dates[rows].astype(str), 'horizon': hs + 1})
                K._atomic(fit_dir / 'context.csv', lambda stream, keys=keys: stream.write(keys.to_csv(index=False).encode()))
                start = time.perf_counter()
                model = make_model(model_name, config, stage, selection_path=select_dir)
                model.fit(X, y, Xv, Yv[vr, vh], fit_dir)
                fit_seconds = time.perf_counter() - start
                fit_info = {'context_hashes': hashes, 'n_context': len(y), 'n_features': X.shape[1],
                            'fit_seconds': fit_seconds, 'model': dict(model.metadata)}
                K.atomic_json(fit_dir / 'fit.json', fit_info)
                if shared and head:
                    additional_refit_seconds += fit_seconds
                del X, Xv
            pieces, seconds = {}, {}
            for key, inp in queries.items():
                Xq = K.features(strategy, inp, predictions[key], np.arange(len(inp.ctx)),
                                np.full(len(inp.ctx), head, dtype=int))
                start = time.perf_counter()
                pred = np.asarray(model.predict(Xq), dtype=float).reshape(-1)
                seconds[key] = time.perf_counter() - start
                if pred.shape != (len(inp.ctx),) or not np.isfinite(pred).all():
                    raise ValueError('invalid model scalar output')
                pieces[key] = np.clip(pred, 1., 3000.)
                predictions[key][:, head] = pieces[key]
            K._atomic(checkpoint, lambda stream, pieces=pieces: np.savez(stream, **pieces))
            info = {'head': head + 1, 'context_hashes': fit_info['context_hashes'],
                    'fit_seconds': fit_seconds if not shared or head == 0 else 0.,
                    'inference_seconds': seconds, 'model': dict(model.metadata),
                    'prediction_sha256': K._file_hash(checkpoint)}
            K.atomic_json(receipt, info)
            heads.append(info)
            print(f'{model_name} {strategy} {stage}: head {head + 1}/72', flush=True)
            if not shared:
                model.clear()
                model = None
    finally:
        if model is not None:
            model.clear()
    summary = {'heads': heads, 'fit_seconds': sum(h['fit_seconds'] for h in heads),
               'additional_refit_seconds_this_invocation': additional_refit_seconds,
               'inference_seconds_per_fold': {
                   key: sum(h['inference_seconds'][key] for h in heads) / len(predictions[key]) for key in queries}}
    K.atomic_json(root / 'summary.json', summary)
    return predictions, summary
