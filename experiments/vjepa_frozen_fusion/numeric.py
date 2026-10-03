"""Frozen numerical recipe plus explicitly allowed, train-fitted visual features."""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from experiments.ch_breakthrough_v2 import common
from experiments.ch_breakthrough_v2.physical_quality import build_physics_frame, build_ch_quality_frame
from experiments.ch_breakthrough_v2.recurrence import RecurrenceResidualForecaster
from experiments.mse20_four_regimes import protocol, tree

END = pd.Timestamp('2026-05-31T23:00:00Z')


def load_frames(root=None, end=END):
    """Predicate-read sources before feature construction; never load official truth."""
    root = Path(root or protocol.ROOT)
    end = pd.Timestamp(end)
    if end > END:
        raise ValueError('official scoring period forbidden')
    ace_path = root / 'store/ch-v1/ace.parquet'
    ch_path = root / 'store/ch-v1/ch-hourly.parquet'
    ace = pd.read_parquet(ace_path, filters=[('timestamp_utc', '<=', end)])
    ace['timestamp_utc'] = pd.to_datetime(ace.timestamp_utc, utc=True)
    ace = ace.sort_values('timestamp_utc').drop_duplicates('timestamp_utc', keep='last').set_index('timestamp_utc')
    values = pd.to_numeric(ace.filled_speed_kms, errors='coerce')
    ace['filled_speed_kms'] = values.where(np.isfinite(values) & values.gt(0))
    ace = ace.loc[ace.filled_speed_kms.first_valid_index():]
    ace = ace.reindex(pd.date_range(ace.index.min(), ace.index.max(), freq='h'))
    ace['was_missing'] = (ace.was_missing.astype('boolean').fillna(True) | ace.filled_speed_kms.isna()).astype(int)
    ace['filled_speed_kms'] = ace.filled_speed_kms.ffill()
    ace = ace.rename_axis('timestamp_utc').reset_index()
    # CH carries source geometry rather than ACE scoring truth; slice before builders.
    ch = common.load_ch_hourly(ch_path).loc[:end]
    raw_paths = sorted(glob.glob(str(root / 'store/ch-breakthrough-v2/upload/raw-ace-snapshot/year=*/part.parquet')))
    parts = [pd.read_parquet(p, filters=[('timestamp_utc', '<=', end)]) for p in raw_paths]
    raw = pd.concat(parts)
    raw = raw.set_index(pd.DatetimeIndex(pd.to_datetime(raw.timestamp_utc, utc=True))).drop(columns='timestamp_utc')
    raw = raw[~raw.index.duplicated(keep='last')].sort_index()
    speed = common.as_speed_series(ace)
    base = common.build_features(ace, ch).replace([np.inf, -np.inf], np.nan)
    index = base.index[base.index.isin(raw.index.floor('h'))]
    return dict(base=base.reindex(index), physics=build_physics_frame(raw, index),
                quality=build_ch_quality_frame(ch, index), speed=speed,
                observed=common.observed_mask(speed, speed.index),
                identities={str(p): protocol.digest(p) for p in [ace_path, ch_path, *raw_paths]})


def train_origins(frames, cutoff, cap=6000):
    features = tree._features(frames)
    cutoff = common.utc(cutoff)
    pool = features.index[features.index + pd.Timedelta(hours=72) <= cutoff]
    depth = frames['speed'].loc[:cutoff].rolling(common.MEAN_WINDOW_HOURS, min_periods=common.MEAN_WINDOW_HOURS).mean()
    pool = pool[depth.reindex(pool).notna().to_numpy()]
    if len(pool) > cap:
        pool = pool[np.linspace(0, len(pool) - 1, cap, dtype=int)]
    if not len(pool):
        raise ValueError('empty training origins')
    return pool


def plan_origins(frames, cap=6000):
    """Frozen split grids: development 6h, selection/audit hourly; 6000 train pool."""
    return {name: {'train': train_origins(frames, spec['cutoff'], cap),
                   'eval': protocol.split_origins(frames['base'].index, name)}
            for name, spec in protocol.SPLITS.items()}


def read_embeddings(path, arm):
    with np.load(path, allow_pickle=False) as z:
        origins = pd.DatetimeIndex(pd.to_datetime(z['origins'], utc=True))
        values = np.asarray(z[arm], dtype=np.float32)
        ends = z[f'{arm}_source_end_ns']
        quality = np.asarray(z[f'{arm}_quality'], dtype=np.float32) if f'{arm}_quality' in z else np.empty((len(origins), 0))
    if not origins.is_unique or not origins.is_monotonic_increasing:
        raise ValueError('embedding origins must be sorted and unique')
    if values.ndim != 2 or quality.ndim != 2 or len(values) != len(origins) or len(quality) != len(origins):
        raise ValueError('embedding shape mismatch')
    if np.shape(ends) != (len(origins),) or np.any(ends > origins.asi8):
        raise ValueError('future image source')
    return origins, values, quality


def visual_features(path, arm, pool, index, output_dir, fit=True):
    origins, values, quality = read_embeddings(path, arm)
    locations = origins.get_indexer(pool)
    if fit and np.any(locations < 0):
        raise ValueError('missing planned train embeddings')
    transform_path = Path(output_dir) / 'visual_transform.npz'
    if fit:
        train = values[locations].astype(np.float64)
        valid = np.isfinite(train)
        mean = np.divide(np.where(valid, train, 0).sum(0), valid.sum(0),
                         out=np.zeros(train.shape[1]), where=valid.sum(0) > 0)
        clean = np.where(valid, train, mean)
        if min(clean.shape) < 16:
            raise ValueError('PCA16 needs at least 16 train rows and dimensions')
        pca = PCA(n_components=16, svd_solver='randomized', random_state=42).fit(clean)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        np.savez(transform_path, impute=mean, center=pca.mean_, components=pca.components_,
                 train_origins=pool.asi8, variance=pca.explained_variance_ratio_)
    with np.load(transform_path, allow_pickle=False) as transform:
        mean, center, components = transform['impute'], transform['center'], transform['components']
    selected = origins.isin(pd.DatetimeIndex(index))
    origins, values, quality = origins[selected], values[selected], quality[selected]
    clean = np.where(np.isfinite(values), values, mean)
    projected = ((clean - center) @ components.T).astype(np.float32)
    missing = (~np.isfinite(values)).mean(axis=1, keepdims=True)
    data = np.concatenate([projected, missing, quality], axis=1)
    columns = [f'vjepa_pc_{i:02d}' for i in range(16)] + ['vjepa_missing_fraction'] + [f'vjepa_quality_{i}' for i in range(quality.shape[1])]
    return pd.DataFrame(data, index=origins, columns=columns).reindex(index)


def fit_predict(frames, cutoff, origins, config, device, output_dir, arm='baseline', embeddings=None):
    if arm == 'baseline':
        return tree.fit_predict(frames, cutoff, origins, config, device, output_dir)
    if arm not in ('image', 'video'):
        raise ValueError('unknown visual arm')
    output_dir = Path(output_dir)
    if (output_dir / 'metadata.json').exists():
        raise FileExistsError(output_dir)
    features = tree._features(frames)
    pool = train_origins(frames, cutoff, config['max_train_origins'])
    origins = tree._origins(origins)
    eorigins, _, _ = read_embeddings(embeddings, arm)
    if len(origins.difference(eorigins)):
        raise ValueError('missing planned eval embeddings')
    visual = visual_features(embeddings, arm, pool, features.index, output_dir)
    features = pd.concat([features, visual], axis=1)
    fore = RecurrenceResidualForecaster(params=dict(config['params']), device=device, seed=config['seed'], use_target=True)
    start = time.perf_counter()
    fore.fit(features.loc[:cutoff], frames['speed'].loc[:cutoff], cutoff, max_origins=config['max_train_origins'])
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    preds = fore.predict(features, frames['speed'], origins)
    prediction_seconds = time.perf_counter() - start
    tree._validate_predictions(preds, origins)
    model_path = output_dir / 'model.ubj'
    fore.model.get_booster().save_model(model_path)
    metadata = dict(family='combined_recurrence_xgboost_visual', arm=arm, config=config,
                    cutoff=str(cutoff), train_label_max=str(pool.max() + pd.Timedelta(hours=72)),
                    train_origin_sha256=hashlib.sha256(pool.asi8.tobytes()).hexdigest(),
                    n_train_origins=len(pool), base_columns=fore.base_columns, columns=fore.columns,
                    actual_device=fore.actual_device, fit_seconds=fit_seconds, prediction_seconds=prediction_seconds,
                    use_target=True, model_sha256=protocol.digest(model_path),
                    transform_sha256=protocol.digest(output_dir / 'visual_transform.npz'),
                    embedding_sha256=protocol.digest(embeddings), input_identities=frames.get('identities', {}))
    gains = fore.model.get_booster().get_score(importance_type='gain')
    metadata['feature_gain'] = gains
    metadata['visual_gain_sum'] = {name: float(sum(v for k, v in gains.items() if k.startswith(prefix))) for name, prefix in [('pca', 'vjepa_pc_'), ('quality', 'vjepa_quality_'), ('missing', 'vjepa_missing_')]}
    with np.load(output_dir / 'visual_transform.npz') as transform:
        metadata['pca_explained_variance_ratio'] = transform['variance'].tolist()
    protocol.write_json(output_dir / 'metadata.json', metadata)
    return preds, metadata


def predict_saved(frames, origins, output_dir, device, embeddings=None):
    from xgboost import Booster
    output_dir = Path(output_dir)
    meta = json.loads((output_dir / 'metadata.json').read_text())
    if meta.get('arm', 'baseline') == 'baseline':
        return tree.predict_saved(frames, origins, output_dir, device)
    if protocol.digest(output_dir / 'model.ubj') != meta['model_sha256'] or protocol.digest(output_dir / 'visual_transform.npz') != meta['transform_sha256']:
        raise ValueError('saved artifact hash mismatch')
    features = tree._features(frames)
    origins = tree._origins(origins)
    embedding_origins, _, _ = read_embeddings(embeddings, meta['arm'])
    if len(origins.difference(embedding_origins)):
        raise ValueError('missing planned eval embeddings')
    pool = train_origins(frames, meta['cutoff'], meta['config']['max_train_origins'])
    visual = visual_features(embeddings, meta['arm'], pool, tree._origins(origins), output_dir, fit=False)
    features = pd.concat([features, visual], axis=1)
    if list(features.columns) != meta['base_columns']:
        raise ValueError('saved feature schema mismatch')
    fore = RecurrenceResidualForecaster(params=meta['config']['params'], device=device, seed=meta['config']['seed'], use_target=True)
    fore.base_columns, fore.columns = meta['base_columns'], meta['columns']
    fore.cutoff = common.utc(meta['cutoff'])
    booster = Booster(model_file=output_dir / 'model.ubj')
    booster.set_param({'device': device, 'nthread': 4})
    origins, design, anchor = fore._build_predict_matrix(features, tree._origins(origins), frames['speed'])
    preds = pd.DataFrame({'origin_last_input_utc': np.repeat(origins, 72), 'horizon_hours': np.tile(np.arange(1, 73), len(origins)),
                          'pred_kms': np.clip(anchor + booster.inplace_predict(design[fore.columns]), 1, 3000)})
    tree._validate_predictions(preds, origins)
    return preds
