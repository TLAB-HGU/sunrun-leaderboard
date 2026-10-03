"""Causal, matched-support and saved-model checks without official truth."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.ch_breakthrough_v2.common import build_features
from experiments.ch_breakthrough_v2.physical_quality import build_ch_quality_frame, build_physics_frame
from experiments.mse20_four_regimes import tree
from experiments.vjepa_frozen_fusion import numeric, compare


def frames():
    index = pd.date_range('2024-01-01', periods=1100, freq='h', tz='UTC')
    speed = pd.Series(420 + 25 * np.sin(np.arange(len(index)) / 30), index=index)
    ace = pd.DataFrame({'filled_speed_kms': speed, 'was_missing': 0}, index=index)
    ch = pd.DataFrame(index=index)
    raw = pd.DataFrame({k: 1.0 for k in ('ace_density_cm3', 'ace_temperature_k', 'ace_bt_nt', 'ace_speed_kms')}, index=index)
    return dict(base=build_features(ace, ch), physics=build_physics_frame(raw, index),
                quality=build_ch_quality_frame(ch, index), speed=speed,
                observed=pd.Series(True, index=index))


def config():
    c = json.loads((Path(__file__).resolve().parents[1] / 'experiments/vjepa_frozen_fusion/leaderboard_reference/comparison_protocol.json').read_text())['config']
    return {**c, 'max_train_origins': 32, 'params': {**c['params'], 'n_estimators': 3, 'max_depth': 2}}


def embeddings(path, index, values=None):
    if values is None:
        values = np.random.default_rng(1).normal(size=(len(index), 24)).astype('float32')
    np.savez(path, origins=index.asi8, image=values, video=values,
             image_source_end_ns=index.asi8, video_source_end_ns=index.asi8,
             image_quality=np.ones((len(index), 2)), video_quality=np.ones((len(index), 2)))


def test_baseline_exact_parity_replay_and_future_numeric_invariance(tmp_path):
    f = frames(); cutoff = f['speed'].index[900]; origins = f['speed'].index[[980, 990]]
    a, ma = numeric.fit_predict(f, cutoff, origins, config(), 'cpu', tmp_path / 'adapter')
    b, mb = tree.fit_predict(f, cutoff, origins, config(), 'cpu', tmp_path / 'frozen')
    np.testing.assert_array_equal(a.pred_kms, b.pred_kms)
    assert ma['train_origin_sha256'] == mb['train_origin_sha256']
    replay = numeric.predict_saved(f, origins, tmp_path / 'adapter', 'cpu')
    np.testing.assert_allclose(a.pred_kms, replay.pred_kms, atol=1e-4)
    poisoned = copy.deepcopy(f)
    for key, data in poisoned.items():
        data.loc[data.index > origins[0]] = False if key == 'observed' else 99999
    deleted = {k: v.loc[:origins[0]] for k, v in f.items()}
    expected = numeric.predict_saved(f, origins[:1], tmp_path / 'adapter', 'cpu')
    for altered in (poisoned, deleted):
        actual = numeric.predict_saved(altered, origins[:1], tmp_path / 'adapter', 'cpu')
        np.testing.assert_array_equal(expected.pred_kms, actual.pred_kms)
    assert len(a) == 144 and np.isfinite(a.pred_kms).all()


def test_visual_train_only_pca_missing_and_future_embedding_invariance(tmp_path):
    f = frames(); index = f['speed'].index; cutoff = index[900]; origins = index[[980, 990]]
    path = tmp_path / 'embeddings.npz'
    values = np.random.default_rng(4).normal(size=(len(index), 24)).astype('float32')
    values[700:720] = np.nan
    embeddings(path, index, values)
    pred, meta = numeric.fit_predict(f, cutoff, origins, config(), 'cpu', tmp_path / 'model', 'image', path)
    replay = numeric.predict_saved(f, origins, tmp_path / 'model', 'cpu', path)
    np.testing.assert_allclose(pred.pred_kms, replay.pred_kms, atol=1e-4)
    assert len(pred) == 144 and np.isfinite(pred.pred_kms).all()
    with np.load(tmp_path / 'model/visual_transform.npz') as z:
        assert np.max(z['train_origins']) + 72 * 3600 * 10**9 <= cutoff.value
        original = z['components'].copy()
    altered = values.copy(); altered[index > cutoff] = 1e9
    embeddings(tmp_path / 'poison.npz', index, altered)
    pool = numeric.train_origins(f, cutoff, 32)
    numeric.visual_features(tmp_path / 'poison.npz', 'image', pool, index, tmp_path / 'transform')
    with np.load(tmp_path / 'transform/visual_transform.npz') as z:
        np.testing.assert_array_equal(original, z['components'])
    expected = numeric.predict_saved(f, origins[:1], tmp_path / 'model', 'cpu', path)
    for truncate in (False, True):
        kept = index <= origins[0] if truncate else np.ones(len(index), bool)
        altered = values.copy(); altered[index > origins[0]] = 1e9
        embeddings(tmp_path / 'future.npz', index[kept], altered[kept])
        actual = numeric.predict_saved(f, origins[:1], tmp_path / 'model', 'cpu', tmp_path / 'future.npz')
        np.testing.assert_array_equal(expected.pred_kms, actual.pred_kms)
    assert meta['visual_gain_sum'].keys() == {'pca', 'quality', 'missing'}


def test_future_source_and_missing_train_fail_closed(tmp_path):
    index = pd.date_range('2024-01-01', periods=32, freq='h', tz='UTC')
    path = tmp_path / 'bad.npz'
    np.savez(path, origins=index.asi8, image=np.ones((32, 24)), image_source_end_ns=index.asi8 + 1)
    with pytest.raises(ValueError, match='future image'):
        numeric.read_embeddings(path, 'image')
    embeddings(path, index)
    with pytest.raises(ValueError, match='missing planned train'):
        numeric.visual_features(path, 'image', index + pd.Timedelta(days=2), index, tmp_path)


def test_selection_checks_only_precision_recall():
    base = dict(precision=.5, recall=.5, height_mae_kms=10, time_mae_hours=2)
    assert compare.eligible({**base, 'height_mae_kms': 100}, base)
    assert not compare.eligible({**base, 'recall': .4}, base)
    assert not compare.eligible({**base, 'precision': None}, base)


def test_scoring_identical_keys_observation_masks_and_buckets():
    f = frames(); origins = f['speed'].index[[980, 990]]
    pred = pd.DataFrame({'origin_last_input_utc': np.repeat(origins, 72), 'horizon_hours': np.tile(np.arange(1, 73), 2), 'pred_kms': 450.})
    f['observed'].iloc[995] = False
    a = compare.score(pred, f, origins)
    pred['pred_kms'] += 20
    b = compare.score(pred, f, origins)
    for key in ('target_observation_mask_sha256', 'prediction_keys_sha256'):
        assert a[key] == b[key]
    assert a['n_targets'] == 144 and a['n_observed'] == 142
    assert set(a['horizon_buckets']) == {'1-24', '25-48', '49-72'}
    assert all(v['n_targets'] == 48 for v in a['horizon_buckets'].values())
