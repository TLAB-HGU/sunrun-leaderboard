"""Synthetic only: chronological calibration, composite cache and paired restoration."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
from experiments.mse20_hss_phase import model as m
from test_mse20_hss_arrival_model import frames


def test_two_fixed_ablations_config_and_chronological_assignments():
    configs = m.candidate_configs()
    assert [c['id'] for c in configs] == ['shift_only', 'shift_gain']
    assert m._hash(m.fixed_backbone_config()) == m.BACKBONE_CONFIG_SHA
    for config in configs: m.validate_config(config)
    with pytest.raises(ValueError): m.validate_config(dict(configs[0], epochs=2))
    cutoff = pd.Timestamp('2025-01-01T00:00Z')
    origins = pd.DatetimeIndex([cutoff-pd.Timedelta(days=d) for d in (179, 90, 89, 4)])
    splits, assignment = m.calibration_assignments(origins, cutoff)
    assert assignment.tolist() == [0, 0, 1, 1]
    assert splits == [cutoff-pd.Timedelta(days=180), cutoff-pd.Timedelta(days=90)]
    with pytest.raises(ValueError, match='noncausal'):
        m.calibration_assignments(pd.DatetimeIndex([cutoff-pd.Timedelta(days=180), cutoff-pd.Timedelta(days=4)]), cutoff)


def test_scenarios_boundary_sign_gain_pair_and_gradients():
    base = np.arange(1, 73, dtype=np.float32)[None]+400
    past = np.arange(-24, 1, dtype=np.float32)[None]+400
    take = np.clip(np.arange(1, 73)[None]+np.array(m.SHIFTS)[:, None], -24, 72)+24
    shifted = np.concatenate([past, base], axis=1)[:, take]
    assert shifted[0, 0, 0] == 377 and shifted[0, -1, -1] == 472
    features = dict(base=base, shifted=shifted)
    np.testing.assert_allclose(m.unit_scenarios(features)[0, 4], base[0])
    m.setup()
    for config in m.candidate_configs():
        net = m.PhaseModel(config, 8).eval()
        with torch.no_grad(): net.shift.bias[8] = 5
        output = net(torch.ones(1, 8), torch.tensor([400.]), torch.from_numpy(base), torch.from_numpy(shifted))
        assert output['mode'].item() == 8 and output['gain'].item() == 1
        assert output['scenarios'][0, 8, 23] == 448  # Positive shift advances profile.
        torch.testing.assert_close(output['mse'], (output['probabilities'][:, :, None]*output['scenarios']).sum(1))
        torch.testing.assert_close(output['peak'], output['scenarios'][:, 8])
        output['mse'].square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_features_future_poison_and_soft_label_cutoff():
    f = frames(); origins = f['speed'].index[[1000, 1002]]; config = m.candidate_configs()[0]
    prediction = m._frame(origins, np.tile(np.arange(72)+400., (2, 1)))
    features = m.make_features(f, origins, prediction, config)
    assert features['values'].shape == (2, 670)
    for source in ('ace', 'ch', 'raw'):
        raw = {k: f[k].copy() for k in ('ace', 'ch', 'raw')}
        raw[source].loc[raw[source].index > origins.max()] = 1e7
        other = m.make_features(m.state.inputs.rebuild_frames(**raw), origins, prediction, config)
        for key in ('values', 'shifted', 'base'):
            np.testing.assert_array_equal(features[key], other[key])
    cutoff = origins.max()+pd.Timedelta(hours=96)
    labels = m.make_labels(f, origins, cutoff, features)
    np.testing.assert_allclose(labels['soft_shift'].sum(1), 1, atol=1e-6)
    assert (labels['soft_shift'] > 0).all()
    poison = copy.deepcopy(f); poison['speed'].loc[poison['speed'].index > cutoff] = 1e7
    assert m.label_hashes(m.make_labels(poison, origins, cutoff, features)) == m.label_hashes(labels)
    with pytest.raises(ValueError, match='crosses cutoff'):
        m.make_labels(f, origins, cutoff-pd.Timedelta(hours=1), features)


def _fake_backbone(monkeypatch):
    calls = []
    real_pool = m.backbone.training_origins
    monkeypatch.setattr(m.backbone, 'training_origins', lambda f, c, config: real_pool(f, c, config)[-4:])
    def predict(f, origins, directory, device):
        origins = m._origins(origins)
        current = f['speed'].reindex(origins).to_numpy()
        return m._frame(origins, current[:, None]+np.arange(1, 73)[None]*.2)
    def fit(f, cutoff, origins, config, device, directory):
        calls.append((cutoff, list(origins)))
        directory = Path(directory); directory.mkdir(parents=True)
        proof = m.backbone_training_proof(f, cutoff)
        (directory/'model-band0.ubj').write_bytes(b'synthetic-native-placeholder')
        metadata = dict(family='adaptive_tree', config=config, cutoff=pd.Timestamp(cutoff).isoformat(),
            train_label_max=proof['train_label_max'], n_train_origins=proof['n_train_origins'],
            train_origin_sha256=proof['train_origin_sha256'],
            model_files=[dict(path='model-band0.ubj', sha256=m.digest(directory/'model-band0.ubj'))])
        (directory/'metadata.json').write_text(json.dumps(metadata))
        (directory/'metadata.sha256').write_text(m.digest(directory/'metadata.json'))
        return predict(f, origins, directory, device), metadata
    monkeypatch.setattr(m.backbone, 'fit_predict', fit)
    monkeypatch.setattr(m.backbone, 'predict_saved', predict)
    return calls


def test_composite_shared_cache_actual_oof_scaler_restore_provenance(tmp_path, monkeypatch):
    f = frames(5900); cutoff = f['speed'].index[5700]; origins = f['speed'].index[[5701, 5702]]
    pool = pd.DatetimeIndex([cutoff-pd.Timedelta(days=120), cutoff-pd.Timedelta(days=30)])
    monkeypatch.setattr(m, 'training_origins', lambda *args: pool)
    calls = _fake_backbone(monkeypatch)
    (tmp_path/'registered.json').write_text('{}')
    metadata = []
    for config in m.candidate_configs():
        directory = tmp_path/config['id']/'model'
        pair, meta = m.fit_predict(f, cutoff, origins, config, 'cpu', directory)
        metadata.append(meta)
        restored = m.predict_saved(f, origins, directory)
        for key in ('mse', 'peak', 'backbone'):
            np.testing.assert_allclose(pair[key].pred_kms, restored[key].pred_kms, rtol=0, atol=1e-3)
        proof = m.verify_training_provenance(f, directory)
        assert proof['train_label_hashes'] == meta['train_label_hashes']
        assert meta['train_assignment'] == [0, 1] and meta['input_width'] == 1340
        for record in meta['backbones']:
            m._verify(record['result']); m._verify(record['training_provenance'])
            claim = json.loads(Path(record['claim']['path']).read_text())
            assert claim['owner_birth'] and claim['claimed_utc']
    assert len(calls) == 3  # Both heads share exactly three successful fits.
    assert metadata[0]['backbones'] == metadata[1]['backbones']
    # Rehashing an altered learned scaler cannot evade independent reconstruction.
    path = tmp_path/'shift_gain/model/metadata.json'; meta = json.loads(path.read_text())
    meta['normalizer']['mean'][0] += 1
    path.write_text(json.dumps(meta)); (path.parent/'metadata.sha256').write_text(m.digest(path))
    with pytest.raises(ValueError, match='scaler'):
        m.verify_training_provenance(f, path.parent)


def test_failed_cache_claim_never_retries(tmp_path, monkeypatch):
    (tmp_path/'registered.json').write_text('{}')
    f = frames(); cutoff = f['speed'].index[1000]; origins = f['speed'].index[[1001]]
    calls = []
    def fail(*args): calls.append(1); raise RuntimeError('synthetic failure')
    monkeypatch.setattr(m.backbone, 'fit_predict', fail)
    for index in range(2):
        with pytest.raises((RuntimeError, ValueError), match='failure|retry forbidden'):
            m.cached_backbone(f, cutoff, cutoff, origins, 'deployment', 'cpu', tmp_path)
    assert calls == [1]


def test_small_real_cpu_backbone_native_fields_and_training_proof(tmp_path):
    (tmp_path/'registered.json').write_text('{}')
    # Exact frozen400-tree configuration, only two eligible synthetic origins.
    f = frames(740); cutoff = f['speed'].index[720]; origins = f['speed'].index[[721, 722]]
    prediction, record = m.cached_backbone(f, cutoff, cutoff, origins, 'deployment', 'cpu', tmp_path)
    restored = m.backbone.predict_saved(f, origins, record['model_dir'], 'cpu')
    np.testing.assert_allclose(prediction.pred_kms, restored.pred_kms, rtol=0, atol=1e-3)
    proof = json.loads(Path(record['training_provenance']['path']).read_text())
    metadata = json.loads((Path(record['model_dir'])/'metadata.json').read_text())
    assert proof == m.backbone_training_proof(f, cutoff)
    for key in ('train_origin_sha256', 'n_train_origins', 'train_label_max'):
        assert proof[key] == metadata[key]
    assert proof['n_train_origins'] <= 3
    _, cached = m.cached_backbone(f, cutoff, cutoff, origins, 'deployment', 'cpu', tmp_path)
    assert cached == record


def test_backbone_budget_and_interrupted_claim_fail_closed(tmp_path, monkeypatch):
    (tmp_path/'registered.json').write_text('{}')
    f = frames(); cutoff = f['speed'].index[1000]; origins = f['speed'].index[[1001]]
    for i in range(42):
        path = tmp_path/'backbones'/f'cutoff-synthetic{i}'/'oof0'/'claim.json'
        path.parent.mkdir(parents=True); path.write_text(json.dumps({'identity': {'scope': 'dev'}}))
    def forbidden(*args): raise AssertionError('budget must reject before fit')
    monkeypatch.setattr(m.backbone, 'fit_predict', forbidden)
    with pytest.raises(ValueError, match='budget exhausted'):
        m.cached_backbone(f, cutoff, cutoff, origins, 'deployment', 'cpu', tmp_path)
