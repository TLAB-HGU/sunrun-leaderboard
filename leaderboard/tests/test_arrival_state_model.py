"""Synthetic CPU evidence for compact recurrence/state-conditioned arrival models."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
from experiments.mse20_hss_state import model as m
from test_mse20_hss_arrival_model import frames


def test_only_fixed_prior_ablation_and_compact_schema():
    configs = m.candidate_configs()
    assert len(configs) == 2
    a, b = [dict(c) for c in configs]
    for c in (a, b):
        m.validate_config(c)
        assert (c['width'], c['bin_hours'], c['slow_hours'], c['epochs']) == (48, 6, 168, 20)
    assert {a.pop('prior_strength'), b.pop('prior_strength')} == {0., 1.}
    a.pop('id'); b.pop('id'); assert a == b
    with pytest.raises(ValueError): m.validate_config(dict(configs[0], prior_sigma_hours=6.))
    f = frames()
    values = m.make_features(f, f['speed'].index[[1000]], configs[0])
    assert values['values'].shape == (1, 653)
    assert values['schema']['version'] == 3


def test_prior_actual_prefix_future_poison_and_missing(monkeypatch):
    f = frames(); origins = f['speed'].index[[1000, 1002]]; config = m.candidate_configs()[1]
    helper = m.onset_times; seen = []
    def recording(speed, observed=None):
        seen.append(speed.index.max())
        return helper(speed, observed)
    monkeypatch.setattr(m, 'onset_times', recording)
    before = m.make_features(f, origins, config)
    assert seen == list(origins)
    for source in ('ace', 'ch', 'raw'):
        raw = {k: f[k].copy() for k in ('ace', 'ch', 'raw')}
        raw[source].loc[raw[source].index > origins.max()] = 1e7
        changed = m.make_features(m.inputs.rebuild_frames(**raw), origins, config)
        for field in ('values', 'slow', 'prior_delta'):
            np.testing.assert_array_equal(before[field], changed[field])
    # Even a helper returning an unresolved future-label onset cannot admit it.
    monkeypatch.setattr(m, 'onset_times', lambda *args: pd.DatetimeIndex([origins[0]-pd.Timedelta(hours=1)]))
    missing = m.make_features(f, origins[:1], config)
    assert np.isnan(missing['prior_delta'][0]) and missing['values'][0, -1] == 1
    known = origins[0]-pd.Timedelta(days=27)+pd.Timedelta(hours=30)
    monkeypatch.setattr(m, 'onset_times', lambda *args: pd.DatetimeIndex([known]))
    valid = m.make_features(f, origins[:1], config)
    assert valid['prior_delta'][0] == 30 and valid['values'][0, -1] == 0


def test_geometry_zero_missing_both_signs_and_transport_moments():
    raw = np.zeros((1, 180), dtype=np.float32)
    summary = m._ch_summary(raw)
    assert np.all(summary[:, :5] == 0) and np.isnan(summary[0, 5:7]).all()
    assert np.isnan(m._ch_summary(np.full_like(raw, np.nan))).all()
    for suffix, column in [('negative', 'ch_t035_lon00_lat1_area'), ('positive', 'ch_t035_lon11_lat1_area')]:
        values = raw.copy(); values[0, list(m.inputs.CH_COLUMNS).index(column)] = 2
        geom = m._ch_summary(values)[0]
        assert geom[0] == 2 and geom[1] == 2
        assert geom[3 if suffix == 'negative' else 4] == 2
    curves = np.zeros((1, 72, 75), dtype=np.float32)
    curves[0, 9, :12] = 2; curves[0, 49, 12:24] = 3
    actual = m._transport_summary(curves)[0]
    np.testing.assert_allclose(actual[:6], [2, 10, 0, 3, 50, 0])
    assert np.isnan(m._transport_summary(np.full_like(curves, np.nan))).all()


def test_gaussian_prior_changes_timing_without_occurrence_or_calibration():
    m.setup()
    models = [m.ArrivalPlateau(c, 8).eval() for c in m.candidate_configs()]
    x = torch.zeros(3, 8); current = torch.full((3,), 500.); slow = torch.full((3,), 380.)
    prior = torch.tensor([15.5, 57.5, float('nan')])
    outputs = [net(x, current, slow, prior) for net in models]
    for output in outputs:
        torch.testing.assert_close(output['probabilities'].sum(1), torch.ones(3))
        torch.testing.assert_close(output['event_probability'], torch.full((3,), .5))
        torch.testing.assert_close(output['mse'], (output['probabilities'][:, :, None]*output['scenarios']).sum(1))
        torch.testing.assert_close(output['peak'], output['scenarios'][torch.arange(3), output['mode']])
    assert outputs[1]['mode'].tolist()[:2] == [3, 10]
    torch.testing.assert_close(outputs[0]['probabilities'][2], outputs[1]['probabilities'][2])
    loss = outputs[1]['mse'].square().mean()+m.conditional_cdf_loss(outputs[1]['logits'], torch.tensor([2, 5, 0])).mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in models[1].parameters())
    q, ht = m.balanced_probabilities(np.array([0, 0, 0, 1]))
    np.testing.assert_allclose(q*ht, np.full(4, .25))
    assert np.sum(q*ht*np.array([0, 0, 0, 1])) == pytest.approx(.25)


def test_train_only_normalizer_and_label_cutoff():
    normalizer = m.fit_normalizer(np.array([[1., np.nan, 0.], [3., np.nan, 0.]], dtype=np.float32))
    assert normalizer['mean'] == [2., 0., 0.] and normalizer['count'] == [2, 0, 2]
    unchanged = copy.deepcopy(normalizer)
    result = m.transform(np.array([[1e8, 50., 0.]], dtype=np.float32), normalizer)
    assert np.isfinite(result).all() and normalizer == unchanged
    f = frames(); cutoff = f['speed'].index[1040]; config = m.candidate_configs()[0]
    pool = m.training_origins(f, cutoff, config)
    assert (pool+pd.Timedelta(hours=96) <= cutoff).all()
    origin = f['speed'].index[[960]]
    with pytest.raises(ValueError, match='crosses cutoff'):
        m.make_labels(f, origin, cutoff, config, np.array([380.]))


def test_occurrence_initialization_uses_two_classes_not_timing_bin_count():
    features = dict(values=np.zeros((10, 3), dtype=np.float32))
    labels = dict(arrival=np.array([0]*9+[4]))
    net, _, history = m.train_model(features, labels, m.candidate_configs()[1], 'cpu', epochs=0)
    assert history == []
    # One positive, nine negative: one pseudo-count per class gives2/12.
    net.eval()
    out = net(torch.zeros(1, 6), torch.tensor([400.]), torch.tensor([380.]), torch.tensor([21.5]))
    assert out['event_probability'].item() == pytest.approx(2/12, abs=1e-7)


def test_cpu_fit_reload_provenance_and_postcutoff_invariance(tmp_path):
    f = frames(); cutoff = f['speed'].index[1040]; origins = f['speed'].index[[1041, 1042, 1043]]
    config = m.candidate_configs()[1]
    pair, metadata = m.fit_predict(f, cutoff, origins, config, 'cpu', tmp_path/'first')
    assert metadata['family'] == 'hss_state_recurrence' and metadata['input_width'] == 1306
    assert any(path.endswith('/mse20_hss_shape/model.py') for path in metadata['sources'])
    reloaded = m.predict_saved(f, origins, tmp_path/'first')
    subset = m.predict_saved(f, origins[1:2], tmp_path/'first')
    for key in ('mse', 'peak'):
        np.testing.assert_allclose(pair[key].pred_kms, reloaded[key].pred_kms, rtol=0, atol=1e-3)
        np.testing.assert_allclose(reloaded[key].pred_kms.to_numpy().reshape(3, 72)[1], subset[key].pred_kms, rtol=0, atol=1e-3)
    proof = m.verify_training_provenance(f, tmp_path/'first')
    assert proof['train_shape_label_sha256'] == metadata['train_shape_label_sha256']
    poisoned = copy.deepcopy(f); poisoned['speed'].loc[poisoned['speed'].index > cutoff] = 1e6
    _, other = m.fit_predict(poisoned, cutoff, origins, config, 'cpu', tmp_path/'second')
    assert other['model_sha256'] == metadata['model_sha256'] and other['normalizer'] == metadata['normalizer']
    path = tmp_path/'first/metadata.json'; record = json.loads(path.read_text())
    record['train_shape_label_sha256'] = '0'*64
    path.write_text(json.dumps(record)); (path.parent/'metadata.sha256').write_text(m.digest(path))
    with pytest.raises(ValueError, match='shape/censor'): m.verify_training_provenance(f, tmp_path/'first')
