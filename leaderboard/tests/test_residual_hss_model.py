"""Synthetic-only residual head tests; no scientific data, no GPU, CPU only."""
import numpy as np
import pandas as pd
import pytest
import torch

from experiments.mse20_hss_residual import model as m


def test_candidate_configs_exact_two():
    configs = m.candidate_configs()
    assert [(c['id'], c['decline']) for c in configs] == [
        ('residual_rise', False), ('residual_rise_decline', True)]
    for config in configs:
        m.validate_config(config)
        assert (config['width'], config['dropout'], config['epochs'], config['batch_size'],
                config['learning_rate'], config['weight_decay'], config['seed']) == \
            (48, .1, 20, 128, .001, .001, 42)
    with pytest.raises(ValueError):
        m.validate_config(dict(configs[0], epochs=21))


def test_sources_include_original_plus_new():
    from experiments.mse20_hss_phase_ff1 import evaluate as ff1eval
    assert set(ff1eval.MODEL_SOURCE_NAMES) < set(m.MODEL_SOURCE_NAMES)
    assert 'experiments/mse20_hss_residual/model.py' in m.MODEL_SOURCE_NAMES
    assert 'experiments/mse20_hss_residual/teacher.py' in m.MODEL_SOURCE_NAMES
    sources = m._sources()
    assert set(sources) == {str((m.ROOT / n).resolve()) for n in m.MODEL_SOURCE_NAMES}


def test_noevent_identity_and_zero_origin_anchor():
    rng = np.random.default_rng(0)
    base = rng.uniform(300, 700, size=(4, 72)).astype(np.float32)
    current = rng.uniform(300, 700, size=4).astype(np.float32)
    curve = m.anchored_base(current, base)
    amp = np.full(4, 50, dtype=np.float32)
    dec = np.full(4, 20, dtype=np.float32)
    for use_decline in (False, True):
        scenarios = m.residual_scenarios(curve, amp, dec, use_decline)
        assert scenarios.shape == (4, 21, 72)
        np.testing.assert_allclose(scenarios[:, m.NOEVENT], np.clip(curve, 1, 3000), rtol=0, atol=0)
        # Residual offset is exactly zero at issue time for every bin.
        for tau in m.BIN_CENTERS:
            zero = np.zeros(4, dtype=np.float32)
            offset = (m.pulse_shape(zero - np.float32(tau), amp, dec, use_decline)
                      - m.pulse_shape(np.full(4, -np.float32(tau), dtype=np.float32),
                                       amp, dec, use_decline))
            np.testing.assert_allclose(offset, 0., atol=1e-4)
    # Near-lead boundary: first 6h ramps from current to backbone.
    np.testing.assert_allclose(curve[:, 0], current + (base[:, 0] - current) / 6, rtol=1e-5)
    np.testing.assert_allclose(curve[:, 5:], base[:, 5:], rtol=1e-5)


def test_flat_baseline_pulse_creates_rise():
    current = np.full(2, 400, dtype=np.float32)
    base = np.full((2, 72), 400, dtype=np.float32)
    curve = m.anchored_base(current, base)
    scenarios = m.residual_scenarios(curve, np.full(2, 200.), np.zeros(2), False)
    assert scenarios[:, :20].max() > 500  # missing rise is created
    assert scenarios.min() >= 1 and scenarios.max() <= 3000


def test_decline_permits_negative_correction():
    current = np.full(2, 600, dtype=np.float32)
    base = np.full((2, 72), 600, dtype=np.float32)
    curve = m.anchored_base(current, base)
    scenarios = m.residual_scenarios(curve, np.full(2, 100.), np.full(2, 250.), True)
    assert scenarios[:, :20].min() < 600  # post-event dip below baseline
    rise_only = m.residual_scenarios(curve, np.full(2, 100.), np.zeros(2), False)
    assert (rise_only[:, :20].min() >= 600 - 1e-3)


def test_expectation_joint_map_consistency():
    torch.manual_seed(0)
    model = m.ResidualModel(m.candidate_configs()[1], 8)
    model.eval()
    x = torch.randn(3, 8)
    current = torch.full((3,), 400.)
    base = torch.full((3, 72), 450.)
    out = model(x, current, base)
    assert out['scenarios'].shape == (3, 21, 72)
    assert ((out['amplitude'] >= 0) & (out['amplitude'] <= 600)).all()
    assert ((out['decline'] >= 0) & (out['decline'] <= 250)).all()
    assert out['mse'].min() >= 1 and out['mse'].max() <= 3000
    # Expectation equals probability-weighted scenarios; peak equals MAP scenario.
    np.testing.assert_allclose(out['mse'].detach().numpy(),
                               (out['probabilities'][:, :, None] * out['scenarios']).sum(1).detach().numpy(),
                               rtol=1e-5)
    idx = torch.arange(3)
    np.testing.assert_allclose(out['peak'].detach().numpy(),
                               out['scenarios'][idx, out['mode']].detach().numpy(), rtol=1e-5)
    # Laplace prior init: small amplitude, near-uniform-ish logits from log prior.
    assert float(out['amplitude'].detach().mean()) < 30


def test_torch_matches_numpy_scenarios():
    torch.manual_seed(1)
    model = m.ResidualModel(m.candidate_configs()[0], 4)
    model.eval()
    x = torch.randn(2, 4)
    current = torch.tensor([380., 520.])
    base = torch.full((2, 72), 430.)
    out = model(x, current, base)
    amp = out['amplitude'].detach().numpy()
    curve = m.anchored_base(current.numpy(), base.numpy())
    expected = m.residual_scenarios(curve, amp, np.zeros(2, dtype=np.float32), False)
    np.testing.assert_allclose(out['scenarios'].detach().numpy(), expected, rtol=1e-4, atol=1e-3)


def test_cpu_forward_determinism_and_roundtrip(tmp_path):
    torch.manual_seed(2)
    model = m.ResidualModel(m.candidate_configs()[1], 6)
    model.eval()
    x = torch.randn(5, 6)
    current = torch.full((5,), 410.)
    base = torch.full((5, 72), 460.)
    first = model(x, current, base)['mse'].detach()
    second = model(x, current, base)['mse'].detach()
    torch.testing.assert_close(first, second)
    path = tmp_path / 'model.pt'
    torch.save(model.state_dict(), path)
    clone = m.ResidualModel(m.candidate_configs()[1], 6)
    clone.load_state_dict(torch.load(path, weights_only=True))
    clone.eval()
    torch.testing.assert_close(clone(x, current, base)['mse'].detach(), first)


def test_masked_normalizer_keeps_nan_rows_finite():
    values = np.array([[1., np.nan, 3.], [4., 5., 6.]], dtype=np.float32)
    normalizer = m.fit_normalizer(values)
    out = m.transform(values, normalizer)
    assert out.shape == (2, 6)  # values plus validity mask, no rows dropped
    assert np.isfinite(out).all()
    assert out[0, 1] == 0 and out[0, 4] == 0 and out[1, 4] == 1


def synthetic_speed():
    idx = pd.date_range('2024-01-01T00:00:00Z', periods=720, freq='h', tz='UTC')
    speed = pd.Series(np.where(np.arange(720) < 500, 400., 700.), index=idx)
    observed = pd.Series(True, index=idx)
    return dict(speed=speed, observed=observed)


def test_labels_bin_and_noevent():
    frames = synthetic_speed()
    onsets = m.state.onset_times(frames['speed'], frames['observed'])
    assert len(onsets) >= 1
    onset = onsets[0]
    origin = onset - pd.Timedelta(hours=12)
    cutoff = origin + pd.Timedelta(hours=96)
    labels = m.make_labels(frames, pd.DatetimeIndex([origin]), cutoff)
    assert labels['arrival'][0] == (12 + 47) // 6
    assert labels['tau_hours'][0] == 12.
    assert labels['profile'].shape == (1, 72) and labels['profile96'].shape == (1, 96)
    far = onset - pd.Timedelta(hours=200)
    quiet = m.make_labels(frames, pd.DatetimeIndex([far]), far + pd.Timedelta(hours=96))
    assert quiet['arrival'][0] == m.NOEVENT and np.isnan(quiet['tau_hours'][0])


def test_labels_reject_crossing_and_ambiguous():
    frames = synthetic_speed()
    origin = pd.Timestamp('2024-01-10T00:00:00Z')
    with pytest.raises(ValueError):
        m.make_labels(frames, pd.DatetimeIndex([origin]), origin + pd.Timedelta(hours=95))
    with pytest.raises(ValueError):
        m.classify_onset(origin, [origin - pd.Timedelta(hours=10), origin + pd.Timedelta(hours=10)])
    assert m.classify_onset(origin, [origin - pd.Timedelta(hours=48)])[0] == m.NOEVENT
    assert m.classify_onset(origin, [origin - pd.Timedelta(hours=47)])[0] == 0
    assert m.classify_onset(origin, [origin + pd.Timedelta(hours=72)])[0] == 19


def test_label_hash_keys():
    frames = synthetic_speed()
    onsets = m.state.onset_times(frames['speed'], frames['observed'])
    origin = onsets[0] - pd.Timedelta(hours=6)
    labels = m.make_labels(frames, pd.DatetimeIndex([origin]), origin + pd.Timedelta(hours=96))
    assert set(m.label_hashes(labels)) == {'profile96', 'arrival', 'tau_hours'}
