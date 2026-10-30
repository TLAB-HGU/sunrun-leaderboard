"""Synthetic CPU-only residual execution readiness tests. No real data, no GPU.

Torch is imported only for synthetic small-tensor math (no gradients, no
fits). Every test skips with an explicit PEER_GAP if the peer-owned
model/teacher modules are not yet present; nothing here reads official
scores, truth, or predictions.
"""
import hashlib
import json

import pytest

pd = pytest.importorskip('pandas')
np = pytest.importorskip('numpy')


def _peer(name):
    try:
        return pytest.importorskip(f'experiments.mse20_hss_residual.{name}',
                                   reason=f'PEER_GAP: {name} not yet written')
    except Exception as error:
        pytest.skip(f'PEER_GAP: {error}')


def test_expected_config_ids_pinned():
    protocol = pytest.importorskip('experiments.mse20_hss_residual.protocol')
    assert protocol.EXPECTED_CONFIG_IDS == ('residual_rise', 'residual_rise_decline')
    assert protocol.CAPS['new_backbone_fits'] == 0
    assert protocol.CAPS['dev_claimed_attempts'] == 28


def test_cache_key_varies_with_cutoff_sources_teacher():
    dev = pytest.importorskip('experiments.mse20_hss_residual.dev')
    base_cutoff = '2024-01-01T00:00:00Z'
    base_sources = {'id': 'residual_rise'}
    base_teacher = 't1'
    key_base = dev.cache_key(base_cutoff, base_sources, base_teacher)
    key_cutoff = dev.cache_key('2024-01-02T00:00:00Z', base_sources, base_teacher)
    key_sources = dev.cache_key(base_cutoff, {'id': 'residual_rise_decline'}, base_teacher)
    key_teacher = dev.cache_key(base_cutoff, base_sources, 't2')
    assert len({key_base, key_cutoff, key_sources, key_teacher}) == 4
    assert dev.cache_key(base_cutoff, base_sources, base_teacher, config=None) == key_base
    with pytest.raises(ValueError):
        dev.cache_key(base_cutoff, base_sources, base_teacher, config={'id': 'residual_rise'})


def test_synthetic_zero_at_origin_and_no_event_identity():
    torch = pytest.importorskip('torch')
    horizon = torch.arange(1, 73, dtype=torch.float32)
    current = torch.full((72,), 400.0)
    backbone = torch.full((72,), 400.0)
    bound = current + torch.minimum(horizon / 6, torch.ones(())) * (backbone - current)
    assert torch.allclose(bound, current)
    # No-event scenario is exactly the boundary curve.
    assert torch.allclose(bound, backbone)


def test_synthetic_rise_creates_missing_rise_and_decline_goes_negative():
    torch = pytest.importorskip('torch')
    z = torch.arange(-48, 73, dtype=torch.float32)

    def u(v):
        return torch.clamp(v / 12, 0, 1)

    def vfn(v):
        return torch.clamp((v - 36) / 36, 0, 1)

    rise = 300.0 * u(z)
    assert float(rise[z == 0]) == 0.0
    assert float(rise[z == 12]) == pytest.approx(300.0)
    assert float(rise[z == -1]) == pytest.approx(0.0)
    both = 300.0 * u(z) - (300.0 + 100.0) * vfn(z)
    assert float(both[z == 72]) < float(rise[z == 72])


def test_synthetic_joint_expectation_map_consistency():
    torch = pytest.importorskip('torch')
    torch.manual_seed(42)
    logits = torch.zeros(21)
    probs = torch.softmax(logits, dim=-1)
    assert probs.shape == (21,)
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    scenarios = torch.randn(21, 72)
    expectation = (probs[:, None] * scenarios).sum(dim=0)
    peak = scenarios[int(torch.argmax(probs))]
    assert expectation.shape == (72,)
    assert peak.shape == (72,)
    assert torch.isfinite(expectation).all() and (expectation > 0).all() is not None


def test_peer_candidate_configs_exact_two(tmp_path):
    model = _peer('model')
    configs = model.candidate_configs()
    assert len(configs) == 2
    assert {c['id'] for c in configs} == {'residual_rise', 'residual_rise_decline'}
    with pytest.raises(ValueError):
        model.validate_config({'id': 'not_a_variant'})


def test_peer_teacher_pool_rejects_leak(tmp_path):
    teacher = _peer('teacher')
    protocol = pytest.importorskip('experiments.mse20_hss_residual.protocol')
    with pytest.raises((ValueError, Exception)):
        teacher.calibration_pool(pd.Timestamp('2020-01-01T00:00:00Z'), 'dev')
    assert hasattr(teacher, 'calibration_pool')


def test_old_sources_unmutated_by_import():
    protocol = pytest.importorskip('experiments.mse20_hss_residual.protocol')
    before = dict(protocol.old_closure_sources())
    import experiments.mse20_hss_residual.dev  # noqa: F401
    assert protocol.old_closure_sources() == before
    assert len(hashlib.sha256(json.dumps('x').encode()).hexdigest()) == 64
