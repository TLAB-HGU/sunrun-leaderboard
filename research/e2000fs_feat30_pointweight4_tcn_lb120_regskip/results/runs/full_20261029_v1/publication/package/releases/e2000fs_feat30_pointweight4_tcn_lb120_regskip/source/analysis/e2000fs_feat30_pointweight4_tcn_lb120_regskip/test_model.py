"""Behavior contracts for history causality, horizon weighting and persistence."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


spec = importlib.util.spec_from_file_location('feat30_tcn_regskip_model', Path(__file__).with_name('model.py'))
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_causal_encoder_ignores_later_timesteps():
    torch.manual_seed(42)
    net = model.TCN(tuple(range(21))).eval()
    history = torch.randn(2, 120, 30)
    changed = history.clone()
    changed[:, 80:] += 100
    with torch.no_grad():
        actual = net.encoder(history.transpose(1, 2))
        poisoned = net.encoder(changed.transpose(1, 2))
    torch.testing.assert_close(actual[:, :, :80], poisoned[:, :, :80], rtol=0, atol=0)
    assert not torch.equal(actual[:, :, 80:], poisoned[:, :, 80:])


def test_oldest_hour_can_influence_last_context():
    net = model.TCN(tuple(range(21))).eval()
    # Positive parameters keep ReLUs active, testing reachability independently
    # of random initialization rather than expecting every random unit to fire.
    with torch.no_grad():
        for block in net.encoder:
            block.conv.weight.fill_(0.01)
            block.conv.bias.fill_(0.1)
    history = torch.ones(1, 120, 30, requires_grad=True)
    net.encode(history).sum().backward()
    assert torch.isfinite(history.grad).all()
    assert (history.grad[0, 0] > 0).all()
    assert (history.grad[0, -1] > 0).all()


def test_predictions_are_finite_and_future_channels_are_horizon_local():
    torch.manual_seed(42)
    net = model.TCN(tuple(range(21))).eval()
    history = torch.randn(3, 120, 30)
    future = torch.randn(3, 72, 9)
    changed = future.clone()
    changed[:, 10] += 100
    with torch.no_grad():
        baseline = net(history, future)
        candidate = net(history, changed)
    assert baseline.shape == (3, 72)
    assert torch.isfinite(baseline).all()
    keep = torch.arange(72) != 10
    torch.testing.assert_close(baseline[:, keep], candidate[:, keep], rtol=0, atol=0)
    assert not torch.equal(baseline[:, 10], candidate[:, 10])


def test_point_weights_normalize_per_horizon_without_neighbor_spill():
    y = np.full((4, 72), 400.)
    y[0, 0] = 550.
    y[1, 0] = 549.
    y[:2, 2] = 700.
    weights = model.point_weights(y)
    np.testing.assert_allclose(weights.mean(axis=0), 1., rtol=1e-6)
    np.testing.assert_allclose(weights[:, 0] / weights[1, 0], [4., 1., 1., 1.])
    np.testing.assert_array_equal(weights[:, 1], np.ones(4))
    np.testing.assert_allclose(weights[:, 2] / weights[2, 2], [4., 4., 1., 1.])


@pytest.mark.parametrize('bad', [np.full((1, 72), np.nan), np.full((1, 72), np.inf),
                                  np.zeros(72), np.zeros((2, 71)), np.zeros((0, 72))])
def test_invalid_target_matrices_rejected(bad):
    with pytest.raises(ValueError, match='targets'):
        model.point_weights(bad)


def test_weighted_mse_preserves_elementwise_scale_and_gradient():
    pred = torch.ones(2, 72, requires_grad=True)
    target = torch.zeros_like(pred)
    weights = torch.ones_like(pred)
    weights[0, 0] = 4
    loss = model.weighted_mse(pred, target, weights)
    torch.testing.assert_close(loss, torch.tensor(147 / 144))
    loss.backward()
    torch.testing.assert_close(pred.grad, 2 * weights / 144)
    with pytest.raises(ValueError, match='match'):
        model.weighted_mse(pred, target, weights[:, :1])


def test_input_shapes_rejected():
    net = model.TCN(tuple(range(21)))
    with pytest.raises(ValueError, match='history'):
        net(torch.zeros(1, 119, 30), torch.zeros(1, 72, 9))
    with pytest.raises(ValueError, match='future'):
        net(torch.zeros(1, 120, 30), torch.zeros(1, 71, 9))


def test_checkpoint_roundtrip_preserves_predictions(tmp_path):
    torch.manual_seed(42)
    net = model.TCN(tuple(range(21))).eval()
    history, future = torch.randn(2, 120, 30), torch.randn(2, 72, 9)
    with torch.no_grad():
        expected = net(history, future)
    path = tmp_path / 'model.pt'
    torch.save(net.state_dict(), path)
    restored = model.TCN(tuple(range(21))).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(restored(history, future), expected, rtol=0, atol=0)


def test_current_speed_reaches_head_when_encoder_is_zero():
    # Route a unit signal only through the current-speed input to the head.
    # The output and gradient must survive completely disabled convolutions.
    current_indices = (2, 0, *range(3, 22))
    net = model.TCN(current_indices).eval()
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
        net.head[0].weight[0, model.ARCH['width'] + current_indices.index(0)] = 1.
        net.head[3].weight[0, 0] = 1.
        net.head[6].weight[0, 0] = 1.
    history = torch.ones(1, 120, 30, requires_grad=True)
    future = torch.zeros(1, 72, 9)
    output = net(history, future)
    torch.testing.assert_close(output, torch.ones(1, 72))
    output.sum().backward()
    assert history.grad[0, -1, 0].item() == 72.
    assert history.grad[:, :-1].count_nonzero().item() == 0
    changed = history.detach().clone()
    changed[:, :-1] = 999.
    with torch.no_grad():
        torch.testing.assert_close(net(changed, future), output.detach(), rtol=0, atol=0)
        changed[:, -1, 0] = 2.
        torch.testing.assert_close(net(changed, future), 2 * output.detach(), rtol=0, atol=0)


def test_dropout_is_active_in_training_and_disabled_for_evaluation():
    torch.manual_seed(42)
    net = model.TCN(tuple(range(21)))
    history, future = torch.randn(4, 120, 30), torch.randn(4, 72, 9)
    with torch.no_grad():
        assert not torch.equal(net(history, future), net(history, future))
        net.eval()
        torch.testing.assert_close(net(history, future), net(history, future), rtol=0, atol=0)
    assert sum(p.numel() for p in net.parameters()) < 85505


@pytest.mark.parametrize('indices', [tuple(range(20)), (0,) * 21,
                                    tuple(range(1, 22)), (0, *range(1, 20), 30),
                                    (0, *range(1, 20), -1), (False, *range(1, 21)),
                                    (0., *range(1, 21))])
def test_invalid_current_channel_mapping_rejected(indices):
    with pytest.raises(ValueError, match='current_indices'):
        model.TCN(indices)
