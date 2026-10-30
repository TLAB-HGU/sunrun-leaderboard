"""Synthetic CPU checks only; no official data or fitting."""
import copy
import itertools
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
from experiments.mse20_ch_dynamics import neural as n


def synthetic_raw(origins):
    raw = {'schema': {'synthetic': True}, 'origins': origins,
           'anchor': np.full((len(origins), 72), 400.),
           'ch_slot_valid': np.ones((len(origins), 120), dtype=bool)}
    for key, shape in n.SHAPES.items():
        row = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / max(1, np.prod(shape))
        raw[key] = np.broadcast_to(row, (len(origins), *shape)).copy()
        raw[key] += np.asarray(origins.hour, dtype=np.float32)[:, None, None] / 24
        raw[key+'_mask'] = np.ones_like(raw[key], dtype=bool)
    return raw


def config(kind='multiscale'):
    return next(c for c in n.candidate_grid() if c['kind'] == kind)


def test_grid_fixed_domains_and_semantic_dedup():
    grid = n.candidate_grid()
    assert grid == n.candidate_grid() and len(grid) == 36
    assert len({n.semantic_digest(c) for c in grid}) == 36
    for c in grid:
        n.validate_config(c)
        assert n.semantic_digest(c) == n.semantic_digest(dict(c, id='display only'))
    assert {c['kind'] for c in grid[:12]} == {'multiscale', 'tcn', 'pulse'}
    with pytest.raises(ValueError):
        n.validate_config(dict(grid[0], max_epochs=1))
    with pytest.raises(ValueError):
        n.validate_config(dict(grid[0], oracle=True))


def test_normalizer_mask_missing_and_channel_statistics():
    raw = synthetic_raw(pd.date_range('2020-01-01', periods=3, freq='h', tz='UTC'))
    raw['ch'][0, 0, 0] = 1e20
    raw['ch_mask'][0, 0, 0] = False
    raw['ace'][:, :, 2] = np.nan
    stats = n.fit_normalizer(raw)
    assert stats['ch']['mean'][0] < 2
    assert stats['ace']['count'][2] == 0
    transformed = n.transform(raw, stats)
    assert transformed['ch'][0, 0, 0] == 0
    assert transformed['ch'][0, 0, 180] == 0
    assert np.isfinite(transformed['ace']).all()
    assert not transformed['ace'][:, :, 2].any()
    raw2 = copy.deepcopy(raw)
    raw2['ch'][0, 0, 0] = -1e30
    assert n.fit_normalizer(raw2) == stats


def test_purged_validation_exact_boundaries():
    cutoff = n.CUTOFF
    pool = pd.date_range(cutoff-pd.Timedelta(days=100), cutoff-pd.Timedelta(hours=72), freq='3h')
    train, valid = n.validation_split(pool, cutoff)
    start = cutoff-pd.Timedelta(days=60)
    assert train.max()+pd.Timedelta(hours=72) <= start-pd.Timedelta(hours=72)
    assert valid.min() >= start+pd.Timedelta(hours=72)
    assert valid.max()+pd.Timedelta(hours=72) <= cutoff


def test_pulse_all_boundaries_and_unequal_rise_fall():
    h = torch.arange(1, 73, dtype=torch.float64)[None]
    for a, on, duration, rise, fall in itertools.product((0., 400.), (-24., 96.), (6., 96.), (1., 24.), (1., 48.)):
        pulse = n.positive_pulse(h, a, on, duration, rise, fall)
        assert torch.isfinite(pulse).all()
        assert (pulse >= 0).all() and (pulse <= a).all()
    old_difference = torch.sigmoid((h-96)/1)-torch.sigmoid((h-102)/48)
    assert (old_difference < 0).any()


@pytest.mark.parametrize('kind', ['multiscale', 'tcn', 'pulse'])
def test_architecture_full_ch_sequence_and_padded_batch(kind):
    n.setup()
    origins = pd.date_range('2020-01-01', periods=3, freq='h', tz='UTC')
    raw = synthetic_raw(origins)
    data = n.transform(raw, n.fit_normalizer(raw))
    model = n.NumericNetwork(config(kind)).eval()
    # Initial zero residual heads intentionally hide encoders; open them for dependency test.
    with torch.no_grad():
        for head in model.heads:
            head.weight.fill_(.001)
    full = n._predict(model, data, raw['anchor'], 'cpu')
    subset = n._predict(model, {k:v[1:2] for k,v in data.items()}, raw['anchor'][1:2], 'cpu')
    np.testing.assert_allclose(full[1:2], subset, atol=1e-3, rtol=0)
    tensors = {k:torch.tensor(v[:1], requires_grad=True) for k,v in data.items()}
    model(tensors).sum().backward()
    gradient = tensors['ch'].grad[0, :, :180].abs().sum(dim=1)
    assert (gradient > 0).all(), 'every CH time slot must reach forecast'
    assert full.shape == (3, 72) and np.isfinite(full).all()


def test_synthetic_fit_roundtrip_refit_and_reconstruction(tmp_path, monkeypatch):
    cutoff = n.CUTOFF
    index = pd.date_range(cutoff-pd.Timedelta(days=100), cutoff+pd.Timedelta(hours=100), freq='h')
    frames = {'speed': pd.Series(400+10*np.sin(np.arange(len(index))/40), index=index)}
    pool = pd.DatetimeIndex([cutoff-pd.Timedelta(days=90), cutoff-pd.Timedelta(days=80),
                             cutoff-pd.Timedelta(days=20), cutoff-pd.Timedelta(days=10)])
    monkeypatch.setattr(n, '_pool', lambda frames, cutoff, config: pool)
    monkeypatch.setattr(n, '_raw', lambda frames, origins, target: synthetic_raw(origins))
    original_train = n.train_model
    calls = []
    def tiny_train(frames, origins, config, device, epochs, validation=None):
        assert frames['speed'].index.max() <= cutoff
        calls.append((list(origins), epochs, validation is not None))
        return original_train(frames, origins, config, device, min(epochs, 2), validation)
    monkeypatch.setattr(n, 'train_model', tiny_train)
    issued = pd.DatetimeIndex([cutoff+pd.Timedelta(hours=h) for h in (1, 2, 3)])
    destination = tmp_path/'fit'
    pred, meta = n.fit_predict(frames, cutoff, issued, config(), 'cpu', destination)
    assert calls[0][2] and not calls[1][2]
    assert calls[1][0] == list(pool)
    assert calls[1][1] == meta['selected_epochs']
    assert len(meta['history']) == meta['selected_epochs']
    restored = n.predict_saved(frames, issued, destination, 'cpu')
    np.testing.assert_allclose(pred.pred_kms, restored.pred_kms, atol=1e-3, rtol=0)
    assert n.reconstruct_normalizer(frames, destination)['normalizer'] == meta['normalizer']
    poisoned = {'speed': frames['speed'].copy()}
    poisoned['speed'].loc[poisoned['speed'].index > cutoff] = 1e20
    pred2, meta2 = n.fit_predict(poisoned, cutoff, issued, config(), 'cpu', tmp_path/'poison')
    assert meta2['normalizer'] == meta['normalizer']
    assert meta2['model_sha256'] == meta['model_sha256']
    np.testing.assert_array_equal(pred.pred_kms, pred2.pred_kms)
    with pytest.raises(FileExistsError):
        n.fit_predict(frames, cutoff, issued, config(), 'cpu', destination)
    for name in ('metadata.json', 'model.pt'):
        path = destination/name
        old = path.read_bytes()
        path.write_bytes(old+b' ')
        with pytest.raises(ValueError, match='hash mismatch'):
            n.predict_saved(frames, issued, destination, 'cpu')
        path.write_bytes(old)
    saved = json.loads((destination/'metadata.json').read_text())
    saved['normalizer']['ch']['mean'][0] += 1
    (destination/'metadata.json').write_text(json.dumps(saved))
    (destination/'metadata.sha256').write_text(n.digest(destination/'metadata.json'))
    with pytest.raises(ValueError, match='scaler mismatch'):
        n.reconstruct_normalizer(frames, destination)


def test_memory_estimate_has_bound_shapes():
    estimate = n.estimate_memory(config())
    assert estimate['max_input_shapes']['ch'] == [12000, 120, 180]
    assert 4*2**30 < estimate['additional_host_bytes'] <= 16*2**30
    assert estimate['gpu_bytes'] < 20*2**30


def test_actual_shared_inputs_network_causality_and_unknown_schema():
    from experiments.mse20_ch_dynamics import inputs
    index = pd.date_range('2020-01-01', periods=1100, freq='h', tz='UTC')
    x = np.arange(len(index))
    ace = pd.DataFrame({'filled_speed_kms': 430+70*np.sin(x/35), 'was_missing': np.zeros(len(x))}, index=index)
    ch = pd.DataFrame({c: .01+.005*np.sin(x/50+j) for j,c in enumerate(inputs.CH_COLUMNS)}, index=index)
    raw_ace = pd.DataFrame({'ace_density_cm3': 5+np.cos(x/10), 'ace_temperature_k': 1e5+x,
                           'ace_bt_nt': 5+np.sin(x/17), 'ace_speed_kms': ace.filled_speed_kms}, index=index)
    frames = inputs.rebuild_frames(ace, ch, raw_ace)
    origins = index[[1000, 1001, 1002]]
    raw = n._raw(frames, origins, 'mr')
    stats = n.fit_normalizer(n._raw(frames, index[[800, 850, 900]], 'mr'))
    n.setup()
    model = n.NumericNetwork(config()).eval()
    with torch.no_grad():
        for head in model.heads:
            head.weight.fill_(.001)
    before = n._predict(model, n.transform(raw, stats), raw['anchor'], 'cpu')
    for source in ('ace', 'ch', 'raw'):
        for drop in (False, True):
            sources = {k: frames[k].copy() for k in ('ace', 'ch', 'raw')}
            future = sources[source].index > origins[-1]
            assert future.sum() > 72
            if drop:
                sources[source] = sources[source].loc[~future]
            else:
                sources[source].loc[future] = 1e7
            rebuilt = inputs.rebuild_frames(**sources)
            changed = n._raw(rebuilt, origins, 'mr')
            after = n._predict(model, n.transform(changed, stats), changed['anchor'], 'cpu')
            np.testing.assert_allclose(before, after, atol=1e-3, rtol=0)
    for source in ('ch', 'physics'):
        changed = dict(frames)
        changed[source] = frames[source].assign(future_regime=1)
        with pytest.raises(ValueError, match='schema|allowlisted'):
            n._raw(changed, origins, 'mr')


@pytest.mark.parametrize('length',[168,120,72,1])
@pytest.mark.parametrize('factor',[1,6,24])
def test_fixed_block_pool_matches_adaptive_values_and_backward(length,factor):
    torch.manual_seed(42)
    original=torch.randn(2,3,length,dtype=torch.float64,requires_grad=True)
    rewritten=original.detach().clone().requires_grad_(True)
    expected=torch.nn.functional.adaptive_avg_pool1d(original,max(1,length//factor))
    actual=n.block_mean_pool(rewritten,factor)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    weight=torch.randn_like(actual)
    (expected*weight).sum().backward();(actual*weight).sum().backward()
    torch.testing.assert_close(rewritten.grad,original.grad,atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('knots',[6,18])
def test_fixed_interpolation_matches_linear_values_and_backward(knots):
    torch.manual_seed(42)
    original=torch.randn(4,knots,requires_grad=True)
    rewritten=original.detach().clone().requires_grad_(True)
    expected=torch.nn.functional.interpolate(original[:,None],size=72,mode='linear',align_corners=True)[:,0]
    interpolation=n.FixedInterpolation(knots)
    actual=interpolation(rewritten)
    torch.testing.assert_close(actual,expected,atol=5e-6,rtol=5e-6)
    assert not list(interpolation.parameters())
    assert torch.equal(actual[:,0],rewritten[:,0]) and torch.equal(actual[:,-1],rewritten[:,-1])
    weight=torch.randn_like(actual)
    (expected*weight).sum().backward();(actual*weight).sum().backward()
    torch.testing.assert_close(rewritten.grad,original.grad,atol=1e-5,rtol=1e-5)


def test_exact72_head_has_identity_interpolator():
    assert isinstance(n.NumericNetwork(config('tcn')).interpolators[0],torch.nn.Identity)
    assert isinstance(n.NumericNetwork(config('multiscale')).interpolators[-1],torch.nn.Identity)


def synthetic_backward_check(kind,device):
    """Called by CPU regression and explicitly leased CUDA preflight; synthetic only."""
    n.setup()
    model=n.NumericNetwork(config(kind)).to(device)
    tensors={key:torch.randn(2,length,2*channels+(key=='ch'),device=device,requires_grad=True)
             for key,(length,channels) in n.SHAPES.items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    losses=[]
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction=model(tensors)
        loss=(prediction-.123).square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in tensors.values())
        assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in model.parameters())
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    return {'kind':kind,'device':str(device),'passed':True,'losses':losses,
            'deterministic_algorithms':True,'warn_only':False,'optimizer_steps':2}


@pytest.mark.parametrize('kind',['multiscale','tcn','pulse'])
def test_synthetic_deterministic_backward(kind):
    assert synthetic_backward_check(kind,'cpu')['passed']
