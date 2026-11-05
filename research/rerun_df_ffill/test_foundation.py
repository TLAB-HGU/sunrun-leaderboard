import importlib.util
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('rerun_foundation', Path(__file__).with_name('foundation.py'))
f = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = f
spec.loader.exec_module(f)


def fixture():
    return f.Inputs(np.array([[90], [91]], dtype=np.float32),
                    np.arange(144, dtype=np.float32).reshape(2, 72, 1),
                    np.tile(np.arange(24, dtype=np.float32), (2, 1)))


@pytest.mark.parametrize('strategy,width', [('direct',26), ('dirrec',28), ('recursive',27), ('pooled',27)])
def test_exact_design(strategy, width):
    inp = fixture()
    previous = np.tile(np.arange(100, 172), (2,1))
    x = f.features(strategy, inp, previous, np.array([0]), np.array([2]))
    assert x.shape == (1, width)
    assert x[0, 0] == 90
    if strategy == 'recursive':
        np.testing.assert_array_equal(x[0,1:25], np.r_[np.arange(2,24),100,101])
    else:
        np.testing.assert_array_equal(x[0,1:25], np.arange(24))
    if strategy == 'dirrec':
        np.testing.assert_array_equal(x[0,25:27], [100,101])
    if strategy in ('pooled','recursive'):
        assert x[0,-1] == 3


def test_context_chronology_and_observed_before_limit():
    dates = pd.to_datetime(['2020-01-03','2020-01-01','2020-01-02'], utc=True)
    y = np.ones((3,72))
    y[0,-1] = np.nan
    r,h = f.context_cells(dates,y,0,True,3)
    assert list(zip(r,h)) == [(0,68),(0,69),(0,70)]
    y[0,0] = np.nan
    r,h = f.context_cells(dates,y,0,False,2)
    np.testing.assert_array_equal(r,[1,2])


def test_teacher_ffill_never_changes_loss_target():
    inp = fixture()
    y = np.full((2,72),np.nan)
    y[:,1] = 100
    previous = f.teacher_inputs(inp,y)
    np.testing.assert_array_equal(previous[:,0], [23,23])
    assert np.all(previous[:,1:] == 100)
    assert np.isnan(y[:,0]).all()


def test_prior_only_selection_purge():
    dates = pd.date_range('2023-01-01',periods=5000,freq='h',tz='UTC')
    tr,va = f.selection_split(dates)
    assert dates[tr].max() + pd.Timedelta(hours=144) < dates[va].min()
    assert dates[va].max() <= dates.max()


def test_query_uncomputed_future_does_not_enter_features():
    inp = fixture()
    p = np.ones((2,72))
    bad = p.copy(); bad[:,3:] = 999999
    for strategy in ['direct','recursive','pooled','dirrec']:
        a=f.features(strategy,inp,p,np.array([0,1]),np.array([3,3]))
        b=f.features(strategy,inp,bad,np.array([0,1]),np.array([3,3]))
        np.testing.assert_array_equal(a,b)


def test_smoke_cannot_merge(tmp_path):
    np.savez(tmp_path/'F1_predictions.npz', pred=np.ones((2,72)),origins=np.array([1,2]))
    with pytest.raises(ValueError,match='incomplete'):
        f.merge(tmp_path)


def test_cached_checkpoint_resolver_never_downloads(monkeypatch):
    import huggingface_hub
    calls=[]
    def resolve(*a,**kw):
        calls.append((a,kw)); return '/cached/model.pt'
    monkeypatch.setattr(huggingface_hub,'hf_hub_download',resolve)
    f.checkpoint('kumo'); f.checkpoint('tabiclv2')
    assert all(kw['local_files_only'] is True for _,kw in calls)
    assert calls[0][0][1] == 'large/regressor.pt'
    assert calls[1][1]['revision'] == '4dcd344ece2c00be9e831fdd35bed57b5ad83e19'


@pytest.mark.parametrize('family',['kumo','tabiclv2','realmlp'])
def test_tiny_actual_cpu_fit(family):
    x=np.random.default_rng(42).normal(size=(16,3)).astype(np.float32)
    y=(300+x[:,0]*10).astype(np.float32)
    model=f.Learner(family,threads=1,smoke=True).fit(x,y)
    pred=model.predict(x[:2])
    assert pred.shape == (2,)
    assert np.isfinite(pred).all()


@pytest.mark.parametrize('device',['cuda:1','cuda','mps'])
def test_unapproved_device_rejected(device):
    with pytest.raises(ValueError,match='cpu or cuda:0'):
        f.Learner('realmlp',device=device)


@pytest.mark.skipif(__import__('os').environ.get('RUN_CAMPAIGN_CUDA_TESTS')!='1',reason='requires an externally held GPU0 lease')
@pytest.mark.parametrize('family',['kumo','tabiclv2','realmlp'])
def test_original_cuda_tiny_fit(family):
    import torch
    assert torch.cuda.device_count()==1  # launcher must isolate physical GPU0
    x=np.random.default_rng(42).normal(size=(16,3)).astype(np.float32)
    y=(300+10*x[:,0]).astype(np.float32)
    # Kumo preflights and loads the full large checkpoint, but uses one tiny estimator.
    model=f.Learner(family,threads=1,smoke=family!='kumo',device='cuda:0')
    if family=='kumo':
        assert model.identity['filename']=='large/regressor.pt'
        model.smoke=True
    model.fit(x,y)
    pred=model.predict(x[:2])
    assert pred.shape==(2,) and np.isfinite(pred).all()
    if family!='realmlp':
        with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):
            expected=np.asarray(model.model.predict(model.table(x[:2])).numerical.float().cpu().numpy(),dtype=np.float64).mean(axis=1)
        np.testing.assert_array_equal(pred,expected)
    del model
    torch.cuda.empty_cache()
