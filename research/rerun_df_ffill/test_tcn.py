import ast
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

spec=importlib.util.spec_from_file_location('rerun_tcn',Path(__file__).with_name('tcn.py'))
t=importlib.util.module_from_spec(spec);sys.modules[spec.name]=t;spec.loader.exec_module(t)
torch.set_num_threads(1)


@pytest.mark.parametrize('regskip',[False,True])
def test_exact_architecture_and_forward_against_archive(regskip):
    suffix='_regskip' if regskip else ''
    src=(Path(__file__).parent/'source_archive'/f'e2000fs_feat30_pointweight4_tcn_lb120{suffix}.py').read_text()
    tree=ast.parse(src)
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ['CausalBlock','TCN']]
    namespace=dict(torch=torch,nn=nn,KERNEL=3,DILATIONS=t.DILATIONS,WIDTH=32 if regskip else 64,DROPOUT=.2)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'archived-model','exec'),namespace)
    original=namespace['TCN'](t.CURRENT) if regskip else namespace['TCN']()
    actual=t.TCN(regskip)
    actual.load_state_dict(original.state_dict(),strict=True)
    original.eval();actual.eval()
    history=torch.randn(2,120,30);future=torch.randn(2,72,9)
    torch.testing.assert_close(actual(history,future),original(history,future),rtol=0,atol=0)


def test_causal_block_never_reads_future():
    model=t.CausalBlock(2,4,2).eval()
    x=torch.randn(1,2,15);alter=x.clone();alter[:,:,8:]+=1000
    torch.testing.assert_close(model(x)[:,:,:8],model(alter)[:,:,:8])


def test_missing_loss_mask_and_original_all_observed_formula():
    p=torch.tensor([[2.,100.] ],requires_grad=True)
    y=torch.tensor([[1.,0.] ]);w=torch.tensor([[4.,0.] ])
    loss=t.weighted_mse(p,y,w);assert loss.item()==4.
    loss.backward();assert p.grad[0,1]==0
    w=torch.tensor([[2.,3.] ])
    assert t.weighted_mse(p,y,w)==(w*(p-y).square()).mean()


def test_history_window_boundary_and_leading_missing():
    h=np.ones((130,30));h[0,0]=np.nan
    positions=np.array([-1,118,119,120,129,130])
    np.testing.assert_array_equal(t.complete_windows(h,positions),[False,False,False,True,True,False])


def test_training_stats_and_observed_weight_normalization():
    history=np.ones((125,30));future=np.ones((2,72,9));y=np.full((2,72),300.)
    y[0,0]=np.nan;y[1,0]=600.
    stats=t.normalization(history,np.array([119,120]),future,y)
    assert stats['weight_mean'][0]==4.
    history[-1]=99999
    after=t.normalization(history,np.array([119,120]),future,y)
    np.testing.assert_array_equal(stats['history']['mean'],after['history']['mean'])
    batch=t.Windows(history,np.array([119,120]),future,y,stats)[0]
    assert batch[3][0]==0 and np.isfinite(batch[2]).all()


@pytest.mark.parametrize('regskip',[False,True])
def test_tiny_train_and_saved_loss(tmp_path,regskip):
    rng=np.random.default_rng(42)
    history=rng.normal(size=(125,30)).astype(np.float32)
    future=rng.normal(size=(2,72,9)).astype(np.float32)
    y=np.full((2,72),300.,np.float32);y[0,2]=np.nan
    positions=np.array([119,120]);stats=t.normalization(history,positions,future,y)
    windows=t.Windows(history,positions,future,y,stats)
    net,epoch=t.train(windows,windows,2,regskip,tmp_path/'loss.csv')
    assert epoch in [1,2] and len(pd.read_csv(tmp_path/'loss.csv'))==2
    assert np.isfinite(next(net.parameters()).detach().numpy()).all()


def test_hourly_history_uses_h1_and_causal_fill(tmp_path):
    dates=pd.date_range('2024-01-01',periods=3,freq='h',tz='UTC')
    frame=pd.DataFrame({'origin':np.repeat(dates,2),'h':np.tile([1,2],3)})
    for n in t.FEATURES:
        frame[t.mapped(n)]=[1.,999.,np.nan,999.,3.,999.]
    frame.to_parquet(tmp_path/'data.parquet')
    index,history=t.hourly_history(tmp_path/'data.parquet')
    np.testing.assert_array_equal(history[:,0],[1.,1.,3.])
    assert index.equals(dates)


@pytest.mark.skipif(__import__('os').environ.get('RUN_CAMPAIGN_CUDA_TESTS')!='1',reason='requires an externally held GPU0 lease')
@pytest.mark.parametrize('regskip',[False,True])
def test_original_cuda_fp32_step(tmp_path,regskip):
    assert torch.cuda.device_count()==1
    rng=np.random.default_rng(42)
    history=rng.normal(size=(125,30)).astype(np.float32)
    future=rng.normal(size=(2,72,9)).astype(np.float32)
    y=np.full((2,72),300.,np.float32)
    positions=np.array([119,120]);stats=t.normalization(history,positions,future,y)
    windows=t.Windows(history,positions,future,y,stats)
    model,epoch=t.train(windows,windows,1,regskip,tmp_path/'cuda_loss.csv',device='cuda:0')
    assert next(model.parameters()).device==torch.device('cuda:0')
    assert next(model.parameters()).dtype==torch.float32
    model.eval()
    with torch.inference_mode():
        gpu=model(torch.from_numpy(history[:120][None]).cuda(),torch.from_numpy(future[:1]).cuda()).cpu()
        cpu=model.cpu()(torch.from_numpy(history[:120][None]),torch.from_numpy(future[:1]))
    torch.testing.assert_close(gpu,cpu,rtol=1e-4,atol=1e-5)
    assert torch.isfinite(gpu).all() and epoch==1
