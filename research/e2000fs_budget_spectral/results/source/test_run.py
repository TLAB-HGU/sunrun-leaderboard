import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

spec=importlib.util.spec_from_file_location('spectral',Path(__file__).with_name('run.py'))
r=importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def test_gradient_and_psd_majorizer():
    rng=np.random.default_rng(42)
    y=rng.uniform(350,650,(2,24))
    p=y+rng.normal(0,75,y.shape)
    def loss(p):
        return .5*np.sum(r.weights(y)*(p-y)**2)+.1*r.spectral_loss(p-y).sum()
    g,h=r.derivatives(p,y)
    eps=.01
    for i in range(24):
        delta=np.zeros_like(p); delta[0,i]=eps
        numerical=(loss(p+delta)-loss(p-delta))/(2*eps)
        np.testing.assert_allclose(g[0,i],numerical,rtol=2e-5,atol=2e-4)
    # Differentiate the exact double-precision spectral gradient to expose cross terms.
    def grad(e):
        z=np.fft.fft(e,norm='ortho')
        return .1*np.fft.ifft(z/np.sqrt(1+abs(z)**2/r.SCALE**2),norm='ortho').real
    e=(p-y)[0]
    actual=np.column_stack([(grad(e+np.eye(24)[j]*eps)-grad(e-np.eye(24)[j]*eps))/(2*eps) for j in range(24)])
    np.testing.assert_allclose(actual,actual.T,atol=1e-8)
    eig=np.linalg.eigvalsh(actual)
    gap=np.linalg.eigvalsh(.1*np.eye(24)-actual)
    assert eig.min()>-1e-7 and gap.min()>-1e-7
    assert np.max(abs(actual-np.diag(np.diag(actual))))>1e-4
    r.write(r.LANE/'hessian-verification.json',dict(actual_hessian=actual.tolist(),
        actual_eigenvalues=eig.tolist(),majorizer_gap_eigenvalues=gap.tolist(),
        frobenius_majorizer_gap=float(np.linalg.norm(.1*np.eye(24)-actual)),
        off_diagonal_max=float(np.max(abs(actual-np.diag(np.diag(actual))))),
        finite_difference_step=eps))


def test_parseval_and_non_noop_phase_coupling():
    rng=np.random.default_rng(1)
    e=rng.normal(size=(3,24))*40
    np.testing.assert_allclose(np.sum(abs(np.fft.fft(e,axis=1,norm='ortho'))**2,axis=1),np.sum(e**2,axis=1))
    y=np.full((1,24),400.)
    p=y+np.linspace(0,100,24)
    g=r.derivatives(p,y)[0]
    q=p.copy(); q[0,3]+=10
    # A residual change at h4 changes the spectral gradient at other horizons.
    assert abs(r.derivatives(q,y)[0][0,8]-g[0,8])>1e-4
    signal=np.sin(np.arange(24)*2*np.pi/24)[None,:]*50
    np.testing.assert_allclose(abs(np.fft.fft(signal)),abs(np.fft.fft(-signal)))
    assert r.spectral_loss(signal-signal)[0]==0
    assert r.spectral_loss(-signal-signal)[0]>0
    assert not np.isclose(r.spectral_loss(e)[0],.5*np.sum(e[0]**2))
    with pytest.raises(ValueError): r.spectral_loss(e,scale=0)


def test_parent_zero_coefficient_and_reload(tmp_path):
    rng=np.random.default_rng(42)
    x=rng.normal(size=(64,4)).astype(np.float32)
    y=(np.tile(np.linspace(400,650,24),(64,1))+x[:,:1]*50).astype(np.float32)
    dt=xgb.DMatrix(x,label=y,nthread=3)
    p=y+2
    a=r.derivatives(p,y,0); b=r.parent.objective(p,dt)
    for aa,bb in zip(a,b): np.testing.assert_array_equal(aa,bb)
    settings=dict(tree_method='hist',multi_strategy='multi_output_tree',num_target=24,max_depth=3,nthread=3,seed=42,base_score=float(y.mean()))
    model=xgb.train(settings,dt,num_boost_round=2,obj=r.parent.objective)
    zero=xgb.train(settings,dt,num_boost_round=2,obj=lambda p,d:r.derivatives(p,d.get_label().reshape(p.shape),0))
    np.testing.assert_array_equal(model.inplace_predict(x),zero.inplace_predict(x))
    spectral=xgb.train(settings,dt,num_boost_round=2,obj=r.objective)
    path=tmp_path/'spectral.ubj'; spectral.save_model(path)
    np.testing.assert_array_equal(spectral.inplace_predict(x),xgb.Booster(model_file=path).inplace_predict(x))
    assert not np.array_equal(spectral.inplace_predict(x),model.inplace_predict(x))


def test_maturity_future_poison_and_550_boundary():
    origins=pd.date_range('2025-01-01',periods=300,freq='h',tz='UTC')
    cutoff=origins[200]
    ix=r.mature_indices(origins,cutoff)
    assert ix[-1]==56 and np.all(origins[ix]+pd.Timedelta(hours=144)<=cutoff)
    rng=np.random.default_rng(42)
    y=rng.uniform(300,650,(300,24)).astype(np.float32)
    p=y[ix]+10
    before=r.derivatives(p,y[ix])
    poisoned=y.copy(); poisoned[57:]=1e6
    for a,b in zip(before,r.derivatives(p,poisoned[ix])): np.testing.assert_array_equal(a,b)
    assert not np.array_equal(r.weights(y),r.weights(poisoned)) # positive control
    w=r.weights(np.array([[549.999,550.],[550.,549.999]]))
    assert w[1,0]/w[0,0]==4
    # Design consumes each row independently, future query poisoning cannot affect train.
    names={'constant':['c'],'varying':['v']}
    bundle={'C':np.arange(300)[:,None], 'E':np.ones((300,72,1))}
    old=r.imported('spectral_base_test',r.ROOT/'analysis/e2000fs_dirmo_24_feat30_lgbm/run.py').design
    original=old(bundle,names,['c','v'],0)
    bundle['C'][57:]=99999; bundle['E'][57:]=99999
    changed=old(bundle,names,['c','v'],0)
    np.testing.assert_array_equal(original[ix],changed[ix])
    assert not np.array_equal(original[57:],changed[57:])
