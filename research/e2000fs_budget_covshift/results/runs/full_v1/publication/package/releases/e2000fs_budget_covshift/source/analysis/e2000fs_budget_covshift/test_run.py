import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

spec=importlib.util.spec_from_file_location('covshift',Path(__file__).with_name('run.py'))
r=importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def test_maturity_inclusive_and_all_horizons():
    cutoff=pd.Timestamp('2026-02-26',tz='UTC')
    oo=pd.date_range(cutoff-pd.Timedelta(hours=145),periods=4,freq='h')
    np.testing.assert_array_equal(r.mature_rows(oo,cutoff),[0,1])
    assert (oo[r.mature_rows(oo,cutoff)]+pd.Timedelta(hours=144)).max()==cutoff


def test_noop_and_speed_band_mass():
    rng=np.random.default_rng(42)
    y=rng.uniform(300,800,(200,72)).astype(np.float32)
    y[0,:4]=[399.9,400,550,700]
    oo=pd.date_range('2024-12-28',periods=200,freq='h',tz='UTC')
    w,_=r.combined_weights(y,oo,np.ones(len(y)))
    np.testing.assert_array_equal(w,r.parent.weights(y))
    w,mass=r.combined_weights(y,oo,np.linspace(.25,4,len(y)))
    np.testing.assert_allclose(w.mean(axis=0),1,rtol=1e-6)
    assert len(mass)>0
    bands=np.digitize(y,[400,550,700])
    base=r.parent.weights(y)
    for era in np.unique(oo.year):
        for h in range(72):
            for b in range(4):
                mask=(oo.year==era)&(bands[:,h]==b)
                np.testing.assert_allclose(w[mask,h].sum(),base[mask,h].sum(),rtol=2e-6)


def test_gradient_and_parent_noop_training_reload(tmp_path):
    rng=np.random.default_rng(42)
    x=rng.normal(size=(96,8)).astype(np.float32)
    y=rng.uniform(300,800,(96,24)).astype(np.float32)
    y[0,0]=550
    w=r.parent.weights(y)
    m=xgb.DMatrix(x,label=y,nthread=3)
    pred=y.astype(float)+1.234
    g,h=r.weighted_objective(w)(pred,m)
    eps=1e-3
    numerical=(.5*w*(pred+eps-y)**2-.5*w*(pred-eps-y)**2)/(2*eps)
    np.testing.assert_allclose(g,numerical,rtol=1e-7)
    np.testing.assert_array_equal(h,w)
    settings=dict(tree_method='hist',multi_strategy='multi_output_tree',num_target=24,nthread=3,
                  max_depth=3,eta=.03,seed=42,base_score=float(y.mean()),subsample=.8,colsample_bytree=.7)
    p=xgb.train(settings,m,num_boost_round=3,obj=r.parent.objective)
    c=xgb.train(settings,m,num_boost_round=3,obj=r.weighted_objective(w))
    np.testing.assert_array_equal(p.inplace_predict(x),c.inplace_predict(x))
    path=tmp_path/'model.ubj'
    c.save_model(path)
    np.testing.assert_array_equal(c.inplace_predict(x),xgb.Booster(model_file=path).inplace_predict(x))


def test_cutoff_poison_and_positive_control():
    cutoff=pd.Timestamp('2026-01-01',tz='UTC')
    oo=pd.date_range('2025-01-01',periods=3120,freq='3h',tz='UTC')
    rng=np.random.default_rng(42)
    x=rng.normal(size=(len(oo),4)).astype(np.float32)
    x[:,0]+=np.arange(len(oo))/100
    idx,a,diag,_=r.density_ratio(x,oo,cutoff,rounds=8)
    future=x.copy()
    future[oo>=cutoff]=1e9
    _,b,_,_=r.density_ratio(future,oo,cutoff,rounds=8)
    np.testing.assert_array_equal(a,b)
    past=x.copy()
    past[oo<cutoff,0]=0
    past[oo<cutoff,1]=0
    past[oo<cutoff,2]=0
    past[oo<cutoff,3]=0
    _,changed,_,_=r.density_ratio(past,oo,cutoff,rounds=8)
    assert np.max(np.abs(a-changed))>1e-5
    assert pd.Timestamp(diag['max_context'])<cutoff
    # Future labels cannot enter weights; positive-control poison mature targets.
    y=rng.uniform(300,800,(len(oo),72)).astype(np.float32)
    train=r.mature_rows(oo,cutoff)
    ratio=a[np.searchsorted(idx,train)]
    before,_=r.combined_weights(y[train],oo[train],ratio)
    y[oo+pd.Timedelta(hours=144)>cutoff]=1e9
    after,_=r.combined_weights(y[train],oo[train],ratio)
    np.testing.assert_array_equal(before,after)
    y[train[:10]]=551
    changed,_=r.combined_weights(y[train],oo[train],ratio)
    assert not np.array_equal(before,changed)
