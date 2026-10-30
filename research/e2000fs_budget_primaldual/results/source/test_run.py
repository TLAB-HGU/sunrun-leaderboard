import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

spec=importlib.util.spec_from_file_location('budget_primaldual',Path(__file__).with_name('run.py'))
r=importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def test_derivatives_and_boundary():
    y=np.array([[549.99,550.],[650.,400.]],dtype=np.float32)
    obj=r.RiskObjective(y,1000.)
    assert not obj.high[0,0] and obj.high[0,1]
    p=y.astype(float)+np.array([[2.,-3.],[4.,-5.]])
    matrix=xgb.DMatrix(np.ones((2,3)),label=y,nthread=3)
    g,h=obj(p,matrix)
    eps=1e-3
    loss=lambda z:.5*obj.weights()*(z-y)**2
    np.testing.assert_allclose(g,(loss(p+eps)-loss(p-eps))/(2*eps),rtol=1e-6)
    np.testing.assert_allclose(h,(loss(p+eps)-2*loss(p)+loss(p-eps))/eps**2,rtol=1e-6)
    assert (h>0).all()
    np.testing.assert_allclose(h.mean(),1.)
    assert np.isclose(h[0,1]/h[0,0],4.)


def test_dual_projection_and_budget_scale():
    obj=r.RiskObjective(np.array([[400.,600.]]),100.)
    assert obj.update(200.)==1.52
    for _ in range(1000): obj.update(1e9)
    assert obj.dual==10.
    for _ in range(1000): obj.update(0.)
    assert obj.dual==0.
    empty=r.RiskObjective(np.array([[400.,500.]]),100.)
    assert empty.update(1e9)==0.
    np.testing.assert_array_equal(empty.weights(),np.ones((1,2)))


def test_maturity_calibration_and_future_poison_positive_control():
    oo=pd.date_range('2020-01-01',periods=5000,freq='h',tz='UTC')
    cutoff=oo[4000]
    fit=r.train_indices(oo,cutoff)
    assert oo[fit[-1]]+pd.Timedelta(hours=144)==cutoff
    assert np.all(oo[fit]+pd.Timedelta(hours=72)+pd.Timedelta(hours=72)<=cutoff)
    ref,cal=r.calibration_indices(oo,cutoff)
    start=cutoff-pd.Timedelta(days=90)
    assert np.all(oo[ref]+pd.Timedelta(hours=144)<=start)
    assert np.all(oo[cal]>=start)
    assert not np.intersect1d(ref,cal).size
    rng=np.random.default_rng(42)
    x=rng.normal(size=(5000,4)); y=rng.uniform(400,650,size=(5000,24))
    poisoned_y=y.copy(); poisoned_x=x.copy()
    # All ineligible labels/query features may be poisoned without affecting objective/calibration inputs.
    future=np.setdiff1d(np.arange(len(oo)),fit)
    poisoned_y[future]=1e9; poisoned_x[future]=-1e9
    np.testing.assert_array_equal(y[fit],poisoned_y[fit])
    np.testing.assert_array_equal(x[ref],poisoned_x[ref])
    np.testing.assert_array_equal(y[cal],poisoned_y[cal])
    a=r.RiskObjective(y[fit],100.); b=r.RiskObjective(poisoned_y[fit],100.)
    np.testing.assert_array_equal(a.weights(),b.weights())
    assert a.update(150.)==b.update(150.)
    assert not np.array_equal(r.RiskObjective(y,100.).weights(),r.RiskObjective(poisoned_y,100.).weights())
    assert not np.array_equal(x,poisoned_x)


def test_parent_noop_vector_reload_and_callback_order(tmp_path):
    rng=np.random.default_rng(42)
    x=rng.normal(size=(64,4)).astype(np.float32)
    y=(np.tile(np.linspace(400,650,24),(64,1))+x[:,:1]*50).astype(np.float32)
    matrix=xgb.DMatrix(x,label=y,nthread=3)
    params=dict(tree_method='hist',multi_strategy='multi_output_tree',num_target=24,
                max_depth=3,nthread=3,seed=42,base_score=float(y.mean()))
    direct=xgb.train(params,matrix,num_boost_round=2,obj=r.parent.objective)
    got,_=r.fit_model(tmp_path_setup(tmp_path),'control',x,y,x,y,params,2)
    np.testing.assert_array_equal(got,np.clip(direct.inplace_predict(x),1.,3000.))
    obj=r.RiskObjective(y,100.)
    r.fit_model(tmp_path,'candidate',x,y,x,y,params,2,objective=obj)
    trace=pd.read_csv(tmp_path/'losses/candidate.csv')
    assert trace.dual_used.iloc[0]==3*obj.p
    np.testing.assert_allclose(trace.dual_used.iloc[1],trace.dual_next.iloc[0])
    expected=np.clip(trace.dual_used+.02*(trace.train_high_mse/100.-1.),0,10)
    np.testing.assert_allclose(trace.dual_next,expected)
    doc=tmp_path/'vector.json'
    direct.save_model(doc)
    trees=r.read(doc)['learner']['gradient_booster']['model']['trees']
    assert all(int(t['tree_param']['size_leaf_vector'])==24 for t in trees)


def tmp_path_setup(path):
    (path/'models').mkdir(); (path/'losses').mkdir()
    return path
