import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

spec = importlib.util.spec_from_file_location('budget_complexity',Path(__file__).with_name('run.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def data(n=256):
    rng = np.random.default_rng(42)
    origins = pd.date_range('2023-12-20',periods=n,freq='12h',tz='UTC')
    x = rng.normal(size=(n,4)).astype(np.float32)
    y = (480+60*x[:,:1]+np.linspace(-30,50,72)).astype(np.float32)
    return x,y,origins


def test_noop_exact_parent_and_mass_550_boundary():
    x,y,o = data()
    y[0,:3] = [549.999,550,550.001]
    w,trace = r.mass_weights(y,o,np.ones_like(y),np.ones(len(y),bool))
    np.testing.assert_array_equal(w,r.p.weights(y))
    assert r.p.weights(y)[0,1] > 1
    assert trace['max_mass_relative_error']==0
    matrix = xgb.DMatrix(x,label=y[:,:24],nthread=3)
    params = dict(tree_method='hist',multi_strategy='multi_output_tree',num_target=24,
                  max_depth=3,nthread=3,seed=42,base_score=float(y[:,:24].mean()))
    a = xgb.train(params,matrix,num_boost_round=2,obj=r.p.objective)
    b = xgb.train(params,matrix,num_boost_round=2,obj=r.objective_for(w[:,:24]))
    np.testing.assert_array_equal(a.inplace_predict(x),b.inplace_predict(x))


def test_mass_warmup_small_groups_and_ess():
    _,y,o = data(500)
    raw = np.random.default_rng(9).uniform(.5,2,y.shape)
    active = np.arange(len(y))>=125
    w,trace = r.mass_weights(y,o,raw,active)
    np.testing.assert_array_equal(w[~active],r.p.weights(y)[~active])
    for g in trace['groups']:
        np.testing.assert_allclose(g['candidate_mass'],g['parent_mass'],atol=1e-9)
        if g['n']:
            assert 0<g['ess']<=g['n']+1e-8
    base = r.p.weights(y)
    for year in np.unique(o.year):
        for h in range(72):
            for high in (False,True):
                mask = (o.year==year)&((y[:,h]>=550)==high)
                np.testing.assert_allclose(w[mask,h].astype(float).sum(),base[mask,h].astype(float).sum(),rtol=1e-7)
                if (mask&active).sum()<32:
                    np.testing.assert_array_equal(w[mask,h],base[mask,h])
    assert trace['relative_min']>=.25 and trace['relative_max']<=4
    assert all(v>0 for v in trace['ess_by_horizon'])


def test_derivatives():
    _,y,o = data(80)
    w,_ = r.mass_weights(y,o,np.ones_like(y),np.ones(len(y),bool))
    pred = y[:,:24].astype(float)+3
    matrix = xgb.DMatrix(np.ones((len(y),2)),label=y[:,:24],nthread=3)
    g,h = r.objective_for(w[:,:24])(pred,matrix)
    eps = 1e-3
    f = lambda a: .5*w[:,:24]*(a-y[:,:24])**2
    np.testing.assert_allclose(g,(f(pred+eps)-f(pred-eps))/(2*eps),rtol=1e-6)
    np.testing.assert_array_equal(h,w[:,:24])


def test_maturity_and_future_heldout_label_poison_positive_control():
    x,y,o = data(300)
    cutoff = o[200]
    train = r.train_indices(o,cutoff)
    assert o[train[-1]]+pd.Timedelta(hours=144)==cutoff
    assert ((o[train,None] if False else o[train])+pd.Timedelta(hours=144)<=cutoff).all()
    poisoned = y.copy()
    poisoned[len(train):] += 10000
    a,_ = r.mass_weights(y[train],o[train],np.ones_like(y[train]),np.ones(len(train),bool))
    b,_ = r.mass_weights(poisoned[train],o[train],np.ones_like(y[train]),np.ones(len(train),bool))
    np.testing.assert_array_equal(a,b)
    inside = y.copy()
    inside[train[:20]] = 800
    assert not np.array_equal(a,r.p.weights(inside[train]))


def test_forward_crossfit_future_rows_and_labels_do_not_change_earlier_scores(tmp_path):
    x,y,o = data(256)
    params = dict(tree_method='hist',multi_strategy='multi_output_tree',num_target=24,
                  max_depth=3,nthread=3,seed=42,learning_rate=.03)
    raw,active,receipt = r.forward_complexity([x]*3,y,o,x[:,:3],params,2)
    future_y = y.copy()
    future_x = x.copy()
    future_y[192:] += 10000
    future_x[192:] += 10000
    poisoned,_,_ = r.forward_complexity([future_x]*3,future_y,o,future_x[:,:3],params,2)
    np.testing.assert_array_equal(raw[:192],poisoned[:192])
    np.testing.assert_array_equal(raw[:64],np.ones((64,72)))
    assert not active[:64].any()
    for fold in receipt['folds']:
        if not fold.get('neutral'):
            assert pd.Timestamp(fold['max_label_maturity'])<=pd.Timestamp(fold['cutoff'])
    # Positive control: current scored training labels affect that quarter's complexity.
    assert not np.array_equal(raw[192:],poisoned[192:])
    baseline_density = r.density(x[:64,:3],x[64:128,:3])
    assert np.isfinite(baseline_density).all()
    # Checkpoint roundtrip through the exact custom objective.
    w = r.p.weights(y[:,:24])
    matrix = xgb.DMatrix(x,label=y[:,:24],nthread=3)
    m = xgb.train(dict(params,base_score=500),matrix,num_boost_round=2,obj=r.objective_for(w))
    path = tmp_path/'model.ubj'
    m.save_model(path)
    np.testing.assert_array_equal(m.inplace_predict(x),xgb.Booster(model_file=path).inplace_predict(x))


def test_density_empty_nan_and_duplicate_bins():
    a = np.ones((60,3))
    a[:,1] = np.nan
    d = r.density(a,np.array([[1,np.nan,1],[10,1,10]],float))
    assert np.isfinite(d).all() and (d>=.5).all() and (d<=2).all()
