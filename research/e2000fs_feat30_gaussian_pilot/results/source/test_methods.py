import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('runner_under_test', HERE/'run.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
m = r.m


def test_decoder_physical_loss_equivalence_and_missing():
    names = {'constant':['v_lag0'],'varying':['rec_mean']}
    bundle = {'C':np.array([[400.],[600.]]),
              'E':np.array([[[420.],[1000.],[np.nan]],[[500.],[600.],[200.]]])}
    a,b,missing = m.decoder(bundle,names)
    np.testing.assert_array_equal(b,[[50,300,100],[-100,50,-300]])
    assert missing == 1
    y = np.array([[500,800,600],[400,700,300]])
    g = np.array([[.1,.3,.8],[-.2,.5,.9]])
    np.testing.assert_allclose(b*b*(g-(y-a[:,None])/b)**2,(a[:,None]+b*g-y)**2)
    bundle['C'][0,0] = np.nan
    with pytest.raises(ValueError,match='anchor'):
        m.decoder(bundle,names)


def test_gaussian_floor_normalization_and_equal_losses():
    y = np.r_[np.zeros(99),100.]
    w,info = m.gaussian_weights(y,np.zeros(100))
    assert np.isfinite(w).all() and np.isclose(w.mean(),1)
    assert info['floor_count'] == 1 and w[-1] < w[0]
    np.testing.assert_array_equal(m.gaussian_weights(np.ones(5),np.zeros(5))[0],np.ones(5))
    with pytest.raises(ValueError):
        m.gaussian_weights([np.nan],[0])


def test_causal_decoder_future_row_poison_with_positive_control():
    names = {'constant':['v_lag0'],'varying':['rec_mean']}
    bundle = {'C':np.array([[400.],[500.]]),'E':np.array([[[480.]],[[650.]]])}
    a,b,_ = m.decoder(bundle,names)
    a = a.copy()
    bundle['C'][1,0] = 1000
    bundle['E'][1,0,0] = 50
    poisoned_a,poisoned_b,_ = m.decoder(bundle,names)
    assert poisoned_a[0] == a[0] and poisoned_b[0,0] == b[0,0]
    assert poisoned_a[1] != a[1] and poisoned_b[1,0] != b[1,0]


def test_calibration_split_strict_label_maturity():
    oo = pd.date_range('2020-01-01',periods=1000,freq='3h',tz='UTC')
    train = m.mature_indices(oo,oo[-1])
    assert (oo[train]+pd.Timedelta(hours=72)).max() < oo[-1]
    prefix,cal = m.calibration_split(oo,train)
    assert len(cal) == len(train)-int(.8*len(train))
    assert (oo[prefix]+pd.Timedelta(hours=72)).max() < oo[cal].min()
    assert not np.intersect1d(prefix,cal).size


def test_gate_rejects_quiet_regression_even_if_total_and_extremes_improve():
    base = dict(mse=100.,mse_lt550=50.,mse_ge550=300.)
    better = dict(mse=98.,mse_lt550=50.,mse_ge550=280.)
    baseline = {k:base for k in ['dev0','dev1','dev2','pooled']}
    candidate = {k:better.copy() for k in baseline}
    assert m.development_gate(candidate,baseline)
    candidate['dev1']['mse_lt550'] = 50.01
    assert not m.development_gate(candidate,baseline)
    assert not m.gate(dict(better,mse=np.nan),base,True)


def test_checkpoint_parity_and_tamper(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(256,3)); y = x[:,0]*10
    params = dict(n_estimators=3,n_jobs=2,verbosity=-1,min_child_samples=10)
    ctx = dict(stage='test')
    model = r.fit_checkpoint(tmp_path,ctx,x,y,params)
    restored = r.fit_checkpoint(tmp_path,ctx,x,y,params)
    np.testing.assert_array_equal(model.predict(x,num_threads=2),restored.predict(x,num_threads=2))
    with pytest.raises(ValueError,match='identity'):
        r.fit_checkpoint(tmp_path,dict(stage='wrong'),x,y,params)
    with (tmp_path/'loss.csv').open('a') as f:
        f.write('tamper')
    with pytest.raises(ValueError,match='Frozen file'):
        r.fit_checkpoint(tmp_path,ctx,x,y,params)
