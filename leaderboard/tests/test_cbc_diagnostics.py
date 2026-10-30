import importlib.util
from pathlib import Path
import numpy as np
s=importlib.util.spec_from_file_location('cbcdiag',Path(__file__).parents[1]/'experiments/d153_shifts/cbc/diagnostics.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_shift_recovers_known_displacement():
    p=np.r_[np.zeros(25),np.ones(47)*100][None,:]+400
    y=p[:,np.clip(np.arange(72)-6,0,71)]
    r,s=m.lag_bounds(y,p,np.array([400]))
    assert r['shift6'][0]==0 and s[6][0]==6 and r['base'][0]>0

def test_bias_oracle_is_frequency_weighted():
    y=np.full((10,72),410.);p=np.full((10,72),400.)
    r=m.bias_deciles(y,p,np.arange(10)/10,np.zeros(10,bool))
    assert abs(r['max_oracle_bias_gain']-100)<1e-9

def test_metrics_handle_single_class_without_fake_auc():
    r=m.metrics(np.zeros(10),np.full(10,.1))
    assert r['auc'] is None and r['f1']==0 and np.isfinite(r['sedi'])

def test_derivative_lag_known_displacement():
    p=np.r_[np.ones(25)*400,np.ones(47)*600][None,:]
    y=p[:,np.clip(np.arange(72)-6,0,71)]
    lag,corr=m.derivative_lag(y,p)
    assert lag[0]==6 and abs(corr[0]-1)<1e-9

def test_pooled_threshold_metrics_preserve_window_thresholds():
    y=np.array([0,1,0,1]);p=np.array([.2,.4,.6,.8]);t=np.array([.3,.3,.7,.7])
    assert m.metrics(y,p,t)['f1']==1
