import importlib.util
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parents[1]/'experiments/d153_shifts/arrival_e2/breakeven.py'
spec=importlib.util.spec_from_file_location('breakeven',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_positive_error_delays_not_advances():
    a=np.array([[0.,0.,1.,0.,0.]])
    assert np.argmax(m.shift(a,1))==3
    assert np.argmax(m.shift(a,-1))==1
    assert np.array_equal(m.shift(a,0),a)

def test_zero_timing_removes_demeaned_residual_only():
    y=np.array([[1.,2.,3.,4.],[4.,5.,6.,7.]])
    p=np.zeros_like(y);mask=np.array([True,False])
    g=m.expected_gain(y,p,mask,np.array([0.,10.]),0,1,1)
    assert np.allclose(g,[np.var(y[0]),0])

def test_fractional_shift_and_zero_boundary():
    a=np.array([[0.,1.,0.,0.]])
    assert np.allclose(m.shift(a,.5),[[0.,.5,.5,0.]])
    assert np.array_equal(m.shift(a,10),np.zeros_like(a))
