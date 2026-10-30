"""Causality and drag physics regression tests for the bounded arrival diagnostic."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parents[1]/'experiments/d153_shifts/arrival_e2'
sys.path.insert(0,str(HERE))
import data
spec=importlib.util.spec_from_file_location('arrival_run',HERE/'run.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)

def test_drag_inversion_fast_and_slow():
    g=torch.tensor([1e-7,1e-7]);v=torch.tensor([1000.,250.]);w=torch.tensor([400.,400.])
    t=run.transit(g,v,w)
    assert torch.allclose(run.distance(t,g,v,w),torch.full_like(g,run.DIST),rtol=1e-5)
    assert t[0]<run.DIST/400/3600<t[1]

def test_no_nat_origin():
    assert data.stamp(None) is None
    assert data.stamp('bad') is None

def test_future_poison_positive_control():
    o=100*data.H
    aa=[dict(sub=o-10*data.H,t21=o-20*data.H,vals=[800,30,0,0],runs=[(o-5*data.H,o+40*data.H)])]
    ambient=pd.DataFrame({'ace_speed_kms':[400.,500.,600.], 'ace_density_cm3':[2.,3.,4.]},index=[o-72*data.H,o-24*data.H,o+data.H])
    assert data.poison({'a':aa},ambient,[dict(aid='a',origin=o)])['passed']
