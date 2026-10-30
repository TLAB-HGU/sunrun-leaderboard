"""Causal adaptive designs on synthetic sources, no official data."""
import copy
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.ch_breakthrough_v2.common import build_features
from experiments.ch_breakthrough_v2.physical_quality import build_ch_quality_frame, build_physics_frame
from experiments.mse20_four_regimes.tree import _features
from experiments.ch_breakthrough_v2.recurrence import RecurrenceResidualForecaster
from experiments.mse20_adaptive.features import design, dynamic_lags, observed_speed


def frames(n=1150):
    index = pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC')
    speed = 430+50*np.sin(np.arange(n)/29)+20*np.sin(np.arange(n)/7)
    ace = pd.DataFrame({'filled_speed_kms': speed, 'was_missing': 0}, index=index)
    ch = pd.DataFrame({f'ch_{tag}_{name}': value for tag in ('t035','t045','t055')
                       for name,value in [('area',.05),('top1_lon',-25.),('top1_lat',10.),('equatorial_area',.02)]}, index=index)
    raw = pd.DataFrame({'ace_density_cm3':5.,'ace_temperature_k':1e5,'ace_bt_nt':5.,'ace_speed_kms':speed}, index=index)
    return {'base':build_features(ace,ch), 'physics':build_physics_frame(raw,index),
            'quality':build_ch_quality_frame(ch,index), 'speed':ace.filled_speed_kms.copy(),
            'observed':pd.Series(True,index=index)}


@pytest.mark.parametrize('variant',['base','anchor','expanded','dynamic'])
def test_all72_future_poison_and_drop(variant):
    f = frames()
    origin = f['speed'].index[[1050]]
    _, expected, anchor = design(f,origin,variant)
    assert len(expected)==72 and all(t == np.dtype('float32') for t in expected.dtypes)
    for drop in (True,False):
        altered=copy.deepcopy(f)
        for key, value in altered.items():
            if drop: altered[key]=value.loc[:origin[0]].copy()
            elif key=='observed': value.loc[value.index>origin[0]]=False
            else: value.loc[value.index>origin[0]]=1900.
        _,actual,a=design(altered,origin,variant)
        pd.testing.assert_frame_equal(expected,actual,check_exact=True)
        np.testing.assert_array_equal(anchor,a)


def test_base_equivalence_and_anchor_present():
    f=frames(); origins=f['speed'].index[[1000,1002]]
    _, actual, anchor=design(f,origins,'base')
    _, old, expected=RecurrenceResidualForecaster(use_target=True)._build_predict_matrix(_features(f),origins,observed_speed(f))
    pd.testing.assert_frame_equal(actual,old.astype(np.float32))
    np.testing.assert_array_equal(anchor,expected)
    _, expanded,_=design(f,origins,'anchor')
    np.testing.assert_array_equal(expanded.anchor_mr.to_numpy(),anchor.astype(np.float32))


def test_missing_observation_and_schema_rejected():
    f=frames(); origins=f['speed'].index[[1000]]
    f['observed']=f['observed'].astype(object)
    f['observed'].iloc[0]=None
    with pytest.raises(ValueError,match='observed'): design(f,origins)
    f=frames(); f['base']['future_regime']=1.
    with pytest.raises(ValueError,match='schema'): design(f,origins)


def test_dynamic_lag_past_correlation_and_fallback():
    rng=np.random.default_rng(42)
    index=pd.date_range('2024-01-01',periods=1800,freq='h',tz='UTC')
    values=np.tile(rng.normal(400,30,600),3)
    speed=pd.Series(values,index=index);speed.attrs['observed']=pd.Series(True,index=index)
    lag,corr,cov,missing=dynamic_lags(speed,index[[1600]])
    assert lag[0]==600 and corr[0]>.999 and cov[0]==1 and missing[0]==0
    speed.attrs['observed'][:]=False
    lag,corr,cov,missing=dynamic_lags(speed,index[[1600]])
    assert lag[0]==648 and np.isnan(corr[0]) and missing[0]==1
