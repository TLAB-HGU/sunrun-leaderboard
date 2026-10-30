"""Hourly coverage, confirmation-time, and availability regression tests."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pandas as pd

D=Path(__file__).resolve().parents[1]/'experiments/d156_nm57_hourly'
sys.path.insert(0,str(D))
import hourly as H


def test_full_hourly_search_is_eligible_and_detector_runs():
    index=pd.date_range('2024-08-01',periods=52,freq='h',tz='UTC')
    z=pd.DataFrame({'signal':np.full(52,5.)},index=index)
    lead,eligible=H.detect(z,index[-1])
    assert eligible and lead==46.  # third point confirms; no backdating


def test_missing_window_not_reported_as_scientific_zero():
    index=pd.date_range('2024-08-01',periods=52,freq='h',tz='UTC')
    z=pd.DataFrame({'signal':np.full(52,np.nan)},index=index)
    assert H.detect(z,index[-1])==(None,False)


def test_confirmation_is_per_marker_not_switching_markers():
    index=pd.date_range('2024-08-01',periods=52,freq='h',tz='UTC')
    vals=np.zeros((52,2)); vals[np.arange(52),np.arange(52)%2]=5
    z=pd.DataFrame(vals,index=index)
    assert not H.onsets(z).any()
    assert H.detect(z,index[-1])==(None,True)


def test_hourly_bin_close_latency_and_future_poison():
    index=pd.date_range('2024-08-01',periods=180,freq='h',tz='UTC')
    s=pd.Series(100+np.random.default_rng(0).normal(size=180),index=index)
    cut=index[120]+pd.Timedelta(minutes=30)
    poisoned=s.copy(); future=index+pd.Timedelta(minutes=64)>cut
    poisoned.loc[future]*=np.arange(future.sum())%5+1
    clean=H.marker_frame(s); dirty=H.marker_frame(poisoned)
    assert clean.index[0]==index[0]+pd.Timedelta(minutes=64)
    before=clean.index<=cut
    np.testing.assert_array_equal(clean.loc[before],dirty.loc[before])
    assert not np.array_equal(clean.loc[~before],dirty.loc[~before],equal_nan=True)


def test_no_lookahead_after_outage():
    index=pd.date_range('2024-08-01',periods=180,freq='h',tz='UTC')
    s=pd.Series(100+np.random.default_rng(1).normal(size=180),index=index)
    s.iloc[80:]=np.nan
    assert H.marker_frame(s).iloc[-1].isna().all()
