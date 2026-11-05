import importlib.util
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest

spec=importlib.util.spec_from_file_location('rerun_tda',Path(__file__).with_name('tda.py'))
t=importlib.util.module_from_spec(spec);sys.modules[spec.name]=t;spec.loader.exec_module(t)


def original():
    source=subprocess.check_output(['git','show','6b85acc0b:research/e2100_tda_pi_suvi-20261031/tda.py'],text=True)
    namespace={};exec(compile(source,'original_tda','exec'),namespace);return namespace


def test_exact_preprocessing_diagrams_and_pi_source_parity():
    source=original()
    rng=np.random.default_rng(42)
    rad=rng.uniform(0,100,(80,80)).astype(np.float32)
    a,mask=t.preprocess(rad,30.,40.,40.)
    b,bmask=source['preprocess'](rad,30.,40.,40.)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(mask,bmask)
    for actual,expected in zip(t.diagrams(a),source['diagrams'](b)):
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(t.persistence_image(actual),source['persistence_image'](expected))


def test_pi_weight_formula_and_empty():
    assert not t.persistence_image(np.zeros((0,2))).any()
    dg=np.array([[.2,.6]])
    expected=.4*np.exp(-((t.CENT_B-.2)**2+(t.CENT_P-.4)**2)/(2*.05**2))/14**2
    np.testing.assert_allclose(t.persistence_image(dg),expected)


def test_causal_frame_and_indefinite_ffill():
    end=pd.Timestamp('2025-01-01T00:59:59Z')
    frame=pd.DataFrame(np.vstack([np.ones(392),np.full(392,2.)]),columns=t.PI_COLS)
    frame.insert(0,'obs_end',[end,end+pd.Timedelta(days=10)])
    origins=pd.DatetimeIndex([end-pd.Timedelta(seconds=1),end,end+pd.Timedelta(days=9)])
    x,ok,age=t.frame_lookup(frame,origins)
    assert list(ok)==[False,True,True]
    assert np.isnan(x[0]).all() and (x[1:]==1.).all()
    assert age[-1]==216


def test_noaa_filename_timestamp():
    p='OR_SUVI-L1b-Fe195_G18_s20223621259048_e20223621259058_c20223621259243.nc'
    assert t.file_end(p)==pd.Timestamp('2022-12-28T12:59:05.8Z')


def test_ace_features_match_original_and_future_poison():
    import ast
    src=(Path(__file__).parent/'source_archive/e2100_tda_pi_suvi72.py').read_text()
    node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='ace_features')
    ns={'pd':pd,'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'original_ace','exec'),ns)
    series=pd.Series(np.arange(200.),index=pd.date_range('2025-01-01',periods=200,freq='h',tz='UTC'))
    origins=series.index[90:95]
    x,names=t.ace_features(series,origins)
    expected,expected_names=ns['ace_features'](series,origins)
    np.testing.assert_array_equal(x,expected);assert names==expected_names and len(names)==23
    series.iloc[95:]=99999
    np.testing.assert_array_equal(x,t.ace_features(series,origins)[0])


def test_actual_nc_image_and_cache_reuse(tmp_path):
    paths=t.frame_paths('data-v2/raw/SUVI/nc/g18')
    if not paths:
        pytest.skip('raw SUVI absent')
    end,path=paths[0]
    actual=t.cached_frame((end,path,str(tmp_path)))
    assert actual['status']=='ok'
    expected,_=original()['features'](path)
    np.testing.assert_array_equal(actual['vector'],expected)
    assert actual['vector'].shape==(392,)
    again=t.cached_frame((end,path,str(tmp_path)))
    assert again['cached']
    np.testing.assert_array_equal(actual['vector'],again['vector'])


def test_ridge_prior_split_and_artifacts(tmp_path):
    dates=pd.date_range('2024-01-01',periods=200,freq='D',tz='UTC')
    rng=np.random.default_rng(42);x=rng.normal(size=(200,5));y=np.tile(300+x[:,0,None],(1,72))
    y[0,0]=np.nan
    pred,chosen,_=t.ridge_fit_predict(x,y,x[:2],dates,tmp_path)
    assert pred.shape==(2,72) and np.isfinite(pred).all()
    assert set(chosen)<=set(t.ALPHAS)
    assert (tmp_path/'h72.joblib').exists()
    import json
    receipt=json.loads((tmp_path/'selection.json').read_text())
    assert pd.Timestamp(receipt['selection_train_end'])+pd.Timedelta(hours=144)<pd.Timestamp(receipt['selection_valid_start'])
