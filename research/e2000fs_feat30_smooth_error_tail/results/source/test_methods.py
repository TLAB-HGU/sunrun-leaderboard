import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('smooth_tail_test_run', Path(__file__).with_name('run.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
m = r.methods


def test_derivatives_both_signs_and_cutoff():
    p = np.array([-250., -100., -50., 0., 50., 100., 250.])
    y = np.zeros_like(p)
    delta = 1e-3
    grad, hess = m.objective()(y,p)
    numeric_grad = (m.loss(y,p+delta)-m.loss(y,p-delta))/(2*delta)
    numeric_hess = (m.objective()(y,p+delta)[0]-m.objective()(y,p-delta)[0])/(2*delta)
    np.testing.assert_allclose(grad,numeric_grad,rtol=1e-7,atol=1e-7)
    np.testing.assert_allclose(hess,numeric_hess,rtol=1e-7,atol=1e-7)
    assert (hess>0).all()
    np.testing.assert_allclose(grad,-grad[::-1])
    np.testing.assert_allclose(hess,hess[::-1])


def test_zero_alpha_is_squared_error():
    y,p=np.array([400.,600.]),np.array([450.,500.])
    g,h=m.objective(alpha=0)(y,p)
    np.testing.assert_array_equal(g,p-y)
    np.testing.assert_array_equal(h,np.ones(2))
    np.testing.assert_array_equal(m.loss(y,p,alpha=0),.5*(p-y)**2)


def test_72_hour_maturity_boundary():
    origins=r.pd.date_range('2026-01-01',periods=5,freq='h',tz='UTC')
    np.testing.assert_array_equal(r.train_indices(origins,origins[2]+r.pd.Timedelta(hours=72)),[0,1])


def test_both_regions_required():
    base=dict(mse_lt550=100.,mse_ge550=400.)
    block=dict(mse_lt550=200.,mse_ge550=300.)
    point=dict(mse_lt550=250.,mse_ge550=250.)
    assert r.comparison(dict(mse_lt550=99.,mse_ge550=249.),base,block,point)['satisfies_point4_and_low']
    assert not r.comparison(dict(mse_lt550=101.,mse_ge550=200.),base,block,point)['satisfies_point4_and_low']
    c=r.comparison(dict(mse_lt550=99.,mse_ge550=275.),base,block,point)
    assert c['satisfies_block4_and_low'] and not c['satisfies_point4_and_low']


def test_model_offset_roundtrip_and_receipt_tamper(tmp_path):
    rng=np.random.default_rng(42)
    x=rng.normal(size=(128,4))
    y=450.+60*x[:,0]
    params=dict(n_estimators=3,num_leaves=4,min_child_samples=5,n_jobs=2,verbosity=-1,random_state=42)
    model,offset=r.save_model(x,y,x[:16],y[:16],params,tmp_path)
    assert offset == float(y.mean())
    context=dict(manifest_sha256='test',horizon=1)
    r.d.write(tmp_path/'receipt.json',dict(**context,sha256={f:r.d.digest(tmp_path/f) for f in ('model.txt','offset.json','loss.csv')}))
    restored,restored_offset=r.validate_receipt(tmp_path,context)
    np.testing.assert_array_equal(model.predict(x)+offset,restored.predict(x)+restored_offset)
    with pytest.raises(ValueError,match='context'):
        r.validate_receipt(tmp_path,dict(manifest_sha256='other',horizon=1))
    (tmp_path/'offset.json').write_text('{"offset":0}')
    with pytest.raises(ValueError,match='changed'):
        r.validate_receipt(tmp_path,context)
