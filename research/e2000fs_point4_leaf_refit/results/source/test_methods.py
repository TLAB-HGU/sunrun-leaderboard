import importlib.util
from pathlib import Path
import numpy as np
import lightgbm as lgb
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('point4_runner_test', HERE/'run.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
m = r.m


def test_train_only_scaling_preserves_missing():
    x = np.array([[1.,np.nan,3.],[5.,np.nan,3.]])
    mean,scale = m.scaling(x)
    np.testing.assert_array_equal(mean,[3.,0.,3.])
    np.testing.assert_array_equal(scale,[2.,1.,1.])
    future = np.array([[1e20,np.nan,-1e20]])
    transformed = (future-mean)/scale
    assert np.isnan(transformed[0,1])
    np.testing.assert_array_equal(m.scaling(x)[0],mean)
    assert np.isnan(((x-mean)/scale)[:,1]).all()


def test_stability_objective_equivalent_gradient_and_loss():
    y,teacher,p = np.array([2.,7.]),np.array([3.,4.]),np.array([8.,1.])
    pseudo,w = m.stable_targets(y,teacher)
    np.testing.assert_allclose(w*(p-pseudo),(p-y)+3*(p-teacher))
    original = .5*(y-p)**2+1.5*(p-teacher)**2
    reduced = .5*w*(pseudo-p)**2
    np.testing.assert_allclose(original-reduced,.375*(y-teacher)**2)
    np.testing.assert_array_equal(w,[4.,4.])


def test_gates_fail_closed_and_low_threshold():
    baseline = dict(zip(m.KEYS,[100.,100.,100.]))
    assert m.gate(dict(zip(m.KEYS,[99.,95.,100.])),baseline,True)
    for row in ([99.,95.1,100.],[99.,95.,100.1],[99.1,95.,100.],[99.,None,100.],[99.,float('nan'),100.]):
        assert not m.gate(dict(zip(m.KEYS,row)),baseline,True)
    candidate = {s:dict(zip(m.KEYS,[99.,95.,100.])) for s in ['dev0','dev1','dev2','pooled']}
    refs = {s:baseline for s in candidate}
    assert m.development_gate(candidate,refs)
    candidate['dev1']['mse_ge550']=101.
    assert not m.development_gate(candidate,refs)


def test_refit_new_booster_topology_reload_and_tamper(tmp_path):
    rng = np.random.default_rng(8)
    x = rng.normal(size=(256,4))
    y = 300.+80*x[:,0]+5*rng.normal(size=256)
    teacher = lgb.LGBMRegressor(n_estimators=8,num_leaves=7,min_child_samples=10,n_jobs=2,verbosity=-1).fit(x,y,sample_weight=np.where(y>350,4.,1.)).booster_
    old = teacher.model_to_string()
    model = r.fit(tmp_path,{'fold':'a'},x,y,{},teacher=teacher)
    assert model is not teacher and teacher.model_to_string()==old
    assert m.topology(model)==m.topology(teacher)
    restored = r.b.checkpoint(tmp_path,{'fold':'a'})
    np.testing.assert_array_equal(model.predict(x),restored.predict(x))
    with pytest.raises(ValueError,match='identity'):
        r.b.checkpoint(tmp_path,{'fold':'b'})
    with (tmp_path/'model.txt').open('a') as f:
        f.write('\nchanged\n')
    with pytest.raises(ValueError,match='Frozen file'):
        r.b.checkpoint(tmp_path,{'fold':'a'})
