import importlib.util
from pathlib import Path
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('point4_v2_test', HERE/'run.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
m = r.m


@pytest.mark.parametrize('lane', m.LANES[:2])
def test_quadratic_identity_and_training_boundary(lane):
    y = np.array([549.,550.,700.,300.])
    teacher = y+np.array([0.,50.,-50.,500.])
    pred = y+100.
    pseudo,w = m.quadratic_targets(lane,y,teacher)
    gamma = w-1
    np.testing.assert_allclose(w*(pred-pseudo),pred-y+gamma*(pred-teacher))
    original = .5*(pred-y)**2+.5*gamma*(pred-teacher)**2
    np.testing.assert_allclose(original-.5*w*(pred-pseudo)**2,
                               .5*gamma/w*(y-teacher)**2)
    if lane == 'high_anchor':
        np.testing.assert_array_equal(w,[4.,16.,16.,4.])
    else:
        assert w[0] == 16. and w[1] == w[2]
        assert np.all((w>=4.) & (w<=16.))


def test_regret_derivatives_and_shift_invariance():
    y = np.array([600.,600.,600.,600.,300.])
    teacher = y+np.array([20.,20.,100.,100.,0.])
    pred = y+np.array([150.,-150.,30.,-30.,300.])
    high = y>=550.
    grad,hess = m.regret_derivatives(y,teacher,pred,high)
    eps = 1e-3
    finite_grad = (m.regret_loss(y,teacher,pred+eps,high)-m.regret_loss(y,teacher,pred-eps,high))/(2*eps)
    finite_hess = (m.regret_derivatives(y,teacher,pred+eps,high)[0]-m.regret_derivatives(y,teacher,pred-eps,high)[0])/(2*eps)
    np.testing.assert_allclose(grad,finite_grad,rtol=1e-8)
    np.testing.assert_allclose(hess,finite_hess,rtol=1e-8)
    np.testing.assert_allclose(m.regret_derivatives(y-500,teacher-500,pred-500,high),(grad,hess))
    assert (hess>=4.).all()
    boundary = m.regret_derivatives(y,teacher,teacher,high)
    np.testing.assert_array_equal(boundary[1],np.full(len(y),4.))
    np.testing.assert_allclose(boundary[0],teacher-y)
    # Both signs of the knot have a continuous gradient and an inactive
    # boundary Hessian; the exterior one-sided curvature is larger.
    for sign in (-1.,1.):
        knot = np.array([600.+sign*20.])
        truth, teach, mask = np.array([600.]),np.array([620.]),np.array([True])
        knot_grad = m.regret_derivatives(truth,teach,knot,mask)[0]
        inside = m.regret_derivatives(truth,teach,knot-sign*1e-5,mask)[0]
        outside = m.regret_derivatives(truth,teach,knot+sign*1e-5,mask)[0]
        np.testing.assert_allclose((knot_grad-inside)/(sign*1e-5),[4.],rtol=1e-6)
        np.testing.assert_allclose((outside-knot_grad)/(sign*1e-5),[4.16],rtol=1e-6)


def test_strict_maturity_and_fail_closed_gates():
    origins = r.pd.date_range('2025-01-01',periods=4,freq='h',tz='UTC')
    np.testing.assert_array_equal(r.b.mature_indices(origins,origins[0]+r.pd.Timedelta(hours=74)),[0,1])
    reference = dict(zip(m.KEYS,[100.,100.,100.]))
    good = dict(zip(m.KEYS,[99.,95.,100.]))
    assert m.gate(good,reference,True)
    for row in ([99.,95.1,100.],[99.,95.,100.1],[99.1,95.,100.],[99.,None,100.],[99.,np.nan,100.]):
        assert not m.gate(dict(zip(m.KEYS,row)),reference,True)
    candidates = {s:good.copy() for s in ('dev0','dev1','dev2','pooled')}
    refs = {s:reference for s in candidates}
    assert m.development_gate(candidates,refs)
    candidates['dev1']['mse_ge550']=101.
    assert not m.development_gate(candidates,refs)
    # Gates were removed from the runner on 2026-10-29 (user decision): every lane reaches the F1/F2 stages.
    source = (HERE/'run.py').read_text()
    assert 'completed_rejected' not in source and 'development_gate' not in source


@pytest.mark.parametrize('lane',m.LANES)
def test_fit_reload_offset_teacher_immutable_and_checkpoint(tmp_path,lane):
    rng = np.random.default_rng(24)
    x = rng.normal(size=(128,4))
    y = 500.+120*x[:,0]
    teacher = y+30*rng.normal(size=128)
    original = teacher.copy()
    params = dict(n_estimators=3,num_leaves=7,min_child_samples=5,n_jobs=2,verbosity=-1)
    model,offset = r.fit(tmp_path,{'stage':'test'},x,y,params,teacher=teacher,lane=lane)
    np.testing.assert_array_equal(teacher,original)
    assert offset == (float(y.mean()) if lane=='high_regret_guard' else 0.)
    restored,restored_offset = r.fit(tmp_path,{'stage':'test'},x,y,params,teacher=teacher,lane=lane)
    np.testing.assert_array_equal(model.predict(x)+offset,restored.predict(x)+restored_offset)
    assert np.isfinite(model.predict(x)+offset).all()
    with pytest.raises(ValueError,match='identity'):
        r.fit(tmp_path,{'stage':'wrong'},x,y,params,teacher=teacher,lane=lane)
    (tmp_path/'offset.json').write_text('{"offset":0}')
    with pytest.raises(ValueError,match='Frozen file'):
        r.fit(tmp_path,{'stage':'test'},x,y,params,teacher=teacher,lane=lane)
