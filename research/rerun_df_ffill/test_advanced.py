"""Original-kernel equivalence, missing-label exclusion and artifact checks."""
import ast
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location('advanced', Path(__file__).with_name('advanced.py'))
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def context(experiment, missing=False):
    env = A.kernel(experiment, 1, True)
    names = A.selected_names(env['FEATURES'])
    rng = np.random.default_rng(51)
    def grid(n):
        return dict(C=rng.uniform(300, 700, (n, len(names['constant']))).astype('float32'),
                    E=rng.uniform(300, 700, (n, 72, len(names['varying']))).astype('float32'),
                    origins=pd.date_range('2020-01-01', periods=n, freq='24h', tz='UTC'))
    train, query = grid(512), grid(8)
    y = np.tile(train['C'][:, :1], (1,72)) + rng.normal(0, 30, (512,72))
    if missing:
        y[::7,0] = np.nan
    ctx = A.prepare_context(experiment, env, train, query, y, names,
                            np.full((512,24), 450.), np.full((8,24), 450.))
    return ctx


@pytest.mark.parametrize('experiment', A.RECIPES)
def test_original_kernel_and_missing_target_fit(experiment, tmp_path):
    ctx = context(experiment, True)
    actual, report = A.fit_horizon(experiment, ctx, 0, tmp_path/'adapter', smoke=True)
    env = A.kernel(experiment, 1, True)
    env.update(A.horizon_inputs(experiment, ctx, 0))
    env['out'] = tmp_path/'original'
    env['out'].mkdir()
    assert np.isfinite(env['y'][env['fit'],0]).all()
    exec(A.RECIPES[experiment]['fit'], env)
    expected = eval(A.RECIPES[experiment]['predict'], env)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert actual.shape == (8,)
    assert all((tmp_path/'adapter'/name).is_file() for name in report['models'])


@pytest.mark.parametrize('experiment', A.RECIPES)
def test_embedded_fit_matches_original_source(experiment):
    source = Path(__file__).with_name('source_archive').joinpath(experiment+'.py').read_text()
    main = next(x for x in ast.parse(source).body if isinstance(x, ast.FunctionDef) and x.name=='main')
    loop = next(x for x in ast.walk(main) if isinstance(x,ast.For) and ast.unparse(x.target)=='h')
    statements=[]
    for statement in loop.body:
        if isinstance(statement,ast.Assign) and ast.unparse(statement.targets[0])=='pred[:, h]':
            assert ast.dump(ast.parse(A.RECIPES[experiment]['predict'],mode='eval').body)==ast.dump(statement.value)
            break
        statements.append(statement)
    assert ast.dump(ast.parse(A.RECIPES[experiment]['fit']))==ast.dump(ast.Module(body=statements,type_ignores=[]))


def test_spectral_matches_original_and_masks_missing():
    source = Path(__file__).with_name('source_archive').joinpath(A.SPECTRAL+'.py').read_text()
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('point4_weights','point4_spectral')]
    env = dict(np=np, COEFFICIENT=.10, SCALE=50.)
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<original>','exec'),env)
    rng = np.random.default_rng(4)
    y=rng.uniform(300,700,(20,24)).astype('float32')
    pred=rng.uniform(300,700,(20,24)).astype('float32')
    matrix=type('Matrix',(),{'get_label':lambda self:y.ravel()})()
    actual=A.spectral_objective(y)(pred,matrix)
    expected=env['point4_spectral'](pred,matrix)
    for x,z in zip(actual,expected):
        np.testing.assert_allclose(x,z,rtol=1e-6,atol=1e-5)
    y[::3,::2]=np.nan
    grad,hess=A.spectral_objective(y)(pred,None)
    assert np.isfinite(grad).all() and np.isfinite(hess).all()
    assert (grad[~np.isfinite(y)]==0).all() and (hess[~np.isfinite(y)]==0).all()


def test_calibration_is_purged_and_missing_labels_excluded():
    exp='e2000fs_feat30_mc_calibration'
    ctx=context(exp,True)
    env=A.horizon_inputs(exp,ctx,0)
    assert (ctx['origins'][env['prefix']]+pd.Timedelta(hours=72)<ctx['origins'][env['calibration']].min()).all()
    assert np.isfinite(ctx['y'][env['calibration'],0]).all()
    assert not set(env['prefix']) & set(env['calibration'])


def test_selected_order_matches_original():
    names=A.selected_names(A.kernel(next(iter(A.RECIPES)))['FEATURES'])
    assert sum(map(len,names.values()))==30
    assert names['constant'][0]=='v_lag0'
    assert names['varying'][:4]==['rec26','rec27','rec28','rec_mean']
    assert A.mapped('ch_ch_t035_area')=='suvi_t035_area'


def test_grid_never_reads_query_targets_and_rejects_immature_train(monkeypatch):
    origin=pd.Timestamp('2025-12-28',tz='UTC')
    calls=[]
    def read(path, columns, filters):
        calls.append(columns)
        n=1 if ('h','==',1) in filters else 72
        values={'origin':[origin]*n,'v_lag0':[400.]*n,'h':np.arange(1,n+1),'y':[450.]*n}
        return pd.DataFrame({c:values[c] for c in columns})
    monkeypatch.setattr(A.pd,'read_parquet',read)
    names={'constant':['v_lag0'],'varying':[]}
    grid,y=A.load_grid('dummy','F1',names,False)
    assert y is None and len(grid['origins'])==1
    assert all('y' not in c for c in calls)
    with pytest.raises(ValueError,match='cutoff'):
        A.load_grid('dummy','F1',names,True)
