"""Finite-data recipe parity and missing-target budget semantics."""
import ast
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
SPEC=importlib.util.spec_from_file_location('structured',Path(__file__).with_name('structured.py'))
S=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(S)


def toy(experiment,missing=False):
    env=S.kernel(experiment,1);names=S.selected_names(env['FEATURES'])
    rng=np.random.default_rng(17)
    def grid(n):
        return dict(C=rng.uniform(300,700,(n,len(names['constant']))).astype('float32'),
                    E=rng.uniform(300,700,(n,72,len(names['varying']))).astype('float32'),
                    origins=pd.date_range('2020-01-01',periods=n,freq='24h',tz='UTC'))
    train,query=grid(512),grid(8)
    y=np.tile(train['C'][:,:1],(1,72))+rng.normal(0,25,(512,72)).astype('float32')
    if missing:
        y[::3,::2]=np.nan
    cutoff=train['origins'][-1]+pd.Timedelta(hours=144)
    return env,names,train,query,y,cutoff


@pytest.mark.parametrize('experiment',[e for e in S.SUPPORTED if 'agnostic_v2_' not in e])
def test_partial_label_fit_and_artifacts(experiment,tmp_path):
    env,names,train,query,y,cutoff=toy(experiment,True)
    pred,reports,_=S.fit_predict(experiment,env,train,query,y,names,cutoff,tmp_path,True,train)
    assert pred.shape==(8,72) and np.isfinite(pred).all()
    assert all((tmp_path/f'block{i}.ubj').exists() for i in range(3))
    assert sum(r['observed_training_targets'] for r in reports)==np.isfinite(y).sum()


@pytest.mark.parametrize('experiment',[e for e in S.SUPPORTED if 'budget_' in e])
def test_finite_recipe_training_parity(experiment,tmp_path):
    env,names,train,query,y,cutoff=toy(experiment)
    actual,_,_=S.fit_predict(experiment,env,train,query,y,names,cutoff,tmp_path,True,train)
    X=[S.design(train,s) for s in (0,24,48)]
    origins=train['origins']
    if experiment.endswith('complexity'):
        ctx=train['C'][:,[names['constant'].index(n) for n in env['CONTEXT']]]
        raw,active=env['forward_complexity'](X,y,origins,ctx,2)
        weights=env['mass_weights'](y,origins,raw,active)
    elif experiment.endswith('covshift'):
        index,ratio=env['density_ratio'](X[0],origins,cutoff,3)
        weights=env['shifted_weights'](y,origins,ratio)
    expected=np.empty_like(actual)
    for b,start in enumerate((0,24,48)):
        target=y[:,start:start+24]
        matrix=xgb.DMatrix(X[b],label=target,nthread=1)
        params=dict(env['PARAMS'],base_score=float(target.mean()))
        if experiment.endswith('primaldual'):
            boundary=cutoff-pd.Timedelta(days=90)
            ref=np.flatnonzero(origins+pd.Timedelta(hours=144)<=boundary)
            cal=np.flatnonzero((origins>=boundary)&(origins+pd.Timedelta(hours=144)<=cutoff))
            ry=target[ref];cy=target[cal]
            parent=xgb.train(dict(env['PARAMS'],base_score=float(ry.mean())),xgb.DMatrix(X[b][ref],label=ry,nthread=1),num_boost_round=2,obj=env['point4_l2'])
            high=cy>=550
            budget=max(1.,float(np.mean((np.clip(parent.inplace_predict(X[b][cal]),1,3000)-cy)[high]**2)))
            objective=env['RiskObjective'](target,budget)
            model=xgb.train(params,matrix,num_boost_round=2,obj=objective,evals=[(matrix,'train')],custom_metric=env['high_mse'],callbacks=[env['DualUpdate'](objective)],verbose_eval=False)
        else:
            model=xgb.train(params,matrix,num_boost_round=2,obj=env['weighted_l2'](weights[:,start:start+24]))
        expected[:,start:start+24]=np.clip(model.inplace_predict(S.design(query,start)),1,3000)
    np.testing.assert_allclose(actual,expected,rtol=1e-6)


def test_missing_weights_preserve_observed_group_mass():
    env,names,train,query,y,cutoff=toy(S.SUPPORTED[0],True)
    base=S.point_weights(y)
    np.testing.assert_allclose(base.sum(0,dtype=float),np.isfinite(y).sum(0),rtol=1e-6)
    assert (base[~np.isfinite(y)]==0).all()
    raw=np.full_like(y,1.5);active=np.ones(len(y),bool)
    mass=S.mass_weights(y,train['origins'],raw,active)
    np.testing.assert_allclose(mass,base)
    shifted=S.shifted_weights(y,train['origins'],np.linspace(.25,4,len(y)))
    for h in range(72):
        for band in range(4):
            mask=(np.digitize(y[:,h],[400,550,700])==band)&np.isfinite(y[:,h])
            np.testing.assert_allclose(shifted[mask,h].sum(),base[mask,h].sum(),rtol=1e-6)


def test_masked_dual_ignores_missing_coordinates():
    target=np.array([[600.,np.nan],[400.,700.]],dtype='float32')
    obj=S.MaskedRiskObjective(target,100.)
    assert obj.p==2/3
    matrix=xgb.DMatrix(np.ones((2,2)),label=np.nan_to_num(target))
    g,h=obj(np.full((2,2),500.,dtype='float32'),matrix)
    assert g[0,1]==h[0,1]==0
    assert np.isfinite(g).all()


@pytest.mark.parametrize('experiment',S.SUPPORTED)
def test_frozen_helpers_match_archived_functions(experiment):
    original=ast.parse(Path(__file__).with_name('source_archive').joinpath(experiment+'.py').read_text())
    embedded=ast.parse(S.RECIPES[experiment])
    functions={n.name:ast.dump(n) for n in original.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name not in ('main','targets')}
    assert functions=={n.name:ast.dump(n) for n in embedded.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}


def test_probtraj_finite_training_and_donor_parity(tmp_path):
    experiment='e2000fs_feat30_probtraj'
    env,names,train,query,y,cutoff=toy(experiment)
    actual,_,_=S.fit_predict(experiment,env,train,query,y,names,cutoff,tmp_path,True)
    fit_leaves=[];query_leaves=[]
    for start in (0,24,48):
        x,xq=S.design(train,start),S.design(query,start)
        model=xgb.XGBRegressor(**env['PARAMS'],n_estimators=2)
        model.fit(x,y[:,start:start+24],verbose=False)
        fit_leaves.append(model.get_booster().predict(xgb.DMatrix(x,nthread=1),pred_leaf=True).astype(np.int32))
        query_leaves.append(model.get_booster().predict(xgb.DMatrix(xq,nthread=1),pred_leaf=True).astype(np.int32))
    expected=np.clip(env['trajectory_means'](np.hstack(fit_leaves),y,np.hstack(query_leaves)),1.,3000.)
    np.testing.assert_allclose(actual,expected,rtol=1e-6)


def test_donor_missing_mass_and_no_fallback():
    y=np.array([[100.,np.nan],[200.,400.]])
    np.testing.assert_allclose(S.observed_donor_mean(np.array([.75,.25]),y),[125.,400.])
    with pytest.raises(ValueError,match='no observed'):
        S.observed_donor_mean(np.array([1.,0.]),y)


@pytest.mark.parametrize('suffix',['context','event'])
def test_agnostic_original_finite_forecast_parity(suffix,tmp_path):
    experiment='e2000fs_feat30_agnostic_v2_'+suffix
    env,names,train,query,y,cutoff=toy(experiment)
    n=32 if suffix=='event' else 160
    train={k:v[:n] for k,v in train.items()};y=y[:n]
    if suffix=='event':
        parameters=np.column_stack((np.linspace(350,500,n),np.linspace(100,300,n),np.linspace(4,24,n),
                                    np.full(n,8.),np.full(n,60.),np.full(n,12.),np.full(n,16.)))
        y=env['event_curve'](parameters)
    actual,reports,_=S.fit_predict(experiment,env,train,query,y,names,cutoff,tmp_path,True)
    assert sum(r['observed_training_targets'] for r in reports)==y.size
    expected=np.empty_like(actual)
    if suffix=='context':
        for b,start in enumerate((0,24,48)):
            x,xq=S.design(train,start),S.design(query,start)
            target=y[:,start:start+24]
            q=env['event_probabilities'](x,target,train['origins'].values,3)
            model=xgb.XGBRegressor(**{**env['PARAMS'],'n_estimators':3,'objective':env['context_loss'](q),'base_score':float(target.astype('float32').mean())})
            model.fit(x,target.astype('float32'),verbose=False)
            expected[:,start:start+24]=model.get_booster().predict(xgb.DMatrix(xq,nthread=1))
            assert (tmp_path/f'block{b}.ubj').exists()
    else:
        curves=np.stack([env['fit_curve'](target) for target in y])
        mean,scale=curves.mean(0),np.maximum(curves.std(0),1.)
        coded=np.asarray((curves-mean)/scale,dtype='float32')
        model=xgb.XGBRegressor(**{**env['PARAMS'],'n_estimators':3,'base_score':float(coded.mean())})
        model.fit(env['design_all'](train,names),coded,verbose=False)
        raw=model.get_booster().predict(xgb.DMatrix(env['design_all'](query,names),nthread=1))
        expected=env['event_curve'](raw*scale+mean)
        assert (tmp_path/'event.ubj').exists()
    np.testing.assert_allclose(actual,np.clip(expected,1,3000),rtol=1e-6)


def test_query_grid_never_reads_truth(monkeypatch):
    calls=[]
    def read(path,columns,filters):
        calls.append(columns)
        n=1 if ('h','==',1) in filters else 72
        values=dict(origin=[pd.Timestamp('2026-01-01',tz='UTC')]*n,h=np.arange(1,n+1),v_lag0=[400.]*n)
        return pd.DataFrame({c:values[c] for c in columns})
    monkeypatch.setattr(S.pd,'read_parquet',read)
    grid,y=S.load_grid('unused','F1',dict(constant=['v_lag0'],varying=[]),False)
    assert y is None and len(grid['origins'])==1
    assert all('y' not in columns for columns in calls)
