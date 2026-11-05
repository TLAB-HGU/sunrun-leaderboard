"""Prior-only calibration boundaries and original pooling formula parity."""
import ast
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
SPEC=importlib.util.spec_from_file_location('pooling',Path(__file__).with_name('pooling.py'))
P=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(P)


def toy():
    rng=np.random.default_rng(14);names=P.selected_names(P.FEATURES)
    def grid(n,start='2020-01-01'):
        return dict(origins=pd.date_range(start,periods=n,freq='24h',tz='UTC'),
                    C=rng.uniform(300,700,(n,len(names['constant']))).astype('float32'),
                    E=rng.uniform(300,700,(n,72,len(names['varying']))).astype('float32'))
    train,query=grid(512),grid(4,'2022-01-01')
    y=np.clip(rng.normal(490,110,(512,72)),100,1000).astype('float32')
    return train,query,y


@pytest.mark.parametrize('component',['body','spectral','detroute','mixture','era','gpd'])
def test_fresh_components_save_finite_models(component,tmp_path):
    train,query,y=toy()
    pred,timing=P.component_predictions(component,train,query,y,1,tmp_path,True)
    assert pred.shape==(4,72) and np.isfinite(pred).all() and timing>=0
    assert len(list(tmp_path.glob('*.ubj')))>=3


def test_prior_split_is_purged_and_disjoint():
    train,query,y=toy()
    reference,calibration,boundary=P.prior_split(train['origins'])
    assert (train['origins'][reference]+pd.Timedelta(hours=144)<=boundary).all()
    assert (train['origins'][calibration]>=boundary).all()
    assert not set(reference)&set(calibration)
    assert train['origins'][calibration].max()<query['origins'].min()


@pytest.mark.parametrize('experiment',[e for e in P.SUPPORTED if e!='e2000fs_calib5_detpool_all'])
def test_frozen_formula_parity(experiment):
    rng=np.random.default_rng(9);y=rng.uniform(300,700,(300,72))
    arms={k:y+rng.normal(0,50,y.shape) for k in P.required_components(experiment)}
    state=P.estimate_state(experiment,arms,y)
    actual=P.apply_state(experiment,arms,state)
    if experiment.endswith('offset_roll'):
        # Append 73 fictitious later origins: the first appended origin with all
        # prior residuals available has exactly the frozen final rolling mean.
        a=arms['spectral'];extra=np.zeros((73,72))
        offsets=P.causal_offsets(np.vstack([a,extra]),np.vstack([y,extra]),336,.25)
        expected=a+np.array([offsets[len(y)+h,h] for h in range(72)])
    elif experiment.endswith('pool_bodytail'):
        expected=arms['body'].copy()
        for start in (0,24,48):
            a=arms['body'][:,start:start+24];b=arms['spectral'][:,start:start+24]
            # Original causal least squares evaluated after every prior row matures.
            zeros=np.zeros((72,24))
            weight=P.causal_pool_weights(np.vstack([a,zeros]),np.vstack([b,zeros]),np.vstack([y[:,start:start+24],zeros]))[-1]
            expected[:,start:start+24]=(1-weight)*a+weight*b
    else:
        arm=state['arm'];a=arms['detroute'];b=arms[arm]
        w=P.choose_weights((a-y)**2,(b-y)**2,(a-y)*(b-y),y>=550,np.array([len(y)]))[0]
        expected=(1-w)*a+w*b
        if experiment.endswith('mix20'):expected=.8*expected+.2*arms['mixture']
    np.testing.assert_allclose(actual,np.clip(expected,1,3000),rtol=1e-12)


@pytest.mark.parametrize('file,function',[
    ('e2000fs_calib4_detpool_gpd.py','choose_weights'),
    ('e2000fs_calib_offset_roll.py','causal_offsets'),
    ('e2000fs_calib_pool_bodytail.py','causal_pool_weights'),
    ('e2000fs_calib5_detpool_all.py','fit_block'),
])
def test_original_formulas_embedded_unchanged(file,function):
    source=Path(__file__).with_name('source_archive').joinpath(file).read_text()
    original=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==function)
    current=next(n for n in ast.parse(Path(P.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==function)
    assert ast.dump(current)==ast.dump(original)


def test_query_target_poison_cannot_change_pool(tmp_path):
    train,query,y=toy();exp='e2000fs_calib_pool_bodytail'
    query['y']=np.full((4,72),np.nan)
    a,state,_=P.fit_pool(exp,train,query,y,1,tmp_path/'a',True)
    query['y'][:]=1e30
    b,state2,_=P.fit_pool(exp,train,query,y,1,tmp_path/'b',True)
    np.testing.assert_array_equal(a,b)
    assert state==state2


def test_loader_excludes_query_truth(monkeypatch):
    calls=[]
    def read(path,columns,filters):
        calls.append(columns);n=1 if ('h','==',1) in filters else 72
        data=dict(origin=[pd.Timestamp('2026-01-01',tz='UTC')]*n,h=np.arange(1,n+1),v_lag0=[400.]*n)
        return pd.DataFrame({c:data[c] for c in columns})
    monkeypatch.setattr(P.pd,'read_parquet',read)
    grid,y=P.load_grid('unused','F1',dict(constant=['v_lag0'],varying=[]),False)
    assert y is None and all('y' not in columns for columns in calls)


def test_poolall_uses_documented_actual_submission_not_smaller_guess():
    assert P.required_components('e2000fs_calib5_detpool_all')==P.POOLALL_ACTUAL_ARMS
    assert len(P.POOLALL_ACTUAL_ARMS)==16
    assert len(P.POOLALL_ARMS)==81
    with pytest.raises(ValueError,match='exact documented'):
        P.apply_poolall(np.ones((3,72)),{'gpd':np.ones((3,72))},np.ones(1))


def test_recovered_81_arm_inventory_matches_archived_weight_metadata():
    import json
    source=Path(__file__).resolve().parents[2]/'docs/explainers/data/calib5.json'
    original=json.loads(source.read_text())['arms']
    assert len(P.POOLALL_ARMS)==len(set(P.POOLALL_ARMS))==original['count']==81
    assert list(P.POOLALL_ARMS)==original['names']
    assert P.POOLALL_BLOCKER['inventory_recovered'] is True
    assert P.POOLALL_BLOCKER['missing_verified_recipe']==['e2000fs_calib3-20261031/runs/monotone_v1/forecast.npz']


def test_actual_submission_identity_and_arm_order_match_receipt():
    import json
    root=Path(__file__).resolve().parents[2]
    directory=root/'docs/visualizations/e2000fs_calib5_detpool_all'
    receipt=json.loads((directory/'weights_receipt.json').read_text())
    record=json.loads((directory/'leaderboard-result.json').read_text())
    assert tuple(receipt['arm_order'])==P.POOLALL_ACTUAL_ARMS
    code=ast.parse((directory/'code/pool_weights.py').read_text())
    allowed=next(ast.literal_eval(n.value) for n in code.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ALLOWED' for t in n.targets))
    assert tuple(allowed)==P.POOLALL_ACTUAL_ARMS
    for key in ['experiment','team_member','submitted_utc']:
        assert record[key]==P.POOLALL_IDENTITY[key]
    for key,value in P.POOLALL_IDENTITY['matching_mse'].items():
        assert record[key]==receipt['verification']['scores'][key]==value
    assert receipt['verification']['recomputed_stack_equals_submission'] is True
    assert receipt['verification']['max_abs_difference_float32']==0.


def test_actual16_nnls_matches_original_formula():
    rng=np.random.default_rng(81);y=rng.uniform(300,700,(100,72));base=y+rng.normal(0,60,y.shape)
    arms={name:y+rng.normal(0,80,y.shape) for name in P.POOLALL_ACTUAL_ARMS}
    delta=np.stack([arms[name]-base for name in P.POOLALL_ACTUAL_ARMS])
    expected=P.fit_block(delta.reshape(16,-1),(y-base).ravel(),(y>=550.).ravel())
    actual=P.poolall_weights(base,arms,y)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_allclose(P.apply_poolall(base,arms,actual),np.clip(base+np.einsum('p,pnh->nh',expected,delta),1,3000))


def test_nested_prior_poolall_never_uses_held_or_query_targets(monkeypatch,tmp_path):
    train,query,y=toy();calls=[]
    def fake(component,fit,apply,target,threads,out,smoke):
        calls.append((component,fit['origins'].max(),apply['origins'].min()))
        # Only fit labels contribute; query can carry arbitrary poison metadata.
        bias=sum(component.encode())%41
        return np.full((len(apply['origins']),72),float(target.mean())+bias),0.
    monkeypatch.setattr(P,'component_predictions',fake)
    query['y']=np.full((len(query['origins']),72),np.nan)
    a,state,_=P.fit_pool('e2000fs_calib5_detpool_all',train,query,y,1,tmp_path/'a',True)
    query['y'][:]=1e30
    b,state2,_=P.fit_pool('e2000fs_calib5_detpool_all',train,query,y,1,tmp_path/'b',True)
    np.testing.assert_array_equal(a,b);assert state==state2
    assert state['arm_order']==list(P.POOLALL_ACTUAL_ARMS)
    assert pd.Timestamp(state['inner_reference_end'])+pd.Timedelta(hours=144)<=pd.Timestamp(state['inner_calibration_start'])
    assert pd.Timestamp(state['inner_calibration_end'])+pd.Timedelta(hours=144)<=pd.Timestamp(state['outer_calibration_start'])
    assert pd.Timestamp(state['outer_reference_end'])+pd.Timedelta(hours=144)<=pd.Timestamp(state['outer_calibration_start'])
    assert all(fit+pd.Timedelta(hours=144)<=apply for _,fit,apply in calls)


@pytest.mark.parametrize('component',['lgbm_extreme4','lgbm_point4','lgbm_recent'])
def test_actual_lgb_components_preserve_original_scalar_fit(component,tmp_path):
    import lightgbm as lgb
    train,query,y=toy()
    actual,_=P.component_predictions(component,train,query,y,1,tmp_path,True)
    keep=np.arange(len(y))
    if component=='lgbm_recent':
        keep=keep[np.asarray((train['origins'].max()-train['origins'])/pd.Timedelta(days=1))<=365]
    x,xq=P.design(train,0)[keep],P.design(query,0)
    weights=P._event_weights(y[keep],0,0) if component=='lgbm_extreme4' else (P._point_weights_h(y[keep],0,0) if component=='lgbm_point4' else None)
    model=lgb.LGBMRegressor(**dict(P.LGB_PARAMS,n_jobs=1,n_estimators=2))
    model.fit(x,y[keep,0],sample_weight=weights)
    np.testing.assert_allclose(actual[:,0],np.clip(model.predict(xq),1.,3000.))
    assert len(list(tmp_path.glob('h*.txt')))==72
