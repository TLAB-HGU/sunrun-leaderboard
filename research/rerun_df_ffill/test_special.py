import ast
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import special as s

ROOT = Path(__file__).resolve().parents[2]


def original_lib():
    path = ROOT/'leaderboard/lb-f1f2/experiments/d143_axisA_analog/a2b/resid_lib_a2b.py'
    spec = importlib.util.spec_from_file_location('old_resid',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract(path,name,context):
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),context)  # noqa: S102
    return context[name]


def test_mixture_stitch_and_blend_parity():
    old = original_lib()
    rng = np.random.default_rng(42)
    h = np.arange(1,73)
    masks = h<=6,(h>=7)&(h<=24),h>24
    for seed in range(3):
        levels,event = s.level_seeds(seed),s.event_seed(seed)
        assert levels == old.level_seeds(seed)
        short1 = {i:rng.normal(size=6) for i in levels}
        short2 = {i:rng.normal(size=18) for i in levels}
        args = short1,short2,rng.normal(size=72),rng.normal(size=48),*masks,levels,event
        actual, expected = s.stitch_members(*args),old.stitch_members(*args)
        weights = {lead:dict(zip(s.MEMBERS,[.2,.3,.5])) for lead in s.LEADS}
        np.testing.assert_array_equal(s.blend_per_lead(actual,weights,h),old.blend_per_lead(expected,weights,h))


def test_residual_finite_parity_and_unfilled_target_mask():
    old = original_lib()
    fn = extract(ROOT/'leaderboard/lb-f1f2/experiments/d147_e0421_lean/lean.py','residual_multi',{'np':np,'pd':pd,'RL':old})
    dates = pd.date_range('2024-01-01',periods=1000,freq='h',tz='UTC')
    values = pd.Series(np.linspace(300,600,1000),index=dates)
    paths = np.ones((1000,72))*400
    args = dates[-2:],values,paths[-2:],paths[-14:-12],dates,paths,(15,),'std_equal',(1.,2.,3.)
    expected = fn(*args)[15]
    actual = s.residual_multi(*args,target_series=values)[15]
    np.testing.assert_array_equal(actual,expected)
    missing = pd.Series(np.nan,index=dates)
    with np.errstate(all='ignore'):
        masked = s.residual_multi(*args,target_series=missing)[15]
    assert np.isnan(masked).all()


def test_hsspeak_29features_and_bump_parity():
    path = ROOT/'leaderboard/experiments/f1_resubmit_20261026/hsspeak_f1.py'
    context = {'np':np,'pd':pd,'HORIZON':72,'WIDTH':12.,'SHRINK':.15}
    oldfeatures = extract(path,'features',context)
    oldbump = extract(path,'bump',context)
    dates = pd.date_range('2023-01-01',periods=1000,freq='h',tz='UTC')
    ace = pd.Series(np.linspace(300,600,1000),index=dates)
    origins = dates[-3:]
    base = np.tile(np.linspace(300,650,72),(3,1))
    hx = base+10
    actual = s.features(origins,base,hx,ace)
    assert actual.shape == (3,29)
    np.testing.assert_array_equal(actual,oldfeatures(origins,base,hx,ace))
    np.testing.assert_array_equal(s.bump(base,np.array([600,700,800]),np.array([25,35,60]),25),
                                  oldbump(base,np.array([600,700,800]),np.array([25,35,60]),25))


def test_calendar_training_boundaries():
    for period in s.CUTOFFS:
        windows,_ = s.calendar(period)
        cutoff = pd.Timestamp(s.CUTOFFS[period],tz='UTC')
        assert len(windows) == 3
        for _,train_end,origins in windows:
            assert train_end < origins.min()
            assert origins.max()+pd.Timedelta(hours=144) <= cutoff


def test_all_seven_finite_models_match_source(tmp_path):
    from types import SimpleNamespace

    import xgboost as xgb
    old = original_lib()
    fit_set = extract(ROOT/'leaderboard/lb-f1f2/experiments/d147_e0421_lean/lean.py','fit_set',
                      {'np':np,'pd':pd,'xgb':xgb,'RL':old,'SP':SimpleNamespace(SPLIT_H=24),'SMOKE':False})
    rng = np.random.default_rng(6)
    origins = pd.date_range('2024-01-01',periods=12,freq='3h',tz='UTC')
    x = rng.normal(size=(12*72,5)).astype(np.float32)
    x[:,0] = np.tile(np.arange(1,73),12)
    y = rng.normal(450,100,size=(12,72)).astype(np.float32)
    grid = SimpleNamespace(labels=lambda _:y.copy(),long=lambda _:x.copy())
    models = s.fit_mixture(grid,origins,pd.Timestamp('2025-01-01',tz='UTC'),0,2,1,tmp_path,'parity')
    frame = pd.DataFrame(x,columns=['h','a','b','c','d']);frame['y']=y.ravel()
    params = dict(s.PARAMS,n_estimators=2,n_jobs=1)
    shorts,long,levels,event = fit_set({'members':'mixture'},3,params,0,frame,['h','a','b','c','d'])
    np.testing.assert_array_equal(models[f'long_{event}'].inplace_predict(x),long.predict(x))
    for seed in levels:
        for kind,mask,idx in [('short1',x[:,0]<=6,0),('short2',(x[:,0]>=7)&(x[:,0]<=24),1)]:
            np.testing.assert_array_equal(models[f'{kind}_{seed}'].inplace_predict(x[mask]),shorts[seed][idx].predict(x[mask]))


def test_complete_smoke_retrains_28_models_without_query_targets(tmp_path,monkeypatch):
    dates = pd.date_range('2024-08-01','2026-03-29',freq='h',tz='UTC')
    n = len(dates)
    con = np.broadcast_to(np.arange(len(s.CONSTANT),dtype=np.float32),(n,len(s.CONSTANT)))
    ext = np.broadcast_to(np.arange(len(s.VARYING),dtype=np.float32),(n,72,len(s.VARYING)))
    timeline = pd.date_range(dates.min(),dates.max()+pd.Timedelta(hours=72),freq='h')
    speeds = pd.Series(450+100*np.sin(np.arange(len(timeline))/100),index=timeline)
    y = np.column_stack([speeds.reindex(dates+pd.Timedelta(hours=h)).to_numpy(np.float32) for h in range(1,73)])
    masks = {p:np.asarray(dates+pd.Timedelta(hours=144)<=pd.Timestamp(cut,tz='UTC')) for p,cut in s.CUTOFFS.items()}
    query = {'F1':pd.date_range('2025-12-31 23:00',periods=s.QUERY_COUNTS['F1'],freq='h',tz='UTC')}
    grid = s.Grid(dates,con,ext,y,np.ones(n,bool),query,masks)
    monkeypatch.setattr(s,'load_grid',lambda *_:(grid,speeds,speeds))
    data=tmp_path/'data';data.write_text('test snapshot')
    s.run_period(data,data,tmp_path,'F1',threads=1,smoke=True)
    assert len(list(tmp_path.glob('*.ubj'))) == 28
    with np.load(tmp_path/'F1.npz') as z:
        assert z['pred'].shape == (4,72)
        assert np.isfinite(z['pred']).all()


def test_loader_excludes_poisoned_query_y_and_preserves_training_ffill(tmp_path):
    dates = pd.date_range('2024-01-01',periods=3,freq='h',tz='UTC')
    frames = []
    for kind in ('train','query'):
        frame = pd.DataFrame({'origin':np.repeat(dates,72),'h':np.tile(np.arange(1,73),3),
            'set':kind,'period':'F1','fit_F1':kind=='train','fit_F2':kind=='train','fit_F3':kind=='train'})
        frame['y'] = np.arange(216).reshape(3,72).ravel() if kind == 'query' else np.tile(np.arange(1,73),3)+np.repeat(np.arange(3),72)
        frame['y'] = frame['y']+1e10 if kind == 'query' else frame['y']+400
        feature = {s.mapped(n):np.ones(216,dtype=np.float32) for n in s.CONSTANT if n != 'img_stale'}
        feature.update({n:np.ones(216,dtype=np.float32) for n in s.VARYING})
        frames.append(pd.concat([frame,pd.DataFrame(feature)],axis=1))
    path = tmp_path/'data.parquet'
    pd.concat(frames,ignore_index=True).to_parquet(path,index=False)
    grid,_,target = s.load_grid(path,path)
    assert grid.y.max()<1000
    assert target.max()<1000
    assert np.isfinite(grid.y).all()
    assert len(grid.queries['F1']) == 3
