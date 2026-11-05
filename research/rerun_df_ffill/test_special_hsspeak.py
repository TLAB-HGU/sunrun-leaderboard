import ast
import subprocess

import numpy as np
import pandas as pd
import pytest
import special_hsspeak as h


def original(path,name,context):
    source=subprocess.check_output(['git','show',f'd55d9f6d6:{path}'],text=True)
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),path,'exec'),context)  # noqa:S102
    return context[name]


def test_physics_summary_feature_and_bump_parity():
    dates=pd.date_range('2024-01-01',periods=1000,freq='h',tz='UTC')
    speed=pd.Series(np.arange(1000)+300.,index=dates);origins=dates[-3:]
    rng=np.random.default_rng(3);base=rng.uniform(300,800,(3,72));physics=rng.uniform(300,800,(3,72))
    summaries=pd.DataFrame(np.column_stack([physics.max(1),physics.argmax(1)+1,physics[:,[0,23,47,71]]]),index=origins,columns=h.HUXT_COLUMNS)
    fn=original('experiments/solar_hsspeak25_20261020/run.py','features',{'np':np,'pd':pd,'HORIZON':72,'CARR':654})
    np.testing.assert_array_equal(h.head_features(origins,base,summaries,speed),fn(origins,base,physics,speed))
    bump=original('experiments/solar_hsspeak25_20261020/run.py','bump',{'np':np,'HORIZON':72,'WIDTH':12.,'SHRINK':.15,'MIN_LEAD':25})
    np.testing.assert_array_equal(h.bump(base,np.array([600.,700.,800.]),np.array([25.,50.,72.]),25),
                                  bump(base,np.array([600.,700.,800.]),np.array([25.,50.,72.])))


def test_l72_residual_source_parity():
    context={'np':np,'pd':pd,'HOUR':h.HOUR,'utc':h.utc,'trailing':h.trailing}
    fn=original('experiments/e0472_ablation_hpo/recipe.py','residual',context)
    dates=pd.date_range('2024-01-01',periods=1400,freq='h',tz='UTC')
    inputs=pd.Series(np.linspace(300,700,1400),index=dates)
    labels=inputs.copy();labels.iloc[100:150]=np.nan
    paths=np.ones((1400,72))*400
    args=(dates[-3:],inputs,labels,paths[-3:],paths[-17:-14],dates,paths,h.CFG,dates[-1])
    actual,ascale=h.residual(*args);expected,escale=fn(*args)
    np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(ascale,escale)


def test_level_summary_and_ridge_source_parity():
    from types import SimpleNamespace
    rng=np.random.default_rng(7)
    dates=pd.date_range('2024-01-01',periods=20,tz='UTC')
    c=rng.normal(size=(20,len(h.CONSTANT)));paths=rng.normal(size=(20,72))
    grid=SimpleNamespace(c=c,index=lambda _:np.arange(20))
    data=SimpleNamespace(frame=lambda _:pd.DataFrame(np.repeat(c[:,[h.CONSTANT.index(n) for n in h.HISTORY]],72,axis=0),columns=h.HISTORY))
    summary=original('experiments/gjc_inc_l72cond_20261014/run_proxy.py','summary',{'np':np,'HISTORY':h.HISTORY})
    np.testing.assert_array_equal(h.level_summary(grid,dates,paths),summary(data,dates,paths))


def test_full_three_seed_sixfold_smoke(tmp_path,monkeypatch):
    dates=pd.date_range('2022-10-01','2026-09-27',freq='h',tz='UTC');n=len(dates)
    con=np.broadcast_to(np.arange(len(h.CONSTANT),dtype=np.float32),(n,len(h.CONSTANT)))
    ext=np.broadcast_to(np.arange(len(h.VARYING),dtype=np.float32),(n,72,len(h.VARYING)))
    timeline=pd.date_range(dates.min(),dates.max()+72*h.HOUR,freq='h')
    inputs=pd.Series(450+50*np.sin(np.arange(len(timeline))/100),index=timeline)
    y=np.column_stack([inputs.reindex(dates+i*h.HOUR).to_numpy(np.float32) for i in range(1,73)])
    train=dates<=h.utc('2026-06-23')
    y[~train]=np.nan
    masks={p:np.asarray(train&(dates+144*h.HOUR<=h.utc(cut))) for p,cut in h.CUTOFFS.items()}
    query={'F3':pd.date_range('2026-06-30 23:00',periods=h.QUERY_COUNTS['F3'],freq='h',tz='UTC')}
    grid=h.Grid(dates,con,ext,y,train,query,masks)
    target=inputs.copy();target.loc[target.index>h.utc('2026-06-26')]=np.nan
    monkeypatch.setattr(h,'load_grid',lambda *_:(grid,inputs,target))
    physics=pd.DataFrame({'origin':dates,'huxt_max':600.,'huxt_argmax':30.,'huxt_h1':400.,'huxt_h24':500.,'huxt_h48':550.,'huxt_h72':450.})
    path=tmp_path/'physics.parquet';physics.to_parquet(path,index=False)
    data=tmp_path/'df_ffill_trainy.parquet';data.write_bytes(b'synthetic')
    calls=[];original_train=h.xgb.train
    def counting(*args,**kwargs):
        calls.append(1)
        return original_train(*args,**kwargs)
    monkeypatch.setattr(h.xgb,'train',counting)
    report=h.run_period(data,data,path,tmp_path,'F3',1,True)
    trained=len(calls)
    assert report['head_training_origins']==48
    assert len(list(tmp_path.glob('F3_fold*_s*.npz')))==18
    assert len(list(tmp_path.glob('*.ubj')))>=7*3*6
    with np.load(tmp_path/'F3_predictions.npz') as z:
        assert z['pred'].shape==(4,72) and np.isfinite(z['pred']).all()
        baseline=z['pred'].copy()
    assert report['checkpoint_nodes']['trained']>trained
    resumed=tmp_path/'resumed'
    interrupt_calls=[]
    def interrupt(*args,**kwargs):
        if len(interrupt_calls)==4:raise RuntimeError('simulated reboot')
        interrupt_calls.append(1)
        return original_train(*args,**kwargs)
    monkeypatch.setattr(h.xgb,'train',interrupt)
    with pytest.raises(RuntimeError,match='simulated reboot'):
        h.run_period(data,data,path,resumed,'F3',1,True)
    assert not (resumed/'F3_metadata.json').exists()
    assert len(list((resumed/'hsspeak_checkpoints').glob('*.json')))==5  # context + four nodes
    calls.clear();monkeypatch.setattr(h.xgb,'train',counting)
    report=h.run_period(data,data,path,resumed,'F3',1,True)
    assert len(calls)==trained-4
    assert report['checkpoint_nodes']['reused']>=4
    with np.load(resumed/'F3_predictions.npz') as z:np.testing.assert_array_equal(z['pred'],baseline)


def test_query_target_poison_is_not_loaded_for_heads_or_residual(tmp_path):
    dates=pd.date_range('2024-01-01',periods=3,freq='h',tz='UTC')
    frames=[]
    for kind in ('train','query'):
        frame=pd.DataFrame({'origin':np.repeat(dates,72),'h':np.tile(np.arange(1,73),3),'set':kind,'period':'F1',
                            'fit_F1':kind=='train','fit_F2':kind=='train','fit_F3':kind=='train'})
        frame['y']=400+np.tile(np.arange(1,73),3)+np.repeat(np.arange(3),72)
        values={h.mapped(n):np.ones(216,dtype=np.float32) for n in h.CONSTANT if n!='img_stale'}
        values.update({n:np.ones(216,dtype=np.float32) for n in h.VARYING})
        frames.append(pd.concat([frame,pd.DataFrame(values)],axis=1))
    data=pd.concat(frames,ignore_index=True);path=tmp_path/'data.parquet'
    outputs=[]
    for poison in (np.nan,1e12):
        data.loc[data['set'].eq('query'),'y']=poison;data.to_parquet(path,index=False)
        grid,inputs,targets=h.load_grid(path,path)
        outputs.append((grid.y,inputs.to_numpy(),targets.to_numpy()))
    for before,after in zip(*outputs):np.testing.assert_array_equal(before,after)


def test_dynamic_level_replay_keeps_original_end_coverage(monkeypatch):
    from types import SimpleNamespace
    end=h.utc('2026-09-26 23:00')
    engine=SimpleNamespace(end=end,cutoff=h.utc('2025-12-29'))
    bank=h.LevelBank(engine)
    seen=[]
    monkeypatch.setattr(h,'L72Replay',lambda engine,cutoff,**kwargs:seen.append(kwargs['end']) or object())
    bank.replay(h.utc('2025-12-29'))
    bank.replay(h.utc('2025-07-01'))
    assert seen==[end,h.utc('2026-01-01')-h.HOUR]


def test_ridge_fit_matches_original():
    context={'make_pipeline':h.make_pipeline,'SimpleImputer':h.SimpleImputer,
             'StandardScaler':h.StandardScaler,'Ridge':h.Ridge}
    fn=original('experiments/gjc_inc_l72cond_20261014/run_proxy.py','head_fit',context)
    rng=np.random.default_rng(17);x=rng.normal(size=(80,18));x[::4,3]=np.nan;y=rng.normal(size=80)
    np.testing.assert_array_equal(h.level_head(x,y).predict(x),fn(x,y).predict(x))


def test_merge_rejects_changed_physics_snapshot(tmp_path,monkeypatch):
    import json

    import pytest
    monkeypatch.setattr(h,'QUERY_COUNTS',{'F1':1,'F2':1})
    for i,period in enumerate(('F1','F2')):
        meta={'smoke':False,'rounds':900,'experiment':h.EXPERIMENT,'data':'same','raw_data':'same',
              'huxt_features':{'path':'physics','bytes':i+1}}
        (tmp_path/f'{period}_metadata.json').write_text(json.dumps(meta))
        np.savez(tmp_path/f'{period}_predictions.npz',pred=np.ones((1,72)),origins=np.array([h.utc('2026-01-01').value]))
    with pytest.raises(ValueError,match='snapshots differ'):h.merge(tmp_path)


def test_checkpoint_corruption_isolated_and_identity_invalidated(tmp_path,monkeypatch):
    data=tmp_path/'input';data.write_bytes(b'fresh input')
    source=tmp_path/'source.py';source.write_bytes(b'frozen source')
    monkeypatch.setattr(h,'__file__',str(source))
    cache=h.NodeCache(tmp_path,data,data,data,1,True)
    x=np.arange(54,dtype=float).reshape(3,18);y=np.arange(3,dtype=float)
    oo=pd.date_range('2025-01-01',periods=3,tz='UTC')
    calls=[];original=h.level_head
    def fit(*args):
        calls.append(1);return original(*args)
    monkeypatch.setattr(h,'level_head',fit)
    expected=cache.ridge('ridge_a',x,y,oo,oo[-1]).predict(x)
    cache.ridge('ridge_b',x,y,oo,oo[-1])
    assert len(calls)==2
    (tmp_path/'ridge_a.pkl').write_bytes(b'broken')
    resumed=h.NodeCache(tmp_path,data,data,data,1,True)
    np.testing.assert_array_equal(resumed.ridge('ridge_a',x,y,oo,oo[-1]).predict(x),expected)
    resumed.ridge('ridge_b',x,y,oo,oo[-1]);assert len(calls)==3
    assert (resumed.trained,resumed.reused)==(1,1)
    # An incomplete NPZ proof invalidates only its node, even with a valid model.
    (resumed.root/'ridge_a.npz').write_bytes(b'PK\x03\x04truncated')
    resumed.ridge('ridge_a',x,y,oo,oo[-1]);resumed.ridge('ridge_b',x,y,oo,oo[-1]);assert len(calls)==4
    for changed,payload in ((data,b'different input'),(source,b'different source')):
        changed.write_bytes(payload)
        resumed=h.NodeCache(tmp_path,data,data,data,1,True)
        resumed.ridge('ridge_a',x,y,oo,oo[-1]);resumed.ridge('ridge_b',x,y,oo,oo[-1])
        assert (resumed.trained,resumed.reused)==(2,0)
    assert len(calls)==8
    assert len(list(tmp_path.glob('hsspeak_invalid_*')))==2
    data.write_bytes(b'changed during execution')
    with pytest.raises(ValueError,match='Input changed during'):
        resumed.ridge('ridge_a',x,y,oo,oo[-1])


def test_booster_checkpoint_checks_rounds_and_loss(tmp_path):
    import json

    data=tmp_path/'input';data.write_bytes(b'fresh input')
    cache=h.NodeCache(tmp_path,data,data,data,1,True)
    x=np.arange(24,dtype=np.float32).reshape(8,3);y=np.arange(8,dtype=np.float32)
    params=dict(h.PARAMS,seed=0,nthread=1,max_bin=256)
    witness={'origins':np.arange(8)}
    calls=[]
    def matrix():
        calls.append(1);return h.xgb.QuantileDMatrix(x,label=y,nthread=1)
    def fit(tag,rounds=2):return cache.booster(tag,params,rounds,['a','b','c'],witness,matrix)
    fit('a');fit('b');assert len(calls)==2
    (tmp_path/'a.ubj').write_bytes(b'corrupt booster')
    fit('a');fit('b');assert len(calls)==3
    (tmp_path/'a_loss.json').write_text(json.dumps({'train':{'rmse':[1.]}}))
    fit('a');fit('b');assert len(calls)==4
    fit('a',3);fit('b');assert len(calls)==5
    assert (cache.trained,cache.reused)==(5,3)
