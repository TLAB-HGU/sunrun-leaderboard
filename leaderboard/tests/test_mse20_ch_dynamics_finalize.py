"""Synthetic static-selection and persisted causality-proof tests; no official scores."""
import contextlib
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.mse20_ch_dynamics import finalize as f


def predictions(origins, value):
    return f.p.canonical_predictions(pd.DataFrame({f.p.KEY[0]:np.repeat(origins,72),
        f.p.KEY[1]:np.tile(np.arange(1,73),len(origins)), 'pred_kms':value}),origins)


def component(tmp_path, key, family='tree', value=400., mse=100.):
    origins = pd.date_range('2020-04-01',periods=3,freq='h',tz='UTC')
    directory = tmp_path/key
    directory.mkdir()
    model = directory/'model.bin';model.write_bytes(key.encode())
    f.p.write_new(directory/'metadata.json',{'config':{'id':key},'cutoff':f.p.CUTOFF})
    f.p.write_new(directory/'pilot.json',{'estimated_final_seconds':100.,'files':{}})
    return dict(key=key,family=family,config={'id':key},model_dir=str(directory),
        metadata=f.p.file_ref(directory/'metadata.json'),model_files=[f.p.file_ref(model)],
        prediction=f.save_prediction(predictions(origins,value),directory/'candidate.parquet'),
        metrics={'official':{'mse':mse},'rank':mse},pilot=f.p.file_ref(directory/'pilot.json'))


def test_simplex_exact_quota_and_negative_cases():
    assert len(f.simplex_weights(3))==45
    assert len(f.simplex_weights(2))==9
    assert len(set(f.simplex_weights(3)))==45
    for row in f.simplex_weights(3):
        assert sum(row)==1 and all(w>=0 and w*8==int(w*8) for w in row)
    with pytest.raises(ValueError):f.simplex_weights(4)


def test_prediction_identity_numeric_not_parquet_path(tmp_path):
    first=component(tmp_path,'one');second=component(tmp_path,'two')
    assert f.prediction_identity(first)==f.prediction_identity(second)
    with pytest.raises(FileExistsError):
        f.save_prediction(pd.read_parquet(first['prediction']['path']),first['prediction']['path'])


def test_three_prospective_rules_and_dedup(tmp_path,monkeypatch):
    monkeypatch.setattr(f.p,'rank',lambda m:(m['rank'],m['official']['mse'],m['official']['mse']))
    ref=component(tmp_path,'reference','reference',450.)
    rows=[component(tmp_path,'a','tree',401.,10.),component(tmp_path,'b','tree',402.,20.),
          component(tmp_path,'c','neural',403.,30.),component(tmp_path,'duplicate','tree',401.,5.)]
    triples=f.proposed_triples(rows,ref)
    assert 1<=len(triples)<=3
    assert all(2<=len(t)<=3 and t[-1]==ref for t in triples)
    assert all(len({f.prediction_identity(c) for c in t})==len(t) for t in triples)
    assert sum(len(f.simplex_weights(len(t))) for t in triples)<=135


def test_mutation_exact72h_and_nonvacuous():
    index=pd.date_range('2020-01-01',periods=150,freq='h',tz='UTC')
    origin=index[2]
    data=pd.DataFrame({'x':np.arange(150,dtype=float),'timestamp_utc':index})
    poisoned,count=f.mutate(data,origin,'poison')
    assert count==72
    np.testing.assert_array_equal(poisoned.timestamp_utc,data.timestamp_utc)
    assert poisoned.loc[2,'x']==2 and poisoned.loc[75,'x']==75
    assert (poisoned.loc[3:74,'x']==987654).all()
    dropped,count=f.mutate(data,origin,'drop')
    assert len(dropped)==78 and count==72
    with pytest.raises(ValueError,match='vacuous'):f.mutate(data,index[-1],'poison')


def test_raw_rebuild_receives_mutation_before_derived(monkeypatch):
    from experiments.mse20_ch_dynamics import inputs
    index=pd.date_range('2020-01-01',periods=100,freq='h',tz='UTC')
    frames={k:pd.DataFrame({'x':np.ones(100)},index=index) for k in f.RAW+f.DERIVED}
    received=[]
    def rebuild(**raw):
        received.append(raw)
        return {'rebuilt':True}
    monkeypatch.setattr(inputs,'rebuild_frames',rebuild)
    changed,counts=f.alter_frames(frames,index[10],'poison','ch')
    assert changed=={'rebuilt':True} and counts=={'ch':72}
    assert received[0]['ch'].iloc[11,0]==987654
    assert received[0]['ace'] is frames['ace']
    assert frames['ch'].iloc[11,0]==1
    changed,counts=f.alter_frames(frames,index[10],'drop')
    assert set(counts)==set(f.DERIVED) and all(v==72 for v in counts.values())
    assert all(len(changed[k])==28 for k in f.DERIVED)
    assert changed['ch'] is frames['ch']


def _freeze_fixture(tmp_path,monkeypatch,partial=False):
    monkeypatch.setattr(f.p,'rank',lambda m:(m['rank'],m['official']['mse'],m['official']['mse']))
    monkeypatch.setattr(f.p,'verify_attempts',lambda *a,**k:[])
    monkeypatch.setattr(f.p,'verify_registration',lambda *a,**k:{})
    count={'scores':0}
    def deadline(run,stage,reserve=0):
        if partial and reserve>0 and count['scores']>=2:raise TimeoutError('fake interrupted triple')
        return 1e6
    monkeypatch.setattr(f.p,'deadline',deadline)
    c=component(tmp_path,'new',value=400.,mse=10.)
    ref=component(tmp_path,'ref','reference',value=450.,mse=30.)
    result=tmp_path/'attempt-results/a.json'
    f.p.write_new(result,{'status':'ok','details':{'fit_completed_utc':'2020-01-01T00:00:00Z',
                                               'score_completed_utc':'2020-01-01T00:00:01Z'}})
    c['attempt_result']=f.p.file_ref(result)
    monkeypatch.setattr(f,'trial_component',lambda path,run:c)
    monkeypatch.setattr(f,'reference_component',lambda paths:ref)
    origins=pd.date_range('2020-04-01',periods=3,freq='h',tz='UTC')
    class Benchmark:
        def __init__(self,run):self.paths={};self.origins=origins
        def score(self,prediction,stage):
            assert stage=='search'
            assert (tmp_path/'blends/triple0/grid.json').exists()
            count['scores']+=1
            return {'official':{'mse':1.},'rank':1.}
    monkeypatch.setattr(f.p,'Benchmark',Benchmark)
    for name in ('registered.json','authorization.json'):f.p.write_new(tmp_path/name,{})
    f.p.write_new(tmp_path/'runtime.json',{'search_stop_utc':'2030-01-01T00:00:00Z'})
    return c,count


def test_complete_grid_can_win_but_reference_only_corner_cannot(tmp_path,monkeypatch):
    _,count=_freeze_fixture(tmp_path,monkeypatch)
    recipe=f.freeze(tmp_path)
    selection=f.p.read_json(tmp_path/'selection.json')
    assert count['scores']==9 and selection['triples'][0]['status']=='complete'
    assert selection['chosen']['kind']=='blend'
    assert any(c['family']!='reference' for c in recipe['components'])
    assert sum(c['weight'] for c in recipe['components'])==1
    before=(tmp_path/'recipe.json').read_bytes()
    assert f.freeze(tmp_path)==recipe
    assert count['scores']==9 and (tmp_path/'recipe.json').read_bytes()==before


def test_partial_blend_is_diagnostic_never_winner(tmp_path,monkeypatch):
    _,count=_freeze_fixture(tmp_path,monkeypatch,partial=True)
    f.freeze(tmp_path)
    selection=f.p.read_json(tmp_path/'selection.json')
    assert count['scores']==2
    assert selection['triples'][0]['status']=='partial_diagnostic_only'
    assert selection['chosen']['kind']=='standalone'


def test_replay_persists24_snapshots_cpu_gpu_and_scores(tmp_path,monkeypatch):
    c=component(tmp_path,'new',value=400.)
    c['weight']=1.
    origins=pd.date_range('2020-04-01',periods=3,freq='h',tz='UTC')
    index=pd.date_range('2020-01-01',periods=2400,freq='h',tz='UTC')
    frames={k:pd.Series(np.ones(len(index)),index=index) for k in f.RAW+f.DERIVED}
    for k in f.RAW:frames[k]=frames[k].to_frame('x')
    for name in ('registered.json','runtime.json','authorization.json','selection.json','prefit-tests.json'):
        f.p.write_new(tmp_path/name,{})
    f.p.write_new(tmp_path/'regression-summary.json',{'passed':True,'files':{}})
    final=f.save_prediction(predictions(origins,400.),tmp_path/'final-predictions.parquet')
    f.p.write_new(tmp_path/'recipe.json',{'components':[c],'files':{},'prediction':final,'verification_budget_seconds':10.})
    monkeypatch.setattr(f.p,'verify_attempts',lambda *a,**k:[])
    monkeypatch.setattr(f.p,'verify_registration',lambda *a,**k:{})
    monkeypatch.setattr(f.p,'deadline',lambda *a,**k:1e6)
    monkeypatch.setattr(f.p,'load_frames',lambda end,run_dir:frames)
    monkeypatch.setattr(f,'predict',lambda component,frames,origins,device:predictions(origins,400.))
    monkeypatch.setattr(f,'training_provenance',lambda *a:{'recomputed':True})
    monkeypatch.setattr(f.p,'gpu_lease',lambda *a,**k:contextlib.nullcontext('cuda:0'))
    monkeypatch.setattr(f.p,'memory_admission',lambda *a,**k:contextlib.nullcontext())
    from experiments.mse20_ch_dynamics import inputs,tree
    monkeypatch.setattr(tree,'estimate_memory',lambda *a,**k:{'gpu_bytes':100})
    monkeypatch.setattr(inputs,'rebuild_frames',lambda **raw:dict(frames,**raw))
    class Benchmark:
        def __init__(self,run):self.origins=origins;self.contract={'files':{}}
        def score(self,prediction,stage):
            assert stage=='final'
            return {'status':'fail','constant':'same both calls'}
    monkeypatch.setattr(f.p,'Benchmark',Benchmark)
    manifest=f.replay(tmp_path)
    proof=f.p.read_json(tmp_path/'replay-proof.json')
    record=proof['components'][0]
    assert len(record['checks'])==24
    assert sum(c['kind']=='raw' for c in record['checks'])==18
    assert sum(c['kind']=='derived' for c in record['checks'])==6
    assert len(record['subset_checks'])==3
    for check in record['checks']:
        assert len(pd.read_parquet(check['baseline']['path']))==72
        assert len(pd.read_parquet(check['altered']['path']))==72
        assert all(count==72 for count in check['mutation_counts'].values())
    assert set(('final_score_1','final_score_2'))<=set(manifest['files'])
    with np.load(proof['source_inventory']['path']) as inventory:
        assert set(inventory.files)==set(f.RAW+f.DERIVED)


def test_difference_rejects_changed_values_or_keys():
    origins=pd.date_range('2020-01-01',periods=1,tz='UTC')
    first=predictions(origins,400.)
    with pytest.raises(ValueError,match='tolerance'):f.difference(first,predictions(origins,401.))
    with pytest.raises(ValueError,match='keys'):f.difference(first,first.iloc[:-1])


def test_restart_uses_complete_grid_without_rescoring(tmp_path,monkeypatch):
    _,count=_freeze_fixture(tmp_path,monkeypatch)
    original=f.p.write_new
    crash={'enabled':True}
    def injected(path,value):
        if Path(path).name=='selection.json' and crash['enabled']:
            raise RuntimeError('synthetic crash before selection')
        return original(path,value)
    monkeypatch.setattr(f.p,'write_new',injected)
    with pytest.raises(RuntimeError,match='synthetic crash'):f.freeze(tmp_path)
    assert count['scores']==9
    crash['enabled']=False
    recipe=f.freeze(tmp_path)
    assert count['scores']==9 and recipe['components']


def test_score_timeout_preserves_partial_diagnostic_grid(tmp_path,monkeypatch):
    _,count=_freeze_fixture(tmp_path,monkeypatch)
    cls=f.p.Benchmark
    previous=cls.score
    def timeout(self,prediction,stage):
        if count['scores']>=1:raise TimeoutError('score deadline')
        return previous(self,prediction,stage)
    monkeypatch.setattr(cls,'score',timeout)
    recipe=f.freeze(tmp_path)
    selection=f.p.read_json(tmp_path/'selection.json')
    assert selection['triples'][0]['status']=='partial_diagnostic_only'
    assert selection['chosen']['kind']=='standalone'
    assert recipe['components'][0]['weight']==1.


def test_pilot_returns_bound_path_and_actual_snapshots(tmp_path,monkeypatch):
    c=component(tmp_path,'new')
    origins=pd.date_range('2020-04-01',periods=3,freq='h',tz='UTC')
    monkeypatch.setattr(f.p,'deadline',lambda *a,**k:1e6)
    monkeypatch.setattr(f.p,'verify_registration',lambda *a,**k:{})
    monkeypatch.setattr(f,'predict',lambda component,frames,origins,device:predictions(origins,400.))
    monkeypatch.setattr(f,'alter_frames',lambda *a,**k:({'changed':True},{'raw':72}))
    monkeypatch.setattr(f,'training_provenance',lambda *a:{'actual_arrays':[1,2,3]})
    path=f.pilot(c,{},origins,tmp_path)
    assert isinstance(path,Path)
    record=f.p.read_json(path)
    assert record['metadata']==c['metadata'] and record['prediction']==c['prediction']
    assert record['mutation_counts']=={'raw':72}
    assert record['cpu_full_seconds']>0 and record['raw_rebuild_seconds']>0
    assert record['estimated_final_seconds']>=720
    for ref in record['files'].values():f.p.verify_ref(ref)
    with pytest.raises(FileExistsError):f.pilot(c,{},origins,tmp_path)


def test_full45_rows_without_summary_recovers_complete_no_scores(tmp_path,monkeypatch):
    first,count=_freeze_fixture(tmp_path,monkeypatch)
    second=component(tmp_path,'second','neural',value=401.,mse=20.)
    path=tmp_path/'attempt-results/b.json'
    f.p.write_new(path,f.p.read_json(first['attempt_result']['path']))
    second['attempt_result']=f.p.file_ref(path)
    monkeypatch.setattr(f,'trial_component',lambda path,run:first if path.name=='a.json' else second)
    original=f.p.write_new
    crash={'enabled':True}
    def injected(path,value):
        if Path(path).name=='summary.json' and crash['enabled']:
            raise RuntimeError('crash after all45 scored rows')
        return original(path,value)
    monkeypatch.setattr(f.p,'write_new',injected)
    with pytest.raises(RuntimeError,match='all45'):f.freeze(tmp_path)
    assert count['scores']==45
    crash['enabled']=False
    f.freeze(tmp_path)
    assert count['scores']==45
    assert f.p.read_json(tmp_path/'blends/triple0/summary.json')['status']=='complete'


@pytest.mark.parametrize('boundary',['selection','prediction'])
def test_completed_selection_prediction_resume_without_optimization(tmp_path,monkeypatch,boundary):
    _,count=_freeze_fixture(tmp_path,monkeypatch)
    crash={'enabled':True}
    original_save=f.save_prediction
    original_write=f.p.write_new
    def save(frame,path):
        if boundary=='selection' and Path(path).name=='final-predictions.parquet' and crash['enabled']:
            raise RuntimeError('crash before final prediction')
        return original_save(frame,path)
    def write(path,value):
        if boundary=='prediction' and Path(path).name=='recipe.json' and crash['enabled']:
            raise RuntimeError('crash before frozen recipe')
        return original_write(path,value)
    monkeypatch.setattr(f,'save_prediction',save)
    monkeypatch.setattr(f.p,'write_new',write)
    with pytest.raises(RuntimeError,match='crash before'):f.freeze(tmp_path)
    saved_selection=(tmp_path/'selection.json').read_bytes()
    saved_prediction=(tmp_path/'final-predictions.parquet').read_bytes() if boundary=='prediction' else None
    crash['enabled']=False
    recipe=f.freeze(tmp_path)
    assert recipe and count['scores']==9
    assert (tmp_path/'selection.json').read_bytes()==saved_selection
    if saved_prediction is not None:assert (tmp_path/'final-predictions.parquet').read_bytes()==saved_prediction


@pytest.mark.parametrize('tamper',['clock','weight','hash'])
def test_recovery_rejects_invalid_scored_row(tmp_path,monkeypatch,tamper):
    _,count=_freeze_fixture(tmp_path,monkeypatch)
    original=f.p.write_new
    def crash(path,value):
        if Path(path).name=='summary.json':raise RuntimeError('synthetic summary crash')
        return original(path,value)
    monkeypatch.setattr(f.p,'write_new',crash)
    with pytest.raises(RuntimeError):f.freeze(tmp_path)
    path=tmp_path/'blends/triple0/00.json'
    row=f.p.read_json(path)
    if tamper=='clock':row['completed_utc']='2040-01-01T00:00:00Z'
    elif tamper=='weight':row['weights']=[.5,.5]
    else:row['prediction']['sha256']='0'*64
    path.write_text(json.dumps(row))
    monkeypatch.setattr(f.p,'write_new',original)
    with pytest.raises((ValueError,AssertionError)):f.freeze(tmp_path)
    assert count['scores']==9


def test_partial_grid_stays_partial_on_recovery(tmp_path,monkeypatch):
    _,count=_freeze_fixture(tmp_path,monkeypatch,partial=True)
    original=f.p.write_new
    crash={'enabled':True}
    def injected(path,value):
        if Path(path).name=='selection.json' and crash['enabled']:raise RuntimeError('selection crash')
        return original(path,value)
    monkeypatch.setattr(f.p,'write_new',injected)
    with pytest.raises(RuntimeError):f.freeze(tmp_path)
    crash['enabled']=False
    recipe=f.freeze(tmp_path)
    assert count['scores']==2 and recipe['components'][0]['weight']==1.
    assert f.p.read_json(tmp_path/'blends/triple0/summary.json')['status']=='partial_diagnostic_only'
