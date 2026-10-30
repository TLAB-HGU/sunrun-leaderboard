"""Deterministic finite simplex, deduplication and issuance mutation checks."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.mse20_adaptive import finalize as f


@pytest.mark.parametrize('count,expected',[(1,1),(2,9),(3,45)])
def test_simplex_exact_finite_caps(count,expected):
    weights=f.simplex_weights(count)
    assert len(weights)==expected==len(set(weights))
    assert all(sum(row)==1 and all(0<=v<=1 and v*8==int(v*8) for v in row) for row in weights)
    for i in range(count):
        assert tuple(float(j==i) for j in range(count)) in weights


def test_other_triples_forbidden():
    with pytest.raises(ValueError,match='one fixed triple'):f.simplex_weights(4)


def test_pick_deduplicates_model_content_and_config_ignoring_id(monkeypatch):
    monkeypatch.setattr(f.p,'rank',lambda m:(m['rank'],m['mse']))
    monkeypatch.setattr(f,'trial_component',lambda r:r['component'])
    def row(name,rank,sha,param):
        return {'family':'tree','config':{'id':name,'param':param},'config_sha256':name,
                'metrics':{'rank':rank,'mse':rank},
                'component':{'key':name,'model_files':[{'sha256':sha}]}}
    reference={'key':'reference','model_files':[{'sha256':'baseline'}]}
    rows=[row('baseline_duplicate',0,'baseline',0),row('a',1,'a',1),
          row('b',2,'b',1),row('c',3,'a',2),row('d',4,'d',3),row('e',5,'e',4)]
    chosen=f.pick_components(rows,reference)
    assert [c['key'] for c in chosen]==['a','d','reference']
    assert chosen==f.pick_components(list(reversed(rows)),reference)


def test_undefined_and_event_regressed_models_rank_after_feasible():
    cells={c:{'mse':5000.} for c in f.ev.old.BASELINE_CELLS}
    def metric(p,r,mse):
        return {'status':'unmet','events':{'precision':p,'recall':r},
                'official':{'mse':mse,'by_regime':cells}}
    assert f.p.rank(metric(1.,1/818,5000))<f.p.rank(metric(.99,.1,1000))
    assert f.p.rank(metric(1.,1/818,5000))<f.p.rank(metric(None,0.,1000))


@pytest.mark.parametrize('mode',['poison','drop'])
def test_every_future_tabular_row_is_changed_without_modifying_past_or_original(mode):
    idx=pd.date_range('2026-06-01',periods=5,freq='h',tz='UTC');origin=idx[2]
    frames={'speed':pd.Series(np.arange(5,dtype=float),index=idx),
            'observed':pd.Series(True,index=idx),
            'base':pd.DataFrame({'a':np.arange(5,dtype=float)},index=idx),
            'ace':pd.DataFrame({'timestamp_utc':idx,'b':np.arange(5,dtype=float),'mask':True}),
            'other':'immutable'}
    before=copy.deepcopy(frames);changed=f.future_frames(frames,origin,mode)
    pd.testing.assert_series_equal(frames['speed'],before['speed'])
    pd.testing.assert_frame_equal(frames['ace'],before['ace'])
    pd.testing.assert_series_equal(changed['speed'].loc[:origin],before['speed'].loc[:origin])
    pd.testing.assert_frame_equal(changed['ace'].iloc[:3],before['ace'].iloc[:3])
    if mode=='drop':
        assert len(changed['speed'])==len(changed['observed'])==len(changed['ace'])==3
    else:
        assert changed['speed'].iloc[-1]==changed['base'].iloc[-1,0]==changed['ace'].b.iloc[-1]==987654
        assert not changed['observed'].iloc[-1] and not changed['ace']['mask'].iloc[-1]


def test_prediction_delta_checks_keys_and_finite():
    left=pd.DataFrame({'origin_last_input_utc':['x'],'horizon_hours':[1],'pred_kms':[400.]})
    right=left.copy();right.pred_kms+=.0001
    assert f.max_difference(left,right)<1e-3
    right.pred_kms+=.1
    with pytest.raises(ValueError,match='difference'):f.max_difference(left,right)
    right=left.copy();right.horizon_hours=2
    with pytest.raises(ValueError,match='keys'):f.max_difference(left,right)


def test_existing_freeze_is_not_overwritten(tmp_path,monkeypatch):
    (tmp_path/'recipe.json').write_text('{}')
    monkeypatch.setattr(f.p,'require_evaluator_ready',lambda _:None)
    monkeypatch.setattr(f.p,'deadline',lambda *args:None)
    with pytest.raises(FileExistsError,match='already frozen'):f.freeze(tmp_path)
    assert json.loads((tmp_path/'recipe.json').read_text())=={}


def test_hash_precedes_scoring_and_frozen_bytes_cannot_change(tmp_path,monkeypatch):
    path=tmp_path/'pred.parquet';pd.DataFrame({'pred_kms':[400.]}).to_parquet(path)
    events=[];original=f.p.file_ref
    def reference(path):events.append('hash');return original(path)
    monkeypatch.setattr(f.p,'file_ref',reference)
    class Benchmark:
        def score(self,frame):events.append('score');return {'mse':0.,'by_horizon':{1:0.}}
    ref,metrics=f.score_frozen(path,Benchmark())
    assert events==['hash','score'] and metrics=={'mse':0.,'by_horizon':{'1':0.}}
    class MutatingBenchmark:
        def score(self,frame):path.write_bytes(b'tampered');return {}
    with pytest.raises(ValueError,match='stale artifact'):f.score_frozen(path,MutatingBenchmark())


def test_future_counts_fail_closed_on_vacuous_last_origin():
    idx=pd.date_range('2026-06-01',periods=3,freq='h',tz='UTC')
    frames={key:pd.Series(400.,index=idx) for key in ('speed','observed','base','physics','quality')}
    assert all(v==1 for v in f.future_counts(frames,idx[1]).values())
    with pytest.raises(ValueError,match='vacuous'):f.future_counts(frames,idx[-1])
