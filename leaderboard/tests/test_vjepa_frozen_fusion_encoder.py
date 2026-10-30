"""Production timestamp/causality and fixed spatial token pooling regressions."""
import numpy as np
import pandas as pd
import torch

from experiments.vjepa_frozen_fusion.extract import plan_sources
from experiments.vjepa_frozen_fusion.encoder import pool_tokens
from experiments.vjepa_frozen_fusion.images import select_clip


def inventory():
    slots=pd.date_range('2025-07-01',periods=80,freq='h',tz='UTC')
    return pd.DataFrame({'slot':slots,'obs_end':slots-pd.Timedelta(seconds=11),
                         'path':[f'{i}.nc' for i in range(len(slots))], 'available':True})


def test_parquet_microseconds_match_reference_selector(tmp_path):
    frame=inventory();frame['slot']=frame.slot.dt.as_unit('us');frame['obs_end']=frame.obs_end.dt.as_unit('us')
    path=tmp_path/'inventory.parquet';frame.to_parquet(path,index=False);frame=pd.read_parquet(path)
    origin=pd.Timestamp('2025-07-03T12:00Z')
    rows,ids,ends,_,_=plan_sources(frame,np.array([origin.value]))
    selected=select_clip(frame,origin)
    assert (ids>=0).all()
    assert [rows.iloc[i].path for i in ids[0]]==selected.path.tolist()
    assert np.array_equal(ends[0],pd.DatetimeIndex(selected.obs_end).as_unit('ns').asi8)
    assert (ends<=origin.value).all()


def test_production_future_delete_poison_invariance():
    frame=inventory();origin=pd.Timestamp('2025-07-03T00:00Z')
    original=plan_sources(frame,np.array([origin.value]))
    poison=frame.copy();poison.loc[poison.slot>origin,'path']='future-poison';poison.loc[poison.slot>origin,'obs_end']=pd.Timestamp('2100-01-01',tz='UTC')
    changed=plan_sources(poison,np.array([origin.value]));deleted=plan_sources(frame[frame.slot<=origin],np.array([origin.value]))
    for got in [changed,deleted]:
        for i in range(1,5):assert np.array_equal(original[i],got[i])


def test_production_bounded_age_and_missing():
    frame=inventory().iloc[:1];origin=pd.Timestamp('2025-07-02T00:00Z')
    _,ids,ends,age,filled=plan_sources(frame,np.array([origin.value]))
    assert ids[0,-1]==-1 and ends[0,-1]==np.iinfo(np.int64).min and age[0,-1]==7 and not filled[0,-1]
    assert (age[ids>=0]<=6).all()


def test_pool_preserves_left_right_and_temporal_mean():
    tokens=torch.zeros(1,4608,768);grid=tokens.reshape(1,8,24,24,768);grid[:,:,:,:12,:]=1;grid[:,:,:,12:,:]=3
    pools=pool_tokens(tokens).reshape(1,5,768)
    assert torch.allclose(pools[:,0],torch.full((1,768),2.))
    assert torch.allclose(pools[:,3],torch.ones(1,768))
    assert torch.allclose(pools[:,4],torch.full((1,768),3.))
    assert torch.equal(pool_tokens(grid[:,:1].reshape(1,576,768)),pool_tokens(tokens))
