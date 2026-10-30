"""Temporal coverage, causal sources, and deterministic chunk identity regressions."""
import json

import numpy as np
import pandas as pd
import pytest

from experiments.vjepa_120h_fusion.temporal import (
    CLIP_OFFSETS, HOUR_NS, NAT, all_quality, coverage_manifest,
    deduplicate_clips, origins_ns, plan_sources,
)
from experiments.vjepa_120h_fusion.extract import atomic_npz, ensure_npz, read_chunk, write_json
from experiments.vjepa_frozen_fusion.encoder import sha256


def inventory():
    times = pd.date_range('2025-01-01',periods=150,freq='h',tz='UTC')
    return pd.DataFrame({'slot':times,'obs_end':times-pd.Timedelta(minutes=5),
                         'available':True,'path':[f'frame{i}' for i in range(150)]})


def test_exact_120_hour_coverage_and_overlap():
    manifest = coverage_manifest()
    assert manifest['clip_starts'] == [0,15,30,45,60,75,90,104]
    assert np.array_equal(np.unique(CLIP_OFFSETS),np.arange(120))
    assert sum(manifest['slot_multiplicities']) == 128
    assert np.all(np.diff(CLIP_OFFSETS,axis=1)==1)


def test_ns_us_equivalence():
    dates = pd.date_range('2025-01-06',periods=3,freq='h',tz='UTC')
    assert np.array_equal(origins_ns(dates.as_unit('us')),origins_ns(dates.as_unit('ns')))
    with pytest.raises(ValueError,match='nanoseconds'):
        origins_ns(dates.as_unit('us').asi8)


def test_source_request_origin_order_and_bounded_fill():
    data = inventory()
    origin = origins_ns([data.slot.iloc[-1]])
    data.loc[145:,'available']=False
    _, ids, ends, age, filled = plan_sources(data,origin)
    requested = origin[:,None]-np.arange(119,-1,-1)*HOUR_NS
    assert ids.shape == (1,120)
    assert np.all(ends <= requested)
    assert np.all(requested <= origin[:,None])
    assert np.all(age[ids>=0] <= 6)
    assert filled[0,-1]
    data.loc[140:,'available']=False
    _,ids,ends,age,_=plan_sources(data,origin)
    assert ids[0,-1] == -1 and ends[0,-1] == NAT and age[0,-1] == 7


def test_future_perturbation_and_deletion_invariance():
    data=inventory()
    origin=origins_ns([data.slot.iloc[125]])
    baseline=plan_sources(data,origin)[1:]
    mutated=data.copy()
    mutated.loc[126:,'path']='FUTURE-PERTURBED'
    variants=[mutated, data.iloc[:126].copy()]
    for variant in variants:
        for before,after in zip(baseline,plan_sources(variant,origin)[1:]):
            assert np.array_equal(before,after)


def test_empty_inventory_and_invalid_observation():
    data=inventory()
    origin=origins_ns([data.slot.iloc[-1]])
    data['available']=False
    _,ids,ends,age,filled=plan_sources(data,origin)
    assert np.all(ids==-1) and np.all(ends==NAT) and not filled.any()
    data.loc[0,'available']=True
    data.loc[0,'obs_end']=data.slot.iloc[0]+pd.Timedelta(seconds=1)
    with pytest.raises(ValueError,match='invalid source'):
        plan_sources(data,origin)


def test_dedup_exact_window_and_quality():
    data=inventory()
    origins=origins_ns(data.slot.iloc[125:])
    _,ids,ends,age,filled=plan_sources(data,origins)
    plan=deduplicate_clips(origins,ids,ends,age,filled)
    assert len(plan['clip_ends']) < len(origins)*8
    assert np.all(plan['clip_ends'][plan['origin_to_clip']] <= origins[:,None])
    for clip,offsets in enumerate(CLIP_OFFSETS):
        assert np.array_equal(plan['source_ids'][plan['origin_to_clip'][:,clip]],ids[:,offsets])
    quality=all_quality(ids,age,filled,np.ones(ids.shape))
    assert quality.shape==(25,45)
    assert np.isfinite(quality).all()
    damaged=ids.copy(); damaged[15,0]=-1
    with pytest.raises(ValueError,match='conflicting'):
        deduplicate_clips(origins,damaged,ends,age,filled)


def test_chunk_identity_owner_order_and_corruption(tmp_path):
    ends=np.arange(300,dtype=np.int64)*HOUR_NS
    path=tmp_path/'000001.npz'
    values=np.ones((44,3840),np.float32)
    atomic_npz(path,clip_ends=ends[256:],embedding=values)
    metadata={'identity':'known','sha256':sha256(path),'chunk_id':1,'gpu':1}
    write_json(path.with_suffix('.json'),metadata)
    assert np.array_equal(read_chunk(path,'known',ends,1)[0],values)
    with pytest.raises(ValueError,match='identity mismatch'):
        read_chunk(path,'wrong',ends,1)
    metadata['gpu']=0; write_json(path.with_suffix('.json'),metadata)
    with pytest.raises(ValueError,match='owner mismatch'):
        read_chunk(path,'known',ends,1)
    metadata['gpu']=1
    atomic_npz(path,clip_ends=ends[256:][::-1],embedding=values)
    metadata['sha256']=sha256(path); write_json(path.with_suffix('.json'),metadata)
    with pytest.raises(ValueError,match='time identity'):
        read_chunk(path,'known',ends,1)
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='identity mismatch'):
        read_chunk(path,'known',ends,1)


def test_preparation_recovers_missing_artifact_and_rejects_drift(tmp_path):
    path = tmp_path/'plan.npz'
    values = np.arange(120)
    ensure_npz(path, offsets=values)
    original = path.read_bytes()
    ensure_npz(path, offsets=values)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match='prepared artifact drift'):
        ensure_npz(path, offsets=values + 1)
    path.unlink()
    ensure_npz(path, offsets=values)
    with np.load(path) as z:
        np.testing.assert_array_equal(z['offsets'], values)


def test_multigpu_merge_preserves_chronological_concatenation(tmp_path, monkeypatch):
    from experiments.vjepa_120h_fusion import extract
    ends=np.arange(300,dtype=np.int64)*HOUR_NS
    origin_to_clip=np.array([[0,1,2,3,256,257,298,299],[299,298,257,256,3,2,1,0]])
    plan={'clip_ends':ends,'origin_to_clip':origin_to_clip,
          'source_ids':np.zeros((300,16),np.int64)}
    monkeypatch.setattr(extract,'prepare',lambda run:(plan,{},'known'))
    folder=tmp_path/'cache/hourly-clip-embeddings'; folder.mkdir(parents=True)
    # Write GPU1 first: merge order must come from deterministic chunk ids.
    for chunk_id in [1,0]:
        lo,hi=extract.chunk_bounds(300,chunk_id)
        path=folder/f'{chunk_id:06d}.npz'
        atomic_npz(path,clip_ends=ends[lo:hi],embedding=np.repeat(np.arange(lo,hi,dtype=np.float32)[:,None],3840,axis=1))
        write_json(path.with_suffix('.json'),{'identity':'known','sha256':sha256(path),
            'chunk_id':chunk_id,'gpu':chunk_id%2,'gpu_seconds':1.})
        write_json(tmp_path/f'extraction-worker-gpu{chunk_id}.json',{'stage':'complete'})
    write_json(tmp_path/'coverage-manifest.json',coverage_manifest())
    atomic_npz(tmp_path/'embedding_metadata.npz',origins=np.array([0,1]))
    extract.merge(tmp_path)
    output=np.load(tmp_path/'embeddings.npy',mmap_mode='r')
    assert output.shape==(2,30720)
    assert np.array_equal(output.reshape(2,8,3840)[:,:,0],origin_to_clip)
    assert json.loads((tmp_path/'extraction.json').read_text())['gpu_seconds']==2.
