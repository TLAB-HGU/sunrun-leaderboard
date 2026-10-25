"""Recomputed temporal perturbation receipts; source availability needs separate review."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import gate


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _delta(a,b):
    gate.require(isinstance(a,pd.DataFrame) and isinstance(b,pd.DataFrame),'builder must return DataFrames')
    gate.require(a.index.equals(b.index) and a.columns.equals(b.columns),'probe output alignment changed')
    x,y=a.to_numpy(float),b.to_numpy(float)
    gate.require(np.isfinite(x).all() and np.isfinite(y).all() and x.size>0,'nonfinite/empty probe output')
    return float(np.max(np.abs(x-y)))


def probe(builder, original, future_perturbed, past_perturbed, *, origin, columns):
    origin=pd.Timestamp(origin)
    gate.require(str(origin.tz)=='UTC','probe origin must be UTC')
    gate.require(isinstance(original.index,pd.DatetimeIndex) and str(original.index.tz)=='UTC'
                 and original.index.is_unique,'raw inputs require unique UTC time index')
    counts=[]
    for modified,future in ((future_perturbed,True),(past_perturbed,False)):
        gate.require(original.index.equals(modified.index) and original.columns.equals(modified.columns),'raw probe keys/columns changed')
        a,b=original.to_numpy(float),modified.to_numpy(float)
        gate.require(np.isfinite(a).all() and np.isfinite(b).all(),'nonfinite raw probe')
        changed=np.any(a!=b,axis=1)
        allowed=original.index>origin if future else original.index<=origin
        gate.require(changed.any() and not (changed & ~allowed).any(),'probe lacks actual correctly timed mutations')
        counts.append(int(changed.sum()))
    base=builder(original.copy(deep=True),origin)
    gate.require(list(base.columns)==list(columns),'probe feature columns mismatch')
    gate.require(_delta(base,builder(original.copy(deep=True),origin))==0,'nondeterministic feature builder')
    future=builder(future_perturbed.copy(deep=True),origin)
    past=builder(past_perturbed.copy(deep=True),origin)
    future_delta,past_delta=_delta(base,future),_delta(base,past)
    gate.require(future_delta==0 and past_delta>0,'future leakage or vacuous past sensitivity')
    return dict(origin=origin.isoformat(),future_rows_changed=counts[0],past_rows_changed=counts[1],
                future_delta=future_delta,past_delta=past_delta,
                raw_sha256={name:hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes()).hexdigest()
                            for name,frame in [('original',original),('future',future_perturbed),('past',past_perturbed)]})


def write_receipt(path, inputs, builder_code_paths, groups):
    gate.require('poison' not in inputs,'receipt inputs must exclude poison reference')
    receipt=dict(version=1,artifact_sha256=gate.digest(inputs),
                 source_availability_sha256=gate.digest(inputs['source_availability']),
                 builder_code_sha256={str(Path(p).resolve()):file_hash(p) for p in builder_code_paths},groups=groups)
    path=Path(path).resolve();gate.atomic(path,receipt)
    return dict(path=str(path),sha256=file_hash(path))


def validate_poison(inputs,recipe):
    gate.require(isinstance(inputs.get('poison'),dict),'actual future/past poison receipt required')
    ref=inputs['poison'];gate.keys(ref,'path sha256');gate.sha(ref['sha256'])
    gate.require(file_hash(ref['path'])==ref['sha256'],'poison receipt hash mismatch')
    receipt=gate.read(ref['path'])
    gate.keys(receipt,'version artifact_sha256 source_availability_sha256 builder_code_sha256 groups')
    gate.require(receipt['version']==1,'unsupported poison receipt')
    bound={k:v for k,v in inputs.items() if k!='poison'}
    gate.require(receipt['artifact_sha256']==gate.digest(bound),'poison input identity mismatch')
    gate.require(receipt['source_availability_sha256']==gate.digest(inputs['source_availability']),'poison source identity mismatch')
    codes=receipt['builder_code_sha256']
    gate.require(isinstance(codes,dict) and codes,'builder code hashes required')
    for path,sha in codes.items():
        gate.sha(sha)
        gate.require(Path(path).is_absolute() and Path(path).suffix=='.py' and file_hash(path)==sha,'builder code identity mismatch')
    gate.require(isinstance(receipt['groups'],dict),'poison groups must be a mapping')
    expected={'base111':inputs['features']}
    for key,value in recipe.items():
        if key.startswith('input_') and value is True:
            name=key[6:]
            gate.require(name in inputs.get('blocks',{}),'enabled block missing')
            expected[name]=inputs['blocks'][name]
    for name,columns in expected.items():
        group=receipt['groups'].get(name)
        gate.require(isinstance(group,dict),'missing poison group: '+name)
        gate.keys(group,'columns cases')
        gate.require(group['columns']==columns and columns and len(set(columns))==len(columns),'poison feature columns mismatch')
        gate.require(isinstance(group['cases'],list) and group['cases'],'actual probe cases required')
        for case in group['cases']:
            gate.keys(case,'origin future_rows_changed past_rows_changed future_delta past_delta raw_sha256')
            gate.require(str(pd.Timestamp(case['origin']).tz)=='UTC','poison origin must be UTC')
            for key in ('future_rows_changed','past_rows_changed'):
                gate.require(type(case[key]) is int and case[key]>0,'actual mutated rows required')
            for key in ('future_delta','past_delta'):
                gate.require(type(case[key]) in (float,int) and np.isfinite(case[key]),'finite probe delta required')
            gate.require(case['future_delta']==0 and case['past_delta']>0,'future leakage or vacuous probe')
            raw=case['raw_sha256'];gate.keys(raw,'original future past')
            for sha in raw.values():gate.sha(sha)
            gate.require(len(set(raw.values()))==3,'raw probe inputs did not change')
    return receipt
