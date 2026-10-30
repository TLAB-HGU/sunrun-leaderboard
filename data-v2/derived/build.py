"""Isolated, provenance-checked v2 feature generation. Never runs model fits."""
from __future__ import annotations
import argparse, ast, hashlib, importlib.util, json, os, sys, traceback
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
LB=HERE.parents[1]/'leaderboard'
STORE_ROOT = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store')
V2=STORE_ROOT/'v2'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def load(name):
    path=HERE/'vendor'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m

def atomic_json(path,value):
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(path)

def ace(job,end,out):
    files=sorted((V2/'ace-snapshot').glob('year=*/part.parquet'))
    if not files:raise FileNotFoundError('v2 ACE snapshot absent')
    raw=pd.concat([pd.read_parquet(p) for p in files],ignore_index=True)
    raw.timestamp_utc=pd.to_datetime(raw.timestamp_utc,utc=True)
    raw=raw[raw.timestamp_utc<=end].drop_duplicates('timestamp_utc',keep='last').sort_values('timestamp_utc')
    hashes={str(p):sha(p) for p in files}
    if job=='alfven':
        m=load('alfvenicity'); m.SOURCE_DIR=V2/'ace-snapshot';m.CACHE_DIR=out/'cache';m.SOURCE_MANIFEST=out/'source.json'
        atomic_json(m.SOURCE_MANIFEST,{'source_hashes':hashes,'local_observation_range_utc':[str(raw.timestamp_utc.min()),str(raw.timestamp_utc.max())]})
        obj=m.AlfvenicityFeatureStore(end); table=obj.table
        atomic_json(out/'poison.json',obj.poison(table.index[-1]))
    else:
        m=load('aceturb');table=m.build_turb_features(raw.set_index('timestamp_utc'));table=table[table.index<=end]
    table.index.name='origin_last_input_utc';table.reset_index().to_parquet(out/'features.parquet',index=False)
    return {'source_hashes':hashes,'source_observation_first':str(raw.timestamp_utc.min()),'source_observation_last':str(raw.timestamp_utc.max()),'rows':len(table),'feature_first':str(table.index.min()),'feature_last':str(table.index.max()),'availability_rule':'ACE observation + 2 hours <= origin'}

def main():
    p=argparse.ArgumentParser();p.add_argument('job',choices=['alfven','aceturb']);p.add_argument('--end',default='now');p.add_argument('--output',type=Path);a=p.parse_args()
    end=pd.Timestamp.now(tz='UTC').floor('h') if a.end=='now' else pd.Timestamp(a.end)
    if end.tz is None:end=end.tz_localize('UTC')
    out=a.output or V2/a.job;out.mkdir(parents=True,exist_ok=True)
    result={'dataset':a.job,'requested_end':str(end),'started_at':str(pd.Timestamp.now(tz='UTC')),'builder':str(Path(__file__)),'builder_sha256':sha(__file__),'vendor_hashes':{('alfvenicity.py' if a.job=='alfven' else 'aceturb.py'):sha(HERE/'vendor'/('alfvenicity.py' if a.job=='alfven' else 'aceturb.py'))}}
    try:
        result.update(ace(a.job,end,out));result.update(status='completed',returncode=0,files={'features.parquet':sha(out/'features.parquet')})
    except Exception as e:
        result.update(status='failed',returncode=1,error=f'{type(e).__name__}: {e}');traceback.print_exc()
    result['finished_at']=str(pd.Timestamp.now(tz='UTC'));atomic_json(out/'MANIFEST.json',result);print(json.dumps(result));return result['returncode']
if __name__=='__main__':raise SystemExit(main())
