"""Refresh S2 image embeddings with existing frozen AE weights and normalization."""
import argparse,json,sys,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from build import HERE,LB,V2,sha,load,atomic_json
from dino import lease

STORE_ROOT = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store')


OLD=STORE_ROOT/'mambats-s2-source-cache/6dd5b1e04b52cfd57bfda07875bdafc5c2c388747579a3f442007d4c8e6bbe84'

def main():
 p=argparse.ArgumentParser();p.add_argument('--end',default='now');p.add_argument('--fe171-meta',type=Path,default=V2/'fe171/metadata-full.parquet');p.add_argument('--output',type=Path,default=V2/'mambats-source');a=p.parse_args()
 end=pd.Timestamp.now(tz='UTC') if a.end=='now' else pd.Timestamp(a.end)
 if end.tz is None:end=end.tz_localize('UTC')
 out=a.output;out.mkdir(parents=True,exist_ok=True);result={'dataset':'mambats-source','as_of':str(pd.Timestamp.now(tz='UTC')),'requested_end':str(end),'builder_sha256':sha(__file__),'status':'running'};atomic_json(out/'MANIFEST.json',result)
 try:
  import torch
  torch.set_num_threads(2);m=load('mambats_source');old=json.loads((OLD/'cache_manifest.json').read_text());kernel=sha(HERE/'vendor/mambats_source.py')
  if kernel!=old['source_code_sha256']:raise ValueError('source preprocessing code differs from frozen AE cache')
  for name,item in old['artifacts'].items():
   if sha(OLD/name)!=item['sha256']:raise ValueError(f'old cache artifact hash mismatch: {name}')
  rows=[json.loads(f.read_text()) for f in (V2/'gong-derived/receipts').glob('*.json')];atomic_json(out/'gong-source.json',{'maps':rows})
  m.SUVI_META=a.fe171_meta;m.GONG_MANIFEST=out/'gong-source.json'
  suvi,gong,identity,counts=m._load_records(end);result.update(metadata_identity=identity,metadata_counts=counts,kernel_sha256=kernel,source_manifest_sha256=sha(out/'gong-source.json'),fe171_metadata_sha256=sha(a.fe171_meta),frozen_manifest_sha256=sha(OLD/'cache_manifest.json'))
  groups={'suvi':suvi,'gong':gong};all_records={};reused={};new={};decode_errors=[]
  for name,records in groups.items():
   previous={r['local_sha256']:i for i,r in enumerate(old['records'][name])};oldimages=np.load(OLD/f'{name}_images.npy',mmap_mode='r');images=np.lib.format.open_memmap(out/f'pending-{name}_images.npy',mode='w+',dtype=np.float16,shape=(len(records),2,m.IMAGE_H,m.IMAGE_W));reused[name]=0;new[name]=0;updated=[]
   for i,r in enumerate(records):
    digest=sha(r['path'])
    if digest in previous:images[len(updated)]=oldimages[previous[digest]];r=dict(r,local_sha256=digest);reused[name]+=1
    else:
     try:decoded_sha,image,_=m._decode_suvi(r) if name=='suvi' else m._decode_gong(r)
     except Exception as e:decode_errors.append({'source':name,'path':r['path'],'error':str(e)});continue
     images[len(updated)]=image;r=dict(r,local_sha256=decoded_sha);new[name]+=1
    updated.append(r)
    if i%1000==0:print(f'S2 {name} {i}/{len(records)} reused={reused[name]} new={new[name]}',flush=True)
   images.flush();np.save(out/f'{name}_images.npy',np.asarray(images[:len(updated)]));del images;(out/f'pending-{name}_images.npy').unlink();all_records[name]=updated
  with lease() as gpu:
   for stage in ['split','final']:
    summary=old[f'{stage}_encoder']
    for name,records in all_records.items():
     stats={'mean':summary[f'{name}_pixel_mean'],'scale':summary[f'{name}_pixel_scale']}
     images=np.load(out/f'{name}_images.npy',mmap_mode='r')
     m._embed_all(images,records,stats,OLD/f'encoder_{stage}.pt',f'cuda:{gpu}',out/f'{name}_embedding_{stage}.npy')
  atomic_json(out/'records.json',all_records);lasts={name:str(pd.Timestamp(max(r['obs_ns'] for r in rs),tz='UTC')) for name,rs in all_records.items()}
  result.update(status='partial' if not (V2/'gong-mrzqs/MANIFEST.json').exists() or decode_errors or any(pd.Timestamp(t)<end-pd.Timedelta(days=1) for t in lasts.values()) else 'completed',returncode=0,source_observation_last=lasts,rows={k:len(v) for k,v in all_records.items()},reused_decodes=reused,new_decodes=new,decode_errors=decode_errors,gpu=gpu,files={f.name:sha(f) for f in out.glob('*.npy')})
 except Exception as e:result.update(status='failed',returncode=1,error=f'{type(e).__name__}: {e}');traceback.print_exc()
 result['finished_at']=str(pd.Timestamp.now(tz='UTC'));atomic_json(out/'MANIFEST.json',result);return result.get('returncode',1)
if __name__=='__main__':raise SystemExit(main())
