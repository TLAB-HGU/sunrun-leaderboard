"""Frozen DINOv3 patch tokens from hash-pinned SUVI inputs; shared GPU lease."""
import argparse, contextlib, fcntl, hashlib, json, os, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from build import HERE, LB, V2, load, sha, atomic_json

STORE_ROOT = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store')



@contextlib.contextmanager
def lease():
    root=STORE_ROOT/'ch-breakthrough-v2/gpu-leases';root.mkdir(parents=True,exist_ok=True)
    while True:
        for gpu in (1,0):
            f=(root/f'gpu{gpu}.lock').open('a+')
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:f.close();continue
            try:yield gpu
            finally:fcntl.flock(f,fcntl.LOCK_UN);f.close()
            return
        print('waiting for shared GPU lease',flush=True);time.sleep(10)

def main():
    p=argparse.ArgumentParser();p.add_argument('--end',default='now');p.add_argument('--output',type=Path,default=V2/'dinov3');a=p.parse_args()
    end=pd.Timestamp.now(tz='UTC') if a.end=='now' else pd.Timestamp(a.end)
    if end.tz is None:end=end.tz_localize('UTC')
    out=a.output;out.mkdir(parents=True,exist_ok=True);cache=out/'cache';cache.mkdir(exist_ok=True)
    result={'dataset':'dinov3','requested_end':str(end),'as_of':str(pd.Timestamp.now(tz='UTC')),'builder_sha256':sha(__file__),'status':'running','rows':0}
    atomic_json(out/'MANIFEST.json',result)
    try:
        import torch
        torch.set_num_threads(2)
        d=load('dinov3');result['kernel_sha256']=sha(HERE/'vendor/dinov3.py')
        if d.model_sha256()!=d.MODEL_SHA256:raise ValueError('DINO model weights hash mismatch')
        parts=sorted(Path('/home/t-lab01/.local/state/sundb/stage/suvi_hourly').glob('year=*/part.parquet'))
        meta=pd.concat([pd.read_parquet(f) for f in parts],ignore_index=True)
        for col in ['slot','obs_end','s3_modified']:meta[col]=pd.to_datetime(meta[col],utc=True)
        meta=meta[(meta.channel=='Fe195')&(meta.status=='ok')&(meta.slot<=end)&meta.nc_path.notna()&meta.s3_modified.notna()&meta.obs_end.notna()].sort_values('slot').drop_duplicates('slot',keep='last')
        meta['available_at']=pd.concat([meta.obs_end+pd.Timedelta(hours=1),meta.s3_modified],axis=1).max(axis=1)
        meta=meta[meta.available_at<=end]
        # Reuse the existing frozen intensity transform, never refit on post-cutoff images.
        scale_path=sorted((STORE_ROOT/'dinov3-cache/v1').glob('*.npz'))[0]
        with np.load(scale_path,allow_pickle=False) as z:low,high=float(z['low']),float(z['high'])
        result['intensity']={'low':low,'high':high,'frozen_source':str(scale_path),'sha256':sha(scale_path)}
        result['metadata_hashes']={str(f):sha(f) for f in parts};result['model_sha256']=d.MODEL_SHA256
        meta.to_parquet(out/'inventory.parquet',index=False)
        records=[];errors=[]
        with lease() as gpu:
            model=d.load_encoder(f'cuda:{gpu}');result['gpu']=gpu
            for i,r in enumerate(meta.itertuples(index=False)):
                path=Path(r.nc_path)
                if not path.is_absolute():path=LB.parent/'sun-img'/path
                if not path.is_file():errors.append({'slot':str(r.slot),'error':'local image absent','path':str(path)});continue
                digest=sha(path);key=hashlib.sha256(json.dumps([result['builder_sha256'],result['kernel_sha256'],digest,d.MODEL_SHA256,low,high]).encode()).hexdigest();dest=cache/f'{key}.npz';receipt=cache/f'{key}.json'
                valid=False
                if dest.exists() and receipt.exists():
                    old=json.loads(receipt.read_text());valid=old.get('output_sha256')==sha(dest) and old.get('source_sha256')==digest
                if not valid:
                    image,vp,geom=d.canvas_image(str(path),low,high)
                    tokens=d.encode_batch(model,image[None],f'cuda:{gpu}')[0]
                    np.savez_compressed(dest,tokens=tokens,valid_patch=vp)
                    atomic_json(receipt,{'key':key,'source_sha256':digest,'output_sha256':sha(dest),'geometry':geom})
                records.append({'slot':r.slot,'observation_time_utc':r.obs_end,'available_at_utc':r.available_at,'path':str(path),'source_sha256':digest,'cache':str(dest),'cache_sha256':sha(dest)})
                if i%100==0:
                    result.update(rows=len(records),errors_count=len(errors),source_observation_last=str(r.obs_end));atomic_json(out/'MANIFEST.json',result);print(f'DINO frames {i+1}/{len(meta)}',flush=True)
        pd.DataFrame(records).to_parquet(out/'tokens.parquet',index=False)
        result.update(status='partial' if errors else 'completed',returncode=0,rows=len(records),errors=errors,files={'tokens.parquet':sha(out/'tokens.parquet'),'inventory.parquet':sha(out/'inventory.parquet')},source_observation_last=str(max(r['observation_time_utc'] for r in records)))
    except Exception as e:result.update(status='blocked_credentials' if ('GatedRepoError' in type(e).__name__ or 'model.safetensors' in str(e) or '401' in str(e)) else 'failed',returncode=1,error=f'{type(e).__name__}: {e}');traceback.print_exc()
    result['finished_at']=str(pd.Timestamp.now(tz='UTC'));atomic_json(out/'MANIFEST.json',result);return result.get('returncode',1)
if __name__=='__main__':raise SystemExit(main())
