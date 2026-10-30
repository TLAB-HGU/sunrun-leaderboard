"""Decode v2 GONG FITS into hash-keyed arrays without changing source maps."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from build import HERE,V2,sha,load,atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--end',default='now');p.add_argument('--output',type=Path,default=V2/'gong-decoded');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);cache=a.output/'cache';cache.mkdir(exist_ok=True)
 end=pd.Timestamp.now(tz='UTC') if a.end=='now' else pd.Timestamp(a.end)
 if end.tz is None:end=end.tz_localize('UTC')
 m=load('gong_polar');kernel=sha(HERE/'vendor/gong_polar.py');records=[];errors=[];rows={}
 for line in (V2/'gong-mrzqs/fetch-log.jsonl').read_text().splitlines():
  try:r=json.loads(line)
  except json.JSONDecodeError:continue
  if r.get('file') and r.get('sha256'):rows[r['file']]=r
 for r in rows.values():
  path=V2/'gong-mrzqs'/r['file']
  try:
   if sha(path)!=r['sha256']:raise ValueError('source hash mismatch')
   header,image=m.read_fits_image(path);obs=pd.Timestamp(header['DATE'],tz='UTC')
   if obs>end:continue
   key=hashlib.sha256((kernel+r['sha256']).encode()).hexdigest();dest=cache/f'{key}.npz'
   receipt=cache/f'{key}.json'
   if not (dest.exists() and receipt.exists() and json.loads(receipt.read_text()).get('output_sha256')==sha(dest)):
    np.savez_compressed(dest,image=image);atomic_json(receipt,{'header':header,'source_sha256':r['sha256'],'output_sha256':sha(dest),'decoder_sha256':kernel})
   records.append({'observation_time_utc':obs,'source':str(path),'source_sha256':r['sha256'],'cache':str(dest),'cache_sha256':sha(dest)})
  except Exception as e:errors.append({'file':str(path),'error':str(e)})
 frame=pd.DataFrame(records);frame.to_parquet(a.output/'index.parquet',index=False);last=frame.observation_time_utc.max() if len(frame) else None
 atomic_json(a.output/'MANIFEST.json',{'dataset':'gong-decoded','status':'partial' if not (V2/'gong-mrzqs/MANIFEST.json').exists() or len(records)<len(list((V2/'gong-mrzqs/maps').rglob('*.fits.gz'))) or errors or last is None or last<end-pd.Timedelta(days=1) else 'completed','returncode':1 if errors else 0,'as_of':str(pd.Timestamp.now(tz='UTC')),'requested_end':str(end),'rows':len(frame),'source_observation_last':str(last),'builder_sha256':sha(__file__),'decoder_sha256':kernel,'files':{'index.parquet':sha(a.output/'index.parquet')},'errors':errors,'availability_note':'Decoded source arrays only; origin consumers must join gong-derived HTTP availability receipts.'})
 return 1 if errors else 0
if __name__=='__main__':raise SystemExit(main())
