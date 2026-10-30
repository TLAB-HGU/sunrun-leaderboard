"""Reuse decoded VJEPA frames only after exact kernel, transform and byte checks."""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np,pandas as pd
from build import HERE,LB,V2,sha,atomic_json
sys.path.insert(0,str(LB));sys.path.insert(0,str(HERE/'vendor'))
from vjepa_frozen.extract import source_binding,plan_sources

STORE_ROOT = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store')


old=STORE_ROOT/'vjepa-frozen-fusion/run-20261003T084905Z'
manifest=json.loads((V2/'vjepa-frozen/MANIFEST.json').read_text());run=Path(manifest['run'])
for name in ['images.py']:
 if sha(HERE/'vendor/vjepa_frozen'/name)!=sha(LB/'experiments/vjepa_frozen_fusion'/name):raise ValueError('preprocessing code changed')
if sha(run/'intensity.json')!=sha(old/'intensity.json'):raise ValueError('frozen intensity changed')
old_binding=json.loads((old/'execution-source-hashes.json').read_text())
for key,digest in old_binding.items():
 if key.endswith(('images.py','xgboost_suvi_fusion/features.py')) and sha(key)!=digest:raise ValueError('original preprocessing source drift')
oldmap=json.loads((old/'frame-cache-manifest.json').read_text())['frames'];by_path={r['raw_path']:r for r in oldmap.values()}
binding=source_binding(run);ph=hashlib.sha256(json.dumps({k:v for k,v in binding.items() if k.endswith(('images.py','xgboost_suvi_fusion/features.py'))},sort_keys=True).encode()).hexdigest();th=hashlib.sha256(json.dumps(json.loads((run/'intensity.json').read_text()),sort_keys=True).encode()).hexdigest()
origins=np.load(run/'planned_origins.npz')['origins'];rows,ids,*_=plan_sources(pd.read_parquet(run/'inventory.parquet'),origins)
count=0;failed=[]
for i in np.unique(ids[ids>=0])[::-1]:
 row=rows.iloc[i];previous=by_path.get(row.path)
 if previous is None:continue
 key=hashlib.sha256(f'{row.path}:{row.size_bytes}:{row.mtime_ns}:{th}:{ph}'.encode()).hexdigest();dest=run/'cache/frames'/(key+'.npy');meta=dest.with_suffix('.json')
 if dest.exists() or meta.exists():continue
 try:
  if sha(row.path)!=previous['raw_sha256']:continue
  src=old/'cache/frames'/(previous['key']+'.npy')
  if sha(src)!=previous['preprocessed_sha256']:raise ValueError('cached frame hash mismatch')
  temp=dest.with_suffix('.seed.tmp');os.link(src,temp);temp.replace(dest)
  receipt={k:v for k,v in previous.items() if k!='key'};receipt.update(bytes=int(row.size_bytes),mtime_ns=int(row.mtime_ns),source_slot=row.slot.isoformat(),obs_end=row.obs_end.isoformat())
  atomic_json(meta,receipt);count+=1
  if count%1000==0:print(f'seeded {count} verified frames',flush=True)
 except Exception as e:failed.append({'path':row.path,'error':str(e)})
atomic_json(run/'frame-reuse-receipt.json',{'reused_frames':count,'errors':failed,'original_manifest_sha256':sha(old/'frame-cache-manifest.json'),'kernel_sha256':sha(HERE/'vendor/vjepa_frozen/images.py'),'intensity_sha256':sha(run/'intensity.json')})
print(count,len(failed))
