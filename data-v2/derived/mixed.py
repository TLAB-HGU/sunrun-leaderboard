"""Causal CH/GONG and CME/CH/GONG feature-only v2 builders."""
import argparse,json,sys,traceback
from pathlib import Path
import pandas as pd
from build import HERE,LB,V2,sha,load,atomic_json
sys.path.insert(0,str(LB));sys.path.insert(0,str(HERE/'vendor'))

def main():
 p=argparse.ArgumentParser();p.add_argument('job',choices=['cmechip','chunip','magconc']);p.add_argument('--end',default='now');p.add_argument('--output',type=Path);a=p.parse_args()
 end=pd.Timestamp.now(tz='UTC').floor('h') if a.end=='now' else pd.Timestamp(a.end)
 if end.tz is None:end=end.tz_localize('UTC')
 out=a.output or V2/a.job;out.mkdir(parents=True,exist_ok=True);result={'dataset':a.job,'requested_end':str(end),'as_of':str(pd.Timestamp.now(tz='UTC')),'builder_sha256':sha(__file__),'status':'running'};atomic_json(out/'MANIFEST.json',result)
 try:
  rows=[json.loads(f.read_text()) for f in (V2/'gong-derived/receipts').glob('*.json')]
  if not rows:raise RuntimeError('v2 GONG source receipts not ready')
  manifest=out/'gong-source.json';atomic_json(manifest,{'maps':rows})
  load('cmechip');m=load(a.job)
  if a.job=='cmechip':catalog=m.ChipFeatureCatalog(end,manifest)
  else:
   spec=m.morph_spec
   spec.STORE=out/'ch-selector';spec.SOURCE_MANIFEST=V2/'euvwave/source_manifest.json';spec.CH_PATH=V2/'ch/ch-hourly.parquet';spec.CH_FEATURE_MANIFEST=V2/'ch/ch-feature-manifest.json';spec.CH_INPUT_IDENTITY=V2/'ch/input-identity.json';spec.SUVI_CACHE=V2/'suvi-fusion/suvi-hourly-v2.parquet'
   metadata_parts=sorted(Path('/home/t-lab01/.local/state/sundb/stage/suvi_hourly').glob('year=*/part.parquet'))
   pd.concat([pd.read_parquet(f) for f in metadata_parts],ignore_index=True).to_parquet(out/'metadata.parquet',index=False)
   spec.SUVI_META_GLOB=str(out/'metadata.parquet');spec.CH_INPUT_IDENTITY=out/'input-identity.json'
   atomic_json(spec.CH_INPUT_IDENTITY,{'inputs':{str(f):sha(f) for f in [spec.CH_PATH,spec.SUVI_CACHE,spec.SOURCE_MANIFEST,out/'metadata.parquet']},'source_metadata':{str(f):sha(f) for f in metadata_parts}})
   class V2MorphologyContext(m.morph_features.MorphologyContext):
    def _origin_sets(self):
     return {'generation':pd.date_range(pd.Timestamp('2022-10-01',tz=self.cap.tz),self.cap,freq='h')}
   m.morph_features.MorphologyContext=V2MorphologyContext
   cls=m.CHUnipCatalog if a.job=='chunip' else m.CHMagConcCatalog;catalog=cls(end,manifest,sha(manifest))
  origins=pd.date_range(pd.Timestamp('2022-10-01',tz=end.tz),end,freq='h').as_unit('ns');chunks=[]
  for start in range(0,len(origins),256):
   block=origins[start:start+256];values=catalog.features_for(block);frame=pd.DataFrame(values,index=block,columns=m.FEATURE_COLUMNS);frame.index.name='origin_last_input_utc';chunks.append(frame)
   print(f'{a.job} origins {min(start+256,len(origins))}/{len(origins)}',flush=True)
  pd.concat(chunks).reset_index().to_parquet(out/'features.parquet',index=False)
  raw_manifest=V2/'gong-mrzqs/MANIFEST.json';raw_done=raw_manifest.exists() and len(rows)>=len(list((V2/'gong-mrzqs/maps').rglob('*.fits.gz')))
  result.update(status='completed' if raw_done else 'partial',returncode=0,rows=len(origins),source_observation_last=max(r['observation_time_utc'] for r in rows),source_manifest_sha256=sha(manifest),kernel_sha256=sha(HERE/'vendor'/f'{a.job}.py'),files={'features.parquet':sha(out/'features.parquet')},coverage_note='GONG raw collection must be complete before full dataset readiness')
 except Exception as e:result.update(status='failed',returncode=1,error=f'{type(e).__name__}: {e}');traceback.print_exc()
 result['finished_at']=str(pd.Timestamp.now(tz='UTC'));atomic_json(out/'MANIFEST.json',result);return result.get('returncode',1)
if __name__=='__main__':raise SystemExit(main())
