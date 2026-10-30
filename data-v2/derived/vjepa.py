"""Frozen V-JEPA frame/video inference on new v2 input snapshots."""
import argparse,json,os,shutil,sys,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from build import HERE,LB,V2,sha,atomic_json
sys.path.insert(0,str(LB));sys.path.insert(0,str(HERE/'vendor'))
from vjepa_frozen.extract import extract

STORE_ROOT = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store')



OLD=STORE_ROOT/'vjepa-frozen-fusion/run-20261003T084905Z'
CHECKPOINT='vjepa2_1_vitb_dist_vitG_384.pt'
EXPECTED='848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d'

def main():
    p=argparse.ArgumentParser();p.add_argument('--end',default='now');p.add_argument('--output',type=Path,default=V2/'vjepa-frozen');a=p.parse_args()
    end=pd.Timestamp.now(tz='UTC').floor('h') if a.end=='now' else pd.Timestamp(a.end)
    if end.tz is None:end=end.tz_localize('UTC')
    out=a.output;out.mkdir(parents=True,exist_ok=True);run=out/end.strftime('%Y%m%dT%H%M%SZ');run.mkdir(exist_ok=True)
    result={'dataset':'vjepa-frozen','status':'running','builder_sha256':sha(__file__),'requested_end':str(end),'as_of':str(pd.Timestamp.now(tz='UTC')),'run':str(run)};atomic_json(out/'MANIFEST.json',result)
    try:
        checkpoint=OLD/'checkpoints'/CHECKPOINT
        if sha(checkpoint)!=EXPECTED:raise ValueError('VJEPA checkpoint hash mismatch')
        (run/'checkpoints').mkdir(exist_ok=True)
        if not (run/'checkpoints'/CHECKPOINT).exists():os.link(checkpoint,run/'checkpoints'/CHECKPOINT)
        if not (run/'upstream').exists():shutil.copytree(OLD/'upstream',run/'upstream',ignore=shutil.ignore_patterns('.git','__pycache__'),copy_function=os.link)
        if not (run/'intensity.json').exists():shutil.copy2(OLD/'intensity.json',run/'intensity.json')
        parts=sorted(Path('/home/t-lab01/.local/state/sundb/stage/suvi_hourly').glob('year=*/part.parquet'))
        meta=pd.concat([pd.read_parquet(f) for f in parts],ignore_index=True)
        for col in ['slot','obs_end','s3_modified']:meta[col]=pd.to_datetime(meta[col],utc=True)
        meta=meta[(meta.channel=='Fe195')&(meta.status=='ok')&(meta.slot<=end)&meta.nc_path.notna()&meta.s3_modified.notna()&meta.obs_end.notna()].sort_values('slot').drop_duplicates('slot',keep='last')
        meta['observation_slot']=meta.slot
        # Slot lookup now uses causal publication eligibility, preserving observation age.
        meta['slot']=pd.concat([meta.slot,meta.obs_end+pd.Timedelta(hours=1),meta.s3_modified],axis=1).max(axis=1)
        meta=meta[meta.slot<=end].sort_values('slot')
        meta['path']=meta.nc_path.map(lambda x:str(Path(x) if Path(x).is_absolute() else LB.parent/'sun-img'/x))
        stats=[Path(x).stat() if Path(x).is_file() else None for x in meta.path]
        meta['size_bytes']=[s.st_size if s else 0 for s in stats];meta['mtime_ns']=[s.st_mtime_ns if s else 0 for s in stats];meta['available']=(meta.size_bytes>0)&(meta.size_bytes==meta.nc_size)
        invalid_geometry=np.zeros(len(meta),dtype=bool)
        for column in ['rsun','crpix1','crpix2','crota']:
            values=pd.to_numeric(meta[column],errors='coerce').to_numpy(dtype=float)
            invalid_geometry|=~np.isfinite(values)
            if column=='rsun':invalid_geometry|=values<=0
        meta['invalid_geometry']=invalid_geometry;meta['available']&=~invalid_geometry
        meta.to_parquet(run/'inventory.parquet',index=False)
        origins=pd.date_range(pd.Timestamp('2022-10-01',tz=end.tz),end,freq='h').as_unit('ns');np.savez(run/'planned_origins.npz',origins=origins.asi8)
        result.update(source_metadata_hashes={str(f):sha(f) for f in parts},source_observation_last=str(meta.obs_end.max()),source_rows=len(meta),missing_images=int((~meta.available).sum()),checkpoint_sha256=EXPECTED,availability_rule='max(metadata slot, obs_end + 1h, S3 LastModified) <= requested clip slot')
        atomic_json(out/'MANIFEST.json',result)
        extract(run,batch_size=2,workers=2,pilot=2)
        result.update(status='partial' if result['missing_images'] else 'completed',returncode=0,rows=len(origins),files={str(run/'embeddings.npz'):sha(run/'embeddings.npz')})
    except Exception as e:result.update(status='failed',returncode=1,error=f'{type(e).__name__}: {e}');traceback.print_exc()
    result['finished_at']=str(pd.Timestamp.now(tz='UTC'));atomic_json(out/'MANIFEST.json',result);return result.get('returncode',1)
if __name__=='__main__':raise SystemExit(main())
