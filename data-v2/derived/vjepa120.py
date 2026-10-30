"""Build 120h frozen clip embeddings from the complete v2 SUVI snapshot."""
import argparse,json,os,shutil,sys,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from build import HERE,LB,V2,sha,atomic_json
sys.path.insert(0,str(LB));sys.path.insert(0,str(HERE/'vendor'))
from vjepa_frozen.extract import prepare_frames
from vjepa_120h.temporal import plan_sources
from vjepa_120h.extract import extract,merge

def main():
    p=argparse.ArgumentParser();p.add_argument('--end',default='now');p.add_argument('--output',type=Path,default=V2/'vjepa-120h');a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True);result={'dataset':'vjepa-120h','as_of':str(pd.Timestamp.now(tz='UTC')),'builder_sha256':sha(__file__)}
    try:
        parent=json.loads((V2/'vjepa-frozen/MANIFEST.json').read_text());source=Path(parent['run'])
        if a.end!='now' and pd.Timestamp(a.end)!=pd.Timestamp(parent['requested_end']):
            raise ValueError('Requested end must match frozen v2 snapshot; rebuild vjepa-frozen at that end first')
        if parent['status'] not in ['completed','partial']:raise RuntimeError('v2 frozen frame preparation/extraction must complete before 120h reuse')
        run=out/source.name;run.mkdir(exist_ok=True);result['run']=str(run)
        for name in ['inventory.parquet','intensity.json','planned_origins.npz']:
            dest=run/name
            if dest.exists() and sha(dest)!=sha(source/name):raise ValueError(f'v2 input drift: {name}')
            if not dest.exists():shutil.copy2(source/name,dest)
        for name in ['upstream','checkpoints','cache/frames']:
            if not (run/name).exists():shutil.copytree(source/name,run/name,copy_function=os.link)
        atomic_json(run/'protocol.json',{'source':'v2 SUVI','origin_end':parent['requested_end'],'checkpoint_sha256':parent['checkpoint_sha256'],'mode':'frozen feature inference only'})
        origins=np.load(run/'planned_origins.npz')['origins'];rows,ids,*_=plan_sources(pd.read_parquet(run/'inventory.parquet'),origins)
        prepare_frames(run,rows,ids,workers=2)
        for gpu in (0,1):extract(run,gpu,batch_size=2)
        merge(run)
        result.update(status=parent['status'],returncode=0,requested_end=parent['requested_end'],source_observation_last=parent['source_observation_last'],source_manifest_sha256=sha(V2/'vjepa-frozen/MANIFEST.json'),rows=len(origins),files={str(p):sha(p) for p in run.glob('*.npz')})
    except Exception as e:result.update(status='failed',returncode=1,error=f'{type(e).__name__}: {e}');traceback.print_exc()
    atomic_json(out/'MANIFEST.json',result);return result.get('returncode',1)
if __name__=='__main__':raise SystemExit(main())
