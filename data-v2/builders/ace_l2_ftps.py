"""Public SPDF FTPS recovery to v2; never publish revised L2 as RTSW."""
import datetime as dt, ftplib, hashlib, json, sys
from pathlib import Path
sys.path.insert(0,'/home/t-lab01/spaceweather-collector/src')
sys.path.append('/home/t-lab01/spaceweather-collector/.venv/lib/python3.12/site-packages')
import pandas as pd
from spaceweather_collector.sources.ace_l2 import parse_ace_l2_file
from remaining_raw import LB, module, finish
from raw_sources import Run,save
m=module('k1_transport',LB/'experiments/solar_inc_alphahe72_20261014/download_k1.py')
r=Run('ace-l2','ftps://spdf.gsfc.nasa.gov/pub/data/ace/')
records=[]; errors=[]; listing=[]; frames=[]
for product,folder,prefix in [('mag','mag/level_2_cdaweb/mfi_h2','ac_h2_mfi'),('swepam','swepam/level_2_cdaweb/swe_h2','ac_h2_swe')]:
    client=None
    try:
        client=m._ftp(); names=[]
        for year in range(2026,2021,-1):
            remote=f'/pub/data/ace/{folder}/{year}'
            try: names=client.nlst(remote)
            except ftplib.error_temp as exc:
                if 'No such file' not in str(exc): raise
                listing.append({'product':product,'remote':remote,'status':'unpublished','error':str(exc)}); continue
            if names: break
        names=sorted(n for n in names if n.rsplit('/',1)[-1].startswith(prefix) and n.endswith('.cdf'))
        listing.append({'product':product,'remote':remote,'files':names,'retrieved_at':dt.datetime.now(dt.timezone.utc).isoformat()})
        for name in names[-3:]:
            basename=name.rsplit('/',1)[-1]; stamp=basename[len(prefix)+1:len(prefix)+9]
            path=r.dir/'raw'/product/basename; path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                tmp=path.with_suffix('.tmp')
                with tmp.open('wb') as f:client.retrbinary('RETR '+name,f.write)
                tmp.replace(path)
            parsed=list(parse_ace_l2_file(path,product)); frame=pd.DataFrame([dict(timestamp_utc=t,**v) for t,v in parsed]); frames.append(frame)
            records.append({'product':product,'url':'ftps://spdf.gsfc.nasa.gov'+name,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'rows':len(frame),'first':str(frame.timestamp_utc.min()),'last':str(frame.timestamp_utc.max()),'retrieved_at':dt.datetime.now(dt.timezone.utc).isoformat()})
            print(product,basename,len(frame),flush=True)
    except Exception as exc:errors.append({'product':product,'error':str(exc)})
    finally:
        if client:
            try:client.quit()
            except Exception:client.close()
    save(r.dir/'listing.json',json.dumps(listing,indent=2).encode());save(r.dir/'receipts.json',json.dumps(records,indent=2).encode());save(r.dir/'errors.json',json.dumps(errors,indent=2).encode())
if frames:
    f=pd.concat(frames,ignore_index=True).groupby('timestamp_utc').first().reset_index(); f.to_parquet(r.dir/'hourly.parquet',index=False)
finish(r,{'builder_entrypoint':__file__,'status':'partial' if errors else 'complete','errors':errors,'products':records,'availability_rule':'verified L2 has product-dependent weeks-to-years archive delay; not ACE RTSW +2h eligible','scope':'latest three published daily CDFs per product; full history remains in solar_l2 PostgreSQL', 'latest_published':{x['product']:max(x.get('files',[]) or ['unpublished']) for x in listing if x.get('files')}})
if errors:raise RuntimeError(str(errors))
