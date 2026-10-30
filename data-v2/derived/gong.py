"""Incremental v2 GONG polar/multipole extraction with real HTTP availability."""
import argparse, concurrent.futures, json, re, sys, time, urllib.request
from email.utils import parsedate_to_datetime
from pathlib import Path
import pandas as pd
from build import HERE, V2, load, sha, atomic_json

def prepare(row, raw_root, receipt_dir):
    path=raw_root/row['file']; digest=sha(path)
    if digest!=row['sha256']:raise ValueError(f'source hash changed: {path}')
    receipt=receipt_dir/f'{digest}.json'
    if receipt.exists():return json.loads(receipt.read_text())
    lm=row.get('last_modified')
    if not lm:
        for attempt in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(row['url'],method='HEAD'),timeout=20) as response:
                    lm=response.headers.get('Last-Modified')
                break
            except Exception:
                if attempt==2:raise
                time.sleep(2*(attempt+1))
    if not lm:raise ValueError(f'HTTP Last-Modified missing: {row["url"]}')
    match=re.search(r'mrzqs(\d{6})t(\d{4})',path.name)
    obs=pd.to_datetime(match.group(1)+match.group(2),format='%y%m%d%H%M',utc=True)
    available=max(obs+pd.Timedelta(hours=2),pd.Timestamp(parsedate_to_datetime(lm)))
    result={**row,'path':str(path),'file':path.name,'last_modified':lm,'observation_time_utc':str(obs),'issue_time_utc':str(obs),'available_at_utc':str(available),'metadata_checked_at':str(pd.Timestamp.now(tz='UTC'))}
    atomic_json(receipt,result);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--end',default='now');parser.add_argument('--output',type=Path,default=V2/'gong-derived');parser.add_argument('--workers',type=int,default=3);a=parser.parse_args()
    if not 1<=a.workers<=4:parser.error('workers must be between 1 and 4')
    end=pd.Timestamp.now(tz='UTC').floor('h') if a.end=='now' else pd.Timestamp(a.end)
    if end.tz is None:end=end.tz_localize('UTC')
    out=a.output;out.mkdir(parents=True,exist_ok=True);receipts=out/'receipts';receipts.mkdir(exist_ok=True);cache=out/'cache';cache.mkdir(exist_ok=True)
    raw_root=V2/'gong-mrzqs';rawlog=raw_root/'fetch-log.jsonl';rows={}
    for line in rawlog.read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        if r.get('sha256') and r.get('file'):rows[r['file']]=r
    polar=load('gong_polar');multi=load('gong_multipole');errors=[];metadata=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(prepare,r,raw_root,receipts):r for r in rows.values()}
        for i,f in enumerate(concurrent.futures.as_completed(futures)):
            try:metadata.append(f.result())
            except Exception as e:errors.append({'file':futures[f]['file'],'error':str(e)})
            if i%100==0:print(f'metadata {i+1}/{len(rows)}, errors={len(errors)}',flush=True)
    atomic_json(out/'source.json',{'maps':metadata,'errors':errors})
    builder_hash=sha(__file__); vendor_hashes={n:sha(HERE/'vendor'/f'{n}.py') for n in ['gong_polar','gong_multipole']}
    records=[]
    for i,row in enumerate(sorted(metadata,key=lambda r:r['observation_time_utc'])):
        if pd.Timestamp(row['observation_time_utc'])>end:continue
        key=__import__('hashlib').sha256(json.dumps([builder_hash,vendor_hashes,row['sha256']],sort_keys=True).encode()).hexdigest();cached=cache/f'{key}.json'
        if cached.exists():record=json.loads(cached.read_text())
        else:
            header,img=polar.read_fits_image(Path(row['path']));observation=pd.Timestamp(row['observation_time_utc'])
            vals=multi.build_multipole_features(observation,{'header':header,'image':img,'observation_time_utc':observation})
            record={'source_sha256':row['sha256'],'polar':polar.polar_cap_summaries(header,img),'multipole':list(vals)};atomic_json(cached,record)
        records.append({**{k:row[k] for k in ['file','observation_time_utc','available_at_utc','sha256']},**dict(zip(polar.POLAR_COLUMNS,record['polar'])),**dict(zip(multi.MULTIPOLE_COLUMNS,record['multipole']))})
        if i%100==0:print(f'features {i+1}/{len(metadata)}',flush=True)
    frame=pd.DataFrame(records);frame.to_parquet(out/'map-features.parquet',index=False)
    last=max((r['observation_time_utc'] for r in metadata),default=None)
    source_complete=(raw_root/'MANIFEST.json').exists() and len(metadata)>=len(list((raw_root/'maps').rglob('*.fits.gz')))
    result={'dataset':'gong-derived','status':'partial' if not source_complete or errors or last is None or pd.Timestamp(last)<end-pd.Timedelta(days=1) else 'completed','returncode':1 if errors else 0,'requested_end':str(end),'as_of':str(pd.Timestamp.now(tz='UTC')),'source_observation_last':last,'source_map_count':len(rows),'rows':len(frame),'errors':errors,'builder_sha256':builder_hash,'vendor_hashes':vendor_hashes,'files':{'map-features.parquet':sha(out/'map-features.parquet'),'source.json':sha(out/'source.json')},'availability_rule':'max(observation + 2h, HTTP Last-Modified); map_age_h stored as zero at map observation and must be recomputed at origin','coverage_note':'Input snapshot is the completed records present in raw fetch-log at start; rerun after raw collection completes.'}
    atomic_json(out/'MANIFEST.json',result);print(json.dumps(result));return result['returncode']
if __name__=='__main__':raise SystemExit(main())
