"""Incremental raw refresh; historical bodies are hardlinked and never overwritten."""
import argparse, concurrent.futures as cf, datetime as dt, hashlib, importlib.util, json, os, re, sys
from pathlib import Path
import pandas as pd
import tarfile
import fcntl
from http_resume import download
from raw_sources import Run, seed, save, get, days
LB=Path('/home/t-lab01/sunrun/leaderboard')
STORE_ROOT = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store')
NOW=pd.Timestamp.now(tz='UTC')
BUILDER_BYTES=Path(__file__).read_bytes()
BUILDER_SHA256=hashlib.sha256(BUILDER_BYTES).hexdigest()
def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def finish(r, extra):
    entrypoint=extra.pop('builder_entrypoint',__file__)
    body=BUILDER_BYTES if entrypoint==__file__ else Path(entrypoint).read_bytes()
    digest=hashlib.sha256(body).hexdigest()
    provenance=r.dir/'provenance'/f'{Path(entrypoint).stem}-{digest}.py'
    save(provenance,body)
    files=[{'file':str(p.relative_to(r.dir)), 'bytes':p.stat().st_size, 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(r.dir.rglob('*')) if p.is_file() and p.name!='MANIFEST.json' and not p.name.endswith('.tmp')]
    r.manifest(dict(extra,files=files,builder=str(provenance),entrypoint=entrypoint,builder_sha256=digest))

def nmdb():
    here=LB/'experiments/d156_nm57_hourly'; sys.path.insert(0,str(here)); m=module('nm_hourly',here/'hourly.py')
    r=Run('nmdb','https://www.nmdb.eu/nest/getNMDB.php'); m.DATA=r.dir/'raw'; m.SP=dict(m.SP,end=NOW.floor('h').isoformat())
    seed(LB.parent/'research/notes/d156_nm57_hourly/data',m.DATA,'*.body'); seed(LB.parent/'research/notes/d156_nm57_hourly/data',m.DATA,'*.receipt.json')
    # Cache keys for historical requests stay intact; the moving last interval is new.
    f=m.retrieve(); finish(r,{'observation_min':str(f.dropna(how='all').index.min()),'observation_max':str(f.dropna(how='all').index.max()),'availability_rule':'label+64 minutes; operational proxy, historical revisions possible','station_coverage':{c:{'last':str(f[c].last_valid_index()),'finite':int(f[c].notna().sum())} for c in f}})

def ambient():
    m=module('ambient_raw',LB/'experiments/solar_inc_ambsw72_20261014/ambient.py'); r=Run('enlil-ambient','NOAA SWPC daily ambient Earth/L1 series')
    parser=m._parser_fingerprint(); old=STORE_ROOT/'solar-inc-ambsw72-20261014'/parser[:16]; target=r.dir/'series'; seed(old,target,'*.parquet')
    receipts=[]
    for p in old.glob('*.json'):
        x=json.loads(p.read_text()); q=target/(p.stem+'.parquet')
        if x.get('ok') and q.exists():
            x['seeded_from']=str(p); x['series_path']=str(q); receipts.append(x)
    obj=object.__new__(m.AmbientInputs); obj.store=target; obj.parser_hash=parser; obj.source_manifest_hash=m.sha256(m.SOURCE_MANIFEST)
    existing={x['issue_time_utc'][:10] for x in receipts}; rows=[]; gaps=[]
    for d in days(dt.date(2026,9,23),NOW.date()):
        if str(d) in existing: continue
        url=f'https://archive.data.noaa.gov/satellite-spaceweather/SWPC/Models/ENLIL/swpc_wsaenlil_bkg/{d:%Y/%m}/swpc_wsaenlil_bkg_{d:%Y%m%d}_0000.tar.gz'
        rows.append({'init_date':str(d),'url':url})
    def retrieve(row):
        issue=pd.Timestamp(row['init_date'],tz='UTC');key=issue.strftime('%Y%m%d');path=target/f'{key}.parquet';receipt=target/f'{key}.json'
        if path.exists() and receipt.exists():
            cached=json.loads(receipt.read_text())
            if cached.get('ok') and cached.get('series_sha256')==m.sha256(path):return cached
        raw=target/f'{key}.tar.gz.tmp'
        try:
            got=download(row['url'],raw)
            with tarfile.open(raw,'r:gz') as archive:
                members=[v for v in archive.getmembers() if v.isfile() and v.name.endswith('.suball.nc')]
                if len(members)!=1:raise ValueError('expected one suball member')
                frame,metadata=m.parse_suball(archive.extractfile(members[0]).read())
            tmp=path.with_suffix('.parquet.tmp');frame.to_parquet(tmp,index=False);tmp.replace(path)
            result=dict(got,source_url=row['url'],issue_time_utc=issue.isoformat(),execution_start_utc=m.utc(metadata.get('execution_start_time') or issue).isoformat(),metadata=metadata,series_path=str(path),series_sha256=m.sha256(path),body_sha256=got['sha256'],parser_sha256=parser,rows=len(frame),ok=True)
            save(receipt,json.dumps(result,indent=2).encode());raw.unlink();return result
        except Exception as exc:return dict(issue_time_utc=issue.isoformat(),source_url=row['url'],ok=False,error=str(exc),http_status=getattr(exc,'code',None))
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        for future in cf.as_completed([pool.submit(retrieve,row) for row in reversed(rows)]):
            x=future.result()
            print('ENLIL',x.get('issue_time_utc'),x.get('ok'),flush=True)
            (receipts if x.get('ok') else gaps).append(x)
            save(r.dir/'progress.json',json.dumps({'status':'running','completed_runs':len(receipts),'gaps':gaps,'checked_at':pd.Timestamp.now(tz='UTC').isoformat()}).encode())
    save(r.dir/'run_receipts.json',json.dumps(receipts,indent=2).encode()); save(r.dir/'gaps.json',json.dumps(gaps,indent=2).encode())
    failures=[x for x in gaps if x.get('http_status')!=404]
    finish(r,{'status':'partial' if failures else ('complete_with_upstream_gaps' if gaps else 'complete'),'latest_init':max(x['issue_time_utc'] for x in receipts),'runs':len(receipts),'gaps':gaps,'availability_rule':'init+96h and execution_start<=origin; operational proxy','historical_inventory':str(m.SOURCE_MANIFEST)})
    if failures:raise RuntimeError(f'{len(failures)} ENLIL transport/parse failures')

def k1():
    m=module('k1_download',LB/'experiments/solar_inc_alphahe72_20261014/download_k1.py'); r=Run('ace-k1','ftps://spdf.gsfc.nasa.gov/pub/data/ace/swepam/level_2_cdaweb/swe_k1/')
    m.DEST=r.dir/'k1'; m.LAST=NOW.date(); seed(STORE_ROOT/'alphahe72-cache/source/k1',m.DEST,'**/*.cdf')
    listing=m._select_files(); save(r.dir/'listing.json',json.dumps(listing).encode()); pending=[]; results=[]
    for day,name,remote in listing:
        path=m.DEST/f'year={day[:4]}'/name
        if path.exists() and path.stat().st_size>=512:
            results.append(dict(date=day,file=name,path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_url='ftps://spdf.gsfc.nasa.gov'+remote))
        else: pending.append((day,name,remote))
    errors=[]
    for item in pending:
        try: results.extend(m._worker([item]))
        except Exception as exc: errors.append({'file':item,'error':str(exc)})
    save(r.dir/'k1_source_manifest.json',json.dumps({'files':results,'errors':errors},indent=2).encode())
    finish(r,{'status':'partial' if errors else 'complete','observation_range':[min(x['date'] for x in results),max(x['date'] for x in results)],'listed_latest':max(x[0] for x in listing),'availability_rule_hours':244,'errors':errors,'note':'archive edge lag; fetched_at does not prove historical availability'})
    if errors: raise RuntimeError(str(errors))

def punch():
    m=module('punch_raw',LB/'experiments/punch_jmap72/punch_features.py'); r=Run('punch-qam','https://umbra.nascom.nasa.gov/punch/Q/QAM/'); records=[]; pages=[]; errors=[]
    def inventory(d):
        url=f'https://umbra.nascom.nasa.gov/punch/Q/QAM/{d:%Y/%m/%d}/'
        previous=r.dir/'listings'/f'{d}.html'; receipt_file=r.dir/'listing_receipts.json'
        previous_receipts=json.loads(receipt_file.read_text()) if receipt_file.exists() else []
        receipt=next((x for x in previous_receipts if x['url']==url),None)
        if d<NOW.date() and previous.exists() and receipt and m._sha_file(previous)==receipt['sha256']:
            page=dict(receipt,body=previous.read_bytes())
        else:page=m._safe_page(url)
        m.BASE_URL=f'https://umbra.nascom.nasa.gov/punch/Q/QAM/{d.year}'
        return d,page,m._parse_files(page,d) if page['status']==200 else []
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        for d,page,files in pool.map(inventory,days(dt.date(2026,6,1),NOW.date())):
            pages.append({k:v for k,v in page.items() if k!='body'}); save(r.dir/'listings'/f'{d}.html',page['body']); records.extend(m._select_cadence_frames(files))
    save(r.dir/'listing_receipts.json',json.dumps(pages,indent=2).encode()); save(r.dir/'selected_frames.json',json.dumps(records,indent=2).encode())
    def fetch(rec):
        path=r.dir/'raw'/rec['day']/rec['filename']; receipt=path.with_suffix('.receipt.json')
        if path.exists() and receipt.exists():
            old=json.loads(receipt.read_text())
            if m._sha_file(path)==old.get('sha256'): return old
        try:
            tmp=path.with_suffix('.fits.tmp'); got=download(rec['url'],tmp); tmp.replace(path)
            if path.read_bytes()[:9]!=b'SIMPLE  =': raise ValueError('not a FITS body')
            got.update(rec,path=str(path)); save(receipt,json.dumps(got,indent=2).encode()); return got
        except Exception as exc: return dict(rec,error=str(exc))
    # Current edge first, then complete the historical backfill; partial bodies resume.
    records.sort(key=lambda x:x['obs_time_filename'],reverse=True)
    got=[]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        for i,x in enumerate(pool.map(fetch,records),1):
            (errors if 'error' in x else got).append(x)
            if i%20==0: print('PUNCH',i,len(records),'errors',len(errors),flush=True)
    save(r.dir/'receipts.json',json.dumps(got,indent=2).encode()); save(r.dir/'errors.json',json.dumps(errors,indent=2).encode())
    finish(r,{'status':'partial' if errors else 'complete','selected_frames':len(records),'downloaded_frames':len(got),'error_count':len(errors),'observation_min':min((x['obs_time_filename'] for x in got),default=None),'observation_max':max((x['obs_time_filename'] for x in got),default=None),'cadence_hours':6,'selection':'existing builder nearest frame within 1.5h of each six-hour target','availability_rule':'use FITS issue time plus conservative listing timestamp and HTTP Last-Modified; fetched_at is not historical availability'})
    if errors: raise RuntimeError(f'{len(errors)} PUNCH downloads failed')

def punch_bounded_samples(output, urls, prior_receipts=None):
    """Explicit current QAM/QNN schema samples: max12 files /256MiB total."""
    from urllib.parse import urlparse
    if not 1 <= len(urls) <= 12 or len(urls) != len(set(urls)):
        raise ValueError('one to twelve unique FITS URLs required')
    for url in urls:
        parsed=urlparse(url)
        if (parsed.scheme != 'https' or parsed.netloc != 'umbra.nascom.nasa.gov'
                or not re.fullmatch(r'/punch/Q/(QAM|QNN)/[0-9]{4}/[0-9]{2}/[0-9]{2}/PUNCH_LQ_\1_[0-9]{14}_v[0-9a-z]+\.fits',parsed.path)
                or parsed.query or parsed.fragment):
            raise ValueError('explicit QAM/QNN FITS URL required')
    previous=[]
    for receipt_file in prior_receipts or []:
        previous.extend(json.loads(Path(receipt_file).read_text()))
    if len(previous)+len(urls)>12 or len({r['url'] for r in previous})!=len(previous):
        raise ValueError('cumulative twelve-file limit or duplicate prior receipt')
    if set(urls)&{r['url'] for r in previous}:raise ValueError('previous sample cannot be downloaded again')
    for r in previous:
        p=Path(r['path'])
        if p.stat().st_size!=r['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:
            raise ValueError('prior body receipt mismatch')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if list(output.glob('*')):
        raise ValueError('new empty diagnostic directory required; no cumulative cap reset')
    cap=256*1024*1024;used=sum(r['bytes'] for r in previous);receipts=[]
    if used>=cap:raise ValueError('cumulative byte budget exhausted')
    for url in urls:
        path=output/Path(urlparse(url).path).name
        got=download(url,path,attempts=1,max_bytes=cap-used)
        used+=got['bytes']
        with path.open('rb') as stream:
            if stream.read(9)!=b'SIMPLE  =':raise ValueError('not a FITS body')
        got.update(path=str(path),first_seen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),historical_eligibility=False)
        receipts.append(got)
        save(path.with_suffix('.receipt.json'),json.dumps(got,indent=2).encode())
    save(output/'receipts.json',json.dumps(receipts,indent=2).encode())
    return receipts

def punch_metadata_inventory(output, urls):
    """Explicit bounded listing probe; no FITS bodies or global refresh changes."""
    from urllib.parse import urlparse
    if not 1 <= len(urls) <= 12:
        raise ValueError('one to twelve metadata URLs required')
    for url in urls:
        parsed=urlparse(url)
        if (parsed.scheme != 'https' or parsed.netloc != 'umbra.nascom.nasa.gov'
                or not parsed.path.startswith('/punch/') or not parsed.path.endswith('/')
                or parsed.query or parsed.fragment):
            raise ValueError('PUNCH directory metadata URLs only')
    output=Path(output); output.mkdir(parents=True, exist_ok=True)
    m=module('punch_metadata',LB/'experiments/punch_jmap72/punch_features.py')
    receipts=[]
    for url in urls:
        key=hashlib.sha256(url.encode()).hexdigest()[:16]
        try:
            page=m._safe_page(url)
            save(output/f'{key}.html',page['body'])
            receipt={k:v for k,v in page.items() if k!='body'}
            receipt.update(metadata_only=True,body_bytes=len(page['body']),local_listing=str(output/f'{key}.html'))
        except Exception as exc:
            receipt={'url':url,'error':str(exc),'metadata_only':True}
        receipts.append(receipt)
        save(output/f'{key}.json',json.dumps(receipt,indent=2).encode())
    return receipts

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('dataset',choices=['nmdb','ambient','k1','punch']); args=p.parse_args()
    lock_path=STORE_ROOT/'v2'/f'.remaining-{args.dataset}.lock'; lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        globals()[args.dataset]()
