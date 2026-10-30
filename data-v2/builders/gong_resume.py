"""Resume GONG by day, retain failed-day receipts, and retry only failed days."""
import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from raw_sources import Run, get, listing, save, days, TODAY

PATTERN=re.compile(r'mrzqs(\d{6})t(\d{4})c(\d+)_(\d+)\.fits\.gz$')
BASE='https://gong.nso.edu/data/magmap/QR/zqs'
# 2026-10-15 (user): SW-012 hourly. The 00/06/12/18 grid stays in maps/ (derived jobs read only maps/);
# the other hourly files go to maps-hourly/ so gong-derived/pfss-wsa counts and pinned inputs do not change.
GRID='hourly-v1'

def reusable(record, day, today, root):
    return (day < today-dt.timedelta(days=7) and record.get('status')=='complete'
            and len(record.get('files',[]))>=4 and all((root/f).is_file() for f in record['files'])
            and record.get('grid')==GRID and all((root/f).is_file() for f in record.get('hourly_files',[])))

def main():
    run=Run('gong-mrzqs','NSO GONG QuickReduce mrzqs: 00/06/12/18 grid in maps/, remaining hourly files in maps-hourly/')
    receipts=run.dir/'inventory';receipts.mkdir(exist_ok=True)
    def one(day):
        receipt=receipts/f'{day}.json'
        if receipt.exists() and day<TODAY:
            old=json.loads(receipt.read_text())
            if reusable(old,day,TODAY,run.dir):
                return old
        url=f'{BASE}/{day:%Y%m}/mrzqs{day:%y%m%d}/'
        try:
            names=[u.rsplit('/',1)[-1] for u in listing(url) if PATTERN.search(u)]
            def observed(name):
                m=PATTERN.search(name);return dt.datetime.strptime(m[1]+m[2],'%y%m%d%H%M')
            chosen=set()
            for hour in (0,6,12,18):
                target=dt.datetime(day.year,day.month,day.day,hour,14)
                if names:chosen.add(min(names,key=lambda n:abs((observed(n)-target).total_seconds())))
            files=[]
            for name in sorted(chosen):
                rel=f'maps/{day:%Y/%m}/{name}'
                run.fetch_to(url+name,rel,replace=(day==TODAY))
                files.append(rel)
            hourly=set()
            for hour in range(24):
                inside=[n for n in names if observed(n).date()==day and observed(n).hour==hour]
                if inside:
                    target=dt.datetime(day.year,day.month,day.day,hour,14)
                    hourly.add(min(inside,key=lambda n:abs((observed(n)-target).total_seconds())))
            hourly_files=[]
            for name in sorted(hourly-chosen):
                rel=f'maps-hourly/{day:%Y/%m}/{name}'
                run.fetch_to(url+name,rel,replace=(day==TODAY))
                hourly_files.append(rel)
            status='complete' if len(files)>=4 else 'upstream_sparse' if files else 'upstream_empty'
            record={'day':str(day),'status':status,'url':url,'files':files,'grid':GRID,'hourly_files':hourly_files,
                    'upstream_names':len(names)}
        except Exception as exc:
            import urllib.error
            status='upstream_missing' if isinstance(exc,urllib.error.HTTPError) and exc.code==404 else 'failed'
            record={'day':str(day),'status':status,'url':url,'error':str(exc),'files':[]}
        save(receipt,(json.dumps(record)+'\n').encode());return record
    first=dt.date.fromisoformat(sys.argv[sys.argv.index('--start')+1]) if '--start' in sys.argv else dt.date(2022,10,1)
    span=sorted(days(first),reverse=True)
    records=[]
    for attempt in range(3):
        pending=span if attempt==0 else [dt.date.fromisoformat(r['day']) for r in records if r['status']=='failed']
        if not pending:break
        revised={}
        with cf.ThreadPoolExecutor(int(os.environ.get('GONG_WORKERS','4'))) as pool:
            for i,record in enumerate(pool.map(one,pending),1):
                revised[record['day']]=record
                if i%50==0:print('GONG',attempt+1,i,len(pending),flush=True)
        if attempt==0:records=list(revised.values())
        else:records=[revised.get(r['day'],r) for r in records]
    gaps=[r for r in records if r['status']!='complete']
    save(run.dir/'gaps.json',(json.dumps(gaps,indent=1)+'\n').encode())
    run.manifest({'status':'failed' if any(r['status']=='failed' for r in gaps) else 'complete_with_upstream_gaps' if gaps else 'complete',
        'selection':'maps/: nearest 00:14/06:14/12:14/18:14 UTC per day; maps-hourly/: per UTC hour the file nearest HH:14 (hourly-v1, 2026-10-15)',
        'availability_rule_v1':'max(observation+2h, HTTP Last-Modified)',
        'inventory_days':len(records),'upstream_gap_days':len(gaps),'builder':str(Path(__file__).resolve())})
    return int(any(r['status']=='failed' for r in gaps))

if __name__=='__main__':sys.exit(main())
