#!/usr/bin/env python3
"""Serialized, resumable data refresh with explicit per-job outcomes."""
import argparse
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from paths import ROOT, LB, STORE, DATASETS

PY = LB / '.venv-fusion/bin/python'

def jobs(mode):
    if mode == 'champion':
        return [('champion', [str(PY), str(ROOT/'builders/champion.py'), '--workers',os.environ.get('DATAV2_WORKERS','4')])]
    if mode == 'derived':
        records=[]
        for registry in (ROOT/'derived/registry.json',ROOT/'suvi-derived/registry.json'):
            if not registry.exists():
                raise FileNotFoundError(f'derived registry is not installed: {registry}')
            data=json.loads(registry.read_text())
            records.extend(data.get('jobs',[]) if isinstance(data,dict) else data)
        return [(r['name'],r['cmd']) for r in records]
    if mode == 'raw':
        result=[('donki',[str(PY),str(ROOT/'builders/donki.py')])]
        # 2026-10-15: worker-collected sources removed from SolarWind DB are no longer refreshed.
        # 2026-10-20 user: SWPC SRS added (SW-065). ENLIL iSWA joins after its backfill succeeds.
        # 2026-10-28 user: SWPC forecast discussion (SW-042) removed.
        for name in ('srs','lasco_c2'):
            result.append((name,[str(PY),str(ROOT/'builders/raw_sources.py'),name,'--seed']))
        result.append(('gong',[str(PY),str(ROOT/'builders/gong_resume.py')]))
        return result
    raise ValueError(mode)

def run(mode):
    STORE.mkdir(parents=True,exist_ok=True)
    logs=ROOT/'logs'; logs.mkdir(exist_ok=True)
    statuses=[]
    with (STORE/'.refresh.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({'status':'skipped','reason':'another refresh or snapshot is active'}));return 75
        # Legacy one-off jobs predate the shared lock; avoid conflicting with them.
        active=subprocess.run(['systemctl','--user','list-units','datav2-*','data-v2-remaining-*','--state=running','--no-legend','--plain'],
                              text=True,capture_output=True)
        if active.returncode==0 and active.stdout.strip():
            print(json.dumps({'status':'skipped','reason':'initial datav2 build still running'}));return 75
        env=os.environ.copy()
        env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        for name,command in jobs(mode):
            started=dt.datetime.now(dt.timezone.utc).isoformat()
            with (logs/f'refresh-{mode}-{name}.log').open('a') as out:
                out.write(f'\n{started} {json.dumps(command)}\n');out.flush()
                try:
                    result=subprocess.run(command,cwd=LB,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=86400)
                    record={'job':name,'started_utc':started,'returncode':result.returncode}
                except subprocess.TimeoutExpired:
                    record={'job':name,'started_utc':started,'returncode':124,'error':'timeout'}
            record['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat();statuses.append(record)
            target=ROOT/'reports'/f'refresh-{mode}.json'
            target.parent.mkdir(exist_ok=True)
            temp=target.with_suffix('.tmp');temp.write_text(json.dumps(statuses,indent=2)+'\n');os.replace(temp,target)
        from manifest import write_manifest
        # Hash only outputs of this mode; never race a concurrent initial download.
        names=('ace-snapshot','suvi-meta','ch','suvi-fusion') if mode=='champion' else DATASETS if mode=='raw' else ()
        for name in names:
            folder=STORE/name
            if folder.exists():write_manifest(folder)
    print(json.dumps(statuses));return int(any(s['returncode'] for s in statuses))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('raw','champion','derived'));args=ap.parse_args()
    sys.exit(run(args.mode))
