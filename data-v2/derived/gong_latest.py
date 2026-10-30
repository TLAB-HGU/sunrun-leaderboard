"""Refresh newly collected recent GONG receipts while historical backfill runs."""
import json
from pathlib import Path
from build import V2,atomic_json
from gong import prepare
root=V2/'gong-mrzqs';dest=V2/'gong-derived/receipts';dest.mkdir(parents=True,exist_ok=True)
rows={}
for line in (root/'fetch-log.jsonl').read_text().splitlines():
 try:r=json.loads(line)
 except json.JSONDecodeError:continue
 if '/2026/' in r.get('file','') and r.get('sha256'):rows[r['file']]=r
errors=[]
for r in sorted(rows.values(),key=lambda x:x['file'],reverse=True):
 try:prepare(r,root,dest)
 except Exception as e:errors.append({'file':r['file'],'error':str(e)})
atomic_json(dest.parent/'recent-receipts.json',{'rows':len(rows),'errors':errors})
print(len(rows),len(errors))
