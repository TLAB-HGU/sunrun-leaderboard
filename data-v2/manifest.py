"""File integrity and coverage receipts; fetched time never means availability."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def coverage(path):
    if path.suffix != '.parquet':
        return None
    import pandas as pd
    import pyarrow.parquet as pq
    names = pq.read_schema(path).names
    cols = [c for c in ('timestamp_utc','slot','origin_last_input_utc','time','timestamp') if c in names]
    try:
        if cols:
            data = pd.read_parquet(path, columns=cols[:1])
            values=data[cols[0]] if cols[0] in data.columns else data.index
            times = pd.to_datetime(values, utc=True, errors='coerce')
        else:
            meta = pq.read_schema(path).metadata or {}
            pmeta = json.loads(meta.get(b'pandas', b'{}'))
            indices = pmeta.get('index_columns', [])
            if not indices or not isinstance(indices[0], str):
                return None
            data = pd.read_parquet(path, columns=[])
            if not isinstance(data.index, pd.DatetimeIndex):
                return None
            times = pd.to_datetime(data.index, utc=True)
        valid = times[~pd.isna(times)]
        if len(valid):
            return {'start_utc':str(valid.min()), 'end_utc':str(valid.max()), 'rows':len(data)}
    except (ValueError, TypeError, KeyError):
        return None
    return None

def write_manifest(folder, *, source=None, builder=None):
    folder = Path(folder)
    old_path = folder/'MANIFEST.json'
    old = json.loads(old_path.read_text()) if old_path.exists() else {}
    receipts = {}
    log = folder/'fetch-log.jsonl'
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                r=json.loads(line)
                if r.get('file'): receipts[r['file']]=r
            except json.JSONDecodeError:
                continue
    entries=[]
    for p in sorted(folder.rglob('*')):
        if not p.is_file() or p.name in ('MANIFEST.json','FILES.json','SHA256SUMS','fetch-log.jsonl') or '.tmp' in p.name or '.partial' in p.name:
            continue
        rel=str(p.relative_to(folder))
        receipt=receipts.get(rel,{})
        entry={'file':rel,'bytes':p.stat().st_size,'sha256':sha256(p)}
        for k in ('url','last_modified','fetched_at_utc'):
            if k in receipt:entry[k]=receipt[k]
        if p.suffix=='.parquet':
            cov=coverage(p)
            if cov:entry['coverage']=cov
        entries.append(entry)
    index=folder/'FILES.json'
    tmp=index.with_suffix('.tmp')
    tmp.write_text(json.dumps(entries,indent=1)+'\n');os.replace(tmp,index)
    old.update({'dataset':folder.name,'version':'v2','verified_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                'n_files':len(entries),'bytes':sum(e['bytes'] for e in entries),
                'file_index':'FILES.json','file_index_sha256':sha256(index),
                'fetched_at_is_not_availability':True})
    if source:old['source']=source
    if builder:old.update(builder=str(builder),builder_sha256=sha256(builder))
    tmp=old_path.with_suffix('.tmp');tmp.write_text(json.dumps(old,indent=1)+'\n');os.replace(tmp,old_path)
    return old

def write_champion_ready(store):
    store=Path(store)
    files=[store/'ch/ace.parquet',store/'ch/ch-hourly.parquet',store/'suvi-fusion/suvi-hourly-v2.parquet']
    files+=sorted((store/'ace-snapshot').glob('year=*/part.parquet'))
    files+=sorted((store/'suvi-meta').glob('year=*/part.parquet'))
    if not list((store/'ace-snapshot').glob('year=*/part.parquet')) or not list((store/'suvi-meta').glob('year=*/part.parquet')):
        raise FileNotFoundError('ACE and SUVI metadata partitions required')
    hashes={str(p.relative_to(store)):sha256(p) for p in files}
    receipt={'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'files':hashes,
             'version':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()}
    temp=store/'CHAMPION_READY.tmp';temp.write_text(json.dumps(receipt,indent=2)+'\n')
    os.replace(temp,store/'CHAMPION_READY.json')
    return receipt

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('folders',nargs='+');ap.add_argument('--verify',action='store_true');args=ap.parse_args()
    for folder in args.folders:
        if args.verify:
            folder=Path(folder);m=json.loads((folder/'MANIFEST.json').read_text())
            assert sha256(folder/'FILES.json')==m['file_index_sha256']
            for e in json.loads((folder/'FILES.json').read_text()):
                assert sha256(folder/e['file'])==e['sha256'],e['file']
            print(folder.name,'verified')
        else:
            result=write_manifest(folder);print(result['dataset'],result['n_files'])
