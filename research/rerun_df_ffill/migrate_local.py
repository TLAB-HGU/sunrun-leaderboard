"""Move verified inactive remote checkpoints to local CPU, retaining original metadata."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from datetime import UTC,datetime
import launch
import controller

HERE=Path(__file__).resolve().parent


def rebase(value,old_root,new_root,old_identity,new_identity):
    if isinstance(value,dict):
        if value==old_identity:
            return copy.deepcopy(new_identity)
        return {k:rebase(v,old_root,new_root,old_identity,new_identity) for k,v in value.items()}
    if isinstance(value,list):
        return [rebase(v,old_root,new_root,old_identity,new_identity) for v in value]
    if isinstance(value,str) and (value==old_root or value.startswith(old_root+'/')):
        return new_root+value[len(old_root):]
    return value


def verify_input(records):
    data=HERE/'input/df_ffill_trainy.parquet'
    stat=data.stat()
    local={'path':str(data),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    code="from pathlib import Path\nimport json\np=Path("+repr(launch.REMOTE+'/input/df_ffill_trainy.parquet')+")\ns=p.stat()\nprint(json.dumps(dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)))\n"
    remote=json.loads(launch.remote(launch.REMOTE_PYTHON+' -',input=code.encode(),capture_output=True).stdout)
    assert all(r['input_identity']==remote for r in records.values()), 'Remote snapshot changed since launch'
    assert (local['bytes'],local['mtime_ns'])==(225755615,1793684739007946726), 'Local prepared snapshot changed'
    assert remote['bytes']==local['bytes'] and local['mtime_ns']-remote['mtime_ns']==26, 'Unexpected deployment identity'
    return local


def assert_inactive(names):
    script='''from pathlib import Path
import json,subprocess
names=NAMES
root=Path(ROOT)
for name in names:
 p=root/'runs'/name
 assert subprocess.run(['tmux','has-session','-t','srff-'+name],capture_output=True).returncode!=0, name+' session alive'
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:args=(proc/'cmdline').read_bytes().decode().split(chr(0))
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  assert not any(str(p/f) in args for f in ['experiment.py','resume_run.py','run.sh']), name+' process alive'
print(json.dumps({'inactive':names}))
'''.replace('NAMES',repr(names)).replace('ROOT',repr(launch.REMOTE))
    return json.loads(launch.remote(launch.REMOTE_PYTHON+' -',input=script.encode(),capture_output=True).stdout)


def migrate_record(name,record,new_identity):
    out=HERE/'runs'/name
    if record['family'] not in ('calib11','top30'):
        raise ValueError('Unsupported partial model migration: '+name)
    original=out/'migration_remote_metadata'
    original.mkdir(exist_ok=False)
    rewritten=[]
    for path in out.glob('*.json'):
        value=json.loads(path.read_text())
        updated=rebase(value,launch.REMOTE,str(HERE),record['input_identity'],new_identity)
        if value!=updated or path.name=='launch.json':
            (original/path.name).write_bytes(path.read_bytes())
            if path.name!='launch.json':
                path.write_text(json.dumps(updated,indent=2)+'\n')
                rewritten.append(path.name)
    local=rebase(record,launch.REMOTE,str(HERE),record['input_identity'],new_identity)
    local.update(host='local',gpu=None,gpu_lease=None)
    local['command'][0]=launch.LOCAL_PYTHON
    local['migration']={'from':'golab2','utc':datetime.now(UTC).isoformat(),'input_verification':'original campaign deployment identity; equal225755615bytes; localmtime minus remotemtime26ns (tar precision); full content comparison not repeated',
                        'original_launch':'migration_remote_metadata/launch.json','rebased_metadata':rewritten}
    (out/'launch.json').write_text(json.dumps(local,indent=2)+'\n')
    for filename in ('train.exit','train.log','run.sh'):
        path=out/filename
        if path.exists():path.rename(original/filename)
    launch.install_resume(local)
    source=out/'experiment.py'
    if source.read_bytes()!=(out/'runner_at_launch.py').read_bytes():
        raise ValueError('Frozen source mismatch after migration')
    return local['migration']


def main():
    records={n:r for n,r in controller.records().items() if r['host']=='golab2' and not (HERE/'runs'/n/'submission.json').exists()}
    assert_inactive(list(records))
    new_identity=verify_input(records)
    receipt={'utc':datetime.now(UTC).isoformat(),'input_verification':'original launch provenance, unchanged bytes/mtime; tar timestamp precision delta26ns; no full content comparison','collected_completed':[], 'migrated':{}}
    for name,record in records.items():
        assert_inactive([name])
        controller.collect(name)
        out=HERE/'runs'/name
        if (out/'train.exit').exists() and (out/'train.exit').read_text().strip()=='0':
            receipt['collected_completed'].append(name)
            continue
        receipt['migrated'][name]=migrate_record(name,record,new_identity)
    assert_inactive(list(records))
    (HERE/'local-migration.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
