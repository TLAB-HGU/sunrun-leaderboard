"""Move exactly six unlaunched, dependency-free fits with both schedulers paused."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime,timezone
from prepare_campaign import ROOT,REMOTE,remote
import importlib.util
_spec=importlib.util.spec_from_file_location('complete_campaign_controller',ROOT/'controller.py')
_controller=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_controller)
atomic,ensure_scheduler=_controller.atomic,_controller.ensure_scheduler


def stop_local(script):
    matches=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v]
        except OSError:continue
        if args and Path(args[0]).name.startswith('python') and str(ROOT/script) in args:
            matches.append(int(p.name))
    if len(matches)!=1:raise RuntimeError(f'Expected exactly one {script}: {matches}')
    os.kill(matches[0],signal.SIGTERM)


def locks(root):
    handles=[]
    for name in ('.controller.lock','.scheduler.lock'):
        p=root/name
        for attempt in range(100):
            f=p.open('a')
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(f);break
            except BlockingIOError:f.close();time.sleep(.1)
        else:raise RuntimeError('Could not acquire '+name)
    return handles


def archive(root,tag):
    for name in ['scheduler-ablation.exit','scheduler-ablation-launch.json','scheduler-ablation.log','host-complete.json']:
        p=root/name
        if p.exists():p.rename(root/(name+'.'+tag))


def main():
    tag='rebalance-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    journal=ROOT/(tag+'.json')
    stop_local('controller.py');stop_local('ablation_scheduler.py')
    held=locks(ROOT)
    local=json.loads((ROOT/'assignment.json').read_text());state=json.loads((ROOT/'scheduler-state.json').read_text());allocation=json.loads((ROOT/'allocation.json').read_text());glob=json.loads((ROOT/'assignment.global.json').read_text());registry=json.loads((ROOT/'registry.json').read_text())
    dependencies={e['config'].get('reuse_run') for e in local['experiments']}
    selected=[e for e in local['experiments'] if state.get(e['experiment'],{}).get('state') in ('scheduled','waiting') and not e['config'].get('reuse_run') and e['experiment'] not in dependencies and not (ROOT/'runs'/e['experiment']).exists()][:6]
    if len(selected)!=6:raise RuntimeError('Fewer than six independent pending fits')
    names=[e['experiment'] for e in selected]
    if any(allocation['owners'][n]!='local' for n in names):raise RuntimeError('Local ownership mismatch')
    atomic(journal,dict(phase='prepared',names=names,previous={n:'local' for n in names},destination='golab2'))
    # Remote pause and assignment mutation are guarded by its exact scheduler lock.
    code=f'''import pathlib,json,os,signal,time,fcntl
root=pathlib.Path({REMOTE!r});pid=int((root/'scheduler.pid').read_text())
args=pathlib.Path(f'/proc/{{pid}}/cmdline').read_bytes().split(b'\\0')
if str(root/'ablation_scheduler.py').encode() not in args:raise RuntimeError('Remote PID mismatch')
os.kill(pid,signal.SIGTERM)
f=(root/'.scheduler.lock').open('a')
for i in range(100):
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);break
 except BlockingIOError:time.sleep(.1)
else:raise RuntimeError('Remote lock unavailable')
p=root/'assignment.json';a=json.loads(p.read_text());s=json.loads((root/'scheduler-state.json').read_text());selected={selected!r};names={names!r}
if any(e['experiment'] in names for e in a['experiments']):raise RuntimeError('Remote already owns selected fit')
if any((root/'runs'/n).exists() or n in s for n in names):raise RuntimeError('Remote selected output/state already exists')
(root/('assignment.json.'+{tag!r})).write_text(p.read_text())
for e in selected:e['execution_host']='golab2';a['experiments'].append(e)
tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(a,indent=2)+'\\n');tmp.replace(p)
for n in ['scheduler-ablation.exit','scheduler-ablation-launch.json','scheduler-ablation.log','host-complete.json']:
 p=root/n
 if p.exists():p.rename(root/(n+'.'+{tag!r}))
print(json.dumps(dict(moved=names)))
'''
    result=remote(code,timeout=30);print(result.stdout.decode())
    # Source is still locked: no experiment can launch between recheck and removal.
    if any((ROOT/'runs'/n).exists() for n in names):raise RuntimeError('Unexpected local launch during lock')
    for filename in ['assignment.json','assignment.global.json','allocation.json','registry.json','scheduler-state.json']:
        (ROOT/(filename+'.'+tag)).write_bytes((ROOT/filename).read_bytes())
    local['experiments']=[e for e in local['experiments'] if e['experiment'] not in names]
    for n in names:
        state.pop(n,None);registry.pop(n,None);allocation['owners'][n]='golab2'
    allocation['new_training_by_host']['local']-=6;allocation['new_training_by_host']['golab2']+=6
    for e in glob['experiments']:
        if e['experiment'] in names:e['execution_host']='golab2'
    for filename,value in [('assignment.json',local),('assignment.global.json',glob),('allocation.json',allocation),('registry.json',registry),('scheduler-state.json',state),('runtime.json',state)]:atomic(ROOT/filename,value)
    archive(ROOT,tag)
    atomic(journal,dict(phase='committed',names=names,source='local',destination='golab2',changed_only='execution ownership; experiment configurations unchanged'))
    for f in held:f.close()
    print(ensure_scheduler(ROOT,'local','ablation'));print(ensure_scheduler(ROOT,'golab2','ablation'))
    session='srac-global-controller-20261106'
    for i in range(50):
        if subprocess.run(['tmux','has-session','-t','='+session],capture_output=True).returncode:break
        time.sleep(.1)
    else:raise RuntimeError('Old central controller session still exists')
    p=ROOT/'controller.exit'
    if p.exists():p.rename(ROOT/('controller.exit.'+tag))
    subprocess.run(['tmux','new-session','-d','-s',session,'bash',str(ROOT/'run-controller.sh')],check=True)
    identifiers={l.split('|')[0]:l.split('|')[1] for l in subprocess.check_output(['tmux','list-sessions','-F','#{session_name}|#{session_id}'],text=True).splitlines()}
    atomic(ROOT/'controller-launch.json',dict(session=session,session_id=identifiers[session]))
    atomic(journal,dict(phase='resumed',names=names,source='local',destination='golab2'))
    print(json.dumps(dict(moved=names,count=6)))


if __name__=='__main__':main()
