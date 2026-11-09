"""Durable two-host ablation collection and once-frozen final confirmation."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time

from prepare_campaign import REMOTE, RPY, remote


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@contextmanager
def ownership(root):
    with (root / '.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def remote_status():
    code = f'''import json,pathlib
root=pathlib.Path({REMOTE!r})
def read(name,default):
 p=root/name
 return json.loads(p.read_text()) if p.exists() else default
print(json.dumps(dict(state=read('scheduler-state.json',{{}}), complete=read('host-complete.json',None))))
'''
    return json.loads(remote(code, timeout=60).stdout)


def ensure_scheduler(root, host, phase):
    """A recorded controller exit is authoritative; never restart failed jobs."""
    remote_root = REMOTE if host == 'golab2' else str(root)
    python = RPY if host == 'golab2' else read(root / 'assignment.json')['python']
    session = f'srac-{host}-{phase}-scheduler-20261106'
    code = f'''import json,pathlib,subprocess,shlex,re
def session_id(name):
 result=subprocess.run(['tmux','list-sessions','-F','#{{session_name}}|#{{session_id}}'],capture_output=True,text=True)
 for line in result.stdout.splitlines():
  session_name,separator,identifier=line.partition('|')
  if separator and session_name==name and re.fullmatch(r'\\$\\d+',identifier):return identifier
 return None
root=pathlib.Path({remote_root!r});session={session!r}
receipt=root/{('scheduler-'+phase+'-launch.json')!r}
exit_path=root/{('scheduler-'+phase+'.exit')!r}
if exit_path.exists():
 code=int(exit_path.read_text())
 if code:raise RuntimeError('Scheduler failed; inspect '+str(exit_path))
 print(json.dumps(dict(state='exited',exit_code=code)))
else:
 alive=subprocess.run(['tmux','has-session','-t','='+session],capture_output=True).returncode==0
 if receipt.exists():
  saved=json.loads(receipt.read_text())
  current=session_id(session)
  if not alive or not current or current!=saved.get('session_id'):raise RuntimeError('Recorded scheduler lost without matching session identity')
  print(json.dumps(dict(state='running',session=session)))
 elif alive:raise RuntimeError('Unowned scheduler session already exists')
 else:
  command=[{python!r},'-u',str(root/'ablation_scheduler.py'),'--root',str(root)]
  log=root/{('scheduler-'+phase+'.log')!r}
  script=root/{('scheduler-'+phase+'.sh')!r}
  script.write_text('#!/bin/bash\\ncd '+shlex.quote(str(root))+'\\n'+shlex.join(command)+' > '+shlex.quote(str(log))+' 2>&1\\ncode=$?\\nprintf \\'%s\\\\n\\' "$code" > '+shlex.quote(str(exit_path)+'.tmp')+'\\nmv '+shlex.quote(str(exit_path)+'.tmp')+' '+shlex.quote(str(exit_path))+'\\nexit "$code"\\n')
  subprocess.run(['tmux','new-session','-d','-s',session,'bash',str(script)],check=True)
  identifier=session_id(session)
  if not identifier:raise RuntimeError('Launched scheduler has no verifiable session ID')
  receipt.write_text(json.dumps(dict(session=session,session_id=identifier,command=command)))
  print(json.dumps(dict(state='started',session=session)))
'''
    if host == 'golab2':
        return json.loads(remote(code, timeout=60).stdout)
    result = subprocess.run([python, '-c', code], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def collect_completed(root, owners, remote_state, collected):
    names = [name for name, host in owners.items() if host == 'golab2'
             and remote_state.get(name, {}).get('state') == 'trained_validated' and name not in collected]
    if not names:
        return []
    for name in names:
        if Path(name).name != name:
            raise ValueError('Unsafe experiment name')
        destination = root / 'runs' / name
        if destination.exists() and not (destination / 'collected.json').exists():
            raise ValueError('Refusing to overwrite local run: ' + name)
    code = f'''import pathlib,sys,tarfile,json
root=pathlib.Path({REMOTE!r});names={names!r}
state=json.loads((root/'scheduler-state.json').read_text())
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name in names:
  out=root/'runs'/name
  if state[name]['state']!='trained_validated' or (out/'train.exit').read_text().strip()!='0':raise RuntimeError('Run no longer complete: '+name)
  for p in out.iterdir():
   if p.is_file() and p.suffix in ('.json','.npz','.parquet','.exit','.log','.py'):
    archive.add(p,arcname=name+'/'+p.name)
'''
    result = remote(code, timeout=600)
    with tempfile.TemporaryDirectory(prefix='.collect-', dir=root) as directory:
        staging = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:gz') as archive:
            for member in archive.getmembers():
                path = Path(member.name)
                if path.is_absolute() or '..' in path.parts or len(path.parts) != 2 or path.parts[0] not in names or not member.isfile():
                    raise ValueError('Unsafe result archive member')
            archive.extractall(staging, filter='data')
        for name in names:
            source = staging / name
            required = ['train.exit', 'complete.json', 'local_score.json', 'paired_no_offset_score.json',
                        'predictions.parquet', 'no_offset_predictions.parquet', 'launch.json']
            if any(not (source / n).is_file() for n in required) or (source / 'train.exit').read_text().strip() != '0':
                raise ValueError('Incomplete collected run: ' + name)
            destination = root / 'runs' / name
            if not destination.exists():
                atomic(source / 'collected.json', dict(host='golab2', source=REMOTE+'/runs/'+name, collected_utc=now()))
                source.replace(destination)
    return names


def complete(recipes, owners, local_state, remote_state):
    for recipe in recipes:
        name = recipe['experiment']
        state = remote_state if owners[name] == 'golab2' else local_state
        if state.get(name, {}).get('state') != 'trained_validated':
            return False
    return True


def analyze(root, final=False):
    name = 'final-analysis' if final else 'global-analysis'
    receipt = root / (name + '-status.json')
    previous = read(receipt, {})
    if previous.get('exit_code') == 0:
        return
    python = read(root / 'assignment.global.json')['python']
    command = [python, str(root / 'analyze_results.py'), '--root', str(root), '--bootstrap', '2000', '--block-hours', '168']
    if final:
        command.append('--final')
    atomic(receipt, dict(started_utc=now(), command=command))
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    with (root / (name + '.log')).open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
    atomic(receipt, dict(finished_utc=now(), command=command, exit_code=result.returncode))
    if result.returncode:
        raise RuntimeError(name + ' failed; inspect log')


def append_final_recipes(root, manifest):
    if manifest.get('status') != 'frozen' or len(manifest.get('final_experiments', [])) != 3:
        raise ValueError('Final confirmation requires frozen three-seed configuration')
    finals = manifest['final_experiments']
    if sorted(r.get('seed') for r in finals) != [42, 43, 44]:
        raise ValueError('Final confirmation must contain exactly seeds 42, 43, 44')
    for recipe in finals:
        if (recipe.get('family') != 'pooling' or recipe.get('stage') != 'ablation'
                or recipe.get('variant') != 'final_pruned'
                or recipe.get('config', {}).get('seed') != recipe['seed']):
            raise ValueError('Invalid final confirmation recipe')
    allocation = read(root / 'allocation.json')
    for recipe in finals:
        host = 'golab2' if recipe['seed'] == 43 else 'local'
        name = recipe['experiment']
        previous = allocation['owners'].get(name)
        if previous is not None and previous != host:
            raise ValueError('Final ownership mismatch')
        allocation['owners'][name] = host
    atomic(root / 'allocation.json', allocation)
    for host in ('local', 'golab2'):
        additions = [r for r in finals if allocation['owners'][r['experiment']] == host]
        if host == 'local':
            path = root / 'assignment.json'
            assignment = read(path)
            merge_recipes(assignment, additions)
            atomic(path, assignment)
            registry_path = root / 'registry.json'
            registry = read(registry_path, {})
            for recipe in additions:
                registry[recipe['experiment']] = dict(recipe, host='local', training_host='local')
            atomic(registry_path, registry)
        else:
            code = f'''import pathlib,json
root=pathlib.Path({REMOTE!r});path=root/'assignment.json';assignment=json.loads(path.read_text())
for recipe in {additions!r}:
 old=next((r for r in assignment['experiments'] if r['experiment']==recipe['experiment']),None)
 if old is not None and old!=recipe:raise ValueError('Final recipe changed')
 if old is None:assignment['experiments'].append(recipe)
temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(assignment,indent=2)+'\\n');temporary.replace(path)
'''
            remote(code, timeout=60)
    return allocation['owners']


def merge_recipes(assignment, additions):
    for recipe in additions:
        previous = next((r for r in assignment['experiments'] if r['experiment'] == recipe['experiment']), None)
        if previous is not None and previous != recipe:
            raise ValueError('Final recipe changed')
        if previous is None:
            assignment['experiments'].append(recipe)


def cleanup_scheduler_sessions(root):
    """Remove only retained dead panes with this controller's recorded session ID."""
    for host in ('local', 'golab2'):
        host_root = REMOTE if host == 'golab2' else str(root)
        code = f'''import pathlib,json,subprocess
root=pathlib.Path({host_root!r})
for phase in ('ablation','final'):
 p=root/('scheduler-'+phase+'-launch.json')
 if not p.exists():continue
 receipt=json.loads(p.read_text())
 sessions=subprocess.run(['tmux','list-sessions','-F','#{{session_name}}|#{{session_id}}'],capture_output=True,text=True)
 if not receipt.get('session_id') or receipt['session']+'|'+receipt['session_id'] not in sessions.stdout.splitlines():continue
 result=subprocess.run(['tmux','list-panes','-s','-t',receipt['session_id'],'-F','#{{session_id}} #{{pane_dead}} #{{session_windows}}'],capture_output=True,text=True)
 if result.returncode==0 and result.stdout.strip()==receipt['session_id']+' 1 1':
  subprocess.run(['tmux','kill-session','-t',receipt['session_id']],check=True)
'''
        if host == 'golab2':
            remote(code, timeout=60)
        else:
            subprocess.run([read(root / 'assignment.json')['python'], '-c', code], check=True)


def step(root, status):
    phase = status.get('phase', 'ablation')
    allocation = read(root / 'allocation.json')
    owners = allocation['owners']
    recipes = read(root / 'assignment.global.json')['experiments']
    if phase == 'final':
        recipes = read(root / 'analysis' / 'removal_manifest.json')['final_experiments']
    host_errors = {}
    for host in ('local', 'golab2'):
        try:
            ensure_scheduler(root, host, phase)
        except (RuntimeError, subprocess.SubprocessError) as error:
            host_errors[host] = str(error)
    if host_errors:
        status['host_errors'] = host_errors
    else:
        status.pop('host_errors', None)
    remote_snapshot = remote_status()
    atomic(root / 'remote-state.json', remote_snapshot)
    local_state = read(root / 'scheduler-state.json', {})
    remote_state = remote_snapshot['state']
    collected = set(status.get('collected', []))
    collected.update(collect_completed(root, owners, remote_state, collected))
    status.update(phase=phase, collected=sorted(collected), updated_utc=now())
    atomic(root / 'controller-state.json', status)
    failed = {r['experiment']: (remote_state if owners[r['experiment']] == 'golab2' else local_state).get(r['experiment'], {})
              for r in recipes if (remote_state if owners[r['experiment']] == 'golab2' else local_state).get(r['experiment'], {}).get('state') in {'failed', 'cancelled_by_user'}}
    if failed or host_errors:
        status['failures'] = failed
        # Preserve independent work while recording failures; do not declare completion.
        return
    if not complete(recipes, owners, local_state, remote_state):
        return
    # Ensure each host scheduler has finished and released its lock before final assignment changes.
    for host in ('local', 'golab2'):
        if ensure_scheduler(root, host, phase).get('state') != 'exited':
            return
    analyze(root, final=phase == 'final')
    if phase == 'ablation':
        manifest = read(root / 'analysis' / 'removal_manifest.json')
        if manifest.get('status') != 'frozen':
            status.update(phase='blocked', reason=manifest.get('status', 'removal manifest unavailable'))
            return
        append_final_recipes(root, manifest)
        status.update(phase='final', final_started_utc=now())
    else:
        cleanup_scheduler_sessions(root)
        status.update(phase='complete', completed_utc=now())
        atomic(root / 'campaign-complete.json', dict(status, state='complete'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.interval <= 60:
        parser.error('interval must be between 1 and 60 seconds')
    root = args.root.resolve()
    with ownership(root):
        path = root / 'controller-state.json'
        status = read(path, dict(phase='ablation', collected=[], started_utc=now()))
        while status['phase'] not in {'complete', 'blocked'}:
            try:
                step(root, status)
                status.pop('last_error', None)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                status.update(last_error=str(error), error_utc=now())
                print(json.dumps(dict(error=str(error), utc=now())), flush=True)
            atomic(path, status)
            print(json.dumps(dict(phase=status['phase'], collected=len(status.get('collected', [])), utc=now())), flush=True)
            if args.once:
                break
            if status['phase'] not in {'complete', 'blocked'}:
                time.sleep(args.interval)


if __name__ == '__main__':
    main()
