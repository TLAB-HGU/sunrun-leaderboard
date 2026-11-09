"""Launch exact-owned durable services once, without touching existing sessions."""
from pathlib import Path
import subprocess
from transport_support import remote, REMOTE, RPY

ROOT = Path(__file__).resolve().parent
PYTHON = str(ROOT.parents[1] / '.venv/bin/python')


def start(root, python, name, script, distant=False):
    code = f'''import json,shlex,subprocess,re
from pathlib import Path
root=Path({str(root)!r});name={name!r};receipt=root/({script!r}+'.launch.json')
if receipt.exists():raise RuntimeError('Service already launched')
if subprocess.run(['tmux','has-session','-t','='+name],capture_output=True).returncode==0:raise RuntimeError('Session collision')
command=[{python!r},'-u',str(root/{script!r}),'--root',str(root)]
log=root/({script!r}+'.log');exitfile=root/({script!r}+'.exit');shell=root/({script!r}+'.sh')
shell.write_text('#!/bin/bash\\ncd '+shlex.quote(str(root))+'\\n'+shlex.join(command)+' > '+shlex.quote(str(log))+' 2>&1\\ncode=$?\\nprintf \\'%s\\\\n\\' "$code" > '+shlex.quote(str(exitfile))+'\\nexit "$code"\\n')
subprocess.run(['tmux','new-session','-d','-s',name,'bash',str(shell)],check=True)
rows=subprocess.check_output(['tmux','list-sessions','-F','#{{session_name}}|#{{session_id}}'],text=True).splitlines()
identifier=next((r.split('|')[1] for r in rows if r.split('|')[0]==name),None)
if not identifier or not re.fullmatch(r'\\$\\d+',identifier):raise RuntimeError('No session identity')
receipt.write_text(json.dumps(dict(session=name,session_id=identifier,command=command)))
print(json.dumps(dict(session=name,session_id=identifier)))
'''
    if distant:
        print(remote(code, timeout=30).stdout.decode())
    else:
        print(subprocess.check_output([PYTHON, '-c', code], text=True))


if __name__ == '__main__':
    start(REMOTE, RPY, 'sr2drop-golab2-scheduler-20261108', 'host_scheduler.py', True)
    start(ROOT, PYTHON, 'sr2drop-local-scheduler-20261108', 'host_scheduler.py')
    start(ROOT, PYTHON, 'sr2drop-controller-20261108', 'controller.py')
