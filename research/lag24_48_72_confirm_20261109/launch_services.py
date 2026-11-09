"""Start/resume only this campaign's local foreground supervisor and collector."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from scheduler_support import atomic, session_id, now


def start(root, python, script, name, resume=False):
    receipt = root / (script + '.launch.json')
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous['campaign_root'] != str(root):
            raise ValueError('Service root ownership mismatch')
        current = session_id(previous['tmux_session'])
        if not resume:
            raise RuntimeError('Existing receipt: use --resume for exact identity recovery')
        if current == previous['tmux_session_id']:
            return dict(state='already_running', **previous)
        if current is not None:
            raise RuntimeError('Service name belongs to a different session')
        archive = root / 'service-history' / (str(time.time_ns()) + '-' + script)
        archive.mkdir(parents=True)
        for suffix in ('.launch.json', '.log', '.sh', '.exit'):
            path = root / (script + suffix)
            if path.exists():
                shutil.move(path, archive / path.name)
    if session_id(name) is not None:
        raise RuntimeError('Refusing an existing session name')
    command = [python, '-u', str(root / script), '--root', str(root)]
    environment = ['env', 'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1']
    log, exitfile, shell = (root / (script + suffix) for suffix in ('.log', '.exit', '.sh'))
    shell.write_text('#!/bin/bash\ncd ' + shlex.quote(str(root)) + '\n' + shlex.join(environment + command)
          + ' > ' + shlex.quote(str(log)) + ' 2>&1\ncode=$?\nprintf \'%s\\n\' "$code" > '
          + shlex.quote(str(exitfile)) + '\nexit "$code"\n')
    subprocess.run(['tmux', 'new-session', '-d', '-s', name, 'bash', str(shell)], check=True)
    identifier = session_id(name)
    if identifier is None:
        raise RuntimeError('Started service lacks verifiable session identity')
    record = dict(campaign_root=str(root), tmux_session=name, tmux_session_id=identifier,
                  started_utc=now(), command=command)
    atomic(receipt, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    assignment = json.loads((root / 'assignment.global.json').read_text())
    if not json.loads((root / 'preflight.json').read_text()).get('passed'):
        raise ValueError('Validated freeze required before launch')
    if not json.loads((root / 'combo-smoke.json').read_text()).get('passed'):
        raise ValueError('Actual-data joint-feature smoke proof required before launch')
    for script, name in (('remote_scheduler_controller.py', 'srlag3-remote-controller-20261109'),
                         ('collector.py', 'srlag3-collect-controller-20261109')):
        print(json.dumps(start(root, assignment['python'], script, name, args.resume)), flush=True)


if __name__ == '__main__':
    main()
