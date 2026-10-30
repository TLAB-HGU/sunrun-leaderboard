"""Launch four isolated tmux sessions backed by durable user services."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PYTHON = '/home/t-lab01/sunrun/.venv/bin/python'
METHODS = ['shape', 'wavelet', 'extreme', 'probtraj']


def shell(arguments):
    return shlex.join([str(a) for a in arguments])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        parser.error('run-id must be a directory name')
    sessions = subprocess.run(['tmux', 'list-sessions', '-F', '#S'], capture_output=True, text=True).stdout.splitlines()
    for method in METHODS:
        if 'fs30-' + method in sessions:
            raise RuntimeError(f'Existing session preserved: fs30-{method}')
    receipts = []
    for method in METHODS:
        session = 'fs30-' + method
        unit = session + '-' + args.run_id.replace('_', '-')
        directory = HERE / 'runs' / args.run_id / method
        directory.mkdir(parents=True, exist_ok=True)
        command = shell([PYTHON, HERE / 'run.py', method, '--run-id', args.run_id])
        if method == 'shape':
            command = shell([PYTHON, HERE / 'run.py', 'prepare', '--run-id', args.run_id]) + ' && ' + command
        service = ['systemd-run', '--user', '--unit=' + unit, '--working-directory=' + str(HERE),
                   '--property=CPUQuota=300%', '--setenv=OMP_NUM_THREADS=3', '--setenv=OPENBLAS_NUM_THREADS=1',
                   '--setenv=MKL_NUM_THREADS=1', '--setenv=CUDA_VISIBLE_DEVICES=',
                   '--property=StandardOutput=append:' + str(directory / 'console.log'),
                   '--property=StandardError=append:' + str(directory / 'console.log'),
                   '/bin/bash', '-c', command]
        # A single failed launch falls back immediately to foreground inside tmux.
        # No process or session belonging to another task is altered.
        pane = (shell(service) + '; started=$?; '
                'if [ "$started" -ne 0 ]; then ' + command + ' 2>&1 | tee -a ' + shell([directory / 'console.log']) + '; '
                'else service_pid=$(systemctl --user show ' + shell([unit]) + ' --property=MainPID --value); '
                'if [ "$service_pid" -gt 0 ]; then tail --pid="$service_pid" -n 30 -F ' + shell([directory / 'console.log']) + '; '
                'else tail -n 30 ' + shell([directory / 'console.log']) + '; fi; fi; '
                'echo "Experiment process ended; inspect status.json and results.json"; exec /bin/bash')
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(HERE), '/bin/bash', '-c', pane], check=True)
        receipts.append(dict(method=method, session=session, unit=unit, directory=str(directory)))
    path = HERE / 'runs' / args.run_id / 'launch.json'
    path.write_text(json.dumps(receipts, indent=2) + '\n')
    print(path.read_text())


if __name__ == '__main__':
    main()
