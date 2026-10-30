"""Launch one durable service and tmux pane per transfer experiment."""
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON = ROOT / '.venv/bin/python'
LANES = ('baseline', 'center', 'recency', 'recent_window')


def command(args, check=True):
    return subprocess.run(args, check=check, text=True, capture_output=True)


def launch():
    out = HERE / 'runs/full_20261028_v1'
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / 'launch.json'
    if receipt_path.exists():
        raise RuntimeError('Existing launch receipt preserved')
    entries = []
    for lane in LANES:
        suffix = lane.replace('_', '-')
        unit = f'fs30-transfer-{suffix}-20261028-v1'
        session = f'fs30-transfer-{suffix}'
        state = command(['systemctl', '--user', 'show', unit+'.service', '--property=LoadState'], check=False)
        if state.stdout.strip() != 'LoadState=not-found':
            raise RuntimeError(f'Existing or uninspectable service: {unit}')
        if command(['tmux', 'has-session', '-t', '='+session], check=False).returncode == 0:
            raise RuntimeError(f'Existing tmux session: {session}')
        entries.append(dict(lane=lane, unit=unit, session=session, status='pending'))
    receipt = dict(status='launching', lanes=entries)
    def save():
        temporary = receipt_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(receipt, indent=2)+'\n')
        temporary.replace(receipt_path)
    save()
    for entry in entries:
        lane = entry['lane']
        lane_dir = out / lane
        lane_dir.mkdir(exist_ok=True)
        log = lane_dir / 'run.log'
        argv = [str(PYTHON), '-u', str(HERE/'run.py'), '--lane', lane]
        entry.update(command=argv, log=str(log))
        save()
        service = command(['systemd-run', '--user', '--unit='+entry['unit'],
                           '--working-directory='+str(ROOT), '--property=CPUQuota=200%',
                           '--setenv=OMP_NUM_THREADS=2', '--setenv=OPENBLAS_NUM_THREADS=2',
                           '--setenv=MKL_NUM_THREADS=2',
                           '--property=StandardOutput=append:'+str(log),
                           '--property=StandardError=append:'+str(log), *argv], check=False)
        if service.returncode:
            entry.update(status='launch_failed', error=service.stderr)
            save()
            # Workspace contract: one failed background launch immediately runs foreground.
            subprocess.run(argv, cwd=ROOT, check=True)
            entry['status'] = 'foreground_completed'
        else:
            entry['status'] = 'started'
        monitor = 'exec '+shlex.join(['tail', '-n', '40', '-F', '--', str(log)])
        command(['tmux', 'new-session', '-d', '-s', entry['session'], '-c', str(ROOT), monitor])
        entry['pane'] = command(['tmux', 'list-panes', '-t', entry['session'], '-F', '#{pane_id}']).stdout.strip()
        save()
        print(json.dumps(entry), flush=True)
    receipt['status'] = 'started'
    save()


if __name__ == '__main__':
    launch()
