"""Status-registered durable runs, each with its own tmux pane."""
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON = ROOT / '.venv/bin/python'
STATUS = ROOT / 'leaderboard/lbstatus.py'
LANES = dict(hyper_gain='e2000fs_feat30_hyper_gain',
             gaussian='e2000fs_feat30_gaussian_pilot',
             mc_calibration='e2000fs_feat30_mc_calibration')


def call(args, check=True):
    return subprocess.run([str(x) for x in args], capture_output=True, text=True, check=check)


def launch():
    root = HERE / 'runs/full_20261028_v1'
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / 'launch.json'
    if receipt.exists():
        raise RuntimeError('Launch receipt exists; preserving prior run')
    entries = []
    for lane, experiment in LANES.items():
        session = 'fs30-v2-'+lane.replace('_', '-')
        unit = session+'-20261028-v1'
        state = call(['systemctl', '--user', 'show', unit+'.service', '--property=LoadState'], False)
        if state.stdout.strip() != 'LoadState=not-found':
            raise RuntimeError(f'Existing/uninspectable service: {unit}')
        if call(['tmux', 'has-session', '-t', '='+session], False).returncode == 0:
            raise RuntimeError(f'Existing session: {session}')
        entries.append(dict(lane=lane, experiment=experiment, session=session, unit=unit, status='pending'))
    def save():
        temporary = receipt.with_suffix('.tmp')
        temporary.write_text(json.dumps(entries, indent=2)+'\n')
        temporary.replace(receipt)
    save()
    for entry in entries:
        # Reassert required waiting status immediately before launch.
        call([PYTHON, STATUS, 'set', entry['experiment'], '--status', '대기',
              '--desc', entry['lane']+'; frozen seasonal and actual-evaluation transfer gates'])
        folder = root / entry['lane']
        folder.mkdir(exist_ok=True)
        log = folder / 'run.log'
        argv = [str(PYTHON), str(STATUS), 'run', entry['experiment'], '--',
                str(PYTHON), '-u', str(HERE/'run.py'), '--lane', entry['lane']]
        entry.update(command=argv, log=str(log))
        save()
        result = call(['systemd-run', '--user', '--unit='+entry['unit'],
                       '--working-directory='+str(ROOT), '--property=CPUQuota=200%',
                       '--setenv=OMP_NUM_THREADS=2', '--setenv=OPENBLAS_NUM_THREADS=2',
                       '--setenv=MKL_NUM_THREADS=2',
                       '--property=StandardOutput=append:'+str(log),
                       '--property=StandardError=append:'+str(log), *argv], False)
        if result.returncode:
            entry.update(status='launch_failed', error=result.stderr)
            save()
            subprocess.run(argv, cwd=ROOT, check=True)
            entry['status'] = 'foreground_completed'
        else:
            entry['status'] = 'started'
        call(['tmux', 'new-session', '-d', '-s', entry['session'], '-c', ROOT,
              'exec '+shlex.join(['tail','-n','40','-F','--',str(log)])])
        entry['pane'] = call(['tmux','list-panes','-t',entry['session'],'-F','#{pane_id}']).stdout.strip()
        save()
        print(json.dumps(entry), flush=True)


if __name__ == '__main__':
    launch()
