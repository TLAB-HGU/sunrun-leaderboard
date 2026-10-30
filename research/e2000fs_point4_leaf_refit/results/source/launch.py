"""Start status-wrapped point4 experiments in independent durable tmux lanes."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON = ROOT / '.venv/bin/python'
STATUS = ROOT / 'leaderboard/lbstatus.py'
LANES = dict(baseline='e2000fs_point4_seasonal_baseline',
             linear_leaf='e2000fs_point4_linear_leaf',
             leaf_refit='e2000fs_point4_leaf_refit',
             stable_retrain='e2000fs_point4_stable_retrain')


def call(argv, check=True):
    return subprocess.run([str(x) for x in argv], capture_output=True, text=True, check=check)


def launch(lane):
    folder = HERE / 'runs/full_20261029_v1' / lane
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / 'launch.json'
    if receipt.exists():
        raise RuntimeError('Existing launch receipt preserved')
    session = 'fs30-p4-'+lane.replace('_','-')
    unit = session+'-20261029-v1'
    state = call(['systemctl','--user','show',unit+'.service','--property=LoadState'], False)
    if state.stdout.strip() != 'LoadState=not-found':
        raise RuntimeError('Existing or uninspectable service')
    if call(['tmux','has-session','-t','='+session], False).returncode == 0:
        raise RuntimeError('Existing tmux session')
    experiment = LANES[lane]
    call([PYTHON,STATUS,'set',experiment,'--status','대기','--desc',
          lane+'; direct pointweight4 parent; ordinary improvement with high-speed preservation'])
    target = [PYTHON,'-u',HERE/('baseline.py' if lane == 'baseline' else 'run.py')]
    if lane != 'baseline':
        target += ['--lane',lane]
    command = [PYTHON,STATUS,'run',experiment,'--',*target]
    log = folder/'run.log'
    record = dict(lane=lane,experiment=experiment,session=session,unit=unit,
                  command=[str(v) for v in command],log=str(log),status='launching')
    def save():
        temp = receipt.with_suffix('.tmp')
        temp.write_text(json.dumps(record,indent=2)+'\n')
        temp.replace(receipt)
    save()
    result = call(['systemd-run','--user','--unit='+unit,'--working-directory='+str(ROOT),
                   '--property=CPUQuota=200%','--setenv=OMP_NUM_THREADS=2',
                   '--setenv=OPENBLAS_NUM_THREADS=2','--setenv=MKL_NUM_THREADS=2',
                   '--property=StandardOutput=append:'+str(log),
                   '--property=StandardError=append:'+str(log),*command], False)
    if result.returncode:
        record.update(status='launch_failed',error=result.stderr)
        save()
        subprocess.run([str(v) for v in command],cwd=ROOT,check=True)
        record['status'] = 'foreground_completed'
    else:
        record['status'] = 'started'
    call(['tmux','new-session','-d','-s',session,'-c',ROOT,
          'exec '+shlex.join(['tail','-n','40','-F','--',str(log)])])
    record['pane'] = call(['tmux','list-panes','-t',session,'-F','#{pane_id}']).stdout.strip()
    save()
    print(json.dumps(record),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane',required=True,choices=list(LANES))
    launch(parser.parse_args().lane)
