"""Launch four independent tmux lanes with durable user services."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
PYTHON='/home/t-lab01/sunrun/.venv/bin/python'
METHODS=['event','context','outputwave','quantile']


def shell(args):
    return shlex.join([str(x) for x in args])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id:
        parser.error('run-id must be a directory name')
    existing=subprocess.run(['tmux','list-sessions','-F','#S'],capture_output=True,text=True).stdout.splitlines()
    for method in METHODS:
        if 'fs30v2-'+method in existing:
            raise RuntimeError('Existing session preserved: fs30v2-'+method)
    receipt=[]
    for method in METHODS:
        session='fs30v2-'+method
        unit=session+'-'+args.run_id.replace('_','-')
        directory=HERE/'runs'/args.run_id/method;directory.mkdir(parents=True,exist_ok=True)
        command=shell([PYTHON,HERE/'run.py',method,'--run-id',args.run_id])
        if method=='event':
            command=shell([PYTHON,HERE/'run.py','prepare','--run-id',args.run_id])+' && '+command
        logfile=directory/'console.log'
        service=['systemd-run','--user','--unit='+unit,'--working-directory='+str(HERE),
                 '--property=CPUQuota=300%','--setenv=OMP_NUM_THREADS=3','--setenv=OPENBLAS_NUM_THREADS=1',
                 '--setenv=MKL_NUM_THREADS=1','--setenv=CUDA_VISIBLE_DEVICES=',
                 '--property=StandardOutput=append:'+str(logfile),
                 '--property=StandardError=append:'+str(logfile),'/bin/bash','-c',command]
        pane=(shell(service)+'; started=$?; if [ "$started" -ne 0 ]; then '+command+' 2>&1 | tee -a '+shell([logfile])+'; '
              'else service_pid=$(systemctl --user show '+shell([unit])+' --property=MainPID --value); '
              'if [ "$service_pid" -gt 0 ]; then tail --pid="$service_pid" -n 30 -F '+shell([logfile])+'; '
              'else tail -n 30 '+shell([logfile])+'; fi; fi; '
              'echo "Experiment process ended; check status.json / results.json"; exec /bin/bash')
        subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(HERE),'/bin/bash','-c',pane],check=True)
        receipt.append(dict(method=method,session=session,unit=unit,directory=str(directory)))
    output=HERE/'runs'/args.run_id/'launch.json'
    output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(output.read_text())


if __name__=='__main__':
    main()
