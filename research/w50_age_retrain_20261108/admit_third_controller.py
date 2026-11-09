"""One-time resource-only admission of the already-registered third seed."""
import json
import os
import shlex
import signal
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCHEDULER_PID = 448


def main():
    assignment = json.loads((ROOT / 'assignment.json').read_text())
    assert assignment['host'] == 'golab2'
    expected = str(ROOT / 'host_scheduler.py')
    argv = (Path('/proc') / str(SCHEDULER_PID) / 'cmdline').read_bytes().decode().split('\0')
    assert expected in argv, 'Exact scheduler ownership could not be verified'
    assert {r['seed'] for r in assignment['experiments']} == {42, 43, 44}
    os.kill(SCHEDULER_PID, signal.SIGSTOP)
    try:
        jobs = []
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                args = (proc / 'cmdline').read_bytes().decode().split('\0')
                if 'python' in Path(args[0]).name and '--out' in args and '--threads' in args:
                    jobs.append({'pid': int(proc.name), 'threads': int(args[args.index('--threads') + 1])})
            except (OSError, ValueError, IndexError):
                continue
        available = next(line for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        resources = {'jobs': jobs, 'available_gib': int(available.split()[1]) / 1024**2,
                     'load1': os.getloadavg()[0]}
        assert len(jobs) < 6 and sum(job['threads'] for job in jobs) + 4 <= 24
        assert resources['available_gib'] >= 11 and resources['load1'] + 4 <= 24
        name = 'age_retrain_s44'
        out = ROOT / 'runs' / name
        out.mkdir(parents=True, exist_ok=True)
        assert not (out / 'launch.json').exists() and not list(out.glob('F*_models'))
        parent = json.loads((ROOT / 'runs/age_retrain_s42/launch.json').read_text())
        command = parent['command'].copy()
        for flag, value in [('--seed', '44'), ('--experiment', name), ('--out', str(out))]:
            command[command.index(flag) + 1] = value
        assert command[command.index('--threads') + 1] == '4'
        assert command[command.index('--device') + 1] == 'cpu'
        session = 'srw50-' + name + '-20261108'
        launch = {'family': 'pooling', 'experiment': name, 'tmux_session': session,
                  'started_utc': datetime.now(UTC).isoformat(), 'host': 'golab2',
                  'training_host': 'golab2', 'campaign_root': str(ROOT), 'command': command,
                  'resources_before': resources, 'admission_amendment': 'strategy.md resource-only amendment; same three seeds'}
        with (out / 'launch.json').open('x') as stream:
            json.dump(launch, stream, indent=2)
        script = out / 'train.sh'
        script.write_text('#!/bin/bash\nset -u\ncd ' + shlex.quote(str(ROOT)) + '\n'
            + 'export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n'
            + shlex.join(command) + ' > ' + shlex.quote(str(out / 'train.log')) + ' 2>&1\n'
            + 'code=$?\nprintf "%s\\n" "$code" > ' + shlex.quote(str(out / 'train.exit.tmp'))
            + '\nmv ' + shlex.quote(str(out / 'train.exit.tmp')) + ' ' + shlex.quote(str(out / 'train.exit'))
            + '\nexit "$code"\n')
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, 'bash ' + shlex.quote(str(script))], check=True)
        assignment['max_jobs'] = 3
        assignment['resource_amendment'] = launch['admission_amendment']
        (ROOT / 'assignment.json').write_text(json.dumps(assignment, indent=2) + '\n')
        print(json.dumps(launch), flush=True)
    finally:
        os.kill(SCHEDULER_PID, signal.SIGCONT)


if __name__ == '__main__':
    main()
