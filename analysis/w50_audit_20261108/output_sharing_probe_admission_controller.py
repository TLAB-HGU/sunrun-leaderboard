"""Admit one no-fit probe through the existing shared host resource policy."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PROBE = ROOT / 'output_sharing_probe'
OUT = PROBE / 'run01'
SESSION = 'srw50-output-sharing-probe-controller-20261109'
ADMISSION_LOCK = Path('/tmp/sunrun-cpu-admission.lock')
sys.path.insert(0, str(REPO / 'research/feature_addition_all_20261108'))
import host_scheduler as host
import scheduler_support as support


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true', help='Inspect admission without starting the probe.')
    args = parser.parse_args(argv)
    PROBE.mkdir(exist_ok=True)
    with (PROBE / '.admission-controller.lock').open('a') as ownership:
        fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.once and (OUT / 'launch.json').exists():
            raise ValueError('Existing probe launch must be inspected, never restarted automatically')
        while True:
            with ADMISSION_LOCK.open('a') as admission:
                fcntl.flock(admission, fcntl.LOCK_EX)
                jobs, memory, busy = host.workloads(), support.memory_gib(), support.busy_cpu_cores()
                measured = {'utc': datetime.now(UTC).isoformat(), 'controller_pid': os.getpid(),
                            'jobs': jobs, 'declared_threads': sum(j['threads'] for j in jobs),
                            'available_gib': memory, 'busy_cpu_cores': busy,
                            'load1': os.getloadavg()[0], 'requested_threads': 1,
                            'eligible': host.capacity(jobs, memory, busy, threads=1)}
                if args.once:
                    print(json.dumps(measured), flush=True)
                    return 0
                if measured['eligible']:
                    OUT.mkdir(exist_ok=False)
                    command = [sys.executable, '-u', str(ROOT / 'output_sharing_probe_controller.py'),
                               '--device', 'cpu', '--threads', '1', '--out', str(OUT)]
                    launch = dict(measured, family='diagnostic', kind='training_only_inference_no_fit',
                                  tmux_session=SESSION, started_utc=measured['utc'], host='local',
                                  training_host='not_applicable', device='cpu', threads=1, command=command,
                                  fresh_training=False)
                    write(OUT / 'launch.json', launch)
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1',
                                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                    log = (OUT / 'probe.log').open('w')
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
                    launch['pid'] = child.pid
                    write(OUT / 'launch.json', launch)
                    # The shared lock stays held until other schedulers can see this exact live job.
                    for _ in range(100):
                        if any(row['pid'] == child.pid for row in host.workloads()):
                            break
                        if child.poll() is not None:
                            break
                        time.sleep(.02)
                    else:
                        child.terminate()
                        child.wait()
                        log.close()
                        raise RuntimeError('Probe did not become visible before releasing admission lock')
                    write(PROBE / 'admission-state.json', dict(launch, state='running'))
                    print(json.dumps({'state': 'running', 'pid': child.pid, 'out': str(OUT)}), flush=True)
                    break
                write(PROBE / 'admission-state.json', dict(measured, state='waiting_capacity'))
                print(json.dumps({'state': 'waiting_capacity', 'jobs': len(jobs),
                                  'threads': measured['declared_threads'], 'live_pids': [j['pid'] for j in jobs]}), flush=True)
            time.sleep(10)
        code = child.wait()
        log.close()
        (OUT / 'probe.exit').write_text(str(code) + '\n')
        complete = json.loads((OUT / 'complete.json').read_text()) if (OUT / 'complete.json').exists() else {}
        valid = code == 0 and complete.get('contexts') == 18 and complete.get('no_fit') is True
        write(PROBE / 'admission-state.json', dict(launch, state='complete' if valid else 'failed',
                                                exit_code=code, output_valid=valid))
        return 0 if valid else (code or 1)


if __name__ == '__main__':
    raise SystemExit(main())
