"""Launch isolated distribution experiments as durable services with tmux monitors."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = Path('/home/t-lab01/sunrun')
PYTHON = ROOT / '.venv/bin/python'
METHODS = ('controls', 'lognormal_nll', 'lognormal_crps', 'smooth_wcrps', 'tailcal_crps')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def run_directory(run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        raise ValueError('run-id must contain 1–80 letters, digits, underscores or hyphens')
    return HERE / 'runs' / run_id


def names(run_id, method):
    run_directory(run_id)
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}')
    slug = run_id.replace('_', '-').lower()
    short = slug if len(slug) <= 30 else slug[:23] + '-' + hashlib.sha256(run_id.encode()).hexdigest()[:6]
    suffix = method.replace('_', '-')
    return f'fs30dist-{slug}-{suffix}', f'fs30dist-{short}-{suffix}'


def execute(arguments):
    return subprocess.run([str(item) for item in arguments], capture_output=True, text=True)


def service_state(unit):
    result = execute(['systemctl', '--user', 'show', unit + '.service',
                      '--property=LoadState,ActiveState,SubState,MainPID,ExecMainStatus,Result'])
    state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if result.returncode and state.get('LoadState') != 'not-found':
        raise RuntimeError(f'Cannot inspect {unit}: {result.stderr.strip()}')
    if not state.get('LoadState'):
        raise RuntimeError(f'Cannot establish service identity for {unit}: {result.stderr.strip()}')
    return state


def has_session(session):
    result = execute(['tmux', 'has-session', '-t', '=' + session])
    if result.returncode not in (0, 1):
        raise RuntimeError(f'Cannot inspect tmux session {session}: {result.stderr.strip()}')
    return result.returncode == 0


def foreground_command(run_id, method, smoke=False):
    command = [str(PYTHON), '-u', str(HERE / 'run.py'), 'run', '--run-id', run_id, '--method', method]
    if smoke:
        command.append('--smoke')
    return command


def launch(run_id, smoke=False):
    directory = run_directory(run_id)
    if not (directory / 'manifest.json').is_file():
        raise FileNotFoundError(f'Prepare the run first: {directory / "manifest.json"}')
    with (directory / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = directory / 'launch.json'
        if receipt_path.exists():
            raise RuntimeError(f'Existing launch receipt preserved: {receipt_path}')
        entries = []
        for method in METHODS:
            unit, session = names(run_id, method)
            if service_state(unit)['LoadState'] != 'not-found':
                raise RuntimeError(f'Existing service preserved: {unit}')
            if has_session(session):
                raise RuntimeError(f'Existing tmux session preserved: {session}')
            entries.append(dict(method=method, unit=unit, session=session, status='pending',
                                directory=str(directory / method)))
        receipt = dict(run_id=run_id, smoke=smoke, started_at=utc_now(), status='launching', lanes=entries)
        write(receipt_path, receipt)
        for entry in entries:
            lane = Path(entry['directory'])
            lane.mkdir(parents=True, exist_ok=True)
            log = lane / 'run.log'
            command = foreground_command(run_id, entry['method'], smoke)
            entry.update(status='launching', foreground_command=command, log=str(log))
            service = ['systemd-run', '--user', '--unit=' + entry['unit'],
                       '--working-directory=' + str(ROOT), '--property=CPUQuota=200%',
                       '--setenv=OMP_NUM_THREADS=2', '--setenv=OPENBLAS_NUM_THREADS=2',
                       '--setenv=MKL_NUM_THREADS=2', '--setenv=CUDA_VISIBLE_DEVICES=',
                       '--property=StandardOutput=append:' + str(log),
                       '--property=StandardError=append:' + str(log), *command]
            entry['service_command'] = service
            write(receipt_path, receipt)
            try:
                result = execute(service)
                entry.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
                if result.returncode:
                    raise RuntimeError('Background launch failed; run this foreground command immediately: '
                                       + shlex.join(command))
                entry['service_state'] = service_state(entry['unit'])
                state = entry['service_state']
                completed = False
                status_path = lane / 'status.json'
                if status_path.exists():
                    completed = json.loads(status_path.read_text()).get('status') == 'completed'
                if state.get('ActiveState') not in ('active', 'activating') and not (
                        completed and state.get('Result') == 'success' and state.get('ExecMainStatus') == '0'):
                    raise RuntimeError(f'Experiment did not remain active: {entry["unit"]}: {state}')
                monitor = 'exec ' + shlex.join(['tail', '-n', '50', '-F', '--', str(log)])
                pane = execute(['tmux', 'new-session', '-d', '-s', entry['session'], '-c', str(ROOT), monitor])
                entry.update(tmux_returncode=pane.returncode, tmux_stderr=pane.stderr)
                if pane.returncode or not has_session(entry['session']):
                    raise RuntimeError(f'Service started but tmux monitor failed: {entry["session"]}')
                entry.update(status='completed' if completed else 'started', verified_at=utc_now())
                write(receipt_path, receipt)
            except Exception as error:
                entry.update(status='launch_failed', error=str(error), failed_at=utc_now())
                receipt.update(status='launch_failed', error=str(error))
                write(receipt_path, receipt)
                raise
        receipt.update(status='started', verified_at=utc_now())
        write(receipt_path, receipt)
        return receipt


def status(run_id):
    directory = run_directory(run_id)
    path = directory / 'launch.json'
    receipt = json.loads(path.read_text())
    result = dict(run_id=run_id, launch_status=receipt['status'], checked_at=utc_now(), lanes=[])
    for entry in receipt['lanes']:
        lane = dict(method=entry['method'], unit=entry['unit'], session=entry['session'],
                    service=service_state(entry['unit']), tmux_exists=has_session(entry['session']))
        status_path = directory / entry['method'] / 'status.json'
        lane['experiment'] = json.loads(status_path.read_text()) if status_path.exists() else None
        result['lanes'].append(lane)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=('launch', 'status'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    result = launch(args.run_id, args.smoke) if args.step == 'launch' else status(args.run_id)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
