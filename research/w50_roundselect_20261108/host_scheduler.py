"""Resource-gated fresh training queue; resumes exact owned runs only."""
import fcntl
import json
import math
import os
import shlex
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def now():
    return datetime.now(UTC).isoformat()


def resources():
    jobs = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().decode().split('\0')
            if 'python' in Path(args[0]).name and '--out' in args and '--threads' in args:
                jobs.append(int(args[args.index('--threads') + 1]))
        except (OSError, ValueError, IndexError, UnicodeError):
            continue
    memory = next(line for line in Path('/proc/meminfo').read_text().splitlines()
                  if line.startswith('MemAvailable:'))
    return {'jobs': len(jobs), 'threads': sum(jobs),
            'available_gib': int(memory.split()[1]) / 1024 ** 2,
            'load1': os.getloadavg()[0]}


def capacity(measured, own_running):
    return (own_running < 3 and measured['jobs'] < 6 and measured['threads'] + 4 <= 24
            and measured['available_gib'] >= 11 and measured['load1'] + 4 <= 24)


def validate_finished(out, seed):
    """Validate completed artifacts without opening evaluation observations."""
    try:
        complete = json.loads((out / 'complete.json').read_text())
        if (complete.get('status') != 'complete' or complete.get('exit_code') != 0
                or complete.get('fresh_training') is not True or complete.get('seed') != seed):
            raise ValueError('Invalid fresh-training completion receipt')
        score = json.loads((out / 'submission.json').read_text())['competition']
        local = json.loads((out / 'local_score.json').read_text())['competition']
        for scope in ('total', 'f1', 'f2', 'f3'):
            if score[scope]['n_replaced'] != 0:
                raise ValueError(f'{scope}: scorer used fallback forecasts')
            for metric in ('s', 'rmse', 'trmse', 'timse', 'mae'):
                value = score[scope][metric]
                if not math.isfinite(value) or value < 0 or value != local[scope][metric]:
                    raise ValueError(f'{scope}/{metric}: invalid or inconsistent score')
        if complete['total_s'] != score['total']['s']:
            raise ValueError('Completion score differs from submission score')
        for period, count in (('F1', 2089), ('F2', 2113), ('F3', 2113)):
            metadata = json.loads((out / f'{period}_metadata.json').read_text())
            if (metadata.get('seed') != seed or metadata.get('prior_rounds') != 900
                    or metadata.get('smoke') or metadata.get('fresh_training') is not True
                    or metadata.get('reused_model_or_predictions') is not False
                    or metadata.get('query_origins') != count):
                raise ValueError(f'{period}: invalid full-training metadata')
            selected = metadata.get('selected_rounds', [])
            if (len(selected) != 3 or any(not isinstance(n, int) or not 1 <= n <= 900 for n in selected)
                    or metadata.get('feature_count') != 28
                    or metadata.get('expanded_feature_count') != 212
                    or metadata.get('control', {}).get('passed') is not True
                    or metadata['control']['max_absolute_difference'] > 1e-6):
                raise ValueError(f'{period}: invalid selected tree counts or prior900 control')
            selection = json.loads((out / f'{period}_models/tree_selection.json').read_text())
            if len(selection) != 3:
                raise ValueError('Missing head selection records')
            for head, selected_count in zip(selection, selected, strict=True):
                curve = head['native_rmse']
                if (len(curve) != 900 or not all(math.isfinite(v) for v in curve)
                        or int(np.argmin(curve)) + 1 != selected_count
                        or head['full_rounds'] != selected_count):
                    raise ValueError('Tree-count rule or full refit differs')
            with np.load(out / f'{period}_predictions.npz', allow_pickle=False) as arrays:
                pred, origins = arrays['pred'], arrays['origins']
                if (pred.shape != (count, 72) or origins.shape != (count,)
                        or not np.isfinite(pred).all() or not (pred > 0).all()
                        or not np.isfinite(origins).all() or not (np.diff(origins) > 0).all()):
                    raise ValueError(f'{period}: invalid forecast dimensions or values')
            for stage in ('prior', 'full'):
                for block in range(3):
                    model = out / f'{period}_models' / stage / f'b{block}.ubj'
                    if not model.is_file() or model.stat().st_size == 0:
                        raise ValueError(f'Missing trained model: {model.name}')
        provenance = json.loads((out / 'provenance.json').read_text())
        if (provenance.get('fresh_training') is not True
                or provenance.get('reused_model_or_predictions') is not False):
            raise ValueError('Missing fresh-training provenance')
        for name in ('predictions.parquet', 'feature_contract.json', 'train.log'):
            if not (out / name).is_file() or (out / name).stat().st_size == 0:
                raise ValueError(f'Missing required artifact: {name}')
        return True, None
    except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
        return False, str(error)


def finished_state(out, seed):
    text = (out / 'train.exit').read_text().strip()
    if not text:
        # Older wrappers wrote this non-atomically. Retry rather than crashing
        # the foreground scheduler and dropping the WSL lifetime connection.
        return {'state': 'running', 'error': 'Exit receipt write pending'}
    try:
        code = int(text)
    except ValueError:
        return {'state': 'failed', 'error': 'Malformed train.exit receipt'}
    valid, error = validate_finished(out, seed) if code == 0 else (False, 'Training exited nonzero')
    return {'state': 'trained_validated' if valid else 'failed', 'exit_code': code,
            'finished_utc': now(), 'error': error,
            'validation_scope': 'artifact structure; independent model/score replay still required'}


def main():
    lock = (ROOT / '.scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assignment = json.loads((ROOT / 'assignment.json').read_text())
    while True:
        state = {}
        active = 0
        for row in assignment['experiments']:
            name = row['experiment']
            out = ROOT / 'runs' / name
            out.mkdir(parents=True, exist_ok=True)
            if (out / 'train.exit').exists():
                state[name] = finished_state(out, row['seed'])
                active += int(state[name]['state'] == 'running')
            elif (out / 'launch.json').exists():
                launch = json.loads((out / 'launch.json').read_text())
                alive = subprocess.run(['tmux', 'has-session', '-t', '=' + launch['tmux_session']],
                                       capture_output=True, check=False).returncode == 0
                state[name] = {'state': 'running' if alive else 'failed',
                               'error': None if alive else 'Session absent without exit receipt'}
                active += int(alive)
            else:
                state[name] = {'state': 'waiting'}
        for row in assignment['experiments']:
            name = row['experiment']
            if state[name]['state'] != 'waiting':
                continue
            measured = resources()
            write(ROOT / 'resources.json', dict(measured, measured_utc=now()))
            if not capacity(measured, active):
                break
            out = ROOT / 'runs' / name
            session = 'srw50-' + name + '-20261108'
            cmd = [assignment['python'], '-u', str(ROOT / 'experiment.py'),
                   '--experiment', name, '--seed', str(row['seed']),
                   '--out', str(out), '--device', 'cpu', '--threads', '4',
                   '--data', assignment['data'],
                   '--observations', assignment['observations']]
            write(out / 'launch.json', {'family': 'pooling', 'experiment': name,
                  'tmux_session': session, 'started_utc': now(), 'host': 'golab2',
                  'training_host': 'golab2', 'campaign_root': str(ROOT), 'command': cmd,
                  'resources_before': measured})
            script = out / 'train.sh'
            script.write_text('#!/bin/bash\nset -u\ncd ' + shlex.quote(str(ROOT)) + '\n'
                + 'export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n'
                + shlex.join(cmd) + ' > ' + shlex.quote(str(out / 'train.log')) + ' 2>&1\n'
                + 'code=$?\nprintf "%s\\n" "$code" > ' + shlex.quote(str(out / 'train.exit.tmp'))
                + '\nmv -- ' + shlex.quote(str(out / 'train.exit.tmp')) + ' '
                + shlex.quote(str(out / 'train.exit'))
                + '\nexit "$code"\n')
            subprocess.run(['tmux', 'new-session', '-d', '-s', session, 'bash ' + shlex.quote(str(script))],
                           check=True)
            state[name] = {'state': 'running', 'started_utc': now()}
            active += 1
            time.sleep(2)
        write(ROOT / 'scheduler-state.json', state)
        print(json.dumps({'time': now(), 'state': state}), flush=True)
        if all(row['state'] in ('trained_validated', 'failed') for row in state.values()):
            write(ROOT / 'host-complete.json', {'state': state, 'completed_utc': now(),
                  'all_terminal': True,
                  'training_success_all': all(row['state'] == 'trained_validated' for row in state.values()),
                  'research_goal_complete': False})
            return
        time.sleep(20)


if __name__ == '__main__':
    main()
