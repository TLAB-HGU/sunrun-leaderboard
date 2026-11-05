"""Run an assigned golab2 queue after its timezone-aware start time.

Training validation is separate from scoring and leaderboard publication.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

DEFAULT_PYTHON = '/home/yehwan/power_forecast/.venv/bin/python'
DL = {'foundation', 'tcn'}
RAW = {'strategies', 'foundation', 'direct', 'tcn', 'special', 'special_hsspeak'}
TERMINAL = {'trained_validated', 'failed', 'cancelled_by_user'}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def load_assignment(root):
    assignment = json.loads((root / 'assignment.json').read_text())
    start = datetime.fromisoformat(assignment['not_before'])
    if start.tzinfo is None:
        raise ValueError('not_before requires an explicit timezone')
    names = set()
    for recipe in assignment['experiments']:
        recipe.setdefault('python', assignment.get('python', DEFAULT_PYTHON))
        recipe.setdefault('git_commit', assignment.get('git_commit'))
        name, family = recipe['experiment'], recipe['family']
        if not re.fullmatch(r'[A-Za-z0-9_]+', name) or name in names:
            raise ValueError('Invalid or duplicate experiment')
        if not re.fullmatch(r'[a-z0-9_]+', family):
            raise ValueError('Invalid family')
        names.add(name)
        compile((root / f'{family}.py').read_text(), f'{family}.py', 'exec')
        if not Path(recipe.get('python', DEFAULT_PYTHON)).is_file():
            raise ValueError('Missing recipe Python executable')
        if family in RAW and not (root / 'raw/df.parquet').is_file():
            raise ValueError('Missing raw input')
        if int(recipe.get('threads', 4)) != 4:
            raise ValueError('Remote assignment requires four threads per job')
    if not names or not (root / 'input/df_ffill_trainy.parquet').is_file():
        raise ValueError('Empty assignment or missing prepared input')
    return assignment, start


def memory_gib():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(values['MemAvailable'].split()[0]) / 1024**2


def gpu_available():
    try:
        result = subprocess.run(['nvidia-smi', '--id=0',
                                 '--query-gpu=memory.free,memory.used,utilization.gpu',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True)
        free, used, utilization = (int(v.strip()) for v in result.stdout.strip().split(','))
        return result.returncode == 0 and free >= 8192 and used < 1024 and utilization < 10
    except (OSError, ValueError):
        return False


def foundation_dependency(recipe):
    """Provisioning may finish after reservation; missing dependencies remain queued."""
    code = ('import torch,sdm; from huggingface_hub import hf_hub_download; '
            'hf_hub_download("nvidia/Kumo-Tabular", "large/regressor.pt", '
            'revision="v1.0.1", local_files_only=True)')
    try:
        result = subprocess.run([recipe.get('python', DEFAULT_PYTHON), '-c', code],
                                capture_output=True, text=True, timeout=45)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f'Waiting for foundation dependencies: {error}'
    if result.returncode:
        details = (result.stderr or result.stdout).strip().splitlines()
        return 'Waiting for foundation dependencies: ' + (details[-1] if details else 'dependency check failed')
    return None


def session_alive(session):
    return subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0


def physics_ready(root):
    try:
        features = root / 'input/huxt_features.parquet'
        meta = json.loads(features.with_suffix('.metadata.json').read_text())
        return ((root / 'gong_cache/exit.code').read_text().strip() == '0'
                and features.stat().st_size > 0 and meta.get('old_huxt_predictions_used') is False
                and meta.get('stage') in ('all', 'forecast') and bool(meta.get('versions')))
    except (OSError, ValueError):
        return False


def identity(path):
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def launch(root, recipe):
    name, family = recipe['experiment'], recipe['family']
    out = root / 'runs' / name
    out.mkdir(parents=True, exist_ok=False)
    source = (root / f'{family}.py').read_text()
    entry = ast.parse(source).body[-1]
    if not isinstance(entry, ast.If) or '__name__' not in ast.unparse(entry.test):
        raise ValueError('Family lacks standalone main guard')
    source = ('\n'.join(source.splitlines()[:entry.lineno - 1])
              + '\nif __name__ == "__main__":\n    import sys\n'
              + '    if "--experiment" not in sys.argv:\n'
              + f'        sys.argv.extend(["--experiment", {name!r}])\n    main()\n')
    for filename in ('experiment.py', 'runner_at_launch.py'):
        target = out / filename
        target.write_text(source)
        target.chmod(0o444)
    command = [recipe.get('python', DEFAULT_PYTHON), '-u', str(out / 'experiment.py'),
               '--experiment', name, '--data', str(root / 'input/df_ffill_trainy.parquet'),
               '--out', str(out), '--threads', '4']
    if family in RAW:
        command += ['--raw-data', str(root / 'raw/df.parquet')]
    if family in DL:
        command += ['--device', 'cuda:0']
    if family == 'special_hsspeak':
        command += ['--huxt-features', str(root / 'input/huxt_features.parquet')]
    session = 'sr23-' + name
    record = dict(recipe, host='golab2', output=str(out), tmux_session=session,
                  command=command, started_utc=now(), input_identity=identity(root / 'input/df_ffill_trainy.parquet'),
                  source_bytes=len(source.encode()), source_snapshot=str(out / 'experiment.py'),
                  training_target_ffill=True, gpu=0 if family in DL else None)
    versions = subprocess.run([recipe['python'], '-c',
        'import importlib.metadata as m,json,platform; '
        'print(json.dumps(dict(python=platform.python_version(),packages={d.metadata["Name"]:d.version for d in m.distributions()})))'],
        capture_output=True, text=True, check=True)
    record['versions'] = json.loads(versions.stdout)
    atomic(out / 'launch.json', record)
    environment = ['env', 'CUDA_VISIBLE_DEVICES=' + ('0' if family in DL else ''),
                   'OMP_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4']
    script = ('#!/bin/bash\nset -o pipefail\ncd ' + shlex.quote(str(root)) + '\n'
              + shlex.join(environment + command) + ' > ' + shlex.quote(str(out / 'train.log'))
              + ' 2>&1\ncode=$?\nprintf "%s\\n" "$code" > '
              + shlex.quote(str(out / 'train.exit.tmp')) + '\nmv '
              + shlex.quote(str(out / 'train.exit.tmp')) + ' ' + shlex.quote(str(out / 'train.exit'))
              + '\nexit "$code"\n')
    (out / 'run.sh').write_text(script)
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, 'bash', str(out / 'run.sh')], check=True)
    return record


def validate_output(out):
    import numpy as np
    import pandas as pd
    for name in ('train.log', 'launch.json', 'experiment.py', 'runner_at_launch.py', 'predictions.parquet'):
        if not (out / name).is_file() or not (out / name).stat().st_size:
            raise ValueError(f'Missing or empty {name}')
    frame = pd.read_parquet(out / 'predictions.parquet')
    keys = ['origin_last_input_utc', 'horizon_hours']
    if len(frame) != (2089 + 2113 + 2113) * 72 or frame.duplicated(keys).any():
        raise ValueError('Incomplete or duplicate prediction keys')
    if not np.isfinite(frame.pred_kms).all() or not (frame.pred_kms > 0).all():
        raise ValueError('Invalid predictions')
    if set(frame.horizon_hours.unique()) != set(range(1, 73)):
        raise ValueError('Incomplete horizons')
    for period in ('F1', 'F2', 'F3'):
        reports = list(out.glob(f'{period}*.json'))
        if not reports:
            raise ValueError(f'Missing {period} metadata')
        for report in reports:
            json.loads(report.read_text())
    models = [p for p in out.rglob('*') if p.is_file() and p.suffix in {'.ubj', '.txt', '.pkl', '.pickle', '.joblib', '.pt', '.pth'}]
    contexts = list(out.rglob('head*_context.npz'))
    for context in contexts:
        report = json.loads(context.with_name(context.name.replace('_context.npz', '.json')).read_text())
        checkpoint = report.get('checkpoint') or {}
        if not Path(checkpoint.get('local_path', '/missing')).is_file():
            raise ValueError('Missing pinned foundation checkpoint')
    if not models and not contexts:
        raise ValueError('No fitted model artifacts')
    result = {'validated_utc': now(), 'rows': len(frame), 'models': len(models), 'fitted_contexts': len(contexts),
              'evaluation_status': 'pending_separate_scoring'}
    atomic(out / 'training-validation.json', result)
    return result


def tick(root, assignment, start, state):
    active = []
    for recipe in assignment['experiments']:
        name = recipe['experiment']
        entry = state.setdefault(name, {'state': 'scheduled'})
        if entry['state'] in TERMINAL:
            continue
        out = root / 'runs' / name
        record_path = out / 'launch.json'
        if not record_path.exists():
            if out.exists():
                entry.update(state='failed', reason='Partial launch directory; manual diagnosis required')
            continue
        record = json.loads(record_path.read_text())
        if (out / 'train.exit').exists():
            try:
                code = int((out / 'train.exit').read_text())
                if code != 0:
                    raise ValueError(f'Training exited {code}')
                proof = validate_output(out)
                entry.update(state='trained_validated', finished_utc=now(), validation=proof)
            except Exception as error:
                entry.update(state='failed', finished_utc=now(), reason=str(error))
            if session_alive(record['tmux_session']):
                subprocess.run(['tmux', 'kill-session', '-t', record['tmux_session']], check=True)
        elif session_alive(record['tmux_session']):
            entry.update(state='running')
            active.append(recipe)
        else:
            entry.update(state='failed', reason='Training session lost without exit receipt')
    if datetime.now(timezone.utc) < start:
        return
    available = memory_gib()
    for recipe in assignment['experiments']:
        entry = state[recipe['experiment']]
        if entry['state'] not in {'scheduled', 'waiting'}:
            continue
        family = recipe['family']
        if len(active) >= 3 or available < 8:
            entry.update(state='waiting', reason='Waiting for CPU/RAM slot')
            continue
        if family in DL and (any(r['family'] in DL for r in active) or not gpu_available()):
            entry.update(state='waiting', reason='Waiting for GPU0 memory')
            continue
        if family == 'special_hsspeak' and not physics_ready(root):
            entry.update(state='waiting', reason='Waiting for verified GONG/HUXt features')
            continue
        if family == 'foundation':
            dependency = foundation_dependency(recipe)
            if dependency:
                entry.update(state='waiting', reason=dependency)
                continue
        try:
            record = launch(root, recipe)
            entry.clear()
            entry.update(state='running', started_utc=record['started_utc'])
            active.append(recipe)
            available -= 8
        except Exception as error:
            entry.update(state='failed', reason=f'Launch failed: {error}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    assignment, start = load_assignment(root)
    if args.check:
        print(json.dumps({'valid': True, 'experiments': len(assignment['experiments']), 'not_before': start.isoformat()}))
        return
    with (root / '.scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root / 'scheduler-state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        while True:
            tick(root, assignment, start, state)
            atomic(state_path, state)
            print(json.dumps({'utc': now(), 'counts': {s: sum(e['state'] == s for e in state.values())
                                                     for s in ('scheduled', 'waiting', 'running', 'trained_validated', 'failed', 'cancelled_by_user')}}), flush=True)
            if args.once or all(e['state'] in TERMINAL for e in state.values()):
                break
            time.sleep(30)


if __name__ == '__main__':
    main()
