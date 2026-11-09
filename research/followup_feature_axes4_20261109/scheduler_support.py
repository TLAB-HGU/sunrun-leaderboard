"""CPU-only locked additive-axis launch and artifact validation."""


from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import uuid


TERMINAL = {'trained_validated', 'failed', 'cancelled_by_user'}
DEFAULT_PYTHON = '/home/yehwan/sunrun-golab2-20261104-17/.venv/bin/python'


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def identity(path):
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


@contextmanager
def scheduler_ownership(root):
    with (root / '.scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pid_path = root / 'scheduler.pid'
        pid = str(os.getpid())
        pid_path.write_text(pid + '\n')
        try:
            yield
        finally:
            if pid_path.exists() and pid_path.read_text().strip() == pid:
                pid_path.unlink()


def seed(recipe):
    return int(recipe.get('seed', recipe['config'].get('seed', 42)))


def memory_gib():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(values['MemAvailable'].split()[0]) / 1024**2


def busy_cpu_cores():
    def sample():
        values = [int(v) for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
        return sum(values), values[3] + values[4]
    first, first_idle = sample()
    time.sleep(.2)
    last, last_idle = sample()
    used = 1 - (last_idle - first_idle) / max(1, last - first)
    return max(os.getloadavg()[0], max(0., used) * (os.cpu_count() or 1))


def session_alive(session):
    return subprocess.run(['tmux', 'has-session', '-t', '=' + session], capture_output=True).returncode == 0


def session_id(session):
    result = subprocess.run(['tmux', 'list-sessions', '-F', '#{session_name}\t#{session_id}'],
                            capture_output=True, text=True)
    if result.returncode:
        return None
    for line in result.stdout.splitlines():
        name, separator, identifier = line.partition('\t')
        if separator and name == session and re.fullmatch(r'\$\d+', identifier):
            return identifier
    return None


def launch(root, assignment, recipe):
    name = recipe['experiment']
    out = root / 'runs' / name
    out.mkdir(parents=True, exist_ok=False)
    config = expected_config(recipe)
    atomic(out / 'config.json', config)
    wrapper = ('\"\"\"Frozen single-experiment entrypoint.\"\"\"\n'
               'from core import main\n\n'
               f'EXPERIMENT = {name!r}\nCONFIG = {config!r}\n\n'
               'if __name__ == \"__main__\":\n'
               '    main(locked_experiment=EXPERIMENT, locked_config=CONFIG)\n')
    (out / 'experiment.py').write_text(wrapper)
    for filename in ('core.py', 'baseline.py', 'competition.py', 'provenance.json', 'inventory.json'):
        source = root / filename
        if filename != 'provenance.json' or source.exists():
            (out / filename).write_bytes(source.read_bytes())
    threads = int(recipe.get('threads', assignment.get('threads', 4)))
    data = input_path(assignment, recipe)
    command = [assignment.get('python', DEFAULT_PYTHON), '-u', str(out / 'experiment.py'),
               '--experiment', name, '--config', str(out / 'config.json'),
               '--data', str(data), '--out', str(out),
               '--observations', assignment['observations'], '--device', 'cpu', '--threads', str(threads)]
    session = f'srax8-{uuid.uuid4().hex[:8]}-{name}'
    if session_alive(session):
        raise ValueError('Refusing to reuse an existing tmux session')
    (out / 'runner_at_launch.py').write_bytes((out / 'experiment.py').read_bytes())
    host = assignment.get('host', 'golab2')
    record = dict(recipe, config=config, family='pooling', host=host, training_host=host,
                  output=str(out), campaign_root=str(root), command=command,
                  tmux_session=session, started_utc=now(), threads=threads,
                  input_identity=identity(data), source_identity=identity(out / 'experiment.py'),
                  source_snapshots={name: identity(out / name) for name in ('experiment.py', 'core.py', 'baseline.py', 'competition.py')})
    atomic(out / 'launch.json', record)
    environment = ['env', 'CUDA_VISIBLE_DEVICES=', f'OMP_NUM_THREADS={threads}',
                   'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1']
    script = ('#!/bin/bash\nset -o pipefail\ncd ' + shlex.quote(str(root)) + '\n'
              + "printf '%s\\n' " + shlex.quote(json.dumps(dict(period='F1', stage='prior', unit=0, units=6))) + ' > ' + shlex.quote(str(out / 'train.log')) + '\n'
              + shlex.join(environment + command) + ' >> ' + shlex.quote(str(out / 'train.log'))
              + ' 2>&1\ncode=$?\nprintf \'%s\\n\' "$code" > ' + shlex.quote(str(out / 'train.exit.tmp'))
              + '\nmv ' + shlex.quote(str(out / 'train.exit.tmp')) + ' ' + shlex.quote(str(out / 'train.exit'))
              + '\nexit "$code"\n')
    (out / 'run.sh').write_text(script)
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, 'bash', str(out / 'run.sh')], check=True)
    identifier = session_id(session)
    if not identifier:
        raise RuntimeError('Launched session has no verifiable tmux session ID')
    record['tmux_session_id'] = identifier
    atomic(out / 'launch.json', record)
    return record


VALIDATE = r'''
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
out = Path(sys.argv[1])
keys = ['origin_last_input_utc', 'horizon_hours']
frame = pd.read_parquet(out / 'predictions.parquet').sort_values(keys).reset_index(drop=True)
if len(frame) != (2089+2113+2113)*72 or frame.duplicated(keys).any():
    raise ValueError('Incomplete or duplicate prediction keys')
if not np.isfinite(frame.pred_kms).all() or not (frame.pred_kms > 0).all():
    raise ValueError('Nonfinite/nonpositive predictions')
for period, count in [('F1',2089),('F2',2113),('F3',2113)]:
    meta = json.loads((out / (period+'_metadata.json')).read_text())
    if meta.get('smoke') or meta.get('period', period) != period:
        raise ValueError('Smoke or mismatched metadata')
    with np.load(out / (period+'_predictions.npz'), allow_pickle=False) as z:
        if z['pred'].shape != (count,72) or len(z['origins']) != count:
            raise ValueError('Incomplete period prediction grid')
        if not np.isfinite(z['pred']).all() or len(np.unique(z['origins'])) != count:
            raise ValueError('Invalid period prediction grid')
        expected = pd.DataFrame({keys[0]: np.repeat(pd.DatetimeIndex(z['origins'],tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ'),72), keys[1]: np.tile(np.arange(1,73),count), 'pred_kms':z['pred'].ravel()})
        joined = expected.merge(frame, on=keys, validate='one_to_one', suffixes=('_npz','_parquet'))
        if len(joined) != count*72 or not np.array_equal(joined.pred_kms_npz, joined.pred_kms_parquet):
            raise ValueError('NPZ and parquet predictions disagree')
score = json.loads((out / 'local_score.json').read_text())
score = score.get('competition', score)
for scope in ['total','f1','f2','f3']:
    if not np.isfinite(float(score[scope]['s'])) or score[scope].get('n_replaced',0):
        raise ValueError('Invalid or replaced official score')
result = {'rows': len(frame), 'total_s': float(score['total']['s'])}
sys.path.insert(0, str(out))
import competition
recomputed = competition.score_competition(frame, competition.prepare_context(
    competition.load_observations(Path(sys.argv[2]))))
if score != recomputed:
    raise ValueError('Saved score differs from recomputed official scorer output')
result['official_score_recomputed'] = True
print(json.dumps(result))
'''


def cleanup_terminal_session(record):
    """Remove only a dead pane in the exact session created for this run."""
    identifier = record.get('tmux_session_id')
    if not identifier or session_id(record['tmux_session']) != identifier:
        return
    result = subprocess.run(['tmux', 'list-panes', '-s', '-t', identifier, '-F',
                             '#{session_id} #{pane_dead} #{session_windows}'], capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip() == identifier + ' 1 1':
        subprocess.run(['tmux', 'kill-session', '-t', identifier], check=True)


def expected_config(recipe):
    import core
    supplied = recipe['config']
    if set(supplied) != {'seed', 'add_features'}:
        raise ValueError('Only seed and additive features may vary')
    if seed(recipe) not in (42, 43, 44) or supplied['seed'] != seed(recipe):
        raise ValueError('Unexpected seed')
    return dict(core.normalize_config(supplied), experiment=recipe['experiment'])


def input_path(assignment, recipe):
    """Each axis consumes its own immutable augmented input table."""
    value = recipe.get('data', assignment.get('data'))
    if value is None:
        raise ValueError('Missing per-axis augmented input path')
    return Path(value).resolve()



def validate_output(root, assignment, recipe):
    out = root / 'runs' / recipe['experiment']
    expected = expected_config(recipe)
    if (out / 'train.exit').read_text().strip() != '0':
        raise ValueError('Training has no successful exit receipt')
    launch_record = json.loads((out / 'launch.json').read_text())
    if (launch_record['experiment'] != recipe['experiment'] or
            launch_record['campaign_root'] != str(root) or launch_record['config'] != expected):
        raise ValueError('Launch receipt differs from locked recipe')
    if launch_record['input_identity'] != identity(input_path(assignment, recipe)):
        raise ValueError('Augmented input changed since launch')
    for filename in ('train.log', 'launch.json', 'runner_at_launch.py', 'local_score.json'):
        if not (out / filename).is_file() or not (out / filename).stat().st_size:
            raise ValueError(f'Missing or empty {filename}')
    for name in ('core.py', 'baseline.py', 'competition.py', 'inventory.json'):
        if (out / name).read_bytes() != (root / name).read_bytes():
            raise ValueError('Scientific source snapshot changed: ' + name)
    if (out / 'experiment.py').read_bytes() != (out / 'runner_at_launch.py').read_bytes():
        raise ValueError('Experiment differs from launch snapshot')
    if json.loads((out / 'config.json').read_text()) != expected:
        raise ValueError('Frozen configuration changed')
    import core
    import baseline
    names = core.feature_names(expected)
    expanded = {str(start): core.expanded_names(names, start, 24) for start in (0, 24, 48)}
    for period in ('F1', 'F2', 'F3'):
        report = json.loads((out / f'{period}_metadata.json').read_text())
        if report['config'] != expected or report['feature_count'] != 28 + len(recipe['config']['add_features']):
            raise ValueError('Metadata configuration mismatch')
        if report['features'] != names or report['expanded_features'] != expanded:
            raise ValueError('Source or expanded input values/order differ from frozen recipe')
        if (report['rounds'] != 900 or report['seed'] != expected['seed'] or
                report['training_cutoff'] != baseline.CUTOFFS[period] or
                report['query_origins'] != baseline.QUERY_COUNTS[period]):
            raise ValueError('Metadata training/evaluation contract mismatch')
        frozen = json.loads((out / f'{period}_models/frozen_calibration.json').read_text())
        if frozen != report['frozen_state']:
            raise ValueError('Frozen calibration artifact differs from metadata')
    result = subprocess.run([assignment.get('python', DEFAULT_PYTHON), '-c', VALIDATE, str(out), assignment['observations']],
                            capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'))
    if result.returncode:
        raise ValueError('Artifact validation failed: ' + result.stderr.strip()[-2000:])
    replay = subprocess.run([assignment.get('python', DEFAULT_PYTHON), str(root / 'replay.py'), '--run', str(out), '--data', str(input_path(assignment, recipe))], capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'))
    if replay.returncode:
        raise ValueError('Saved model replay failed: ' + replay.stderr[-2000:])
    proof = dict(json.loads(result.stdout), saved_model_replay=json.loads(replay.stdout), validated_utc=now())
    atomic(out / 'training-validation.json', proof)
    return proof
