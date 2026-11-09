"""Run host-assigned CPU ablations after baseline reproduction and resource checks.

The supervisor uses only stdlib; artifact checks use the experiment interpreter.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import uuid

TERMINAL = {'trained_validated', 'failed', 'cancelled_by_user'}
DEFAULT_PYTHON = '/home/yehwan/power_forecast/.venv/bin/python'


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


def replay(recipe):
    return bool(recipe['config'].get('reuse_run'))


def threads_for(assignment, recipe):
    return int(recipe.get('threads', 1 if replay(recipe) else assignment.get('threads', 4)))


def load_assignment(root):
    assignment = json.loads((root / 'assignment.json').read_text())
    if assignment.get('host', 'golab2') not in {'local', 'golab2'}:
        raise ValueError('Unknown training host')
    names = set()
    for recipe in assignment['experiments']:
        name = recipe['experiment']
        if not re.fullmatch(r'[A-Za-z0-9_]+', name) or name in names:
            raise ValueError('Experiment names must be unique safe identifiers')
        names.add(name)
        if recipe.get('family', 'pooling') != 'pooling':
            raise ValueError('Only CPU pooling ablations are supported')
        if recipe['stage'] not in {'baseline', 'ablation'}:
            raise ValueError('Unknown experiment stage')
        if 'seed' in recipe and seed(recipe) != int(recipe['config'].get('seed', seed(recipe))):
            raise ValueError('Recipe and model seeds disagree')
        if threads_for(assignment, recipe) < 1:
            raise ValueError('threads must be positive')
    for recipe in assignment['experiments']:
        dependency = recipe['config'].get('reuse_run')
        if dependency and (dependency not in names or dependency == recipe['experiment']):
            raise ValueError('Invalid reuse_run dependency')
        visited = {recipe['experiment']}
        while dependency:
            if dependency in visited:
                raise ValueError('Cyclic reuse_run dependency')
            visited.add(dependency)
            parent = next(r for r in assignment['experiments'] if r['experiment'] == dependency)
            dependency = parent['config'].get('reuse_run')
    baselines = [r for r in assignment['experiments'] if r['stage'] == 'baseline']
    expected_seeds = assignment.get('baseline_seeds', [42, 43, 44])
    if (not names or not expected_seeds or 42 not in expected_seeds
            or len(set(expected_seeds)) != len(expected_seeds)
            or sorted(seed(r) for r in baselines) != sorted(expected_seeds)):
        raise ValueError('Exactly one baseline per configured seed, including seed42, is required')
    for key in ('data', 'observations'):
        path = Path(assignment[key])
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise ValueError(f'Missing {key}: {path}')
        assignment[key] = str(path.resolve())
    if not Path(assignment.get('python', DEFAULT_PYTHON)).is_file():
        raise ValueError('Missing experiment interpreter')
    for key, default in [('max_jobs', 1), ('threads', 4), ('max_cpu_threads_total', os.cpu_count() or 1)]:
        if int(assignment.get(key, default)) < 1:
            raise ValueError(f'{key} must be positive')
    for key, default in [('reserve_memory_gib', 6), ('estimated_job_memory_gib', 5)]:
        value = float(assignment.get(key, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if assignment.get('baseline_reference_s') is None and not assignment.get('baseline_reference_predictions'):
        raise ValueError('A baseline score or prediction reference is required')
    if assignment.get('baseline_reference_s') is not None and not math.isfinite(float(assignment['baseline_reference_s'])):
        raise ValueError('Baseline reference score must be finite')
    for key in ('baseline_prediction_tolerance', 'baseline_score_tolerance'):
        value = float(assignment.get(key, 1e-6))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    compile((root / 'core.py').read_text(), 'core.py', 'exec')
    return assignment


def memory_gib():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(values['MemAvailable'].split()[0]) / 1024**2


def host_gate_error(root, assignment):
    """A copied baseline receipt cannot prove reproduction on this host."""
    filename = assignment.get('host_reproduction_gate')
    if not filename:
        return None
    path = Path(filename)
    if not path.is_absolute():
        path = root / path
    try:
        proof = json.loads(path.read_text())
        if proof.get('host') != assignment.get('host', 'golab2') or proof.get('validated') is not True:
            raise ValueError('host or validation flag mismatch')
        current_data = identity(Path(assignment['data']))
        if proof.get('data') != current_data:
            raise ValueError('host reproduction input identity mismatch')
        smoke_path = Path(assignment['smoke_proof'])
        if not smoke_path.is_absolute():
            smoke_path = root / smoke_path
        smoke = json.loads(smoke_path.read_text())
        if smoke.get('all_equal') is not True or smoke.get('data') != current_data:
            raise ValueError('training smoke proof missing or input identity mismatch')
        for field, tolerance in [('baseline_max_abs_difference', 'baseline_prediction_tolerance'),
                                 ('baseline_score_difference', 'baseline_score_tolerance')]:
            difference = float(proof[field])
            if not math.isfinite(difference) or difference < 0 or difference > float(assignment.get(tolerance, 1e-6)):
                raise ValueError(field + ' exceeds tolerance')
    except (OSError, ValueError, KeyError, TypeError) as error:
        return 'Waiting for host baseline reproduction: ' + str(error)
    return None


def validate_assignment_update(root, previous, updated, state):
    """Only unlaunched work may move; started recipes and input identity are frozen."""
    for key in ('host', 'data', 'observations', 'python', 'observation_revision',
                'baseline_reference_predictions', 'baseline_reference_s', 'host_reproduction_gate', 'smoke_proof'):
        if previous.get(key) != updated.get(key):
            raise ValueError('Cannot change active scheduler identity: ' + key)
    old = {r['experiment']: r for r in previous['experiments']}
    new = {r['experiment']: r for r in updated['experiments']}
    for name, recipe in old.items():
        launched = (root / 'runs' / name).exists()
        status = state.get(name, {}).get('state', 'scheduled')
        if launched or status not in {'scheduled', 'waiting'}:
            if new.get(name) != recipe:
                raise ValueError('Cannot remove or change started recipe: ' + name)
    for name in set(old) - set(new):
        state.pop(name, None)


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
    config = dict(recipe['config'], seed=seed(recipe))
    atomic(out / 'config.json', config)
    wrapper = ('\"\"\"Frozen single-experiment entrypoint.\"\"\"\n'
               'from core import main\n\n'
               f'EXPERIMENT = {name!r}\nCONFIG = {config!r}\n\n'
               'if __name__ == \"__main__\":\n'
               '    main(locked_experiment=EXPERIMENT, locked_config=CONFIG)\n')
    (out / 'experiment.py').write_text(wrapper)
    for filename in ('core.py', 'baseline.py', 'competition.py', 'provenance.json'):
        source = root / filename
        if filename != 'provenance.json' or source.exists():
            (out / filename).write_bytes(source.read_bytes())
    threads = threads_for(assignment, recipe)
    command = [assignment.get('python', DEFAULT_PYTHON), '-u', str(out / 'experiment.py'),
               '--experiment', name, '--config', str(out / 'config.json'),
               '--data', assignment['data'], '--out', str(out),
               '--observations', assignment['observations'], '--device', 'cpu', '--threads', str(threads)]
    session = f'srac-{uuid.uuid4().hex[:10]}-{name}'
    if session_alive(session):
        raise ValueError('Refusing to reuse an existing tmux session')
    (out / 'runner_at_launch.py').write_bytes((out / 'experiment.py').read_bytes())
    host = assignment.get('host', 'golab2')
    record = dict(recipe, config=config, family='pooling', host=host, training_host=host,
                  output=str(out), campaign_root=str(root), command=command,
                  tmux_session=session, started_utc=now(), threads=threads,
                  input_identity=identity(Path(assignment['data'])), source_identity=identity(out / 'experiment.py'),
                  source_snapshots={name: identity(out / name) for name in ('experiment.py', 'core.py', 'baseline.py', 'competition.py')})
    atomic(out / 'launch.json', record)
    environment = ['env', 'CUDA_VISIBLE_DEVICES=', f'OMP_NUM_THREADS={threads}',
                   'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1']
    script = ('#!/bin/bash\nset -o pipefail\ncd ' + shlex.quote(str(root)) + '\n'
              + shlex.join(environment + command) + ' > ' + shlex.quote(str(out / 'train.log'))
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
if len(sys.argv) > 2:
    reference = pd.read_parquet(sys.argv[2]).sort_values(keys).reset_index(drop=True)
    if not frame[keys].equals(reference[keys]):
        raise ValueError('Baseline reference prediction keys disagree')
    if not np.isfinite(reference.pred_kms).all():
        raise ValueError('Baseline reference predictions must be finite')
    difference = float(np.max(np.abs(frame.pred_kms.to_numpy()-reference.pred_kms.to_numpy())))
    result['baseline_max_abs_difference'] = difference
    if difference > float(sys.argv[3]):
        raise ValueError('Baseline prediction reproduction mismatch: ' + str(difference))
print(json.dumps(result))
'''


def validate_output(root, assignment, recipe):
    out = root / 'runs' / recipe['experiment']
    for filename in ('train.log', 'launch.json', 'runner_at_launch.py', 'local_score.json'):
        if not (out / filename).is_file() or not (out / filename).stat().st_size:
            raise ValueError(f'Missing or empty {filename}')
    launch_record = json.loads((out / 'launch.json').read_text())
    for name, snapshot in launch_record.get('source_snapshots', {}).items():
        if Path(name).name != name or not (out / name).is_file() or (out / name).stat().st_size != snapshot['bytes']:
            raise ValueError('Missing or changed source snapshot: ' + name)
    if (out / 'experiment.py').exists() and (out / 'experiment.py').read_bytes() != (out / 'runner_at_launch.py').read_bytes():
        raise ValueError('Experiment source differs from launch snapshot')
    command = [assignment.get('python', DEFAULT_PYTHON), '-c', VALIDATE, str(out)]
    baseline = recipe['stage'] == 'baseline' and seed(recipe) == 42
    if baseline and assignment.get('baseline_reference_predictions'):
        reference = Path(assignment['baseline_reference_predictions'])
        if not reference.is_absolute():
            reference = root / reference
        command += [str(reference), str(assignment.get('baseline_prediction_tolerance', 1e-6))]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise ValueError('Artifact validation failed: ' + result.stderr.strip()[-2000:])
    proof = json.loads(result.stdout)
    if baseline and assignment.get('baseline_reference_s') is not None:
        difference = abs(proof['total_s'] - float(assignment['baseline_reference_s']))
        proof['baseline_score_difference'] = difference
        if difference > float(assignment.get('baseline_score_tolerance', 1e-6)):
            raise ValueError(f'Baseline score reproduction mismatch: {difference}')
    proof['validated_utc'] = now()
    atomic(out / 'training-validation.json', proof)
    atomic(out / 'complete.json', dict(proof, publish='not_required',
           publish_reason='Preregistered diagnostic ablation; leaderboard publication is not requested.'))
    return proof


def cleanup_terminal_session(record):
    """Remove only a dead pane in the exact session created for this run."""
    identifier = record.get('tmux_session_id')
    if not identifier or session_id(record['tmux_session']) != identifier:
        return
    result = subprocess.run(['tmux', 'list-panes', '-s', '-t', identifier, '-F',
                             '#{session_id} #{pane_dead} #{session_windows}'], capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip() == identifier + ' 1 1':
        subprocess.run(['tmux', 'kill-session', '-t', identifier], check=True)


def tick(root, assignment, state):
    active = []
    for recipe in assignment['experiments']:
        name = recipe['experiment']
        entry = state.setdefault(name, {'state': 'scheduled'})
        if entry['state'] == 'trained_validated' and entry.get('imported_from') and not entry.get('import_checked'):
            try:
                out = root / 'runs' / name
                if (out / 'train.exit').read_text().strip() != '0' or not (out / 'imported.json').is_file():
                    raise ValueError('Imported run lacks successful exit/provenance receipt')
                proof = validate_output(root, assignment, recipe)
                entry.update(import_checked=True, validation=proof)
            except Exception as error:
                entry.update(state='failed', error=f'Imported artifact validation failed: {error}')
        if entry['state'] in TERMINAL:
            continue
        out = root / 'runs' / name
        if not (out / 'launch.json').exists():
            if out.exists():
                entry.update(state='failed', error='Partial launch directory; inspect before retry')
            continue
        record = json.loads((out / 'launch.json').read_text())
        if record.get('campaign_root') != str(root) or record.get('experiment') != name:
            entry.update(state='failed', error='Launch ownership does not match campaign')
            continue
        if (out / 'train.exit').exists():
            try:
                code = int((out / 'train.exit').read_text())
                if code:
                    raise ValueError(f'Training exited {code}; inspect train.log')
                proof = validate_output(root, assignment, recipe)
                entry.update(state='trained_validated', finished_utc=now(), validation=proof)
                entry.pop('reason', None)
                cleanup_terminal_session(record)
            except Exception as error:
                entry.update(state='failed', finished_utc=now(), error=str(error))
        elif session_alive(record['tmux_session']):
            entry.update(state='running')
            active.append(recipe)
        else:
            entry.update(state='failed', error='Session lost without an exit receipt')

    baselines = [r for r in assignment['experiments'] if r['stage'] == 'baseline']
    first = next(r for r in baselines if seed(r) == 42)
    first_state = state[first['experiment']]['state']
    baseline_ready = all(state[r['experiment']]['state'] == 'trained_validated' for r in baselines)
    baseline_failed = any(state[r['experiment']]['state'] in {'failed', 'cancelled_by_user'} for r in baselines)
    gate_error = host_gate_error(root, assignment)
    available = memory_gib()
    reserve = float(assignment.get('reserve_memory_gib', 6))
    cpu_budget = int(assignment.get('max_cpu_threads_total', os.cpu_count() or 1))
    free_cpu = max(0., cpu_budget - busy_cpu_cores())
    active_threads = sum(threads_for(assignment, r) for r in active)
    for recipe in assignment['experiments']:
        entry = state[recipe['experiment']]
        if entry['state'] not in {'scheduled', 'waiting'}:
            continue
        if gate_error:
            entry.update(state='waiting', reason=gate_error)
            continue
        if baseline_failed:
            entry.update(state='waiting', reason='Blocked by failed baseline reproduction/validation')
            continue
        if recipe is not first and first_state != 'trained_validated':
            entry.update(state='waiting', reason='Waiting for seed42 baseline reproduction')
            continue
        if recipe['stage'] != 'baseline' and not baseline_ready:
            entry.update(state='waiting', reason=('Waiting for all three validated baselines' if len(baselines) == 3 else 'Waiting for configured validated baseline'))
            continue
        dependency = recipe['config'].get('reuse_run')
        if dependency and state[dependency]['state'] != 'trained_validated':
            if state[dependency]['state'] in {'failed', 'cancelled_by_user'}:
                entry.update(state='failed', error=f'Required source run failed: {dependency}')
            else:
                entry.update(state='waiting', reason=f'Waiting for source run {dependency}')
            continue
        threads = threads_for(assignment, recipe)
        per_job = .5 if replay(recipe) else float(assignment.get('estimated_job_memory_gib', 5))
        if (len(active) >= int(assignment.get('max_jobs', 1)) or available < reserve + per_job
                or active_threads + threads > cpu_budget or free_cpu < threads):
            entry.update(state='waiting', reason='Waiting for CPU/RAM capacity')
            continue
        try:
            record = launch(root, assignment, recipe)
            entry.clear()
            entry.update(state='running', started_utc=record['started_utc'])
            active.append(recipe)
            available -= per_job
            active_threads += threads
            free_cpu -= threads
        except Exception as error:
            entry.update(state='failed', finished_utc=now(), error=f'Launch failed: {error}')
            if recipe['stage'] == 'baseline':
                baseline_failed = True


def analyze(root, assignment, state, *, final):
    """Finish the report before declaring the campaign complete."""
    entries = [state[r['experiment']] for r in assignment['experiments']]
    validated = sum(e['state'] == 'trained_validated' for e in entries)
    full = final and validated == len(entries)
    if not assignment.get('run_analysis', True):
        status = dict(finished_utc=now(), validated=validated, experiments=len(entries),
                      full=full, exit_code=0, analysis_deferred=True,
                      state='complete' if full else 'incomplete', host=assignment.get('host', 'golab2'))
        if final:
            atomic(root / 'host-complete.json', status)
        return status
    status_path = root / ('analysis-status.json' if final else 'analysis-progress.json')
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if (previous.get('exit_code') == 0 and previous.get('validated') == validated
                and previous.get('full') == full):
            return previous
    command = [assignment.get('python', DEFAULT_PYTHON), str(root / 'analyze_results.py'),
               '--root', str(root), '--bootstrap', '2000' if full else '0', '--block-hours', '168']
    if not full:
        command.append('--partial')
    prefix = 'analysis' if final else 'analysis-progress'
    status = dict(started_utc=now(), command=command, validated=validated,
                  experiments=len(entries), full=full)
    atomic(status_path, status)
    try:
        with (root / f'{prefix}.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        code = result.returncode
    except OSError as error:
        (root / f'{prefix}.log').write_text(str(error) + '\n')
        code = 127
    (root / f'{prefix}.exit').write_text(str(code) + '\n')
    status.update(finished_utc=now(), exit_code=code)
    atomic(status_path, status)
    if final:
        atomic(root / 'campaign-complete.json', dict(status,
               state='complete' if full and code == 0 else 'incomplete',
               failures={r['experiment']: state[r['experiment']] for r in assignment['experiments']
                         if state[r['experiment']]['state'] != 'trained_validated'}))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--interval', type=float, default=30)
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 1:
        parser.error('interval must be at least one second')
    root = args.root.resolve()
    assignment = load_assignment(root)
    if args.check:
        print(json.dumps({'valid': True, 'experiments': len(assignment['experiments'])}))
        return
    with scheduler_ownership(root):
        path = root / 'scheduler-state.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        while True:
            updated = load_assignment(root)
            validate_assignment_update(root, assignment, updated, state)
            assignment = updated
            tick(root, assignment, state)
            atomic(path, state)
            counts = {s: sum(e['state'] == s for e in state.values())
                      for s in ('scheduled', 'waiting', 'running', 'trained_validated', 'failed')}
            print(json.dumps({'utc': now(), 'counts': counts}), flush=True)
            failed_baseline = any(r['stage'] == 'baseline' and state[r['experiment']]['state'] in {'failed', 'cancelled_by_user'}
                                  for r in assignment['experiments'])
            terminal = all(state[r['experiment']]['state'] in TERMINAL for r in assignment['experiments'])
            blocked = failed_baseline and not counts['running']
            if terminal or blocked:
                status = analyze(root, assignment, state, final=True)
                if status['exit_code']:
                    raise SystemExit('Analysis failed; inspect analysis.log')
                if not status['full']:
                    raise SystemExit('Campaign incomplete; partial report and failure evidence preserved')
                break
            if args.once:
                break
            progress_path = root / 'analysis-progress.json'
            previous = json.loads(progress_path.read_text()) if progress_path.exists() else {}
            if counts['trained_validated'] >= previous.get('validated', 0) + 10:
                analyze(root, assignment, state, final=False)
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
