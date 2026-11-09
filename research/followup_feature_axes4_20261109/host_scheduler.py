"""Resource-gated locked additive-axis CPU scheduler; report-only experiments."""
import argparse
import json
from pathlib import Path
import time
import shutil
from collections import Counter

import scheduler_support as support


def workloads(proc_root=Path('/proc')):
    """Count live CPU experiment interpreters, not their shell parents."""
    result = []
    for folder in proc_root.iterdir():
        if not folder.name.isdigit():
            continue
        try:
            argv = (folder / 'cmdline').read_bytes().decode().split('\0')
            if not argv or 'python' not in Path(argv[0]).name:
                continue
            if any(Path(a).name == 'analyze_results.py' for a in argv) and '--root' in argv:
                result.append({'pid': int(folder.name), 'threads': 1, 'out': argv[argv.index('--root') + 1]})
                continue
            if '--threads' not in argv or '--out' not in argv:
                continue
            if '--device' in argv and argv[argv.index('--device') + 1] != 'cpu':
                continue
            result.append({'pid': int(folder.name), 'threads': int(argv[argv.index('--threads') + 1]),
                           'out': argv[argv.index('--out') + 1]})
        except (OSError, ValueError, IndexError, UnicodeError):
            continue
    return result


def capacity(jobs, available, busy, threads=4):
    return (len(jobs) < 6 and sum(j['threads'] for j in jobs) + threads <= 24
            and available >= 11 and busy + threads <= 24)


def tick(root, assignment, state):
    active = []
    for recipe in assignment['experiments']:
        name = recipe['experiment']
        row = state.setdefault(name, {'state': 'waiting'})
        if row['state'] in support.TERMINAL:
            continue
        run = root / 'runs' / name
        if not (run / 'launch.json').exists():
            if run.exists():
                row.update(state='failed', error='Partial run directory without launch receipt')
            continue
        launch = json.loads((run / 'launch.json').read_text())
        if launch['campaign_root'] != str(root) or launch['experiment'] != name:
            raise ValueError('Launch identity mismatch')
        if (run / 'train.exit').exists():
            try:
                if (run / 'train.exit').read_text().strip() != '0':
                    raise ValueError('Training failed; inspect train.log')
                proof = support.validate_output(root, assignment, recipe)
                support.atomic(run / 'complete.json', dict(proof, publish='not_required',
                               publish_reason='Preregistered follow-up ablation; report only, no publication.'))
                row.update(state='trained_validated', finished_utc=support.now(), validation=proof)
                row.pop('reason', None)
                support.cleanup_terminal_session(launch)
            except Exception as error:
                row.update(state='failed', error=str(error), finished_utc=support.now())
        elif support.session_id(launch['tmux_session']) == launch['tmux_session_id']:
            row.update(state='running')
            active.append(name)
        else:
            row.update(state='failed', error='Exact run session lost without exit receipt')
    for recipe in assignment['experiments']:
        row = state[recipe['experiment']]
        if row['state'] != 'waiting':
            continue
        jobs = workloads()
        available = support.memory_gib()
        busy = support.busy_cpu_cores()
        free_disk_gib = shutil.disk_usage(root).free / 1024**3
        if free_disk_gib < 15:
            row.update(reason='Waiting for at least15GiB free disk; retained models are never deleted',
                       resources={'free_disk_gib': free_disk_gib})
            break
        observed = {job['out'] for job in jobs}
        pending = [name for name in active if str(root / 'runs' / name) not in observed]
        reserved_jobs = jobs + [{'threads': 4} for name in pending]
        if len(active) >= assignment.get('max_jobs', 6) or not capacity(reserved_jobs, available - 5 * len(pending), busy + 4 * len(pending)):
            row.update(reason='Waiting for one of six global CPU slots; <=24 threads and >=6GiB reserve',
                       resources={'workloads': len(jobs), 'threads': sum(j['threads'] for j in jobs),
                                  'available_gib': available, 'busy_cores': busy})
            break
        try:
            record = support.launch(root, assignment, recipe)
        except Exception as error:
            row.update(state='failed', error='Launch failure: ' + str(error), finished_utc=support.now())
            continue
        row.clear()
        row.update(state='running', started_utc=record['started_utc'])
        active.append(recipe['experiment'])


def validate_preflight(root, assignment):
    """Reject source/input drift before admitting any queued job."""
    gate = json.loads((root / 'preflight.json').read_text())
    if not gate.get('passed'):
        raise ValueError('Follow-up parent lock and feature smoke gate has not passed')
    host_proof = gate[assignment['host']]
    inputs = host_proof['input_identities']
    for recipe in assignment['experiments']:
        support.expected_config(recipe)
        path = support.input_path(assignment, recipe)
        if support.identity(path) != inputs.get(str(path)):
            raise ValueError('Prepared per-axis input changed after host preflight')
        if int(recipe.get('threads', assignment.get('threads', 4))) != 4:
            raise ValueError('Each CPU ablation requires exactly four threads')
    sources = host_proof['source_identity']
    required = {'core.py', 'baseline.py', 'competition.py', 'inventory.json',
                'replay.py', 'scheduler_support.py', 'host_scheduler.py', 'assignment.json'}
    if not required.issubset(sources):
        raise ValueError('Preflight lacks complete frozen source/assignment identity')
    for filename, identity in sources.items():
        if support.identity(root / filename) != identity:
            raise ValueError('Frozen source or assignment changed after preflight: ' + filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    assignment = json.loads((root / 'assignment.json').read_text())
    validate_preflight(root, assignment)
    with support.scheduler_ownership(root):
        state = json.loads((root / 'scheduler-state.json').read_text()) if (root / 'scheduler-state.json').exists() else {}
        if set(state) - {recipe['experiment'] for recipe in assignment['experiments']}:
            raise ValueError('Scheduler state contains experiments outside frozen assignment')
        # A persisted terminal status alone is never proof of completion.
        for recipe in assignment['experiments']:
            row = state.get(recipe['experiment'], {})
            if row.get('state') == 'trained_validated':
                try:
                    support.validate_output(root, assignment, recipe)
                except Exception as error:
                    row.update(state='failed', error='Resume validation failed: ' + str(error))
        while True:
            # Shared host admission lock covers resource inspection and launch atomically.
            import fcntl
            with Path('/tmp/sunrun-cpu-admission.lock').open('a') as admission:
                fcntl.flock(admission, fcntl.LOCK_EX)
                tick(root, assignment, state)
            support.atomic(root / 'scheduler-state.json', state)
            support.atomic(root / 'runtime.json', state)
            print(json.dumps({'utc': support.now(), 'counts': dict(Counter(r['state'] for r in state.values()))}), flush=True)
            if all(row['state'] in support.TERMINAL for row in state.values()):
                valid = all(row['state'] == 'trained_validated' for row in state.values())
                support.atomic(root / 'host-complete.json', {'finished_utc': support.now(), 'complete': valid})
                raise SystemExit(0 if valid else 1)
            if args.once:
                break
            time.sleep(20)


if __name__ == '__main__':
    main()
