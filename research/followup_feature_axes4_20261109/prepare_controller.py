"""Prepare four fixed additions and execute after the preceding ready axes finish."""
import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import pyarrow as pa
import build_data
import host_scheduler
import scheduler_support as support

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PREVIOUS = ROOT.parent / 'followup_feature_axes8_20261109'
AXES = (1, 2, 3, 4)


def read(path):
    return json.loads(Path(path).read_text())


def previous_ready(previous=PREVIOUS):
    """Require validated ready-arm completion, without waiting for blocked CME."""
    previous = Path(previous)
    receipt = previous / 'ready-axes-complete.json'
    if not receipt.exists():
        return False
    value = read(receipt)
    if value.get('complete') is not True or value.get('analysis_exit') != 0:
        return False
    if value.get('completed_axes') != list(range(1, 8)):
        return False
    assignment = read(previous / 'assignment.global.json')
    ready = [r for r in assignment['experiments'] if r['axis'] in range(1, 8)]
    if len(ready) != 21 or len({r['experiment'] for r in ready}) != 21:
        raise ValueError('Unexpected preceding campaign ready-arm manifest')
    state = read(previous / 'runtime.json')
    return all(state.get(r['experiment'], {}).get('state') == 'trained_validated' for r in ready)


def validate_assignment(root, assignment):
    from analyze_results import conditions_from
    conditions_from(assignment)
    parent = read(root / 'parent-lock.json')
    if parent.get('parent') != 'e2000fs_calib_offset_roll_drop_t035_v650' or parent.get('base_add_features', []) != []:
        raise ValueError('Follow-up requires the frozen D parent')
    for recipe in assignment['experiments']:
        expected = build_data.describe(recipe['axis'])
        if recipe['config']['add_features'] != list(expected['constant']) + list(expected['varying']):
            raise ValueError('Recipe features differ from the frozen provider')
    return assignment


def validate_sources(root, assignment):
    if read(root / 'source-smoke.json').get('passed') is not True:
        raise ValueError('Actual-source smoke gate has not passed')
    if support.identity(Path(assignment['data'])) != assignment['original_data_identity']:
        raise ValueError('Original prepared input changed after registration')
    common = ('core.py', 'baseline.py', 'competition.py', 'replay.py',
              'scheduler_support.py', 'host_scheduler.py')
    for name in common:
        if (root / name).read_bytes() != (PREVIOUS / name).read_bytes():
            raise ValueError('Validated common runner changed: ' + name)
    build_data.validate_minute_sources(root)


@contextmanager
def admitted(root, stage):
    with Path('/tmp/sunrun-cpu-admission.lock').open('a') as lock:
        while True:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if host_scheduler.capacity(host_scheduler.workloads(), support.memory_gib(), support.busy_cpu_cores(), threads=1):
                break
            fcntl.flock(lock, fcntl.LOCK_UN)
            support.atomic(root / 'controller-state.json', dict(phase='waiting_for_prepare_resources', stage=stage, utc=support.now()))
            time.sleep(10)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def start_service(root, script, name):
    spec = importlib.util.spec_from_file_location('axes4_launcher', ROOT.parent / 'lag24_48_72_confirm_20261109/launch_services.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.start(root, str(REPO / '.venv/bin/python'), script, name, resume=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    from remote_services import stage, remote_snapshot, collect_run
    with (root / '.prepare-controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assignment = validate_assignment(root, read(root / 'assignment.global.json'))
        validate_sources(root, assignment)
        source = assignment['data']
        for axis in AXES:
            cache = root / 'input' / f'axis{axis}.features.parquet'
            if cache.exists():
                if not cache.with_suffix('.json').exists():
                    raise RuntimeError('Partial feature cache; preserve before recovery: ' + str(cache))
                if read(cache.with_suffix('.json'))['source_identity'] != support.identity(Path(source)):
                    raise RuntimeError('Prepared input changed after feature cache freeze')
                continue
            support.atomic(root / 'controller-state.json', dict(phase='preparing_features', axis=axis, utc=support.now()))
            with admitted(root, f'axis{axis}_cache'):
                build_data.build_cache(source, cache, axis)
        while not previous_ready():
            summary = dict(phase='waiting_for_preceding_axes', predecessor=str(PREVIOUS), ready_axes_required=list(range(1, 8)), blocked_axis8_waited=False, utc=support.now())
            support.atomic(root / 'controller-state.json', summary)
            print(json.dumps(summary), flush=True)
            time.sleep(30)
        support.atomic(root / 'predecessor-release.json', dict(passed=True, source=str(PREVIOUS), receipt=read(PREVIOUS / 'ready-axes-complete.json'), utc=support.now()))
        support.atomic(root / 'controller-state.json', dict(phase='staging_remote', utc=support.now()))
        stage(root, assignment, list(AXES))
        start_service(root, 'remote_scheduler_controller.py', 'sraxes4-remote-controller-20261109')
        while True:
            try:
                snapshot = remote_snapshot()
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                support.atomic(root / 'collector-error.json', dict(utc=support.now(), error=str(error), retry=True))
                time.sleep(30)
                continue
            distant = snapshot.get('state', snapshot)
            names = {r['experiment'] for r in assignment['experiments']}
            if not isinstance(distant, dict) or set(distant) != names:
                raise ValueError('Remote state does not match all twelve frozen runs')
            state = {}
            for name, row in distant.items():
                row = dict(row)
                if row['state'] == 'running' and not snapshot.get('scheduler_alive', False):
                    row.update(state='interrupted', reason='Actual foreground scheduler absent; stale state rejected')
                if row['state'] == 'trained_validated':
                    try:
                        collect_run(root, name)
                    except (OSError, ValueError, subprocess.SubprocessError) as error:
                        row.update(state='collection_pending', reason=str(error))
                state[name] = row
            support.atomic(root / 'runtime.json', state)
            done = all(row['state'] == 'trained_validated' for row in state.values())
            blockers = {k: v for k, v in state.items() if v['state'] in ('failed', 'interrupted')}
            summary = dict(phase='analyzing' if done else 'training_with_blockers' if blockers else 'training', utc=support.now(), counts=dict(Counter(v['state'] for v in state.values())), parent='e2000fs_calib_offset_roll_drop_t035_v650', publication='not_required')
            support.atomic(root / 'controller-state.json', summary)
            if blockers:
                support.atomic(root / 'training-blocker.json', dict(utc=support.now(), affected=blockers))
            if done:
                with admitted(root, 'paired_analysis'):
                    with (root / 'analysis.log').open('a') as log:
                        result = subprocess.run([assignment['python'], str(root / 'analyze_results.py'), '--root', str(root), '--bootstrap', '2000'], stdout=log, stderr=subprocess.STDOUT)
                report = read(root / 'analysis/results.json') if (root / 'analysis/results.json').exists() else {}
                complete = result.returncode == 0 and report.get('status') == 'complete'
                support.atomic(root / 'campaign-complete.json', dict(complete=complete, completed_axes=list(AXES), analysis_exit=result.returncode, publication='not_required', utc=support.now()))
                support.atomic(root / 'controller-state.json', dict(summary, phase='complete' if complete else 'analysis_blocked'))
                return 0 if complete else 1
            print(json.dumps(summary), flush=True)
            time.sleep(30)


if __name__ == '__main__':
    raise SystemExit(main())
