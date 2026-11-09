"""Prepare eight frozen additive axes, then run/collect ready axes after parent selection."""
import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pyarrow as pa
import build_data
import host_scheduler
import scheduler_support as support
from transport_support import REMOTE

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
COMBO = ROOT.parent / 'lag24_48_72_confirm_20261109'
PARENT = ROOT.parent / 'drop035_v650_confirm_20261108'
SOURCE = ROOT.parent / 'feature_addition_all_20261108'
AXES = {1: 'daily_shape', 2: 'recurrence_spread', 3: 'recurrence_disagreement',
        4: 'direct_morphology', 5: 'suvi_strip_reprojection', 6: 'ace_minute_variability',
        7: 'ace_minute_trend', 8: 'issued_cme_kinematics'}


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def admitted(root, stage):
    """One-thread preparation/analysis; never displace any live experiment."""
    with Path('/tmp/sunrun-cpu-admission.lock').open('a') as lock:
        while True:
            fcntl.flock(lock, fcntl.LOCK_EX)
            jobs = host_scheduler.workloads()
            if host_scheduler.capacity(jobs, support.memory_gib(), support.busy_cpu_cores(), threads=1):
                break
            fcntl.flock(lock, fcntl.LOCK_UN)
            support.atomic(root / 'controller-state.json', dict(phase='waiting_for_prepare_resources', stage=stage, utc=support.now()))
            time.sleep(10)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def select_parent(result):
    if result.get('status') != 'complete' or result.get('seeds') != [42, 43, 44]:
        raise ValueError('Parent decision requires a complete three-seed report')
    if result.get('add_features') != ['v_lag24', 'v_lag48', 'v_lag72']:
        raise ValueError('Unexpected parent experiment')
    use_combo = result['gate']['passed'] is True
    return dict(use_combo=use_combo, base_add_features=['v_lag24', 'v_lag48', 'v_lag72'] if use_combo else [],
                parent='D_plus_lag24_48_72' if use_combo else 'e2000fs_calib_offset_roll_drop_t035_v650',
                reason='Predeclared full improvement gate passed' if use_combo else 'Joint lag addition did not pass the predeclared adoption gate')


def freeze(root, decision):
    if (root / 'assignment.global.json').exists():
        frozen = read(root / 'parent-lock.json')
        if frozen['use_combo'] != decision['use_combo']:
            raise ValueError('Refuse changing frozen parent')
        return read(root / 'assignment.global.json')
    source = read(SOURCE / 'assignment.global.json')
    inventory = read(SOURCE / 'inventory.json')
    for axis in AXES:
        spec = build_data.describe(axis)
        for kind in ('constant', 'varying'):
            for name in spec[kind]:
                if any(r['name'] == name for r in inventory['candidates']):
                    raise ValueError('Derived feature collides with existing inventory: ' + name)
                inventory['candidates'].append(dict(name=name, kind=kind, status='eligible', group=f'axis{axis:02d}'))
    support.atomic(root / 'inventory.json', inventory)
    recipes = []
    for axis, label in AXES.items():
        spec = build_data.describe(axis)
        features = decision['base_add_features'] + list(spec['constant']) + list(spec['varying'])
        for seed in (42, 43, 44):
            recipes.append(dict(experiment=f'axis{axis:02d}_{label}_s{seed}', axis=axis, seed=seed,
                family='pooling', execution_host='golab2', threads=4,
                data=f'{REMOTE}/input/axis{axis}.parquet', config={'seed': seed, 'add_features': features},
                hypothesis=f'Conditional incremental information of axis{axis}:{label} above the fixed parent'))
    controls = {str(s): str((COMBO / 'runs' / f'lag24_48_72_s{s}') if decision['use_combo']
                           else (PARENT / 'runs' / f'drop035_v650_s{s}')) for s in (42, 43, 44)}
    assignment = dict(campaign=root.name, host='local', python=str(REPO / '.venv/bin/python'),
        data=source['data'], observations=source['observations'], observation_revision=source['observation_revision'],
        remote_observations='/home/yehwan/sunrun-offset-roll-complete-ablation-20261106/competition-observations.parquet',
        original_data_identity=support.identity(Path(source['data'])),
        experiments=recipes, baseline_runs=controls, threads=4, max_jobs=6, publication='not_required')
    support.atomic(root / 'parent-lock.json', dict(decision, selection_report=str(COMBO / 'analysis/results.json'), utc=support.now()))
    support.atomic(root / 'assignment.global.json', assignment)
    support.atomic(root / 'provenance.json', dict(parent=decision['parent'], base_add_features=decision['base_add_features'],
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        git_dirty=subprocess.check_output(['git', 'status', '--short'], cwd=REPO, text=True).splitlines(),
        observation_revision=source['observation_revision'], frozen_utc=support.now(),
        interpretation='Eight predeclared additive groups; repeated official-period exploratory evidence; no automatic feature combination'))
    support.atomic(root / 'configuration-diff.json', dict(
        base_add_features=decision['base_add_features'], retained_parent='all fixed parent input values and expanded input prefix',
        axes={str(axis): dict(added_constant=list(build_data.describe(axis)['constant']),
                             added_varying=list(build_data.describe(axis)['varying']),
                             expected_source_count=28 + len(decision['base_add_features']) + len(build_data.describe(axis)['constant']) + len(build_data.describe(axis)['varying']),
                             expected_expanded_count=212 + len(decision['base_add_features']) + len(build_data.describe(axis)['constant']) + 24 * len(build_data.describe(axis)['varying'])) for axis in AXES}))
    state = {r['experiment']: {'state': 'waiting' if r['axis'] <= 7 else 'blocked_source',
                               'reason': None if r['axis'] <= 7 else 'Historical issued-body provenance unproven'} for r in recipes}
    support.atomic(root / 'runtime.json', state)
    return assignment


def start_service(root, script, name):
    # Exact-owned, tested launch helper; no shutdown/restart of live training.
    import importlib.util
    spec = importlib.util.spec_from_file_location('axes8_launcher', COMBO / 'launch_services.py')
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
        source = read(SOURCE / 'assignment.global.json')['data']
        for axis in range(1, 8):
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
        report = COMBO / 'analysis/results.json'
        while not report.exists():
            support.atomic(root / 'controller-state.json', dict(phase='waiting_for_parent_analysis', utc=support.now(), prepared_axes=list(range(1, 8))))
            time.sleep(20)
        assignment = freeze(root, select_parent(read(report)))
        support.atomic(root / 'controller-state.json', dict(phase='staging_remote', utc=support.now()))
        stage(root, assignment, list(range(1, 8)))
        start_service(root, 'remote_scheduler_controller.py', 'sraxes8-remote-controller-20261109')
        last_analysis = 0.
        while True:
            try:
                snapshot = remote_snapshot()
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                support.atomic(root / 'collector-error.json', dict(utc=support.now(), error=str(error), retry=True))
                time.sleep(30)
                continue
            distant = snapshot.get('state', snapshot)
            if not isinstance(distant, dict):
                raise ValueError('Unexpected remote state snapshot')
            state = {r['experiment']: {'state': 'blocked_source', 'reason': 'Historical issued-body provenance unproven'}
                     for r in assignment['experiments'] if r['axis'] == 8}
            for name, row in distant.items():
                row = dict(row)
                if row['state'] == 'running' and not snapshot.get('scheduler_alive', False):
                    row.update(state='interrupted', reason='Actual foreground scheduler is absent; stale running state rejected')
                if row['state'] == 'trained_validated':
                    try:
                        collect_run(root, name)
                    except (OSError, ValueError, subprocess.SubprocessError) as error:
                        row.update(state='collection_pending', reason=str(error))
                state[name] = row
            support.atomic(root / 'runtime.json', state)
            ready_done = len(distant) == 21 and all(state[name]['state'] == 'trained_validated' for name in distant)
            summary = dict(phase='waiting_for_axis8_provenance' if ready_done else 'training', utc=support.now(),
                           counts=dict(Counter(v['state'] for v in state.values())), parent=assignment.get('parent'), publication='not_required')
            if any(v['state'] in ('interrupted', 'failed') for v in state.values()):
                summary['phase'] = 'training_with_blockers'
                support.atomic(root / 'training-blocker.json', dict(utc=support.now(), affected={k:v for k,v in state.items() if v['state'] in ('interrupted','failed')}))
            support.atomic(root / 'controller-state.json', summary)
            if ready_done or time.monotonic() - last_analysis > 3600:
                with admitted(root, 'paired_analysis'):
                    with (root / 'analysis.log').open('a') as log:
                        result = subprocess.run([assignment['python'], str(root / 'analyze_results.py'), '--root', str(root), '--bootstrap', '2000'], stdout=log, stderr=subprocess.STDOUT)
                last_analysis = time.monotonic()
                if result.returncode:
                    support.atomic(root / 'analysis-error.json', dict(exit_code=result.returncode, utc=support.now()))
                if ready_done:
                    support.atomic(root / 'ready-axes-complete.json', dict(complete=True, completed_axes=list(range(1, 8)), blocked_axes=[8],
                         campaign_complete=False, analysis_exit=result.returncode, utc=support.now()))
                    return 0 if result.returncode == 0 else 1
            print(json.dumps(summary), flush=True)
            time.sleep(30)


if __name__ == '__main__':
    raise SystemExit(main())
