"""Adopt stopped golab2 artifacts and restart the remaining original recipes locally.

The recovery directory is an untouched copy of the remote run. Only copied JSON
path/identity fields are rebased; executable snapshots, models and predictions
retain their bytes. Failed/interrupted runs are archived before a fresh attempt.
"""
from __future__ import annotations

import ast
import filecmp
import json
from datetime import UTC, datetime
from pathlib import Path
import shutil

from migrate_local import rebase

HERE = Path(__file__).resolve().parent
RECOVERED = HERE / 'recovered_golab2_20261104'
REMOTE_ROOT = '/home/yehwan/sunrun-golab2-20261103-2300'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def expected_source(family, name):
    source = (HERE / f'{family}.py').read_text()
    entry = ast.parse(source).body[-1]
    return ('\n'.join(source.splitlines()[:entry.lineno - 1])
            + '\nif __name__ == "__main__":\n    import sys\n'
            + '    if "--experiment" not in sys.argv:\n'
            + f'        sys.argv.extend(["--experiment", {name!r}])\n    main()\n').encode()


def archive_attempt(name, reason):
    out = HERE / 'runs' / name
    if not out.exists():
        return None
    record = json.loads((out / 'launch.json').read_text())
    assert not (out / 'submission.json').exists(), 'Never archive submitted results'
    assert (out / 'experiment.py').read_bytes() == expected_source(record['family'], name), name + ': model code drift'
    assert (out / 'experiment.py').read_bytes() == (out / 'runner_at_launch.py').read_bytes()
    target = HERE / 'interrupted_local_20261104' / name
    target.parent.mkdir(exist_ok=True)
    assert not target.exists()
    out.rename(target)
    write_json(target / 'retry-decision.json', {'utc': datetime.now(UTC).isoformat(), 'reason': reason,
                                               'source_bytes_equal_current_recipe': True,
                                               'previous_outputs_used_as_new_predictions': False})
    return str(target)


def main():
    assert (RECOVERED / 'transfer-gzip.exit').read_text().strip() == '0'
    stop = json.loads((RECOVERED / 'local-transfer-stop.json').read_text())
    assert not stop['remaining_training'] and stop['start_script_disabled']
    assert stop['task']['returncode'] == 0
    # Compare complete bytes directly: file-size checks alone do not prove input equality.
    comparisons = {}
    for copied, local in [('input/df_ffill_trainy.parquet', HERE / 'input/df_ffill_trainy.parquet'),
                          ('raw/df.parquet', HERE.parents[1] / 'data-v2/input/df.parquet')]:
        assert filecmp.cmp(RECOVERED / copied, local, shallow=False), copied + ': input differs'
        comparisons[copied] = {'equal_bytes': True, 'bytes': local.stat().st_size}
    data = HERE / 'input/df_ffill_trainy.parquet'
    identity = {'path': str(data), 'bytes': data.stat().st_size, 'mtime_ns': data.stat().st_mtime_ns}
    assignment = json.loads((RECOVERED / 'assignment.json').read_text())
    remote_state = json.loads((RECOVERED / 'scheduler-state.json').read_text())
    registry = json.loads((HERE / 'registry.json').read_text())
    runtime = json.loads((HERE / 'runtime.json').read_text())
    write_json(HERE / 'registry.before-local-adoption-20261104.json', registry)
    write_json(HERE / 'runtime.before-local-adoption-20261104.json', runtime)
    receipt = {'utc': datetime.now(UTC).isoformat(), 'inputs': comparisons, 'adopted_completed': [],
               'remote_to_local_queue': [], 'local_retries': {}, 'added_missing_original_recipes': []}
    for recipe in assignment['experiments']:
        name = recipe['experiment']
        assert name not in registry, 'Duplicate assignment: ' + name
        registry[name] = {'family': recipe['family'], 'threads': 2}
        state = remote_state[name]
        if state['state'] != 'trained_validated':
            assert not (HERE / 'runs' / name).exists()
            runtime[name] = {'state': 'queued', 'migration_from': 'golab2', 'previous_state': state['state']}
            receipt['remote_to_local_queue'].append({'experiment': name, 'previous_state': state['state']})
            continue
        original = RECOVERED / 'runs' / name
        out = HERE / 'runs' / name
        assert not out.exists()
        assert (original / 'train.exit').read_text().strip() == '0'
        assert (original / 'predictions.parquet').is_file()
        assert (original / 'experiment.py').read_bytes() == (original / 'runner_at_launch.py').read_bytes()
        shutil.copytree(original, out, copy_function=shutil.copy2)
        old = json.loads((original / 'launch.json').read_text())
        # Original launch and all fit reports remain in RECOVERED for provenance.
        for path in out.rglob('*.json'):
            value = json.loads(path.read_text())
            changed = rebase(value, REMOTE_ROOT, str(HERE), old['input_identity'], identity)
            changed = rebase(changed, str(HERE / 'raw'), str(HERE.parents[1] / 'data-v2/input'), {}, {})
            if path.name == 'launch.json':
                changed['training_host'] = 'golab2'
                changed['artifact_location'] = 'local'
                changed['migration'] = {'utc': receipt['utc'], 'original_launch': str(original / 'launch.json'),
                                        'input_content_equal': True, 'metadata_path_rebase_only': True}
            if changed != value:
                path.write_text(json.dumps(changed, indent=2) + '\n')
        assert filecmp.cmp(original / 'predictions.parquet', out / 'predictions.parquet', shallow=False)
        assert filecmp.cmp(original / 'experiment.py', out / 'experiment.py', shallow=False)
        runtime[name] = {'state': 'trained', 'training_host': 'golab2', 'migration_to': 'local'}
        receipt['adopted_completed'].append(name)
    for name in ('e2000fs_mimo', 'e2000fs_recmo_24', 'e2000fs_dirmo_6', 'e2000fs_feat30_extreme_histprior'):
        receipt['local_retries'][name] = archive_attempt(name, 'Host memory exhaustion interrupted prior campaign; retry with conservative memory reservation and at most two local jobs')
        runtime[name] = {'state': 'queued', 'retry_diagnosis': 'prior host memory exhaustion; partial outputs archived; new admission policy'}
    manifest = json.loads((HERE / 'manifest.json').read_text())
    excluded = set(manifest.get('user_excluded_experiments', []))
    for name in ('e2000fs_realmlp_direct', 'e2000fs_tabiclv2_direct'):
        assert name not in excluded and name not in registry
        registry[name] = {'family': 'foundation', 'threads': 2}
        runtime[name] = {'state': 'queued'}
        receipt['added_missing_original_recipes'].append(name)
    expected = {e['experiment'] for e in manifest['experiments']} - excluded
    assert set(registry) == expected, {'missing': sorted(expected - set(registry)), 'extra': sorted(set(registry) - expected)}
    receipt['total_admitted'] = len(registry)
    receipt['already_submitted'] = sum((HERE / 'runs' / name / 'submission.json').exists() for name in registry)
    receipt['remaining_training'] = len(registry) - receipt['already_submitted'] - len(receipt['adopted_completed'])
    write_json(HERE / 'registry.json', registry)
    write_json(HERE / 'runtime.json', runtime)
    write_json(HERE / 'local-adoption-20261104.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
