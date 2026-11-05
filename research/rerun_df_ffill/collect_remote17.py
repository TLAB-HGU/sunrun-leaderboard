"""Collect the exact golab2 17-run assignment and publish validated results.

Original remote artifacts stay in a separate recovery directory. Publication is
serialized by publish.py's existing lock; receipts also resolve races with the
local controller. This program never starts or retries remote training.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import time

from golab2_transport import remote
from migrate_local import rebase
import publish

HERE = Path(__file__).resolve().parent
REMOTE_ROOT = '/home/yehwan/sunrun-golab2-20261104-17'
ASSIGNMENT = HERE / 'golab2-assignment-20261104-17.json'
RECOVERY = HERE / 'recovered_golab2_20261104_17'
STATE = HERE / 'remote17-collection-state.json'
TERMINAL = {'submitted', 'cancelled_by_user', 'remote_failed', 'collection_failed', 'publication_failed'}
MAX_ATTEMPTS = 5


def now():
    return datetime.now(UTC).isoformat()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def emit(event, **fields):
    print(json.dumps({'utc': now(), 'event': event, **fields}), flush=True)


def identity(path):
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def load_assignment(path):
    assignment = json.loads(path.read_text())
    if assignment.get('remote_root') != REMOTE_ROOT:
        raise ValueError('Collector requires the exact 20261104-17 remote root')
    names = [entry['experiment'] for entry in assignment['experiments']]
    # 17 original + 4 added on 2026-11-04 (remote4-addition-20261104.json); names must stay unique
    if len(names) < 17 or len(set(names)) != len(names) or any(not re.fullmatch(r'[A-Za-z0-9_]+', n) for n in names):
        raise ValueError('Expected at least seventeen unique experiment names')
    for remote_key, local_key, local_path, remote_path in (
        ('input_identity', 'local_input_identity', HERE / 'input/df_ffill_trainy.parquet', REMOTE_ROOT + '/input/df_ffill_trainy.parquet'),
        ('raw_identity', 'local_raw_identity', HERE.parents[1] / 'data-v2/input/df.parquet', REMOTE_ROOT + '/raw/df.parquet'),
    ):
        expected = assignment[remote_key]
        if expected['path'] != remote_path:
            raise ValueError('Unexpected deployed input path')
        local = identity(local_path)
        recorded_local = assignment.get(local_key, {**expected, 'path': str(local_path)})
        if local != recorded_local or local['bytes'] != expected['bytes']:
            raise ValueError('Local input differs from the deployment receipt')
    return assignment


def read_remote_state(assignment):
    code = ('from pathlib import Path\nimport json\nr=Path(' + repr(REMOTE_ROOT) + ')\n'
            'a=json.loads((r/"assignment.json").read_text())\n'
            'assert [e["experiment"] for e in a["experiments"]]=='
            + repr([e['experiment'] for e in assignment['experiments']]) + '\n'
            'for expected in ' + repr([assignment['input_identity'], assignment['raw_identity']]) + ':\n'
            ' p=Path(expected["path"]);s=p.stat()\n'
            ' assert (s.st_size,s.st_mtime_ns)==(expected["bytes"],expected["mtime_ns"]), "Remote input changed"\n'
            'print((r/"scheduler-state.json").read_text())\n')
    return json.loads(remote(code, timeout=30).stdout)


def recover(name, assignment, remote_entry):
    """Stream one closed run, retaining its original JSON and executable bytes."""
    RECOVERY.mkdir(exist_ok=True)
    original = RECOVERY / name
    marker = RECOVERY / (name + '.recovery.json')
    if original.exists():
        receipt = json.loads(marker.read_text())
        if receipt['remote_root'] != REMOTE_ROOT or receipt['experiment'] != name:
            raise ValueError('Conflicting recovered directory')
        return original
    partial = RECOVERY / (name + '.tar.gz.part')
    archive = RECOVERY / (name + '.tar.gz')
    completed = remote_entry['state'] == 'trained_validated'
    code = ('from pathlib import Path\nimport json,sys,tarfile\n'
            'r=Path(' + repr(REMOTE_ROOT) + ')\np=r/"runs"/' + repr(name) + '\n'
            'assert p.is_dir(), "Remote run directory absent"\n'
            'if ' + repr(completed) + ':\n'
            ' assert (p/"train.exit").read_text().strip()=="0"\n'
            ' record=json.loads((p/"launch.json").read_text())\n'
            ' assert record["input_identity"]==' + repr(assignment['input_identity']) + '\n'
            ' assert record["experiment"]==' + repr(name) + '\n'
            'with tarfile.open(fileobj=sys.stdout.buffer,mode="w|gz",compresslevel=1) as t:\n'
            ' t.add(p,arcname=' + repr(name) + ')\n')
    with partial.open('wb') as output, (RECOVERY / (name + '.transfer.log')).open('ab') as log:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                                 'golab2', 'wsl -d Ubuntu -- python3 -'], input=code.encode(),
                                stdout=output, stderr=log, timeout=1800,
                                env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    if result.returncode:
        raise RuntimeError(f'Artifact transfer exited {result.returncode}')
    partial.replace(archive)
    with tempfile.TemporaryDirectory(prefix='.extract-', dir=RECOVERY) as temporary:
        with tarfile.open(archive) as stream:
            for member in stream.getmembers():
                parts = Path(member.name).parts
                if not parts or parts[0] != name or '..' in parts or not (member.isfile() or member.isdir()):
                    raise ValueError('Unexpected archive member')
            stream.extractall(temporary, filter='data')
        extracted = Path(temporary) / name
        # Write the receipt before rename; an installed original always has provenance.
        atomic(marker, {'experiment': name, 'remote_root': REMOTE_ROOT, 'collected_utc': now(),
                        'remote_state': remote_entry})
        extracted.rename(original)
    return original


def verify_existing(name):
    out = HERE / 'runs' / name
    record = json.loads((out / 'launch.json').read_text())
    if record.get('experiment') != name or record.get('migration', {}).get('remote_root') != REMOTE_ROOT:
        raise ValueError('Existing local run belongs to a different execution')
    publish.validate_run(name)
    return out


def is_placeholder(out, name):
    if not out.is_dir() or sorted(p.name for p in out.iterdir()) != ['remote-assignment.json']:
        return False
    marker = json.loads((out / 'remote-assignment.json').read_text())
    return (marker.get('experiment') == name and marker.get('host') == 'golab2'
            and marker.get('remote_root') == REMOTE_ROOT)


def install(name, original, assignment):
    """Expose a ready local run only after copying and rebasing all metadata."""
    out = HERE / 'runs' / name
    if out.exists() and not is_placeholder(out, name):
        return verify_existing(name)
    record = json.loads((original / 'launch.json').read_text())
    if record.get('experiment') != name or record['input_identity'] != assignment['input_identity']:
        raise ValueError('Recovered launch differs from the frozen assignment input')
    if (original / 'train.exit').read_text().strip() != '0':
        raise ValueError('Cannot install failed training')
    if (original / 'experiment.py').read_bytes() != (original / 'runner_at_launch.py').read_bytes():
        raise ValueError('Recovered source differs from launch snapshot')
    local_input = identity(HERE / 'input/df_ffill_trainy.parquet')
    staging_root = HERE / '.remote17-staging'
    staging_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=name + '-', dir=staging_root) as temporary:
        staged = Path(temporary) / name
        shutil.copytree(original, staged)
        for path in staged.rglob('*.json'):
            value = json.loads(path.read_text())
            changed = rebase(value, REMOTE_ROOT, str(HERE), record['input_identity'], local_input)
            changed = rebase(changed, str(HERE / 'raw'), str(HERE.parents[1] / 'data-v2/input'), {}, {})
            if path.name == 'launch.json':
                changed.update(training_host='golab2', artifact_location='local',
                               migration={'remote_root': REMOTE_ROOT, 'collected_utc': now(),
                                          'original_launch': str(original / 'launch.json'),
                                          'metadata_path_rebase_only': True})
            if changed != value:
                path.write_text(json.dumps(changed, indent=2) + '\n')
        # Validate before exposure using the same validator with a staging root.
        old_here = publish.HERE
        try:
            publish.HERE = Path(temporary)
            (Path(temporary) / 'runs').mkdir()
            staged.rename(Path(temporary) / 'runs' / name)
            staged = Path(temporary) / 'runs' / name
            (Path(temporary) / 'input').symlink_to(HERE / 'input', target_is_directory=True)
            publish.validate_run(name)
        finally:
            publish.HERE = old_here
        if out.exists():
            if not is_placeholder(out, name):
                raise ValueError('Local run appeared during collection; refusing overwrite')
            RECOVERY.mkdir(exist_ok=True)
            placeholder_archive = RECOVERY / (name + '.reservation')
            if placeholder_archive.exists():
                raise ValueError('Placeholder archive already exists; inspect interrupted install')
            out.rename(placeholder_archive)
        staged.rename(out)
    return out


def publish_ready(name):
    out = verify_existing(name)
    if (out / 'submission.json').exists():
        return
    with (out / 'remote17-publish.log').open('a') as log:
        result = subprocess.run([str(HERE.parents[1] / '.venv/bin/python'), str(HERE / 'publish.py'), name],
                                stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    # The local controller can publish while this process waits for publish.py's lock.
    if not (out / 'submission.json').exists():
        raise RuntimeError(f'Publication returned {result.returncode} without a submission receipt')


def tick(assignment, state):
    remote_state = read_remote_state(assignment)
    manifest_path = HERE / 'manifest.json'
    excluded = set(json.loads(manifest_path.read_text()).get('user_excluded_experiments', [])) if manifest_path.exists() else set()
    for recipe in assignment['experiments']:
        name = recipe['experiment']
        entry = state.setdefault(name, {})
        out = HERE / 'runs' / name
        if name in excluded or remote_state.get(name, {}).get('state') == 'cancelled_by_user':
            entry.update(phase='cancelled_by_user', reason='Explicit user exclusion; artifacts preserved')
            continue
        if (out / 'submission.json').exists():
            verify_existing(name)
            entry.update(phase='submitted', submitted_utc=now())
            continue
        if entry.get('phase') in TERMINAL or time.time() < entry.get('retry_after', 0):
            continue
        remote_entry = remote_state.get(name, {'state': 'scheduled'})
        entry['remote_state'] = remote_entry
        if remote_entry['state'] not in {'trained_validated', 'failed'}:
            entry['phase'] = 'waiting_remote'
            continue
        try:
            original = recover(name, assignment, remote_entry)
            if remote_entry['state'] == 'failed':
                entry.update(phase='remote_failed', original=str(original), reason=remote_entry.get('reason'))
                emit('remote_failed_preserved', experiment=name, reason=entry['reason'])
                continue
            install(name, original, assignment)
            entry.update(phase='ready', original=str(original))
            atomic(STATE, state)
            publish_ready(name)
            entry.update(phase='submitted', submitted_utc=now())
            emit('submitted', experiment=name)
        except Exception as error:
            entry['attempts'] = entry.get('attempts', 0) + 1
            entry.update(error=f'{type(error).__name__}: {error}', retry_after=time.time() + 300)
            if entry['attempts'] >= MAX_ATTEMPTS or isinstance(error, ValueError):
                entry['phase'] = 'publication_failed' if entry.get('phase') == 'ready' else 'collection_failed'
            emit('collection_error', experiment=name, attempts=entry['attempts'], error=entry['error'])
        finally:
            atomic(STATE, state)
    atomic(STATE, state)
    phases = [state.get(e['experiment'], {}).get('phase') for e in assignment['experiments']]
    success = {'submitted', 'cancelled_by_user'}
    emit('summary', submitted=phases.count('submitted'), cancelled=phases.count('cancelled_by_user'),
         failed=sum(p in TERMINAL - success for p in phases), total=len(phases))
    return 0 if all(p in success for p in phases) else (1 if all(p in TERMINAL for p in phases) else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assignment', type=Path, default=ASSIGNMENT)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    assignment = load_assignment(args.assignment)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    with (HERE / '.remote17-collector.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                result = tick(assignment, state)
                if result is not None:
                    return result
                if args.once:
                    return 2  # Poll succeeded; the assignment is still incomplete.
            except Exception as error:
                emit('poll_error', error=f'{type(error).__name__}: {error}')
                if args.once:
                    raise
            time.sleep(60)


if __name__ == '__main__':
    raise SystemExit(main())
