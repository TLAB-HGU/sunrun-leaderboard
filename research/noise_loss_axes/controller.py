"""Durable completion watcher for the fixed five-arm loss campaign.

This controller never trains, retries fits, changes parameters or publishes.
It releases only the named Colab session, after the six remote runs have been
retrieved and independently validated. Failed fits keep their logs and runtime.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import fcntl
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
REMOTE_ROOT = '/content/sunrun/research/noise_loss_axes'
SESSION = 'sunrun-noise-loss-cpu'
REMOTE_ARMS = ('gaussian', 'lognormal')
LOCAL_ARMS = ('sera_l2', 'plain_l2', 'mixture')
ARMS = (*LOCAL_ARMS, *REMOTE_ARMS)
SEEDS = (42, 43, 44)
REMOTE_ARCHIVE = '/content/noise_loss_results.tar.gz'


def timestamp():
    return dt.datetime.now(dt.UTC).isoformat()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def read_exit(path):
    path = Path(path)
    return int(path.read_text().strip()) if path.exists() else None


def expected_features(source, block):
    values = {}
    for node in ast.parse(Path(source).read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('CONSTANT', 'VARYING'):
                    values[target.id] = ast.literal_eval(node.value)
    return values['CONSTANT'] + [f'{name}@h{h}' for h in range(block*24+1, (block+1)*24+1)
                                 for name in values['VARYING']]


def finite_numbers(value):
    if isinstance(value, dict):
        return all(finite_numbers(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_numbers(v) for v in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def validate_seed(root, arm, seed, source):
    """Validate complete model, prediction and metric artifacts, without fitting."""
    import numpy as np
    import pandas as pd

    out = Path(root)/'results'/arm/f'seed{seed}'
    if read_exit(out/'process.exit') != 0:
        raise ValueError(f'{arm}/{seed}: successful process exit missing')
    identity = json.loads((out/'run.json').read_text())
    expected = {'arm': arm, 'seed': seed, 'rounds': 300, 'smoke': False,
                'maturity_hours': 144, 'target_scale': 500.0, 'sigma_floor': .02,
                'train_cutoff': '2025-06-29 00:00:00+00:00',
                'validation_start': '2025-07-01 00:00:00+00:00',
                'validation_end': '2025-10-01 00:00:00+00:00'}
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError(f'{arm}/{seed}: registered run identity mismatch')
    if (out/'source.py').read_text() != Path(source).read_text():
        raise ValueError(f'{arm}/{seed}: runner source differs')
    complete = json.loads((out/'complete.json').read_text())
    metrics = json.loads((out/'metrics.json').read_text())
    if complete.get('status') != 'complete' or complete.get('exit_code') != 0 or complete.get('identity') != identity:
        raise ValueError(f'{arm}/{seed}: completion metadata mismatch')
    if metrics.get('identity') != identity or metrics.get('arm') != arm or metrics.get('seed') != seed:
        raise ValueError(f'{arm}/{seed}: metrics identity mismatch')
    if not finite_numbers(metrics) or any(k not in metrics for k in ('W', 'mae', 'rmse', 'bias', 'n_pairs')):
        raise ValueError(f'{arm}/{seed}: invalid metrics')
    if metrics['n_pairs'] <= 0 or metrics['W'] < 0:
        raise ValueError(f'{arm}/{seed}: invalid score population')
    by_lead = metrics.get('by_lead', [])
    if [r.get('lead') for r in by_lead] != list(range(1, 73)) or any(r.get('n', 0) <= 0 for r in by_lead):
        raise ValueError(f'{arm}/{seed}: incomplete lead metrics')
    if arm in ('gaussian', 'lognormal', 'mixture'):
        distribution = metrics.get('distribution', {})
        if not all(k in distribution for k in ('nll', 'coverage50', 'coverage90', 'coverage95', 'pit_hist')):
            raise ValueError(f'{arm}/{seed}: distribution metrics missing')
        if any(not 0 <= distribution[k] <= 1 for k in ('coverage50', 'coverage90', 'coverage95')):
            raise ValueError(f'{arm}/{seed}: invalid interval coverage')
    origins = pd.date_range('2025-07-01T00:00Z', '2025-09-27T18:00Z', freq='6h')
    frame = pd.read_parquet(out/'predictions.parquet')
    required = {'origin', 'h', 'pred_kms', 'y'}
    if not required.issubset(frame.columns) or len(frame) != len(origins)*72:
        raise ValueError(f'{arm}/{seed}: prediction shape mismatch')
    if frame.duplicated(['origin', 'h']).any():
        raise ValueError(f'{arm}/{seed}: duplicate prediction key')
    frame = frame.sort_values(['origin', 'h'])
    actual_origins = pd.to_datetime(frame.origin, utc=True).array.asi8.reshape(-1, 72)
    if not np.array_equal(actual_origins, np.repeat(origins.asi8[:, None], 72, axis=1)):
        raise ValueError(f'{arm}/{seed}: prediction origin mismatch')
    if not np.array_equal(frame.h.to_numpy().reshape(-1, 72), np.tile(np.arange(1, 73), (len(origins), 1))):
        raise ValueError(f'{arm}/{seed}: prediction horizon mismatch')
    pred, truth = frame.pred_kms.to_numpy(), frame.y.to_numpy()
    if not np.isfinite(pred).all() or (pred <= 0).any() or np.isinf(truth).any():
        raise ValueError(f'{arm}/{seed}: invalid predictions/truth')
    if int(np.isfinite(truth).sum()) != metrics['n_pairs'] or complete.get('rows') != len(frame):
        raise ValueError(f'{arm}/{seed}: completed population mismatch')
    for block in range(3):
        meta = json.loads((out/f'b{block}.json').read_text())
        if meta.get('identity') != identity or meta.get('block') != block or meta.get('feature_count') != 237:
            raise ValueError(f'{arm}/{seed}/b{block}: metadata mismatch')
        if meta.get('features') != expected_features(source, block):
            raise ValueError(f'{arm}/{seed}/b{block}: feature list changed')
        if not (out/f'b{block}.ubj').stat().st_size:
            raise ValueError(f'{arm}/{seed}/b{block}: empty model')
        curves = json.loads((out/f'b{block}.loss.json').read_text())
        curves = curves.get('train', {})
        if not curves or any(len(v) != 300 for v in curves.values()) or not finite_numbers(curves):
            raise ValueError(f'{arm}/{seed}/b{block}: training curve incomplete')
        with np.load(out/f'b{block}.npz', allow_pickle=False) as data:
            sl = slice(block*24, (block+1)*24)
            if not np.array_equal(data['origins'], origins.asi8):
                raise ValueError(f'{arm}/{seed}/b{block}: block origins differ')
            if not np.array_equal(data['pred'], pred.reshape(-1, 72)[:, sl]):
                raise ValueError(f'{arm}/{seed}/b{block}: predictions differ from assembly')
            if not np.array_equal(data['truth'], truth.reshape(-1, 72)[:, sl], equal_nan=True):
                raise ValueError(f'{arm}/{seed}/b{block}: truth differs from assembly')
            if arm in ('gaussian', 'lognormal', 'mixture'):
                width = 5 if arm == 'mixture' else 2
                if data['theta'].shape != (len(origins), 24, width) or not np.isfinite(data['theta']).all():
                    raise ValueError(f'{arm}/{seed}/b{block}: invalid distribution parameters')
    return {'arm': arm, 'seed': seed, 'rows': len(frame), 'n_pairs': metrics['n_pairs'], 'W': metrics['W'],
            'libraries': identity.get('libraries'), 'data': identity.get('data'), 'truth': identity.get('truth')}


def validate_arms(root, arms, source):
    records = []
    for arm in arms:
        if read_exit(Path(root)/'results'/arm/'process.exit') != 0:
            raise ValueError(f'{arm}: successful arm exit missing')
        records.extend(validate_seed(root, arm, seed, source) for seed in SEEDS)
    return records


def allowed_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or not path.parts:
        return False
    parts = path.parts
    if parts[0] == 'results' and len(parts) >= 2 and parts[1] in REMOTE_ARMS:
        return (len(parts) == 3 and parts[2] == 'process.exit') or (len(parts) >= 4 and parts[2] in ('seed42', 'seed43', 'seed44'))
    if parts[0] == 'logs' and len(parts) == 2:
        return re.fullmatch(r'(gaussian|lognormal)[A-Za-z0-9_.-]*\.log', parts[1]) is not None
    return parts == ('evidence', 'remote_resources.json')


def secure_extract(archive, destination):
    """Extract regular files only into a fresh staging tree with a fixed allowlist."""
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Extraction destination must be empty')
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as handle:
        members = handle.getmembers()
        seen = set()
        total = 0
        for member in members:
            if not member.isfile() or not allowed_member(member.name) or member.name in seen:
                raise ValueError(f'Unsafe archive member: {member.name}')
            seen.add(member.name)
            total += member.size
            if total > 10*1024**3:
                raise ValueError('Archive exceeds the 10 GiB artifact limit')
        for member in members:
            target = destination/PurePosixPath(member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.extractfile(member) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output)
    return len(members)


def install_staged(staging, root):
    """Merge retrieved artifacts without overwriting any existing different file."""
    files = [p for p in Path(staging).rglob('*') if p.is_file()]
    for source in files:
        target = Path(root)/source.relative_to(staging)
        if target.exists() and (not target.is_file() or source.read_bytes() != target.read_bytes()):
            raise ValueError(f'Retrieved artifact conflicts with existing file: {target}')
    for source in files:
        target = Path(root)/source.relative_to(staging)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def remote_probe_source():
    return '''import json, os, platform, subprocess, tarfile
from pathlib import Path
import importlib.metadata
root = Path(__REMOTE_ROOT__)
arms = ('gaussian', 'lognormal')
def code(path):
    return int(path.read_text().strip()) if path.exists() else None
result = {'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'arms': {}, 'archive': None}
for arm in arms:
    result['arms'][arm] = {'exit': code(root/'results'/arm/'process.exit'), 'seeds': {}}
    for seed in (42,43,44):
        path = root/'results'/arm/f'seed{seed}'
        result['arms'][arm]['seeds'][str(seed)] = {'exit': code(path/'process.exit'), 'complete': (path/'complete.json').exists()}
if all(result['arms'][arm]['exit'] == 0 and all(v['exit'] == 0 and v['complete'] for v in result['arms'][arm]['seeds'].values()) for arm in arms):
    evidence = root/'evidence'/'remote_resources.json'
    evidence.parent.mkdir(parents=True, exist_ok=True)
    if not evidence.exists():
        versions = {p: importlib.metadata.version(p) for p in ('numpy','pandas','xgboost','scikit-learn','pyarrow')}
        record = {'boot_id': result['boot_id'], 'platform': platform.platform(), 'cpu_count': os.cpu_count(), 'versions': versions,
                  'memory': Path('/proc/meminfo').read_text(),
                  'processes': subprocess.run(['ps','-eo','pid,etime,pcpu,pmem,comm'], capture_output=True, text=True).stdout}
        evidence.write_text(json.dumps(record, indent=2))
    archive = Path(__REMOTE_ARCHIVE__)
    if not archive.exists():
        files = [p for arm in arms for p in (root/'results'/arm).rglob('*') if p.is_file()]
        files += [p for arm in arms for p in (root/'logs').glob(arm+'*.log') if p.is_file()]
        files += [evidence]
        with tarfile.open(str(archive)+'.partial','w:gz') as tar:
            for path in files:
                if path.is_symlink(): raise RuntimeError('Symlink in experiment artifacts')
                tar.add(path, arcname=str(path.relative_to(root)), recursive=False)
        Path(str(archive)+'.partial').replace(archive)
    result['archive'] = {'path': str(archive), 'bytes': archive.stat().st_size}
print('NOISE_STATUS='+json.dumps(result))
'''.replace('__REMOTE_ROOT__', repr(REMOTE_ROOT)).replace('__REMOTE_ARCHIVE__', repr(REMOTE_ARCHIVE))


def parse_probe(stdout):
    for line in reversed(stdout.splitlines()):
        if 'NOISE_STATUS=' in line:
            return json.loads(line.split('NOISE_STATUS=', 1)[1])
    raise RuntimeError('Colab probe returned no NOISE_STATUS record')


def run_command(command, log, timeout=240):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    with Path(log).open('a') as handle:
        handle.write(f'\n[{timestamp()}] {command!r}\n{result.stdout}\n{result.stderr}\nexit={result.returncode}\n')
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {command[:2]}')
    return result.stdout


def local_progress(root):
    arms = {}
    failures = []
    for arm in ARMS:
        arms[arm] = {'exit': read_exit(Path(root)/'results'/arm/'process.exit'), 'seeds': {}}
        if arms[arm]['exit'] not in (None, 0):
            failures.append(f'{arm}: exit={arms[arm]["exit"]}')
        for seed in SEEDS:
            path = Path(root)/'results'/arm/f'seed{seed}'
            status = {'exit': read_exit(path/'process.exit'), 'complete': (path/'complete.json').exists(),
                      'blocks': sum((path/f'b{block}.json').exists() for block in range(3))}
            arms[arm]['seeds'][str(seed)] = status
            if status['exit'] not in (None, 0):
                failures.append(f'{arm}/{seed}: exit={status["exit"]}')
    return arms, failures


def monitor(root, session, interval, once=False):
    root = Path(root)
    evidence = root/'evidence'
    evidence.mkdir(parents=True, exist_ok=True)
    logfile = root/'logs'/'controller.log'
    logfile.parent.mkdir(parents=True, exist_ok=True)
    probe_path = evidence/'remote_probe.py'
    probe_path.write_text(remote_probe_source())
    receipt_path = evidence/'remote_receipt.json'
    identity_path = evidence/'remote_boot_identity.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    iteration = 0
    while True:
        iteration += 1
        state = {'updated_at': timestamp(), 'iteration': iteration, 'session': session, 'status': 'running'}
        progress, failures = local_progress(root)
        state.update(arms=progress, failures=failures,
                     completed_seeds=sum(v['exit'] == 0 and v['complete'] for a in progress.values() for v in a['seeds'].values()))
        try:
            if receipt is None or not receipt.get('session_stopped'):
                remote = parse_probe(run_command(['colab', 'exec', '-s', session, '--file', str(probe_path), '--timeout', '120'], logfile))
                state['remote'] = remote
                if identity_path.exists():
                    frozen = json.loads(identity_path.read_text())
                    if frozen != {'session': session, 'boot_id': remote['boot_id']}:
                        raise ValueError('Named Colab runtime identity changed; refusing retrieval or stop')
                else:
                    write_json(identity_path, {'session': session, 'boot_id': remote['boot_id']})
                for arm, value in remote['arms'].items():
                    if value['exit'] not in (None, 0):
                        failures.append(f'remote {arm}: exit={value["exit"]}')
                    for seed, seed_value in value['seeds'].items():
                        if seed_value['exit'] not in (None, 0):
                            failures.append(f'remote {arm}/{seed}: exit={seed_value["exit"]}')
                if remote.get('archive') and receipt is None:
                    if remote['archive']['path'] != REMOTE_ARCHIVE:
                        raise ValueError('Unexpected remote archive path')
                    archive = evidence/'remote_results.tar.gz'
                    run_command(['colab', 'download', '-s', session, REMOTE_ARCHIVE, str(archive)], logfile, timeout=600)
                    if archive.stat().st_size != remote['archive']['bytes']:
                        raise ValueError('Downloaded archive byte count mismatch')
                    with tempfile.TemporaryDirectory(prefix='remote-stage-', dir=evidence) as temporary:
                        members = secure_extract(archive, temporary)
                        records = validate_arms(temporary, REMOTE_ARMS, root/'experiment.py')
                        install_staged(temporary, root)
                    validate_arms(root, REMOTE_ARMS, root/'experiment.py')
                    receipt = {'session': session, 'boot_id': remote['boot_id'], 'retrieved_at': timestamp(),
                               'archive': str(archive), 'bytes': archive.stat().st_size, 'members': members,
                               'verified_runs': records, 'session_stopped': False}
                    write_json(receipt_path, receipt)
                if receipt is not None and not receipt['session_stopped']:
                    # Revalidate the exact runtime after transfer and immediately before stop.
                    remote = parse_probe(run_command(['colab', 'exec', '-s', session, '--file', str(probe_path), '--timeout', '120'], logfile))
                    if remote['boot_id'] != receipt['boot_id'] or not remote.get('archive'):
                        raise ValueError('Runtime identity/completion changed before stop')
                    validate_arms(root, REMOTE_ARMS, root/'experiment.py')
                    run_command(['colab', 'stop', '-s', session], logfile)
                    receipt.update(session_stopped=True, stopped_at=timestamp())
                    write_json(receipt_path, receipt)
            if failures:
                state.update(status='failed', failures=failures, note='Healthy jobs and remote runtime left intact; no fit retries')
                write_json(root/'status.json', state)
                return 2
            progress, failures = local_progress(root)
            state.update(arms=progress, failures=failures,
                         completed_seeds=sum(v['exit'] == 0 and v['complete'] for a in progress.values() for v in a['seeds'].values()))
            if all(a['exit'] == 0 for a in progress.values()):
                records = validate_arms(root, ARMS, root/'experiment.py')
                write_json(evidence/'completion_validation.json', {'validated_at': timestamp(), 'runs': records})
                run_command([sys.executable, str(root/'summarize.py')], logfile)
                finalizer = root/'finalize.py'
                if not finalizer.exists():
                    state['status'] = 'awaiting_finalizer'
                else:
                    run_command([sys.executable, str(finalizer)], logfile, timeout=600)
                    state.update(status='complete', completed_at=timestamp(), completed_seeds=15)
                    write_json(root/'status.json', state)
                    return 0
        except (ValueError, FileNotFoundError, KeyError, TypeError, tarfile.TarError, zipfile.BadZipFile) as error:
            state.update(status='failed', last_tool_error=str(error), note='Artifact or identity validation failed; runtimes preserved')
            write_json(root/'status.json', state)
            return 2
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as error:
            state['last_tool_error'] = str(error)
            state['status'] = 'retrying_transport'
        state['remote_retrieved'] = receipt is not None
        write_json(root/'status.json', state)
        print(json.dumps({k: state[k] for k in ('updated_at', 'status', 'completed_seeds')}), flush=True)
        if once:
            return 0
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE)
    parser.add_argument('--session', default=SESSION)
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.session != SESSION:
        raise ValueError('This controller is restricted to its exact assigned Colab session')
    if args.interval < 10:
        raise ValueError('Polling interval must be at least 10 seconds')
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root/'controller.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Controller is already active') from None
        lock.seek(0)
        lock.truncate()
        lock.write(f'{os.getpid()}\n')
        lock.flush()
        raise SystemExit(monitor(args.root, args.session, args.interval, args.once))


if __name__ == '__main__':
    main()
