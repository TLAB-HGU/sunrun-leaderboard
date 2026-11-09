"""Collect three frozen combined fits and publish their fixed mean with visualization."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time

from scheduler_support import atomic, now
from transport_support import remote, REMOTE


def collect(root, names):
    if not names:
        return
    code = f'''import pathlib,sys,tarfile
root=pathlib.Path({REMOTE!r})
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name in {names!r}:
  out=root/'runs'/name
  if (out/'train.exit').read_text().strip()!='0':raise RuntimeError('Unsuccessful run')
  for p in out.iterdir():
   if p.is_file() and p.suffix in ('.json','.npz','.parquet','.exit','.log','.py'):
    archive.add(p,arcname=name+'/'+p.name)
'''
    payload = remote(code, timeout=300).stdout
    with tempfile.TemporaryDirectory(prefix='.collect-', dir=root) as directory:
        staging = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
            for member in archive.getmembers():
                path = Path(member.name)
                if path.is_absolute() or '..' in path.parts or len(path.parts) != 2 or path.parts[0] not in names or not member.isfile():
                    raise ValueError('Unsafe archive member')
            archive.extractall(staging, filter='data')
        for name in names:
            source = staging / name
            if not (source / 'training-validation.json').exists():
                raise ValueError('Missing remote validation')
            atomic(source / 'collected.json', {'host': 'golab2', 'collected_utc': now()})
            source.rename(root / 'runs' / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    (root / 'runs').mkdir(exist_ok=True)
    # Separate controller lock: the host scheduler owns .scheduler.lock.
    import fcntl
    with (root / '.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                local = json.loads((root / 'scheduler-state.json').read_text())
                distant = json.loads(remote(f"from pathlib import Path;print((Path({REMOTE!r})/'scheduler-state.json').read_text())", timeout=30).stdout)
                atomic(root / 'remote-state.json', {'state': distant})
                fresh = [n for n, v in distant.items() if v['state'] == 'trained_validated' and not (root / 'runs' / n).exists()]
                collect(root, fresh)
                state = dict(local, **distant)
                atomic(root / 'runtime.json', state)
                failed = {n: v for n, v in state.items() if v['state'] == 'failed'}
                summary = {'phase': 'training', 'updated_utc': now(), 'states': state}
                if failed:
                    summary.update(phase='failed', failures=failed)
                    atomic(root / 'controller-state.json', summary)
                    raise SystemExit('Training failure; inspect preserved run logs')
                if len(state) == 3 and all(v['state'] == 'trained_validated' for v in state.values()):
                    publisher = root / 'publish.py'
                    if publisher.exists():
                        summary['phase'] = 'publication'
                        atomic(root / 'controller-state.json', summary)
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
                        python = json.loads((root / 'assignment.global.json').read_text())['python']
                        with (root / 'publication.log').open('a') as log:
                            result = subprocess.run([python, str(publisher), '--apply'], stdout=log, stderr=subprocess.STDOUT, env=env)
                        if result.returncode:
                            raise RuntimeError('Publication failed; safe idempotent retry after 60s; see publication.log')
                        if not (root / 'campaign-complete.json').exists():
                            raise RuntimeError('Publisher returned without verified campaign-complete receipt')
                        summary.update(phase='complete', finished_utc=now())
                        atomic(root / 'controller-state.json', summary)
                        return
                    summary['phase'] = 'waiting_for_publisher'
                atomic(root / 'controller-state.json', summary)
                print(json.dumps(summary), flush=True)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                atomic(root / 'controller-error.json', {'utc': now(), 'error': str(error)})
                print(json.dumps({'utc': now(), 'error': str(error)}), flush=True)
            time.sleep(60)


if __name__ == '__main__':
    main()
