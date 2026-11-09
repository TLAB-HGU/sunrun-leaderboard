"""Collect all three validated fresh runs; publication has a separate owner."""
import argparse
from collections import Counter
import fcntl
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time

from scheduler_support import atomic, now
from transport_support import remote, REMOTE


def remote_status():
    code = f'''import json
from pathlib import Path
r=Path({REMOTE!r});state=json.loads((r/'scheduler-state.json').read_text());pid=(r/'scheduler.pid').read_text().strip() if (r/'scheduler.pid').exists() else ''
alive=False
if pid.isdigit():
 p=Path('/proc')/pid/'cmdline'
 alive=p.exists() and str(r/'host_scheduler.py').encode() in p.read_bytes().split(bytes([0]))
print(json.dumps(dict(state=state,scheduler_alive=alive)))
'''
    return json.loads(remote(code, timeout=35).stdout)


def collect(root, name):
    out = root / 'runs' / name
    if out.exists():
        if (out / 'collected.json').is_file():
            return
        raise ValueError('Incomplete local destination; preserve and diagnose: ' + name)
    code = f'''import sys,tarfile,json
from pathlib import Path
out=Path({REMOTE!r})/'runs'/{name!r}
if (out/'train.exit').read_text().strip()!='0':raise ValueError('Remote run did not exit0')
if not (out/'training-validation.json').exists():raise ValueError('Missing remote replay validation')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:archive.add(out,arcname={name!r})
'''
    data = remote(code, timeout=180).stdout
    with tempfile.TemporaryDirectory(prefix='.collect-', dir=root) as directory:
        staging = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
            for member in archive.getmembers():
                path = Path(member.name)
                if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != name or not (member.isfile() or member.isdir()):
                    raise ValueError('Unsafe remote artifact path')
            archive.extractall(staging, filter='data')
        source = staging / name
        proof = json.loads((source / 'training-validation.json').read_text())
        if proof['rows'] != 454680 or set(proof['saved_model_replay']) != {'F1', 'F2', 'F3'}:
            raise ValueError('Incomplete remote model replay receipt')
        for report in proof['saved_model_replay'].values():
            if report['max_abs_error'] > 1e-6 or report['boosters'] != 3 or report['rounds'] != 900:
                raise ValueError('Remote replay validation failed')
        for filename in ('core.py', 'baseline.py', 'competition.py', 'inventory.json'):
            if (source / filename).read_bytes() != (root / filename).read_bytes():
                raise ValueError('Remote scientific source differs: ' + filename)
        atomic(source / 'collected.json', dict(host='golab2', utc=now(), includes_saved_models=True))
        source.rename(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    recipes = json.loads((root / 'assignment.global.json').read_text())['experiments']
    names = {r['experiment'] for r in recipes}
    with (root / '.collector.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                snapshot = remote_status()
                state = snapshot['state']
                if set(state) != names:
                    raise ValueError('Remote queue differs from frozen three-seed assignment')
                for name, row in state.items():
                    if row['state'] == 'trained_validated':
                        collect(root, name)
                    elif row['state'] == 'running' and not snapshot['scheduler_alive']:
                        row.update(state='interrupted', reason='Remote foreground scheduler is absent; stale running receipt')
                atomic(root / 'runtime.json', state)
                atomic(root / 'remote-state.json', snapshot)
                summary = dict(utc=now(), counts=dict(Counter(v['state'] for v in state.values())),
                               scheduler_alive=snapshot['scheduler_alive'])
                atomic(root / 'collector-state.json', summary)
                print(json.dumps(summary), flush=True)
                if all(v['state'] == 'trained_validated' for v in state.values()):
                    atomic(root / 'training-complete.json', dict(complete=True, utc=now(), experiments=sorted(names), publication='pending'))
                    return 0
                if any(v['state'] in ('failed', 'interrupted') for v in state.values()):
                    atomic(root / 'training-blocker.json', dict(utc=now(), state=state, recovery='Preserve failed attempts; resume only after diagnosis'))
                if args.once:
                    return 0
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                atomic(root / 'collector-error.json', dict(utc=now(), error=str(error)))
                print(json.dumps(dict(utc=now(), error=str(error))), flush=True)
                if args.once:
                    return 1
            time.sleep(20)


if __name__ == '__main__':
    raise SystemExit(main())
