"""Mirror exact remote campaign artifacts; never selects a new hypothesis."""
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'rerun_df_ffill_20261103'))
from golab2_transport import remote

REMOTE = '/home/yehwan/w50-roundselect-workspace-20261108/research/w50_roundselect_20261108'


def collect_once():
    code = """
from pathlib import Path
import io,sys,tarfile
root=Path(REMOTE_ROOT)
done=(root/'host-complete.json').exists()
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w:gz') as tar:
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        relative=path.relative_to(root)
        if relative.parts[0] in ('frozen_controls','code_snapshot'):continue
        if ('__pycache__' in relative.parts or path.suffix=='.ubj') and not done:
            continue
        if path.suffix in ('.json','.exit','.log','.npz','.parquet','.ubj','.sh'):
            tar.add(path,arcname=str(relative),recursive=False)
sys.stdout.buffer.write(buffer.getvalue())
""".replace('REMOTE_ROOT', repr(REMOTE))
    result = remote(code, timeout=120)
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:gz') as tar:
        # Use this remote snapshot, not a stale local receipt from an old attempt.
        remote_terminal = 'host-complete.json' in tar.getnames()
        # Extraction filter rejects traversal, unsafe links and special files.
        tar.extractall(ROOT, filter='data')
    state = json.loads((ROOT / 'scheduler-state.json').read_text())
    (ROOT / 'runtime.json').write_text(json.dumps(state, indent=2) + '\n')
    return remote_terminal


def main():
    failures = 0
    while True:
        try:
            done = collect_once()
            failures = 0
            state = json.loads((ROOT / 'runtime.json').read_text())
            success = bool(done and state and all(
                row.get('state') == 'trained_validated' for row in state.values()))
            print(json.dumps({'stage': 'collected', 'all_terminal': done,
                              'training_success_all': success,
                              'research_goal_complete': False}), flush=True)
            if done:
                (ROOT / 'next-decision.json').write_text(json.dumps({
                    'status': 'awaiting_scientific_review' if success else 'training_failure_review',
                    'action': 'Check exits and source/model replay; then analyze matched seeds, diagnose failure before next preregistration.',
                    'research_goal_complete': False}, indent=2) + '\n')
                return
        except (OSError, ValueError, tarfile.TarError, subprocess.SubprocessError) as error:
            failures += 1
            print(json.dumps({'stage': 'collection_error', 'attempt': failures,
                              'error': str(error)}), flush=True)
        time.sleep(45)


if __name__ == '__main__':
    main()
