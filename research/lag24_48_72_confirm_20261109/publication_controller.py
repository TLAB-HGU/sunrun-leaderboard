"""Publish the authorized three-seed combination after all fits are validated."""
import argparse
from contextlib import contextmanager
from datetime import UTC, datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(UTC).isoformat()



def analysis_ready(root):
    path = root / 'analysis/results.json'
    if not path.exists():
        return False
    report = json.loads(path.read_text())
    return (report.get('status') == 'complete' and report.get('seeds') == [42, 43, 44]
            and report.get('add_features') == ['v_lag24', 'v_lag48', 'v_lag72']
            and report.get('bootstrap', {}).get('replicates', 0) >= 2000)


@contextmanager
def analysis_slot():
    import host_scheduler
    import scheduler_support as support
    with Path('/tmp/sunrun-cpu-admission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield host_scheduler.capacity(host_scheduler.workloads(), support.memory_gib(),
                                      support.busy_cpu_cores(), threads=1)


def attempt_publication(root):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if not analysis_ready(root):
        with analysis_slot() as admitted:
            if not admitted:
                return 'waiting_for_analysis_resources'
            write(root / 'publication-controller-state.json', dict(state='analyzing', utc=now()))
            with (root / 'analysis.log').open('a') as log:
                result = subprocess.run([sys.executable, '-u', str(root / 'analyze_results.py'),
                                         '--root', str(root), '--bootstrap', '2000'],
                                        stdout=log, stderr=subprocess.STDOUT, env=environment)
            if result.returncode or not analysis_ready(root):
                raise RuntimeError('Full matched-seed analysis failed; inspect analysis.log')
    # A failed performance gate is reported, but does not veto explicit publication.
    write(root / 'publication-controller-state.json', dict(state='publishing', utc=now()))
    with (root / 'publication.log').open('a') as log:
        result = subprocess.run([sys.executable, '-u', str(root / 'publish.py'), '--apply'],
                                stdout=log, stderr=subprocess.STDOUT, env=environment)
    if result.returncode:
        raise RuntimeError(f'Publication exited{result.returncode}; inspect publication.log')
    if not (root / 'campaign-complete.json').exists():
        raise RuntimeError('Publisher returned without verified completion receipt')
    return 'published'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / '.publication-controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            completed = root / 'campaign-complete.json'
            if completed.exists():
                result = json.loads(completed.read_text())
                if result.get('complete') and result.get('visible_rows') == 3 and result.get('all_scores_match'):
                    write(root / 'publication-controller-state.json', dict(state='complete', utc=now(), records=result['records']))
                    return 0
            trained = root / 'training-complete.json'
            receipt = json.loads(trained.read_text()) if trained.exists() else {}
            if receipt.get('complete') or receipt.get('status') == 'complete':
                try:
                    state = attempt_publication(root)
                    if state == 'published':
                        continue
                    write(root / 'publication-controller-state.json', dict(state=state, utc=now()))
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                    write(root / 'publication-controller-state.json', dict(state='retry_pending', utc=now(),
                          error=str(error), reason='Preserve receipts and retry idempotently after60s'))
                    if args.once:
                        return 1
                if args.once:
                    return 0
            else:
                write(root / 'publication-controller-state.json', dict(state='waiting_for_three_validated_seeds', utc=now()))
                if args.once:
                    return 0
            time.sleep(60)


if __name__ == '__main__':
    raise SystemExit(main())
