"""Local axes4 analysis controller: wait for the local scheduler, then run the paired report.

Replaces the golab2 collect/analysis branch of prepare_controller.py after the
2026-11-09 move to local execution. It never trains, never publishes.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time

import scheduler_support as support
from prepare_controller import admitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    assignment = json.loads((root / 'assignment.json').read_text())
    names = {r['experiment'] for r in assignment['experiments']}
    while True:
        state = json.loads((root / 'scheduler-state.json').read_text())
        if set(state) != names:
            raise ValueError('Local state does not match all twelve frozen runs')
        counts = dict(Counter(v['state'] for v in state.values()))
        done = all(v['state'] == 'trained_validated' for v in state.values())
        blockers = {k: v for k, v in state.items() if v['state'] == 'failed'}
        summary = dict(phase='analyzing' if done else 'training_with_blockers' if blockers else 'training',
                       execution_host='local', utc=support.now(), counts=counts,
                       parent='e2000fs_calib_offset_roll_drop_t035_v650', publication='not_required')
        support.atomic(root / 'controller-state.json', summary)
        if blockers:
            support.atomic(root / 'training-blocker.json', dict(utc=support.now(), affected=blockers))
            if (root / 'host-complete.json').exists():
                support.atomic(root / 'controller-state.json', dict(summary, phase='training_failed'))
                return 1
        if done:
            with admitted(root, 'paired_analysis'):
                with (root / 'analysis.log').open('a') as log:
                    result = subprocess.run([assignment['python'], str(root / 'analyze_results.py'), '--root', str(root),
                                             '--bootstrap', '2000'], stdout=log, stderr=subprocess.STDOUT)
            report = json.loads((root / 'analysis/results.json').read_text()) if (root / 'analysis/results.json').exists() else {}
            complete = result.returncode == 0 and report.get('status') == 'complete'
            support.atomic(root / 'campaign-complete.json', dict(complete=complete, execution_host='local',
                           analysis_exit=result.returncode, publication='not_required', utc=support.now()))
            support.atomic(root / 'controller-state.json', dict(summary, phase='complete' if complete else 'analysis_blocked'))
            return 0 if complete else 1
        print(json.dumps(summary), flush=True)
        time.sleep(60)


if __name__ == '__main__':
    raise SystemExit(main())
