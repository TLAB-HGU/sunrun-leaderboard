"""Run immutable registered arms sequentially, then score, publish and read back."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LB = ROOT / 'leaderboard'
PYTHON = ROOT / '.venv/bin/python'


def command(args, cwd=HERE):
    print('+', ' '.join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=HERE / 'outputs')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text())
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    log = args.out / 'supervision.json'
    state = {'status': 'running', 'experiments': registration['experiments'], 'completed': []}

    def save():
        state['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        temporary = log.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(log)

    save()
    try:
        for strategy, eid in registration['experiments'].items():
            state['current'] = {'strategy': strategy, 'eid': eid}
            save()
            # run() records a nonzero child return code without returning a CLI failure.
            command([PYTHON, '-m', 'expos', 'run', eid, '--timeout', '172800'], cwd=LB)
            sys.path.insert(0, str(LB))
            from expos import core
            latest = core.state()['experiments'][eid]['runs'][-1]
            if latest['rc'] != 0:
                raise RuntimeError(f'{eid} failed: {latest["rc"]}; see {latest["log"]}')
            state['completed'].append(eid)
            save()
        command([PYTHON, HERE / 'run.py', 'report', '--out', args.out])
        if args.publish:
            for action in ('build', 'dryrun', 'push', 'github', 'submit', 'verify'):
                state['current'] = {'publication': action}
                save()
                command([PYTHON, HERE / 'publish.py', '--out', args.out, action])
        state['status'] = 'completed'
    except Exception as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
