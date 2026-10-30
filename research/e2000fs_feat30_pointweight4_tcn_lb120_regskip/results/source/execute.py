"""Status-wrapped training followed by publication; safe to resume the same run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STATUS = ROOT / 'leaderboard/lbstatus.py'
EXPERIMENT = 'e2000fs_feat30_pointweight4_tcn_lb120_regskip'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    code = subprocess.call([sys.executable, str(STATUS), 'run', EXPERIMENT, '--',
                            sys.executable, '-u', str(HERE / 'run.py'), '--out', str(out)], cwd=ROOT)
    if code:
        return code
    state = json.loads((out / 'status.json').read_text())
    if state['status'] == 'completed_internal_rejected':
        return subprocess.call([sys.executable, str(STATUS), 'set', EXPERIMENT, '--status', '완료',
                                '--note', 'Internal improvement gate rejected; negative result preserved, no official upload'], cwd=ROOT)
    if state['status'] != 'completed' or state['smoke']:
        raise RuntimeError('Only completed full experiments may publish')
    try:
        subprocess.run([sys.executable, '-u', str(HERE / 'publish.py'), '--out', str(out)], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(STATUS), 'sync'], cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        subprocess.run([sys.executable, str(STATUS), 'set', EXPERIMENT, '--status', '실패',
                        '--note', f'Training complete; publication failed rc={exc.returncode}. Predictions preserved at {out}'],
                       cwd=ROOT, check=False)
        return exc.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
