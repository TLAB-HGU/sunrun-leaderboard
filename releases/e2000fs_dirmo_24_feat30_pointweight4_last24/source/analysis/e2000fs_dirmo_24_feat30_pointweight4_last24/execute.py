"""Status-wrapped training followed by publication; safe to resume the same run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STATUS = ROOT / 'leaderboard/lbstatus.py'
EXPERIMENT = 'e2000fs_dirmo_24_feat30_pointweight4_last24'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    code = subprocess.call([sys.executable, str(STATUS), 'run', EXPERIMENT, '--',
                            sys.executable, '-u', str(HERE / 'run.py'), '--out', str(out)], cwd=ROOT)
    if code:
        return code
    try:
        state = json.loads((out / 'status.json').read_text())
        if state['status'] != 'completed' or state['smoke']:
            raise RuntimeError('Only completed full experiments may publish')
        subprocess.run([sys.executable, '-u', str(HERE / 'publish.py'), '--out', str(out)], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(STATUS), 'sync'], cwd=ROOT, check=True)
    except (subprocess.CalledProcessError, OSError, ValueError, KeyError, RuntimeError) as exc:
        code = exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 1
        subprocess.run([sys.executable, str(STATUS), 'set', EXPERIMENT, '--status', '실패',
                        '--note', f'Training returned success; publication failed ({type(exc).__name__}, rc={code}). Predictions preserved at {out}'],
                       cwd=ROOT, check=False)
        print(f'Publication failed: {exc}', file=sys.stderr, flush=True)
        return code
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
