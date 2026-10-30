"""Register the experiment and create its dedicated training tmux via systemd."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = 'e2000fs_feat30_pointweight4_tcn_lb120'
SESSION = 'e2000fs-tcn-lb120'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if subprocess.run(['tmux', 'has-session', '-t', '=' + SESSION], capture_output=True, check=False).returncode == 0:
        raise SystemExit(f'Refusing to overwrite existing tmux session {SESSION}')
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / 'leaderboard/lbstatus.py'), 'set', EXPERIMENT,
                    '--status', '대기', '--desc', 'TCN feat30 hourly lookback120; pointwise extreme4; physical GPU1; seed42'],
                   cwd=ROOT, check=True)
    command = shlex.join(['/bin/bash', str(HERE / 'tmux_entry.sh'), str(out)])
    subprocess.run(['systemd-run', '--user', '--collect', '--unit=e2000fs-tcn-lb120-launch',
                    '--working-directory=' + str(ROOT), '/usr/bin/tmux', 'new-session', '-d',
                    '-s', SESSION, '-c', str(ROOT), command], check=True)
    print(f'Launched tmux session {SESSION}; log={out / "run.log"}', flush=True)


if __name__ == '__main__':
    main()
