"""Run the real golab2 scheduler in the foreground of a persistent SSH call."""
import argparse
from pathlib import Path
import subprocess

from transport_support import REMOTE, RPY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    code = ('import os\n'
            f'os.chdir({REMOTE!r})\n'
            'env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",OMP_NUM_THREADS="1")\n'
            f'os.execve({RPY!r},[{RPY!r},"-u",{REMOTE + "/host_scheduler.py"!r},"--root",{REMOTE!r}],env)\n')
    # This connection performs actual scheduling; no placeholder keeps resources occupied.
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                             '-o', 'ServerAliveInterval=20', '-o', 'ServerAliveCountMax=3',
                             'golab2', 'wsl -d Ubuntu -- python3 -'], input=code.encode(), check=False)
    temporary = root / 'remote_scheduler_controller.exit.tmp'
    temporary.write_text(str(result.returncode) + '\n')
    temporary.replace(root / 'remote_scheduler_controller.exit')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
