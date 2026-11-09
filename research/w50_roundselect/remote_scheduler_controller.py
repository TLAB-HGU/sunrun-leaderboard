"""Keep the real remote scheduler foregrounded across Windows OpenSSH/WSL."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE = '/home/yehwan/w50-roundselect-workspace-20261108/research/w50_roundselect'
PYTHON = '/home/yehwan/sunrun-golab2-20261104-17/.venv/bin/python'


def main():
    # This connection performs scheduling, not a placeholder resource reservation.
    code = ('import os,subprocess\n'
            f'r=subprocess.run([{PYTHON!r},"-u",{REMOTE + "/host_scheduler.py"!r}],check=False,'
            'env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1"))\n'
            'raise SystemExit(r.returncode)\n')
    with (ROOT / 'remote_scheduler_controller.log').open('ab') as log:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                                 '-o', 'ServerAliveInterval=20', '-o', 'ServerAliveCountMax=3',
                                 'golab2', 'wsl -d Ubuntu -- python3 -'],
                                input=code.encode(), stdout=log, stderr=log, check=False)
    (ROOT / 'remote_scheduler_controller.exit').write_text(str(result.returncode) + '\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
