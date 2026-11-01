"""Run the (arm, stage, block) parts with bounded concurrency, then assemble each arm.

usage: python run_lane.py [--only control|chgeo|both] [--out DIR]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = '/home/t-lab01/sunrun/.venv/bin/python'
ALL_ARMS = ('chgeo', 'control')
CONCURRENCY = 3
THREADS = 3


def parts(order):
    return [(arm, s, b) for arm in order for s in ('F3', 'F2', 'F1') for b in (0, 1, 2)]


def run(part):
    arm, stage, block = part
    cmd = [PY, str(HERE / 'chgeo_lane.py'), '--arm', arm, '--stage', stage, '--block', str(block),
           '--out', str(OUT), '--threads', str(THREADS)]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    line = (proc.stdout.strip().splitlines() or [''])[-1]
    print(f'[{time.time() - t0:7.1f}s rc={proc.returncode}] {arm} {stage} b{block} {line}', flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-2000:], flush=True)
    return proc.returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=('control', 'chgeo', 'both'), default='both')
    ap.add_argument('--out', type=Path, default=HERE / 'runs/v1')
    args = ap.parse_args(argv)
    order = ALL_ARMS if args.only == 'both' else (args.only,)
    globals()['OUT'] = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        codes = list(pool.map(run, parts(order)))
    if any(codes):
        print('some parts failed', codes, flush=True)
        return 1
    for arm in order:
        rc = subprocess.run([PY, str(HERE / 'assemble.py'), '--out', str(OUT), '--arms', arm]).returncode
        if rc:
            return rc
    return 0


if __name__ == '__main__':
    sys.exit(main())
