"""Detached supervisor: status-wrapped train, five-minute records, publish and sync."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LANE = ROOT/'research/tail-budget-20261029/lanes/complexity'
STATUS = ROOT/'leaderboard/lbstatus.py'
EXPERIMENT = HERE.name


def record(out, phase, detail=''):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    state = (out/'status.json').read_text() if (out/'status.json').exists() else '{}'
    text = (f'# complexity — {phase}\n\nUpdated: {now}\nSession: tb29-complexity\n'
            f'Supervisor PID: {os.getpid()}\nOwned code: {HERE}\nOwned records: {LANE}\n'
            f'Run: {out}\n\n{detail}\n\nTraining state:\n```json\n{state}\n```\n'
            'No target achievement claimed before official scores. Published3556/5923 is historical.\n')
    (LANE/'STATUS.md').write_text(text)


def run_supervised(command,out,phase):
    record(out,phase,'Command: '+repr(command))
    process = subprocess.Popen(command,cwd=ROOT)
    tick = 0
    while process.poll() is None:
        if time.monotonic()>=tick:
            record(out,phase,f'Child PID: {process.pid}; command: {command!r}')
            tick = time.monotonic()+300
        time.sleep(2)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode,command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    try:
        if not (out/'status.json').exists():
            run_supervised([sys.executable,str(STATUS),'run',EXPERIMENT,'--',sys.executable,'-u',str(HERE/'run.py'),'--out',str(out)],out,'training')
        state = json.loads((out/'status.json').read_text())
        if state['status']!='completed' or state['smoke']:
            raise RuntimeError('Only a completed full run may publish; failed artifacts are preserved')
        # A publication transport race may retry the same frozen identity, never retrain/select.
        for attempt in range(3):
            try:
                run_supervised([sys.executable,'-u',str(HERE/'publish.py'),'--out',str(out)],out,'publishing')
                break
            except subprocess.CalledProcessError:
                if attempt==2:
                    raise
                time.sleep(10)
        run_supervised([sys.executable,str(STATUS),'sync'],out,'sync')
        spec = __import__('importlib.util',fromlist=['util'])
        s = spec.spec_from_file_location('complexity_report',HERE/'run.py')
        r = spec.module_from_spec(s)
        s.loader.exec_module(r)
        r.report(out,json.loads((out/'paired_results.json').read_text()),out/'publication/submission.json')
        record(out,'completed; submitted and synced',f'Receipts: {out}/publication/; report: {LANE}/REPORT.md')
        return 0
    except BaseException:
        detail = traceback.format_exc()
        record(out,'infrastructure/training failure; artifacts preserved',detail)
        subprocess.run([sys.executable,str(STATUS),'set',EXPERIMENT,'--status','실패','--note',f'Execution/publication error; preserved {out}; see complexity STATUS.md'],cwd=ROOT,check=False)
        raise


if __name__=='__main__':
    raise SystemExit(main())
